import argparse
import os
import os.path as osp
import mmengine
import numpy as np
import torch
import copy
import mmcv
import torch.nn.functional as F
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import CameraInstance3DBoxes
from mmdet3d.utils import register_all_modules
from mmengine.structures import InstanceData

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Loss Final')
    parser.add_argument('img_dir', help='Directory containing images')
    parser.add_argument('ann_file', help='The validation .pkl file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--cam-type', default='CAM_FRONT', help='Camera type')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def apply_lidar2cam(bboxes_3d, lidar2cam_matrix):
    """Transform 3D boxes from LiDAR frame to Camera frame."""
    if len(bboxes_3d) == 0:
        return bboxes_3d
    
    # Transform Centers (x, y, z)
    centers = bboxes_3d[:, :3] 
    centers_hom = np.hstack([centers, np.ones((centers.shape[0], 1))]) 
    centers_cam = (lidar2cam_matrix @ centers_hom.T).T 
    bboxes_3d[:, :3] = centers_cam[:, :3]
    return bboxes_3d

def main():
    register_all_modules(init_default_scope=True)
    args = parse_args()

    # 1. Load Config & Model
    print(f"Loading config: {args.config}")
    cfg = mmengine.Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    # Determine Model Classes (Standard NuScenes 10 classes)
    if hasattr(cfg, 'class_names'):
        model_classes = cfg.class_names
    else:
        # Fallback to standard 10 classes if not in config
        model_classes = [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]
    print(f"Model expects classes: {model_classes}")

    # 2. Load Annotations
    print(f"Loading annotations from {args.ann_file}...")
    full_infos = mmengine.load(args.ann_file)
    data_list = full_infos['data_list']
    
    # Get Pickle Classes
    if 'metainfo' in full_infos and 'classes' in full_infos['metainfo']:
        pkl_classes = full_infos['metainfo']['classes']
    else:
        # Fallback if metainfo is missing (common in old pkls)
        pkl_classes = model_classes 
    
    file_index = {}
    for item in data_list:
        if args.cam_type in item['images']:
            path_in_pkl = item['images'][args.cam_type]['img_path']
            fname = osp.basename(path_in_pkl)
            file_index[fname] = item

    # 3. Iterate
    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Found {len(img_files)} images. Calculating detailed losses...")

    results = [] 

    # FCOS3D Normalization
    mean = torch.tensor([103.530, 116.280, 123.675], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([1.0, 1.0, 1.0], device=args.device).view(1, 3, 1, 1)

    for i, img_file in enumerate(img_files):
        if img_file not in file_index:
            continue
            
        raw_info = copy.deepcopy(file_index[img_file])
        full_img_path = osp.join(args.img_dir, img_file)
        
        # --- A. Manual Preprocessing ---
        img_bytes = mmengine.fileio.get(full_img_path)
        img = mmcv.imfrombytes(img_bytes) 
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(args.device)
        img_tensor = (img_tensor - mean) / std
        
        # Pad to divisor 32
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        divisor = 32
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), value=0)
        
        inputs_dict = {'imgs': img_tensor}

        # --- B. Prepare Data Sample ---
        cam2img_arr = np.array(raw_info['images'][args.cam_type]['cam2img'], dtype=np.float32)
        lidar2cam_arr = np.array(raw_info['images'][args.cam_type]['lidar2cam'], dtype=np.float32)

        data_sample = Det3DDataSample()
        data_sample.set_metainfo(dict(
            img_path=full_img_path,
            ori_shape=(h, w),
            img_shape=(h + pad_h, w + pad_w),
            pad_shape=(h + pad_h, w + pad_w),
            scale_factor=1.0,
            cam2img=cam2img_arr,
            lidar2cam=lidar2cam_arr,
            cam_type=args.cam_type
        ))
        
        # --- C. Extract & Remap GT ---
        gt_bboxes_3d, gt_labels_3d = [], []
        gt_bboxes_2d, gt_labels_2d = [], []
        
        for inst in raw_info['instances']:
            # 1. Get original label index
            orig_label = inst['bbox_label_3d'] if 'bbox_label_3d' in inst else inst['bbox_label']
            
            # 2. Get Class Name
            if orig_label < len(pkl_classes):
                class_name = pkl_classes[orig_label]
            else:
                continue # Skip invalid indices
            
            # 3. Check if this class exists in the Model's config
            if class_name in model_classes:
                # 4. Map to Model's Index
                new_label = model_classes.index(class_name)
                
                # Append Data
                if 'bbox_3d' in inst:
                    gt_bboxes_3d.append(inst['bbox_3d'])
                    gt_labels_3d.append(new_label) # Use NEW label
                    
                    if 'bbox' in inst:
                        gt_bboxes_2d.append(inst['bbox'])
                        gt_labels_2d.append(new_label)
                    else:
                        gt_bboxes_2d.append([0,0,10,10])
                        gt_labels_2d.append(new_label)

        if len(gt_bboxes_3d) > 0:
            gt_bboxes_3d_np = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_labels_3d_np = np.array(gt_labels_3d, dtype=np.int64)

            # --- Transform LiDAR -> Camera ---
            gt_bboxes_3d_np = apply_lidar2cam(gt_bboxes_3d_np, lidar2cam_arr)

            # Pad to 9 dims
            if gt_bboxes_3d_np.shape[1] == 7:
                padding = np.zeros((gt_bboxes_3d_np.shape[0], 2), dtype=np.float32)
                gt_bboxes_3d_np = np.hstack([gt_bboxes_3d_np, padding])
            
            box_dim = gt_bboxes_3d_np.shape[-1]
            gt_bboxes_3d_inst = CameraInstance3DBoxes(
                torch.from_numpy(gt_bboxes_3d_np).to(args.device), 
                box_dim=box_dim, origin=(0.5, 0.5, 0.5) 
            )
            data_sample.gt_instances_3d = InstanceData()
            data_sample.gt_instances_3d.bboxes_3d = gt_bboxes_3d_inst
            data_sample.gt_instances_3d.labels_3d = torch.from_numpy(gt_labels_3d_np).to(args.device)

            # 2D Projections
            centers_3d = torch.from_numpy(gt_bboxes_3d_np[:, :3]).to(args.device) 
            cam2img_tensor = torch.from_numpy(cam2img_arr[:3, :3]).to(args.device) 
            points_2d = centers_3d @ cam2img_tensor.T
            z = points_2d[:, 2].clamp(min=1e-5)
            centers_2d = torch.stack([points_2d[:, 0] / z, points_2d[:, 1] / z], dim=1)
            
            data_sample.gt_instances_3d.centers_2d = centers_2d
            data_sample.gt_instances_3d.depths = points_2d[:, 2]

            # 2D Instances
            data_sample.gt_instances = InstanceData()
            data_sample.gt_instances.bboxes = torch.from_numpy(np.array(gt_bboxes_2d)).to(args.device)
            data_sample.gt_instances.labels = torch.from_numpy(np.array(gt_labels_2d)).to(args.device)
        else:
            # Handle Empty
            data_sample.gt_instances_3d = InstanceData()
            data_sample.gt_instances_3d.bboxes_3d = CameraInstance3DBoxes(torch.empty((0, 9)).to(args.device))
            data_sample.gt_instances_3d.labels_3d = torch.empty(0).to(args.device).long()
            data_sample.gt_instances_3d.centers_2d = torch.empty((0, 2)).to(args.device)
            data_sample.gt_instances_3d.depths = torch.empty(0).to(args.device)
            data_sample.gt_instances = InstanceData()
            data_sample.gt_instances.bboxes = torch.empty((0, 4)).to(args.device)
            data_sample.gt_instances.labels = torch.empty(0).to(args.device).long()

        # --- D. Run Loss ---
        try:
            with torch.no_grad():
                losses = model.loss(inputs_dict, [data_sample.to(args.device)])
            
            total_loss = sum(v.item() for v in losses.values())
            breakdown = {k: round(v.item(), 4) for k, v in losses.items()}
            results.append((img_file, total_loss, breakdown))
            
            print(f"[{i+1}/{len(img_files)}] {img_file}")
            print(f"   Total: {total_loss:.4f} | Breakdown: {breakdown}")

        except Exception as e:
            print(f"ERROR on {img_file}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"{'Total Loss':<12} | {'Filename':<40} | {'Breakdown'}")
    print("="*80)
    
    results.sort(key=lambda x: x[1], reverse=True)

    out_path = 'loss_rankings_detailed.txt'
    with open(out_path, 'w') as f:
        for fname, loss, details in results:
            detail_str = str(details)
            line = f"{loss:.4f} | {fname:<30} | {detail_str}"
            print(line)
            f.write(line + "\n")
            
    print(f"\nSaved detailed rankings to {out_path}")

if __name__ == '__main__':
    main()


