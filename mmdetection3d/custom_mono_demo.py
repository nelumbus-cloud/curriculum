import argparse
import os
import os.path as osp
import mmengine
import copy
import traceback
import numpy as np
from mmdet3d.apis import MonoDet3DInferencer

def parse_args():
    parser = argparse.ArgumentParser(description='Robust Mono3D Demo with Matrix Fix')
    parser.add_argument('img_dir', help='Directory containing images to test')
    parser.add_argument('ann_file', help='The .pkl file containing metadata')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    
    # Optional args
    parser.add_argument('--cam-type', default='CAM_FRONT', help='Camera type')
    parser.add_argument('--out-dir', default='outputs/final_results', help='Where to save results')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Bbox score threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Annotations
    print(f"Loading annotations from {args.ann_file}...")
    full_infos = mmengine.load(args.ann_file)
    data_list = full_infos['data_list']
    metainfo = full_infos.get('metainfo', {})
    
    # 2. Build Index
    print(f"Indexing metadata for {args.cam_type}...")
    file_index = {}
    for item in data_list:
        if args.cam_type in item['images']:
            path_in_pkl = item['images'][args.cam_type]['img_path']
            fname = osp.basename(path_in_pkl)
            file_index[fname] = item
    print(f"Indexed {len(file_index)} samples.")

    # 3. Initialize Model
    print("Initializing model...")
    inferencer = MonoDet3DInferencer(args.config, args.checkpoint, args.device)
    
    # 4. Process Images
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Found {len(img_files)} images in {args.img_dir}")
    
    temp_pkl_path = osp.join(args.out_dir, 'temp_single_info.pkl')

    for i, img_file in enumerate(img_files):
        if img_file not in file_index:
            print(f"[{i+1}/{len(img_files)}] SKIP: {img_file} not found in annotations.")
            continue
            
        print(f"[{i+1}/{len(img_files)}] Inference: {img_file}")
        
        try:
            # Prepare metadata
            matched_info = copy.deepcopy(file_index[img_file])
            
            # --- PATCH: CALCULATE LIDAR2IMG MANUALLY ---
            # Get the matrices
            cam2img_3x3 = np.array(matched_info['images'][args.cam_type]['cam2img'], dtype=np.float32)
            lidar2cam_4x4 = np.array(matched_info['images'][args.cam_type]['lidar2cam'], dtype=np.float32)
            
            # Pad intrinsic to 4x4
            # [ K  0 ]
            # [ 0  1 ]
            cam2img_4x4 = np.eye(4, dtype=np.float32)
            cam2img_4x4[:3, :3] = cam2img_3x3
            
            # Calculate lidar2img
            lidar2img_4x4 = cam2img_4x4 @ lidar2cam_4x4
            
            # Inject back into info
            # We inject it into BOTH the requested camera AND the 'CAM2' hack
            matched_info['images'][args.cam_type]['lidar2img'] = lidar2img_4x4.tolist()
            matched_info['images'][args.cam_type]['img_path'] = img_file
            
            # THE HACK: Duplicate to CAM2
            matched_info['images']['CAM2'] = copy.deepcopy(matched_info['images'][args.cam_type])
            
            # Write temp pkl
            mini_info = {
                'metainfo': metainfo,
                'data_list': [matched_info]
            }
            mmengine.dump(mini_info, temp_pkl_path)
            
            # Run Inference
            full_img_path = osp.join(args.img_dir, img_file)
            inferencer(
                inputs=dict(img=full_img_path, infos=temp_pkl_path),
                out_dir=args.out_dir,
                cam_type=args.cam_type,
                pred_score_thr=args.score_thr,
                show=False, 
                no_save_vis=False
            )
        except Exception as e:
            print(f"ERROR processing {img_file}: {e}")
            traceback.print_exc()

    # Cleanup
    if osp.exists(temp_pkl_path):
        os.remove(temp_pkl_path)
    print("Done.")

if __name__ == '__main__':
    main()



