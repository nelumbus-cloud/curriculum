import os
import torch
import pandas as pd
import numpy as np
import sys
import mmcv
from mmengine.config import Config, ConfigDict
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.datasets import NuScenesDataset

# 1. Register Modules
register_all_modules()

# --- CONFIGURATION ---
data_root = 'data/nuscenes/'
ann_file = 'nuscenes_infos_train.pkl'

# 2. Load Config & Model
mmdet_path = '/home/sb2ek/curriculum/mmdetection3d'
config_file = os.path.join(mmdet_path, 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py')
checkpoint_file = 'checkpoints/fcos3d_baseline.pth'

cfg = Config.fromfile(config_file)
cfg.model.train_cfg = ConfigDict(allowed_border=0, code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05], pos_weight=-1, debug=False)

model = init_model(cfg, checkpoint_file, device='cuda:0')

# 3. INITIALIZE DATASET
# We explicitly set version to v1.0-mini just in case, though loading by ann_file usually overrides it
print("Initializing Dataset...")
dataset = NuScenesDataset(
    data_root=data_root,
    ann_file=ann_file,
    version='v1.0-mini', 
    data_prefix=dict(
        pts='samples/LIDAR_TOP',
        CAM_FRONT='samples/CAM_FRONT', 
        CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
        CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
        CAM_BACK='samples/CAM_BACK',
        CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
        CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
        sweeps='sweeps/LIDAR_TOP'
    ),
    pipeline=[], 
    test_mode=False,
    metainfo=dict(classes=('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'))
)
dataset.full_init()
print(f"Dataset Loaded: {len(dataset)} samples found.")

# 4. MODIFY PIPELINE (The Bypass Trick)
# We filter out 'LoadImageFromFileMono3D' because it keeps crashing on your paths.
print("Building pipeline (Skipping LoadImageFromFileMono3D)...")
new_pipeline_cfg = []
for transform in cfg.train_pipeline:
    if transform['type'] != 'LoadImageFromFileMono3D':
        new_pipeline_cfg.append(transform)

pipeline = Compose(new_pipeline_cfg)

print("\n🚀 Starting Robust Loss Analysis...")
results_list = []

# 5. LOOP
for i in range(len(dataset)):
    base_filename = "Unknown"
    try:
        # A. Get Info (Annotations)
        data_info = dataset.get_data_info(i)
        
        # B. Filter for CAM_FRONT
        process_info = None
        
        # Standard Nested Structure
        if 'images' in data_info and 'CAM_FRONT' in data_info['images']:
            cam_info = data_info['images']['CAM_FRONT']
            data_info.update(cam_info)
            process_info = data_info
        # Flat Structure
        elif 'img_path' in data_info:
            if 'CAM_FRONT' in data_info['img_path']:
                process_info = data_info

        if process_info is None:
            continue

        # C. FIND IMAGE FILE
        base_filename = os.path.basename(process_info.get('img_path', ''))
        
        # Look in all possible mini/full locations
        real_path = ""
        possible_paths = [
            os.path.join(data_root, 'samples', 'CAM_FRONT', base_filename),
            os.path.join(data_root, 'v1.0-mini', 'samples', 'CAM_FRONT', base_filename),
            os.path.join(data_root, base_filename)
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                real_path = p
                break
        
        if not real_path:
            # File missing (normal for mini dataset)
            continue

        # D. MANUAL LOAD (Bypass the Loader Error)
        # We load the image using MMCV, which is what the pipeline would have done
        img = mmcv.imread(real_path)
        
        # We manually populate the dictionary keys the pipeline expects
        process_info['img'] = img
        process_info['img_shape'] = img.shape[:2]
        process_info['ori_shape'] = img.shape[:2]
        process_info['img_path'] = real_path
        process_info['filename'] = real_path
        process_info['ori_filename'] = base_filename
        
        # Ensure array types
        if 'cam2img' in process_info: process_info['cam2img'] = np.array(process_info['cam2img'])
        if 'lidar2cam' in process_info: process_info['lidar2cam'] = np.array(process_info['lidar2cam'])

        # E. Run Remaining Pipeline
        data_batch = pipeline(process_info)
        data_batch = pseudo_collate([data_batch])
        data_batch = model.data_preprocessor(data_batch, training=True)
        
        with torch.no_grad():
            losses = model.forward(data_batch['inputs'], data_batch['data_samples'], mode='loss')
            total_loss = sum(l.item() for l in losses.values() if isinstance(l, torch.Tensor))
            
            row = {
                'filename': base_filename,
                'total_loss': total_loss,
                'loss_cls': losses.get('loss_cls', torch.tensor(0.0)).item(),
                'loss_bbox': losses.get('loss_bbox', torch.tensor(0.0)).item(),
                'loss_attr': losses.get('loss_attr', torch.tensor(0.0)).item()
            }
            results_list.append(row)
            print(f"✅ [{len(results_list)}] {base_filename} | Loss: {total_loss:.4f}")

    except Exception as e:
        print(f"⚠️ Error on {base_filename}: {e}")
        continue

print(f"\nDone. Processed {len(results_list)} images.")

if results_list:
    df = pd.DataFrame(results_list)
    df = df.sort_values(by='total_loss', ascending=False)
    df.to_csv('fcos3d_losses.csv', index=False)
    print("Saved results to fcos3d_losses.csv")
else:
    print("❌ No images processed.")
