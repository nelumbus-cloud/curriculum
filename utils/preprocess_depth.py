#calibrate depth map in meters


import numpy as np
import os
import mmengine
import argparse
from joblib import Parallel, delayed

import logging

  


def get_depth_extremad(cam_info, lidar_path, w, h):
    
    l2c = np.array(cam_info['lidar2cam']).reshape(4, 4)
    c2i = np.array(cam_info['cam2img']).reshape(3, 3)
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    pts_cam = (np.hstack((pts, np.ones((len(pts), 1)))) @ l2c.T)[:, :3]
    mask_z = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask_z]
    pts_img_hom = pts_cam @ c2i.T
    pts_img = pts_img_hom[:, :2] / pts_img_hom[:, 2:3]
    mask_fov = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h)
    final_z = pts_cam[mask_fov, 2]
    final_img = pts_img[mask_fov]
    return final_z, final_img.astype(np.uint32)



def depth_in_meters(depth_path, pts_img, pts_z, method='p95', p_lo=5, p_hi=95, max_depth_meters=200, eps=1e-6):

    d_map = np.load(depth_path).astype(np.float32)

    # [255,0] to [0,255] depth -> depth-like
    d_map = d_map.max() - d_map

    # normalize
    d01 = (d_map - d_map.min()) / (d_map.max() - d_map.min() + eps)

    h, w = d_map.shape
    uv = pts_img.copy().astype(np.int32)

    uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)

    x = d01[uv[:, 1], uv[:, 0]]
    z = pts_z.astype(np.float32)

    # filter lidar
    mask = (z > 1.0) & (z < max_depth_meters)
    x = x[mask]
    z = z[mask]

    if method == "p95":
        x_lo, x_hi = np.percentile(x, [p_lo, p_hi])
        z_lo, z_hi = np.percentile(z, [p_lo, p_hi])

        a = (z_hi - z_lo) / (x_hi - x_lo + eps)
        b = z_lo - a * x_lo

        depth_meters = a * d01 + b

        logging.info(f"Depth stats: min={depth_meters.min():.2f}, max={depth_meters.max():.2f}")
        return depth_meters
    elif method == "poly2":
        coeff = np.polyfit(x, z, deg=2)
        depth_meters = coeff[0] * d01**2 + coeff[1] * d01 + coeff[2]

        return depth_meters
    else:
        raise NotImplementedError


def process_sample(sample_info, img_width, img_height,args):
    #unpack args
    logging.basicConfig(level=logging.INFO)

    try:
        img_file_name = sample_info['images'][args.cam]['img_path']
        depth_file_name = img_file_name.replace('.jpg', '.npy')
        depth_path = os.path.join(args.depth_dir, depth_file_name)
        #for some reason this is weirdly structure
        lidar_path = os.path.join(args.dataroot, 'samples/LIDAR_TOP', sample_info['lidar_points']['lidar_path'])


        cam_info = sample_info['images'][args.cam]
        z, uv = get_depth_extremad(cam_info, lidar_path, w=img_width, h=img_height)

        depth_map = depth_in_meters(depth_path, uv, z, method=args.method)

        #save depth map
        np.save(os.path.join(args.out_dir, depth_file_name), depth_map)
        logging.info(f"Saved depth map to {os.path.join(args.out_dir, depth_file_name)}")

    except Exception as e:
        logging.error(f"Error processing sample {sample_info['sample_idx']}: {e}")

#Example: 
# python mmdetection3d/utils/preprocess_depth.py \
# --num-workers 8 \
# --dataroot mmdetection3d/data/nuscenes \
# --depth_root  mmdetection3d/data/nuscenes_depth \
# --out_root mmdetection3d/data/nuscenes_depth_calibrated \
# --pkl_path mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl \
# --method poly2 \
# --cam CAM_FRONT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--dataroot", type=str, required=True)
    ap.add_argument("--depth_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--pkl_path", type=str, required=True)
    ap.add_argument("--method", type=str, default="poly2")
    ap.add_argument("--cam", type=str, default="CAM_FRONT")
    args = ap.parse_args()


    #since image is not opened, we pass width and height by hardcoding

    img_width = 1600
    img_height = 900

    #check critical only 
    out_dir = os.path.join(args.out_root, f'samples/{args.cam}')
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir

    #similary
    depth_dir = os.path.join(args.depth_root, f'samples/{args.cam}')
    args.depth_dir = depth_dir

    sample_info = mmengine.load(args.pkl_path)['data_list']
    #get index from workers
    indices = list(range(len(sample_info)))

    results = Parallel(
        n_jobs=args.num_workers,
        backend="loky",
        verbose=10,
    )(delayed(process_sample)(sample_info[i], img_width, img_height, args) for i in indices)

    
