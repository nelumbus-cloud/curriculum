import cv2
import numpy as np
import os
import mmengine
import argparse

import logging

logging.basicConfig(level=logging.INFO)   

def depth_in_meters(depth_path, pts_img, pts_z, method='p95', p_lo=5, p_hi=95, max_depth_meters=100, eps=1e-6):

    d_map = np.load(depth_path).astype(np.float32)

    # inverse depth -> depth-like
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
    mask = (z > 1.0) & (z < 80.0)
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

        logging.info(f"[poly2] Depth stats: min={depth_meters.min():.2f}, max={depth_meters.max():.2f}")
        return depth_meters
    else:
        raise NotImplementedError
#beta >= 2.29e-3 
def add_fog_beta(image, depth_in_km, beta=0.02, airlight=220):
    """
    Applies fog attenuation: I(x) = I(x) * exp(-beta * d(x)) + L * (1 - exp(-beta * d(x)))
    """
    # Transmission map
    t = np.exp(-beta * depth_in_km)
    
    # Expand dims for broadcasting if image is RGB (H, W, 3)
    if len(image.shape) == 3 and len(t.shape) == 2:
        t = t[..., np.newaxis]
        
    img_float = image.astype(np.float32)
    
    # Fog equation
    foggy_image = img_float * t + airlight * (1 - t)
    
    return np.clip(foggy_image, 0, 255).astype(np.uint8)



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


@MMENGINE.register_module()
def add_fog(cfg):
    # z, uv = get_depth_extremad(cam_info, lidar_path, w=img_width, h=img_height)

    # depth_map = depth_in_meters(depth_path, uv, z, method='poly2')

    # #foggy image
    # image = cv2.imread(os.path.join(args.dataroot, f'samples/{cam}/{img_file_name}'))

    # #
    # foggy_image = add_fog_beta(image, depth_map, beta=0.05)
    pass

if __name__ == '__main__':
    # Example usage
    pkl_path = 'data/nuscenes/nuscenes_infos_train.pkl'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--pkl_path', default='data/nuscenes/nuscenes_infos_train.pkl')
    parser.add_argument('--depth_root', default='data/nuscenes_depth')
    parser.add_argument('--out_root', default='outputs')

    args = parser.parse_args()
    dataroot = args.dataroot

    #let sample_idx be 0
    sample_idx = 0
    cam = 'CAM_FRONT'
    img_width = 1600
    img_height = 900

    sample_info = mmengine.load(pkl_path)['data_list'][sample_idx]

    # depthPath is left after striping image path and adding after depth_root
    #info[‘images’][‘CAM_XXX’][‘img_path’]: The filename of the image.

    img_file_name = sample_info['images'][cam]['img_path']
    depth_file_name = img_file_name.replace('.jpg', '.npy')
    depth_path = os.path.join(args.depth_root, f'samples/{cam}/{depth_file_name}')
    #for some reason this is weirdly structure
    lidar_path = os.path.join(args.dataroot, 'samples/LIDAR_TOP', sample_info['lidar_points']['lidar_path'])


    cam_info = sample_info['images'][cam]
    z, uv = get_depth_extremad(cam_info, lidar_path, w=img_width, h=img_height)

    depth_map = depth_in_meters(depth_path, uv, z, method='poly2')

    #foggy image
    image = cv2.imread(os.path.join(args.dataroot, f'samples/{cam}/{img_file_name}'))

    #
    foggy_image = add_fog_beta(image, depth_map, beta=0.05)
    #check max and min of fog beta
    logging.info(f"Max foggy image: {foggy_image.max()}")
    logging.info(f"Min foggy image: {foggy_image.min()}")
    #save foggy image
    mmengine.utils.mkdir_or_exist(os.path.join(args.out_root, f'{cam}'))
    cv2.imwrite(os.path.join(args.out_root, f'{cam}/{img_file_name}'), foggy_image)
    
    

