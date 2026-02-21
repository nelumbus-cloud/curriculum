
from utils.preprocess_depth import get_depth_extremad, depth_in_meters
import argparse
import mmengine
import os
import cv2
import numpy as np
import logging


#beta >= 2.29e-3 per m
def add_fog_beta(image, depth_in_meters, beta=0.02, airlight=220):
    """
    Applies fog attenuation: I(x) = I(x) * exp(-beta * d(x)) + L * (1 - exp(-beta * d(x)))
    """
    # Transmission map
    t = np.exp(-beta * depth_in_meters)
    
    # Expand dims for broadcasting if image is RGB (H, W, 3)
    if len(image.shape) == 3 and len(t.shape) == 2:
        t = t[..., np.newaxis]
        
    img_float = image.astype(np.float32)
    
    # Fog equation
    foggy_image = img_float * t + airlight * (1 - t)
    
    return np.clip(foggy_image, 0, 255).astype(np.uint8)

def estimate_airlight(image, patch_size=15, brightest_fraction=1/1000):
    """
    image: HxWx3 BGR uint8 image
    patch_size: size of erosion patch
    brightest_fraction: fraction of brightest dark-channel pixels
    """

    H, W, _ = image.shape
    num_pixels = H * W

    # 1. Dark channel (min over channels + erosion)
    min_channel = np.min(image, axis=2)
    kernel = np.ones((patch_size, patch_size), np.uint8)
    dark_channel = cv2.erode(min_channel, kernel)

    # 2. Pick top brightest pixels in dark channel
    brightest_count = max(1, int(brightest_fraction * num_pixels))
    dark_vec = dark_channel.reshape(-1)
    brightest_indices = np.argpartition(dark_vec, -brightest_count)[-brightest_count:]

    # 3. Among them, pick highest grayscale intensity in original image
    gray_vec = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(-1)
    best_idx = brightest_indices[np.argmax(gray_vec[brightest_indices])]

    # 4. Recover atmospheric light
    row = best_idx // W
    col = best_idx % W
    airlight = image[row, col].astype(np.float32)

    return airlight, (row, col)



if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    
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

    #estimate airlight
    airlight, _ = estimate_airlight(image)
    foggy_image = add_fog_beta(image, depth_map, beta=0.05, airlight=airlight)
    #check max and min of fog beta
    logging.info(f"Max foggy image: {foggy_image.max()}")
    logging.info(f"Min foggy image: {foggy_image.min()}")
    #save foggy image
    mmengine.utils.mkdir_or_exist(os.path.join(args.out_root, f'{cam}'))
    cv2.imwrite(os.path.join(args.out_root, f'{cam}/{img_file_name}'), foggy_image)
    
    

