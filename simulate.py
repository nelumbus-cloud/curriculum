import math
import os.path as osp
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


DEFAULT_RAIN_MM_HR = 100.0
DROP_DIAMETER_MM = 2.0
RADIUS_M = 100.0
HEIGHT_M = 4.0
FREQUENCY_HZ = 120.0


def convert_map_to_meters(
    depth_map,
    u, v, z_lidar,
    scaling_method="p95",
    max_range_m=100.0, #100m by inspection
    eps=1e-6,
    p_lo=5, p_hi=95
):
    d = depth_map.astype(np.float32)
    d = d.max() - d
    d01 = (d - d.min()) / (d.max() - d.min() + eps)

    x = d01[v.astype(np.int32), u.astype(np.int32)]
    z = z_lidar.astype(np.float32)

    if scaling_method == "none":
        depth_m = d01 * max_range_m
    elif scaling_method == "median":
        s = np.median(z) / (np.median(x) + eps)
        depth_m = s * d01
    elif scaling_method == "p95":
        x_lo, x_hi = np.percentile(x, [p_lo, p_hi])
        z_lo, z_hi = np.percentile(z, [p_lo, p_hi])
        a = (z_hi - z_lo) / (x_hi - x_lo + eps)
        b = z_lo - a * x_lo
        depth_m = a * d01 + b
    else:
        raise ValueError("scaling_method must be one of: none, median, p95")

    depth_m = np.clip(depth_m, 0.0, max_range_m).astype(np.float32)
    return depth_m


def get_nusc_depth_from_lidar(nusc, camera_token, pointsensor_token):
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])

    pc = LidarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    z = pc.points[2, :]

    pts = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    u, v = pts[0, :], pts[1, :]

    mask = np.ones(z.shape[0], dtype=bool)
    mask &= (z > 1.0)
    mask &= (u > 1) & (u < im.size[0] - 1)
    mask &= (v > 1) & (v < im.size[1] - 1)

    u = u[mask].astype(np.int32)
    v = v[mask].astype(np.int32)
    z = z[mask].astype(np.float32)
    return u, v, z


def terminal_velocity(D_mm: torch.Tensor) -> torch.Tensor:
    v = 9.65 - 10.3 * torch.exp(-0.6 * D_mm)
    return torch.clamp(v, min=0.0)


def sample_disk_xy(n: int, R: float, device):
    u = torch.rand(n, device=device)
    th = 2.0 * math.pi * torch.rand(n, device=device)
    r = R * torch.sqrt(u)
    return r * torch.cos(th), r * torch.sin(th)


def get_cam_K_Rt(nusc, camera_token, device):
    cam_sd = nusc.get("sample_data", camera_token)
    cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32, device=device)

    R = torch.tensor(Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32, device=device)
    t = torch.tensor(cs["translation"], dtype=torch.float32, device=device)
    R_ec = R.t()
    t_ec = -R.t() @ t
    img_path = osp.join(nusc.dataroot, cam_sd["filename"])
    return K, R_ec, t_ec, img_path, cam_sd


def project_points(K, Xc):
    X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    eps = 1e-6
    ok = Z > eps
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (X / (Z + eps)) + cx
    v = fy * (Y / (Z + eps)) + cy
    return u, v, ok

#todo : add rain streak database for more realistic rendering

def rasterize_streaks(rain, u0, v0, u1, v1, alpha_per=0.08, samples=16):
    H, W = rain.shape
    t = torch.linspace(0.0, 1.0, steps=samples, device=rain.device).view(1, samples)
    uu = u0.view(-1, 1) * (1 - t) + u1.view(-1, 1) * t
    vv = v0.view(-1, 1) * (1 - t) + v1.view(-1, 1) * t
    ui = torch.round(uu).long()
    vi = torch.round(vv).long()
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[inb]
    vi = vi[inb]
    idx = vi * W + ui
    rain.view(-1).index_add_(0, idx, torch.full_like(idx, alpha_per / samples, dtype=torch.float32))
    return rain


@torch.no_grad()
def add_rain_single_frame(
    nusc,
    camera_token: str,
    lidar_token: str,
    depth_root: str ,
    rain_mm_hr: float = DEFAULT_RAIN_MM_HR,
    drop_diam_mm: float = DROP_DIAMETER_MM,
    radius_m: float = RADIUS_M,
    height_m: float = HEIGHT_M,
    frequency_hz: float = FREQUENCY_HZ,
    base_lambda_per_mmhr: float = 2e-6,
    max_particles: int = 20000,
    device: str = "cuda" # force cuda for speed
):
    device = torch.device(device)
    K, R_ec, t_ec, img_path, cam_sd = get_cam_K_Rt(nusc, camera_token, device)

    rgb = Image.open(img_path).convert("RGB")
    W, H = rgb.size
    img = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).to(device).permute(2, 0, 1).contiguous()

    depth_path = (Path(depth_root) / cam_sd["filename"]).with_suffix(".npy")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    depth_raw = np.load(depth_path)

    u, v, z_lidar = get_nusc_depth_from_lidar(nusc, camera_token, lidar_token)
    depth_m = convert_map_to_meters(depth_raw, u, v, z_lidar, scaling_method="p95", max_range_m=100.0)

    assert depth_m.shape == (H, W), f"Depth shape mismatch: {depth_m.shape} != ({H}, {W})"
       
    exp_s = 1.0 / float(frequency_hz)
    area = math.pi * (radius_m ** 2)
    expected = (base_lambda_per_mmhr * float(rain_mm_hr)) * area * exp_s
    n_new = int(np.random.poisson(expected))
    n = min(n_new, int(max_particles))
    if n <= 0:
        return img, {"depth_path": str(depth_path), "spawn_expected": expected, "spawn_poisson": n_new, "spawn_used": 0}

    x, y = sample_disk_xy(n, radius_m, device)
    z0 = torch.rand(n, device=device) * float(height_m)
    p0_ego = torch.stack([x, y, z0], dim=1)

    D = torch.full((n,), float(drop_diam_mm), device=device, dtype=torch.float32)
    vt = terminal_velocity(D)
    w_ego = torch.stack([torch.zeros_like(vt), torch.zeros_like(vt), -vt], dim=1)
    p1_ego = p0_ego + w_ego * exp_s

    p0_cam = (R_ec @ p0_ego.t()).t() + t_ec.view(1, 3)
    p1_cam = (R_ec @ p1_ego.t()).t() + t_ec.view(1, 3)

    u0, v0, ok0 = project_points(K, p0_cam)
    u1, v1, ok1 = project_points(K, p1_cam)
    ok = ok0 & ok1

    u0, v0, u1, v1, p0_cam = u0[ok], v0[ok], u1[ok], v1[ok], p0_cam[ok]
    if u0.numel() == 0:
        return img, {"depth_path": str(depth_path), "spawn_expected": expected, "spawn_poisson": n_new, "spawn_used": n}

    ui = torch.clamp(torch.round(u0).long(), 0, W - 1)
    vi = torch.clamp(torch.round(v0).long(), 0, H - 1)
    depth_at = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)[vi, ui]
    keep = torch.isfinite(depth_at) & (p0_cam[:, 2] <= (depth_at + 2.0))
    u0, v0, u1, v1 = u0[keep], v0[keep], u1[keep], v1[keep]

    rain = torch.zeros((H, W), device=device, dtype=torch.float32)
    cap_scale = float(n_new) / float(max(1, n))
    alpha_per = 0.08 * cap_scale
    rain = rasterize_streaks(rain, u0, v0, u1, v1, alpha_per=alpha_per, samples=16)
    rain = torch.clamp(rain, 0.0, 1.0)

    a = torch.clamp(rain * 0.9, 0.0, 0.9).unsqueeze(0).repeat(3, 1, 1)
    rainy = img * (1.0 - a) + torch.ones_like(img) * a

    return rainy, {"depth_path": str(depth_path), "spawn_expected": expected, "spawn_poisson": n_new, "spawn_used": n}

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--dataroot", type=str)
    arg.add_argument("--depth_root", type=str)
    arg.add_argument("--output_dir", type=str)
    args = arg.parse_args()
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot=args.dataroot, verbose=True)
    sample = nusc.sample[0]
    cam_token = sample["data"]["CAM_FRONT"]
    lidar_token = sample["data"]["LIDAR_TOP"]

    rainy_img, info = add_rain_single_frame(
        nusc,
        camera_token=cam_token,
        lidar_token=lidar_token,
        depth_root=args.depth_root,
    )
    print(info)
    rainy_img_np = (rainy_img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    rainy_pil = Image.fromarray(rainy_img_np)
    output_path = f"{args.output_dir}/rainy_image.png"
    rainy_pil.save(output_path)
    print(f"Saved rainy image to {output_path}")
