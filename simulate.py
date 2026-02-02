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


CAMS = ("CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT")

def terminal_velocity(D_mm):
    return torch.clamp(9.65 - 10.3 * torch.exp(-0.6 * D_mm), min=0.0)

def sample_disk_xy(n, R, device):
    u = torch.rand(n, device=device)
    th = 2.0 * math.pi * torch.rand(n, device=device)
    r = R * torch.sqrt(u)
    return r * torch.cos(th), r * torch.sin(th)

def pose_world_from_ego(nusc, ego_pose_token, device):
    pose = nusc.get("ego_pose", ego_pose_token)
    R_we = torch.tensor(Quaternion(pose["rotation"]).rotation_matrix, dtype=torch.float32, device=device)
    t_we = torch.tensor(pose["translation"], dtype=torch.float32, device=device)
    return R_we, t_we

def cam_extrinsic_ego_to_cam(nusc, cam_sd, device):
    cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    R_se = torch.tensor(Quaternion(cs["rotation"]).rotation_matrix, dtype=torch.float32, device=device)
    t_se = torch.tensor(cs["translation"], dtype=torch.float32, device=device)
    R_ec = R_se.t()
    t_ec = -R_se.t() @ t_se
    K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32, device=device)
    return K, R_ec, t_ec

def build_global_to_cam(nusc, cam_sd, device):
    R_we, t_we = pose_world_from_ego(nusc, cam_sd["ego_pose_token"], device)
    R_ew = R_we.t()
    t_ew = -R_we.t() @ t_we
    K, R_ec, t_ec = cam_extrinsic_ego_to_cam(nusc, cam_sd, device)
    A = R_ec @ R_ew
    b = (R_ec @ t_ew) + t_ec
    return K, A, b

def project_batch(Ks, Xc):
    X, Y, Z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    eps = 1e-6
    ok = Z > eps
    fx = Ks[:, 0, 0].view(-1, 1)
    fy = Ks[:, 1, 1].view(-1, 1)
    cx = Ks[:, 0, 2].view(-1, 1)
    cy = Ks[:, 1, 2].view(-1, 1)
    u = fx * (X / (Z + eps)) + cx
    v = fy * (Y / (Z + eps)) + cy
    return u, v, ok

def rasterize_lines(H, W, u0, v0, u1, v1, alpha, samples, device):
    rain = torch.zeros((H, W), device=device, dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, steps=samples, device=device).view(1, samples)
    uu = u0.view(-1, 1) * (1 - t) + u1.view(-1, 1) * t
    vv = v0.view(-1, 1) * (1 - t) + v1.view(-1, 1) * t
    ui = torch.round(uu).long()
    vi = torch.round(vv).long()
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = (vi[inb] * W + ui[inb]).view(-1)
    rain.view(-1).index_add_(0, idx, torch.full_like(idx, alpha / samples, dtype=torch.float32))
    return torch.clamp(rain, 0.0, 1.0)

@torch.no_grad()
def add_rain_one_sample_batched(
    nusc,
    sample_token,
    depth_root,
    lidar_channel="LIDAR_TOP",
    rain_mm_hr=DEFAULT_RAIN_MM_HR,
    drop_diam_mm=DROP_DIAMETER_MM,
    radius_m=RADIUS_M,
    height_m=HEIGHT_M,
    frequency_hz=FREQUENCY_HZ,
    base_lambda_per_mmhr=2e-6,
    max_particles=20000,
    z_buffer_margin_m=2.0,
    samples_per_streak=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"][lidar_channel]
    lidar_sd = nusc.get("sample_data", lidar_token)

    exp_s = 1.0 / float(frequency_hz)
    area = math.pi * radius_m * radius_m
    expected = (base_lambda_per_mmhr * float(rain_mm_hr)) * area * exp_s
    n_new = int(np.random.poisson(expected))
    n = min(n_new, int(max_particles))
    if n <= 0:
        return {}, {"spawn_expected": expected, "spawn_poisson": n_new, "spawn_used": 0}

    x, y = sample_disk_xy(n, radius_m, device)
    z0 = torch.rand(n, device=device) * float(height_m)
    p0_ego = torch.stack([x, y, z0], dim=1)
    D = torch.full((n,), float(drop_diam_mm), device=device, dtype=torch.float32)
    vt = terminal_velocity(D)
    p1_ego = p0_ego + torch.stack([torch.zeros_like(vt), torch.zeros_like(vt), -vt], dim=1) * exp_s

    R_wE0, t_wE0 = pose_world_from_ego(nusc, lidar_sd["ego_pose_token"], device)
    p0_w = (R_wE0 @ p0_ego.t()).t() + t_wE0.view(1, 3)
    p1_w = (R_wE0 @ p1_ego.t()).t() + t_wE0.view(1, 3)

    Ks, As, bs, cam_sds = [], [], [], []
    imgs, sizes, depth_paths = [], [], []
    for cam_name in CAMS:
        cam_token = sample["data"][cam_name]
        cam_sd = nusc.get("sample_data", cam_token)
        cam_sds.append(cam_sd)

        K, A, b = build_global_to_cam(nusc, cam_sd, device)
        Ks.append(K); As.append(A); bs.append(b)

        img_path = osp.join(nusc.dataroot, cam_sd["filename"])
        rgb = Image.open(img_path).convert("RGB")
        W, H = rgb.size
        sizes.append((H, W))
        imgs.append(torch.from_numpy(np.asarray(rgb, np.float32) / 255.0).to(device).permute(2, 0, 1).contiguous())

        dp = (Path(depth_root) / cam_sd["filename"]).with_suffix(".npy")
        depth_paths.append(dp)

    Ks = torch.stack(Ks, dim=0)          # (6,3,3)
    As = torch.stack(As, dim=0)          # (6,3,3)
    bs = torch.stack(bs, dim=0)          # (6,3)

    p0_cam = torch.einsum("cij,nj->cni", As, p0_w) + bs[:, None, :]
    p1_cam = torch.einsum("cij,nj->cni", As, p1_w) + bs[:, None, :]

    u0, v0, ok0 = project_batch(Ks, p0_cam)
    u1, v1, ok1 = project_batch(Ks, p1_cam)
    ok = ok0 & ok1

    out = {}
    info = {"spawn_expected": expected, "spawn_poisson": n_new, "spawn_used": n}

    for ci, cam_name in enumerate(CAMS):
        img = imgs[ci]
        H, W = sizes[ci]
        dp = depth_paths[ci]
        if not dp.exists():
            raise FileNotFoundError(f"Depth not found: {dp}")

        depth_raw = np.load(dp)
        cam_token = sample["data"][cam_name]
        u_l, v_l, z_l = get_nusc_depth_from_lidar(nusc, cam_token, lidar_token)
        depth_m = convert_map_to_meters(depth_raw, u_l, v_l, z_l, scaling_method="p95", max_range_m=100.0)

        if depth_m.shape != (H, W):
            dm = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)
            dm = F.interpolate(dm, size=(H, W), mode="nearest").squeeze().cpu().numpy()
            depth_m = dm.astype(np.float32)

        depth_t = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)

        m = ok[ci]
        if m.sum() == 0:
            out[cam_name] = img
            continue

        uu0 = u0[ci][m]; vv0 = v0[ci][m]
        uu1 = u1[ci][m]; vv1 = v1[ci][m]
        p0c = p0_cam[ci][m]

        ui = torch.clamp(torch.round(uu0).long(), 0, W - 1)
        vi = torch.clamp(torch.round(vv0).long(), 0, H - 1)
        depth_at = depth_t[vi, ui]
        keep = torch.isfinite(depth_at) & (p0c[:, 2] <= (depth_at + float(z_buffer_margin_m)))
        uu0, vv0, uu1, vv1 = uu0[keep], vv0[keep], uu1[keep], vv1[keep]

        cap_scale = float(n_new) / float(max(1, n))
        alpha = 0.08 * cap_scale
        rain = rasterize_lines(H, W, uu0, vv0, uu1, vv1, alpha=alpha, samples=samples_per_streak, device=device)

        a = torch.clamp(rain * 0.9, 0.0, 0.9).unsqueeze(0).repeat(3, 1, 1)
        out[cam_name] = img * (1.0 - a) + torch.ones_like(img) * a

    return out, info



if __name__ == "__main__":
    parseer = argparse.ArgumentParser()
    # parse dataroot, depth_root, output_dir, data version
    parseer.add_argument("--dataroot", type=str, required=True)
    parseer.add_argument("--depth_root", type=str, required=True)
    parseer.add_argument("--output_dir", type=str, required=True)
    parseer.add_argument("--data_version", type=str, default="v1.0-mini")
    args = parseer.parse_args()

    nusc = NuScenes(version=args.data_version, dataroot=args.dataroot, verbose=False)
    
    #first sample tests
    sample_token = scene0["first_sample_token"]

    rainy, info = add_rain_one_sample_batched(
        nusc=nusc,
        sample_token=sample_token,
        depth_root=args.depth_root,
        rain_mm_hr=100.0,
        drop_diam_mm=2.0,
        radius_m=100.0,
        height_m=4.0,
        frequency_hz=120.0,
        device="cuda"  # or "cpu"
    )

    print(info)

    # save all 6 rainy camera images    
    timestamp = time.time() # current time
    file_name = args.output_dir + "/" + str(timestamp) + ".png"
    for cam_name, img_t in rainy.items():
        arr = (img_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(file_name)