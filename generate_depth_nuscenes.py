import os, argparse
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import pipeline
from nuscenes.nuscenes import NuScenes

CAMS = (
    "CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
    "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"
)

def depth_fn(est, img):
    d = np.array(est(img)["depth"], dtype=np.float32)
    if d.shape != (img.height, img.width):
        raise RuntimeError("shape mismatch")
    return d

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--version", default="v1.0-trainval")
ap.add_argument("--out-root", default="/tmp/nuscenes_depth")
ap.add_argument("--model", default="LiheYoung/depth-anything-small-hf")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

nusc = NuScenes(args.version, args.data_root, verbose=False)
est = pipeline("depth-estimation", model=args.model, device=args.device, use_fast=True)

for scene in nusc.scene:
    tok = scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        for cam in CAMS:
            sd = nusc.get("sample_data", s["data"][cam])
            img_p = os.path.join(args.data_root, sd["filename"])
            out_p = Path(args.out_root, sd["filename"]).with_suffix(".npy")
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if not out_p.exists():
                img = Image.open(img_p).convert("RGB")
                np.save(out_p, depth_fn(est, img))
        tok = s["next"]

