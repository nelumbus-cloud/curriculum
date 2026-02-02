#!/bin/bash
#SBATCH -J nusc_depth
#SBATCH -p research-gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sb2ek@mtmail.mtsu.edu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


DATAROOT=/home/sb2ek/uhome/datasets/nuscenes_mini
DEPTHROOT=/projects/sb2ek/datasets/nuscenes_depth_mini
OUT_DIR=/home/sb2ek/curriculum/simulation_outputs

set -Eeuo pipefail

mkdir -p logs "$OUT_DIR"

module load miniconda

source activate rain_sim

python simulate.py --dataroot "$DATAROOT" --depth_root "$DEPTHROOT" --output_dir "$OUT_DIR"