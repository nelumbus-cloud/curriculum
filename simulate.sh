#!/bin/bash
#SBATCH -J rain_sim
#SBATCH -p research-gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00

#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -Eeo pipefail

source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate rain_sim

DATAROOT=/home/sb2ek/uhome/datasets/nuscenes_mini
DEPTHROOT=/projects/sb2ek/datasets/nuscenes_depth_mini
OUT_DIR=/home/sb2ek/curriculum/simulation_outputs


mkdir -p logs "$OUT_DIR"



python simulate.py --dataroot "$DATAROOT" --depth_root "$DEPTHROOT" --output_dir "$OUT_DIR"