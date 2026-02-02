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

set -Eeuo pipefail

module load miniconda
source activate nusc_depth


SCR=/scratch/job_${SLURM_JOB_ID}

DATA=/home/sb2ek/uhome/datasets/nuscenes_mini
#DATA=/projects/sb2ek/datasets/nuscenes
FINAL=/projects/sb2ek/datasets/nuscenes_depth_mini
#FINAL=/projects/sb2ek/datasets/nuscenes_depth
VERSION=v1.0-mini

ls "$DATA"
mkdir -p logs "$FINAL" "$SCR"


python generate_depth_nuscenes.py --data-root "$DATA" --out-root "$SCR" --version "$VERSION"

rsync -a --ignore-existing "$SCR"/ "$FINAL"/


