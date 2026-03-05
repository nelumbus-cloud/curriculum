#!/bin/bash
#SBATCH --job-name=depth_process
#SBATCH --partition=research-gpu
#SBATCH --nodelist c13
#SBATCH --gres=gpu:2080Ti:2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=sb2ek@mtmail.mtsu.edu
#SBATCH --mail-type=END,FAIL
set -e
set -x
mkdir -p logs

export APPCONTAINER="$HOME/cuda121-conda.sif"
export PROJECT_DIR="$HOME/curriculum"
export APPTAINER_BIND="/projects:/projects,/opt:/opt,/scratch:/scratch"
export DATA_ROOT=mmdetection3d/data/nuscenes
export OUT_DIR=$PROJECT_DIR/mmdetection3d/data/nuscenes_depth_meters
mkdir -p $OUT_DIR
cd $PROJECT_DIR


apptainer exec --nv "$APPCONTAINER" bash -c "\
source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh && \
conda activate nusc_depth && \
python generate_meter_depth.py \
  --data-root $DATA_ROOT \
  --version v1.0-trainval \
  --out-root $OUT_DIR \\
  --model LiheYoung/depth-anything-small-hf \
  --method poly2"

