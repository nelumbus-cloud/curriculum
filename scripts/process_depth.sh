#!/bin/bash
#SBATCH --job-name=depth_process
#SBATCH --partition=research-cpu
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=sb2ek@mtmail.mtsu.edu
#SBATCH --mail-type=END,FAIL
set -e
set -x
mkdir -p logs


export APPCONTAINER="$HOME/cuda121-uv.sif"
export PROJECT_DIR="$HOME/curriculum"
export APPTAINER_BIND="/projects:/projects,/home/sb2ek/uhome:/opt,/scratch:/scratch"
export OUT_DIR=$PROJECT_DIR/mmdetection3d/data/nuscenes_depth_meters
mkdir -p $OUT_DIR
cd $PROJECT_DIR

apptainer exec "$APPCONTAINER" bash -c "\
source /opt/venvs/mmdet/bin/activate && \
python utils/preprocess_depth.py \
  --num-workers 16 \
  --dataroot mmdetection3d/data/nuscenes \
  --depth_root /projects/sb2ek/datasets/nuscenes_depth_mini \
  --out_root /scratch \
  --pkl_path mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl \
  --cam CAM_FRONT"

mkdir -p $OUT_DIR/samples/
rsync -av /scratch/samples/ $OUT_DIR/samples/
    

