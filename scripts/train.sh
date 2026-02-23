#!/bin/bash
#SBATCH --job-name=fcos3d-train
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e

export APPCONTAINER="$HOME/cuda121-uv.sif"
export PROJECT_DIR="$HOME/curriculum"
export CONFIG="$PROJECT_DIR/configs/fcos3d.py"


export WORK_DIR="$PROJECT_DIR/mmdetection3d/work_dirs"
export TEMP_WORK_DIR="/scratch"
mkdir -p "$TEMP_WORK_DIR"

export APPTAINER_BIND="/projects:/projects,/home/sb2ek/uhome:/opt,/scratch:/scratch"

mkdir -p "$WORK_DIR"

cd "$PROJECT_DIR"


apptainer exec --nv "$APPCONTAINER" bash -c '
    source /opt/venvs/mmdet/bin/activate && \
    export PYTHONPATH=$PYTHONPATH:mmdetection3d && \
    python -u mmdetection3d/tools/train.py "$1" --work-dir="$2" \
    --cfg-options \
    data_root=data/nuscenes \
    train_dataloader.dataset.data_root=mmdetection3d/data/nuscenes \
    val_dataloader.dataset.data_root=mmdetection3d/data/nuscenes \
    val_evaluator.data_root=mmdetection3d/data/nuscenes
' _ "$CONFIG" "$TEMP_WORK_DIR"

rsync -av --progress "$TEMP_WORK_DIR/work_dirs/" "$WORK_DIR/"
