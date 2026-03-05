#!/bin/bash
#SBATCH --job-name=train_fcos3d
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:A5000:2
#SBATCH --nodelist c19
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=sb2ek@mtmail.mtsu.edu
#SBATCH --mail-type=END,FAIL

set -e
set -x

mkdir -p logs

########################################
# SLURM AUTOMATIC VARIABLES
########################################

# Number of GPUs on this node
GPUS=2

# Total nodes 
NNODES=1

# Rank of this node
NODE_RANK=${SLURM_NODEID:-0}

# Master address = first node in allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Port (random but fixed per job)
MASTER_PORT=29500

########################################
# PATHS
########################################

export APPTAINER_BIND="/opt:/opt,/scratch:/scratch,/projects:/projects,/tmp:/tmp"

export PROJECT_DIR="$HOME/curriculum"
export CONFIG="$PROJECT_DIR/configs/fcos3d.py"
export WORK_DIR="$PROJECT_DIR/mmdetection3d/work_dirs"
export TEMP_WORK_DIR="/scratch/sb2ek"

mkdir -p "$TEMP_WORK_DIR" "$TEMP_WORK_DIR/work_dirs"

mkdir -p "$WORK_DIR"

cd "$PROJECT_DIR"

########################################
# TRAIN
########################################

export APPCONTAINER="$HOME/cuda121-conda.sif"


apptainer exec --nv "$APPCONTAINER" bash -c "
set -xe
source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh && \
conda activate mmdet3d && \
cd $PROJECT_DIR && \
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR && \
cd $PROJECT_DIR/mmdetection3d && \
torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    tools/train.py \
    "$CONFIG" \
    --work-dir="$TEMP_WORK_DIR" \
    --launcher pytorch \
    --cfg-options data_root=data/nuscenes
"

rsync -av --progress $TEMP_WORK_DIR/work_dirs/ $WORK_DIR/