#!/usr/bin/env bash
set -e

########################
# User config
########################
export PARTITION=research-gpu
export JOB_NAME=fcos3d-train
export GPUS=1
export GPUS_PER_NODE=1
export CPUS_PER_TASK=8
export SRUN_ARGS=""
export PY_ARGS=""

########################
# Paths
########################
export APPCONTAINER="$HOME/cuda121-uv.sif"
export PROJECT_DIR="$HOME/curriculum"
export CONFIG="$PROJECT_DIR/configs/fcos3d.py"
export WORK_DIR="$PROJECT_DIR/work_dirs/fcos3d"

########################
# Apptainer binds
########################
export APPTAINER_BIND=$(tr '\n' ',' <<END
/etc/passwd
/etc/group
/etc/nsswitch.conf
/etc/slurm
/etc/sssd/
/lib64/libnss_sss.so.2:/lib/libnss_sss.so.2
/usr/bin/sacct
/usr/bin/salloc
/usr/bin/sbatch
/usr/bin/scancel
/usr/bin/scontrol
/usr/bin/scrontab
/usr/bin/sinfo
/usr/bin/squeue
/usr/bin/srun
/usr/bin/sshare
/usr/bin/sstat
/usr/bin/strace
/usr/lib64/libmunge.so.2
/usr/lib64/slurm
/var/lib/sss
/var/run/munge:/run/munge
END
)

export APPTAINER_BIND+=",/projects:/projects,/home/sb2ek/uhome:/opt,/scratch:/scratch"

########################
# Output dir
########################
export OUT_DIR="$PROJECT_DIR/mmdetection3d/data/nuscenes_depth_meters"
mkdir -p "$OUT_DIR"
mkdir -p "$WORK_DIR"

cd "$PROJECT_DIR"

########################
# Sanity prints
########################
echo "Running on partition: $PARTITION"
echo "GPUs: $GPUS (per node: $GPUS_PER_NODE)"
echo "Container: $APPCONTAINER"
echo "Config: $CONFIG"
echo "Work dir: $WORK_DIR"
echo "Apptainer bind: $APPTAINER_BIND"

########################
# Launch
########################
srun -p "${PARTITION}" \
    --job-name="${JOB_NAME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks="${GPUS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    apptainer exec --nv "${APPCONTAINER}" \
    python -u mmdetection3d/tools/train.py "${CONFIG}" \
        --work-dir="${WORK_DIR}" \
        --launcher="slurm" \
        ${PY_ARGS}