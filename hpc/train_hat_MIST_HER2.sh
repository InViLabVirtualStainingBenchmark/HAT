#!/bin/bash
#SBATCH --job-name=hat_MIST_HER2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/hat_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/hat_%j.err

set -euo pipefail

CONTAINER="$VSC_SCRATCH/containers/basicsr_nvidia.sif"
CODE_DIR="$VSC_DATA/projects/code"
DATA_SQSH="$VSC_SCRATCH/MIST_HER2.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/hat_MIST_HER2"

mkdir -p "$OUTPUT_DIR"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$OUTPUT_DIR/gpu_usage.csv" &
GPU_LOG_PID=$!

srun apptainer exec \
    --nv \
    -B "$CODE_DIR":/code \
    -B "$DATA_SQSH":/data:image-src=/ \
    -B "$OUTPUT_DIR":/output \
    "$CONTAINER" \
    bash /code/HAT/train_hat.sh options/train/train_HAT_MIST_HER2_HPC.yml

kill $GPU_LOG_PID || true
