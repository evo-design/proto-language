#!/bin/bash
#SBATCH --job-name=evocas9_rejection_sampling
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_batch,gpu_batch_high_mem
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/slurm/evocas9_rejection_sampling_%j.log
#SBATCH --error=logs/slurm/evocas9_rejection_sampling_%j.log

echo "=========================================="
echo "evocas9_rejection_sampling.py — 2000 samples per combo"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU')"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate gpro

set -euo pipefail

cd /home/brianhie/proto-language

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

mkdir -p logs/slurm

# 2000 samples × 3 temps × 2 top_ks = 12000 total sequences
python examples/scripts/evocas9_rejection_sampling.py \
    --n-samples 2000 \
    --batch-size 200 \
    --output cas9_rejection_sampling_2000_proposals.fasta \
    --verbose

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
