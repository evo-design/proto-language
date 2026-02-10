#!/bin/bash
#SBATCH --job-name=proteinhunter_sweep
#SBATCH --partition=gpu,gpu_high_mem
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --output=logs/slurm/proteinhunter_sweep_%A_%a.out
#SBATCH --error=logs/slurm/proteinhunter_sweep_%A_%a.err
#SBATCH --requeue

# Define Arrays.
TOOLS=("boltz2" "chai1")
LENGTHS=(200 300 400 500 600 700)

# Map Array ID to Parameters
# Logic: We have 3 lengths per tool.
# TOOL_IDX = ID / 3  (Integer division)
# LEN_IDX  = ID % 3  (Modulus)

TOOL_IDX=$((SLURM_ARRAY_TASK_ID / 3))
LEN_IDX=$((SLURM_ARRAY_TASK_ID % 3))

TOOL=${TOOLS[$TOOL_IDX]}
LENGTH=${LENGTHS[$LEN_IDX]}

echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID} | Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration: Tool=${TOOL} | Length=${LENGTH}"
echo "=========================================================="

# Run Protein Hunter.
# Using --cycles 5 and --candidates 1 to keep it lightweight.
python examples/scripts/protein_hunter.py \
    --structure-tool $TOOL \
    --length $LENGTH \
    --cycles 5 \
    --candidates 1 \
    --output-dir ./outputs
