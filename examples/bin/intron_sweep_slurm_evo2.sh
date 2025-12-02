#!/bin/bash
#SBATCH --job-name=intron_design_evo2
#SBATCH --partition=gpu,gpu_batch
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --array=0-99%10
#SBATCH --output=log/intron_design_evo2_%A_%a.out
#SBATCH --error=log/intron_design_evo2_%A_%a.err

mkdir -p log

JOB_ID=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Starting sampling job $JOB_ID..."

DIR=intron_design_multi-context_evo2
mkdir -p $DIR

python -m examples.scripts.program_intron_design \
    --n_steps 500 \
    --multicontext True \
    --intron_generator evo2 \
    --specificity_type max_brain_min_blood \
    > $DIR/intron_design_temp${log_value}_max-brain-min-blood_${JOB_ID}.log 2>&1

python -m examples.scripts.program_intron_design \
    --n_steps 500 \
    --multicontext True \
    --intron_generator evo2 \
    --specificity_type min_blood \
    > $DIR/intron_design_temp${log_value}_min-blood-only_${JOB_ID}.log 2>&1
