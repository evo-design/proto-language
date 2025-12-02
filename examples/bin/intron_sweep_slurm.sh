#!/bin/bash
#SBATCH --job-name=intron_design
#SBATCH --partition=gpu_20gb,gpu_40gb,gpu,gpu_batch
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --array=0-99%10
#SBATCH --output=log/intron_design_%A_%a.out
#SBATCH --error=log/intron_design_%A_%a.err

mkdir -p log

JOB_ID=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Starting sampling job $JOB_ID..."

INIT_LIST=("random" "hbb2c")

for INIT in "${INIT_LIST[@]}"
do
    DIR=intron_design_multi-context_$INIT
    mkdir -p $DIR
    echo "Processing INIT=$INIT, writing to directory $DIR"

    for i in $(seq 1 8 100); do
        log_value=$(awk -v i="$i" 'BEGIN { exponent = (1 - i) / 33; print 10 ^ exponent }')
        
        echo "Running i=$i, temperature=$log_value..."

        python -m examples.scripts.program_intron_design \
            --temperature $log_value \
            --n_steps 5000 \
            --multicontext True \
            --initialization $INIT \
            --specificity_type max_brain_min_blood \
            > $DIR/intron_design_temp${log_value}_max-brain-min-blood_${JOB_ID}.log 2>&1

        python -m examples.scripts.program_intron_design \
            --temperature $log_value \
            --n_steps 5000 \
            --multicontext True \
            --initialization $INIT \
            --specificity_type min_blood \
            > $DIR/intron_design_temp${log_value}_min-blood-only_${JOB_ID}.log 2>&1
    done
done
