#!/bin/bash
#SBATCH --job-name=symmetric_protein_sweep
#SBATCH --output=logs/slurm/sym_protein_%x_%A_%a.log
#SBATCH --error=logs/slurm/sym_protein_%x_%A_%a.log
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu_batch,gpu_batch_high_mem
#SBATCH --requeue

# Parameter sweep for symmetric protein design
# Grid:
#   monomer_length: 50, 100
#   n_symmetric_units: 2, 3, 4, 5, 6, 7, 8
#   n_steps: 5000, 10000
# Total combinations: 2 * 7 * 2 = 28

# Define parameter arrays
MONOMER_LENGTHS=(50) # 100)
N_SYMMETRIC_UNITS=(6 7 8) #(2 3 4 5 6 7 8)
N_STEPS=(30000) #(5000 10000)

# Calculate total number of combinations
N_LENGTHS=${#MONOMER_LENGTHS[@]}
N_UNITS=${#N_SYMMETRIC_UNITS[@]}
N_STEP_OPTIONS=${#N_STEPS[@]}
TOTAL_COMBOS=$((N_LENGTHS * N_UNITS * N_STEP_OPTIONS))

# Get parameters for this array task
# SLURM_ARRAY_TASK_ID ranges from 0 to TOTAL_COMBOS-1
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Decode task ID to parameter indices
# Order: iterate steps fastest, then units, then lengths
STEP_IDX=$((TASK_ID % N_STEP_OPTIONS))
UNIT_IDX=$(((TASK_ID / N_STEP_OPTIONS) % N_UNITS))
LENGTH_IDX=$((TASK_ID / (N_STEP_OPTIONS * N_UNITS)))

# Get actual parameter values
MONOMER_LENGTH=${MONOMER_LENGTHS[$LENGTH_IDX]}
N_SYM_UNITS=${N_SYMMETRIC_UNITS[$UNIT_IDX]}
STEPS=${N_STEPS[$STEP_IDX]}

# Output directory
OUTPUT_DIR="outputs/symmetric_protein_sweep"

# Override PROTO_LANGUAGE_DIR to point at your checkout; defaults to ~/proto-language.
cd "${PROTO_LANGUAGE_DIR:-$HOME/proto-language}"

# Create necessary directories
mkdir -p logs/slurm
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "=========================================="
echo "Symmetric Protein Design - Parameter Sweep"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo ""
echo "Parameters for this job:"
echo "  Monomer length: $MONOMER_LENGTH"
echo "  N symmetric units: $N_SYM_UNITS"
echo "  N steps: $STEPS"
echo "  Output directory: $OUTPUT_DIR"
echo "=========================================="

# Run the optimization
python examples/scripts/program_symmetric_proteins.py \
    --monomer-length "$MONOMER_LENGTH" \
    --n-symmetric-units "$N_SYM_UNITS" \
    --n-steps "$STEPS" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
