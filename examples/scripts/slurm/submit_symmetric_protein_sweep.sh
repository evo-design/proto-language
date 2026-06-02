#!/bin/bash
# Submit the symmetric protein parameter sweep as a SLURM array job
#
# Usage:
#   ./submit_sweep.sh           # Submit all 28 jobs
#   ./submit_sweep.sh 0-13      # Submit subset (first 14 jobs)
#   ./submit_sweep.sh 0-27%4    # Submit all with max 4 concurrent jobs

# Parameter grid summary:
#   monomer_length: 50, 100 (2 values)
#   n_symmetric_units: 2, 3, 4, 5, 6, 7, 8 (7 values)
#   n_steps: 5000, 10000 (2 values)
#   Total: 2 * 7 * 2 = 28 combinations

TOTAL_JOBS=3
ARRAY_SPEC="${1:-0-$((TOTAL_JOBS-1))}"

# Override PROTO_LANGUAGE_DIR to point at your checkout; defaults to ~/proto-language.
cd "${PROTO_LANGUAGE_DIR:-$HOME/proto-language}"

# Create log directory
mkdir -p logs/slurm

echo "Submitting symmetric protein parameter sweep"
echo "Array specification: $ARRAY_SPEC"
echo "Total parameter combinations: $TOTAL_JOBS"
echo ""

# Print parameter grid
echo "Parameter grid:"
echo "  Monomer lengths: 50, 100"
echo "  N symmetric units: 2, 3, 4, 5, 6, 7, 8"
echo "  N steps: 5000, 10000"
echo ""

# Submit array job
sbatch --array="$ARRAY_SPEC" examples/scripts/slurm/run_symmetric_protein_sweep.sh

echo ""
echo "Use 'squeue -u \$USER' to monitor jobs"
echo "Logs will be in logs/slurm/"
echo "Results will be in outputs/symmetric_protein_sweep/"
