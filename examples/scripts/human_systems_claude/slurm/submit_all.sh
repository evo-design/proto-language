#!/bin/bash
# =============================================================================
# Submit all complex diversification jobs to SLURM
# =============================================================================
#
# This script should be run from the proto-language root directory.
#
# Usage:
#   ./examples/scripts/human_systems_claude/slurm/submit_all.sh
#   ./examples/scripts/human_systems_claude/slurm/submit_all.sh --dry-run
#   ./examples/scripts/human_systems_claude/slurm/submit_all.sh --single 5
#   ./examples/scripts/human_systems_claude/slurm/submit_all.sh --range 0-10
#
# Prerequisites:
#   1. Run generate_all.py to create configs, programs, and manifest.json
#   2. Ensure conda environment is properly configured in job_template.sbatch
#
# =============================================================================

set -e

# Determine script location and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLEX_PROGRAMS_DIR="$(dirname "$SCRIPT_DIR")"
MANIFEST="$COMPLEX_PROGRAMS_DIR/manifest.json"
JOB_TEMPLATE="$SCRIPT_DIR/job_template.sbatch"

# Try to find the proto-language root (look for common markers)
# Default assumes we're running from the root
PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-$(pwd)}"

# Compute relative path from root to complex programs dir
RELATIVE_PATH="${COMPLEX_PROGRAMS_DIR#$PROTO_LANGUAGE_ROOT/}"

# Parse arguments
DRY_RUN=false
SINGLE_JOB=""
PARTITION=""
JOB_RANGE=""
MAX_CONCURRENT=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --single|-s)
            SINGLE_JOB="$2"
            shift 2
            ;;
        --range|-r)
            JOB_RANGE="$2"
            shift 2
            ;;
        --max-concurrent|-m)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --partition|-p)
            PARTITION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--single N] [--range N-M]"
            echo ""
            echo "Run this script from the proto-language root directory."
            echo ""
            echo "Options:"
            echo "  --dry-run, -n    Show what would be submitted"
            echo "  --single N, -s   Submit only job index N"
            echo "  --range N-M, -r  Submit jobs N through M"
            echo "  --max-concurrent N, -m  Max concurrent array tasks (default: 20)"
            echo "  --partition P, -p  Override partition (e.g., gpu, normal)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest.json not found at $MANIFEST"
    echo "Run 'python $RELATIVE_PATH/scripts/generate_all.py --excel <spreadsheet> --output-dir $RELATIVE_PATH' first"
    exit 1
fi

if [ ! -f "$JOB_TEMPLATE" ]; then
    echo "ERROR: Job template not found at $JOB_TEMPLATE"
    exit 1
fi

# Get total number of programs
TOTAL_PROGRAMS=$(python3 -c "
import json
with open('$MANIFEST') as f:
    manifest = json.load(f)
print(manifest['total_programs'])
")

echo "=============================================="
echo "Complex Diversification Job Submission"
echo "=============================================="
echo "Proto-language root: $PROTO_LANGUAGE_ROOT"
echo "Complex programs dir: $COMPLEX_PROGRAMS_DIR"
echo "Relative path: $RELATIVE_PATH"
echo "Total programs: $TOTAL_PROGRAMS"
echo ""

# Determine job range
if [ -n "$SINGLE_JOB" ]; then
    ARRAY_SPEC="$SINGLE_JOB"
    echo "Mode: Single job (index $SINGLE_JOB)"
elif [ -n "$JOB_RANGE" ]; then
    ARRAY_SPEC="$JOB_RANGE"
    echo "Mode: Job range ($JOB_RANGE)"
else
    ARRAY_SPEC="0-$((TOTAL_PROGRAMS - 1))"
    echo "Mode: All jobs (0-$((TOTAL_PROGRAMS - 1)))"
fi

echo "Array specification: $ARRAY_SPEC"
echo ""

# Create outputs directory
mkdir -p "$COMPLEX_PROGRAMS_DIR/outputs"

# Create SLURM log directory (job_template writes --output to logs/slurm/;
# sbatch does not create parent dirs and the job fails if it is missing)
mkdir -p "$PROTO_LANGUAGE_ROOT/logs/slurm"

# Show what will be submitted
echo "Jobs to submit:"
python3 -c "
import json
with open('$MANIFEST') as f:
    manifest = json.load(f)

# Parse array spec
spec = '$ARRAY_SPEC'
if '-' in spec:
    start, end = map(int, spec.split('-'))
    indices = range(start, end + 1)
else:
    indices = [int(spec)]

for i in indices:
    if i < len(manifest['programs']):
        prog = manifest['programs'][i]
        print(f\"  [{i}] {prog['filename_base']} ({prog['n_genes']} genes, {prog['n_complexes']} complexes)\")
"
echo ""

SBATCH_ARGS="--array=${ARRAY_SPEC}%${MAX_CONCURRENT}"
if [ -n "$PARTITION" ]; then
    SBATCH_ARGS="$SBATCH_ARGS --partition=$PARTITION"
fi

# Submit or dry-run
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - would execute:"
    echo "  cd $PROTO_LANGUAGE_ROOT"
    echo "  export COMPLEX_PROGRAMS_DIR=$RELATIVE_PATH"
    echo "  sbatch $SBATCH_ARGS $JOB_TEMPLATE"
else
    echo "Submitting jobs..."

    # Change to proto-language root for submission
    cd "$PROTO_LANGUAGE_ROOT"

    # Export the complex programs directory path
    export COMPLEX_PROGRAMS_DIR="$RELATIVE_PATH"

    JOB_ID=$(sbatch $SBATCH_ARGS "$JOB_TEMPLATE" | awk '{print $4}')

    echo ""
    echo "=============================================="
    echo "Jobs submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "=============================================="
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f complex_div_${JOB_ID}_*.log"
    echo ""
    echo "Cancel with:"
    echo "  scancel $JOB_ID"
fi
