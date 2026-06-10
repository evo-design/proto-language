#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
cd "$REPO_ROOT"

export SPLICE_TRANSFORMER_CHECKPOINT="${SPLICE_TRANSFORMER_CHECKPOINT:-}"
export PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS="${PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS:-true}"
RUN_ROOT_DEFAULT="$REPO_ROOT/scratch/intron_alphagenome_work/examples_outputs/intron_alphagenome_runs"
DESIGN_TARGET_ONTOLOGY_TERMS_DEFAULT="CL:0002319,CL:0011012,CL:0011020,CL:0000047"
VIS_TARGET_ONTOLOGY_TERMS_DEFAULT="CL:0002319"
OFFTARGET_ONTOLOGY_TERMS_DEFAULT="EFO:0002067"

resolve_alphagenome_checkpoint() {
    if [[ -n "${ALPHAGENOME_CHECKPOINT_PATH:-}" ]]; then
        return 0
    fi
    local cache_root="$HOME/.cache/huggingface/hub/models--google--alphagenome-all-folds"
    if [[ -f "$cache_root/refs/main" ]]; then
        local snapshot_ref
        snapshot_ref="$(cat "$cache_root/refs/main")"
        export ALPHAGENOME_CHECKPOINT_PATH="$cache_root/snapshots/$snapshot_ref"
    fi
    if [[ -z "${ALPHAGENOME_CHECKPOINT_PATH:-}" || ! -d "${ALPHAGENOME_CHECKPOINT_PATH:-}" ]]; then
        echo "ERROR: Could not resolve ALPHAGENOME_CHECKPOINT_PATH." >&2
        exit 1
    fi
}

resolve_splice_transformer_checkpoint() {
    if [[ -n "${SPLICE_TRANSFORMER_CHECKPOINT:-}" ]]; then
        if [[ -f "$SPLICE_TRANSFORMER_CHECKPOINT" ]]; then
            return 0
        fi
        echo "WARNING: SPLICE_TRANSFORMER_CHECKPOINT is set but file does not exist: $SPLICE_TRANSFORMER_CHECKPOINT" >&2
        echo "WARNING: Falling back to HuggingFace cache/runtime download for SpliceTransformer checkpoint." >&2
        unset SPLICE_TRANSFORMER_CHECKPOINT
    fi

    local cache_root="$HOME/.cache/huggingface/hub/models--brianhie--SpTransformer"
    if [[ -f "$cache_root/refs/main" ]]; then
        local snapshot_ref
        snapshot_ref="$(cat "$cache_root/refs/main")"
        local candidate="$cache_root/snapshots/$snapshot_ref/SpTransformer_pytorch.ckpt"
        if [[ -f "$candidate" ]]; then
            export SPLICE_TRANSFORMER_CHECKPOINT="$candidate"
            return 0
        fi
    fi

    echo "INFO: SpliceTransformer checkpoint not found in local cache; runtime will download it from HuggingFace." >&2
}

usage() {
    cat <<'EOF'
Usage:
  examples/bin/run_intron_alphagenome_gpu.sh [mode] [args...]

Modes:
  passthrough (default)
    Run program_intron_alphagenome with provided args.
    Example:
      examples/bin/run_intron_alphagenome_gpu.sh --n_steps 10 --multicontext true

  mcmc5000_cl0002319
    Run the full local 5000-step multi-target neural profile
    (CL:0002319, CL:0011012, CL:0011020, CL:0000047) vs EFO:0002067, then
    auto-generate AlphaGenome SVG tracks for initial/final sequences plus HBB controls.
    Visualization is intentionally focused on CL:0002319.

Environment:
  PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS
                            Use shared in-process ST+AG GPU worker (default: true).
  INTRON_AG_RUN_ROOT        Optional output root for mcmc5000_cl0002319 mode.
  INTRON_AG_TARGET_ONTOLOGY_TERMS
                            Optional override for design target ontology list.
  INTRON_AG_VIS_TARGET_ONTOLOGY_TERMS
                            Optional override for visualization target terms.
  INTRON_AG_OFFTARGET_ONTOLOGY_TERMS
                            Optional override for off-target terms.
  SPLICE_TRANSFORMER_CHECKPOINT
  ALPHAGENOME_CHECKPOINT_PATH
EOF
}

run_passthrough() {
    resolve_alphagenome_checkpoint
    resolve_splice_transformer_checkpoint
    env \
        PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS="$PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS" \
        python -u -m examples.scripts.program_intron_alphagenome "$@"
}

run_mcmc5000_cl0002319() {
    resolve_alphagenome_checkpoint
    resolve_splice_transformer_checkpoint

    local run_root="${INTRON_AG_RUN_ROOT:-$RUN_ROOT_DEFAULT}"
    local target_terms="${INTRON_AG_TARGET_ONTOLOGY_TERMS:-$DESIGN_TARGET_ONTOLOGY_TERMS_DEFAULT}"
    local vis_target_terms="${INTRON_AG_VIS_TARGET_ONTOLOGY_TERMS:-$VIS_TARGET_ONTOLOGY_TERMS_DEFAULT}"
    local offtarget_terms="${INTRON_AG_OFFTARGET_ONTOLOGY_TERMS:-$OFFTARGET_ONTOLOGY_TERMS_DEFAULT}"
    local ts run_id run_dir run_log post_log status_path cmd_path plot_fasta plot_dir
    ts="$(date +%Y%m%d_%H%M%S)"
    run_id="mcmc5000_cl0002319_${ts}"
    run_dir="$run_root/$run_id"
    run_log="$run_dir/mcmc_run.log"
    post_log="$run_dir/postprocess.log"
    status_path="$run_dir/status.txt"
    cmd_path="$run_dir/command.txt"
    plot_fasta="$run_dir/plot_designs.fasta"
    plot_dir="$run_dir/plots_all_contexts"

    mkdir -p "$run_dir"

    {
        echo "run_id=$run_id"
        echo "start_time=$(date --iso-8601=seconds)"
        echo "repo_root=$REPO_ROOT"
        echo "run_log=$run_log"
        echo "post_log=$post_log"
        echo "status=$status_path"
        echo "plot_dir=$plot_dir"
        echo "target_ontology_terms=$target_terms"
        echo "visualization_target_ontology_terms=$vis_target_terms"
        echo "offtarget_ontology_terms=$offtarget_terms"
        echo "alphagenome_checkpoint_path=$ALPHAGENOME_CHECKPOINT_PATH"
        echo "splice_transformer_checkpoint=$SPLICE_TRANSFORMER_CHECKPOINT"
        echo "proto_language_rna_splicing_shared_process=$PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS"
    } > "$status_path"

    local cmd=(
        python -u -m examples.scripts.program_intron_alphagenome
        --n_steps 5000
        --multicontext true
        --intron_generator uniform
        --initialization hbb2c
        --enable_alphagenome true
        --enable_splice_transformer true
        --enable_splice_specificity true
        --specificity_type max_brain_min_blood
        --temperature 0.572237
        --temperature_min 0.001
        --alphagenome_device cuda
        --splice_transformer_device cuda
        --alphagenome_track_strand positive
        --alphagenome_brain_weight 1.0
        --alphagenome_blood_weight 1.0
        --target_ontology_terms "$target_terms"
        --offtarget_ontology_terms "$offtarget_terms"
    )

    if [[ $# -gt 0 ]]; then
        cmd+=("$@")
    fi

    {
        printf 'SPLICE_TRANSFORMER_CHECKPOINT=%q\n' "$SPLICE_TRANSFORMER_CHECKPOINT"
        printf 'ALPHAGENOME_CHECKPOINT_PATH=%q\n' "$ALPHAGENOME_CHECKPOINT_PATH"
        printf 'PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS=%q\n' "$PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS"
        printf 'COMMAND='
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$cmd_path"

    echo "[run] $run_id"
    echo "[run] log: $run_log"
    echo "[run] status: $status_path"
    echo "[run] command: $cmd_path"

    set +e
    /usr/bin/time -f "WALL_SECONDS=%e" env \
        SPLICE_TRANSFORMER_CHECKPOINT="$SPLICE_TRANSFORMER_CHECKPOINT" \
        ALPHAGENOME_CHECKPOINT_PATH="$ALPHAGENOME_CHECKPOINT_PATH" \
        PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS="$PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS" \
        "${cmd[@]}" 2>&1 | tee "$run_log"
    local run_rc=${PIPESTATUS[0]}
    set -e

    {
        echo "run_exit_code=$run_rc"
        echo "run_end_time=$(date --iso-8601=seconds)"
    } >> "$status_path"

    if [[ "$run_rc" -ne 0 ]]; then
        echo "[error] MCMC run failed with exit code $run_rc"
        echo "postprocess_skipped=true" >> "$status_path"
        return "$run_rc"
    fi

    python - "$run_log" "$plot_fasta" <<'PY'
import re
import sys
from types import SimpleNamespace

from examples.scripts.program_intron_design import get_initial_intron

log_path = sys.argv[1]
out_fasta = sys.argv[2]
text = open(log_path).read()

pattern = re.compile(
    r"sequence \(left_flank\):\s*([ACGTNacgtn]+)\s*"
    r"\n\s*sequence \(intron\):\s*([ACGTNacgtn]+)\s*"
    r"\n\s*sequence \(right_flank\):\s*([ACGTNacgtn]+)"
)
triples = [(a.upper(), b.upper(), c.upper()) for a, b, c in pattern.findall(text)]
if not triples:
    raise SystemExit("No (left_flank, intron, right_flank) triples found in run log.")

def reconstruct_intron(left: str, intron_core: str, right: str) -> str:
    if len(left) < 2 or len(right) < 2:
        raise ValueError("Cannot reconstruct full intron: flank too short.")
    return left[-2:] + intron_core + right[:2]

initial = reconstruct_intron(*triples[0])
final = reconstruct_intron(*triples[-1])

controls = {
    "hbb1": get_initial_intron(SimpleNamespace(initialization="hbb1", intron_length=301)),
    "hbb2": get_initial_intron(SimpleNamespace(initialization="hbb2", intron_length=301)),
    "hbb2c": get_initial_intron(SimpleNamespace(initialization="hbb2c", intron_length=301)),
}

items = [
    ("run_initial", initial),
    ("run_final", final),
    ("hbb1_control", controls["hbb1"]),
    ("hbb2_control", controls["hbb2"]),
    ("hbb2c_control", controls["hbb2c"]),
]

seen = set()
with open(out_fasta, "w") as handle:
    for name, seq in items:
        seq = seq.upper()
        if seq in seen:
            continue
        seen.add(seq)
        handle.write(f">{name}\n{seq}\n")
PY

    set +e
    env \
        ALPHAGENOME_CHECKPOINT_PATH="$ALPHAGENOME_CHECKPOINT_PATH" \
        PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS="$PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS" \
        python -u examples/bin/visualize_intron_alphagenome_tracks.py \
            --design_sequences_path "$plot_fasta" \
            --max_designs 0 \
            --target_cell "$vis_target_terms" \
            --offtarget_cell "$offtarget_terms" \
            --target_ontology_terms "$vis_target_terms" \
            --offtarget_ontology_terms "$offtarget_terms" \
            --alphagenome_device cuda \
            --output_dir "$plot_dir" \
            --filename_prefix "${run_id}_" 2>&1 | tee "$post_log"
    local plot_rc=${PIPESTATUS[0]}
    set -e

    {
        echo "plot_exit_code=$plot_rc"
        echo "plot_end_time=$(date --iso-8601=seconds)"
    } >> "$status_path"

    if [[ "$plot_rc" -ne 0 ]]; then
        echo "[error] Post-run visualization failed with exit code $plot_rc"
        return "$plot_rc"
    fi

    echo "[ok] completed run + plotting"
    echo "[ok] run directory: $run_dir"
}

mode="${1:-passthrough}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$mode" in
    passthrough)
        run_passthrough "$@"
        ;;
    mcmc5000_cl0002319)
        run_mcmc5000_cl0002319 "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        # Backward-compatible: first token is likely a program arg, not a mode.
        run_passthrough "$mode" "$@"
        ;;
esac
