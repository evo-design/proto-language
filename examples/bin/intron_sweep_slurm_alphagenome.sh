#!/bin/bash
#SBATCH --job-name=intron_alphagenome
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --partition=preemptible
#SBATCH --requeue
#SBATCH --output=log/slurm/%x_%A_%a.out
#SBATCH --error=log/slurm/%x_%A_%a.err

set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
if [[ -n "${PROTO_LANGUAGE_REPO_ROOT:-}" ]]; then
    REPO_ROOT="$PROTO_LANGUAGE_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
fi
cd "$REPO_ROOT"
SCRATCH_ROOT_CANDIDATE_1="/scratch/users/${USER}/${USER}/proto-language"
SCRATCH_ROOT_CANDIDATE_2="/scratch/users/${USER}/proto-language"
if [[ -d "$(dirname "$SCRATCH_ROOT_CANDIDATE_1")" ]]; then
    DEFAULT_WORK_ROOT="$SCRATCH_ROOT_CANDIDATE_1"
elif [[ -d "$(dirname "$SCRATCH_ROOT_CANDIDATE_2")" ]]; then
    DEFAULT_WORK_ROOT="$SCRATCH_ROOT_CANDIDATE_2"
else
    DEFAULT_WORK_ROOT="$REPO_ROOT"
fi
WORK_ROOT="${PROTO_LANGUAGE_WORK_ROOT:-$DEFAULT_WORK_ROOT}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$WORK_ROOT/log/slurm}"
SWEEP_LOG_ROOT="${SWEEP_LOG_ROOT:-$WORK_ROOT/log/intron_alphagenome}"
SWEEP_RUN_ROOT="${SWEEP_RUN_ROOT:-$WORK_ROOT/runs/intron_alphagenome}"
mkdir -p "$SLURM_LOG_DIR" "$SWEEP_LOG_ROOT" "$SWEEP_RUN_ROOT"

DEFAULT_GRID_PATH="${SWEEP_GRID_PATH:-examples/bin/intron_sweep_alphagenome_grid.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
SLURM_ARRAY_SPEC="${SLURM_ARRAY_SPEC:-}"
SLURM_PARTITION="${SLURM_PARTITION:-preemptible}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-GPUCACE}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_TIME="${SLURM_TIME:-}"
SWEEP_STEPS="${SWEEP_STEPS:-5000}"
SWEEP_INTRON_LENGTH="${SWEEP_INTRON_LENGTH:-301}"
SWEEP_REPLICATES="${SWEEP_REPLICATES:-1}"
SWEEP_INITIALIZATION_VALUES="${SWEEP_INITIALIZATION_VALUES:-random,hbb2c}"
TEMPERATURE_VALUES="${TEMPERATURE_VALUES:-0.01,0.1,0.5,1}"
ST_SPECIFICITY_VALUES="${ST_SPECIFICITY_VALUES:-True,False}"
SPECIFICITY_TYPE_ON_VALUES="${SPECIFICITY_TYPE_ON_VALUES:-max_brain_min_blood}"
SPECIFICITY_TYPE_OFF_VALUES="${SPECIFICITY_TYPE_OFF_VALUES:-}"
ALPHAGENOME_TRACK_STRAND_VALUES="${ALPHAGENOME_TRACK_STRAND_VALUES:-positive}"
ALPHAGENOME_BRAIN_WEIGHT_VALUES="${ALPHAGENOME_BRAIN_WEIGHT_VALUES:-1.0,5.0,20.0}"
ALPHAGENOME_BLOOD_WEIGHT_VALUES="${ALPHAGENOME_BLOOD_WEIGHT_VALUES:-1.0}"
ALPHAGENOME_PRED_TIMEOUT="${ALPHAGENOME_PRED_TIMEOUT:-3600}"
ALPHAGENOME_DEVICE="${ALPHAGENOME_DEVICE:-cuda}"
SPLICE_TRANSFORMER_DEVICE="${SPLICE_TRANSFORMER_DEVICE:-cuda}"
SHARED_RNA_SPLICING_PROCESS="${SHARED_RNA_SPLICING_PROCESS:-true}"
GENERATE_AGGREGATE_AG_VIS="${GENERATE_AGGREGATE_AG_VIS:-true}"
GENERATE_AGGREGATE_AG_ST_VIS="${GENERATE_AGGREGATE_AG_ST_VIS:-true}"
AG_VIS_VIEW_MODE="${AG_VIS_VIEW_MODE:-full}"
AG_ST_VIS_VIEW_MODE="${AG_ST_VIS_VIEW_MODE:-full}"
MCMC_CANDIDATES_PER_RESULT="${MCMC_CANDIDATES_PER_RESULT:-1}"
DEFAULT_TARGET_ONTOLOGY_TERMS="CL:0002319,CL:0011012,CL:0011020,CL:0000047"
DEFAULT_OFFTARGET_ONTOLOGY_TERMS="EFO:0002067"
SWEEP_TARGET_CELL="${SWEEP_TARGET_CELL:-CL:0002319}"
SWEEP_TARGET_ONTOLOGY_TERMS="${SWEEP_TARGET_ONTOLOGY_TERMS:-$DEFAULT_TARGET_ONTOLOGY_TERMS}"
SWEEP_OFFTARGET_CELL="${SWEEP_OFFTARGET_CELL:-k562}"
SWEEP_OFFTARGET_ONTOLOGY_TERMS="${SWEEP_OFFTARGET_ONTOLOGY_TERMS:-$DEFAULT_OFFTARGET_ONTOLOGY_TERMS}"
DEFAULT_VIS_PLASMID_CONTEXT_PATHS="examples/data/plasmid_context_cmv_20260308.txt,examples/data/plasmid_context_Ef1a.txt,examples/data/plasmid_context_sffv.txt"
DEFAULT_VIS_GENOMIC_CONTEXT_PATHS="examples/data/alphagenome_context_aavs1.txt,examples/data/alphagenome_context_ccr5.txt,examples/data/alphagenome_context_clybl.txt,examples/data/alphagenome_context_hrosa26.txt"
VIS_PRIMARY_PLASMID_CONTEXT_PATH="${VIS_PRIMARY_PLASMID_CONTEXT_PATH:-examples/data/plasmid_context_cmv_20260308.txt}"
VIS_PLASMID_CONTEXT_PATHS="${VIS_PLASMID_CONTEXT_PATHS:-$DEFAULT_VIS_PLASMID_CONTEXT_PATHS}"
VIS_GENOMIC_CONTEXT_PATHS="${VIS_GENOMIC_CONTEXT_PATHS:-$DEFAULT_VIS_GENOMIC_CONTEXT_PATHS}"
VIS_GENE_SEQUENCE_PATH="${VIS_GENE_SEQUENCE_PATH:-examples/data/mscarlet_ires_zsgreen.txt}"
VIS_GENE_INSERTION_POS="${VIS_GENE_INSERTION_POS:-477}"
ALPHAGENOME_CKPT_DEFAULT="$HOME/.cache/huggingface/hub/models--google--alphagenome-all-folds/snapshots/$(cat "$HOME/.cache/huggingface/hub/models--google--alphagenome-all-folds/refs/main" 2>/dev/null || true)"
ALPHAGENOME_CHECKPOINT_PATH="${ALPHAGENOME_CHECKPOINT_PATH:-$ALPHAGENOME_CKPT_DEFAULT}"
CUDA_LIB_DIRS=(
    "/usr/local/cuda/targets/x86_64-linux/lib"
    "/usr/local/cuda/lib64"
)
CUDA_LD_LIBRARY_PATH=""
for cuda_lib_dir in "${CUDA_LIB_DIRS[@]}"; do
    if [[ -d "$cuda_lib_dir" ]]; then
        if [[ -n "$CUDA_LD_LIBRARY_PATH" ]]; then
            CUDA_LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}:$cuda_lib_dir"
        else
            CUDA_LD_LIBRARY_PATH="$cuda_lib_dir"
        fi
    fi
done

infer_target_tissue() {
    local specificity_type_lc="${1,,}"
    local target_cell_lc="${2,,}"
    if [[ "$specificity_type_lc" == *brain* ]]; then
        echo "BRAIN"
        return
    fi
    if [[ "$specificity_type_lc" == *liver* ]]; then
        echo "LIVER"
        return
    fi
    if [[ "$specificity_type_lc" == *blood* ]]; then
        echo "BLOOD"
        return
    fi
    case "$target_cell_lc" in
        *0001187*|hepg2)
            echo "LIVER"
            ;;
        *0002067*|k562)
            echo "BLOOD"
            ;;
        *0002319*|shsy5y|sh-sy5y)
            echo "BRAIN"
            ;;
        *)
            echo "BRAIN"
            ;;
    esac
}

infer_offtarget_tissue() {
    local specificity_type_lc="${1,,}"
    local offtarget_cell_lc="${2,,}"
    if [[ "$specificity_type_lc" == *blood* ]]; then
        echo "BLOOD"
        return
    fi
    if [[ "$specificity_type_lc" == *brain* ]]; then
        echo "BRAIN"
        return
    fi
    if [[ "$specificity_type_lc" == *liver* ]]; then
        echo "LIVER"
        return
    fi
    case "$offtarget_cell_lc" in
        *0001187*|hepg2)
            echo "LIVER"
            ;;
        *0002067*|k562)
            echo "BLOOD"
            ;;
        *0002319*|shsy5y|sh-sy5y)
            echo "BRAIN"
            ;;
        *)
            echo "BLOOD"
            ;;
    esac
}

prepare_grid() {
    local grid_path="$1"
    mkdir -p "$(dirname "$grid_path")"

    {
        echo -e "config_id\tinitialization\tintron_generator\ttemperature\tn_steps\tmulticontext\tspecificity_type\tenable_splice_transformer\tenable_splice_specificity\tenable_alphagenome\ttarget_cell\ttarget_ontology_terms\tofftarget_cell\tofftarget_ontology_terms\talphagenome_track_strand\talphagenome_brain_weight\talphagenome_blood_weight"

        local config_id=0
        local init_list
        local init
        local enable_splice_specificity
        local spec
        local spec_values
        local temperature
        local alphagenome_track_strand
        local alphagenome_brain_weight
        local alphagenome_blood_weight
        local temperature_values
        local st_specificity_values
        local specificity_type_on_values
        local specificity_type_off_values
        local sweep_initialization_values
        local alphagenome_track_strand_values
        local alphagenome_brain_weight_values
        local alphagenome_blood_weight_values
        local replicate_idx
        IFS=',' read -r -a sweep_initialization_values <<< "$SWEEP_INITIALIZATION_VALUES"
        IFS=',' read -r -a temperature_values <<< "$TEMPERATURE_VALUES"
        IFS=',' read -r -a st_specificity_values <<< "$ST_SPECIFICITY_VALUES"
        IFS=',' read -r -a specificity_type_on_values <<< "$SPECIFICITY_TYPE_ON_VALUES"
        IFS=',' read -r -a specificity_type_off_values <<< "$SPECIFICITY_TYPE_OFF_VALUES"
        IFS=',' read -r -a alphagenome_track_strand_values <<< "$ALPHAGENOME_TRACK_STRAND_VALUES"
        IFS=',' read -r -a alphagenome_brain_weight_values <<< "$ALPHAGENOME_BRAIN_WEIGHT_VALUES"
        IFS=',' read -r -a alphagenome_blood_weight_values <<< "$ALPHAGENOME_BLOOD_WEIGHT_VALUES"

        init_list=("${sweep_initialization_values[@]}")
        for init in "${init_list[@]}"; do
            for enable_splice_specificity in "${st_specificity_values[@]}"; do
                if [[ "$enable_splice_specificity" == "True" ]]; then
                    spec_values=("${specificity_type_on_values[@]}")
                else
                    if [[ "${#specificity_type_off_values[@]}" -gt 0 && -n "${specificity_type_off_values[0]}" ]]; then
                        spec_values=("${specificity_type_off_values[@]}")
                    else
                        spec_values=("${specificity_type_on_values[@]}")
                    fi
                fi

                for spec in "${spec_values[@]}"; do
                    for temperature in "${temperature_values[@]}"; do
                        for alphagenome_track_strand in "${alphagenome_track_strand_values[@]}"; do
                            for alphagenome_brain_weight in "${alphagenome_brain_weight_values[@]}"; do
                                for alphagenome_blood_weight in "${alphagenome_blood_weight_values[@]}"; do
                                    for replicate_idx in $(seq 1 "$SWEEP_REPLICATES"); do
                                        echo -e "${config_id}\t${init}\tuniform\t${temperature}\t${SWEEP_STEPS}\tTrue\t${spec}\tTrue\t${enable_splice_specificity}\tTrue\t${SWEEP_TARGET_CELL}\t${SWEEP_TARGET_ONTOLOGY_TERMS}\t${SWEEP_OFFTARGET_CELL}\t${SWEEP_OFFTARGET_ONTOLOGY_TERMS}\t${alphagenome_track_strand}\t${alphagenome_brain_weight}\t${alphagenome_blood_weight}"
                                        config_id=$((config_id + 1))
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    } > "$grid_path"
}

submit_sweep() {
    local grid_path="$1"
    local sweep_id="$2"
    echo "SWEEP_ID=${sweep_id}"

    local total_configs
    total_configs=$(($(wc -l < "$grid_path") - 1))
    if [[ "$total_configs" -le 0 ]]; then
        echo "No configs found in grid: $grid_path" >&2
        exit 1
    fi

    local exclude_nodes="${SLURM_EXCLUDE_NODES}"
    local grid_path_abs
    grid_path_abs="$(readlink -f "$grid_path")"
    local default_array_spec="0-$((total_configs - 1))"
    if [[ -n "${MAX_PARALLEL}" ]]; then
        default_array_spec="${default_array_spec}%${MAX_PARALLEL}"
    fi
    local array_spec="${SLURM_ARRAY_SPEC:-$default_array_spec}"
    local -a sbatch_args=(
        --partition "$SLURM_PARTITION"
        --output "${SLURM_LOG_DIR}/%x_%A_%a.out"
        --error "${SLURM_LOG_DIR}/%x_%A_%a.err"
    )
    if [[ -n "$SLURM_TIME" ]]; then
        sbatch_args+=(--time "$SLURM_TIME")
    fi
    if [[ -n "$SLURM_QOS" ]]; then
        sbatch_args+=(--qos "$SLURM_QOS")
    fi
    if [[ -n "$SLURM_CONSTRAINT" ]]; then
        sbatch_args+=(--constraint "$SLURM_CONSTRAINT")
    fi
    if [[ -n "$exclude_nodes" ]]; then
        sbatch_args+=(--exclude "$exclude_nodes")
    fi

    echo "Submitting ${total_configs} configs with array spec ${array_spec}"
    echo "SLURM placement: partition=${SLURM_PARTITION} qos=${SLURM_QOS:-<default>} constraint=${SLURM_CONSTRAINT:-<none>} exclude=${exclude_nodes:-<none>}"
    echo "Output roots: work=${WORK_ROOT} slurm_log=${SLURM_LOG_DIR} sweep_log=${SWEEP_LOG_ROOT} sweep_run=${SWEEP_RUN_ROOT}"
    sbatch \
        "${sbatch_args[@]}" \
        --array="$array_spec" \
        --export=ALL,GRID_FILE="$grid_path_abs",SWEEP_ID="$sweep_id",PROTO_LANGUAGE_REPO_ROOT="$REPO_ROOT",PROTO_LANGUAGE_WORK_ROOT="$WORK_ROOT",SLURM_LOG_DIR="$SLURM_LOG_DIR",SWEEP_LOG_ROOT="$SWEEP_LOG_ROOT",SWEEP_RUN_ROOT="$SWEEP_RUN_ROOT" \
        "$SCRIPT_PATH" run
}

run_task() {
    local grid_file="${GRID_FILE:-$DEFAULT_GRID_PATH}"
    local sweep_id="${SWEEP_ID:-manual_$(date +%Y%m%d_%H%M%S)}"

    if [[ ! -f "$grid_file" ]]; then
        echo "Grid file not found: $grid_file" >&2
        exit 1
    fi
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "SLURM_ARRAY_TASK_ID is required in run mode." >&2
        exit 1
    fi

    local row_num=$((SLURM_ARRAY_TASK_ID + 2))
    local row
    row=$(sed -n "${row_num}p" "$grid_file")
    if [[ -z "$row" ]]; then
        echo "No config for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in $grid_file" >&2
        exit 1
    fi

    IFS=$'\t' read -r \
        config_id initialization intron_generator temperature n_steps multicontext specificity_type \
        enable_splice_transformer enable_splice_specificity enable_alphagenome \
        target_cell target_ontology_terms offtarget_cell offtarget_ontology_terms \
        alphagenome_track_strand alphagenome_brain_weight alphagenome_blood_weight \
        <<< "$row"

    local task_log_dir="${SWEEP_LOG_ROOT}/${sweep_id}/task_${SLURM_ARRAY_TASK_ID}"
    local run_dir="${SWEEP_RUN_ROOT}/${sweep_id}/config_${config_id}"
    local vis_out_dir="${run_dir}/alphagenome_final_aggregate"
    local ag_st_vis_out_dir="${run_dir}/ag_st_final_aggregate"
    local stdout_log="${task_log_dir}/stdout.log"
    local stderr_log="${task_log_dir}/stderr.log"
    local vis_log="${task_log_dir}/visualize_final_aggregate.log"
    local ag_st_vis_log="${task_log_dir}/visualize_ag_st_final_aggregate.log"
    local restart_count="${SLURM_RESTART_COUNT:-0}"
    local resolved_vis_plasmid_context_paths="$VIS_PLASMID_CONTEXT_PATHS"
    local resolved_vis_genomic_context_paths="$VIS_GENOMIC_CONTEXT_PATHS"
    local ag_st_target_tissue
    local ag_st_offtarget_tissue
    mkdir -p "$task_log_dir" "$run_dir"

    if [[ "$multicontext" != "True" ]]; then
        resolved_vis_plasmid_context_paths="$VIS_PRIMARY_PLASMID_CONTEXT_PATH"
        resolved_vis_genomic_context_paths="${VIS_GENOMIC_CONTEXT_PATHS%%,*}"
    fi
    ag_st_target_tissue="$(infer_target_tissue "$specificity_type" "$target_cell")"
    ag_st_offtarget_tissue="$(infer_offtarget_tissue "$specificity_type" "$offtarget_cell")"

    local cmd=(
        python -m examples.scripts.program_intron_alphagenome
        --intron_length "$SWEEP_INTRON_LENGTH"
        --n_steps "$n_steps"
        --temperature "$temperature"
        --multicontext "$multicontext"
        --intron_generator "$intron_generator"
        --initialization "$initialization"
        --specificity_type "$specificity_type"
        --enable_splice_transformer "$enable_splice_transformer"
        --enable_splice_specificity "$enable_splice_specificity"
        --enable_alphagenome "$enable_alphagenome"
        --target_cell "$target_cell"
        --target_ontology_terms "$target_ontology_terms"
        --offtarget_cell "$offtarget_cell"
        --offtarget_ontology_terms "$offtarget_ontology_terms"
        --alphagenome_track_strand "$alphagenome_track_strand"
        --alphagenome_brain_weight "$alphagenome_brain_weight"
        --alphagenome_blood_weight "$alphagenome_blood_weight"
        --alphagenome_prediction_timeout "$ALPHAGENOME_PRED_TIMEOUT"
        --alphagenome_device "$ALPHAGENOME_DEVICE"
        --splice_transformer_device "$SPLICE_TRANSFORMER_DEVICE"
        --mcmc_candidates_per_result "$MCMC_CANDIDATES_PER_RESULT"
    )

    {
        echo "SWEEP_ID=$sweep_id"
        echo "GRID_FILE=$grid_file"
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
        echo "CONFIG_ID=$config_id"
        echo "INITIALIZATION=$initialization"
        echo "INTRON_GENERATOR=$intron_generator"
        echo "INTRON_LENGTH=$SWEEP_INTRON_LENGTH"
        echo "TEMPERATURE=$temperature"
        echo "N_STEPS=$n_steps"
        echo "MULTICONTEXT=$multicontext"
        echo "SPECIFICITY_TYPE=$specificity_type"
        echo "ENABLE_SPLICE_TRANSFORMER=$enable_splice_transformer"
        echo "ENABLE_SPLICE_SPECIFICITY=$enable_splice_specificity"
        echo "ENABLE_ALPHAGENOME=$enable_alphagenome"
        echo "TARGET_CELL=$target_cell"
        echo "TARGET_ONTOLOGY_TERMS=$target_ontology_terms"
        echo "OFFTARGET_CELL=$offtarget_cell"
        echo "OFFTARGET_ONTOLOGY_TERMS=$offtarget_ontology_terms"
        echo "ALPHAGENOME_TRACK_STRAND=$alphagenome_track_strand"
        echo "ALPHAGENOME_BRAIN_WEIGHT=$alphagenome_brain_weight"
        echo "ALPHAGENOME_BLOOD_WEIGHT=$alphagenome_blood_weight"
        echo "ALPHAGENOME_PRED_TIMEOUT=$ALPHAGENOME_PRED_TIMEOUT"
        echo "ALPHAGENOME_DEVICE=$ALPHAGENOME_DEVICE"
        echo "SPLICE_TRANSFORMER_DEVICE=$SPLICE_TRANSFORMER_DEVICE"
        echo "MCMC_CANDIDATES_PER_RESULT=$MCMC_CANDIDATES_PER_RESULT"
        echo "SLURM_RESTART_COUNT=$restart_count"
        echo "SHARED_RNA_SPLICING_PROCESS=$SHARED_RNA_SPLICING_PROCESS"
        echo "GENERATE_AGGREGATE_AG_VIS=$GENERATE_AGGREGATE_AG_VIS"
        echo "GENERATE_AGGREGATE_AG_ST_VIS=$GENERATE_AGGREGATE_AG_ST_VIS"
        echo "AG_VIS_VIEW_MODE=$AG_VIS_VIEW_MODE"
        echo "AG_ST_VIS_VIEW_MODE=$AG_ST_VIS_VIEW_MODE"
        echo "VIS_OUTPUT_DIR=$vis_out_dir"
        echo "VIS_LOG_PATH=$vis_log"
        echo "AG_ST_VIS_OUTPUT_DIR=$ag_st_vis_out_dir"
        echo "AG_ST_VIS_LOG_PATH=$ag_st_vis_log"
        echo "VIS_PRIMARY_PLASMID_CONTEXT_PATH=$VIS_PRIMARY_PLASMID_CONTEXT_PATH"
        echo "VIS_PLASMID_CONTEXT_PATHS=$resolved_vis_plasmid_context_paths"
        echo "VIS_GENOMIC_CONTEXT_PATHS=$resolved_vis_genomic_context_paths"
        echo "VIS_GENE_SEQUENCE_PATH=$VIS_GENE_SEQUENCE_PATH"
        echo "VIS_GENE_INSERTION_POS=$VIS_GENE_INSERTION_POS"
        echo "AG_ST_TARGET_TISSUE=$ag_st_target_tissue"
        echo "AG_ST_OFFTARGET_TISSUE=$ag_st_offtarget_tissue"
        echo "ALPHAGENOME_CHECKPOINT_PATH=$ALPHAGENOME_CHECKPOINT_PATH"
    } > "${task_log_dir}/config.env"

    printf '%q ' "${cmd[@]}" > "${task_log_dir}/command.sh"
    printf '\n' >> "${task_log_dir}/command.sh"
    chmod +x "${task_log_dir}/command.sh"

    echo "Running config ${config_id} (task ${SLURM_ARRAY_TASK_ID})"
    {
        echo
        echo "===== $(date --iso-8601=seconds) restart_count=${restart_count} job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID} config=${config_id} ====="
    } >> "$stdout_log"
    {
        echo
        echo "===== $(date --iso-8601=seconds) restart_count=${restart_count} job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID} config=${config_id} ====="
    } >> "$stderr_log"

    env \
        PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS="$SHARED_RNA_SPLICING_PROCESS" \
        ALPHAGENOME_CHECKPOINT_PATH="$ALPHAGENOME_CHECKPOINT_PATH" \
        LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "${cmd[@]}" \
        >> "$stdout_log" \
        2>> "$stderr_log"

    if [[ "$GENERATE_AGGREGATE_AG_VIS" == "true" && "$enable_alphagenome" == "True" ]]; then
        local vis_cmd=(
            python -u examples/bin/visualize_intron_alphagenome_tracks.py
            --stdout_log "${task_log_dir}/stdout.log"
            --log_selection last
            --max_designs 1
            --aggregate_only true
            --output_dir "$vis_out_dir"
            --filename_prefix "config_${config_id}_"
            --target_cell "$target_cell"
            --target_ontology_terms "$target_ontology_terms"
            --offtarget_cell "$offtarget_cell"
            --offtarget_ontology_terms "$offtarget_ontology_terms"
            --plasmid_context_paths "$resolved_vis_plasmid_context_paths"
            --genomic_context_paths "$resolved_vis_genomic_context_paths"
            --gene_sequence_path "$VIS_GENE_SEQUENCE_PATH"
            --gene_insertion_pos "$VIS_GENE_INSERTION_POS"
            --alphagenome_device "$ALPHAGENOME_DEVICE"
            --view_mode "$AG_VIS_VIEW_MODE"
        )

        printf '%q ' "${vis_cmd[@]}" > "${task_log_dir}/visualize_command.sh"
        printf '\n' >> "${task_log_dir}/visualize_command.sh"
        chmod +x "${task_log_dir}/visualize_command.sh"

        env \
            PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS="$SHARED_RNA_SPLICING_PROCESS" \
            ALPHAGENOME_CHECKPOINT_PATH="$ALPHAGENOME_CHECKPOINT_PATH" \
            LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            "${vis_cmd[@]}" \
            >> "$vis_log" \
            2>&1
    fi

    if [[ "$GENERATE_AGGREGATE_AG_ST_VIS" == "true" && "$enable_alphagenome" == "True" && "$enable_splice_transformer" == "True" ]]; then
        local ag_st_vis_cmd=(
            python -u examples/bin/visualize_intron_ag_st_tracks.py
            --stdout_log "${task_log_dir}/stdout.log"
            --log_selection last
            --max_designs 1
            --aggregate_only true
            --output_dir "$ag_st_vis_out_dir"
            --filename_prefix "config_${config_id}_"
            --target_cell "$target_cell"
            --target_ontology_terms "$target_ontology_terms"
            --offtarget_cell "$offtarget_cell"
            --offtarget_ontology_terms "$offtarget_ontology_terms"
            --target_tissue "$ag_st_target_tissue"
            --offtarget_tissue "$ag_st_offtarget_tissue"
            --plasmid_context_paths "$resolved_vis_plasmid_context_paths"
            --genomic_context_paths "$resolved_vis_genomic_context_paths"
            --gene_sequence_path "$VIS_GENE_SEQUENCE_PATH"
            --gene_insertion_pos "$VIS_GENE_INSERTION_POS"
            --alphagenome_track_strand "$alphagenome_track_strand"
            --alphagenome_device "$ALPHAGENOME_DEVICE"
            --splice_transformer_device "$SPLICE_TRANSFORMER_DEVICE"
            --view_mode "$AG_ST_VIS_VIEW_MODE"
        )

        printf '%q ' "${ag_st_vis_cmd[@]}" > "${task_log_dir}/visualize_ag_st_command.sh"
        printf '\n' >> "${task_log_dir}/visualize_ag_st_command.sh"
        chmod +x "${task_log_dir}/visualize_ag_st_command.sh"

        env \
            PROTO_LANGUAGE_RNA_SPLICING_SHARED_PROCESS="$SHARED_RNA_SPLICING_PROCESS" \
            ALPHAGENOME_CHECKPOINT_PATH="$ALPHAGENOME_CHECKPOINT_PATH" \
            LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            "${ag_st_vis_cmd[@]}" \
            >> "$ag_st_vis_log" \
            2>&1
    fi

    cp "${task_log_dir}/config.env" "${run_dir}/config.env"
    cp "${task_log_dir}/command.sh" "${run_dir}/command.sh"
    if [[ -f "${task_log_dir}/visualize_command.sh" ]]; then
        cp "${task_log_dir}/visualize_command.sh" "${run_dir}/visualize_command.sh"
    fi
    if [[ -f "${task_log_dir}/visualize_ag_st_command.sh" ]]; then
        cp "${task_log_dir}/visualize_ag_st_command.sh" "${run_dir}/visualize_ag_st_command.sh"
    fi
}

main() {
    local mode="${1:-prepare}"

    if [[ "$mode" == "prepare" ]]; then
        prepare_grid "$DEFAULT_GRID_PATH"
        local total_configs
        total_configs=$(($(wc -l < "$DEFAULT_GRID_PATH") - 1))
        echo "Wrote grid with ${total_configs} configs: $DEFAULT_GRID_PATH"
        echo "Submit with:"
        echo "  ${SCRIPT_PATH} submit"
        exit 0
    fi

    if [[ "$mode" == "submit" ]]; then
        prepare_grid "$DEFAULT_GRID_PATH"
        local sweep_id="${SWEEP_ID_OVERRIDE:-sweep_$(date +%Y%m%d_%H%M%S)}"
        submit_sweep "$DEFAULT_GRID_PATH" "$sweep_id"
        exit 0
    fi

    if [[ "$mode" == "run" ]]; then
        run_task
        exit 0
    fi

    echo "Unknown mode: $mode" >&2
    echo "Usage: $SCRIPT_PATH [prepare|submit|run]" >&2
    exit 1
}

main "${1:-prepare}"
