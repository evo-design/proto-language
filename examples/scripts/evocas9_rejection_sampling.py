"""Cas9 generation pipeline using a single Rejection Sampling optimizer with filter constraints.

Expresses the multi-stage Cas9 generation pipeline as a proto-language Program
with one RejectionSamplingOptimizer. Reusable filters are core proto-language constraints
ordered cheap -> expensive; the AF3-specific structural screen remains local to this script.
The optimizer's built-in filter short-circuiting (score_energy mask propagation) ensures
expensive filters (AF3) only run on proposals that pass all cheaper ones.

Architecture:
    1 Rejection Sampling optimizer with 1 Evo1Generator + 8 filter constraints:
        1. orf_filter          - ORFipy, ORF >= 3000 bp
        2. cas9_phmm_filter    - PyHmmer vs cas9.hmm
        3. crispr_array_filter - MinCED, >= 3 repeats
        4. identity_filter     - MMseqs2, identity < 90%
        5. gap_gini_filter     - MAFFT + gap Gini < 0.1
        6. domain_filter       - PyHmmer vs cas9_domains.hmm
        7. tracr_filter        - CRISPRtracrRNA + IntaRNA
        8. structure_filter    - AF3 + structure metrics

Usage:
    python evocas9_rejection_sampling.py
    python evocas9_rejection_sampling.py --n-samples 2000 --batch-size 100
    python evocas9_rejection_sampling.py --n-samples 150 --filter-log-output cas9_filter_diagnostics.tsv
"""

import argparse
import csv
import gzip
import logging
import math
import os
from pathlib import Path
from typing import Any

from proto_language.core import ConstraintOutput

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Generation
PROMPT = "`A"  # Backtick = Cas9 class token + one nucleotide
NUM_TOKENS = 8000
MODEL_NAME = "evo-1-8k-crispr"
# Temperatures [0.1, 0.3, 0.5] and top_k [2, 4] were all tested in the paper;
# subsequent analysis showed temp=0.5 dominates the top structural hits.
TEMPERATURES = [0.5]
TOP_KS = [2, 4]

# Data paths (all relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _REPO_ROOT / "examples" / "data"
CAS9_HMM_PATH = str(_DATA_DIR / "cas_hmms" / "cas9.hmm")
DOMAIN_HMM_PATH = str(_DATA_DIR / "cas_hmms" / "cas9_domains.hmm")
TRAINING_FASTA_DIR = _DATA_DIR / "cas_training_proteins"
TRAINING_FASTA_CACHE = _REPO_ROOT / "data" / "cas_training_combined.fasta"

# Stage 1 thresholds
ORF_MIN_LEN = 3000  # Nucleotides
CAS9_PHMM_EVALUE = 1e-3

# Stage 2 thresholds
IDENTITY_THRESHOLD_PCT = 90.0
GAP_GINI_THRESHOLD = 0.1

# Stage 3 thresholds
DOMAIN_EVALUE_THRESHOLD = 1e-10
REQUIRED_DOMAINS = {"RuvC_1", "RuvC_2", "RuvC_3", "HNH"}

# Stage 4 thresholds
PLDDT_THRESHOLD = 75.0
GYRATION_RADIUS_THRESHOLD = 45.0
LONGEST_ALPHA_THRESHOLD = 50

# ============================================================================
# Module-level cache
# ============================================================================

# Key: DNA sequence string
# Values are only AF3/structure artifacts reused if the same sequence recurs.
STRUCTURE_CACHE: dict[str, dict[str, Any]] = {}

FILTER_LOG_COLUMNS = [
    "temperature",
    "top_k",
    "round",
    "proposal_idx",
    "accepted_as_result",
    "passed_all_filters",
    "outcome",
    "failed_filter",
    "energy_score",
    "filter_status_path",
    "dna_length",
    "dna_sequence",
    "protein_sequence",
    "identity",
    "gap_gini",
    "domains_found",
    "crispr_repeat",
    "tracr_rna_sequence",
    "interaction_energy",
    "plddt",
    "gyration_radius",
    "longest_alpha",
    "pdb_path",
]

# ============================================================================
# Helpers
# ============================================================================


def _get_training_fasta() -> Path:
    """Build combined training FASTA from individual .gz files (cached)."""
    if TRAINING_FASTA_CACHE.exists():
        logger.info("Using cached combined training FASTA: %s", TRAINING_FASTA_CACHE)
        return TRAINING_FASTA_CACHE

    if not TRAINING_FASTA_DIR.exists():
        raise FileNotFoundError(f"Training FASTA directory not found: {TRAINING_FASTA_DIR}")

    fasta_files = sorted(TRAINING_FASTA_DIR.glob("*.fasta.gz"))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta.gz files found in {TRAINING_FASTA_DIR}")

    TRAINING_FASTA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    total_seqs = 0
    with open(TRAINING_FASTA_CACHE, "w") as out:
        for fasta in fasta_files:
            with gzip.open(fasta, "rt") as f:
                content = f.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                total_seqs += content.count(">")

    logger.info(
        "Built combined training FASTA (%d sequences from %d files): %s",
        total_seqs,
        len(fasta_files),
        TRAINING_FASTA_CACHE,
    )
    return TRAINING_FASTA_CACHE


# ============================================================================
# Constraint functions
# ============================================================================


def structure_filter(
    input_sequences: list[tuple[Any, ...]],
    config: dict[str, Any],
) -> list[ConstraintOutput]:
    """Filter by AF3 structure prediction + metrics.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import (
        AlphaFold3Config,
        AlphaFold3Input,
        Complex,
        Mmseqs2HomologySearchConfig,
        run_alphafold3,
    )

    from proto_language.utils.orf_selection import predict_longest_canonical_cds

    plddt_threshold = config.get("plddt_threshold", PLDDT_THRESHOLD)
    rg_threshold = config.get("rg_threshold", GYRATION_RADIUS_THRESHOLD)
    alpha_threshold = config.get("alpha_threshold", LONGEST_ALPHA_THRESHOLD)
    af3_dir = config.get("af3_output_dir", "af3_pdbs")
    af3_name = "cas9"

    dna_sequence_objs = [seq_tuple[0] for seq_tuple in input_sequences]
    dna_seqs = [seq.sequence for seq in dna_sequence_objs]
    selected_orfs = predict_longest_canonical_cds(dna_sequence_objs)
    scores = [1.0] * len(input_sequences)
    metadata_per_idx: dict[int, dict[str, Any]] = {}
    structures_per_idx: dict[int, Any] = {}

    for i, (dna, (selected_orf, orf_metadata)) in enumerate(zip(dna_seqs, selected_orfs, strict=True)):
        cache_entry = STRUCTURE_CACHE.setdefault(dna, {})
        metadata: dict[str, Any] = dict(orf_metadata)

        if selected_orf is None:
            metadata.update(
                {
                    "selected_protein_sequence": None,
                    "selected_orf_nucleotide_length": None,
                    "selected_orf_amino_acid_length": None,
                }
            )
            metadata_per_idx[i] = metadata
            continue

        protein = selected_orf.amino_acid_sequence
        metadata.update(
            {
                "selected_protein_sequence": protein,
                "selected_orf_nucleotide_length": selected_orf.nucleotide_length,
                "selected_orf_amino_acid_length": selected_orf.amino_acid_length,
            }
        )

        # Skip AF3 if we already have a PDB for this sequence
        if "pdb_path" in cache_entry:
            pdb_file = Path(cache_entry["pdb_path"])
            if pdb_file.exists():
                plddt = cache_entry.get("plddt")
                rg = cache_entry.get("gyration_radius")
                alpha = cache_entry.get("longest_alpha")
                plddt_ok = plddt is not None and plddt >= plddt_threshold
                rg_ok = rg is not None and rg < rg_threshold
                alpha_ok = alpha is not None and alpha < alpha_threshold
                metadata.update(
                    {
                        "plddt": plddt,
                        "gyration_radius": rg,
                        "longest_alpha": alpha,
                        "pdb_path": str(pdb_file),
                    }
                )
                metadata_per_idx[i] = metadata
                if plddt_ok and rg_ok and alpha_ok:
                    scores[i] = 0.0
                continue

        # AF3 structure prediction
        af3_idx = getattr(structure_filter, "_next_idx", 0)
        structure_filter._next_idx = af3_idx + 1  # type: ignore[attr-defined]
        proposal_dir = f"{af3_dir}/{af3_name}_{af3_idx}"
        try:
            af3_result = run_alphafold3(
                AlphaFold3Input(complexes=[Complex(chains=[protein])]),
                AlphaFold3Config(
                    name=af3_name,
                    output_dir=proposal_dir,
                    use_msa=True,
                    msa_search_config=Mmseqs2HomologySearchConfig(search_mode="local"),
                ),
            )
        except Exception as e:
            logger.error("  structure_filter: AF3 failed for proposal: %s", e)
            continue

        structure = af3_result.structures[0]
        plddt = structure.metrics.get("avg_plddt")
        cache_entry["plddt"] = plddt

        # Find PDB path
        output_dir = Path(f"{proposal_dir}_af3_results")
        pdb_file = output_dir / f"{af3_name}_0_af3.pdb"
        if not pdb_file.exists():
            # Try to find any PDB in the output directory
            pdb_files = list(output_dir.glob("*.pdb")) if output_dir.exists() else []
            if pdb_files:
                pdb_file = pdb_files[0]
            else:
                logger.error("  structure_filter: PDB not found for proposal")
                continue

        cache_entry["pdb_path"] = str(pdb_file)

        # Structure metrics
        from proto_tools import (
            StructureMetricsConfig,
            StructureMetricsInput,
            run_structure_metrics,
        )

        metrics_result = run_structure_metrics(
            StructureMetricsInput(pdb_paths=[str(pdb_file)]),
            StructureMetricsConfig(),
        )

        if metrics_result.metrics:
            m = metrics_result.metrics[0]
            cache_entry["gyration_radius"] = m.gyration_radius
            cache_entry["longest_alpha"] = m.longest_alpha_helix

            # Combined filter
            plddt_ok = plddt is not None and plddt >= plddt_threshold
            rg_ok = m.gyration_radius is not None and m.gyration_radius < rg_threshold
            alpha_ok = m.longest_alpha_helix is not None and m.longest_alpha_helix < alpha_threshold

            metadata.update(
                {
                    "plddt": plddt,
                    "gyration_radius": m.gyration_radius,
                    "longest_alpha": m.longest_alpha_helix,
                    "pdb_path": str(pdb_file),
                }
            )
            metadata_per_idx[i] = metadata
            structures_per_idx[i] = structure

            if plddt_ok and rg_ok and alpha_ok:
                scores[i] = 0.0
            else:
                reasons = []
                if not plddt_ok:
                    reasons.append(f"pLDDT={plddt}")
                if not rg_ok:
                    reasons.append(f"Rg={m.gyration_radius}")
                if not alpha_ok:
                    reasons.append(f"alpha={m.longest_alpha_helix}")
                logger.info("  structure_filter: FAIL (%s)", "; ".join(reasons))

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info("structure_filter: %d/%d passed structure checks", n_pass, len(scores))

    return [
        ConstraintOutput(
            score=scores[i],
            metadata=metadata_per_idx.get(i, {}),
            structures=(structures_per_idx.get(i),),
        )
        for i in range(len(scores))
    ]


structure_filter._next_idx = 0  # type: ignore[attr-defined]


# Suppress validation warnings by setting supported sequence types
structure_filter._constraint_supported_sequence_types = ["dna"]  # type: ignore[attr-defined]
structure_filter._constraint_num_input_sequences_per_tuple = 1  # type: ignore[attr-defined]


# ============================================================================
# Program builder
# ============================================================================


def build_program(
    n_samples: int,
    temperature: float,
    top_k_val: int,
    batch_size: int,
    verbose: bool = False,
    af3_output_dir: str = "af3_pdbs",
    filter_log_output: Path | None = None,
) -> tuple[Any, Any]:
    """Build a Program with a single Rejection Sampling optimizer for one (temp, top_k) combo.

    Returns:
        (program, segment) tuple for result collection.
    """
    from proto_tools import CrisprTracrRNAConfig, MincedConfig, PyHmmsearchConfig

    from proto_language import (
        Evo1Generator,
        Evo1GeneratorConfig,
        RejectionSamplingOptimizer,
        RejectionSamplingOptimizerConfig,
    )
    from proto_language.constraint import (
        crispr_array_constraint,
        crispr_tracr_rna_constraint,
        longest_orf_length_constraint,
        protein_max_identity_constraint,
        protein_nearest_neighbor_gap_gini_constraint,
        protein_profile_hmm_constraint,
    )
    from proto_language.constraint.protein_quality.protein_max_identity_constraint import (
        ProteinMaxIdentityConfig,
    )
    from proto_language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint import (
        ProteinNearestNeighborGapGiniConfig,
    )
    from proto_language.constraint.protein_quality.protein_profile_hmm_constraint import (
        ProteinProfileHMMConfig,
    )
    from proto_language.constraint.sequence_annotation.crispr_array_constraint import CrisprArrayConfig
    from proto_language.constraint.sequence_annotation.orf_length_constraint import LongestOrfLengthConfig
    from proto_language.constraint.sequence_annotation.tracr_rna_constraint import (
        CrisprTracrRNAConstraintConfig,
    )
    from proto_language.core import Constraint, Construct, Program, Segment

    # Segment: DNA sequence of length NUM_TOKENS (Evo1 output length)
    # prepend_prompt=False means max_new_tokens = segment.sequence_length
    segment = Segment(
        length=NUM_TOKENS,
        sequence_type="dna",
        label="crispr_locus",
    )
    construct = Construct([segment], label="cas9_construct")

    # Generator
    gen_config = Evo1GeneratorConfig(
        prompts=[PROMPT],
        model_checkpoint=MODEL_NAME,
        top_k=top_k_val,
        temperature=temperature,
        prepend_prompt=False,
        batch_size=batch_size,
        verbose=verbose,
    )
    generator = Evo1Generator(gen_config)
    generator.assign(segment)

    training_fasta = str(_get_training_fasta())
    tracr_workers = len(os.sched_getaffinity(0)) or 1

    # Every constraint below has a threshold, so Optimizer.score_energy treats them
    # as ordered filters. A proposal that fails one filter is not evaluated by
    # later filters; optional filter diagnostics record those later filters as SKIPPED.
    constraints = [
        Constraint(
            inputs=[segment],
            function=longest_orf_length_constraint,
            function_config=LongestOrfLengthConfig(min_nucleotide_length=ORF_MIN_LEN),
            label="orf_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=protein_profile_hmm_constraint,
            function_config=ProteinProfileHMMConfig(
                hmm_path=CAS9_HMM_PATH,
                hmmsearch_config=PyHmmsearchConfig(
                    evalue_threshold=CAS9_PHMM_EVALUE,
                    domain_evalue_threshold=CAS9_PHMM_EVALUE,
                ),
            ),
            label="cas9_phmm_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=crispr_array_constraint,
            function_config=CrisprArrayConfig(
                minced_config=MincedConfig(min_num_repeats=3, min_repeat_length=23),
            ),
            label="crispr_array_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=protein_max_identity_constraint,
            function_config=ProteinMaxIdentityConfig(
                mmseqs_db=training_fasta,
                reference_fasta=training_fasta,
                max_identity=IDENTITY_THRESHOLD_PCT,
                pass_no_hits=True,
            ),
            label="identity_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=protein_nearest_neighbor_gap_gini_constraint,
            function_config=ProteinNearestNeighborGapGiniConfig(
                mmseqs_db=training_fasta,
                reference_fasta=training_fasta,
                max_gap_gini=GAP_GINI_THRESHOLD,
                pass_no_hits=True,
            ),
            label="gap_gini_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=protein_profile_hmm_constraint,
            function_config=ProteinProfileHMMConfig(
                hmm_path=DOMAIN_HMM_PATH,
                required_profiles=sorted(REQUIRED_DOMAINS),
                profile_match_field="query_name",
                hmmsearch_config=PyHmmsearchConfig(
                    evalue_threshold=DOMAIN_EVALUE_THRESHOLD,
                    domain_evalue_threshold=DOMAIN_EVALUE_THRESHOLD,
                ),
            ),
            label="domain_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=crispr_tracr_rna_constraint,
            function_config=CrisprTracrRNAConstraintConfig(
                tracr_config=CrisprTracrRNAConfig(model_type="II", num_workers=tracr_workers)
            ),
            label="tracr_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=structure_filter,
            function_config={
                "plddt_threshold": PLDDT_THRESHOLD,
                "rg_threshold": GYRATION_RADIUS_THRESHOLD,
                "alpha_threshold": LONGEST_ALPHA_THRESHOLD,
                "af3_output_dir": af3_output_dir,
            },
            label="structure_filter",
            threshold=0.5,
        ),
    ]

    # Rejection Sampling optimizer: num_results = num_samples to keep all survivors
    optimizer_config = RejectionSamplingOptimizerConfig(
        num_samples=n_samples,
        num_results=n_samples,
        verbose=verbose,
    )
    optimizer = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=optimizer_config,
    )
    if filter_log_output is not None:
        filter_specs = [
            (constraint.label, constraint.threshold) for constraint in constraints if constraint.threshold is not None
        ]
        optimizer.custom_logging = _make_filter_result_logger(
            optimizer=optimizer,
            filter_specs=filter_specs,
            output_path=filter_log_output,
            temperature=temperature,
            top_k_val=top_k_val,
        )

    program = Program(optimizers=[optimizer], num_results=n_samples, verbose=verbose)
    return program, segment


# ============================================================================
# Result collection
# ============================================================================


def collect_results(
    segment: Any,
    temperature: float,
    top_k_val: int,
) -> list[dict[str, Any]]:
    """Collect passing proposals from segment result sequences and constraint metadata."""
    results = []
    for i, seq in enumerate(segment.result_sequences):
        dna = seq.sequence
        if not dna:
            continue  # Skip empty padding

        orf_data = _constraint_data(seq, "orf_filter")
        identity_data = _constraint_data(seq, "identity_filter")
        gap_data = _constraint_data(seq, "gap_gini_filter")
        domain_data = _constraint_data(seq, "domain_filter")
        crispr_data = _constraint_data(seq, "crispr_array_filter")
        tracr_data = _constraint_data(seq, "tracr_filter")
        structure_data = _constraint_data(seq, "structure_filter")
        generator_data = seq._generator_metadata.get("evo1", {})
        results.append(
            {
                "proposal_idx": i,
                "temperature": temperature,
                "top_k": top_k_val,
                "score": generator_data.get("score"),
                "identity": identity_data.get("identity"),
                "gap_gini": gap_data.get("gap_gini"),
                "domains_found": domain_data.get("profiles_found", []),
                "interaction_energy": tracr_data.get("interaction_energy"),
                "plddt": structure_data.get("plddt"),
                "gyration_radius": structure_data.get("gyration_radius"),
                "longest_alpha": structure_data.get("longest_alpha"),
                "pdb_path": structure_data.get("pdb_path"),
                "dna_sequence": dna,
                "crispr_repeat": crispr_data.get("crispr_repeat"),
                "tracr_rna_sequence": tracr_data.get("tracr_sequence"),
                "protein_sequence": orf_data.get("selected_protein_sequence"),
            }
        )

    return results


def _constraint_data(seq: Any, label: str) -> dict[str, Any]:
    """Return custom metadata stored by a named constraint."""
    data = seq._constraints_metadata.get(label, {}).get("data", {})
    return data if isinstance(data, dict) else {}


def _make_filter_result_logger(
    optimizer: Any,
    filter_specs: list[tuple[str, float]],
    output_path: Path,
    temperature: float,
    top_k_val: int,
) -> Any:
    """Create a callback that writes per-proposal filter diagnostics."""

    def _log_filter_results(round_num: int, segments: tuple[Any, ...]) -> None:
        rows = _collect_filter_log_rows(
            round_num=round_num,
            segments=segments,
            filter_specs=filter_specs,
            outcomes=optimizer._proposal_outcomes,
            energy_scores=optimizer._proposal_energy_scores,
            temperature=temperature,
            top_k_val=top_k_val,
        )
        _append_filter_log_rows(output_path, rows)
        summary = _summarize_filter_log_rows(rows)
        logger.info(
            "filter diagnostics: round=%d temp=%s top_k=%s accepted=%d passed_filters=%d failed=%s",
            round_num,
            temperature,
            top_k_val,
            summary["accepted_as_result"],
            summary["passed_all_filters"],
            summary["failed_by_filter"],
        )

    return _log_filter_results


def _collect_filter_log_rows(
    round_num: int,
    segments: tuple[Any, ...],
    filter_specs: list[tuple[str, float]],
    outcomes: list[str],
    energy_scores: list[float],
    temperature: float,
    top_k_val: int,
) -> list[dict[str, Any]]:
    """Collect per-proposal filter diagnostics, including short-circuited filters."""
    if not segments:
        return []

    rows: list[dict[str, Any]] = []
    for proposal_idx, seq in enumerate(segments[0].proposal_sequences):
        filter_statuses, failed_filter = _filter_status_path(seq, filter_specs)
        passed_all_filters = failed_filter is None
        outcome = outcomes[proposal_idx] if proposal_idx < len(outcomes) else "unknown"
        orf_data = _constraint_data(seq, "orf_filter")
        identity_data = _constraint_data(seq, "identity_filter")
        gap_data = _constraint_data(seq, "gap_gini_filter")
        domain_data = _constraint_data(seq, "domain_filter")
        crispr_data = _constraint_data(seq, "crispr_array_filter")
        tracr_data = _constraint_data(seq, "tracr_filter")
        structure_data = _constraint_data(seq, "structure_filter")

        rows.append(
            {
                "temperature": temperature,
                "top_k": top_k_val,
                "round": round_num,
                "proposal_idx": proposal_idx,
                "accepted_as_result": outcome == "accepted",
                "passed_all_filters": passed_all_filters,
                "outcome": outcome,
                "failed_filter": failed_filter,
                "energy_score": _format_float(
                    energy_scores[proposal_idx] if proposal_idx < len(energy_scores) else None
                ),
                "filter_status_path": ";".join(filter_statuses),
                "dna_length": len(seq.sequence),
                "dna_sequence": seq.sequence,
                "protein_sequence": orf_data.get("selected_protein_sequence"),
                "identity": _format_float(identity_data.get("identity")),
                "gap_gini": _format_float(gap_data.get("gap_gini")),
                "domains_found": ",".join(domain_data.get("profiles_found", [])),
                "crispr_repeat": crispr_data.get("crispr_repeat"),
                "tracr_rna_sequence": tracr_data.get("tracr_sequence"),
                "interaction_energy": _format_float(tracr_data.get("interaction_energy")),
                "plddt": _format_float(structure_data.get("plddt")),
                "gyration_radius": _format_float(structure_data.get("gyration_radius")),
                "longest_alpha": _format_float(structure_data.get("longest_alpha")),
                "pdb_path": structure_data.get("pdb_path"),
            }
        )
    return rows


def _filter_status_path(seq: Any, filter_specs: list[tuple[str, float]]) -> tuple[list[str], str | None]:
    """Return per-filter PASS/FAIL/SKIPPED statuses and the first failing filter."""
    statuses: list[str] = []
    failed_filter = None
    for label, threshold in filter_specs:
        constraint_entry = seq._constraints_metadata.get(label)
        if constraint_entry is None:
            status = "SKIPPED" if failed_filter is not None else "NOT_EVALUATED"
        else:
            score = constraint_entry.get("score")
            if score is not None and score <= threshold:
                status = "PASS"
            else:
                status = "FAIL"
                failed_filter = failed_filter or label
        statuses.append(f"{label}:{status}")
    return statuses, failed_filter


def _append_filter_log_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Append filter diagnostics rows to a TSV file."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with open(output_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILTER_LOG_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _summarize_filter_log_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize filter diagnostics for concise console logging."""
    failed_by_filter: dict[str, int] = {}
    for row in rows:
        failed_filter = row["failed_filter"]
        if failed_filter:
            failed_by_filter[failed_filter] = failed_by_filter.get(failed_filter, 0) + 1
    return {
        "accepted_as_result": sum(1 for row in rows if row["accepted_as_result"]),
        "passed_all_filters": sum(1 for row in rows if row["passed_all_filters"]),
        "failed_by_filter": failed_by_filter,
    }


def _format_float(value: Any) -> str:
    """Format optional numeric values for TSV output."""
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.6g}"


def save_results(
    results: list[dict[str, Any]],
    output_tsv: Path,
    output_fasta: Path,
) -> None:
    """Save results to TSV and FASTA files."""
    columns = [
        "proposal_idx",
        "temperature",
        "top_k",
        "score",
        "identity",
        "gap_gini",
        "domains_found",
        "interaction_energy",
        "plddt",
        "gyration_radius",
        "longest_alpha_helix",
        "pdb_path",
        "dna_sequence",
        "crispr_repeat",
        "tracr_rna_sequence",
        "protein_sequence",
    ]

    with open(output_tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "proposal_idx": r["proposal_idx"],
                    "temperature": r["temperature"],
                    "top_k": r["top_k"],
                    "score": (f"{r['score']:.4f}" if r["score"] is not None else ""),
                    "identity": (f"{r['identity']:.4f}" if r["identity"] is not None else ""),
                    "gap_gini": (f"{r['gap_gini']:.4f}" if r["gap_gini"] is not None else ""),
                    "domains_found": (",".join(r["domains_found"]) if r["domains_found"] else ""),
                    "interaction_energy": (
                        f"{r['interaction_energy']:.2f}" if r["interaction_energy"] is not None else ""
                    ),
                    "plddt": (f"{r['plddt']:.1f}" if r["plddt"] is not None else ""),
                    "gyration_radius": (f"{r['gyration_radius']:.1f}" if r["gyration_radius"] is not None else ""),
                    "longest_alpha_helix": (r["longest_alpha"] if r["longest_alpha"] is not None else ""),
                    "pdb_path": r["pdb_path"] or "",
                    "dna_sequence": r["dna_sequence"],
                    "crispr_repeat": r["crispr_repeat"] or "",
                    "tracr_rna_sequence": r["tracr_rna_sequence"] or "",
                    "protein_sequence": r["protein_sequence"] or "",
                }
            )

    logger.info("Summary TSV written to: %s", output_tsv)

    with open(output_fasta, "w") as f:
        for r in results:
            plddt_str = f" plddt={r['plddt']:.1f}" if r["plddt"] is not None else ""
            header = f">cas9_proposal_{r['proposal_idx']} temp={r['temperature']} top_k={r['top_k']}{plddt_str}"
            f.write(f"{header}\n{r['dna_sequence']}\n")

    logger.info("FASTA written to: %s", output_fasta)


# ============================================================================
# Main
# ============================================================================


def main() -> list[dict[str, Any]]:
    """Run the EvoCas9 rejection-sampling pipeline."""
    parser = argparse.ArgumentParser(description="Cas9 generation pipeline (Rejection Sampling optimizer version)")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Samples per sweep combination (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for generation (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cas9_proposals.fasta",
        help="Output FASTA for passing proposals (default: cas9_proposals.fasta)",
    )
    parser.add_argument(
        "--filter-log-output",
        type=str,
        default=None,
        help="Optional TSV path for per-proposal filter diagnostics, including skipped filters after short-circuiting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    batch_size = args.batch_size or args.n_samples
    filter_log_output = Path(args.filter_log_output) if args.filter_log_output else None
    if filter_log_output is not None:
        filter_log_output.unlink(missing_ok=True)

    # Enable tool caching
    from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache

    _program_tool_cache.set(ToolCache())

    n_combos = len(TEMPERATURES) * len(TOP_KS)
    total_seqs = args.n_samples * n_combos
    logger.info("=" * 60)
    logger.info("Cas9 Generation Pipeline (Rejection Sampling Optimizer)")
    logger.info("=" * 60)
    logger.info("Sweep: %d temps x %d top_k = %d combos", len(TEMPERATURES), len(TOP_KS), n_combos)
    logger.info("Samples per combo: %d", args.n_samples)
    logger.info("Total sequences: %d", total_seqs)
    logger.info("Batch size: %d", batch_size)
    logger.info("=" * 60)

    all_results: list[dict[str, Any]] = []
    combo_idx = 0

    for temp in TEMPERATURES:
        for top_k_val in TOP_KS:
            combo_idx += 1
            logger.info("\nCombo %d/%d: temp=%s, top_k=%s", combo_idx, n_combos, temp, top_k_val)

            STRUCTURE_CACHE.clear()
            structure_filter._next_idx = 0  # type: ignore[attr-defined]

            output_base = Path(args.output).stem.replace("_proposals", "")
            af3_output_dir = f"{output_base}_af3_pdbs/temp{temp}_topk{top_k_val}"
            program, segment = build_program(
                n_samples=args.n_samples,
                temperature=temp,
                top_k_val=top_k_val,
                batch_size=batch_size,
                verbose=args.verbose,
                af3_output_dir=af3_output_dir,
                filter_log_output=filter_log_output,
            )
            program.run()

            results = collect_results(segment, temp, top_k_val)
            all_results.extend(results)

            logger.info("Combo %d: %d proposals passed all filters", combo_idx, len(results))

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("Total combos: %d", n_combos)
    logger.info("Total sequences generated: %d", total_seqs)
    logger.info("Total passing proposals: %d", len(all_results))
    logger.info("=" * 60)

    if all_results:
        output_fasta = Path(args.output)
        output_tsv = output_fasta.with_suffix(".tsv")
        save_results(all_results, output_tsv, output_fasta)
        logger.info("\nPassing proposals written to: %s", output_fasta)
        logger.info("Summary TSV written to: %s", output_tsv)
    else:
        logger.info("\nNo proposals passed all filters.")
    if filter_log_output is not None:
        logger.info("Filter diagnostics TSV written to: %s", filter_log_output)

    return all_results


if __name__ == "__main__":
    main()
