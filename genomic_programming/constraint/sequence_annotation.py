"""
Sequence annotation constraints for gene prediction and regulatory element identification.
"""

from __future__ import annotations
import tempfile
import os
import subprocess
import math
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Literal
import numpy as np
import pandas as pd
from promoter_calculator.wrapper import promoter_calculator
from ..base import *
from ..utils import resolve_paths
from ..tool_cache import ToolCache
from ..schemas import ORFipyKwargs, MMseqsKwargs
from ..tools.orf_prediction.orfipy import run_orfipy, parse_orfipy_results_to_df
from ..tools.gene_annotation.mmseqs import run_mmseqs_search_proteins
from .utils import (
    MIN_ENERGY,
    MAX_ENERGY,
    _calculate_range_deviation,
    _calculate_percentage_range_deviation,
)


def _run_orfipy_mmseqs_pipeline(
    input_sequence: Sequence,
    orfipy_kwargs: Optional[ORFipyKwargs] = None,
    mmseqs_kwargs: Optional[MMseqsKwargs] = None,
) -> None:
    """
    Run the ORFipy + MMseqs pipeline for sequence analysis.

    Args:
        input_sequence: The sequence to evaluate.
        orfipy_kwargs: ORFipy configuration arguments.
        mmseqs_kwargs: MMseqs configuration arguments.

    Note:
        Results are cached based on sequence and parameters to avoid redundant analysis.
        Updates metadata with 'orfipy_orfs', 'mmseqs_results', and 'unique_orfs_with_hits'.
    """
    # Use defaults if not provided
    if orfipy_kwargs is None:
        orfipy_kwargs = ORFipyKwargs()
    if mmseqs_kwargs is None:
        raise ValueError("MMseqs database path is required")

    # Convert to dictionaries and resolve paths
    orfipy_kwargs_dict = resolve_paths(orfipy_kwargs.model_dump())
    mmseqs_kwargs_dict = resolve_paths(mmseqs_kwargs.model_dump())

    # Check if analysis already cached
    cached_results = ToolCache.get_cached_results(
        input_sequence,
        "orfipy_mmseqs",
        orfipy_kwargs=orfipy_kwargs_dict,
        mmseqs_kwargs=mmseqs_kwargs_dict,
    )
    if cached_results:
        input_sequence._metadata.update(cached_results)
        return

    # Preprocess sequence by removing all characters that are not ACTG
    sequence_to_analyze = "".join(
        char for char in input_sequence.sequence.upper() if char in "ACTG"
    )

    # Run the expensive analysis
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write sequence to temporary FASTA file
        input_fasta = temp_path / "input.fasta"
        with open(input_fasta, "w") as f:
            f.write(f">input_sequence\n{sequence_to_analyze}\n")

        # Run ORFipy
        orfipy_output = temp_path / "orfipy_output"
        aa_fasta, nt_fasta = run_orfipy(
            input_fasta, output_dir=orfipy_output, **orfipy_kwargs_dict
        )

        # Parse ORFipy results
        orfs_df = parse_orfipy_results_to_df(aa_fasta, nt_fasta)

        if orfs_df.empty:
            # No ORFs found (store as empty lists for JSON serialization)
            results = {
                "orfipy_orfs": [],
                "mmseqs_results": [],
                "unique_orfs_with_hits": 0,
            }
        else:
            # Run MMseqs search for each ORF
            mmseqs_output = temp_path / "mmseqs_output"
            mmseqs_results = run_mmseqs_search_proteins(
                aa_fasta,
                mmseqs_kwargs_dict.get(
                    "database", ""
                ),  # Database path should be provided in config
                mmseqs_output,
                **{k: v for k, v in mmseqs_kwargs_dict.items() if k != "database"},
            )

            # Count unique ORFs with hits
            unique_orfs_with_hits = (
                len(mmseqs_results) if not mmseqs_results.empty else 0
            )

            # Store results (convert DataFrames to dicts for JSON serialization)
            results = {
                "orfipy_orfs": orfs_df.to_dict("records") if not orfs_df.empty else [],
                "mmseqs_results": (
                    mmseqs_results.to_dict("records")
                    if not mmseqs_results.empty
                    else []
                ),
                "unique_orfs_with_hits": unique_orfs_with_hits,
            }

    # Cache results and update metadata
    ToolCache.cache_results(
        input_sequence,
        "orfipy_mmseqs",
        results,
        orfipy_kwargs=orfipy_kwargs_dict,
        mmseqs_kwargs=mmseqs_kwargs_dict,
    )
    input_sequence._metadata.update(results)


def orfipy_mmseqs_gene_hit_count_constraint(
    input_sequence: Sequence,
    min_hits: int,
    max_hits: int,
    orfipy_kwargs: Optional[ORFipyKwargs] = None,
    mmseqs_kwargs: Optional[MMseqsKwargs] = None,
) -> float:
    """
    Evaluate whether the number of unique ORFs with hits falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        min_hits: Minimum acceptable number of unique ORFs with hits.
        max_hits: Maximum acceptable number of unique ORFs with hits.
        orfipy_kwargs: ORFipy configuration arguments.
        mmseqs_kwargs: MMseqs configuration arguments (database path required).

    Returns:
        Constraint score where 0.0 indicates the hit count is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Examples:
        Evaluating ORF hit count constraint:

        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> orfipy_kwargs = ORFipyKwargs(threads=48)
        >>> mmseqs_kwargs = MMseqsKwargs(database="/path/to/protein_db")
        >>> score = orfipy_mmseqs_gene_hit_count_constraint(seq, 1, 5, orfipy_kwargs, mmseqs_kwargs)
    """
    # Run the pipeline
    _run_orfipy_mmseqs_pipeline(input_sequence, orfipy_kwargs, mmseqs_kwargs)

    # Get the count of unique ORFs with hits (directly from metadata)
    unique_orfs_with_hits = input_sequence._metadata.get("unique_orfs_with_hits", 0)

    # Calculate range deviation
    return _calculate_range_deviation(unique_orfs_with_hits, min_hits, max_hits)


def orfipy_mmseqs_gene_homology_constraint(
    input_sequence: Sequence,
    min_homology: float,
    max_homology: float,
    orfipy_kwargs: Optional[ORFipyKwargs] = None,
    mmseqs_kwargs: Optional[MMseqsKwargs] = None,
) -> float:
    """
    Evaluate the homology (percent identity) of each individual ORF hit.

    Args:
        input_sequence: The sequence to evaluate.
        min_homology: Minimum acceptable percent identity (0-100) for each ORF.
        max_homology: Maximum acceptable percent identity (0-100) for each ORF.
        orfipy_kwargs: ORFipy configuration arguments.
        mmseqs_kwargs: MMseqs configuration arguments (database path required).

    Returns:
        Constraint score where 0.0 indicates all ORF homologies are within acceptable range
        and higher values indicate more ORFs with homology outside the acceptable range.

    Examples:
        Evaluating ORF homology constraint:

        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> orfipy_kwargs = ORFipyKwargs(threads=48)
        >>> mmseqs_kwargs = MMseqsKwargs(database="/path/to/protein_db")
        >>> score = orfipy_mmseqs_gene_homology_constraint(seq, 50.0, 90.0, orfipy_kwargs, mmseqs_kwargs)
    """
    # Run the pipeline
    _run_orfipy_mmseqs_pipeline(input_sequence, orfipy_kwargs, mmseqs_kwargs)

    # Get the MMseqs results (convert from dict records if needed)
    mmseqs_results_data = input_sequence._metadata.get("mmseqs_results", [])
    if isinstance(mmseqs_results_data, list):
        mmseqs_results = (
            pd.DataFrame(mmseqs_results_data) if mmseqs_results_data else pd.DataFrame()
        )
    else:
        mmseqs_results = mmseqs_results_data
    total_orfs_with_hits = input_sequence._metadata.get("unique_orfs_with_hits", 0)

    if mmseqs_results.empty:
        # No hits found - return max penalty
        input_sequence._metadata["orfs_with_acceptable_homology"] = 0
        input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
        input_sequence._metadata["homology_compliance_rate"] = 0.0
        return MAX_ENERGY

    # Use standardized identity column
    if "identity" not in mmseqs_results.columns:
        input_sequence._metadata["orfs_with_acceptable_homology"] = 0
        input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
        input_sequence._metadata["homology_compliance_rate"] = 0.0
        return MAX_ENERGY

    # Check each ORF's homology individually
    acceptable_homology_count = 0
    homology_violations = []

    for _, row in mmseqs_results.iterrows():
        homology = row["identity"]
        if min_homology <= homology <= max_homology:
            acceptable_homology_count += 1
        else:
            # Calculate how far this ORF's homology deviates from acceptable range
            deviation = _calculate_percentage_range_deviation(
                homology, min_homology, max_homology
            )
            homology_violations.append(deviation)

    # Store metadata for inspection
    input_sequence._metadata["orfs_with_acceptable_homology"] = (
        acceptable_homology_count
    )
    input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
    input_sequence._metadata["homology_compliance_rate"] = (
        acceptable_homology_count / total_orfs_with_hits
    )

    # If all ORFs have acceptable homology, return 0
    if not homology_violations:
        return MIN_ENERGY

    # Return the average deviation of ORFs that violate the homology constraint
    return min(MAX_ENERGY, np.mean(homology_violations))


def sigma70_promoter_constraint(
    sequences: Union["Sequence", List["Sequence"]],
    config: Optional[Dict[str, Any]] = None,
) -> Union[float, List[float]]:
    """
    Evaluate σ70 promoter strength for one or more Sequence objects.
    Results are cached in each Sequence's metadata under key 'sigma70'.

    Args:
        sequences: A Sequence or list of Sequences (DNA only).
        config: Optional override parameters for sigma 70 promoter
        scoring.

    Returns:
        A float penalty (0 best, 1 worst) or a list of penalties if batch.
    """

    # Default sigma 70 scoring parameters
    default_config = {
        "consensus_35": "TTGACA",
        "consensus_10": "TATAAT",
        "probs_35": [0.69, 0.79, 0.61, 0.56, 0.54, 0.54],
        "probs_10": [0.77, 0.76, 0.60, 0.61, 0.56, 0.82],
        "optimal_spacer": 17,
        "spacer_sigma": 1.5,
        "spacer_weight": 0.3,
        "gamma": 0.1,
        "k_opt": 8,
        "match_sigma": 2.0,
        "match_weight": 0.3,
        "min_spacer": 14,
        "max_spacer": 20,
    }
    config = {**default_config, **(config or {})}

    CONS_35 = config["consensus_35"].upper()
    CONS_10 = config["consensus_10"].upper()
    PROBS_35 = np.array(config["probs_35"])
    PROBS_10 = np.array(config["probs_10"])
    max_pwm = np.prod(PROBS_35) * np.prod(PROBS_10)

    def _score_promoter(box35: str, box10: str, spacer_len: int):
        prob_35 = np.prod(
            [
                prob if b == c else (1.0 - prob)
                for b, c, prob in zip(box35, CONS_35, PROBS_35)
            ]
        )
        prob_10 = np.prod(
            [
                prob if b == c else (1.0 - prob)
                for b, c, prob in zip(box10, CONS_10, PROBS_10)
            ]
        )
        raw_pwm = prob_35 * prob_10
        normalized_pwm = (raw_pwm / max_pwm) if max_pwm > 0 else 0
        pwm_score = normalized_pwm ** config["gamma"]
        pwm_penalty = 1.0 - pwm_score

        total_matches = sum(a == c for a, c in zip(box35, CONS_35)) + sum(
            a == c for a, c in zip(box10, CONS_10)
        )
        match_dev = (total_matches - config["k_opt"]) / config["match_sigma"]
        match_penalty = 1.0 - math.exp(-(match_dev**2))

        spacer_dev = (spacer_len - config["optimal_spacer"]) / config["spacer_sigma"]
        spacer_penalty = 1.0 - math.exp(-(spacer_dev**2))

        box_penalty = (1 - config["match_weight"]) * pwm_penalty + config[
            "match_weight"
        ] * match_penalty
        total_penalty = (1 - config["spacer_weight"]) * box_penalty + config[
            "spacer_weight"
        ] * spacer_penalty

        return max(0.0, min(1.0, total_penalty)), {
            "pwm_penalty": pwm_penalty,
            "match_penalty": match_penalty,
            "spacer_penalty": spacer_penalty,
            "total_matches": total_matches,
            "spacer_len": spacer_len,
        }

    is_single = isinstance(sequences, Sequence)
    if is_single:
        sequences = [sequences]

    penalties: List[float] = []

    for seq_obj in sequences:
        seq = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        seq_len = len(seq)

        best_score, best_info = 1.0, {}
        if seq_len < 12:
            best_score, best_info = 1.0, {"reason": "too_short"}
        elif seq_len <= 32:  # treat as fixed promoter
            spacer_len = seq_len - 12
            if config["min_spacer"] <= spacer_len <= config["max_spacer"]:
                box35, box10 = seq[0:6], seq[-6:]
                best_score, best_info = _score_promoter(box35, box10, spacer_len)
                best_info.update({"pos": 0, "box35": box35, "box10": box10})
            else:
                best_score, best_info = 1.0, {"reason": "invalid_spacer"}
        else:  # scan for best
            for spacer_len in range(config["min_spacer"], config["max_spacer"] + 1):
                promoter_len = 12 + spacer_len
                if promoter_len > seq_len:
                    continue
                for pos in range(seq_len - promoter_len + 1):
                    box35 = seq[pos : pos + 6]
                    box10 = seq[pos + 6 + spacer_len : pos + 12 + spacer_len]
                    score, info = _score_promoter(box35, box10, spacer_len)
                    if score < best_score:
                        best_score, best_info = score, {
                            **info,
                            "pos": pos,
                            "box35": box35,
                            "box10": box10,
                        }

        # Cache results in metadata
        seq_obj._metadata["sigma70"] = {
            "sigma70_score": best_score,
            **best_info,
        }
        penalties.append(best_score)

    return penalties[0] if is_single else penalties


def seq_motif_constraint(
    sequences: Union["Sequence", List["Sequence"]],
    motifs_path: str,
    meme_bin_path: str,
    wanted: Union[str, List[str], None] = None,
    not_wanted: Union[str, List[str], None] = None,
    scale: float = 1.0,
    exclusive: bool = False,
    aggregation: Literal["smart", "average", "max", "percentile"] = "smart",
    percentile_value: float = 95.0,
    unwanted_focus: bool = True,
) -> Union[float, List[float]]:
    """
    Score one or more DNA Sequences against motifs using MEME.

    Modified scoring:
    - Unwanted motifs: Strong matches (low e-value) get high penalties
    - Wanted motifs: Strong matches (low e-value) get LOW penalties (rewards)

    Aggregation strategies for handling many motifs:
    - "smart": Uses max/percentile for unwanted, average for wanted
    - "average": Simple average of all penalties
    - "max": Takes maximum penalty
    - "percentile": Uses specified percentile of penalties

    Args:
        sequences: Sequence or list of sequences to evaluate
        motifs_path: Path to MEME motif file
        meme_bin_path: Path to MEME binaries
        wanted: Motifs that should be present
        not_wanted: Motifs that should not be present
        scale: Scaling factor for penalties
        exclusive: If True, automatically sets complement (e.g., one TF motif set for wanted, sets unwanted to all others)
        aggregation: Aggregation strategy to combine multiple penalties
        percentile_value: Which percentile to use (if aggregation="percentile", e.g., 5% takes penalties of top 5% of hits)
        unwanted_focus: Prioritize scoring of unwanted motifs

    Returns:
        float or list[float]: penalty scores (0=best, 1=worst).
    """

    # Parse motif names
    motif_names = []
    with open(motifs_path) as f:
        for line in f:
            if line.startswith("MOTIF"):
                motif_names.append(line.split()[1])

    # Normalize "all"/"none"
    if (
        isinstance(wanted, list)
        and len(wanted) == 1
        and wanted[0].lower() in ("all", "none")
    ):
        wanted = wanted[0].lower()
    if (
        isinstance(not_wanted, list)
        and len(not_wanted) == 1
        and not_wanted[0].lower() in ("all", "none")
    ):
        not_wanted = not_wanted[0].lower()

    # Expand wanted/not_wanted
    if wanted == "all":
        wanted = set(motif_names)
    elif wanted in (None, "none"):
        wanted = set()
    else:
        wanted = set(wanted)

    if not_wanted == "all":
        not_wanted = set(motif_names)
    elif not_wanted in (None, "none"):
        not_wanted = set()
    else:
        not_wanted = set(not_wanted)

    # Exclusive settings to automatically set wanted/unwanted
    if exclusive:
        if wanted and not not_wanted:
            not_wanted = set(motif_names) - wanted
        elif not_wanted and not wanted:
            wanted = set(motif_names) - not_wanted

    is_single = isinstance(sequences, Sequence)
    if is_single:
        sequences = [sequences]

    penalties: List[float] = []

    for seq_obj in sequences:
        seq = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")

        # Run MEME with FIMO
        found: Dict[str, float] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "seq.fa")
            with open(fasta_path, "w") as f:
                f.write(">query\n" + seq + "\n")

            fimo_out = os.path.join(tmpdir, "fimo_out")
            fimo_bin = os.path.join(meme_bin_path, "fimo")
            subprocess.run(
                [fimo_bin, "--oc", fimo_out, motifs_path, fasta_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            fimo_tsv = os.path.join(fimo_out, "fimo.tsv")
            if os.path.exists(fimo_tsv):
                with open(fimo_tsv) as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split("\t")
                        if not parts or parts[0] == "motif_id":
                            continue
                        if len(parts) < 8:
                            continue
                        motif_id = parts[0]
                        e_val = float(parts[7])
                        if motif_id not in found or e_val < found[motif_id]:
                            found[motif_id] = e_val

        # Scoring
        details = {}
        if not wanted and not not_wanted:
            if not found:
                penalty = 0.0
            else:
                # Calculate penalty based on strongest unwanted match
                strongest_eval = min(found.values())
                if strongest_eval > 0:
                    log_penalty = -np.log10(strongest_eval) / 10.0
                    penalty = min(1.0, scale * log_penalty)
                else:
                    penalty = 1.0

            seq_obj._metadata["motif_constraint"] = {
                "penalty": penalty,
                "wanted": wanted,
                "not_wanted": not_wanted,
                "found": found,
                "details": {},
                "aggregation_info": {
                    "method": "none_wanted",
                    "unwanted_count": 0,
                    "wanted_count": 0,
                },
            }
            penalties.append(penalty)
            continue

        unwanted_penalties = []
        wanted_penalties = []

        # Penalize unwanted motifs (lower e-value = stronger match = higher penalty)
        for motif in not_wanted:
            if motif in found:
                e_val = found[motif]
                if e_val > 0:
                    # Using -log10 transform
                    log_penalty = -np.log10(e_val)
                    penalty_val = min(1.0, scale * (log_penalty / 10.0))
                else:
                    penalty_val = 1.0 * scale
                unwanted_penalties.append(penalty_val)
                details[motif] = {
                    "penalty": penalty_val,
                    "status": "unwanted",
                    "e_value": e_val,
                }
            else:
                details[motif] = {"penalty": 0.0, "status": "unwanted_absent"}

        # Reward wanted motifs (lower e-value = stronger match = lower penalty)
        for motif in wanted:
            if motif not in found:
                wanted_penalties.append(1.0 * scale)
                details[motif] = {"penalty": 1.0 * scale, "status": "wanted_missing"}
            else:
                e_val = found[motif]
                if e_val > 0:
                    penalty_val = min(
                        1.0, scale * (1.0 / (1.0 + np.exp(-10 * (e_val - 0.1))))
                    )
                else:
                    penalty_val = 0.0
                wanted_penalties.append(penalty_val)
                details[motif] = {
                    "penalty": penalty_val,
                    "status": "wanted_found",
                    "e_value": e_val,
                }

        # Aggregate penalties based on specified aggregation methods
        final_penalty = 0.0

        if aggregation == "average":
            # Simple average
            all_penalties = unwanted_penalties + wanted_penalties
            if all_penalties:
                final_penalty = np.mean(all_penalties)

        elif aggregation == "max":
            # Strictest, take worst penalty across all methods
            all_penalties = unwanted_penalties + wanted_penalties
            if all_penalties:
                final_penalty = max(all_penalties)

        elif aggregation == "percentile":
            # Use specified percentile to aggregate top n% penalties
            all_penalties = unwanted_penalties + wanted_penalties
            if all_penalties:
                final_penalty = np.percentile(all_penalties, percentile_value)

        else:
            # Different strategies for wanted vs unwanted
            unwanted_score = 0.0
            wanted_score = 0.0

            if unwanted_penalties:
                # For unwanted: focus on worst offenders
                if len(unwanted_penalties) <= 3:
                    # Few motifs: use maximum
                    unwanted_score = max(unwanted_penalties)
                elif len(unwanted_penalties) <= 10:
                    # Medium number: use 90th percentile
                    unwanted_score = np.percentile(unwanted_penalties, 90)
                else:
                    # Many motifs: Take average of top 5% worst penalties
                    k = max(1, int(len(unwanted_penalties) * 0.05))
                    top_k = sorted(unwanted_penalties, reverse=True)[:k]
                    unwanted_score = np.mean(top_k)

            if wanted_penalties:
                # For wanted: all should be present, so use average
                wanted_score = np.mean(wanted_penalties)

            if unwanted_penalties and wanted_penalties:
                if unwanted_focus:
                    # Give more weight to unwanted motifs when many are scanned
                    total_motifs = len(motif_names)
                    unwanted_ratio = len(not_wanted) / total_motifs
                    # Weight increases with the proportion of unwanted motifs
                    unwanted_weight = 1.0 + unwanted_ratio
                    wanted_weight = 1.0
                else:
                    unwanted_weight = 1.0
                    wanted_weight = 1.0
                final_penalty = (
                    unwanted_weight * unwanted_score + wanted_weight * wanted_score
                ) / (unwanted_weight + wanted_weight)
            elif unwanted_penalties:
                final_penalty = unwanted_score
            else:
                final_penalty = wanted_score

        penalty = min(1.0, final_penalty)

        # Store results in metadata
        seq_obj._metadata["motif_constraint"] = {
            "penalty": penalty,
            "wanted": wanted,
            "not_wanted": not_wanted,
            "found": found,
            "details": details,
            "aggregation_info": {
                "method": aggregation,
                "unwanted_count": len(unwanted_penalties),
                "wanted_count": len(wanted_penalties),
                "unwanted_matches": len([p for p in unwanted_penalties if p > 0]),
                "wanted_matches": len([p for p in wanted_penalties if p < 1.0 * scale]),
            },
        }
        penalties.append(penalty)

    return penalties[0] if is_single else penalties


def promoter_strength_constraint(
    sequences: Union["Sequence", List["Sequence"]],
    config: Optional[Dict[str, Any]] = None,
) -> Union[float, List[float]]:
    """
    Run Barrick Lab Promoter Calculator and return a [0,1] penalty score.

    Also caches the full promoter_calculator output in each Sequence's metadata
    under the "promoter_strength" key.

    Penalty scheme:
        For tx_rate penalty:
        - Tx_rate < 1500: weak     (penalty = 1.0)
        - 1500–5000: moderate (linear from 1.0 --> 0.5)
        - > 5000: strong   (linear from 0.5 --> 0.0, capped at 10,000)
        For dG penalty:
        - dG > -1.5: weak (penalty = 1.0)
        - dG between -1.5 and -3.0: moderate (linear scale from 1 to 0.5)
        - dG < -3.0: strong (linear from 0.5 --> 0.0, capped at -5.0)

    Args:
        sequences: Sequence or list[Sequence] (DNA only).
        config: optional params for promoter_calculator:
            - add_context (bool, default False, adds additional nucleotides to end of sequence to ensure
            sequence meets promoter calcualtor length minimums)
            - context_length (int, default 10, amount of additional nucleotides to add)
            - threads (int, default 1)
            - verbosity (int, default 0)
            - circular (bool, default False, circularizes sequence if needed)
            - batch_size (int, default None, process all at once)
            - scoring_type (string, default = dG)

    Returns:
        float or list[float]: penalty scores.
    """
    if config is None:
        config = {}

    is_single = isinstance(sequences, Sequence)
    if is_single:
        sequences = [sequences]

    # Extract config parameters
    add_context = bool(config.get("add_context", False))
    context_length = int(config.get("context_length", 10))
    threads = int(config.get("threads", 8))
    verbosity = int(config.get("verbosity", 0))
    circular = bool(config.get("circular", False))
    batch_size = config.get("batch_size", None)  # If None, process all at once
    scoring_type = config.get("scoring_type", "dG")

    # Clean all sequences
    processed_sequences = []
    for seq_obj in sequences:
        s = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        if add_context:
            s = ("A" * context_length) + s + ("A" * context_length)
        processed_sequences.append(s)

    penalties: List[float] = []

    # Process in batches if batch_size is specified, otherwise all at once
    if batch_size and batch_size < len(processed_sequences):
        # Process in batches
        all_results = []
        for i in range(0, len(processed_sequences), batch_size):
            batch = processed_sequences[i : i + batch_size]
            batch_results = []
            for seq in batch:
                res = (
                    promoter_calculator(
                        seq, threads=threads, verbosity=verbosity, circular=circular
                    )
                    or []
                )
                batch_results.append(res)
            all_results.extend(batch_results)
    else:
        try:
            # Attempt batch processing
            all_results = promoter_calculator(
                processed_sequences,
                threads=threads,
                verbosity=verbosity,
                circular=circular,
            )
            if not isinstance(all_results[0], list):
                raise NotImplementedError("Batch processing format not recognized")
        except (TypeError, AttributeError, NotImplementedError):
            all_results = []
            for seq in processed_sequences:
                res = (
                    promoter_calculator(
                        seq, threads=threads, verbosity=verbosity, circular=circular
                    )
                    or []
                )
                all_results.append(res)

    # Process results for each sequence
    for seq_obj, res in zip(sequences, all_results):
        # Keep only + strand
        res = [r for r in res if getattr(r, "strand", "+") == "+"]

        if not res:
            penalty = 1.0
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "reason": "no_promoter_found",
                "raw_output": [],
            }
            penalties.append(penalty)
            continue

        if scoring_type == "tx_rate":
            # Extract tx_rate
            tx_rate = max(float(r.Tx_rate) for r in res if hasattr(r, "Tx_rate"))

            # Penalty mapping
            if tx_rate < 1500.0:
                penalty = 1.0
            elif tx_rate <= 5000.0:
                penalty = 1.0 - 0.5 * ((tx_rate - 1000.0) / (5000.0 - 1000.0))
            else:
                penalty = 0.5 - min((tx_rate - 5000.0) / 5000.0 * 0.5, 0.5)
                penalty = max(0.0, penalty)

            # Store metadata
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "tx_rate": tx_rate,
                "raw_output": [r.__dict__ for r in res],
            }
            penalties.append(penalty)
        else:
            dG = min(float(r.dG_total) for r in res if hasattr(r, "dG_total"))
            if dG >= 0:
                penalty = 1.0
            elif dG > -1.5:
                penalty = 1.0
            elif dG >= -3.0:
                penalty = 1.0 - 0.5 * ((dG + 1.5) / -1.5)
            else:
                normalized = (dG - (-3.0)) / (-5.0 - (-3.0))
                penalty = 0.5 * (1 - normalized**2)
                penalty = max(0.0, min(0.5, penalty))

            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "dG_rate": dG,
                "raw_output": [r.__dict__ for r in res],
            }
    return penalties[0] if is_single else penalties
