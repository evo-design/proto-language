"""
Sequence motif constraint for scoring DNA sequences against motifs using MEME.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List, Literal, Optional, Union

import numpy as np
from pydantic import Field

from ...base import Sequence
from proto_language.base_config import BaseConfig
from ..registry import ConstraintRegistry


class SeqMotifConfig(BaseConfig):
    """Configuration for sequence motif constraint."""
    motifs_path: str = Field(description="Path to MEME format motif file (.meme) containing PWMs")
    meme_bin_path: str = Field(description="Path to directory containing MEME Suite binaries (must include fimo)")
    wanted: Optional[Union[str, List[str]]] = Field(default=None, description="Motifs that should be present: 'all' (all motifs), 'none' (no requirement), or list of motif names")
    not_wanted: Optional[Union[str, List[str]]] = Field(default=None, description="Motifs that should NOT be present: 'all' (reject all), 'none' (allow all), or list of motif names")
    scale: float = Field(default=1.0, description="Scaling factor to adjust penalty magnitude (>1 = stricter, <1 = more lenient)")
    exclusive: bool = Field(default=False, description="If True, automatically sets unwanted motifs as complement of wanted motifs")
    aggregation: Literal["smart", "average", "max", "percentile"] = Field(default="smart", description="How to aggregate penalties: 'smart' (adaptive), 'average', 'max' (strictest), 'percentile'")
    percentile_value: float = Field(default=95.0, ge=0.0, le=100.0, description="Which percentile to use when aggregation='percentile' (0-100)")
    unwanted_focus: bool = Field(default=True, description="When both wanted and unwanted motifs exist, weight unwanted motifs more heavily in final score")


@ConstraintRegistry.register(
    key="seq-motif",
    config=SeqMotifConfig,
    description="Score DNA sequences against motifs using MEME",
    vectorized=True,
    concatenate=True
)
def seq_motif_constraint(
    sequences: List[Sequence],
    config: SeqMotifConfig
) -> List[float]:
    """
    Score DNA sequences against motifs using MEME FIMO.

    Scoring strategy:
    - Unwanted motifs: Strong matches (low e-value) → high penalties
    - Wanted motifs: Strong matches (low e-value) → low penalties (rewards)
    - Missing wanted motifs → high penalties

    Aggregation strategies for handling multiple motifs:
    - "smart": Adaptive (uses max/percentile for unwanted, average for wanted)
    - "average": Simple average of all penalties
    - "max": Takes maximum penalty (strictest)
    - "percentile": Uses specified percentile of penalties

    Args:
        sequences: List of DNA Sequences to evaluate.
        config: Configuration containing motif paths, wanted/unwanted lists, and scoring parameters.

    Returns:
        List of penalty scores (0.0 = best, 1.0 = worst).
    """

    # Parse motif names
    motif_names = []
    with open(config.motifs_path) as f:
        for line in f:
            if line.startswith("MOTIF"):
                motif_names.append(line.split()[1])

    # Normalize "all"/"none"
    wanted = config.wanted
    not_wanted = config.not_wanted
    
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
    if config.exclusive:
        if wanted and not not_wanted:
            not_wanted = set(motif_names) - wanted
        elif not_wanted and not wanted:
            wanted = set(motif_names) - not_wanted

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
            fimo_bin = os.path.join(config.meme_bin_path, "fimo")
            subprocess.run(
                [fimo_bin, "--oc", fimo_out, config.motifs_path, fasta_path],
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
                    penalty = min(1.0, config.scale * log_penalty)
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
                    penalty_val = min(1.0, config.scale * (log_penalty / 10.0))
                else:
                    penalty_val = 1.0 * config.scale
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
                wanted_penalties.append(1.0 * config.scale)
                details[motif] = {"penalty": 1.0 * config.scale, "status": "wanted_missing"}
            else:
                e_val = found[motif]
                if e_val > 0:
                    penalty_val = min(
                        1.0, config.scale * (1.0 / (1.0 + np.exp(-10 * (e_val - 0.1))))
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

        if config.aggregation == "average":
            # Simple average
            all_penalties = unwanted_penalties + wanted_penalties
            if all_penalties:
                final_penalty = np.mean(all_penalties)

        elif config.aggregation == "max":
            # Strictest, take worst penalty across all methods
            all_penalties = unwanted_penalties + wanted_penalties
            if all_penalties:
                final_penalty = max(all_penalties)

        elif config.aggregation == "percentile":
            # Use specified percentile to aggregate top n% penalties
            all_penalties = unwanted_penalties + wanted_penalties
            if all_penalties:
                final_penalty = np.percentile(all_penalties, config.percentile_value)

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
                if config.unwanted_focus:
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
                "method": config.aggregation,
                "unwanted_count": len(unwanted_penalties),
                "wanted_count": len(wanted_penalties),
                "unwanted_matches": len([p for p in unwanted_penalties if p > 0]),
                "wanted_matches": len([p for p in wanted_penalties if p < 1.0 * config.scale]),
            },
        }
        penalties.append(penalty)

    return penalties