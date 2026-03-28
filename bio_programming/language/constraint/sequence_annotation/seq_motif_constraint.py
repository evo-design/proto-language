"""
proto_language/language/constraint/sequence_annotation/seq_motif_constraint.py

Sequence motif constraint for scoring DNA sequences against motifs using MEME.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence


class SeqMotifConfig(BaseConfig):
    """Configuration for sequence motif constraint using MEME.

    This class defines configuration parameters for evaluating DNA sequences against
    known transcription factor binding motifs using MEME Suite's Find Individual
    Motif Occurrences tool. The constraint searches for position weight matrix
    motifs in sequences and can either encourage specific motifs (wanted) or discourage
    them (not_wanted), enabling design of sequences with controlled sites.

    Attributes:
        motifs_path (str): Path to MEME format motif file (.meme) containing position
            weight matrices. Must be a valid file path. MEME format files can
            be obtained from databases like JASPAR, TRANSFAC, or created with MEME
            Suite tools. Example: "/data/motifs/jaspar_vertebrates.meme" or
            "~/databases/tf_motifs.meme".

        meme_bin_path (str): Path to directory containing MEME Suite binaries. Must
            include the ``fimo`` executable. The directory should contain the full
            MEME Suite installation. Example: "/usr/local/meme/bin" or
            "/opt/meme-5.5.0/bin". Install MEME Suite from https://meme-suite.org/

        wanted (list[str] | None): Motifs that should be present in
            sequences. Options:
            - ["all"]: All motifs in the file must be present
            - ["none"] or None: No requirement for specific motifs
            - List of motif names: Specific motifs that should be present,
              e.g., ["SP1", "NF-kB", "lacO"]
            Can also pass a single string which will be converted to a list.
            Strong matches to wanted motifs result in low penalties (rewards).
            Default: None.

        not_wanted (list[str] | None): Motifs that should NOT be
            present in sequences. Options:
            - ["all"]: No motifs should be present (avoid all binding sites)
            - ["none"] or None: Allow any motifs (default)
            - List of motif names: Specific motifs to avoid.
            Can also pass a single string which will be converted to a list.
            Strong matches to unwanted motifs result in high penalties. Default: None.

        scale (float): Scaling factor to adjust penalty magnitude. Values >1 make
            the constraint stricter (larger penalties), values <1 make it more
            lenient (smaller penalties). For example, 2.0 doubles all penalties,
            0.5 halves them. Must be positive. Default: 1.0.

        exclusive (bool): If True, automatically sets unwanted motifs as the complement
            of wanted motifs (and vice versa). For example, if wanted=["SP1", "NF-kB"]
            and exclusive=True, all other motifs in the file become unwanted. Useful
            for enforcing strict motif specificity. Default: True.

        aggregation (Literal['smart', 'average', 'max', 'percentile']): Method for
            aggregating penalties across multiple motifs:
            - "smart": Adaptive strategy that uses max/percentile for unwanted motifs
              and average for wanted motifs (recommended for most cases)
            - "average": Simple average of all penalties (treats all motifs equally)
            - "max": Takes maximum penalty (strictest, most conservative)
            - "percentile": Uses specified percentile of penalties (see percentile_value)
            Default: "smart".

        percentile_value (float): Percentile to use when aggregation="percentile".
            For example, 95.0 combines the 95th percentile (top 5% worst penalties).
            Must be between 0.0 and 100.0. Higher values are more lenient (focus
            on worst offenders), lower values are stricter. Default: 95.0.

        unwanted_focus (bool): When both wanted and unwanted motifs are specified,
            whether to weight unwanted motifs more heavily in the final score. If True,
            unwanted motif penalties are given higher weight, making it harder to
            pass the constraint if unwanted motifs are present. Useful when avoiding
            specific binding sites is critical. Default: False.

    Note:
        Motif names must match exactly with the names in the MEME file (case-sensitive).
        Use the MOTIF lines in the .meme file to identify available motif names.
    """
    # TODO: Make parameters compatible with client. Ideally no union.
    # Required parameters
    motifs_path: str = ConfigField(
        title="Path to MEME format motif file",
        description="Path to MEME format motif file (.meme) containing PWMs.",
    )
    meme_bin_path: str = ConfigField(
        title="Path to MEME Suite binaries",
        description="Path to directory containing MEME Suite binaries (must include fimo).",
    )
    wanted: Optional[List[str]] = ConfigField(
        title="Wanted Motifs",
        default=None,
        description="Motifs that should be present: 'all' (all motifs), 'none' (no requirement), or list of motif names.",
        examples=[["motif1", "motif2"], ["all"], ["none"]],
    )
    not_wanted: Optional[List[str]] = ConfigField(
        title="Unwanted Motifs",
        default=None,
        description="Motifs that should NOT be present: 'all' (reject all), 'none' (allow all), or list of motif names.",
        examples=[["motif1", "motif2"], ["all"], ["none"]],
    )

    @field_validator('wanted', 'not_wanted', mode='before')
    @classmethod
    def convert_motif_list(cls, v):
        """Convert single string to list of strings."""
        if isinstance(v, str):
            return [v]
        return v

    # Advanced parameters
    scale: float = ConfigField(
        title="Scale",
        default=1.0,
        description="Scaling factor to adjust penalty magnitude (>1 = stricter, <1 = more lenient). Example: 1.0",
        advanced=True,
    )
    exclusive: bool = ConfigField(
        title="Exclusive",
        default=True,
        description="If True, automatically sets unwanted motifs as complement of wanted motifs",
        advanced=True,
    )
    aggregation: Literal["smart", "average", "max", "percentile"] = ConfigField(
        title="Aggregation Method",
        default="smart",
        description="How to aggregate penalties: 'smart' (adaptive), 'average', 'max' (strictest), 'percentile'",
        advanced=True,
    )
    percentile_value: float = ConfigField(
        title="Percentile Value",
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Which percentile to use when aggregation='percentile' (0-100)",
        advanced=True,
        depends_on={"field": "aggregation", "value": "percentile"},
    )
    unwanted_focus: bool = ConfigField(
        title="Unwanted Focus",
        default=True,
        description="When both wanted and unwanted motifs exist, weight unwanted motifs more heavily in final score",
        advanced=True,
    )


@constraint(
    key="seq-motif",
    label="Sequence Motif Match",
    config=SeqMotifConfig,
    description="Score DNA sequences against motifs using MEME",
    tools_called=["meme"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
    num_input_sequences_per_tuple=1,
)
def seq_motif_constraint(input_sequences: List[Tuple[Sequence, ...]], config: SeqMotifConfig) -> List[float]:
    """Score DNA sequences against sequence motifs using MEME.

    This constraint function uses MEME Suite's Find Individual Motif
    Occurrences tool to search for sequence  motifs represented as position weight matrices
    in DNA sequences. It evaluates whether sequences contain desired motifs (wanted)
    or unwanted motifs (not_wanted).

    The scoring strategy penalizes sequences based on motif presence:
    - **Unwanted motifs**: Strong matches (low E-values) result in high penalties,
      encouraging sequences without these binding sites
    - **Wanted motifs**: Strong matches result in low penalties (rewards), while
      missing wanted motifs result in high penalties
    - **No motif specification**: Any motif matches are penalized (novelty constraint)

    Args:
        sequences: List of DNA sequences to evaluate. Each sequence
            is independently scanned for motif occurrences using FIMO. Sequences can
            be any length, though motif detection accuracy improves with longer
            sequences (50+ bp recommended).

        config (SeqMotifConfig): Configuration object containing ``motifs_path``
            (MEME motif file), ``meme_bin_path`` (MEME Suite binary directory),
            ``wanted`` (default: None), ``not_wanted`` (default: None), ``aggregation``
            (default: "smart"), and other scoring parameters.
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.

    Returns:
        List[float]: Penalty scores for each sequence, ranging from 0.0 (best,
            all motif criteria satisfied) to 1.0 (worst, severe violations). The
            scoring depends on wanted/unwanted configuration:
            - **Only unwanted specified**: 0.0 if no unwanted motifs found, higher
              scores for stronger unwanted matches
            - **Only wanted specified**: 0.0 if all wanted motifs found with strong
              E-values, 1.0 if wanted motifs missing
            - **Both specified**: Weighted combination based on aggregation method

    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary with the following keys:

        - ``motif_constraint``: Dictionary containing:
          - ``penalty``: Float overall penalty score (0.0-1.0)
          - ``wanted``: Set of wanted motif names
          - ``not_wanted``: Set of unwanted motif names
          - ``found``: Dictionary mapping motif names to their best (lowest) E-values
          - ``details``: Dictionary with per-motif scoring details including:
            - ``penalty``: Individual motif penalty
            - ``status``: "wanted_found", "wanted_missing", "unwanted", or "unwanted_absent"
            - ``e_value``: E-value if motif was found
          - ``aggregation_info``: Dictionary with aggregation statistics:
            - ``method``: Aggregation method used
            - ``unwanted_count``: Number of unwanted motif evaluations
            - ``wanted_count``: Number of wanted motif evaluations
            - ``unwanted_matches``: Number of unwanted motifs found
            - ``wanted_matches``: Number of wanted motifs found

    Examples:
        Requiring specific transcription factor binding sites:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> promoter_seq = Sequence("ATCGGCGGGATCGTAATATAGCATGC", "dna")
        >>> config = SeqMotifConfig(
        ...     motifs_path="/data/jaspar_vertebrates.meme",
        ...     meme_bin_path="/usr/local/meme/bin",
        ...     wanted=["SP1", "lacI"],  # Must have these motifs
        ...     aggregation="average"
        ... )
        >>> scores = seq_motif_constraint([promoter_seq], config)
        >>> print(scores[0])  # e.g., 0.15 (both motifs found with good E-values)
        >>> metadata = promoter_seq._metadata["motif_constraint"]
        >>> print(metadata["found"])  # e.g., {"SP1": 1e-8, "lacI": 3e-6}
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

    for (seq_obj,) in input_sequences:
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
