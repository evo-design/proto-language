"""Consensus operator-readout specificity constraint (C = z_NA-MPNN + z_DeepPBS).

This constraint combines the NA-MPNN and DeepPBS dyad-aware specificity margins
into a single consensus discrimination score. For a candidate protein-DNA complex
both readout models are run, each produces a sliding-window log-prob specificity
margin (target vs. best off-target, see ``dna_motif_specificity_constraint``), and
each margin is z-normalized against a precomputed reference distribution of margins
over natural / scrambled HTH operators. The consensus statistic is

    C = z_NA-MPNN + z_DeepPBS

where a higher ``C`` means the design discriminates its target more strongly than
typical natural operators. The score squashes ``C`` to ``[0, 1]`` (lower is better)
via ``0.5 - 0.5 * tanh(C / 2)`` so a large positive consensus drives the score
toward 0. DeepPBS failures degrade gracefully to ``z_DeepPBS = 0`` rather than
crashing.

Examples:
    >>> # cfg = ConsensusSpecificityConfig(
    >>> #     target_motif="TACGATATATCGTG", off_target_motifs=["GTATTATATAAGAC"],
    >>> #     dna_indices=list(range(10, 24)), structure_tool="alphafold3")
    >>> # consensus_operator_specificity_constraint([(protein_seq, dna_seq)], cfg)
"""

import json
import logging
from collections.abc import Sequence as TypingSequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from proto_tools import (
    DeepPBSSpecificityConfig,
    DeepPBSSpecificityInput,
    NAMPNNSpecificityConfig,
    NAMPNNSpecificityInput,
    run_deeppbs_specificity,
    run_na_mpnn_specificity,
)
from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.dna_binding_structure_helper import resolve_structure_paths
from proto_language.constraint.protein_structure.dna_motif_specificity_constraint import (
    DNA_TO_INT,
    _to_ppm_matrix,
    sliding_logprob_advantage,
)
from proto_language.constraint.protein_structure.structure_constraint_config import StructureBasedConstraintConfig
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import ConfigField

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_PATH = str(
    Path(__file__).resolve().parents[3] / "examples" / "data" / "consensus_specificity_reference.json"
)


def z_score(value: float, mean: float, std: float) -> float:
    """Z-normalize a value against a reference mean/std (std<=0 -> 0.0).

    Args:
        value (float): Observed specificity margin.
        mean (float): Reference distribution mean.
        std (float): Reference distribution standard deviation.

    Returns:
        float: ``(value - mean) / std``, or ``0.0`` when ``std`` is non-positive.
    """
    if std <= 0.0 or not np.isfinite(std):
        return 0.0
    return float((value - mean) / std)


def consensus_score(consensus: float) -> float:
    """Map a consensus statistic ``C`` to a [0, 1] score (lower is better).

    Squashes ``C`` via ``0.5 - 0.5 * tanh(C / 2)`` so larger ``C`` (stronger
    discrimination) drives the score toward 0 (best) and smaller/negative ``C``
    toward 1 (worst).

    Args:
        consensus (float): Consensus statistic ``z_na_mpnn + z_deeppbs``.

    Returns:
        float: Score in [0, 1] where lower is better.
    """
    return float(np.clip(0.5 - 0.5 * np.tanh(consensus / 2.0), 0.0, 1.0))


def load_reference_stats(reference_path: str) -> dict[str, dict[str, float]]:
    """Load the NA-MPNN / DeepPBS reference margin mean/std from a JSON file.

    Args:
        reference_path (str): Path to a JSON file with ``na_mpnn`` and ``deeppbs``
            objects, each carrying ``mean`` and ``std`` floats.

    Returns:
        dict[str, dict[str, float]]: Mapping ``{"na_mpnn": {"mean", "std"}, "deeppbs": {...}}``.

    Raises:
        FileNotFoundError: If ``reference_path`` does not exist.
        ValueError: If a required key is missing.
    """
    path = Path(reference_path)
    if not path.exists():
        raise FileNotFoundError(f"Consensus reference distribution not found: {reference_path}")
    with path.open() as handle:
        raw = json.load(handle)
    stats: dict[str, dict[str, float]] = {}
    for key in ("na_mpnn", "deeppbs"):
        if key not in raw or "mean" not in raw[key] or "std" not in raw[key]:
            raise ValueError(f"Reference JSON missing '{key}' mean/std: {reference_path}")
        stats[key] = {"mean": float(raw[key]["mean"]), "std": float(raw[key]["std"])}
    return stats


class ConsensusSpecificityConfig(StructureBasedConstraintConfig):
    """Config for the consensus-operator-specificity constraint.

    Attributes:
        target_motif (str): Target DNA motif in the A/C/G/T alphabet.
        off_target_motifs (list[str]): Off-target motifs (same length as the target).
        dna_indices (list[int]): 0-based DNA positions mapped to motif positions.
        desired_margin (float): Sliding-logprob margin scale (kept for parity / metadata).
        dna_chain_label (int): Canonical DNA chain label for motif indexing (fwd strand typically 0).
        reference_path (str): Path to the JSON reference margin mean/std distribution.
        na_mpnn_config (NAMPNNSpecificityConfig): Tool config for na-mpnn-specificity predictions.
        deeppbs_config (DeepPBSSpecificityConfig): Tool config for deeppbs-specificity predictions.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Structure-prediction tool; defaults to a DNA-capable predictor.
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    target_motif: str = ConfigField(
        title="Target Motif",
        description="Target DNA motif in A/C/G/T alphabet.",
    )
    off_target_motifs: list[str] = ConfigField(
        title="Off-target Motifs",
        description="Off-target motifs used for the specificity margin.",
    )
    dna_indices: list[int] = ConfigField(
        title="DNA Indices",
        description="0-based DNA indices mapped to motif positions.",
    )
    desired_margin: float = ConfigField(
        title="Desired Margin",
        default=1.0,
        gt=0.0,
        description="Sliding-logprob margin scale (kept for parity and metadata).",
    )
    dna_chain_label: int = ConfigField(
        title="DNA Chain Label",
        default=0,
        ge=0,
        description="Canonical DNA chain label for motif indexing when multiple DNA chains exist (fwd strand 0).",
    )
    reference_path: str = ConfigField(
        title="Reference Path",
        default=DEFAULT_REFERENCE_PATH,
        description="Path to JSON reference margin mean/std for NA-MPNN and DeepPBS.",
    )
    na_mpnn_config: NAMPNNSpecificityConfig = ConfigField(
        title="NA-MPNN Config",
        default_factory=NAMPNNSpecificityConfig,
        description="Tool config for na-mpnn-specificity predictions.",
    )
    deeppbs_config: DeepPBSSpecificityConfig = ConfigField(
        title="DeepPBS Config",
        default_factory=DeepPBSSpecificityConfig,
        description="Tool config for deeppbs-specificity predictions.",
    )
    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )

    @field_validator("target_motif", mode="before")
    @classmethod
    def normalize_target_motif(cls, value: str) -> str:
        """Normalize and validate the target motif string."""
        if not isinstance(value, str):
            raise TypeError("target_motif must be a string")
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("target_motif cannot be empty")
        if any(base not in DNA_TO_INT for base in normalized):
            raise ValueError("target_motif must only contain A, C, G, T")
        return normalized

    @field_validator("off_target_motifs", mode="before")
    @classmethod
    def normalize_off_targets(cls, value: TypingSequence[str]) -> list[str]:
        """Normalize and validate the off-target motif strings."""
        if not isinstance(value, (list, tuple)):
            raise TypeError("off_target_motifs must be a list of motif strings")
        normalized = [str(item).strip().upper() for item in value]
        if not normalized:
            raise ValueError("off_target_motifs must contain at least one motif")
        for motif in normalized:
            if not motif:
                raise ValueError("off_target_motifs cannot contain empty motifs")
            if any(base not in DNA_TO_INT for base in motif):
                raise ValueError("off_target_motifs entries must only contain A, C, G, T")
        return normalized

    @field_validator("dna_indices")
    @classmethod
    def validate_dna_indices(cls, value: list[int]) -> list[int]:
        """Validate explicit DNA indices."""
        if not value:
            raise ValueError("dna_indices cannot be empty")
        if any(idx < 0 for idx in value):
            raise ValueError("dna_indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("dna_indices must be unique")
        return value

    @model_validator(mode="after")
    def validate_lengths(self) -> "ConsensusSpecificityConfig":
        """Enforce motif-length consistency with the explicit DNA indices."""
        motif_len = len(self.target_motif)
        if len(self.dna_indices) != motif_len:
            raise ValueError("target_motif length must match dna_indices length")
        for motif in self.off_target_motifs:
            if len(motif) != motif_len:
                raise ValueError("All off_target_motifs must match target_motif length")
        return self


def _select_chain_ppm(result: Any, dna_chain_label: int, motif_len: int) -> np.ndarray:
    """Row-normalize a result's PPM and select the canonical DNA chain rows.

    Falls back to all rows when chain labels are unavailable or the selected chain
    is shorter than the motif.
    """
    ppm = _to_ppm_matrix(result.predicted_ppm)
    chain_labels = np.asarray(getattr(result, "chain_labels", []), dtype=np.int64)
    if chain_labels.shape[0] == ppm.shape[0]:
        chain_mask = chain_labels == int(dna_chain_label)
        if int(np.sum(chain_mask)) >= motif_len:
            ppm = ppm[chain_mask]
    return ppm


def _margin_for_result(result: Any, config: ConsensusSpecificityConfig) -> dict[str, float]:
    """Sliding-logprob specificity advantage (margin) for one tool result."""
    ppm = _select_chain_ppm(result, config.dna_chain_label, len(config.target_motif))
    return sliding_logprob_advantage(ppm, config.target_motif, config.off_target_motifs)


@constraint(
    key="consensus-operator-specificity",
    label="Consensus Operator Specificity",
    config=ConsensusSpecificityConfig,
    description="Consensus DNA-readout specificity (C = z_NA-MPNN + z_DeepPBS)",
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "na-mpnn-specificity", "deeppbs-specificity"],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def consensus_operator_specificity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: ConsensusSpecificityConfig,
) -> list[ConstraintOutput]:
    """Consensus operator-readout specificity score (C = z_NA-MPNN + z_DeepPBS).

    Predicts (or reuses) a protein-DNA complex for each candidate, runs BOTH
    NA-MPNN and DeepPBS readout models, computes each model's dyad-aware
    sliding-window specificity margin (target vs. best off-target), z-normalizes
    each margin against the shipped reference distribution, and combines them as
    ``C = z_NA-MPNN + z_DeepPBS``. DeepPBS failures degrade to ``z_DeepPBS = 0``.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-candidate tuples (protein +
            DNA chains) folded into a complex.
        config (ConsensusSpecificityConfig): Constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per candidate; lower score is better.
            Score is ``0.5 - 0.5 * tanh(C / 2)`` so a large positive consensus
            (strong discrimination) maps toward 0.0. Metadata carries each model's
            margin, z-score, and the consensus ``C``.
    """
    if not input_sequences:
        return []

    reference = load_reference_stats(config.reference_path)
    na_ref = reference["na_mpnn"]
    pbs_ref = reference["deeppbs"]

    pdb_paths = resolve_structure_paths(
        input_sequences,
        structure_tool=config.structure_tool,
        tool_config=config.tool_config,
    )

    na_output = run_na_mpnn_specificity(NAMPNNSpecificityInput(pdb_paths=pdb_paths), config.na_mpnn_config)
    na_results = list(na_output.results)
    if len(na_results) != len(pdb_paths):
        raise RuntimeError("NA-MPNN specificity returned a mismatched number of predictions.")

    deeppbs_results: list[Any] | None = None
    deeppbs_error: str | None = None
    try:
        pbs_output = run_deeppbs_specificity(DeepPBSSpecificityInput(pdb_paths=pdb_paths), config.deeppbs_config)
        candidate_results = list(pbs_output.results)
        if len(candidate_results) != len(pdb_paths):
            raise RuntimeError("DeepPBS specificity returned a mismatched number of predictions.")
        deeppbs_results = candidate_results
    except Exception as exc:  # DeepPBS failures degrade to z_deeppbs=0, not crash
        deeppbs_error = str(exc)
        logger.warning("consensus-operator-specificity: DeepPBS failed; using z_deeppbs=0 for this candidate: %s", exc)

    outputs: list[ConstraintOutput] = []
    for idx, pdb_path in enumerate(pdb_paths):
        na_margin = _margin_for_result(na_results[idx], config)
        na_advantage = na_margin["advantage"]
        z_na = z_score(na_advantage, na_ref["mean"], na_ref["std"]) if np.isfinite(na_advantage) else 0.0

        pbs_advantage = float("nan")
        z_pbs = 0.0
        if deeppbs_results is not None:
            pbs_margin = _margin_for_result(deeppbs_results[idx], config)
            pbs_advantage = pbs_margin["advantage"]
            z_pbs = z_score(pbs_advantage, pbs_ref["mean"], pbs_ref["std"]) if np.isfinite(pbs_advantage) else 0.0

        consensus = z_na + z_pbs
        score = consensus_score(consensus)

        metadata: dict[str, object] = {
            "pdb_path": pdb_path,
            "source_method": "consensus",
            "na_mpnn_margin": na_advantage,
            "na_mpnn_z": z_na,
            "deeppbs_margin": pbs_advantage,
            "deeppbs_z": z_pbs,
            "consensus": consensus,
            "desired_margin": config.desired_margin,
            "target_motif": config.target_motif,
            "off_target_motifs": list(config.off_target_motifs),
            "dna_chain_label": int(config.dna_chain_label),
            "reference_path": config.reference_path,
        }
        if deeppbs_error is not None:
            metadata["deeppbs_error"] = deeppbs_error
        outputs.append(ConstraintOutput(score=score, metadata=metadata))

    return outputs
