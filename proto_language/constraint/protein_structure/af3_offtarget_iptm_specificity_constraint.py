"""AF3 off-target ipTM specificity constraint for protein-DNA binders.

This constraint rewards protein binders that fold confidently against their
target DNA operator while folding poorly against a set of off-target operators.
For each candidate it builds the protein + target-DNA complex and one protein +
off-target-DNA complex per off-target motif, folds each with the configured
structure tool (AlphaFold3), and reads the interface predicted TM-score
(``iptm``). The score measures the specificity margin
(``target_iptm - best_off_target_iptm``) against a desired margin.

Off-target DNA sequences are formed by substituting the motif region
(``target_motif`` -> ``off_target_motif``) inside ``target_dna_sequence`` at the
explicit ``dna_indices``, with optional reverse-complement strand inclusion for
double-stranded folding.

Constraints:
- af3-offtarget-iptm-specificity: Margin of target vs. best off-target ipTM.

Examples:
    >>> # Build the off-target operator for a single motif swap.
    >>> _replace_motif_at_indices("AAAA", "GC", [1, 2])  # 'AGCA'
    'AGCA'
"""

from collections.abc import Sequence as TypingSequence
from logging import getLogger
from typing import Literal

import numpy as np
from proto_tools import Complex, Structure, predict_structures
from pydantic import field_validator, model_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.base import ConfigField

logger = getLogger(__name__)


# ============================================================================
# Constants and pure helpers
# ============================================================================

DNA_BASES: frozenset[str] = frozenset("ACGT")
DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(sequence: str) -> str:
    """Return reverse-complement DNA string in A/C/G/T alphabet."""
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def _clean_dna(sequence: str) -> str:
    """Normalize and validate a DNA string into upper-case A/C/G/T."""
    normalized = str(sequence).strip().upper()
    if not normalized:
        raise ValueError("DNA sequence cannot be empty")
    if any(base not in DNA_BASES for base in normalized):
        raise ValueError("DNA sequence must contain only A, C, G, T")
    return normalized


def _replace_motif_at_indices(scaffold: str, motif: str, dna_indices: list[int]) -> str:
    """Create a new scaffold by replacing motif positions at explicit indices."""
    chars = list(scaffold)
    for idx, base in zip(dna_indices, motif, strict=False):
        chars[idx] = base
    return "".join(chars)


def _margin_score(target_iptm: float, best_off_target_iptm: float, desired_margin: float) -> float:
    """Convert a target/off-target ipTM advantage into a [0, 1] score (lower is better).

    The advantage is ``target_iptm - best_off_target_iptm``. A score of 0.0 means
    the advantage meets or exceeds ``desired_margin``; 1.0 means the advantage is
    zero or negative (no specificity).

    Args:
        target_iptm (float): ipTM of the protein + target-DNA complex.
        best_off_target_iptm (float): Highest ipTM across off-target complexes.
        desired_margin (float): Desired ``target_iptm - best_off_target_iptm`` margin.

    Returns:
        float: Specificity score in ``[0.0, 1.0]`` (0.0 best, 1.0 worst).
    """
    advantage = target_iptm - best_off_target_iptm
    return float(np.clip((desired_margin - advantage) / desired_margin, 0.0, 1.0))


# ============================================================================
# Config
# ============================================================================


class AF3OffTargetIPTMSpecificityConfig(StructureBasedConstraintConfig):
    """Configuration for the AF3 off-target ipTM specificity constraint.

    Attributes:
        target_dna_sequence (str): Full forward DNA scaffold (A/C/G/T) used for
            target folding; off-target scaffolds substitute the motif region into it.
        target_motif (str): Target motif (A/C/G/T) placed at ``dna_indices`` in the scaffold.
        off_target_motifs (list[str]): Off-target motifs (each same length as
            ``target_motif``) substituted at ``dna_indices`` for specificity folding.
        dna_indices (list[int]): 0-based, unique indices of motif positions in
            ``target_dna_sequence``; length must equal ``target_motif`` length.
        desired_margin (float): Desired ``target_iptm - best_off_target_iptm`` margin.
        include_reverse_complement (bool): Include the reverse-complement DNA
            strand so the operator folds as double-stranded DNA.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): DNA-capable structure-prediction tool for the protein-DNA complex (default alphafold3).
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    target_dna_sequence: str = ConfigField(
        title="Target DNA Sequence",
        description="Full forward DNA scaffold (A/C/G/T) used for target folding.",
    )
    target_motif: str = ConfigField(
        title="Target Motif",
        description="Target motif in A/C/G/T placed at dna_indices in the scaffold.",
    )
    off_target_motifs: list[str] = ConfigField(
        title="Off-target Motifs",
        description="Off-target motifs substituted at dna_indices for AF3 specificity.",
    )
    dna_indices: list[int] = ConfigField(
        title="DNA Indices",
        description="0-based indices of motif positions in target_dna_sequence.",
    )
    desired_margin: float = ConfigField(
        title="Desired ipTM Margin",
        default=0.05,
        gt=0.0,
        description="Desired margin target_iptm - best_off_target_iptm.",
    )
    include_reverse_complement: bool = ConfigField(
        title="Use Reverse Complement",
        default=True,
        description="Include reverse-complement DNA strand for dsDNA AF3 folding.",
    )
    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )

    @field_validator("target_dna_sequence", "target_motif", mode="before")
    @classmethod
    def _validate_dna_string(cls, value: str) -> str:
        return _clean_dna(value)

    @field_validator("off_target_motifs", mode="before")
    @classmethod
    def _validate_off_targets(cls, value: TypingSequence[str]) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("off_target_motifs must be a list of DNA motifs")
        motifs = [_clean_dna(item) for item in value]
        if not motifs:
            raise ValueError("off_target_motifs must contain at least one motif")
        return motifs

    @field_validator("dna_indices")
    @classmethod
    def _validate_indices(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("dna_indices cannot be empty")
        if any(idx < 0 for idx in value):
            raise ValueError("dna_indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("dna_indices must be unique")
        return value

    @model_validator(mode="after")
    def _validate_lengths(self) -> "AF3OffTargetIPTMSpecificityConfig":
        motif_len = len(self.target_motif)
        if len(self.dna_indices) != motif_len:
            raise ValueError("target_motif length must match dna_indices length")
        if max(self.dna_indices) >= len(self.target_dna_sequence):
            raise ValueError("dna_indices exceed target_dna_sequence bounds")
        for motif in self.off_target_motifs:
            if len(motif) != motif_len:
                raise ValueError("off_target_motifs must match target_motif length")
        return self


# ============================================================================
# Constraint
# ============================================================================


def _build_complex_chains(proteins: list[Sequence], dna_forward: str, include_reverse_complement: bool) -> Complex:
    """Build a Complex of the protein chain(s) plus the (optionally double-stranded) DNA."""
    chains: list[dict[str, str]] = [{"sequence": seq.sequence, "entity_type": seq.sequence_type} for seq in proteins]
    chains.append({"sequence": dna_forward, "entity_type": "dna"})
    if include_reverse_complement:
        chains.append({"sequence": _reverse_complement(dna_forward), "entity_type": "dna"})
    return Complex(chains=chains)


@constraint(
    key="af3-offtarget-iptm-specificity",
    label="AF3 Off-target ipTM Specificity",
    config=AF3OffTargetIPTMSpecificityConfig,
    description="Prefer high target ipTM and lower off-target ipTM using AF3",
    uses_gpu=True,
    tools_called=[
        "alphafold3-prediction",
        "boltz2-prediction",
        "protenix-prediction",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def af3_offtarget_iptm_specificity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: AF3OffTargetIPTMSpecificityConfig,
) -> list[ConstraintOutput]:
    """Score AF3-based off-target specificity margin using ipTM.

    For each candidate tuple, builds the protein + target-DNA complex and one
    protein + off-target-DNA complex per off-target motif (motif substituted at
    ``dna_indices``), folds them all in a single ``predict_structures`` batch,
    and reads ``iptm`` from each. The score is the specificity margin
    ``target_iptm - best_off_target_iptm`` against ``desired_margin``: 0.0 when
    the margin is met or exceeded, 1.0 when there is no advantage.

    **Supported tools**: AlphaFold3 (any ipTM-producing structure tool configured
    via ``structure_tool``).

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-candidate tuples; each
            must contain at least one protein chain (DNA scaffolds come from config).
        config (AF3OffTargetIPTMSpecificityConfig): Constraint configuration.

    Returns:
        list[ConstraintOutput]: Per-candidate specificity score in ``[0, 1]``
            (lower is better) with ``target_iptm`` / ``best_off_target_iptm`` /
            ``iptm_advantage`` / ``desired_margin`` metadata (plus an ``iptm_error``
            flag when the predictor omits ``iptm``). The predicted target complex
            attaches to slot 0.

    Raises:
        ValueError: If a candidate tuple has no protein chain.
        RuntimeError: If the structure prediction count does not match the request.

    Examples:
        Off-target specificity for a designed DNA-binding protein with AF3:

        >>> from proto_language.core import Segment
        >>> binder = Segment(length=120, sequence_type="protein")
        >>> specificity = Constraint(
        ...     inputs=[binder],
        ...     function=af3_offtarget_iptm_specificity_constraint,
        ...     function_config={
        ...         "structure_tool": "alphafold3",
        ...         "target_dna_sequence": "ACGTACGTACGT",
        ...         "target_motif": "GTAC",
        ...         "off_target_motifs": ["AAAA", "TTTT"],
        ...         "dna_indices": [2, 3, 4, 5],
        ...     },
        ... )
    """
    if not input_sequences:
        return []

    complexes: list[Complex] = []
    mapping: list[tuple[int, str]] = []

    target_forward = config.target_dna_sequence

    for cand_idx, candidate in enumerate(input_sequences):
        proteins = [seq for seq in candidate if seq.sequence_type == "protein"]
        if not proteins:
            raise ValueError("AF3 off-target specificity requires at least one protein input.")

        complexes.append(_build_complex_chains(proteins, target_forward, config.include_reverse_complement))
        mapping.append((cand_idx, "target"))

        for off_target_motif in config.off_target_motifs:
            off_forward = _replace_motif_at_indices(target_forward, off_target_motif, config.dna_indices)
            complexes.append(_build_complex_chains(proteins, off_forward, config.include_reverse_complement))
            mapping.append((cand_idx, off_target_motif))

    output = predict_structures(complexes, config.structure_tool, config.tool_config, msas=None)
    if len(output.structures) != len(mapping):
        raise RuntimeError("AF3 off-target specificity: prediction count mismatch.")

    target_iptm: dict[int, float] = {}
    target_structure: dict[int, Structure] = {}
    off_iptm: dict[int, list[float]] = {idx: [] for idx in range(len(input_sequences))}
    iptm_missing: dict[int, bool] = dict.fromkeys(range(len(input_sequences)), False)

    for structure, (cand_idx, tag) in zip(output.structures, mapping, strict=True):
        metric = structure.metrics.get("iptm")
        if metric is None:
            logger.warning("Metric 'iptm' missing from %s output, treating as 0.0.", config.structure_tool)
            iptm_missing[cand_idx] = True
        value = float(metric) if metric is not None else 0.0
        if tag == "target":
            target_iptm[cand_idx] = value
            target_structure[cand_idx] = structure
        else:
            off_iptm[cand_idx].append(value)

    results: list[ConstraintOutput] = []
    for cand_idx, candidate in enumerate(input_sequences):
        tgt = float(target_iptm.get(cand_idx, 0.0))
        best_off = float(max(off_iptm.get(cand_idx) or [0.0]))
        advantage = tgt - best_off
        # A missing ipTM (coerced to 0.0 above) would deflate best_off and falsely
        # inflate specificity — a predictor failure must not look like a better
        # design. Soft-fail the candidate to the worst score instead.
        score = MAX_ENERGY if iptm_missing[cand_idx] else _margin_score(tgt, best_off, config.desired_margin)

        metadata: dict[str, object] = {
            "target_iptm": tgt,
            "best_off_target_iptm": best_off,
            "iptm_advantage": advantage,
            "desired_margin": float(config.desired_margin),
            "target_motif": config.target_motif,
            "off_target_motifs": list(config.off_target_motifs),
            "dna_indices": list(config.dna_indices),
            "structure_tool": config.structure_tool,
        }
        if iptm_missing[cand_idx]:
            metadata["iptm_error"] = f"iptm missing from {config.structure_tool} output (treated as 0.0)"
        structure = target_structure.get(cand_idx)
        n = len(candidate)
        structures = (structure,) + (None,) * (n - 1) if structure is not None else ()
        results.append(ConstraintOutput(score=score, metadata=metadata, structures=structures))

    return results
