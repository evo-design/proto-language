"""DNA motif specificity constraints powered by NA-MPNN and DeepPBS.

These constraints score how preferentially a designed protein reads its target
DNA motif over a set of off-target motifs. A protein-DNA complex is predicted
(or reused) for each candidate, a per-position base-preference matrix (PPM) is
predicted by NA-MPNN or DeepPBS, and the target motif's specificity advantage
over the best off-target is scored. The score is 0.0 when the target beats every
off-target by at least ``desired_margin`` and rises to 1.0 as the advantage
vanishes.

Examples:
    >>> # cfg = NAMPNNMotifSpecificityConfig(
    >>> #     target_motif="TACGATATATCGTG", off_target_motifs=["GTATTATATAAGAC"],
    >>> #     dna_indices=list(range(10, 24)), structure_tool="alphafold3")
    >>> # na_mpnn_motif_specificity_constraint([(protein_seq, dna_seq)], cfg)
"""

import logging
from collections.abc import Callable
from collections.abc import Sequence as TypingSequence
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
from proto_language.constraint.protein_structure.structure_constraint_config import StructureBasedConstraintConfig
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.base import ConfigField

logger = logging.getLogger(__name__)

DNA_ORDER = "ACGT"
DNA_TO_INT = {base: idx for idx, base in enumerate(DNA_ORDER)}
MIN_PROB = 1e-8


class _BaseDNAMotifSpecificityConfig(StructureBasedConstraintConfig):
    """Shared configuration for DNA motif specificity constraints.

    Attributes:
        target_motif (str): Target DNA motif in the A/C/G/T alphabet.
        off_target_motifs (list[str]): Off-target motifs (same length as the target)
            used to compute the specificity margin.
        dna_indices (list[int]): 0-based DNA positions mapped to motif positions
            (required and used only when scoring_mode='cross_entropy').
        desired_margin (float): Specificity advantage at which the term is satisfied
            (cross-entropy margin in cross_entropy mode; log-prob scale in sliding_logprob mode).
            The default (1.0) suits the default sliding_logprob mode; for cross_entropy, where
            the advantage is bounded in [0, 1], set a smaller value (e.g. ~0.2).
        dna_chain_label (int): Canonical DNA chain label for motif indexing when multiple
            DNA chains exist (forward strand is typically 0).
        scoring_mode (Literal['cross_entropy', 'sliding_logprob']): Motif scoring mode
            (fixed-index cross-entropy or sliding-window log-prob).
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Structure-prediction tool (must be DNA-capable).
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
        default_factory=list,
        description="0-based DNA indices mapped to motif positions (required only when scoring_mode='cross_entropy').",
    )
    desired_margin: float = ConfigField(
        title="Desired Margin",
        default=1.0,
        gt=0.0,
        description="Specificity advantage satisfying the term; scale follows scoring_mode (CE or log-prob).",
    )
    dna_chain_label: int = ConfigField(
        title="DNA Chain Label",
        default=0,
        ge=0,
        description="Canonical DNA chain label for motif indexing when multiple DNA chains exist (fwd strand 0).",
    )
    scoring_mode: Literal["cross_entropy", "sliding_logprob"] = ConfigField(
        title="Scoring Mode",
        default="sliding_logprob",
        description="Motif scoring: fixed-index cross_entropy or sliding-window log-prob.",
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
        """Validate explicit DNA indices (presence is checked per scoring_mode)."""
        if any(idx < 0 for idx in value):
            raise ValueError("dna_indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("dna_indices must be unique")
        return value

    @model_validator(mode="after")
    def validate_lengths(self) -> "_BaseDNAMotifSpecificityConfig":
        """Enforce motif-length consistency (dna_indices only required for cross_entropy)."""
        motif_len = len(self.target_motif)
        for motif in self.off_target_motifs:
            if len(motif) != motif_len:
                raise ValueError("All off_target_motifs must match target_motif length")
        if self.scoring_mode == "cross_entropy":
            if not self.dna_indices:
                raise ValueError("dna_indices is required when scoring_mode='cross_entropy'")
            if len(self.dna_indices) != motif_len:
                raise ValueError("target_motif length must match dna_indices length")
        return self


class NAMPNNMotifSpecificityConfig(_BaseDNAMotifSpecificityConfig):
    """Config for the na-mpnn-motif-specificity constraint.

    Attributes:
        target_motif (str): Target DNA motif in the A/C/G/T alphabet.
        off_target_motifs (list[str]): Off-target motifs (same length as the target).
        dna_indices (list[int]): 0-based DNA positions mapped to motif positions
            (required and used only when scoring_mode='cross_entropy').
        desired_margin (float): Specificity advantage at which the term is satisfied
            (cross-entropy margin in cross_entropy mode; log-prob scale in sliding_logprob mode).
            The default (1.0) suits the default sliding_logprob mode; for cross_entropy, where
            the advantage is bounded in [0, 1], set a smaller value (e.g. ~0.2).
        dna_chain_label (int): Canonical DNA chain label for motif indexing (fwd strand typically 0).
        scoring_mode (Literal['cross_entropy', 'sliding_logprob']): Fixed-index cross-entropy or sliding log-prob.
        na_mpnn_config (NAMPNNSpecificityConfig): Tool config for na-mpnn-specificity predictions.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Structure-prediction tool (must be DNA-capable).
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    na_mpnn_config: NAMPNNSpecificityConfig = ConfigField(
        title="NA-MPNN Config",
        default_factory=NAMPNNSpecificityConfig,
        description="Tool config for na-mpnn-specificity predictions.",
    )


class DeepPBSMotifSpecificityConfig(_BaseDNAMotifSpecificityConfig):
    """Config for the deeppbs-motif-specificity constraint.

    Attributes:
        target_motif (str): Target DNA motif in the A/C/G/T alphabet.
        off_target_motifs (list[str]): Off-target motifs (same length as the target).
        dna_indices (list[int]): 0-based DNA positions mapped to motif positions
            (required and used only when scoring_mode='cross_entropy').
        desired_margin (float): Specificity advantage at which the term is satisfied
            (cross-entropy margin in cross_entropy mode; log-prob scale in sliding_logprob mode).
            The default (1.0) suits the default sliding_logprob mode; for cross_entropy, where
            the advantage is bounded in [0, 1], set a smaller value (e.g. ~0.2).
        dna_chain_label (int): Canonical DNA chain label for motif indexing (fwd strand typically 0).
        scoring_mode (Literal['cross_entropy', 'sliding_logprob']): Fixed-index cross-entropy or sliding log-prob.
        deeppbs_config (DeepPBSSpecificityConfig): Tool config for deeppbs-specificity predictions.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Structure-prediction tool (must be DNA-capable).
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    deeppbs_config: DeepPBSSpecificityConfig = ConfigField(
        title="DeepPBS Config",
        default_factory=DeepPBSSpecificityConfig,
        description="Tool config for deeppbs-specificity predictions.",
    )


def _normalized_cross_entropy(ppm: np.ndarray, motif: str, dna_indices: list[int]) -> float:
    """Compute normalized cross entropy for a motif on the selected DNA indices."""
    probs = [
        float(np.clip(ppm[idx, DNA_TO_INT[base]], MIN_PROB, 1.0)) for base, idx in zip(motif, dna_indices, strict=True)
    ]
    ce = -float(np.mean(np.log(probs)))
    return ce / float(np.log(4.0))


def _slide_best_logprob(ppm: np.ndarray, motif: str) -> float:
    """Best average per-position log-prob of ``motif`` over all sliding offsets.

    Slides ``motif`` across the rows of ``ppm`` (the selected DNA chain). At each
    offset the score is ``sum(log(max(ppm[s + j, base], MIN_PROB))) / len(motif)``
    and the maximum over offsets is returned (best-window match).

    Args:
        ppm (np.ndarray): Row-normalized base-preference matrix (L x 4, A/C/G/T order).
        motif (str): DNA motif in the A/C/G/T alphabet.

    Returns:
        float: Best average log-prob over offsets, or ``-inf`` if the motif is empty
            or longer than the available rows.
    """
    mlen = len(motif)
    if mlen == 0 or ppm.shape[0] < mlen:
        return float("-inf")
    best = float("-inf")
    for start in range(ppm.shape[0] - mlen + 1):
        total = sum(
            float(np.log(max(float(ppm[start + j, DNA_TO_INT[base]]), MIN_PROB)))
            for j, base in enumerate(motif)
            if base in DNA_TO_INT
        )
        avg = total / mlen
        if avg > best:
            best = avg
    return best


def sliding_logprob_advantage(
    ppm: np.ndarray, target_motif: str, off_target_motifs: TypingSequence[str]
) -> dict[str, float]:
    """Sliding-window specificity advantage of a target over off-targets.

    Computes the best sliding-window average log-prob for the target motif and each
    off-target, then the specificity advantage ``target_best_lp - best_off_target_lp``.
    A positive advantage means the target reads more favorably than the best off-target.

    Args:
        ppm (np.ndarray): Row-normalized base-preference matrix (L x 4, A/C/G/T order).
        target_motif (str): Target DNA motif.
        off_target_motifs (TypingSequence[str]): Off-target DNA motifs.

    Returns:
        dict[str, float]: Keys ``target_lp``, ``best_off_lp``, and ``advantage``.
    """
    target_lp = _slide_best_logprob(ppm, target_motif)
    off_lps = [_slide_best_logprob(ppm, motif) for motif in off_target_motifs]
    best_off_lp = max(off_lps) if off_lps else float("-inf")
    return {"target_lp": target_lp, "best_off_lp": best_off_lp, "advantage": target_lp - best_off_lp}


def sliding_logprob_score(advantage: float, desired_margin: float) -> float:
    """Map a sliding-logprob specificity advantage to a [0, 1] score (0 = best).

    Log-prob margins are not on the [0, 1] cross-entropy scale, so the advantage is
    squashed via ``0.5 - 0.5 * tanh(advantage / desired_margin)``: a positive margin
    (target reads better) drives the score toward 0 (best), a negative margin toward 1.
    ``desired_margin`` is the advantage at which the term is roughly half-satisfied.

    Args:
        advantage (float): ``target_best_lp - best_off_target_lp``.
        desired_margin (float): Advantage at which the term is ~half-satisfied.

    Returns:
        float: Score in [0, 1] where lower is better.
    """
    return float(np.clip(0.5 - 0.5 * np.tanh(advantage / desired_margin), 0.0, 1.0))


def _to_ppm_matrix(predicted_ppm: list[list[float]]) -> np.ndarray:
    """Validate and row-normalize a predicted PPM matrix (L x 4, A/C/G/T order)."""
    matrix = np.asarray(predicted_ppm, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("predicted_ppm must be a 2D matrix")
    if matrix.shape[1] != 4:
        raise ValueError("predicted_ppm must have shape (L, 4) in A/C/G/T order")
    matrix = np.clip(matrix, MIN_PROB, np.inf)
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return np.asarray(matrix / row_sum, dtype=np.float64)


def _evaluate_motif_specificity(
    input_sequences: list[tuple[Sequence, ...]],
    config: _BaseDNAMotifSpecificityConfig,
    run_specificity: Callable[[list[str]], Any],
    source_method: str,
    soft_fail_on_tool_error: bool = False,
) -> list[ConstraintOutput]:
    """Shared implementation for NA-MPNN and DeepPBS motif specificity constraints.

    Structure resolution and the prediction-count check raise (batch-level
    failures). Only the specificity tool call is soft-failable: when
    ``soft_fail_on_tool_error`` is set, a tool crash degrades every candidate to
    the worst score with an explanatory metadata key instead of raising.
    """
    if not input_sequences:
        return []

    pdb_paths = resolve_structure_paths(
        input_sequences,
        structure_tool=config.structure_tool,
        tool_config=config.tool_config,
    )

    try:
        tool_output = run_specificity(pdb_paths)
    except Exception as exc:
        if not soft_fail_on_tool_error:
            raise
        logger.warning("%s motif specificity tool call failed; assigning worst score: %s", source_method, exc)
        return [
            ConstraintOutput(
                score=MAX_ENERGY, metadata={"source_method": source_method, f"{source_method}_error": str(exc)}
            )
            for _ in input_sequences
        ]

    results = list(tool_output.results)
    if len(results) != len(input_sequences):
        raise RuntimeError("Specificity tool returned a mismatched number of predictions.")

    outputs: list[ConstraintOutput] = []
    for result, pdb_path in zip(results, pdb_paths, strict=True):
        ppm = _to_ppm_matrix(result.predicted_ppm)
        chain_labels = np.asarray(getattr(result, "chain_labels", []), dtype=np.int64)

        metadata: dict[str, object] = {
            "pdb_path": pdb_path,
            "source_method": source_method,
            "scoring_mode": config.scoring_mode,
            "desired_margin": config.desired_margin,
            "target_motif": config.target_motif,
            "off_target_motifs": list(config.off_target_motifs),
            "output_npz_path": getattr(result, "output_npz_path", None),
            "dna_chain_label": int(config.dna_chain_label),
        }

        if config.scoring_mode == "sliding_logprob":
            # Sliding-window scoring over the selected DNA chain rows
            # (falling back to all rows when chain labels are unavailable).
            chain_ppm = ppm
            if chain_labels.shape[0] == ppm.shape[0]:
                chain_mask = chain_labels == int(config.dna_chain_label)
                if int(np.sum(chain_mask)) >= len(config.target_motif):
                    chain_ppm = ppm[chain_mask]
            adv = sliding_logprob_advantage(chain_ppm, config.target_motif, config.off_target_motifs)
            advantage = adv["advantage"]
            score = sliding_logprob_score(advantage, config.desired_margin)
            metadata.update(
                {
                    "target_lp": adv["target_lp"],
                    "best_off_lp": adv["best_off_lp"],
                    "advantage": advantage,
                }
            )
            if hasattr(result, "used_fallback"):
                metadata["deeppbs_used_fallback"] = bool(result.used_fallback)
            if getattr(result, "fallback_reason", None):
                metadata["deeppbs_fallback_reason"] = str(result.fallback_reason)
            outputs.append(ConstraintOutput(score=score, metadata=metadata))
            continue

        # cross_entropy indexing: dna_indices always address the selected DNA chain
        # rows when chain labels are present, else the full ppm rows. Defining one
        # row space keeps the same index pointing at the same physical row.
        if chain_labels.shape[0] == ppm.shape[0]:
            chain_mask = chain_labels == int(config.dna_chain_label)
            ppm = ppm[chain_mask]

        if max(config.dna_indices) >= ppm.shape[0]:
            raise ValueError("dna_indices reference positions outside the selected DNA chain rows")

        target_ce = _normalized_cross_entropy(ppm, config.target_motif, config.dna_indices)
        off_target_ces = [
            _normalized_cross_entropy(ppm, motif, config.dna_indices) for motif in config.off_target_motifs
        ]
        best_off_ce = min(off_target_ces)
        advantage = best_off_ce - target_ce
        score = float(np.clip((config.desired_margin - advantage) / config.desired_margin, 0.0, 1.0))

        metadata.update(
            {
                "target_ce": target_ce,
                "best_off_ce": best_off_ce,
                "advantage": advantage,
                "dna_indices": list(config.dna_indices),
            }
        )
        if hasattr(result, "used_fallback"):
            metadata["deeppbs_used_fallback"] = bool(result.used_fallback)
        if getattr(result, "fallback_reason", None):
            metadata["deeppbs_fallback_reason"] = str(result.fallback_reason)

        outputs.append(ConstraintOutput(score=score, metadata=metadata))

    return outputs


@constraint(
    key="na-mpnn-motif-specificity",
    label="NA-MPNN Motif Specificity",
    config=NAMPNNMotifSpecificityConfig,
    description="Score DNA motif specificity using NA-MPNN predictions",
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "na-mpnn-specificity"],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def na_mpnn_motif_specificity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: NAMPNNMotifSpecificityConfig,
) -> list[ConstraintOutput]:
    """Specificity score from NA-MPNN predicted base-preference matrices.

    Predicts (or reuses) a protein-DNA complex for each candidate, predicts a
    per-position base PPM with NA-MPNN, and scores the specificity advantage of the
    target motif over the best off-target relative to ``desired_margin`` (using the
    configured ``scoring_mode``).

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-candidate tuples (protein +
            DNA chains) folded into a complex.
        config (NAMPNNMotifSpecificityConfig): Constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per candidate; score 0.0 (best) when the
            target beats every off-target by ``desired_margin``, 1.0 (worst) otherwise.
            Metadata carries the target/off-target scores and advantage.
    """

    def _run(pdb_paths: list[str]) -> object:
        return run_na_mpnn_specificity(NAMPNNSpecificityInput(pdb_paths=pdb_paths), config.na_mpnn_config)

    return _evaluate_motif_specificity(
        input_sequences=input_sequences,
        config=config,
        run_specificity=_run,
        source_method="na_mpnn",
    )


@constraint(
    key="deeppbs-motif-specificity",
    label="DeepPBS Motif Specificity",
    config=DeepPBSMotifSpecificityConfig,
    description="Score DNA motif specificity using DeepPBS predictions",
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "deeppbs-specificity"],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def deeppbs_motif_specificity_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: DeepPBSMotifSpecificityConfig,
) -> list[ConstraintOutput]:
    """Specificity score from DeepPBS predicted base-preference matrices.

    Same scoring as the NA-MPNN constraint, but with DeepPBS predictions. If the
    DeepPBS tool call fails, every candidate receives the worst score with an
    explanatory metadata key rather than crashing (structure prediction still raises).

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-candidate tuples (protein +
            DNA chains) folded into a complex.
        config (DeepPBSMotifSpecificityConfig): Constraint configuration.

    Returns:
        list[ConstraintOutput]: One result per candidate; score 0.0 (best) when the
            target beats every off-target by ``desired_margin``, 1.0 (worst) otherwise.
    """

    def _run(pdb_paths: list[str]) -> object:
        return run_deeppbs_specificity(DeepPBSSpecificityInput(pdb_paths=pdb_paths), config.deeppbs_config)

    return _evaluate_motif_specificity(
        input_sequences=input_sequences,
        config=config,
        run_specificity=_run,
        source_method="deeppbs",
        soft_fail_on_tool_error=True,
    )
