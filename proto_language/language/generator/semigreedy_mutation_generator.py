"""Generator for semigreedy single-point mutations guided by logits."""

from collections.abc import Iterable
from typing import Any, Literal, final

import numpy as np
from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    PROTEIN_AMINO_ACIDS,
    Generator,
    Segment,
    Sequence,
)
from proto_language.language.generator.generator_registry import generator
from proto_language.utils import softmax
from proto_language.utils.sequence_logit_bias import (
    SequenceLogitBiasConfig,
    build_sequence_logit_bias_matrix,
    combine_logit_biases,
)


class SemigreedyMutationGeneratorConfig(BaseConfig):
    """Configuration for semigreedy single-point mutation sampling.

    Converts ``seq.logits`` (from a preceding gradient-based optimizer) to a PSSM
    via softmax and samples single-point mutations from it. Stage 2 of the Germinal
    pipeline: paired with ``MCMCOptimizer`` at near-zero temperature for
    greedy/semigreedy discrete refinement.

    Attributes:
        position_weighting (Literal["uniform", "entropy", "plddt"]): Strategy for
            selecting which position to mutate. ``"uniform"`` picks uniformly at
            random. ``"entropy"`` weights positions proportionally to their Shannon
            entropy in the PSSM (higher entropy = more uncertain = more likely to
            be selected). ``"plddt"`` weights by ``(1 - per_residue_plddt)`` from
            ``proposal.structure``, so low-confidence residues are mutated more.
        temperature (float): Softmax temperature applied to logits before building
            the PSSM. Lower values sharpen the distribution.
        exclude_current (bool): Whether to zero out the probability of the current
            amino acid at the selected position before sampling the replacement.
            Guarantees every mutation actually changes the sequence.
        logit_bias (list[list[float]] | None): Additive bias matrix of shape
            ``(L, 20)`` added to ``proposal.logits`` before AA sampling. Position
            weighting still uses ``proposal.logits`` alone.
        sequence_bias (SequenceLogitBiasConfig | None): Declarative additive
            bias built against the assigned segment vocabulary. Combined with
            ``logit_bias``.
        clear_logits (bool): If True, ignore ``proposal.logits`` when sampling the
            replacement amino acid; sample from ``logit_bias`` only (or uniform if
            ``logit_bias`` is None). Incompatible with ``position_weighting="entropy"``.
        frozen_positions (list[int] | None): Zero-indexed positions excluded from
            mutation; the residue at each listed index is preserved from the
            proposal sequence. E.g., disulfide or epitope preservation via
            ``[2, 7]``. Duplicates are ignored.
    """

    position_weighting: Literal["uniform", "entropy", "plddt"] = ConfigField(
        default="uniform",
        title="Position Weighting",
        description="Strategy for selecting mutation positions.",
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Softmax temperature for converting logits to the PSSM.",
        advanced=True,
    )
    exclude_current: bool = ConfigField(
        default=True,
        title="Exclude Current AA",
        description="Zero out the current amino acid before sampling to guarantee a mutation.",
        advanced=True,
    )
    logit_bias: list[list[float]] | None = ConfigField(
        default=None,
        title="Logit Bias",
        description="Additive bias matrix (L x 20) added to logits before AA sampling.",
        advanced=True,
        hidden=True,
    )
    sequence_bias: SequenceLogitBiasConfig | None = ConfigField(
        default=None,
        title="Sequence Bias",
        description="Optional declarative sequence-symbol bias applied before AA sampling.",
        advanced=True,
    )
    clear_logits: bool = ConfigField(
        default=False,
        title="Clear Logits",
        description="Sample replacement AAs from logit_bias only (or uniform), ignoring proposal.logits.",
        advanced=True,
    )
    frozen_positions: list[int] | None = ConfigField(
        default=None,
        title="Frozen Positions",
        description="Zero-indexed positions excluded from mutation.",
        advanced=True,
    )

    @field_validator("logit_bias")
    @classmethod
    def validate_logit_bias(cls, v: Any) -> Any:
        """Validate logit_bias shape and finiteness."""
        if v is None:
            return v
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(PROTEIN_AMINO_ACIDS):
            raise ValueError(f"logit_bias must have shape (L, {len(PROTEIN_AMINO_ACIDS)}), got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("logit_bias must contain only finite values.")
        return v

    @field_validator("frozen_positions")
    @classmethod
    def validate_frozen_positions(cls, v: Any) -> Any:
        """Validate frozen_positions are non-negative indices."""
        if v is None:
            return v
        if not v:
            raise ValueError("frozen_positions must not be empty; use None to disable.")
        if any(p < 0 for p in v):
            raise ValueError(f"frozen_positions must all be non-negative, got {v}.")
        return v

    @model_validator(mode="after")
    def _check_clear_logits_with_entropy(self) -> "SemigreedyMutationGeneratorConfig":
        """Reject the incoherent combination of clear_logits and entropy weighting."""
        if self.clear_logits and self.position_weighting == "entropy":
            raise ValueError(
                "clear_logits=True is incompatible with position_weighting='entropy'; use 'uniform' or 'plddt'."
            )
        return self


@generator(
    key="semigreedy-mutation",
    label="Semigreedy Mutation Generator",
    config=SemigreedyMutationGeneratorConfig,
    description="Logit-guided single-point mutations for semigreedy discrete refinement",
    uses_gpu=False,
    tools_called=[],
    category="mutation",
    supported_sequence_types=["protein"],
)
@final
class SemigreedyMutationGenerator(Generator):
    """Introduce single-point mutations guided by a PSSM derived from ``seq.logits``.

    Each call to ``sample()`` selects one position per proposal sequence and
    replaces the amino acid there by sampling from the softmax distribution over
    logits (with the current residue optionally excluded). Position selection is
    controlled by ``position_weighting``:

    * ``"uniform"``: every position is equally likely.
    * ``"entropy"``: positions with higher Shannon entropy in the PSSM are more
      likely, targeting the most uncertain residues.
    * ``"plddt"``: positions are weighted by ``(1 - pLDDT)`` read from the
      canonical ``proposal.structure.per_residue_plddt`` property, so
      structurally uncertain residues are mutated more frequently. Requires
      each proposal to have a ``Structure`` whose ``b_factor_type`` is
      ``PLDDT`` or ``NORMALIZED_PLDDT``.

    ``frozen_positions`` hard-excludes listed indices from selection (deterministic
    counterpart to ``logit_bias``); whatever residue is there stays. Implements
    Germinal's ``design_semigreedy`` phase (``MCMCOptimizer`` at near-zero
    temperature, ``proposals_per_result > 1``).

    Attributes:
        config (SemigreedyMutationGeneratorConfig): Generator configuration.
        position_weighting (Literal["uniform", "entropy", "plddt"]): Position
            selection strategy.
        temperature (float): Softmax temperature for PSSM construction.
        exclude_current (bool): Whether to exclude the current AA when sampling.
        clear_logits (bool): If True, sample replacement AAs from ``logit_bias``
            only (or uniform), ignoring ``proposal.logits``.

    Example:
        >>> from proto_language.language.core import Segment
        >>> segment = Segment(sequence="ACDEF", sequence_type="protein")
        >>> gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig(position_weighting="entropy"))
        >>> gen.assign(segment)
        >>> # Normally logits come from a GradientOptimizer; here we set them manually:
        >>> import numpy as np
        >>> segment.proposal_sequences[0].logits = np.random.randn(5, 20)
        >>> gen.sample()
        >>> # Exactly one position differs from "ACDEF"
    """

    def __init__(self, config: SemigreedyMutationGeneratorConfig) -> None:
        """Initialize the semigreedy mutation generator."""
        super().__init__()
        self.config = config
        self._raw_logit_bias = np.asarray(config.logit_bias, dtype=float) if config.logit_bias is not None else None
        self._sequence_bias_config = config.sequence_bias
        self._logit_bias: np.ndarray | None = None
        self._frozen_positions: frozenset[int] | None = (
            frozenset(config.frozen_positions) if config.frozen_positions is not None else None
        )
        self.position_weighting = config.position_weighting
        self.temperature = config.temperature
        self.exclude_current = config.exclude_current
        self.clear_logits = config.clear_logits

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        """Assign segment(s) and validate length-dependent config against them."""
        super().assign(segments)
        seq_len = self.segment.sequence_length
        if self._raw_logit_bias is not None and self._raw_logit_bias.shape[0] != seq_len:
            raise ValueError(f"logit_bias has {self._raw_logit_bias.shape[0]} rows but sequence length is {seq_len}.")
        sequence_logit_bias = build_sequence_logit_bias_matrix(self._sequence_bias_config, self.segment)
        self._logit_bias = combine_logit_biases(self._raw_logit_bias, sequence_logit_bias)
        if self._frozen_positions is not None:
            for pos in self._frozen_positions:
                if pos >= seq_len:
                    raise ValueError(f"frozen_positions index {pos} out of range; sequence length is {seq_len}.")
            if len(self._frozen_positions) == seq_len:
                raise ValueError("All positions are frozen; no mutation is possible.")

    def _sample(self) -> None:
        """Introduce one single-point mutation per proposal.

        For each proposal sequence:

        1. Read ``proposal.logits`` and convert to a PSSM via softmax at the
           configured temperature.
        2. Select a position using the configured ``position_weighting`` strategy.
        3. Sample a replacement amino acid at that position. By default, sample
           from ``proposal.logits + logit_bias``. If ``clear_logits=True``, sample
           from ``logit_bias`` alone (or uniform if no bias). Optionally exclude
           the current residue via a logit penalty.
        4. Write the mutated sequence back to ``proposal.sequence``.

        Raises:
            RuntimeError: If called before ``assign()`` or if a proposal has no logits when
                ``clear_logits=False``.
            ValueError: If logits have the wrong shape or ``plddt`` weighting is
                requested but the proposal has no per-residue pLDDT on its structure.
        """
        self._validate_generator()
        vocab = list(PROTEIN_AMINO_ACIDS)
        vocab_size = len(vocab)
        seq_len = self.segment.sequence_length
        rng = np.random.default_rng(self._next_seed())

        for proposal in self.segment.proposal_sequences:
            # When clear_logits=True the entropy weighting is rejected by the validator and
            # AA sampling reads only logit_bias, so proposal.logits is unused — skip building it.
            pssm: np.ndarray | None = None
            if not self.clear_logits:
                if proposal.logits is None:
                    raise RuntimeError(f"Proposal on segment '{self.segment.label}' has no logits.")
                pssm = self._build_pssm(proposal.logits, vocab_size)
            position_weights = self._compute_position_weights(pssm, proposal)
            if self._frozen_positions is not None:
                for pos in self._frozen_positions:
                    position_weights[pos] = 0.0
                total = position_weights.sum()
                if total < 1e-12:
                    raise ValueError(
                        f"All non-frozen positions have zero weight under position_weighting={self.position_weighting!r}."
                    )
                position_weights = position_weights / total
            position = rng.choice(seq_len, p=position_weights)

            if self.clear_logits:
                aa_logits = (
                    self._logit_bias[position].copy()
                    if self._logit_bias is not None
                    else np.zeros(vocab_size, dtype=float)
                )
            else:
                assert proposal.logits is not None  # noqa: S101 -- guarded above when clear_logits=False
                aa_logits = proposal.logits[position].copy()
                if self._logit_bias is not None:
                    aa_logits = aa_logits + self._logit_bias[position]
            aa_logits = aa_logits / self.temperature
            if self.exclude_current:
                aa_logits[vocab.index(proposal.sequence[position])] -= 1e8
            aa_probs = softmax(aa_logits.reshape(1, -1))[0]
            new_aa = vocab[rng.choice(vocab_size, p=aa_probs)]

            seq_list = list(proposal.sequence)
            seq_list[position] = new_aa
            proposal.sequence = "".join(seq_list)

    def _build_pssm(self, logits: np.ndarray, vocab_size: int) -> np.ndarray:
        """Convert raw logits to a PSSM via temperature-scaled softmax."""
        matrix = np.asarray(logits, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Logit matrix must be a 2D array with shape (sequence_length, vocab_size).")
        expected_shape = (self.segment.sequence_length, vocab_size)
        if matrix.shape != expected_shape:
            raise ValueError(f"Logit matrix shape {matrix.shape} does not match expected shape {expected_shape}.")
        if not np.isfinite(matrix).all():
            raise ValueError("Logit matrix must contain only finite values.")
        return softmax(matrix / self.temperature)

    def _compute_position_weights(self, pssm: np.ndarray | None, proposal: Sequence) -> np.ndarray:
        """Compute normalized position selection weights for the configured strategy."""
        seq_len = self.segment.sequence_length
        uniform = np.full(seq_len, 1.0 / seq_len)

        if self.position_weighting == "uniform":
            return uniform

        if self.position_weighting == "entropy":
            assert pssm is not None  # noqa: S101 -- validator rejects entropy + clear_logits
            safe_pssm = np.where(pssm > 0, pssm, 1.0)  # avoid log(0)
            entropy = -np.sum(pssm * np.log(safe_pssm), axis=1)
            total = entropy.sum()
            if total < 1e-12:
                return uniform
            result = entropy / total
            assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy arithmetic for mypy
            return result

        # 1-plddt weighting; uniform fallback when no pLDDT yet.
        if proposal.structure is None:
            return uniform
        per_residue = proposal.structure.per_residue_plddt
        if per_residue is None:
            return uniform
        plddt_array = np.asarray(per_residue, dtype=float)
        if plddt_array.shape != (seq_len,):
            raise ValueError(f"per_residue_plddt length {len(plddt_array)} does not match sequence length {seq_len}.")
        weights = 1.0 - np.clip(plddt_array, 0.0, 1.0)
        total = weights.sum()
        if total < 1e-12:
            return uniform
        result = weights / total
        assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy arithmetic for mypy
        return result
