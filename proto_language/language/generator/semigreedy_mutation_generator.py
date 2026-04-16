"""Generator for semigreedy single-point mutations guided by logits."""

from typing import Any, Literal, final

import numpy as np
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    PROTEIN_AMINO_ACIDS,
    Generator,
    Sequence,
)
from proto_language.language.generator.generator_registry import generator
from proto_language.utils import softmax


class SemigreedyMutationGeneratorConfig(BaseConfig):
    """Configuration for semigreedy single-point mutation sampling.

    This generator reads ``seq.logits`` (produced by a preceding gradient-based
    optimizer), converts them to a position-specific scoring matrix (PSSM) via
    softmax, and introduces single-point mutations by sampling amino acids from
    the PSSM distribution at selected positions.

    Designed for Stage 2 of the Germinal pipeline: ``MCMCOptimizer`` at near-zero
    temperature with this generator performs greedy/semigreedy discrete refinement.

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
        logit_bias (list[list[float]] | None): Optional additive bias matrix of
            shape ``(L, 20)`` added to ``proposal.logits`` before AA sampling.
            Position weighting still uses ``proposal.logits`` alone. Matches
            Germinal's ``self._inputs["bias"]`` (``shared/model.py:155``).
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
    * ``"plddt"``: positions are weighted by ``(1 - pLDDT)`` read from
      ``proposal.structure.metrics["per_residue_plddt"]``, so structurally
      uncertain residues are mutated more frequently. Requires a
      ``Structure`` with the ``per_residue_plddt`` metric on each proposal.

    This generator implements Germinal's ``design_semigreedy`` phase, where an
    ``MCMCOptimizer`` with near-zero temperature acts as a greedy optimizer and
    ``proposals_per_result > 1`` tries multiple mutations per step.

    Attributes:
        config (SemigreedyMutationGeneratorConfig): Generator configuration.
        position_weighting (Literal["uniform", "entropy", "plddt"]): Position
            selection strategy.
        temperature (float): Softmax temperature for PSSM construction.
        exclude_current (bool): Whether to exclude the current AA when sampling.

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
        self._logit_bias = np.asarray(config.logit_bias, dtype=float) if config.logit_bias is not None else None
        self.position_weighting = config.position_weighting
        self.temperature = config.temperature
        self.exclude_current = config.exclude_current

    def sample(self) -> None:
        """Introduce one single-point mutation per proposal.

        For each proposal sequence:

        1. Read ``proposal.logits`` and convert to a PSSM via softmax at the
           configured temperature.
        2. Select a position using the configured ``position_weighting`` strategy.
        3. Sample a replacement amino acid from ``proposal.logits + logit_bias``
           at that position (optionally excluding current residue via logit penalty).
        4. Write the mutated sequence back to ``proposal.sequence``.

        Raises:
            RuntimeError: If called before ``assign()`` or if a proposal has no logits.
            ValueError: If logits have the wrong shape or ``plddt`` weighting is
                requested but the proposal has no structure with ``per_residue_plddt``.
        """
        self._validate_generator()
        vocab = list(PROTEIN_AMINO_ACIDS)
        vocab_size = len(vocab)
        rng = np.random.default_rng(self._next_seed())

        if self._logit_bias is not None and self._logit_bias.shape[0] != self.segment.sequence_length:
            raise ValueError(
                f"logit_bias has {self._logit_bias.shape[0]} rows but sequence length is {self.segment.sequence_length}."
            )

        for proposal in self.segment.proposal_sequences:
            if proposal.logits is None:
                raise RuntimeError(f"Proposal on segment '{self.segment.label}' has no logits.")

            pssm = self._build_pssm(proposal.logits, vocab_size)
            position_weights = self._compute_position_weights(pssm, proposal)
            position = rng.choice(len(pssm), p=position_weights)

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
            raise ValueError(
                f"Logit matrix shape {matrix.shape} does not match expected shape {expected_shape} "
                f"for segment '{self.segment.label or 'unlabeled'}'."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Logit matrix must contain only finite values.")
        return softmax(matrix / self.temperature)

    def _compute_position_weights(self, pssm: np.ndarray, proposal: Sequence) -> np.ndarray:
        """Compute normalized position selection weights for the configured strategy."""
        seq_len = pssm.shape[0]
        uniform = np.full(seq_len, 1.0 / seq_len)

        if self.position_weighting == "uniform":
            return uniform

        if self.position_weighting == "entropy":
            safe_pssm = np.where(pssm > 0, pssm, 1.0)  # avoid log(0)
            entropy = -np.sum(pssm * np.log(safe_pssm), axis=1)
            total = entropy.sum()
            if total < 1e-12:
                return uniform
            result = entropy / total
            assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy arithmetic for mypy
            return result

        # plddt weighting: (1 - plddt) so low-confidence positions are favored
        if proposal.structure is None:
            raise ValueError("position_weighting='plddt' requires a Structure on each proposal.")
        per_residue = proposal.structure.metrics.get("per_residue_plddt")
        if per_residue is None:
            raise ValueError("position_weighting='plddt' requires 'per_residue_plddt' in proposal.structure.metrics.")
        plddt_array = np.asarray(per_residue, dtype=float)
        if plddt_array.shape != (seq_len,):
            raise ValueError(
                f"per_residue_plddt length {plddt_array.shape[0]} does not match sequence length {seq_len}."
            )
        weights = 1.0 - np.clip(plddt_array, 0.0, 1.0)
        total = weights.sum()
        if total < 1e-12:
            return uniform
        result = weights / total
        assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy arithmetic for mypy
        return result
