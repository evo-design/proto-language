"""Generator for sampling sequences from logit distributions."""

from collections.abc import Iterable
from typing import Literal, final

import numpy as np
from pydantic import field_validator

from proto_language.core import Generator, GeneratorInputType, Segment
from proto_language.generator.generator_registry import generator
from proto_language.utils import mean_peak_probability, softmax
from proto_language.utils.base import BaseConfig, ConfigField
from proto_language.utils.sequence_matrices import SequenceLogitBiasConfig, build_sequence_logit_bias_matrix


class PositionWeightGeneratorConfig(BaseConfig):
    """Configuration for position-specific sequence sampling.

    This generator is intended for optimizers that own position-specific
    sequence distributions and need to materialize discrete proposal sequences
    for the rest of the framework.

    Attributes:
        sampling_mode (Literal["argmax", "categorical"]): Whether to decode the
            most likely token at each position or sample stochastically from the
            per-position distribution.
        temperature (float): Softmax temperature applied to logits in ``sample()``.
        sequence_bias (SequenceLogitBiasConfig | None): Optional declarative
            per-position symbol bias resolved against the assigned segment
            vocabulary; added to logits before decoding handoff sequences.
        logit_scale (float): Optional scale factor applied to logits before the
            additive bias and temperature-scaled softmax.
        entropy_positions (list[int] | None): Zero-based positions to include
            when computing the ``mean_peak_probability`` metric (mean per-position
            peak probability). ``None`` = all positions. Segment-length bounds
            are checked at ``sample()`` time (config doesn't know the segment).
    """

    sampling_mode: Literal["argmax", "categorical"] = ConfigField(
        default="argmax",
        title="Sampling Mode",
        description="How to convert position-specific distributions into discrete proposals.",
    )
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Softmax temperature on logits before decoding. Below 1 sharpens; above 1 flattens. Must be > 0.",
    )
    sequence_bias: SequenceLogitBiasConfig | None = ConfigField(
        default=None,
        title="Sequence Bias",
        description="Optional declarative sequence-symbol bias applied before decoding.",
    )
    logit_scale: float = ConfigField(
        default=1.0,
        ge=0.0,
        title="Logit Scale",
        description="Optional scale factor applied to logits before adding sequence_bias and decoding.",
    )
    entropy_positions: list[int] | None = ConfigField(
        default=None,
        title="Peak Prob Positions",
        description="Position indices used in the mean per-position peak-probability metric; None averages over all.",
    )

    @field_validator("entropy_positions")
    @classmethod
    def _check_entropy_positions(cls, value: list[int] | None) -> list[int] | None:
        """Reject empty lists + negative indices at config time; segment bounds are checked at sample()."""
        if value is None:
            return value
        if not value:
            raise ValueError("entropy_positions must be None or non-empty; got [].")
        negative = [p for p in value if p < 0]
        if negative:
            raise ValueError(f"entropy_positions must be non-negative; got {negative}.")
        return value


@generator(
    key="position-weight",
    label="Position Weight Generator",
    config=PositionWeightGeneratorConfig,
    description="Sample sequences from position-specific logit distributions",
    uses_gpu=False,
    tools_called=[],
    supported_sequence_types=["dna", "rna", "protein"],
)
@final
class PositionWeightGenerator(Generator):
    """Convert logit distributions into discrete proposal sequences.

    Reads ``seq.logits`` from each proposal sequence, applies softmax, and
    writes the decoded string back to ``seq.sequence``. Supports deterministic
    argmax decoding or stochastic categorical sampling.

    Designed for gradient-based optimizers that update ``seq.logits`` directly
    and need discrete sequences for handoff or tracking.

    Attributes:
        config (PositionWeightGeneratorConfig): Generator configuration.
        sampling_mode (Literal["argmax", "categorical"]): Decoding strategy.
        temperature (float): Softmax temperature for logits.
        sequence_bias (SequenceLogitBiasConfig | None): Declarative per-position
            symbol bias resolved against the assigned segment vocabulary; added
            to logits before decoding.
        logit_scale (float): Scale factor applied to logits before the additive
            bias and temperature-scaled softmax.
        entropy_positions (list[int] | None): Rows included when computing
            ``mean_peak_probability`` on each proposal's
            ``_generator_metadata["position-weight"]``. ``None`` = all.

    Example:
        >>> segment = Segment(sequence="ACGT", sequence_type="dna")
        >>> gen = PositionWeightGenerator(PositionWeightGeneratorConfig(sampling_mode="argmax"))
        >>> gen.assign(segment)
        >>> segment.proposal_sequences[0].logits = np.array([[5, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]])
        >>> gen.sample()
        >>> segment.proposal_sequences[0].sequence
        'ACGT'
    """

    input_type = GeneratorInputType.LOGITS

    def __init__(self, config: PositionWeightGeneratorConfig) -> None:
        """Initialize the position-weight generator."""
        super().__init__()
        self.config = config
        self.sampling_mode = config.sampling_mode
        self.temperature = config.temperature
        self._sequence_bias_config = config.sequence_bias
        self._logit_bias: np.ndarray | None = None
        self.logit_scale = config.logit_scale
        self.entropy_positions = config.entropy_positions

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        """Assign segment(s) and resolve the declarative bias against the segment vocab."""
        super().assign(segments)
        self._logit_bias = build_sequence_logit_bias_matrix(self._sequence_bias_config, self.segment)

    def _sample(self) -> None:
        """Decode discrete sequences from ``seq.logits`` on each proposal.

        Reads ``.logits`` from each proposal sequence, applies softmax at the
        configured temperature, and writes the decoded string to ``.sequence``.
        Also stashes ``mean_peak_probability`` (mean per-position peak probability,
        optionally restricted to ``entropy_positions``, computed on the
        temperature-scaled softmax; duplicates in ``entropy_positions`` double-count)
        onto ``proposal._generator_metadata["position-weight"]`` so entropy-based
        gates can read it without re-running the softmax.

        Raises:
            RuntimeError: If called before ``assign()`` or if a proposal has no logits.
            ValueError: If logits shape or contents are invalid, or ``entropy_positions``
                references an out-of-range index.
        """
        self._validate_generator()
        vocab = self.segment.ordered_vocab()
        seq_len = self.segment.sequence_length
        if self.entropy_positions is not None:
            out_of_range = [p for p in self.entropy_positions if p >= seq_len]
            if out_of_range:
                raise ValueError(f"entropy_positions {out_of_range} are >= sequence_length ({seq_len}).")

        rng = np.random.default_rng(self._next_seed()) if self.sampling_mode == "categorical" else None
        key = self._spec.key

        for proposal in self.segment.proposal_sequences:
            if proposal.logits is None:
                raise RuntimeError(f"Proposal on segment '{self.segment.label}' has no logits.")
            matrix = self._prepare_matrix(logits=proposal.logits, vocab_size=len(vocab))
            if self.sampling_mode == "argmax":
                proposal.sequence = self._decode_argmax(matrix, vocab)
            else:
                assert rng is not None  # noqa: S101 -- categorical branch always sets rng
                proposal.sequence = self._decode_categorical(matrix, vocab, rng)
            proposal._generator_metadata[key] = {
                "mean_peak_probability": mean_peak_probability(matrix, self.entropy_positions)
            }

    def _prepare_matrix(self, *, logits: np.ndarray, vocab_size: int) -> np.ndarray:
        """Validate, optionally bias/scale, and convert logits to probabilities."""
        matrix = np.asarray(logits, dtype=float)
        self._validate_matrix_shape(matrix, vocab_size)
        matrix = self.logit_scale * matrix
        if self._logit_bias is not None:
            matrix = matrix + self._logit_bias
        return softmax(matrix / self.temperature)

    def _validate_matrix_shape(self, matrix: np.ndarray, vocab_size: int) -> None:
        """Validate the logit matrix shape and numeric contents."""
        if matrix.ndim != 2:
            raise ValueError("Logit matrix must be a 2D array with shape (sequence_length, vocab_size).")
        expected_shape = (self.segment.sequence_length, vocab_size)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Logit matrix shape {matrix.shape} does not match expected shape {expected_shape} for segment '{self.segment.label or 'unlabeled'}'."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Logit matrix must contain only finite values.")

    @staticmethod
    def _decode_argmax(matrix: np.ndarray, vocab: list[str]) -> str:
        """Decode the most likely token at each position."""
        token_indices = np.argmax(matrix, axis=1)
        return "".join(vocab[index] for index in token_indices)

    @staticmethod
    def _decode_categorical(matrix: np.ndarray, vocab: list[str], rng: np.random.Generator) -> str:
        """Sample one discrete sequence from a per-position categorical distribution."""
        token_indices = [rng.choice(len(vocab), p=row_probabilities) for row_probabilities in matrix]
        return "".join(vocab[index] for index in token_indices)
