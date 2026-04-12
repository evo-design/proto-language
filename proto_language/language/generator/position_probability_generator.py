"""Generator for sampling sequences from position-specific probability distributions."""

from typing import Literal, final

import numpy as np

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    DNA_NUCLEOTIDES,
    PROTEIN_AMINO_ACIDS,
    RNA_NUCLEOTIDES,
    Generator,
)
from proto_language.language.generator.generator_registry import generator


class PositionProbabilityGeneratorConfig(BaseConfig):
    """Configuration for position-specific sequence sampling.

    This generator is intended for optimizers that own position-specific
    sequence distributions and need to materialize discrete proposal sequences
    for the rest of the framework.

    Attributes:
        sampling_mode (Literal["argmax", "categorical"]): Whether to decode the
            most likely token at each position or sample stochastically from the
            per-position distribution.
        temperature (float): Softmax temperature applied when logits are
            provided to ``sample()``.
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
        description="Softmax temperature used when logits are provided.",
        advanced=True,
    )


@generator(
    key="position-probability",
    label="Position Probability Generator",
    config=PositionProbabilityGeneratorConfig,
    description="Sample sequences from position-specific probability distributions",
    uses_gpu=False,
    tools_called=[],
    category="mutation",
    supported_sequence_types=["dna", "rna", "protein"],
)
@final
class PositionProbabilityGenerator(Generator):
    """Convert position-specific distributions into discrete proposal sequences.

    The optimizer owns the continuous state and calls ``sample()`` with either
    probabilities or logits. Logits are converted into probabilities internally.
    This generator only handles deterministic argmax decoding or stochastic
    categorical sampling.

    Note:
        ``sample()`` requires ``probabilities`` or ``logits`` and is not
        compatible with optimizers that call ``sample()`` with no arguments
        (e.g., ``MCMCOptimizer``). Designed for optimizers that own continuous
        state and pass distributions at each iteration.

    Attributes:
        config (PositionProbabilityGeneratorConfig): Generator configuration.
        sampling_mode (Literal["argmax", "categorical"]): Decoding strategy.
        temperature (float): Default softmax temperature for logits.

    Example:
        >>> segment = Segment(sequence="ACGT", sequence_type="dna")
        >>> gen = PositionProbabilityGenerator(PositionProbabilityGeneratorConfig(sampling_mode="argmax"))
        >>> gen.assign(segment)
        >>> logits = np.array([[5, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]])
        >>> gen.sample(logits=logits)
        >>> segment.proposal_sequences[0].sequence
        'ACGT'
    """

    def __init__(self, config: PositionProbabilityGeneratorConfig) -> None:
        """Initialize the position-probability generator."""
        super().__init__()
        self.config = config
        self.sampling_mode = config.sampling_mode
        self.temperature = config.temperature

    def sample(
        self,
        probabilities: np.ndarray | None = None,
        logits: np.ndarray | None = None,
        temperature: float | None = None,
    ) -> None:
        """Populate proposal_sequences from position-specific distributions.

        Exactly one of ``probabilities`` or ``logits`` must be provided. The
        matrix is expected to have shape ``(sequence_length, vocab_size)`` using
        the canonical vocab order for the assigned segment's sequence type.

        Args:
            probabilities (np.ndarray | None): Position-specific probability
                matrix. Rows are normalized internally.
            logits (np.ndarray | None): Position-specific logit matrix.
                Softmax is applied internally with the configured temperature.
            temperature (float | None): Override for the config temperature.
                Only valid when logits are provided.

        Raises:
            ValueError: If neither or both inputs are provided, or if the
                matrix shape or contents are invalid.
            RuntimeError: If called before ``assign()``.
        """
        self._validate_generator()

        if (probabilities is None) == (logits is None):
            raise ValueError("Provide exactly one of probabilities or logits.")
        if temperature is not None and logits is None:
            raise ValueError("temperature is only supported with logits, not probabilities.")

        vocab = self._ordered_vocab()
        matrix = self._prepare_matrix(
            probabilities=probabilities,
            logits=logits,
            temperature=temperature,
            vocab_size=len(vocab),
        )

        if self.sampling_mode == "argmax":
            sequence = self._decode_argmax(matrix, vocab)
            for proposal in self.segment.proposal_sequences:
                proposal.sequence = sequence
            return

        rng = np.random.default_rng(self._next_seed())
        for proposal in self.segment.proposal_sequences:
            proposal.sequence = self._decode_categorical(matrix, vocab, rng)

    def _ordered_vocab(self) -> list[str]:
        """Return a deterministic vocab order for the assigned segment."""
        canonical_vocab = {
            "dna": DNA_NUCLEOTIDES,
            "rna": RNA_NUCLEOTIDES,
            "protein": PROTEIN_AMINO_ACIDS,
        }[self.segment.sequence_type]

        assert self.segment.valid_chars is not None  # noqa: S101 -- validated by Segment for non-ligands
        valid_chars = set(self.segment.valid_chars)
        ordered_vocab = [char for char in canonical_vocab if char in valid_chars]
        ordered_vocab.extend(sorted(valid_chars - set(ordered_vocab)))  # defensive: custom valid_chars

        if not ordered_vocab:
            raise ValueError(f"Segment '{self.segment.label or 'unlabeled'}' has no valid characters for sampling.")
        return ordered_vocab

    def _prepare_matrix(
        self,
        *,
        probabilities: np.ndarray | None,
        logits: np.ndarray | None,
        temperature: float | None,
        vocab_size: int,
    ) -> np.ndarray:
        """Validate and normalize the position-specific sampling matrix."""
        if logits is not None:
            resolved_temperature = self.temperature if temperature is None else temperature
            if resolved_temperature <= 0:
                raise ValueError("temperature must be positive.")
            matrix = np.asarray(logits, dtype=float)
            self._validate_matrix_shape(matrix, vocab_size)
            return self._softmax(matrix / resolved_temperature)

        assert probabilities is not None  # noqa: S101 -- probabilities/logits exclusivity checked by caller
        matrix = np.asarray(probabilities, dtype=float)
        self._validate_matrix_shape(matrix, vocab_size)
        return self._normalize_probabilities(matrix)

    def _validate_matrix_shape(self, matrix: np.ndarray, vocab_size: int) -> None:
        """Validate the sampling matrix shape and numeric contents."""
        if matrix.ndim != 2:
            raise ValueError("Sampling state must be a 2D array with shape (sequence_length, vocab_size).")
        expected_shape = (self.segment.sequence_length, vocab_size)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Sampling state shape {matrix.shape} does not match expected shape {expected_shape} for segment '{self.segment.label or 'unlabeled'}'."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Sampling state must contain only finite values.")

    def _normalize_probabilities(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize non-negative row weights into probabilities."""
        if (matrix < 0).any():
            raise ValueError("Probabilities must be non-negative.")

        row_sums = matrix.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("Each position must have a positive probability mass.")
        result = matrix / row_sums
        assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy scalar arithmetic for mypy
        return result

    @staticmethod
    def _softmax(matrix: np.ndarray) -> np.ndarray:
        """Compute a numerically stable row-wise softmax."""
        shifted = matrix - np.max(matrix, axis=1, keepdims=True)
        exp_matrix = np.exp(shifted)
        result = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
        assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy scalar arithmetic for mypy
        return result

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
