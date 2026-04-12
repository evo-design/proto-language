"""Generator for sampling sequences from logit distributions."""

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
        temperature (float): Softmax temperature applied to logits in ``sample()``.
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
    description="Sample sequences from position-specific logit distributions",
    uses_gpu=False,
    tools_called=[],
    category="mutation",
    supported_sequence_types=["dna", "rna", "protein"],
)
@final
class PositionProbabilityGenerator(Generator):
    """Convert position-specific logit distributions into discrete proposal sequences.

    The optimizer owns the continuous state and calls ``sample()`` with logits.
    Logits are converted into probabilities via softmax internally. This
    generator only handles deterministic argmax decoding or stochastic
    categorical sampling.

    Note:
        ``sample()`` requires ``logits`` and is not compatible with optimizers
        that call ``sample()`` with no arguments (e.g., ``MCMCOptimizer``).
        Designed for optimizers that own continuous state and pass logit
        distributions at each iteration.

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
        logits: np.ndarray | None = None,
        temperature: float | None = None,
    ) -> None:
        """Populate proposal_sequences from a position-specific logit matrix.

        The matrix is expected to have shape ``(sequence_length, vocab_size)``
        using the canonical vocab order for the assigned segment's sequence type.

        Args:
            logits (np.ndarray | None): Position-specific logit matrix. Softmax
                is applied internally with the configured temperature.
            temperature (float | None): Override for the config temperature.

        Raises:
            ValueError: If logits is not provided, or if the matrix shape or
                contents are invalid, or if temperature is not positive.
            RuntimeError: If called before ``assign()``.
        """
        self._validate_generator()

        if logits is None:
            raise ValueError("logits is required.")

        vocab = self._ordered_vocab()
        matrix = self._prepare_matrix(logits=logits, temperature=temperature, vocab_size=len(vocab))

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
        logits: np.ndarray,
        temperature: float | None,
        vocab_size: int,
    ) -> np.ndarray:
        """Validate logits and convert to a probability matrix via softmax."""
        resolved_temperature = self.temperature if temperature is None else temperature
        if resolved_temperature <= 0:
            raise ValueError("temperature must be positive.")
        matrix = np.asarray(logits, dtype=float)
        self._validate_matrix_shape(matrix, vocab_size)
        return self._softmax(matrix / resolved_temperature)

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
