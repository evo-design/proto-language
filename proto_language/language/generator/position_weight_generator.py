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
    key="position-weight",
    label="Position Weight Generator",
    config=PositionWeightGeneratorConfig,
    description="Sample sequences from position-specific logit distributions",
    uses_gpu=False,
    tools_called=[],
    category="mutation",
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

    Example:
        >>> segment = Segment(sequence="ACGT", sequence_type="dna")
        >>> gen = PositionWeightGenerator(PositionWeightGeneratorConfig(sampling_mode="argmax"))
        >>> gen.assign(segment)
        >>> segment.proposal_sequences[0].logits = np.array([[5, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]])
        >>> gen.sample()
        >>> segment.proposal_sequences[0].sequence
        'ACGT'
    """

    def __init__(self, config: PositionWeightGeneratorConfig) -> None:
        """Initialize the position-weight generator."""
        super().__init__()
        self.config = config
        self.sampling_mode = config.sampling_mode
        self.temperature = config.temperature

    def sample(self) -> None:
        """Decode discrete sequences from ``seq.logits`` on each proposal.

        Reads ``.logits`` from each proposal sequence, applies softmax at the
        configured temperature, and writes the decoded string to ``.sequence``.

        Raises:
            RuntimeError: If called before ``assign()`` or if a proposal has no logits.
            ValueError: If logits shape or contents are invalid.
        """
        self._validate_generator()
        vocab = self._ordered_vocab()

        rng = np.random.default_rng(self._next_seed()) if self.sampling_mode == "categorical" else None

        for proposal in self.segment.proposal_sequences:
            if proposal.logits is None:
                raise RuntimeError(f"Proposal on segment '{self.segment.label}' has no logits.")
            matrix = self._prepare_matrix(logits=proposal.logits, vocab_size=len(vocab))
            if self.sampling_mode == "argmax":
                proposal.sequence = self._decode_argmax(matrix, vocab)
            else:
                assert rng is not None  # noqa: S101 -- categorical branch always sets rng
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

    def _prepare_matrix(self, *, logits: np.ndarray, vocab_size: int) -> np.ndarray:
        """Validate logits and convert to a probability matrix via softmax."""
        matrix = np.asarray(logits, dtype=float)
        self._validate_matrix_shape(matrix, vocab_size)
        return self._softmax(matrix / self.temperature)

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
