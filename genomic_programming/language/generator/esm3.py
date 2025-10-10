"""
Esm3 Generator

Extracted from generator.py for better code organization.
"""

from typing import List, final

from ..base import Generator, Segment


@final
class ESM3Generator(Generator):
    """
    A protein sequence generator using the ESM-3 open protein language model.

    This generator uses the (open) ESM-3 protein language model to propose sequences and
    mutations based on the model's logits. It supports various decoding strategies
    for selecting positions to mutate and uses temperature-controlled sampling
    for amino acid selection.

    Examples:
        Basic protein generation:
        >>> segment = Segment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen = ESM3Generator(
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5,
        ...     batch_size=3
        ... )
        >>> gen.assign(segment)  # Creates random initial sequences from mask tokens
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(
        self,
        sequence_length: int = 100,
        temperature: float = 1.0,
        decoding_method: str = "entropy",
        top_k: int = 5,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the ESM3 generator with model and sampling configuration.

        Args:
            sequence_length: Length of protein sequences to generate.
            temperature: Sampling temperature for amino acid selection.
            decoding_method: Strategy for selecting positions to sample:
                - 'entropy': Choose positions with highest prediction entropy
                - 'max_logit': Choose positions with highest maximum logits
                - 'random': Choose positions randomly
            top_k: Number of positions to sample per iteration.
            batch_size: Number of sequences to generate simultaneously.
        """
        super().__init__(batch_size=batch_size)
        if top_k > sequence_length:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed sequence_length ({sequence_length})"
            )

        self.sequence_length = sequence_length
        self.temperature = temperature
        self.decoding_method = decoding_method
        self.top_k = top_k
        self.batch_size = batch_size

    def assign(
        self, assigned_segments: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        Creates initial sequences by running ESM3 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If the segment already contains sequences, they will be used as starting points.

        Args:
            assigned_segments: A single Segment to be assigned to this generator.

        Raises:
            ValueError: If assigned_segments is not a single Segment object.
            AssertionError: If provided sequence length doesn't match configured length.
        """
        # Validate that we received a single Segment, not a list or other type
        if not isinstance(assigned_segments, Segment):
            raise ValueError(
                f"ESM3Generator.assign() expects a single Segment object, "
                f"got {type(assigned_segments).__name__}. If you have multiple segments, "
                f"assign them to separate generator instances."
            )

        # Validate provided sequence length if not empty
        initial_sequence = assigned_segments.batch_sequences[0].sequence
        if initial_sequence != "":
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )

        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM3 logits,
        selects top-k positions based on the decoding method, and samples new amino acids
        at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()
        sequences = [
            self._generator_output.batch_sequences[i].sequence
            for i in range(self.batch_size)
        ]

        # Use ESM3 sampling tool
        from ...tools.models.language_models.esm3.esm3 import run_esm3_sample, ESM3SampleConfig
        
        config = ESM3SampleConfig(
            sequences=sequences,
            sequence_length=self.sequence_length,
            temperature=self.temperature,
            decoding_method=self.decoding_method,
            top_k=self.top_k,
            keep_on_device=True,  # Keep for repeated calls
            verbose=False
        )
        
        result = run_esm3_sample(config)
        mutated_sequences = result.sequences

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._generator_output.batch_sequences[i].sequence = sequence
