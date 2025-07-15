"""
Defines the base classes for the high-level programming language framework.

This module provides the core abstractions for sequence programming:
- ConstructSegment: Individual sequence variables with validation and metadata
- Construct: Fully-defined biological construct consisting of a collection of ConstructSegment objects
- Constraint: Scoring functions that evaluate sequence quality
- Generator: Base class for sequence generation algorithms
- IterativeGenerator: Specialized generator for iterative optimization

These classes work together to enable flexible, constraint-driven sequence design
using various generation strategies like MCMC, autoregressive models, etc.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Dict, Any, Set, Optional, Iterator, Iterable
from enum import Enum
from itertools import zip_longest
import numpy as np
import copy


class SequenceType(Enum):
    """Enumeration of supported biological sequence types."""

    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"


class ConstraintType(Enum):
    """Enumeration of constraint evaluation strategies for multiple inputs."""

    CONTIGUOUS = "contiguous"  # Concatenate sequences before evaluation
    DISJOINT = "disjoint"  # Evaluate sequences separately as a group


class Sequence:
    """
    Internal data structure for the basic unit of the programming language.

    Represents a single DNA, RNA, or protein sequence. The class enforces sequence type
    constraints and maintains metadata that gets updated when the sequence changes.
    """

    def __init__(
        self,
        sequence: str = "",
        sequence_type: SequenceType = SequenceType.DNA,
        valid_chars: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a Sequence with sequence data and metadata.

        Args:
            sequence: The biological sequence string. Defaults to empty string.
            sequence_type: Type of biological sequence (SequenceType.DNA, SequenceType.RNA, or SequenceType.PROTEIN). Defaults to DNA.
            valid_chars: Optional custom set of valid characters for sequence validation.
                If provided, overrides the default character set for the sequence_type.
            metadata: Additional data associated with this sequence.
        """
        self.sequence_type: SequenceType = sequence_type
        # Set up character validation based on sequence type or custom valid_chars
        if valid_chars:
            self._valid_chars: Optional[Set[str]] = valid_chars
        elif self.sequence_type == SequenceType.DNA:
            self._valid_chars = set("ACGT")
        elif self.sequence_type == SequenceType.RNA:
            self._valid_chars = set("ACGU")
        elif self.sequence_type == SequenceType.PROTEIN:
            self._valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        else:
            raise ValueError(f"Unsupported sequence_type: {self.sequence_type}")

        self._validate_sequence(sequence)
        self._sequence: str = sequence
        self._metadata: Dict[str, Any] = {
            "sequence": sequence,
            "sequence_length": len(sequence),
            "energy_score": "N/A",
            "time_step": "N/A",
        }
        # Update metadata with any input metadata
        if metadata:
            self._metadata.update(metadata)

    def _validate_sequence(self, sequence: str) -> None:
        """
        Validate that sequence contains only allowed characters for its type.

        Args:
            sequence: The sequence string to validate.

        Raises:
            ValueError: If sequence contains invalid characters for this sequence type.
        """
        invalid_chars = set(sequence) - self._valid_chars
        if invalid_chars:
            raise ValueError(
                f"Invalid characters found: {', '.join(invalid_chars)}. "
                f"Valid characters are: {', '.join(sorted(self._valid_chars))}"
            )

    @property
    def sequence(self) -> str:
        """
        Get the current sequence string.

        Returns:
            The sequence string.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, new_sequence: str) -> None:
        """
        Set a new sequence string with validation and metadata updates.

        Args:
            new_sequence: The new sequence string to set.

        Raises:
            ValueError: If the new sequence contains invalid characters.
        """
        # TODO: REVISIT THIS TRUNCATION
        # # Truncate at the first space character (EOS/space token)
        # space_index = new_sequence.find(' ')
        # if space_index != -1:
        #     new_sequence = new_sequence[:space_index]

        self._validate_sequence(new_sequence)
        self._sequence = new_sequence
        self._metadata["sequence"] = new_sequence
        self._metadata["sequence_length"] = len(new_sequence)

    def __len__(self) -> int:
        """
        Get the length of the sequence.

        Returns:
            Number of characters in the sequence.
        """
        return len(self._sequence)

    def __str__(self) -> str:
        """
        Get the sequence as a string.

        Returns:
            The sequence string.
        """
        return self._sequence


class ConstructSegment:
    """
    External class that represents the building blocks for a Construct.

    This is the most modular user-facing unit for the programming language.

    Examples:
        Creating a ConstructSegment:
        >>> segment = ConstructSegment(sequence="ATCG", sequence_type=SequenceType.DNA)
        >>> segment.batch_sequences  # [Sequence(sequence="ATCG", sequence_type=SequenceType.DNA)]
    """

    def __init__(
        self,
        sequence: str = "",
        sequence_type: SequenceType = SequenceType.DNA,
        valid_chars: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a ConstructSegment with a single sequence.

        Args:
            sequence: The biological sequence string. Defaults to empty string.
            sequence_type: Type of biological sequence (DNA, RNA, or PROTEIN). Defaults to DNA.
            valid_chars: Optional custom set of valid characters for sequence validation.
            metadata: Additional data associated with this sequence.
        """
        seq = Sequence(
            sequence=sequence,
            sequence_type=sequence_type,
            metadata=metadata,
            valid_chars=valid_chars,
        )
        self.batch_sequences: List[Sequence] = [seq]
        self.sequence_type: SequenceType = seq.sequence_type
        self._valid_chars: Optional[Set[str]] = seq._valid_chars
        self._is_assigned: bool = False

    def create_batch(self, batch_size: int) -> None:
        """
        Set the batch size by replicating the first sequence across the batch.

        Args:
            batch_size: The desired batch size.
        """
        self.batch_sequences = [
            copy.deepcopy(self.batch_sequences[0]) for _ in range(batch_size)
        ]

    def __len__(self) -> int:
        """
        Get the batch size of the ConstructSegment.

        Returns:
            Number of Sequence objects in the ConstructSegment.
        """
        return len(self.batch_sequences)

    def __iter__(self) -> Iterator[Sequence]:
        """
        Iterate over all Sequence objects in the ConstructSegment.

        Returns:
            Iterator over Sequence objects.
        """
        return iter(self.batch_sequences)

    def __getitem__(self, index: int) -> Sequence:
        """
        Get a specific sequence from the batch by index.

        Args:
            index: The index of the sequence to retrieve.

        Returns:
            The Sequence object at the specified index.
        """
        return self.batch_sequences[index]


class Construct:
    """
    External class that represents a full biological construct.

    Consists of multiple ConstructSegment objects that are concatenated together.

    Examples:
        Creating a construct from multiple segments:
        >>> seg1 = ConstructSegment("ATG", SequenceType.DNA)  # Start codon
        >>> seg2 = ConstructSegment("GCTAGC", SequenceType.DNA)  # Coding region
        >>> seg3 = ConstructSegment("TAG", SequenceType.DNA)  # Stop codon
        >>> construct = Construct([seg1, seg2, seg3])
        >>> construct.batch_sequences  # [Sequence("ATGGCTAGCTAG", SequenceType.DNA)]
    """

    def __init__(
        self,
        segments: Iterable[ConstructSegment],
    ) -> None:
        """
        Initialize a Construct with ConstructSegment objects.

        Args:
            segments: An iterable of ConstructSegment objects in order.

        Raises:
            ValueError: If construct contains no segments, segments have different
                sequence types, or segments have different valid characters.
        """
        # Convert to tuple for validation and storage
        self.segments: Tuple[ConstructSegment, ...] = tuple(segments)

        # Ensure segments are valid
        if not self.segments:
            raise ValueError("Construct must contain at least one segment")
        if not all(
            segment.sequence_type == self.segments[0].sequence_type
            for segment in self.segments
        ):
            all_types = set(segment.sequence_type for segment in self.segments)
            raise ValueError(
                f"All segments in a construct must have the same sequence_type. Found: {all_types}"
            )
        if not all(
            segment._valid_chars == self.segments[0]._valid_chars
            for segment in self.segments
        ):
            raise ValueError(
                "All segments in a construct must have the same valid_chars."
            )

        self.sequence_type: SequenceType = self.segments[0].sequence_type
        self._valid_chars: Optional[Set[str]] = self.segments[0]._valid_chars

    @property
    def batch_sequences(self) -> Tuple[Sequence, ...]:
        """
        Get the concatenated Sequence objects batch that represent one user-facing Construct.

        Returns:
            Tuple of concatenated Sequence objects where each element represents
            the concatenation of the i-th sequence from each segment in order.
            The final batch size is equal to the smallest batch size of the segments.
        """
        sequences = []
        segment_sequences = [segment.batch_sequences for segment in self.segments]
        for corresponding_seqs in zip(*segment_sequences):
            concatenated_seq = "".join(
                seq.sequence for seq in corresponding_seqs if seq is not None
            )

            # TODO: REVISIT THIS TO AVOID METADATA COLLISION
            # Merge metadata from all corresponding sequences
            merged_metadata = {}
            for seq in corresponding_seqs:
                if seq is not None:
                    merged_metadata.update(seq._metadata)

            # Create new Sequence object with concatenated sequence and merged metadata
            sequence_obj = Sequence(
                sequence=concatenated_seq,
                sequence_type=self.sequence_type,
                metadata=merged_metadata,
                valid_chars=self._valid_chars,
            )
            # Ensure metadata reflects the concatenated sequence
            sequence_obj._metadata["sequence"] = concatenated_seq
            sequence_obj._metadata["sequence_length"] = len(concatenated_seq)
            sequences.append(sequence_obj)
        return tuple(sequences)


class Constraint:
    """
    Constraints define the objective function for construct optimization.

    Constraints score how well segments satisfy biological or design requirements by
    taking in ConstructSegment objects and returning scores where lower values
    indicate better satisfaction of the constraint.

    Examples:
        Creating a length constraint:
        >>> def length_constraint(sequence, config):
        ...     target = config['target_length']
        ...     return abs(len(sequence) - target) / target
        >>> segment = ConstructSegment("ATCG", SequenceType.DNA)
        >>> constraint = Constraint(
        ...     inputs=[segment],
        ...     scoring_function=length_constraint,
        ...     scoring_function_config={'target_length': 4},
        ...     constraint_type=ConstraintType.CONTIGUOUS,
        ...     name='length_constraint'
        ... )
        >>> constraint.evaluate()  # [0.0]
    """

    def __init__(
        self,
        inputs: Iterable[ConstructSegment],
        scoring_function: Callable[[Sequence | Tuple[Sequence], Dict[str, Any]], float],
        scoring_function_config: Dict[str, Any] = {},
        constraint_type: ConstraintType = ConstraintType.CONTIGUOUS,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a constraint with its inputs and scoring function.

        Args:
            inputs: The ConstructSegment object(s) this constraint evaluates.
                Can be a single ConstructSegment or an iterable of ConstructSegment objects.
            scoring_function: Function that scores sequences.
                Inputs are single sequence for contiguous constraints or tuple of sequences for disjoint constraints.
                Outputs are a float score between 0.0 and 1.0. Lower values are better.
            scoring_function_config: Configuration parameters passed to scoring_function.
            constraint_type: How to process multiple inputs:
                - CONTIGUOUS: Concatenate sequences before evaluation
                - DISJOINT: Evaluate sequences separately as a group
            name: Optional unique name for this constraint. Metadata keys will be prefixed with this name to avoid collisions.
        """
        self.inputs: Tuple[ConstructSegment] = tuple(inputs)
        self.scoring_function: Callable[
            [Sequence | Tuple[Sequence], Dict[str, Any]], float
        ] = scoring_function
        self.scoring_function_config: Dict[str, Any] = scoring_function_config
        self.constraint_type: ConstraintType = constraint_type
        self.name: Optional[str] = name

    def _process_inputs(
        self, inputs: Tuple[ConstructSegment], constraint_type: ConstraintType
    ) -> List[Sequence] | List[Tuple[Sequence, ...]]:
        """
        Transform batched ConstructSegment inputs into the format expected by the scoring function.

        Processes segments by corresponding indices across batches. If ConstructSegment inputs have
        different batch sizes, the missing segments are padded with None. # TODO: REVIEW THIS APPROACH

        Args:
            inputs: Tuple of ConstructSegment objects to process.
            constraint_type: Strategy for combining inputs:
                - CONTIGUOUS: Concatenate corresponding segments
                - DISJOINT: Group corresponding segments as tuples

        Returns:
            For CONTIGUOUS: List of Sequence objects with concatenated sequences.
            For DISJOINT: List of tuples, each containing corresponding Sequence objects.

        Raises:
            ValueError: Inconsistent sequence_type or valid_chars between inputs in CONTIGUOUS case.
        """
        if constraint_type == ConstraintType.CONTIGUOUS:
            # Ensure sequence_type and valid_chars consistency across all inputs
            if not all(
                input_batch.sequence_type == inputs[0].sequence_type
                for input_batch in inputs
            ):
                all_types = {input_batch.sequence_type for input_batch in inputs}
                raise ValueError(
                    f"Inconsistent sequence_type across inputs. Found: {all_types}"
                )
            if not all(
                input_batch._valid_chars == inputs[0]._valid_chars
                for input_batch in inputs
            ):
                raise ValueError("Inconsistent valid_chars across inputs.")

            # Get sequence_type and valid_chars from the first input
            sequence_type = inputs[0].sequence_type
            valid_chars = inputs[0]._valid_chars

            # Concatenate sequences by corresponding indices across batches
            result = []
            segment_sequences = [input_batch.batch_sequences for input_batch in inputs]
            for corresponding_idx_seqs in zip_longest(
                *segment_sequences, fillvalue=None
            ):
                concatenated_sequence = "".join(
                    seq.sequence for seq in corresponding_idx_seqs if seq is not None
                )
                # Merge metadata from all corresponding sequences
                merged_metadata = {}
                for seq in corresponding_idx_seqs:
                    if seq is not None:
                        merged_metadata.update(seq._metadata)

                result.append(
                    Sequence(
                        sequence=concatenated_sequence,
                        sequence_type=sequence_type,
                        metadata=merged_metadata,
                        valid_chars=valid_chars,
                    )
                )
            return result

        elif constraint_type == ConstraintType.DISJOINT:
            # Extract sequences from each ConstructSegment and group by corresponding indices
            result = []
            segment_sequences = [input_batch.batch_sequences for input_batch in inputs]
            for corresponding_idx_seqs in zip_longest(
                *segment_sequences, fillvalue=None
            ):
                result.append(tuple(corresponding_idx_seqs))
            return result

    def evaluate(self) -> List[float]:
        """
        Evaluate each Sequence object in the ConstructSegment batch.

        Returns:
            List of scores between 0.0 and 1.0, one per Sequence object in the batch.
            Returns float('inf') for invalid sequences.
        """
        # Preprocess inputs depending on the constraint type
        scoring_function_inputs = self._process_inputs(
            self.inputs, self.constraint_type
        )

        # TODO: REFACTOR METADATA
        scores = []
        for i, input in enumerate(scoring_function_inputs):
            # Check for invalid input scenarios
            if isinstance(input, tuple) and None in input:
                scores.append(float("inf"))
            else:
                scores.append(
                    self.scoring_function(input, self.scoring_function_config)
                )
                if isinstance(input, Sequence):
                    # Propagate metadata back to all input sequences at index i
                    for batch in self.inputs:
                        original_seq = batch.batch_sequences[i]
                        if original_seq is not None:
                            for key in input._metadata.keys():
                                if (
                                    key != "sequence"
                                ):  # Don't overwrite original sequence content
                                    value = input._metadata[key]
                                    # Prefix metadata key with constraint name if provided
                                    prefixed_key = (
                                        f"{self.name}_{key}" if self.name else key
                                    )
                                    original_seq._metadata[prefixed_key] = value
                elif isinstance(input, tuple):
                    raise NotImplementedError(
                        "Handle DISJOINT case where input is a Tuple."
                    )
        return scores


class Generator(ABC):
    """
    Generator base class that creates/modifies sequences during optimization.

    Subclasses must implement assign() to assign sequences to generators and
    sample() to propose modified sequences.
    """

    def __init__(
        self,
        batch_size: int = 1,
        **hyperparameters: Any,
    ) -> None:
        """
        Initialize the generator with configuration parameters.

        Args:
            batch_size: Number of sequence variants to generate simultaneously.
            **hyperparameters: Additional configuration specific to the generator type.
                These are stored and can be accessed by subclasses.
        """
        self.batch_size: int = batch_size
        self.hyperparameters: Dict[str, Any] = hyperparameters
        self._is_initialized: bool = False

        # Support both single output (common) and multiple outputs (less common)
        self._generator_output: Optional[ConstructSegment] = None
        self._generator_outputs: Optional[Tuple[ConstructSegment, ...]] = None

    def _validate_generator(self) -> None:
        """
        Validate that the generator has been assigned and outputs are properly initialized.

        Raises:
            RuntimeError: If generator hasn't been assigned or outputs aren't initialized.
        """
        if not self._is_initialized:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been assigned. "
                "Call assign() first."
            )

        if self._generator_output is None and self._generator_outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} must initialize either "
                "_generator_output (for single output) or _generator_outputs (for multiple outputs) "
                "in the assign() method."
            )

    @abstractmethod
    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign ConstructSegment objects to the generator and initialize the generator.
        The generator will modify these ConstructSegment objects internally during sampling.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            NotImplementedError: If not implemented by subclass.

        Note:
            This method must set self._is_initialized = True and initialize either:
            - self._generator_output: ConstructSegment (for single output - most common)
            - self._generator_outputs: Iterable[ConstructSegment] (for multiple outputs)

            Most generators should use _generator_output for simplicity.
            Use _generator_outputs only when the generator manages multiple distinct outputs.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the assign method."
        )

    @abstractmethod
    def sample(self, **kwargs: Any) -> None:
        """
        Sample new sequences by modifying generator outputs in-place.

        Args:
            **kwargs: Keyword arguments for sampling (subclass-specific).

        Raises:
            RuntimeError: If called before assign() has been called.
            NotImplementedError: If not implemented by subclass.

        Note:
            This method should modify sequences in-place for efficiency.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    def get_generator_outputs(self) -> Tuple[ConstructSegment, ...]:
        """
        Access the internal _generator_output or _generator_outputs as a tuple.

        Returns:
            Tuple of ConstructSegment objects that are modified during sampling.
            For single outputs, returns a tuple with one element.
            For multiple outputs, returns the tuple directly.

        Raises:
            RuntimeError: If called before assign() or if neither output is initialized.
        """
        # Ensure generator is properly initialized
        self._validate_generator()

        # Return the appropriate outputs as a tuple
        return self._generator_outputs or (self._generator_output,)

    def __len__(self) -> int:
        """
        Get the number of ConstructSegments managed by this generator.

        Returns:
            Number of ConstructSegment objects (1 for single output, N for multiple outputs).

        Raises:
            RuntimeError: If called before assign() has been called.
        """
        return len(self.get_generator_outputs())


class IterativeGenerator(Generator):
    """
    Specialized generator for iterative optimization with energy-based evaluation.

    Extends Generator to support iterative algorithms like MCMC that require
    energy evaluation and state tracking. The class works with multiple
    sub-generators and constraints.
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        **hyperparameters: Any,
    ) -> None:
        """
        Initialize the IterativeGenerator.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            **hyperparameters: Additional configuration parameters.
        """
        super().__init__(**hyperparameters)
        self.constructs = constructs
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights = constraint_weights or [1.0] * len(constraints)
        self.current_step = 0
        self.history: List[Tuple[Construct, ...]] = []

        # Set self._generator_outputs to be a flat tuple of all ConstructSegment objects from all sub-generators
        self._generator_outputs = tuple(
            seq for gen in self.generators for seq in gen.get_generator_outputs()
        )  # Unused
        self._is_initialized = True

    def _validate_generator(self) -> None:
        """
        Validate that constructs, generators, constraints are properly configured.
        Must be called in final subclass __init__ to ensure all attributes are set.

        Raises:
            RuntimeError: If called before assign() has been called.
            ValueError: If any validation checks fail.
        """
        # Ensure basic generator validation
        super()._validate_generator()

        # Ensure constructs, generators, and constraints are non-empty lists
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        if not self.generators:
            raise ValueError("Generators list cannot be empty")
        if not self.constraints:
            raise ValueError("Constraints list cannot be empty")

        # Ensure constraint_weights are positive and finite
        invalid_weights = [
            w for w in self.constraint_weights if w <= 0 or not np.isfinite(w)
        ]
        if invalid_weights:
            raise ValueError(
                f"Constraint weights must be positive and finite. Found invalid weights: {invalid_weights}"
            )

        # Ensure constraint count matches weight count
        if len(self.constraints) != len(self.constraint_weights):
            raise ValueError(
                f"Constraint count ({len(self.constraints)}) must match weight count ({len(self.constraint_weights)})"
            )

        # Ensure types for all constructs, generators, and constraints are correct
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise ValueError(
                    f"Construct {i} has type {type(construct)}, expected Construct"
                )

        for i, generator in enumerate(self.generators):
            if not isinstance(generator, Generator):
                raise ValueError(
                    f"Generator {i} has type {type(generator)}, expected Generator"
                )

        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise ValueError(
                    f"Constraint {i} has type {type(constraint)}, expected Constraint"
                )

        # Ensure all generators are assigned construct segments
        if not all(generator._is_initialized for generator in self.generators):
            raise ValueError("Not all generators have been initialized.")

        # Ensure all construct segments are assigned to a generator
        unassigned_segments = [
            segment
            for construct in self.constructs
            for segment in construct.segments
            if not segment._is_assigned
        ]
        if unassigned_segments:
            raise ValueError(
                f"Found {len(unassigned_segments)} construct segments not assigned to any generator."
            )

        # Ensure all constraints have at least one generator-assigned input ConstructSegment
        generator_segment_ids = set(
            id(segment)
            for generator in self.generators
            for segment in generator.get_generator_outputs()
        )
        for i, constraint in enumerate(self.constraints):
            if not constraint.inputs:
                raise ValueError(f"Constraint {i} has no inputs assigned")
            if not any(id(inp) in generator_segment_ids for inp in constraint.inputs):
                raise ValueError(f"Constraint {i} has no generator-connected inputs")

    def _replicate_best_sequence(self, best_idx: int) -> None:
        """
        Copy the best sequence to all positions within each ConstructSegment.

        This helper method ensures that when a proposal is accepted, the sequence
        with the best energy is propagated to all positions within each batch.
        This is essential for maintaining consistency in constructs access.

        Args:
            best_idx: Index of the best sequence to propagate across all batches.

        Raises:
            RuntimeError: If called before assign() has been called.
            ValueError: If any batch has fewer sequences than best_idx.

        Note:
            This method modifies sequences in-place for all generator outputs.
        """
        # Get generator outputs (works with both single and multiple outputs)
        generator_outputs = self.get_generator_outputs()

        for construct_segment in generator_outputs:
            if len(construct_segment.batch_sequences) <= best_idx:
                raise ValueError(
                    f"ConstructSegment has only {len(construct_segment.batch_sequences)} sequences, "
                    f"cannot propagate best sequence at index {best_idx}"
                )

            best_sequence = construct_segment.batch_sequences[best_idx]
            # Propagate the best sequence to all positions in this ConstructSegment
            # TODO: Check if this propagation makes sense for top_k impelemenetation
            for sequence in construct_segment.batch_sequences:
                sequence.sequence = best_sequence.sequence
                sequence._metadata = best_sequence._metadata.copy()

    @abstractmethod
    def sample(self, **kwargs: Any) -> None:
        """
        Run one or more steps of iterative generation.

        Subclasses should implement this method to run the generation process.
        Implementations should modify generator outputs in-place and may store
        snapshots of constructs in `self.history`.

        Args:
            **kwargs: Keyword arguments for sampling (subclass-specific).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        IterativeGenerator doesn't support manual assignment.

        Raises:
            NotImplementedError: Always, as IterativeGenerator auto-initializes from pre-assigned generators.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} auto-initializes from pre-assigned generators. "
            "Manual assignment is not supported. Ensure all sub-generators are assigned before "
            "creating the IterativeGenerator."
        )

    def score_energy(self, operation: str = "add") -> List[float]:
        """
        Compute energy scores by combining constraint evaluation scores and store them in metadata.

        The energy function is computed as a weighted combination of all
        constraint scores. Lower energy values indicate better solutions.
        Results are automatically stored in sequence metadata.

        Args:
            operation: How to combine constraint scores across constraints:
                - 'add': Sum weighted constraint scores (default)
                - 'multiply': Multiply weighted constraint scores

        Returns:
            List of energy values, one per batch element. Lower values indicate
            better constraint satisfaction.

        Raises:
            ValueError: If generator is not properly initialized or operation is not 'add' or 'multiply'.
            RuntimeError: If called before assign() has been called.

        Note:
            The returned list length equals the batch size of the sequences.
            Energy computation uses current sequence values, so it reflects
            the most recent state after any sampling operations.
            Energy scores and time steps are automatically stored in metadata.
        """
        # Ensure generator is properly initialized
        self._validate_generator()

        # Get weighted scores from all constraints: shape (n_constraints, n_samples)
        # TODO: REVISIT HOW THIS BEHAVES WHEN SEGMENTS HAVE DIFFERENT BATCH SIZES
        constraint_scores = np.array(
            [
                np.array(constraint.evaluate()) * weight
                for constraint, weight in zip(self.constraints, self.constraint_weights)
            ]
        )

        # Combine across constraints for each sample
        if operation == "multiply":
            energies = np.prod(constraint_scores, axis=0)
        elif operation == "add":
            energies = np.sum(constraint_scores, axis=0)
        else:
            raise ValueError(f"Operation must be 'multiply' or 'add', got {operation}")

        # Update sequence metadata with respective energy_score and time_step
        energies_list = energies.tolist()
        
        for batch_idx, energy_score in enumerate(energies_list):
            metadata_update = {"energy_score": energy_score, "time_step": self.current_step}
            for construct in self.constructs:
                for segment in construct.segments:
                    segment.batch_sequences[batch_idx]._metadata.update(metadata_update)

        return energies_list
