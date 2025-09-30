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
from typing import Callable, List, Tuple, Dict, Any, Set, Optional, Iterator, Iterable, Union, final
import warnings
import numpy as np
import copy
from .utils import propagate_metadata, SequenceType, ConstraintType


@final
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
        elif self.sequence_type == SequenceType.LIGAND:
            self._valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'- ")
        else:
            raise ValueError(f"Unsupported sequence_type: {self.sequence_type}")

        self._validate_sequence(sequence)
        self._sequence: str = sequence
        self._metadata = {}
        protected_metadata = {
            "sequence": sequence,
            "sequence_length": len(sequence),
        }
        
        # Add user metadata, warning if they try to override protected keys
        if metadata:
            conflicting_keys = [key for key in metadata if key in protected_metadata]
            if conflicting_keys:
                warnings.warn(
                    f"System-managed metadata for {conflicting_keys} cannot be manually set and will be silently overridden",
                    UserWarning,
                    stacklevel=2
                )
            self._metadata.update(metadata)
        self._metadata.update(protected_metadata)

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

    @staticmethod
    def from_sequences(
        subsequences: List['Sequence'],
        merge_metadata: bool = False
    ) -> 'Sequence':
        """
        Create a sequence by joining subsequences with optional metadata propagation.
        
        This alternative constructor joins subsequences and optionally merges
        their metadata with sequence label prefixing to avoid key collisions.
        
        Args:
            subsequences: List of Sequence objects to join
            merge_metadata: If True, merge non-system metadata; if False, start clean
            
        Returns:
            Single joined Sequence object with only system metadata (if merge_metadata=False)
            or with merged non-system metadata (if merge_metadata=True)
            
        Example:
            >>> sequences = [Seq("ATG"), Seq("CCC")]
            >>> clean_seq = Sequence.from_sequences(sequences, merge_metadata=False)
            >>> # Returns Seq("ATGCCC") with only system metadata
        """
        combined_sequence_string = "".join(sequence.sequence for sequence in subsequences)
        combined_metadata = {}
        
        if merge_metadata:
            for sequence in subsequences:
                # Only propagate non-system metadata (no prefix needed)
                propagate_metadata(sequence._metadata, combined_metadata)
        
        return Sequence(
            sequence=combined_sequence_string,
            sequence_type=subsequences[0].sequence_type, # assumed to be the same for all subsequences
            valid_chars=subsequences[0]._valid_chars, # assumed to be the same for all subsequences
            metadata=combined_metadata
        )

    @property 
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata dictionary with consistent ordering.
        
        Returns:
            Dict with system keys first, then constraint keys in chronological order.
        """
        system_keys = ["sequence", "sequence_length"]
        
        return {
            **{k: self._metadata[k] for k in system_keys if k in self._metadata},  # System keys first
            **{k: v for k, v in self._metadata.items() if k not in set(system_keys)}    # Constraint keys
        }


class ConstructSegment:
    """
    External class that represents the building blocks for a Construct.

    This is the most modular user-facing unit for the programming language.

    Examples:
        Creating a ConstructSegment:
        >>> promoter = ConstructSegment(sequence="TATA", sequence_type=SequenceType.DNA, label="promoter")
        >>> promoter.label  # "promoter"
    """

    def __init__(
        self,
        sequence: str = "",
        sequence_type: SequenceType = SequenceType.DNA,
        valid_chars: Optional[Set[str]] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a ConstructSegment with a single sequence.

        Args:
            sequence: The biological sequence string. Defaults to empty string.
            sequence_type: Type of biological sequence (DNA, RNA, or PROTEIN). Defaults to DNA.
            valid_chars: Optional custom set of valid characters for sequence validation.
            metadata: Additional data associated with this sequence.
            label: Optional label for this segment (e.g., "promoter", "coding_region").
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
        self.label: Optional[str] = label

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
        Creating a construct from labeled segments:
        >>> promoter = ConstructSegment("TATA", SequenceType.DNA, label="promoter")
        >>> coding = ConstructSegment("ATGCCC", SequenceType.DNA, label="coding_region")
        >>> terminator = ConstructSegment("TTTT", SequenceType.DNA, label="terminator")
        >>> gene = Construct([promoter, coding, terminator])
        >>> gene.batch_sequences  # [Sequence("TATAATGCCCTTTT", SequenceType.DNA)]
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
                sequence types, segments have different valid characters, or segments have different batch sizes.
        """
        # Convert to tuple for validation and storage
        self.segments: Tuple[ConstructSegment, ...] = tuple(segments)

        # Any unlabeled segments will be labeled as segment_i
        for i, segment in enumerate(self.segments):
            if segment.label is None:
                segment.label = f"segment_{i}"

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
        
        # Ensure consistent batch sizes across all segments
        batch_sizes = [len(segment.batch_sequences) for segment in self.segments]
        if not all(size == batch_sizes[0] for size in batch_sizes):
            raise ValueError(
                f"Inconsistent batch sizes across construct segments. Found: {batch_sizes}. "
                f"All segments must have the same batch size."
            )

        self.sequence_type: SequenceType = self.segments[0].sequence_type
        self._valid_chars: Optional[Set[str]] = self.segments[0]._valid_chars

    @property
    def batch_sequences(self) -> Tuple[Sequence, ...]:
        """
        Get the joined Sequence objects batch that represent one user-facing Construct.

        Returns:
            Tuple of joined Sequence objects where each element represents
            the joining of the i-th sequence from each segment in order.
        """
        # Join corresponding sequences from each segment with metadata propagation
        # Example: [Seq("AAA"), Seq("TTT"), Seq("GGG")] → [Sequence("AAATTTGGG")]
        joined_sequences = []
        batch_size = len(self.segments[0].batch_sequences)
        
        for batch_position in range(batch_size):
            sequences_to_combine = [segment.batch_sequences[batch_position] for segment in self.segments]
            joined_seq = Sequence.from_sequences(
                subsequences=sequences_to_combine,
                merge_metadata=True
            )
                
            joined_sequences.append(joined_seq)
            
        return tuple(joined_sequences)


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
        ...     label='length_constraint'
        ... )
        >>> constraint.evaluate()  # [0.0]
    """

    def __init__(
        self,
        inputs: Iterable[ConstructSegment],
        scoring_function: Callable[[Sequence | Tuple[Sequence], Dict[str, Any]], float],
        scoring_function_config: Dict[str, Any] = {},
        constraint_type: ConstraintType = ConstraintType.CONTIGUOUS,
        label: Optional[str] = None,
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
            label: Optional custom label for this constraint. If not provided, defaults to scoring function name. 
                   Prefixed to prevent metadata collisions.
        """
        self.inputs: Tuple[ConstructSegment] = tuple(inputs)
        self.scoring_function: Callable[
            [Sequence | Tuple[Sequence], Dict[str, Any]], float
        ] = scoring_function
        self.scoring_function_config: Dict[str, Any] = self._normalize_config(scoring_function_config)
        self.constraint_type: ConstraintType = constraint_type
        self.label: str = label or scoring_function.__name__
        
        # Validate input consistency and store common properties
        self._validate_input_consistency(self.inputs)
        self.sequence_type = self.inputs[0].sequence_type
        self.valid_chars = self.inputs[0]._valid_chars
        self.batch_size = len(self.inputs[0].batch_sequences)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically convert dict configs to Pydantic models where needed.
        
        This handles the conversion for constraint configurations that expect Pydantic models
        for their kwargs parameters but receive dictionaries from JSON parsing or direct calls.
        
        Args:
            config: Configuration dictionary that may contain nested dicts for Pydantic models
            
        Returns:
            Normalized configuration with dicts converted to Pydantic models where appropriate
        """
        if not config:
            return {}
        
        # Import here to avoid circular imports
        try:
            from .schemas import ESMFoldKwargs, ORFipyKwargs, MMseqsKwargs
        except ImportError:
            # If schemas aren't available, return config unchanged
            return config.copy()
        
        normalized = config.copy()
        
        # Define the mapping of config keys to their Pydantic models
        PYDANTIC_MAPPINGS = {
            'esmfold_kwargs': ESMFoldKwargs,
            'orfipy_kwargs': ORFipyKwargs, 
            'mmseqs_kwargs': MMseqsKwargs,
        }
        
        for key, value in normalized.items():
            if key in PYDANTIC_MAPPINGS and isinstance(value, dict):
                try:
                    normalized[key] = PYDANTIC_MAPPINGS[key](**value)
                except Exception:
                    # If conversion fails, leave as dict (backward compatibility)
                    pass
                    
        return normalized

    def _process_inputs(self) -> List[Sequence] | List[Tuple[Sequence, ...]]:
        """
        Processes segments by constraint type to accommodate scoring function inputs.

        Returns:
            For CONTIGUOUS: List of Sequence objects with joined sequences.
            For DISJOINT: List of tuples, each containing corresponding Sequence objects.
        """
        processed_inputs = []

        # CONTIGUOUS CASE: Join corresponding sequences from each segment into a single dummy Sequence object
        # Example: [Seq("AAA"), Seq("TTT"), Seq("GGG")] → [Sequence("AAATTTGGG")]
        # Note: Metadata isn't propagated to dummy sequences since scoring functions won't use it.
        if self.constraint_type == ConstraintType.CONTIGUOUS:
            # Join sequences without metadata propagation (dummy sequences for scoring)
            for batch_position in range(self.batch_size):
                sequences_to_combine = [segment.batch_sequences[batch_position] for segment in self.inputs] # [Sequence("A"), Sequence("T"), Sequence("C"), Sequence("G")]
                dummy_seq = Sequence.from_sequences(
                    subsequences=sequences_to_combine,
                    merge_metadata=False  # Clean metadata - only basic system keys
                )
                processed_inputs.append(dummy_seq)
        # DISJOINT CASE: Group corresponding sequences from each segment as tuples
        # Example: segment_0=[Seq("AAA"), Seq("TTT"), Seq("GGG")], segment_1=[Seq("CCC"), Seq("AAC"), Seq("TTC")] 
        #          → [(Seq("AAA"), Seq("CCC")), (Seq("TTT"), Seq("AAC")), (Seq("GGG"), Seq("TTC"))]
        # Note: Create dummy sequences without old metadata, similar to CONTIGUOUS case
        elif self.constraint_type == ConstraintType.DISJOINT:
            for batch_idx in range(self.batch_size):
                # Create clean sequences for scoring (without old metadata)
                clean_sequences = []
                for segment in self.inputs:
                    original_seq = segment.batch_sequences[batch_idx]
                    # Create clean sequence with only system metadata
                    dummy_seq = Sequence(
                        sequence=original_seq.sequence,
                        sequence_type=original_seq.sequence_type,
                        valid_chars=original_seq._valid_chars
                    )
                    clean_sequences.append(dummy_seq)
                processed_inputs.append(tuple(clean_sequences))

        return processed_inputs
    
    def evaluate(self) -> List[float]:
        """
        Evaluate each Sequence object in the ConstructSegment batch.

        Returns:
            List of scores between 0.0 and 1.0, one per Sequence object in the batch.
            Returns float('inf') for invalid sequences.
        """
        scores = []
        
        # Preprocess inputs to accommodate scoring function
        scoring_function_inputs = self._process_inputs()
            
        # Score all inputs and propagate metadata back with prefixing
        for i, input in enumerate(scoring_function_inputs):
            scores.append(self.scoring_function(input, **self.scoring_function_config)) # this adds metadata to the dummy input sequence

            if self.constraint_type == ConstraintType.CONTIGUOUS:
                # Create segment labels string (e.g. promoter-cds-terminator)
                segment_labels = [seg.label or f"segment_{idx}" for idx, seg in enumerate(self.inputs)]
                segments_str = "-".join(segment_labels)
                
                for segment in self.inputs:
                    original_seq = segment.batch_sequences[i]
                    propagate_metadata(
                        source_metadata=input._metadata,
                        target_metadata=original_seq._metadata,
                        prefix=f"{segments_str}.{self.label}"
                    )

            elif self.constraint_type == ConstraintType.DISJOINT:
                # For DISJOINT: input is a tuple of Sequence objects, propagate metadata from each
                # sequence in the tuple back to the corresponding original sequence
                for j, scored_sequence in enumerate(input):  # input is tuple of sequences
                    original_seq = self.inputs[j].batch_sequences[i]  # Get original sequence from j-th segment
                    segment_label = self.inputs[j].label or f"segment_{j}"
                    
                    propagate_metadata(
                        source_metadata=scored_sequence._metadata,
                        target_metadata=original_seq._metadata,
                        prefix=f"{segment_label}.{self.label}"
                    )
                
        return scores
    
    def _validate_input_consistency(self, inputs: Tuple[ConstructSegment]) -> None:
        """
        Validate that all input segments have consistent properties.

        Args:
            inputs: Tuple of ConstructSegment objects to validate.

        Raises:
            ValueError: If inputs have inconsistent sequence_type, valid_chars, or batch sizes.
        """
        # Check sequence_type consistency
        if not all(
            input_batch.sequence_type == inputs[0].sequence_type
            for input_batch in inputs
        ):
            all_types = {input_batch.sequence_type for input_batch in inputs}
            raise ValueError(
                f"Inconsistent sequence_type across inputs. Found: {all_types}"
            )
            
        # Check valid_chars consistency
        if not all(
            input_batch._valid_chars == inputs[0]._valid_chars
            for input_batch in inputs
        ):
            raise ValueError("Inconsistent valid_chars across inputs.")
            
        # Check batch size consistency
        batch_sizes = [len(input_batch.batch_sequences) for input_batch in inputs]
        if not all(size == batch_sizes[0] for size in batch_sizes):
            raise ValueError(
                f"Inconsistent batch sizes across inputs. Found: {batch_sizes}. "
                f"All input segments must have the same batch size."
            )


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
        self.iteration_count: int = 0

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

    def get_iteration_count(self) -> int:
        """
        Get the number of times sample() has been invoked since the last reset.

        Returns:
            Current iteration count as an integer.
        """
        return self.iteration_count

    def reset_iteration_count(self) -> None:
        """
        Reset the internal iteration counter to zero.
        """
        self.iteration_count = 0

    def _increment_iteration_count(self) -> None:
        """
        Protected helper to increment the iteration counter by one.

        Subclasses should call this at the end of their sample() implementations
        if they want to use the shared iteration counter and any schedulers
        that depend on it.
        """
        self.iteration_count += 1


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
        self.history: List[Dict[str, Any]] = []  # Each entry: {"time_step": int, "energy_scores": List[float], "constructs": List[Construct]}
        self.energy_scores: List[float] = []  # Each index corresponds to a batch element, empty until first score_energy() call

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

    def score_energy(self, operation: str = "add") -> None:
        """
        Compute energy scores by combining constraint evaluation scores
        Energy scores are stored in self.energy_scores.

        The energy function is computed as a weighted combination of all
        constraint scores. Lower energy values indicate better solutions.

        Args:
            operation: How to combine constraint scores across constraints:
                - 'add': Sum weighted constraint scores (default)
                - 'multiply': Multiply weighted constraint scores

        Raises:
            ValueError: If generator is not properly initialized or operation is not 'add' or 'multiply'.
            RuntimeError: If called before assign() has been called.

        Note:
            Energy computation uses current sequence values, so it reflects
            the most recent state after any sampling operations. The computed
            energy scores are accessible via self.energy_scores.
        """
        # Ensure generator is properly initialized
        self._validate_generator()

        # Get weighted scores from all constraints: shape (n_constraints, n_samples)
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

        energies_list = energies.tolist()
        self.energy_scores = energies_list
    
    def append_snapshot_to_history(self) -> None:
        """Save snapshot of current construct state and energy scores to history."""
        # Store as structured history entry with separate metadata
        history_entry = {
            "time_step": self.current_step,
            "energy_scores": self.energy_scores.copy(),
            "constructs": copy.deepcopy(self.constructs)
        }
        
        self.history.append(history_entry)
