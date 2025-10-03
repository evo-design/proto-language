"""
Constraint class for the proto-language.

Constraints define the objective function for construct optimization.
"""

from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections.abc import Iterable as AbcIterable

from .sequence import Sequence, SequenceType
from .segment import Segment
from ...utils.metadata import propagate_metadata


class ConstraintType(Enum):
    """Enumeration of constraint evaluation strategies for multiple inputs."""

    CONTIGUOUS = "contiguous"  # Concatenate sequences before evaluation
    DISJOINT = "disjoint"  # Evaluate sequences separately as a group


class InputMode(Enum):
    """Enumeration of inputs modes for constraints."""

    SINGLE = "single"  # Single input mode: sequence objects are passed to the scoring function one at a time
    MULTI = "multi"  # Multi input mode: a list of sequence objects is passed to the scoring function


class Constraint:
    """
    Constraints define the objective function for construct optimization.

    Constraints score how well segments satisfy biological or design requirements by
    taking in Segment objects and returning scores where lower values
    indicate better satisfaction of the constraint.

    Examples:
        Creating a length constraint:
        >>> def length_constraint(sequence, config):
        ...     target = config['target_length']
        ...     return abs(len(sequence) - target) / target
        >>> segment = Segment("ATCG", SequenceType.DNA)
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
        inputs: Iterable[Segment],
        scoring_function: Callable[[Sequence | Tuple[Sequence], Dict[str, Any]], float],
        scoring_function_config: Optional[Dict[str, Any]] = None,
        input_mode: Optional[Union[InputMode, str]] = InputMode.SINGLE,
        constraint_type: Optional[
            Union[ConstraintType, str]
        ] = ConstraintType.CONTIGUOUS,
        label: Optional[str] = None,
    ) -> None:
        """
        Initialize a constraint with its inputs and scoring function.

        Args:
            inputs: The Segment object(s) this constraint evaluates.
                Can be a single Segment or an iterable of Segment objects. Each
                segment object can contain a batch of sequences, but all segments
                must have the same batch size.
            scoring_function: Function that scores sequences.
                Inputs are single sequence for contiguous constraints or tuple of sequences for disjoint constraints.
                Outputs are a float score between 0.0 and 1.0. Lower values are better.
            scoring_function_config: Configuration parameters passed to scoring_function.
            input_mode: How to pass sequence objects to the scoring function:
                - SINGLE: Single input mode: sequence objects are passed to the scoring function one at a time.
                    The scoring function should expect a single sequence object in it's first parameter. (This is the default mode.)
                - MULTI: Multi input mode: a list of sequence objects is passed to the scoring function.
                    The scoring function should expect a list of sequence objects in it's first parameter. This is helpful for
                    scoring functions that can batch inference for multiple sequences at once (such as functions that involve
                    forward passes through a model)
            constraint_type: How multiple sequences within a segment batch position are processed:
                - CONTIGUOUS: Concatenate sequences before evaluation.
                - DISJOINT: Evaluate sequences separately as a group.
            label: Optional custom label for this constraint. If not provided, defaults to scoring function name.
                   Prefixed to prevent metadata collisions.
        """
        self.inputs: Tuple[Segment] = tuple(inputs)
        self.scoring_function: Callable[
            [Sequence | Tuple[Sequence], Dict[str, Any]], float
        ] = scoring_function
        self.scoring_function_config: Dict[str, Any] = self._normalize_config(scoring_function_config)
        self.constraint_type: ConstraintType = ConstraintType(constraint_type)
        self.label: str = label or scoring_function.__name__
        self.input_mode: InputMode = InputMode(input_mode)

        # Validate input consistency and store common properties
        self._validate_input_consistency(self.inputs)
        self.sequence_type = self.inputs[0].sequence_type
        self.valid_chars = self.inputs[0]._valid_chars

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
            from ...schemas import ESMFoldKwargs, ORFipyKwargs, MMseqsKwargs
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
            for batch_position in range(self.inputs[0].batch_size):
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
            for batch_idx in range(self.inputs[0].batch_size):
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
        Evaluate each Sequence object in the Segment batch.

        Returns:
            List of scores between 0.0 and 1.0, one per Sequence object in the batch.
            Returns float('inf') for invalid sequences.
        """

        # Preprocess inputs to accommodate scoring function
        scoring_function_inputs = self._process_inputs()

        # Score all inputs
        scores = []
        if self.input_mode == InputMode.MULTI:
            # Multi-input scoring function: pass in list of inputs
            scores = self.scoring_function(
                scoring_function_inputs, **self.scoring_function_config
            )
        else:
            # Single-input scoring function: pass in each input separately
            scores = [
                self.scoring_function(input, **self.scoring_function_config)
                for input in scoring_function_inputs
            ]

        # Propagate metadata back with prefixing
        for i, input in enumerate(scoring_function_inputs):

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

    def _validate_input_consistency(self, inputs: Tuple[Segment]) -> None:
        """
        Validate that all input segments have consistent properties.

        Args:
            inputs: Tuple of Segment objects to validate.

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
