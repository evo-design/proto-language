"""
Constraint class for biological programming language.

Constraints score how well sequences satisfy biological or design requirements,
returning values between 0.0 (perfect) and 1.0 (worst).

Key Features:
    - Batch evaluation (sequential or vectorized processing)
    - Segment coordination (contiguous concatenation or disjoint evaluation)
    - Automatic metadata propagation back to original sequences
"""

from typing import Callable, List, Optional, Tuple, Union, Dict, Any, get_type_hints, Type

from pydantic import BaseModel

from .sequence import Sequence
from .segment import Segment
from ...utils.helpers import propagate_metadata


class Constraint:
    """
    Constraints handle batching, metadata propagation, and evaluation of sequences.
    
    Examples:
        Library Usage (Direct instantiation):
        >>> from proto_language.language.core import Constraint
        >>> from proto_language.language.constraint import gc_content_constraint, GCContentConfig
        >>> 
        >>> config = GCContentConfig(min_gc=40, max_gc=60)
        >>> constraint = Constraint(
        ...     inputs=[dna_segment],
        ...     scoring_function=gc_content_constraint,
        ...     scoring_function_config=config
        ... )
        >>> scores = constraint.evaluate()  # [0.0, 0.1, ...]
        
        API/Client Usage (Registry for discovery):
        >>> from proto_language.language.constraint import ConstraintRegistry
        >>> 
        >>> # List available constraints
        >>> all_constraints = ConstraintRegistry.list_all()
        >>> 
        >>> # Get schema for client form generation
        >>> schema = ConstraintRegistry.get_schema("gc_content")
        >>> 
        >>> # Create from user input (dict from client)
        >>> constraint = ConstraintRegistry.create(
        ...     key="gc_content",
        ...     segments=[dna_segment],
        ...     config_dict={"min_gc": 40, "max_gc": 60}
        ... )
    """
    
    def __init__(
        self,
        inputs: List[Segment],
        scoring_function: Callable,
        scoring_function_config: Union[BaseModel, Dict[str, Any]],
        vectorized: bool = False,
        concatenate: bool = True,
        label: Optional[str] = None,
    ):
        """
        Initialize a constraint.
        
        Args:
            inputs: List of Segment objects to evaluate
            scoring_function: The constraint scoring function that returns scores between 0.0-1.0.
                For sequential mode: (Sequence, config=ConfigModel) -> float or (Tuple[Sequence, ...], config=ConfigModel) -> float
                For vectorized mode: (List[Sequence], config=ConfigModel) -> List[float] or (List[Tuple[Sequence, ...]], config=ConfigModel) -> List[float]
            scoring_function_config: Configuration parameters. Can be either:
                Pydantic BaseModel instance (recommended)
                Dict that will be converted to the appropriate config model
            vectorized: If True, pass all sequences at once for batch processing.
                If False, evaluate sequences one at a time.
            concatenate: If True, concatenate multiple segments before evaluation. If False, pass segments separately (tuple of sequences).
            label: Optional label for metadata tracking. Defaults to the name of the scoring function.
        """
        self.inputs = inputs
        self.scoring_function = scoring_function
        self.vectorized = vectorized
        self.concatenate = concatenate
        self.label = label or scoring_function.__name__
        
        # Convert dict configs to Pydantic models for validation
        if isinstance(scoring_function_config, dict):
            config_class = self._try_extract_config_class(scoring_function)
            self.scoring_function_config = (
                config_class(**scoring_function_config) if config_class 
                else scoring_function_config
            )
        else:
            self.scoring_function_config = scoring_function_config
        
        # Validate inputs
        self._validate_inputs()
    
    def evaluate(self) -> List[float]:
        """
        Evaluate the constraint on all candidate sequences.

        This method orchestrates the evaluation by:
        1. Extracting candidate sequences from input segments
        2. Calling the scoring function (vectorized or sequential)
        3. Propagating scores back to original candidate sequence metadata

        Returns:
            List of scores (0.0-1.0), one per candidate.
            Lower scores are better (0.0 = perfect).
        """
        num_candidates = self.inputs[0].num_candidates

        if self.vectorized:
            # Vectorized mode: process all candidates at once
            sequences_vector = [self._preprocess_candidate_at_index(candidate_idx) for candidate_idx in range(num_candidates)]
            scores = self.scoring_function(sequences_vector, config=self.scoring_function_config)

            # Propagate metadata from each scored sequence back to originals
            for candidate_idx, scored_seq in enumerate(sequences_vector):
                self._propagate_metadata_to_candidate(candidate_idx, scored_seq)

            return scores
        else:
            # Sequential mode: process one candidate at a time
            scores = []
            for candidate_idx in range(num_candidates):
                sequence = self._preprocess_candidate_at_index(candidate_idx)
                score = self.scoring_function(sequence, config=self.scoring_function_config)
                scores.append(score)

                # Propagate metadata from scored sequence back to original
                self._propagate_metadata_to_candidate(candidate_idx, sequence)

            return scores
    
    def _preprocess_candidate_at_index(self, candidate_idx: int) -> Sequence | Tuple[Sequence, ...]:
        """
        Preprocess sequence(s) at a specific batch position for scoring by creating dummy Sequence 
        objects with clean metadata to pass to the scoring function.

        Args:
            candidate_idx: Index position in the candidate pool (0-based)

        Returns:
            If concatenate=True: Sequence - merged Sequence object from all segments (contiguous)
            If concatenate=False: Tuple[Sequence, ...] - tuple of clean Sequence objects (disjoint)
        """
        if self.concatenate:
            # CONTIGUOUS: Merge all segments into single Sequence object
            # Example: candidate_idx=0, segments with candidates=[Seq("AAA"), ...], [Seq("CCC"), ...]
            #          → Sequence("AAACCC")
            return Sequence.from_sequences(
                subsequences=[seg.candidate_sequences[candidate_idx] for seg in self.inputs],
                merge_metadata=False  # Clean metadata - only basic system keys
            )
        else:
            # DISJOINT: Return tuple of clean Sequence objects
            # Example: candidate_idx=0, segments with candidates=[Seq("AAA"), ...], [Seq("CCC"), ...]
            #          → (Seq("AAA"), Seq("CCC"))
            dummy_sequences = []
            for seg in self.inputs:
                original = seg.candidate_sequences[candidate_idx]
                # Create clean Sequence with only essential properties
                dummy_seq = Sequence(
                    sequence=original.sequence,
                    sequence_type=original.sequence_type,
                    valid_chars=original._valid_chars
                )
                dummy_sequences.append(dummy_seq)
            return tuple(dummy_sequences)
    
    def _propagate_metadata_to_candidate(self, candidate_idx: int, scored_sequence: Sequence | Tuple[Sequence, ...]) -> None:
        """
        Write constraint results back to original candidate sequence metadata.

        Extracts metadata from the scored Sequence object(s) and propagates it back to
        the original candidate sequences in the input segments. Metadata keys are prefixed
        with segment labels and constraint name to prevent collisions.

        Args:
            candidate_idx: Index position in the candidate pool (0-based)
            scored_sequence: The Sequence (or tuple of Sequences) that was scored,
                           containing metadata written by the scoring function
        """
        if self.concatenate:
            # For contiguous: propagate from single scored Sequence to all original segments
            # Create combined label from all segments
            segment_labels = [seg.label or f"segment_{i}" for i, seg in enumerate(self.inputs)]
            combined_label = "-".join(segment_labels)
            prefix = f"{combined_label}.{self.label}"

            for segment in self.inputs:
                original_seq = segment.candidate_sequences[candidate_idx]
                propagate_metadata(
                    source_metadata=scored_sequence._metadata,
                    target_metadata=original_seq._metadata,
                    prefix=prefix
                )
        else:
            # For disjoint: propagate from each scored Sequence to its corresponding original
            # scored_sequence is a tuple of Sequences, one per segment
            for seg_idx, (segment, scored_seq) in enumerate(zip(self.inputs, scored_sequence)):
                original_seq = segment.candidate_sequences[candidate_idx]
                segment_label = segment.label or f"segment_{seg_idx}"
                prefix = f"{segment_label}.{self.label}"

                propagate_metadata(
                    source_metadata=scored_seq._metadata,
                    target_metadata=original_seq._metadata,
                    prefix=prefix
                )
                
    def _try_extract_config_class(self, func: Callable) -> Optional[Type[BaseModel]]:
        """
        Extract config class from function's type hints, returning None if unavailable.
        
        Supports production constraints with proper type hints and test mocks without them.
        """
        try:
            hints = get_type_hints(func)
            if 'config' not in hints:
                return None
            
            config_type = hints['config']
            
            # Handle Union types (e.g., Optional[ConfigClass])
            if hasattr(config_type, '__origin__'):
                args = getattr(config_type, '__args__', ())
                config_type = next((arg for arg in args if arg is not type(None)), config_type)
            
            return config_type if isinstance(config_type, type) else None
        except Exception:
            return None
    
    def _validate_inputs(self) -> None:
        """Validate that all input segments have consistent candidate pool sizes and sequence types."""
        if not self.inputs:
            raise ValueError("At least one segment must be provided")

        # Check that all segments have the same number of candidates
        candidate_sizes = [seg.num_candidates for seg in self.inputs]
        if not all(size == candidate_sizes[0] for size in candidate_sizes):
            raise ValueError(
                f"All segments must have the same number of candidates. Found: {candidate_sizes}"
            )

        # Check that all segments have the same sequence type
        sequence_types = [seg.sequence_type for seg in self.inputs]
        if not all(st == sequence_types[0] for st in sequence_types):
            raise ValueError(
                f"All segments must have the same sequence type. Found: {sequence_types}"
            )

        # Check that all segments have the same valid_chars (alphabet)
        first_valid_chars = self.inputs[0]._valid_chars
        for seg in self.inputs[1:]:
            if seg._valid_chars != first_valid_chars:
                raise ValueError(
                    f"All segments must have the same valid_chars. "
                    f"Expected: {first_valid_chars}, Found: {seg._valid_chars}"
                )
