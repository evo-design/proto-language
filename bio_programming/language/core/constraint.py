"""
Constraint class for biological programming language.

Constraints score how well sequences satisfy biological or design requirements,
returning values between 0.0 (perfect) and 1.0 (worst). Constraints can optionally
act as filters by providing a threshold parameter.

Key Features:
    - Batch evaluation (sequential or batched processing)
    - Segment coordination (contiguous concatenation or disjoint evaluation)
    - Automatic metadata propagation back to original sequences
    - Threshold-based filtering (converts scores to boolean accept/reject)
"""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict, Any

from pydantic import BaseModel

from .sequence import Sequence
from .segment import Segment
from proto_language.utils.helpers import propagate_metadata


class Constraint:
    """
    Constraints handle batching, metadata propagation, and evaluation of sequences.

    Examples (Library Usage):
        >>> from proto_language.language.core import Constraint
        >>> from proto_language.language.constraint import gc_content_constraint, GCContentConfig
        >>>
        >>> config = GCContentConfig(min_gc=40, max_gc=60)
        >>> constraint = Constraint(
        ...     inputs=[dna_segment],
        ...     function=gc_content_constraint,
        ...     function_config=config
        ... )
        >>> scores = constraint.evaluate()  # [0.0, 0.1, ...]
        >>>
        >>> # Use as a filter by adding threshold
        >>> filter_constraint = Constraint(
        ...     inputs=[dna_segment],
        ...     function=gc_content_constraint,
        ...     function_config=config,
        ...     threshold=0.5
        ... )
        >>> passed = filter_constraint.evaluate()  # [True, False, True, ...]

        API/Client Usage (Registry for discovery):
        >>> from proto_language.language.constraint import ConstraintRegistry
        >>>
        >>> # List available constraints
        >>> all_constraints = ConstraintRegistry.list_all()
        >>>
        >>> # Get schema for client form generation
        >>> schema = ConstraintRegistry.get_schema("gc_content")
        >>>
        >>> # Create from user input (dict from client) - scoring mode
        >>> constraint = ConstraintRegistry.create(
        ...     key="gc_content",
        ...     segments=[dna_segment],
        ...     config_dict={"min_gc": 40, "max_gc": 60}
        ... )
        >>>
        >>> # Create as filter by adding threshold
        >>> filter_constraint = ConstraintRegistry.create(
        ...     key="gc_content",
        ...     segments=[dna_segment],
        ...     config_dict={"min_gc": 40, "max_gc": 60},
        ...     threshold=0.5
        ... )
    """

    def __init__(
        self,
        inputs: List[Segment],
        function: Callable,
        function_config: BaseModel | Dict[str, Any],
        label: Optional[str] = None,
        threshold: Optional[float] = None,
        weight: Optional[float] = None,
    ):
        """
        Initialize a constraint.

        Args:
            inputs: List of Segment objects to evaluate
            function: The constraint scoring function that returns scores between 0.0-1.0.
                For sequential mode: (Sequence, config=ConfigModel) -> float or (Tuple[Sequence, ...], config=ConfigModel) -> float
                For batched mode: (List[Sequence], config=ConfigModel) -> List[float] or (List[Tuple[Sequence, ...]], config=ConfigModel) -> List[float]
            function_config: Configuration as Pydantic BaseModel or dict (auto-converted to BaseModel)
            label: Optional label for metadata tracking. Defaults to function.__name__
            threshold: Optional threshold for filtering mode. If provided, scores <= threshold are accepted (True),
                scores > threshold are rejected (False). If None, returns raw float scores.
                Mutually exclusive with ``weight`` (setting both raises a ValueError).
            weight: Optional weight to multiply the raw constraint score by. Defaults to 1.0 if not provided.
                Only meaningful for scoring constraints (when threshold is None).
                Mutually exclusive with ``threshold`` (setting both raises a ValueError).
        """
        self.inputs = inputs
        self.function = function
        self.label = label or function.__name__

        if threshold is not None and weight is not None:
            raise ValueError(f"Both threshold ({threshold}) and weight ({weight}) are set, cannot weigh a boolean threshold")
        weight = 1.0 if weight is None else weight
        self.threshold = threshold
        self.weight = weight

        # Read metadata from function attributes (set by registry decorator)
        self.batched = function._constraint_batched
        self.concatenate = function._constraint_concatenate

        # Convert dict configs to Pydantic models for validation
        if isinstance(function_config, dict):
            config_class = function._constraint_config_class
            self.function_config = config_class(**function_config)
        else:
            self.function_config = function_config

        # Validate inputs
        self._validate_inputs()
        self._validate_sequence_types()

    def evaluate(
        self,
        mask: Optional[List[bool]] = None,
        verbose: bool = False
    ) -> List[float] | List[bool]:
        """
        Evaluate the constraint on candidates.

        This method orchestrates the evaluation:
        1. Extract candidate sequences from input segments (only those that passed)
        2. Call the scoring function (batched or sequential)
        3. Propagate scores back to original candidate sequence metadata
        4. Convert scores to boolean filters if threshold is set, or apply weight if not
        5. Build a dense result array (one entry per candidate)

        Args:
            mask: Boolean mask indicating which candidates to evaluate. If None, evaluates all.
            verbose: If true, logs evaluation details.

        Returns:
            List of results.
            - Filter constraints: False for unevaluated candidates
            - Scoring constraints: 0.0 for unevaluated candidates
        """
        num_candidates = self.inputs[0].num_candidates

        # Default: evaluate all candidates
        if mask is None:
            mask = [True] * num_candidates
        if len(mask) != num_candidates:
            raise ValueError(f"Mask length ({len(mask)}) must match number of candidates ({num_candidates})")

        # Convert mask to indices for sparse evaluation
        indices_to_evaluate = [i for i in range(num_candidates) if mask[i]]

        # Early return if no candidates to evaluate
        if not indices_to_evaluate:
            return [float('nan')] * num_candidates if self.threshold is None else [False] * num_candidates

        # Evaluate candidates at specified indices only for performance
        if self.batched:
            # Batched mode: evaluate all sequences in one batch
            indexed_sequences = [(idx, self._preprocess_sequence_at_index(idx)) for idx in indices_to_evaluate] # (original_idx, sequence) pairs
            sequences_to_evaluate = [seq for _, seq in indexed_sequences]                           # list of raw sequences to evaluate
            raw_scores = self.function(sequences_to_evaluate, config=self.function_config)          # list of raw scores


            for (original_idx, scored_seq) in indexed_sequences:
                self._propagate_metadata_to_sequence(original_idx, scored_seq)
        else:
            # Sequential mode: evaluate one at a time
            raw_scores = []
            for idx in indices_to_evaluate:
                sequence = self._preprocess_sequence_at_index(idx)
                score = self.function(sequence, config=self.function_config)
                raw_scores.append(score)
                self._propagate_metadata_to_sequence(idx, sequence)

        # Rebuild dense result array. Skipped candidates get NaN (scoring) or False (filter)
        if self.threshold is None:
            # Scoring constraint: apply weight to raw scores
            final_scores = [float('nan')] * num_candidates
            for j, idx in enumerate(indices_to_evaluate):
                final_scores[idx] = raw_scores[j] * self.weight
        else:
            # Filter constraint: convert scores to boolean (pass if score <= threshold)
            final_scores = [False] * num_candidates
            for j, idx in enumerate(indices_to_evaluate):
                final_scores[idx] = raw_scores[j] <= self.threshold

        if verbose:
            evaluated_set = set(indices_to_evaluate)
            for i in range(num_candidates):
                if i in evaluated_set:
                    j = indices_to_evaluate.index(i)
                    if self.threshold is None:
                        print(f"  Candidate {i}: {final_scores[i]:.4f} = {raw_scores[j]:.4f} * {self.weight}")
                    else:
                        print(f"  Candidate {i}: {'PASS' if final_scores[i] else 'FAIL'} ({raw_scores[j]:.4f})")
                else:
                    print(f"  Candidate {i}: SKIPPED")

        return final_scores

    def _preprocess_sequence_at_index(self, sequence_idx: int) -> Sequence | Tuple[Sequence, ...]:
        """
        Preprocess sequence(s) at a specific batch position for scoring by creating dummy Sequence
        objects with clean metadata to pass to the scoring function.

        Args:
            sequence_idx: Index position in the sequence pool (0-based)

        Returns:
            If concatenate=True: Sequence - merged Sequence object from all segments (contiguous)
            If concatenate=False: Tuple[Sequence, ...] - tuple of clean Sequence objects (disjoint)
        """
        if self.concatenate:
            # CONTIGUOUS: Merge all segments into single Sequence object
            # Example: sequence_idx=0, segments with sequences=[Seq("AAA"), ...], [Seq("CCC"), ...] → Sequence("AAACCC")
            return Sequence.from_sequences(
                subsequences=[seg.candidate_sequences[sequence_idx] for seg in self.inputs],
                merge_metadata=False  # Clean metadata - only basic system keys
            )
        else:
            # DISJOINT: Return tuple of clean Sequence objects
            # Example: sequence_idx=0, segments with sequences=[Seq("AAA"), ...], [Seq("CCC"), ...] → (Seq("AAA"), Seq("CCC"))
            dummy_sequences = []
            for seg in self.inputs:
                original = seg.candidate_sequences[sequence_idx]
                # Create clean Sequence with only essential properties
                dummy_seq = Sequence(
                    sequence=original.sequence,
                    sequence_type=original.sequence_type,
                    valid_chars=original._valid_chars
                )
                dummy_sequences.append(dummy_seq)
            return tuple(dummy_sequences)

    def _propagate_metadata_to_sequence(self, sequence_idx: int, scored_sequence: Sequence | Tuple[Sequence, ...]) -> None:
        """
        Write constraint results back to original sequence metadata.

        Extracts metadata from the scored Sequence object(s) and propagates it back to
        the original sequences in the input segments. Metadata keys are prefixed
        with segment labels and constraint name to prevent collisions.

        Args:
            sequence_idx: Index position in the sequence pool (0-based)
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
                original_seq = segment.candidate_sequences[sequence_idx]
                propagate_metadata(
                    source_metadata=scored_sequence._metadata,
                    target_metadata=original_seq._metadata,
                    prefix=prefix
                )
        else:
            # For disjoint: propagate from each scored Sequence to its corresponding original
            # scored_sequence is a tuple of Sequences, one per segment
            for seg_idx, (segment, scored_seq) in enumerate(zip(self.inputs, scored_sequence)):
                original_seq = segment.candidate_sequences[sequence_idx]
                segment_label = segment.label or f"segment_{seg_idx}"
                prefix = f"{segment_label}.{self.label}"

                propagate_metadata(
                    source_metadata=scored_seq._metadata,
                    target_metadata=original_seq._metadata,
                    prefix=prefix
                )

    def _validate_inputs(self) -> None:
        """Validate that all input segments have consistent candidate pool sizes and sequence types."""
        if not self.inputs:
            raise ValueError("At least one segment must be provided")

        # Check that all segments have the same number of candidates
        candidate_sizes = [seg.num_candidates for seg in self.inputs]
        if not all(size == candidate_sizes[0] for size in candidate_sizes):
            raise ValueError(f"All segments must have the same number of candidate sequences. Found: {candidate_sizes}")

        # If concatenate is True, all segments must have the same sequence type and valid_chars
        if self.concatenate:
            ex = self.inputs[0]
            sequence_type = ex.sequence_type
            valid_chars = ex._valid_chars

            # Check the sequences in all other segments
            for ind, seg in enumerate(self.inputs[1:]):
                if seg.sequence_type != sequence_type:
                    raise ValueError(f"All segments must have the same sequence type. Expected: {sequence_type}, Found: {seg.sequence_type} at index {ind}")
                if seg._valid_chars != valid_chars:
                    raise ValueError(f"All segments must have the same valid_chars. Expected: {valid_chars}, Found: {seg._valid_chars} at index {ind}")

    def _validate_sequence_types(self) -> None:
        """Validate that input segment sequence types are supported by this constraint."""
        supported_types = getattr(self.function, '_constraint_supported_sequence_types', None)
        
        if supported_types is None:
            raise ValueError(f"Constraint function '{self.function.__name__}' missing supported_sequence_types attribute")
        
        for seg in self.inputs:
            if seg.sequence_type not in supported_types:
                supported_str = ", ".join(supported_types)
                raise ValueError(
                    f"Constraint '{self.label}' does not support sequence type '{seg.sequence_type}'. "
                    f"Supported types: [{supported_str}]"
                )
