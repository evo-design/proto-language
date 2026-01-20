"""
Constraint class for biological programming language.

Constraints score how well sequences satisfy biological or design requirements,
returning values between 0.0 (perfect) and 1.0 (worst). Constraints can optionally
act as filters by providing a threshold parameter.

Key Features:
    - Batch evaluation (sequential or batched processing)
    - Multi-segment support (pass tuple of sequences per candidate)
    - Automatic metadata propagation back to original sequences
    - Threshold-based filtering (converts scores to boolean accept/reject)
"""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict, Any

from pydantic import BaseModel

from .sequence import Sequence
from .segment import Segment
from proto_language.utils.helpers import filter_inf_nan_scores


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
                Signature depends on batched and multi_input flags:
                - batched=False, multi_input=False: (Sequence, config) -> float
                - batched=True,  multi_input=False: (List[Sequence], config) -> List[float]
                - batched=False, multi_input=True:  (Tuple[Sequence, ...], config) -> float
                - batched=True,  multi_input=True:  (List[Tuple[Sequence, ...]], config) -> List[float]
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
        self.multi_input = getattr(function, '_constraint_multi_input', False)

        # Convert dict configs to Pydantic models for validation
        if isinstance(function_config, dict):
            config_class = function._constraint_config_class
            self.function_config = config_class(**function_config)
        else:
            self.function_config = function_config

        # Validate inputs
        self._validate_constraint()

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
            # indexed_sequences stores (original_idx, tuple_for_metadata) pairs
            indexed_sequences = [(idx, self._preprocess_sequence_at_index(idx)) for idx in indices_to_evaluate]
            
            # Transform based on multi_input flag
            if self.multi_input:
                # Multi-input: pass List[Tuple[Sequence, ...]]
                sequences_to_evaluate = [seq_tuple for _, seq_tuple in indexed_sequences]
            else:
                # Single-input: unpack tuples -> List[Sequence]
                sequences_to_evaluate = [seq_tuple[0] for _, seq_tuple in indexed_sequences]
            
            raw_scores = self.function(sequences_to_evaluate, config=self.function_config)

            for j, (original_idx, scored_tuple) in enumerate(indexed_sequences):
                self._propagate_metadata_to_sequence(original_idx, scored_tuple, raw_scores[j])
        else:
            # Sequential mode: evaluate one at a time
            raw_scores = []
            for idx in indices_to_evaluate:
                seq_tuple = self._preprocess_sequence_at_index(idx)
                
                # Transform based on multi_input flag
                if self.multi_input:
                    # Multi-input: pass Tuple[Sequence, ...]
                    score = self.function(seq_tuple, config=self.function_config)
                else:
                    # Single-input: unpack tuple -> Sequence
                    score = self.function(seq_tuple[0], config=self.function_config)
                
                raw_scores.append(score)
                self._propagate_metadata_to_sequence(idx, seq_tuple, score)

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
                    # Get custom data from propagated metadata
                    constraint_data = self.inputs[0].candidate_sequences[i]._metadata["constraints"][self.label]
                    custom_data = constraint_data["data"]
                    data_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in custom_data.items()]
                    data_str = f" [{', '.join(data_strs)}]" if data_strs else ""
                    
                    if self.threshold is None:
                        print(f"  Candidate {i}: {final_scores[i]:.4f} = {raw_scores[j]:.4f} * {self.weight}. Data: {data_str}")
                    else:
                        print(f"  Candidate {i}: {'PASS' if final_scores[i] else 'FAIL'} ({raw_scores[j]:.4f}). Data: {data_str}")
                else:
                    print(f"  Candidate {i}: SKIPPED")

        return final_scores

    def _preprocess_sequence_at_index(self, sequence_idx: int) -> Tuple[Sequence, ...]:
        """
        Preprocess sequence(s) at a specific batch position for scoring by creating clean Sequence
        objects with fresh metadata to pass to the scoring function.

        Args:
            sequence_idx: Index position in the sequence pool (0-based)

        Returns:
            Tuple[Sequence, ...] - tuple of clean Sequence objects, one per input segment
        """
        # Return tuple of clean Sequence objects
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

    def _propagate_metadata_to_sequence(self, sequence_idx: int, scored_sequence: Tuple[Sequence, ...], score: float) -> None:
        """
        Write constraint results to original sequences in structured format.

        Stores constraint data under _metadata["constraints"][constraint_label] with:
        - Standard evaluation fields at top level (score, weight, weighted_score)
        - Custom data from scoring function nested under "data"
        - Multi-input linking info when applicable

        Args:
            sequence_idx: Index position in the sequence pool (0-based)
            scored_sequence: Tuple of Sequences that were scored, containing metadata
                           written by the scoring function (one per segment)
            score: Raw score returned by the constraint function (before weight applied)

        Example:
            Scoring constraint (weight=2.0):

            >>> seq._metadata["constraints"]["gc_content_constraint"]
            {
                "score": 0.12,
                "weight": 2.0,
                "weighted_score": 0.24,
                "multi_input": False,
                "data": {"gc_content": 52.3}
            }

            Multi-input constraint on two segments:

            >>> protein_a._metadata["constraints"]["interaction_constraint"]
            {
                "score": 0.05,
                "weight": 1.0,
                "weighted_score": 0.05,
                "multi_input": True,
                "input_segments": ["construct_0.protein_a", "construct_0.protein_b"],
                "position_in_inputs": 0,
                "data": {"binding_energy": -8.2}
            }
            >>> protein_b._metadata["constraints"]["interaction_constraint"]
            {
                "score": 0.05,  # Same score - joint evaluation
                "weight": 1.0,
                "weighted_score": 0.05,
                "multi_input": True,
                "input_segments": ["construct_0.protein_a", "construct_0.protein_b"],
                "position_in_inputs": 1,
                "data": {"interface_residues": 12}
            }
        """
        for seg_idx, (segment, scored_seq) in enumerate(zip(self.inputs, scored_sequence)):
            original_seq = segment.candidate_sequences[sequence_idx]

            # Extract custom data from scoring function (nested under "data")
            custom_data = {k: v for k, v in scored_seq._metadata.items()
                          if k not in {"sequence", "sequence_length", "constraints"}}

            # Build structured constraint data
            constraint_data = {
                "score": filter_inf_nan_scores(score),
                "weight": self.weight,
                "weighted_score": filter_inf_nan_scores(score * self.weight),
                "multi_input": self.multi_input,
                "data": custom_data if custom_data else {},
            }
            if self.multi_input:
                constraint_data["input_segments"] = [f"{s.construct_label}.{s.label}" for s in self.inputs]
                constraint_data["position_in_inputs"] = seg_idx

            original_seq._metadata["constraints"][self.label] = constraint_data

    def _validate_constraint(self) -> None:
        """Validate constraint inputs: segments, sequence types, and segment count."""
        if not self.inputs:
            raise ValueError("At least one segment must be provided")

        # Single-input constraints only accept 1 segment
        if not self.multi_input and len(self.inputs) > 1:
            raise ValueError(
                f"Constraint '{self.label}' is single-input (multi_input=False) but received "
                f"{len(self.inputs)} segments. Single-input constraints only accept 1 segment."
            )

        # All segments must have same number of candidates
        candidate_sizes = [seg.num_candidates for seg in self.inputs]
        if not all(size == candidate_sizes[0] for size in candidate_sizes):
            raise ValueError(f"All segments must have the same number of candidate sequences. Found: {candidate_sizes}")

        # Check sequence types are supported
        supported_types = getattr(self.function, '_constraint_supported_sequence_types', None)
        if supported_types is None:
            raise ValueError(f"Constraint function '{self.function.__name__}' missing supported_sequence_types attribute")
        
        for seg in self.inputs:
            if seg.sequence_type not in supported_types:
                raise ValueError(
                    f"Constraint '{self.label}' does not support sequence type '{seg.sequence_type}'. "
                    f"Supported types: [{', '.join(supported_types)}]"
                )
