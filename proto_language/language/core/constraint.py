"""
proto_language/language/core/constraint.py

Constraints score how well sequences satisfy biological or design requirements,
returning values between 0.0 (perfect) and 1.0 (worst). Constraints can optionally
act as filters by providing a threshold parameter.

Key Features:
    - Evaluation of all proposals as a batch
    - Multi-segment support (pass tuple of sequences per proposal)
    - Automatic metadata propagation back to original sequences
    - Threshold-based filtering (converts scores to boolean accept/reject)
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel

from proto_language.utils.helpers import filter_inf_nan_scores

from .segment import Segment
from .sequence import Sequence

logger = logging.getLogger(__name__)


# Reserved keys used in constraint data structure — constraint scoring functions
# must not write these to seq._metadata as they would collide with infrastructure fields.
_RESERVED_CONSTRAINT_KEYS = frozenset({
    "score", "weight", "weighted_score",
    "data", "input_segments", "position_in_inputs",
})


class ConstraintFunction(Protocol):
    """Protocol defining the standardized constraint function signature.

    All constraint functions must conform to this signature:
    - Accept a list of sequence tuples (one tuple per proposal to evaluate)
    - Accept a Pydantic config object
    - Return a list of float scores between 0.0 and 1.0

    The input tuples allow multi-segment constraints where each proposal
    consists of multiple sequences evaluated together (e.g., protein-protein
    interactions). For single-segment constraints, each tuple contains one sequence.

    Example:
        >>> def my_constraint(
        ...     input_sequences: List[Tuple[Sequence, ...]],
        ...     config: MyConfig
        ... ) -> List[float]:
        ...     scores = []
        ...     for (seq,) in input_sequences:  # Single-segment
        ...         scores.append(compute_score(seq, config))
        ...     return scores
    """

    def __call__(
        self,
        input_sequences: List[Tuple[Sequence, ...]],
        config: BaseModel
    ) -> List[float]:
        """Evaluate sequences and return scores between 0.0 and 1.0."""
        ...


class Constraint:
    """
    Constraints handle evaluation and metadata propagation for sequences.

    All constraint functions use a standardized signature:
        (input_sequences: List[Tuple[Sequence, ...]], config) -> List[float]

    Examples:
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

    Examples:
        >>> # Define your own constraint function (no decorator needed)
        >>> def my_custom_constraint(input_sequences, config) -> List[float]:
        ...     return [0.0 if config["value"] > 0.5 else 1.0 for _ in input_sequences]
        >>>
        >>> # Just pass a dict - no Pydantic model needed
        >>> constraint = Constraint(
        ...     inputs=[segment],
        ...     function=my_custom_constraint,
        ...     function_config={"value": 0.5}
        ... )

    API/Client Usage (Registry for discovery):
        >>> from proto_language.language.constraint import constraint
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
            inputs (list[Segment]): List of Segment objects to evaluate. Each proposal is evaluated
                as a tuple of sequences (one from each segment).
            function (Callable): The constraint scoring function with signature:
                (input_sequences: List[Tuple[Sequence, ...]], config) -> List[float]
                Returns scores between 0.0-1.0 for each proposal.
            function_config (BaseModel | dict[str, Any]): Configuration as Pydantic BaseModel or dict (auto-converted to BaseModel)
            label (str | None): Optional label for metadata tracking. Defaults to function.__name__
            threshold (float | None): Optional threshold for filtering mode. If provided, scores <= threshold are accepted (True),
                scores > threshold are rejected (False). If None, returns raw float scores.
                Mutually exclusive with ``weight`` (setting both raises a ValueError).
            weight (float | None): Optional weight to multiply the raw constraint score by. Defaults to 1.0 if not provided.
                Only meaningful for scoring constraints (when threshold is None).
                Mutually exclusive with ``threshold`` (setting both raises a ValueError).
        """
        self._inputs = inputs
        self._function = function
        self.label = label or function.__name__

        if threshold is not None and weight is not None:
            raise ValueError(f"Both threshold ({threshold}) and weight ({weight}) are set, cannot weigh a boolean threshold")
        self._threshold = threshold
        self._weight = 1.0 if weight is None else weight

        # Validate dict config with Pydantic if registered (has config class from decorator)
        config_cls = getattr(function, '_constraint_config_class', None)
        if isinstance(function_config, dict) and config_cls:
            self._function_config = config_cls(**function_config)
        else:
            self._function_config = function_config

        # Validate inputs
        self._validate_constraint()

    # Read-only properties for external access
    @property
    def inputs(self) -> List[Segment]:
        """Input segments (read-only)."""
        return self._inputs

    @property
    def function(self) -> Callable:
        """Constraint scoring function (read-only)."""
        return self._function

    @property
    def function_config(self) -> BaseModel:
        """Function configuration (read-only)."""
        return self._function_config

    @property
    def threshold(self) -> Optional[float]:
        """Threshold for filtering mode (read-only)."""
        return self._threshold

    @property
    def weight(self) -> float:
        """Weight multiplier for scores (read-only)."""
        return self._weight

    def evaluate(
        self,
        mask: Optional[List[bool]] = None,
        verbose: bool = False
    ) -> List[float] | List[bool]:
        """
        Evaluate the constraint on proposals.

        This method orchestrates the evaluation:
        1. Extract proposal sequences from input segments (only those that passed)
        2. Call the scoring function with List[Tuple[Sequence, ...]]
        3. Propagate scores back to original proposal sequence metadata
        4. Convert scores to boolean filters if threshold is set, or apply weight if not
        5. Build a dense result array (one entry per proposal)

        Args:
            mask (list[bool] | None): Boolean mask indicating which proposals to evaluate. If None, evaluates all.
            verbose (bool): If true, logs evaluation details.

        Returns:
            list[float] | list[bool]: List of results. Filter constraints
                return False for unevaluated proposals; scoring constraints
                return NaN for unevaluated proposals.
        """
        num_proposals = self._inputs[0].num_proposals
        logger.debug(f"Constraint.evaluate: {self.label}, proposals={num_proposals}, threshold={self._threshold}")

        # Default: evaluate all proposals
        if mask is None:
            mask = [True] * num_proposals
        if len(mask) != num_proposals:
            raise ValueError(f"Mask length ({len(mask)}) does not match num_proposals ({num_proposals})")

        # Convert mask to indices for sparse evaluation
        indices_to_evaluate = [i for i in range(num_proposals) if mask[i]]

        # Early return if no proposals to evaluate
        if not indices_to_evaluate:
            return [float('nan')] * num_proposals if self._threshold is None else [False] * num_proposals

        # Prepare sequences for batched evaluation
        # indexed_sequences stores (original_idx, tuple_for_metadata) pairs
        indexed_sequences = [(idx, self._preprocess_sequence_at_index(idx)) for idx in indices_to_evaluate]

        # Pass List[Tuple[Sequence, ...]] to the constraint function
        input_sequences_to_evaluate = [seq_tuple for _, seq_tuple in indexed_sequences]
        raw_scores = self._function(input_sequences_to_evaluate, config=self._function_config)

        # Validate output: correct count and range [0, 1]
        if len(raw_scores) != len(input_sequences_to_evaluate):
            raise ValueError(f"Constraint '{self.label}' returned {len(raw_scores)} scores but expected {len(input_sequences_to_evaluate)}")
        for i, score in enumerate(raw_scores):
            if not (0.0 <= score <= 1.0):
                logger.warning(f"Constraint '{self.label}' returned out-of-range score {score:.4f} at index {i}. Scores should be in [0.0, 1.0].")

        # Propagate metadata back to original sequences
        for j, (original_idx, scored_tuple) in enumerate(indexed_sequences):
            self._propagate_metadata_to_sequence(original_idx, scored_tuple, raw_scores[j])

        # Rebuild dense result array. Skipped proposals get NaN (scoring) or False (filter)
        if self._threshold is None:
            # Scoring constraint: apply weight to raw scores
            final_scores = [float('nan')] * num_proposals
            for j, idx in enumerate(indices_to_evaluate):
                final_scores[idx] = raw_scores[j] * self._weight
        else:
            # Filter constraint: convert scores to boolean (pass if score <= threshold)
            final_scores = [False] * num_proposals
            for j, idx in enumerate(indices_to_evaluate):
                final_scores[idx] = raw_scores[j] <= self._threshold

        if verbose:
            evaluated_set = set(indices_to_evaluate)
            for i in range(num_proposals):
                if i in evaluated_set:
                    j = indices_to_evaluate.index(i)
                    # Get custom data directly from scored tuples to avoid relying on
                    # segment candidate indexing/broadcasting details.
                    scored_tuple = indexed_sequences[j][1]
                    custom_data = {}
                    for scored_seq in scored_tuple:
                        if scored_seq._metadata:
                            custom_data = scored_seq._metadata
                            break
                    data_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in custom_data.items()]
                    data_str = f" [{', '.join(data_strs)}]" if data_strs else ""

                    if self._threshold is None:
                        logger.info(f"  Proposal {i}: {final_scores[i]:.4f} = {raw_scores[j]:.4f} * {self._weight}. Data: {data_str}")
                    else:
                        logger.info(f"  Proposal {i}: {'PASS' if final_scores[i] else 'FAIL'} ({raw_scores[j]:.4f}). Data: {data_str}")
                else:
                    logger.info(f"  Proposal {i}: SKIPPED")

        return final_scores

    def _preprocess_sequence_at_index(self, sequence_idx: int) -> Tuple[Sequence, ...]:
        """
        Preprocess sequence(s) at a specific batch position for scoring by creating clean Sequence
        objects with fresh metadata to pass to the scoring function.

        Args:
            sequence_idx (int): Index position in the sequence pool (0-based)

        Returns:
            tuple[Sequence, ...]: Tuple[Sequence, ...] - tuple of clean Sequence objects, one per input segment
        """
        # Return tuple of clean Sequence objects
        # Example: sequence_idx=0, segments with sequences=[Seq("AAA"), ...], [Seq("CCC"), ...] → (Seq("AAA"), Seq("CCC"))
        dummy_sequences = []
        for seg in self._inputs:
            original = seg.proposal_sequences[sequence_idx]
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

        Stores constraint data under _constraints_metadata[constraint_label] with:
        - Standard evaluation fields at top level (score, weight, weighted_score)
        - Custom data from scoring function nested under "data"
        - Input segment linking info for multi-segment constraints

        Args:
            sequence_idx (int): Index position in the sequence pool (0-based)
            scored_sequence (tuple[Sequence, ...]): Tuple of Sequences that were scored, containing metadata
                           written by the scoring function (one per segment)
            score (float): Raw score returned by the constraint function (before weight applied)

        Example:
            Scoring constraint (weight=2.0):

            >>> seq._constraints_metadata["gc_content_constraint"]
            {
                "score": 0.12,
                "weight": 2.0,
                "weighted_score": 0.24,
                "data": {"gc_content": 52.3}
            }

            Multi-segment constraint on two segments:

            >>> protein_a._constraints_metadata["interaction_constraint"]
            {
                "score": 0.05,
                "weight": 1.0,
                "weighted_score": 0.05,
                "input_segments": ["construct_0.protein_a", "construct_0.protein_b"],
                "position_in_inputs": 0,
                "data": {"binding_energy": -8.2}
            }
            >>> protein_b._constraints_metadata["interaction_constraint"]
            {
                "score": 0.05,  # Same score - joint evaluation
                "weight": 1.0,
                "weighted_score": 0.05,
                "input_segments": ["construct_0.protein_a", "construct_0.protein_b"],
                "position_in_inputs": 1,
                "data": {"interface_residues": 12}
            }
        """
        # Skip duplicate segments within the same constraint to avoid overwriting metadata with empty data
        # (e.g., inputs=[protomer, protomer, protomer] for symmetric proteins)
        processed_original_ids = set()

        for seg_idx, (segment, scored_seq) in enumerate(zip(self._inputs, scored_sequence)):
            original_seq = segment.proposal_sequences[sequence_idx]
            original_id = id(original_seq)
            if original_id in processed_original_ids:
                continue
            processed_original_ids.add(original_id)

            custom_data = dict(scored_seq._metadata)
            collisions = _RESERVED_CONSTRAINT_KEYS & custom_data.keys()
            if collisions:
                raise ValueError(f"Constraint '{self.label}' wrote reserved keys to seq._metadata: {collisions}. Change the metadata key.")

            # Build structured constraint data
            constraint_data = {
                "score": filter_inf_nan_scores(score),
                "weight": self._weight,
                "weighted_score": filter_inf_nan_scores(score * self._weight),
                "data": custom_data if custom_data else {},
            }

            # Add segment linking info for multi-segment constraints
            if len(self._inputs) > 1:
                constraint_data["input_segments"] = [f"{s.construct_label}.{s.label}" for s in self._inputs]
                constraint_data["position_in_inputs"] = seg_idx

            original_seq._constraints_metadata[self.label] = constraint_data

    def _validate_constraint(self) -> None:
        """
        Validate constraint configuration.

        Checks:
            1. Non-empty: At least one segment must be provided.
            2. Consistent proposals: All segments must have the same number of proposals.
            3. Supported types: Constraint function must declare supported sequence types.
            4. Type compatibility: Each segment's sequence type must be supported by the constraint.
            5. Input count: Number of input segments must match num_input_sequences_per_tuple if specified.

        Raises:
            ValueError: If any validation check fails.
        """
        if not self._inputs:
            raise ValueError("At least one segment must be provided")

        # All segments must have same number of proposals
        proposal_sizes = [seg.num_proposals for seg in self._inputs]
        if not all(size == proposal_sizes[0] for size in proposal_sizes):
            raise ValueError(f"All segments must have the same number of proposal sequences. Found: {proposal_sizes}")

        # Check sequence types are supported
        supported_types = getattr(self._function, '_constraint_supported_sequence_types', None)
        if supported_types is None:
            warnings.warn(f"Constraint function '{self._function.__name__}' missing supported_sequence_types attribute. Allowing all sequence types as input to constraint.")
        else:
            for seg in self._inputs:
                if seg.sequence_type not in supported_types:
                    raise TypeError(f"Constraint '{self.label}' does not support sequence type '{seg.sequence_type}'. "
                                  f"Supported types: [{', '.join(supported_types)}]")

        # Check number of input sequences per tuple matches requirement
        num_input_sequences_per_tuple = getattr(self._function, '_constraint_num_input_sequences_per_tuple', None)
        if num_input_sequences_per_tuple is None:
            warnings.warn(f"Constraint '{self.label}' does not specify required number of input sequences per tuple. Using {len(self._inputs)} input segment(s).")
        else:
            num_inputs = len(self._inputs)
            if num_inputs != num_input_sequences_per_tuple:
                raise ValueError(f"Constraint '{self.label}' requires exactly {num_input_sequences_per_tuple} input sequence(s) per tuple, but {num_inputs} segment(s) were provided.")
