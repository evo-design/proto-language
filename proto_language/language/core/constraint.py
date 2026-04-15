"""Constraint evaluation, gradient computation, and metadata propagation for sequences.

Constraints score how well sequences satisfy biological or design requirements.
They support two evaluation modes, both optional (at least one required):

- **Discrete** (``function``): scores proposals via ``evaluate()``, returning
  values in [0.0, 1.0] where 0.0 is perfect and 1.0 is worst. Supports
  threshold-based filtering.
- **Gradient** (``backward``): computes gradients via ``compute_gradient()``,
  returning a ``GradientResult`` with gradient, loss, and metrics for
  gradient-based optimizers.

Key Features:
    - Evaluation of all proposals as a batch
    - Multi-segment support (pass tuple of sequences per proposal)
    - Automatic metadata propagation back to original sequences
    - Threshold-based filtering (converts scores to boolean accept/reject)
    - Gradient computation for continuous sequence optimization.
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel

from proto_language.language.core.segment import Segment
from proto_language.language.core.sequence import Sequence
from proto_language.utils.helpers import filter_inf_nan_scores

logger = logging.getLogger(__name__)


# Reserved keys used in constraint data structure. Constraint scoring functions
# must not write these to seq._metadata as they would collide with infrastructure fields.
_RESERVED_CONSTRAINT_KEYS = frozenset(
    {
        "score",
        "weight",
        "weighted_score",
        "data",
        "input_segments",
        "position_in_inputs",
    }
)


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
        >>> def my_constraint(input_sequences: List[Tuple[Sequence, ...]], config: MyConfig) -> List[float]:
        ...     scores = []
        ...     for (seq,) in input_sequences:  # Single-segment
        ...         scores.append(compute_score(seq, config))
        ...     return scores
    """

    def __call__(self, input_sequences: list[tuple[Sequence, ...]], config: BaseModel) -> list[float]:
        """Evaluate sequences and return scores between 0.0 and 1.0."""
        ...


@dataclass(frozen=True)
class GradientResult:
    """Result of a gradient computation through a differentiable model.

    Attributes:
        gradient (tuple[np.ndarray, ...]): Per-segment gradients of the scalar
            objective. One array per input segment, each matching that segment's
            logits shape. Single-segment constraints use a 1-tuple.
        loss (float): Scalar objective value returned by the differentiable
            backend. One value per constraint (not per segment).
        metrics (dict[str, Any]): Optional model-specific auxiliary metrics
            (e.g., pLDDT, pTM) reported alongside ``loss``.
    """

    gradient: tuple[np.ndarray, ...]
    loss: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a compact repr that does not dump the full gradient arrays."""
        shapes = ", ".join(str(g.shape) for g in self.gradient)
        return f"GradientResult(gradient=({shapes}), loss={self.loss}, metrics={self.metrics})"


class Constraint:
    """Constraints handle evaluation and metadata propagation for sequences.

    A constraint can support discrete evaluation (``function``), gradient
    computation (``backward``), or both. At least one must be provided.

    Discrete evaluation uses a standardized signature:
        (input_sequences: List[Tuple[Sequence, ...]], config) -> List[float]

    Gradient computation uses a backward callable:
        (inputs: tuple[Sequence, ...], temperature: float, *, config) -> GradientResult

    Examples:
        Discrete-only constraint:

        >>> config = GCContentConfig(min_gc=40, max_gc=60)
        >>> constraint = Constraint(inputs=[dna_segment], function=gc_content_constraint, function_config=config)
        >>> scores = constraint.evaluate()  # [0.0, 0.1, ...]

        Gradient-only constraint:

        >>> constraint = Constraint(inputs=[segment], backward=my_backward, backward_config=config)
        >>> results = constraint.compute_gradient(temperature=1.0)
        >>> results[0].gradient[0].shape  # (L, vocab_size) — one array per segment

        Both modes (discrete + gradient):

        >>> constraint = Constraint(
        ...     inputs=[segment],
        ...     function=scoring_fn,
        ...     function_config=config,
        ...     backward=gradient_fn,
        ... )
        >>> constraint.supports_discrete  # True
        >>> constraint.supports_gradient  # True

    API/Client Usage (Registry for discovery):
        >>> constraint = ConstraintRegistry.create(
        ...     key="gc-content", segments=[dna_segment], config_dict={"min_gc": 40, "max_gc": 60}
        ... )
    """

    def __init__(
        self,
        inputs: list[Segment],
        function: Callable[..., Any] | None = None,
        function_config: BaseModel | dict[str, Any] | None = None,
        backward: Callable[..., GradientResult] | None = None,
        backward_config: BaseModel | dict[str, Any] | None = None,
        label: str | None = None,
        threshold: float | None = None,
        weight: float | None = None,
    ):
        """Initialize a constraint.

        At least one of ``function`` or ``backward`` must be provided.

        Args:
            inputs (list[Segment]): List of Segment objects to evaluate. Each proposal is evaluated
                as a tuple of sequences (one from each segment).
            function (Callable[..., Any] | None): The constraint scoring function with signature:
                ``(input_sequences: List[Tuple[Sequence, ...]], config) -> List[float]``.
                Returns scores between 0.0-1.0 for each proposal. Required for discrete
                evaluation via ``evaluate()``.
            function_config (BaseModel | dict[str, Any] | None): Configuration for the scoring function.
            backward (Callable[..., GradientResult] | None): Gradient computation callable with signature
                ``(inputs: tuple[Sequence, ...], temperature: float, *, config: BaseModel) -> GradientResult``.
                Receives a tuple of Sequences from input segments (parallel with the scoring function).
                Reads ``.logits`` from optimized segments, ``.sequence`` from context segments.
            backward_config (BaseModel | dict[str, Any] | None): Configuration for the backward callable.
            label (str | None): Optional label for metadata tracking. Defaults to
                ``function.__name__`` or ``backward.__name__``.
            threshold (float | None): Optional threshold for filtering mode. If provided, scores <= threshold are accepted (True),
                scores > threshold are rejected (False). If None, returns raw float scores.
                Mutually exclusive with ``weight`` (setting both raises a ValueError).
            weight (float | None): Optional weight to multiply the raw constraint score by. Defaults to 1.0 if not provided.
                Only meaningful for scoring constraints (when threshold is None).
                Mutually exclusive with ``threshold`` (setting both raises a ValueError).

        Raises:
            ValueError: If neither ``function`` nor ``backward`` is provided.
        """
        if function is None and backward is None:
            raise ValueError("At least one of 'function' or 'backward' must be provided")

        self._inputs = inputs
        self._function = function
        self._backward_fn = backward

        # Label: prefer explicit, then function name, then backward name
        if label is not None:
            self.label = label
        elif function is not None:
            self.label = function.__name__
        else:
            assert backward is not None  # noqa: S101 -- guaranteed by neither-None guard above
            self.label = backward.__name__

        if threshold is not None and weight is not None:
            raise ValueError(
                f"Both threshold ({threshold}) and weight ({weight}) are set, cannot weigh a boolean threshold"
            )
        self._threshold = threshold
        self._weight = 1.0 if weight is None else weight

        # Validate dict config with Pydantic if callable has a registered config class
        self._function_config: BaseModel | dict[str, Any] | None = self._coerce_config(function, function_config)
        self._backward_config: BaseModel | dict[str, Any] | None = self._coerce_config(backward, backward_config)

        # Validate inputs
        self._validate_constraint()

    @staticmethod
    def _coerce_config(
        func: Callable[..., Any] | None, config: BaseModel | dict[str, Any] | None
    ) -> BaseModel | dict[str, Any] | None:
        """Coerce dict config to Pydantic model if the callable has a registered config class."""
        if func is not None and isinstance(config, dict):
            config_cls: type[BaseModel] | None = getattr(func, "_constraint_config_class", None)
            if config_cls is not None:
                return config_cls(**config)
        return config

    # Read-only properties for external access
    @property
    def inputs(self) -> list[Segment]:
        """Input segments (read-only)."""
        return self._inputs

    @property
    def function(self) -> Callable[..., Any] | None:
        """Constraint scoring function (read-only). None for gradient-only constraints."""
        return self._function

    @property
    def function_config(self) -> BaseModel | dict[str, Any] | None:
        """Function configuration (read-only)."""
        return self._function_config

    @property
    def backward(self) -> Callable[..., GradientResult] | None:
        """Backward callable used to compute gradients (read-only). None for discrete-only constraints."""
        return self._backward_fn

    @property
    def backward_config(self) -> BaseModel | dict[str, Any] | None:
        """Configuration for the backward callable (read-only)."""
        return self._backward_config

    @property
    def threshold(self) -> float | None:
        """Threshold for filtering mode (read-only)."""
        return self._threshold

    @property
    def weight(self) -> float:
        """Weight multiplier for scores (read-only)."""
        return self._weight

    @property
    def supports_discrete(self) -> bool:
        """Whether this constraint supports discrete evaluation via ``evaluate()``."""
        return self._function is not None

    @property
    def supports_gradient(self) -> bool:
        """Whether this constraint supports gradient computation via ``compute_gradient()``."""
        return self._backward_fn is not None

    def evaluate(self, mask: list[bool] | None = None, verbose: bool = False) -> list[float] | list[bool]:
        """Evaluate the constraint on proposals using discrete scoring.

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

        Raises:
            RuntimeError: If this constraint has no discrete scoring function.
        """
        if self._function is None:
            raise RuntimeError(
                f"Constraint '{self.label}' does not support discrete evaluation "
                "(no scoring function provided). Use compute_gradient() for gradient-based optimization."
            )
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
            return [float("nan")] * num_proposals if self._threshold is None else [False] * num_proposals

        # Prepare sequences for batched evaluation
        # indexed_sequences stores (original_idx, tuple_for_metadata) pairs
        indexed_sequences = [(idx, self._preprocess_sequence_at_index(idx)) for idx in indices_to_evaluate]

        # Pass List[Tuple[Sequence, ...]] to the constraint function
        input_sequences_to_evaluate = [seq_tuple for _, seq_tuple in indexed_sequences]
        raw_scores = self._function(input_sequences_to_evaluate, config=self._function_config)

        # Validate output: correct count and range [0, 1]
        if len(raw_scores) != len(input_sequences_to_evaluate):
            raise ValueError(
                f"Constraint '{self.label}' returned {len(raw_scores)} scores but expected {len(input_sequences_to_evaluate)}"
            )
        for i, score in enumerate(raw_scores):
            if not (0.0 <= score <= 1.0):
                logger.warning(
                    f"Constraint '{self.label}' returned out-of-range score {score:.4f} at index {i}. Scores should be in [0.0, 1.0]."
                )

        # Propagate metadata back to original sequences
        for j, (original_idx, scored_tuple) in enumerate(indexed_sequences):
            self._propagate_metadata_to_sequence(original_idx, scored_tuple, raw_scores[j])

        # Rebuild dense result array. Skipped proposals get NaN (scoring) or False (filter)
        if self._threshold is None:
            # Scoring constraint: apply weight to raw scores
            final_scores = [float("nan")] * num_proposals
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
                    data_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in custom_data.items()]
                    data_str = f" [{', '.join(data_strs)}]" if data_strs else ""

                    if self._threshold is None:
                        logger.info(
                            f"  Proposal {i}: {final_scores[i]:.4f} = {raw_scores[j]:.4f} * {self._weight}. Data: {data_str}"
                        )
                    else:
                        logger.info(
                            f"  Proposal {i}: {'PASS' if final_scores[i] else 'FAIL'} ({raw_scores[j]:.4f}). Data: {data_str}"
                        )
                else:
                    logger.info(f"  Proposal {i}: SKIPPED")

        return final_scores

    def _preprocess_sequence_at_index(self, sequence_idx: int) -> tuple[Sequence, ...]:
        """Preprocess sequence(s) at a specific batch position for scoring by creating clean Sequence.

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
                sequence=original.sequence, sequence_type=original.sequence_type, valid_chars=original.valid_chars
            )
            dummy_sequences.append(dummy_seq)
        return tuple(dummy_sequences)

    def _propagate_metadata_to_sequence(
        self, sequence_idx: int, scored_sequence: tuple[Sequence, ...], score: float
    ) -> None:
        """Write constraint results to original sequences in structured format.

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

        for seg_idx, (segment, scored_seq) in enumerate(zip(self._inputs, scored_sequence, strict=True)):
            original_seq = segment.proposal_sequences[sequence_idx]
            original_id = id(original_seq)
            if original_id in processed_original_ids:
                continue
            processed_original_ids.add(original_id)

            custom_data = dict(scored_seq._metadata)
            collisions = _RESERVED_CONSTRAINT_KEYS & custom_data.keys()
            if collisions:
                raise ValueError(
                    f"Constraint '{self.label}' wrote reserved keys to seq._metadata: {collisions}. Change the metadata key."
                )

            # Build structured constraint data
            constraint_data: dict[str, Any] = {
                "score": filter_inf_nan_scores(score),
                "weight": self._weight,
                "weighted_score": filter_inf_nan_scores(score * self._weight),
                "data": custom_data or {},
            }

            # Add segment linking info for multi-segment constraints
            if len(self._inputs) > 1:
                constraint_data["input_segments"] = [f"{s.construct_label}.{s.label}" for s in self._inputs]
                constraint_data["position_in_inputs"] = seg_idx

            original_seq._constraints_metadata[self.label] = constraint_data

    def _validate_constraint(self) -> None:
        """Validate constraint configuration.

        Checks:
            1. Non-empty: At least one segment must be provided.
            2. Consistent proposals: All segments must have the same number of proposals.
            3. Supported types: Constraint callable must declare supported sequence types.
            4. Type compatibility: Each segment's sequence type must be supported by the constraint.
            5. Input count: Number of input segments must match input_labels length if specified.

        Raises:
            ValueError: If any validation check fails.
        """
        if not self._inputs:
            raise ValueError("At least one segment must be provided")

        # All segments must have same number of proposals
        proposal_sizes = [seg.num_proposals for seg in self._inputs]
        if not all(size == proposal_sizes[0] for size in proposal_sizes):
            raise ValueError(f"All segments must have the same number of proposal sequences. Found: {proposal_sizes}")

        # Use whichever callable is available for reading decorator-set attributes
        source_fn = self._function if self._function is not None else self._backward_fn

        # Check sequence types are supported
        supported_types = getattr(source_fn, "_constraint_supported_sequence_types", None)
        if supported_types is None:
            warnings.warn(
                f"Constraint '{self.label}' missing supported_sequence_types attribute. Allowing all sequence types as input to constraint.",
                stacklevel=2,
            )
        else:
            for seg in self._inputs:
                if seg.sequence_type not in supported_types:
                    raise TypeError(
                        f"Constraint '{self.label}' does not support sequence type '{seg.sequence_type}'. "
                        f"Supported types: [{', '.join(supported_types)}]"
                    )

        # Check number of input segments matches input_labels length
        expected_inputs = getattr(source_fn, "_constraint_num_input_sequences_per_tuple", None)
        if expected_inputs is None:
            warnings.warn(
                f"Constraint '{self.label}' does not specify input_labels. Using {len(self._inputs)} input segment(s).",
                stacklevel=2,
            )
        else:
            num_inputs = len(self._inputs)
            if num_inputs != expected_inputs:
                raise ValueError(
                    f"Constraint '{self.label}' requires exactly {expected_inputs} input segment(s) "
                    f"(per input_labels), but {num_inputs} segment(s) were provided."
                )

    def compute_gradient(self, temperature: float) -> list[GradientResult]:
        """Compute gradients for all proposals, parallel with ``evaluate()``.

        Iterates over all proposals in the input segments. For each proposal,
        passes a ``tuple[Sequence, ...]`` to the backward callable and propagates
        metrics to ``_constraints_metadata``. The backward reads ``.logits`` from
        optimized segments and ``.sequence`` from context segments.

        Args:
            temperature (float): Softmax temperature for continuous relaxation.

        Returns:
            list[GradientResult]: One result per proposal. Raw gradient, loss, and
                metrics from each backward pass. Weight is NOT applied — the
                optimizer reads ``constraint.weight`` and handles weighting during
                gradient merging.

        Raises:
            RuntimeError: If this constraint has no backward callable, or if any
                proposal has no logits set.
            TypeError: If the backward callable does not return ``GradientResult``.
            ValueError: If ``temperature`` is not positive, or a returned gradient
                shape does not match the logits shape.
        """
        if self._backward_fn is None:
            raise RuntimeError(
                f"Constraint '{self.label}' does not support gradient computation "
                "(no backward callable provided). Use evaluate() for discrete scoring."
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        num_proposals = self._inputs[0].num_proposals
        results: list[GradientResult] = []

        for idx in range(num_proposals):
            inputs = tuple(seg.proposal_sequences[idx] for seg in self._inputs)

            logits_seq = next((seq for seq in inputs if seq.logits is not None), None)
            if logits_seq is None:
                labels = [seg.label for seg in self._inputs]
                raise RuntimeError(
                    f"Constraint '{self.label}': no input segment has logits set on proposal {idx}. "
                    f"Segments: {labels}. Set seq.logits before calling compute_gradient()."
                )

            result = self._backward_fn(inputs, temperature, config=self._backward_config)
            if not isinstance(result, GradientResult):
                raise TypeError(
                    f"backward callable for constraint '{self.label}' must return GradientResult, "
                    f"got {type(result).__name__}"
                )
            if len(result.gradient) != len(inputs):
                raise ValueError(
                    f"Constraint '{self.label}': got {len(result.gradient)} gradient(s), expected {len(inputs)}"
                )
            for seg_idx, (grad, seq) in enumerate(zip(result.gradient, inputs, strict=True)):
                if seq.logits is not None and grad.shape != seq.logits.shape:
                    raise ValueError(
                        f"Constraint '{self.label}': gradient {seg_idx} shape {grad.shape} != logits shape {seq.logits.shape}"
                    )

            scored_tuple = tuple(
                Sequence(
                    sequence=seq.sequence,
                    sequence_type=seq.sequence_type,
                    valid_chars=seq.valid_chars,
                    metadata=result.metrics,
                )
                for seq in inputs
            )
            self._propagate_metadata_to_sequence(idx, scored_tuple, result.loss)
            results.append(result)

        return results
