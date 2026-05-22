"""Constraint evaluation, gradient computation, and metadata propagation for sequences.

Constraints score how well sequences satisfy biological or design requirements.
They support two evaluation modes, both optional (at least one required):

- **Discrete** (``function``): scores proposals via ``evaluate()``. The
  registered scoring function returns ``list[ConstraintOutput]`` — a
  typed per-proposal record carrying score, metadata, and optional per-segment
  predicted structures / logits. Supports threshold-based filtering on the score.
- **Gradient** (``backward``): computes gradients via ``compute_gradient()``,
  returning ``list[GradientConstraintOutput]`` with per-segment gradients, scalar
  loss, metrics, and optional structures for gradient-based optimizers.

Key Features:
    - Evaluation of all proposals as a batch
    - Multi-segment support (pass tuple of sequences per proposal)
    - Typed results for both forward and backward; no sequence mutation by scoring functions
    - Threshold-based filtering (converts scores to boolean accept/reject)
    - Gradient computation for continuous sequence optimization.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import numpy as np
from proto_tools.entities.structures import Structure
from pydantic import BaseModel, ConfigDict, Field

from proto_language.core.segment import Segment
from proto_language.core.sequence import Sequence
from proto_language.utils.serialization import is_plain_int, make_json_safe

logger = logging.getLogger(__name__)


class ConstraintFunction(Protocol):
    """Protocol for forward constraint scoring functions.

    Takes per-proposal input tuples plus a Pydantic config and returns one
    ``ConstraintOutput`` per proposal. Read-only — must not mutate inputs.

    The input tuples allow multi-segment constraints where each proposal consists
    of multiple sequences evaluated together (e.g. protein-protein interactions).
    Single-segment constraints receive a 1-tuple per proposal.

    Example:
        >>> def my_constraint(input_sequences: list[tuple[Sequence, ...]], config: MyConfig) -> list[ConstraintOutput]:
        ...     results = []
        ...     for (seq,) in input_sequences:  # Single-segment
        ...         score, metric = compute_score(seq, config)
        ...         results.append(ConstraintOutput(score=score, metadata={"metric": metric}))
        ...     return results
    """

    def __call__(self, input_sequences: list[tuple[Sequence, ...]], config: BaseModel) -> list["ConstraintOutput"]:
        """Evaluate sequences and return typed results."""
        ...


class InputSlot(BaseModel):
    """Per-slot declaration used by ``@constraint(input_labels=[...])`` for swap-detection.

    Attributes:
        label (str): Slot name, surfaced in error messages.
        requires_logits (bool): Proposal Sequence in this slot must have ``.logits``.
        requires_structure (bool): Proposal Sequence in this slot must have ``.structure``.
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(title="Slot Label", description="Slot name surfaced in error messages")
    requires_logits: bool = Field(
        default=False,
        title="Requires Logits",
        description="When True, the proposal sequence supplied to this slot must include logits.",
    )
    requires_structure: bool = Field(
        default=False,
        title="Requires Structure",
        description="When True, the proposal sequence supplied to this slot must include a predicted structure.",
    )


class ConstraintOutput(BaseModel):
    """Typed result of a single-proposal forward constraint.

    Attributes:
        score (float): Scalar score, ``[0.0, 1.0]`` by default (or any finite value with
            ``_constraint_allow_raw_scores``). Filter constraints may return non-finite.
        metadata (dict[str, Any]): Flat per-proposal data stored under
            ``_constraints_metadata[label]["data"]``.
        structures (tuple[Structure | None, ...]): Optional per-segment structures, aligned
            with the input tuple. Non-``None`` entries are written to ``inputs[i].structure``.
        logits (tuple[np.ndarray | None, ...]): Optional per-segment logits, same semantics.
        metadata_recipient (str | None): Optional unique input label that should receive
            ``metadata``. Constraints with declared ``InputSlot``s resolve this against
            slot labels; otherwise this resolves against segment labels. If unset,
            metadata is written to every input segment, which is the intended default
            for metrics that describe the full input tuple or complex.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    score: float = Field(
        title="Score",
        description="Per-proposal scalar score; conventionally in [0, 1] where lower is better.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        title="Metadata",
        description="Per-segment auxiliary metadata returned alongside the score",
    )
    structures: tuple[Structure | None, ...] = Field(
        default=(),
        title="Per-Segment Structures",
        description="Optional per-segment predicted structures, aligned with the input tuple.",
    )
    logits: tuple[np.ndarray | None, ...] = Field(
        default=(),
        title="Per-Segment Logits",
        description="Optional per-segment logits, aligned with the input tuple.",
    )
    metadata_recipient: str | None = Field(
        default=None,
        title="Metadata Recipient",
        description="Input slot or segment label that should receive metadata; None broadcasts to all.",
    )


class GradientConstraintOutput(BaseModel):
    """Typed result of a gradient computation through a differentiable backend.

    Attributes:
        gradient (tuple[np.ndarray, ...]): Per-segment gradients of the scalar objective,
            aligned with the input tuple. Each array matches that segment's logits shape.
        loss (float): Scalar objective value returned by the differentiable backend.
        metrics (dict[str, Any]): Auxiliary metrics (e.g. pLDDT, pTM) reported alongside ``loss``.
        structures (tuple[Structure | None, ...]): Optional per-segment predicted Structures,
            aligned with ``gradient``. Non-``None`` entries are assigned to ``inputs[i].structure``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    gradient: tuple[np.ndarray, ...] = Field(
        title="Per-Segment Gradients",
        description="Gradient of the scalar loss with respect to each input segment's logits.",
    )
    loss: float = Field(
        title="Loss",
        description="Scalar objective from the differentiable backend; lower is better.",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        title="Metrics",
        description="Auxiliary metrics (e.g. pLDDT, pTM) reported alongside the loss",
    )
    structures: tuple[Structure | None, ...] = Field(
        default=(),
        title="Per-Segment Structures",
        description="Optional per-segment predicted structures, aligned with the gradient tuple.",
    )

    def __repr__(self) -> str:
        """Return a compact repr that does not dump the full gradient arrays."""
        shapes = ", ".join(str(g.shape) for g in self.gradient)
        return f"GradientConstraintOutput(gradient=({shapes}), loss={self.loss}, metrics={self.metrics})"


class Constraint:
    """Constraints handle evaluation and metadata propagation for sequences.

    A constraint can support discrete evaluation (``function``), gradient
    computation (``backward``), or both. At least one must be provided.

    Discrete evaluation uses a standardized signature:
        (input_sequences: list[tuple[Sequence, ...]], config) -> list[ConstraintOutput]

    Gradient computation uses a backward callable:
        (input_sequences: list[tuple[Sequence, ...]], *, config, **kwargs) -> list[GradientConstraintOutput]

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

    Attributes:
        label (str): Metadata label — explicit arg, else ``function.__name__``
            or ``backward.__name__``. Mutable; optimizers may rename to disambiguate duplicates.
    """

    label: str

    def __init__(
        self,
        inputs: list[Segment],
        function: Callable[..., Any] | None = None,
        function_config: BaseModel | dict[str, Any] | None = None,
        backward: Callable[..., list[GradientConstraintOutput]] | None = None,
        backward_config: BaseModel | dict[str, Any] | None = None,
        label: str | None = None,
        threshold: float | None = None,
        weight: float | None = None,
        input_slots: list[InputSlot] | None = None,
        gradient_positions: Iterable[int] | None = None,
    ):
        """Initialize a constraint.

        At least one of ``function`` or ``backward`` must be provided.

        Args:
            inputs (list[Segment]): List of Segment objects to evaluate. Each proposal is evaluated
                as a tuple of sequences (one from each segment).
            function (Callable[..., Any] | None): The constraint scoring function with signature:
                ``(input_sequences: list[tuple[Sequence, ...]], config) -> list[ConstraintOutput]``.
                Returns one ``ConstraintOutput`` per proposal carrying score (in ``[0.0, 1.0]``
                by default), optional flat metadata, and optional per-segment structures / logits.
                Specialized scorers may opt into raw finite scores via
                ``_constraint_allow_raw_scores = True``. Required for discrete evaluation
                via ``evaluate()``.
            function_config (BaseModel | dict[str, Any] | None): Configuration for the scoring function.
            backward (Callable[..., list[GradientConstraintOutput]] | None): Batched gradient callable
                with signature ``(input_sequences: list[tuple[Sequence, ...]], *, config: BaseModel, **kwargs) ->
                list[GradientConstraintOutput]``. Receives all proposals at once and returns one result
                per proposal. Reads ``.logits`` from optimized segments, ``.sequence`` from context
                segments. Additional kwargs (e.g., ``temperature``, ``soft``, ``hard``) are forwarded
                from ``compute_gradient()``.
            backward_config (BaseModel | dict[str, Any] | None): Configuration for the backward callable.
            label (str | None): Optional label for metadata tracking. Defaults to
                ``function.__name__`` or ``backward.__name__``.
            threshold (float | None): Optional threshold for filtering mode. If provided, scores <= threshold are accepted (True),
                scores > threshold are rejected (False). If None, returns raw float scores.
                Mutually exclusive with ``weight`` (setting both raises a ValueError).
            weight (float | None): Optional weight to multiply the raw constraint score by. Defaults to 1.0 if not provided.
                Only meaningful for scoring constraints (when threshold is None).
                Mutually exclusive with ``threshold`` (setting both raises a ValueError).
            input_slots (list[InputSlot] | None): Per-slot requirements enforced in
                ``compute_gradient``. Normally plumbed by ``ConstraintRegistry.create()``.
            gradient_positions (Iterable[int] | None): Optional zero-based
                sequence positions allowed to receive gradient. When set, all
                other rows in the single gradient-bearing input are zeroed after
                the backward callable returns. Currently requires exactly one
                input with logits; multi-input or tied-segment constraints must
                mask their gradients in the backward callable instead.

        Raises:
            ValueError: If neither ``function`` nor ``backward`` is provided.
        """
        if function is None and backward is None:
            raise ValueError("At least one of 'function' or 'backward' must be provided")

        self._inputs = inputs
        self._function = function
        self._backward_fn = backward
        self._input_slots = input_slots or []
        self._gradient_positions = self._normalize_gradient_positions(gradient_positions)

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

    @staticmethod
    def _normalize_gradient_positions(positions: Iterable[int] | None) -> tuple[int, ...] | None:
        """Normalize and validate user-provided gradient positions."""
        if positions is None:
            return None
        normalized = tuple(positions)
        if not normalized:
            raise ValueError("gradient_positions must be None or non-empty; got [].")
        invalid = [position for position in normalized if not is_plain_int(position)]
        if invalid:
            raise ValueError(f"gradient_positions must contain integers; got {invalid}.")
        negative = [position for position in normalized if position < 0]
        if negative:
            raise ValueError(f"gradient_positions must be non-negative; got {negative}.")
        return tuple(sorted(set(normalized)))

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
    def backward(self) -> Callable[..., list[GradientConstraintOutput]] | None:
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
    def gradient_positions(self) -> tuple[int, ...] | None:
        """Positions allowed to receive gradient, or ``None`` when unmasked."""
        return self._gradient_positions

    @property
    def supports_discrete(self) -> bool:
        """Whether this constraint supports discrete evaluation via ``evaluate()``."""
        return self._function is not None

    @property
    def supports_gradient(self) -> bool:
        """Whether this constraint supports gradient computation via ``compute_gradient()``."""
        return self._backward_fn is not None

    def _set_program_seed(self, seed: int | None) -> None:
        """Set runtime seeds and reset private seed cursors."""

        def apply(value: Any) -> None:
            if isinstance(value, BaseModel):
                if hasattr(value, "_evaluation_seed_offset"):
                    value._evaluation_seed_offset = 0
                fields = value.__class__.model_fields
                config: Any = value
                if seed is not None:
                    if "seed" in fields:
                        config.seed = seed
                    if "seeds" in fields:
                        config.seeds = [seed]
                for field_name in fields:
                    apply(getattr(value, field_name))
            elif isinstance(value, dict):
                if seed is not None:
                    if "seed" in value:
                        value["seed"] = seed
                    if "seeds" in value:
                        value["seeds"] = [seed]
                for child in value.values():
                    apply(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    apply(child)

        for config in (self._function_config, self._backward_config):
            apply(config)

    def evaluate(self, mask: list[bool] | None = None, verbose: bool = False) -> list[float] | list[bool]:
        """Evaluate the constraint on proposals using discrete scoring.

        This method orchestrates the evaluation:
        1. Resolve ``mask`` to indices; early-exit with NaN/False if none pass.
        2. Pass original proposals directly to the scoring function (no dummy copies).
        3. Validate each ``ConstraintOutput``: type, score range, per-segment tuple arity.
        4. Write ``_constraints_metadata[label]`` and propagate non-``None`` per-segment
           ``structures`` / ``logits`` onto the originals; homomers deduplicated by ``id()``.
        5. Build the dense per-proposal output: ``score <= threshold`` (filter mode) or
           ``score * weight`` (scoring); skipped proposals get NaN/False.

        Args:
            mask (list[bool] | None): Boolean mask of which proposals to evaluate. None = all.
            verbose (bool): If true, logs per-proposal scores and metadata.

        Returns:
            list[float] | list[bool]: Per-proposal results. Filter constraints return False
                for unevaluated proposals; scoring constraints return NaN.

        Raises:
            RuntimeError: This constraint has no discrete scoring function.
            TypeError: A returned element is not a ``ConstraintOutput``.
            ValueError: Result count, score range, or per-segment tuple arity is invalid.
        """
        if self._function is None:
            raise RuntimeError(f"Constraint '{self.label}' has no scoring function; use compute_gradient() instead.")
        num_proposals = self._inputs[0].num_proposals
        logger.debug(f"Constraint.evaluate: {self.label}, proposals={num_proposals}, threshold={self._threshold}")

        if mask is None:
            mask = [True] * num_proposals
        if len(mask) != num_proposals:
            raise ValueError(f"Mask length {len(mask)} != num_proposals {num_proposals}")

        indices_to_evaluate = [i for i in range(num_proposals) if mask[i]]
        if not indices_to_evaluate:
            return [float("nan")] * num_proposals if self._threshold is None else [False] * num_proposals

        input_sequences = [tuple(seg.proposal_sequences[idx] for seg in self._inputs) for idx in indices_to_evaluate]
        results = self._function(input_sequences, config=self._function_config)

        allow_raw_scores = bool(getattr(self._function, "_constraint_allow_raw_scores", False))
        is_filter = self._threshold is not None
        if len(results) != len(input_sequences):
            raise ValueError(f"'{self.label}' returned {len(results)} results, expected {len(input_sequences)}")
        n_inputs = len(self._inputs)
        metadata_recipient_positions: list[int | None] = []
        for i, result in enumerate(results):
            idx = indices_to_evaluate[i]
            if not isinstance(result, ConstraintOutput):
                raise TypeError(
                    f"'{self.label}' proposal {idx}: expected ConstraintOutput, got {type(result).__name__}"
                )
            score = result.score
            if not np.isfinite(score) and not is_filter and not allow_raw_scores:
                raise ValueError(f"'{self.label}' proposal {idx}: non-finite score {score!r}")
            if not allow_raw_scores and not is_filter and not (0.0 <= score <= 1.0):
                raise ValueError(f"'{self.label}' proposal {idx}: score {score!r} not in [0.0, 1.0]")
            metadata_recipient_positions.append(
                self._resolve_metadata_recipient_position(idx, result.metadata_recipient)
            )
            if result.structures and len(result.structures) != n_inputs:
                raise ValueError(
                    f"'{self.label}' proposal {idx}: {len(result.structures)} structures, expected {n_inputs}"
                )
            if result.logits and len(result.logits) != n_inputs:
                raise ValueError(f"'{self.label}' proposal {idx}: {len(result.logits)} logits, expected {n_inputs}")

        for original_idx, result, metadata_recipient_position in zip(
            indices_to_evaluate, results, metadata_recipient_positions, strict=True
        ):
            self._write_forward_result(original_idx, result, metadata_recipient_position)

        if self._threshold is None:
            final_scores: list[float] | list[bool] = [float("nan")] * num_proposals
            for j, idx in enumerate(indices_to_evaluate):
                final_scores[idx] = results[j].score * self._weight
        else:
            final_scores = [False] * num_proposals
            for j, idx in enumerate(indices_to_evaluate):
                final_scores[idx] = results[j].score <= self._threshold

        if verbose:
            evaluated_set = set(indices_to_evaluate)
            for i in range(num_proposals):
                if i in evaluated_set:
                    j = indices_to_evaluate.index(i)
                    custom_data = results[j].metadata
                    data_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in custom_data.items()]
                    data_str = f" [{', '.join(data_strs)}]" if data_strs else ""
                    if self._threshold is None:
                        logger.info(
                            f"  Proposal {i}: {final_scores[i]:.4f} = {results[j].score:.4f} * {self._weight}. Data: {data_str}"
                        )
                    else:
                        logger.info(
                            f"  Proposal {i}: {'PASS' if final_scores[i] else 'FAIL'} ({results[j].score:.4f}). Data: {data_str}"
                        )
                else:
                    logger.info(f"  Proposal {i}: SKIPPED")

        return final_scores

    def _write_forward_result(
        self, sequence_idx: int, result: ConstraintOutput, metadata_recipient_position: int | None
    ) -> None:
        """Write a ``ConstraintOutput`` onto the original proposal sequences.

        Stores score/weight/weighted_score and nested ``data`` under
        ``original._constraints_metadata[self.label]``; assigns non-``None`` per-segment
        structures and logits to matching proposal sequences. Skips duplicate segments
        that share a proposal Sequence instance (e.g., homomers).
        """
        self._write_constraint_metadata(sequence_idx, result.score, result.metadata, metadata_recipient_position)
        if result.structures or result.logits:
            processed_ids: set[int] = set()
            for seg_idx, segment in enumerate(self._inputs):
                original = segment.proposal_sequences[sequence_idx]
                if id(original) in processed_ids:
                    continue
                processed_ids.add(id(original))
                if result.structures:
                    s = result.structures[seg_idx]
                    if s is not None:
                        original.structure = s
                if result.logits:
                    lg = result.logits[seg_idx]
                    if lg is not None:
                        original.logits = lg

    def _write_constraint_metadata(
        self, sequence_idx: int, score: float, metadata: dict[str, Any], metadata_recipient_position: int | None = None
    ) -> None:
        """Write ``_constraints_metadata[self.label]`` on each unique input proposal.

        ``metadata_recipient_position`` restricts custom metadata to one input.
        When unset, custom metadata is copied to all inputs for discoverability.
        """
        originals_by_id: dict[int, Sequence] = {}
        metadata_by_original: dict[int, dict[str, Any]] = {}
        position_by_original: dict[int, int] = {}
        for seg_idx, segment in enumerate(self._inputs):
            original = segment.proposal_sequences[sequence_idx]
            original_id = id(original)
            originals_by_id.setdefault(original_id, original)
            metadata_by_original.setdefault(original_id, {})
            position_by_original.setdefault(original_id, seg_idx)
            if metadata_recipient_position is None or metadata_recipient_position == seg_idx:
                metadata_by_original[original_id].update(metadata)

        for original_id, original in originals_by_id.items():
            seg_idx = position_by_original[original_id]
            constraint_data: dict[str, Any] = {
                "score": make_json_safe(score),
                "weight": self._weight,
                "weighted_score": make_json_safe(score * self._weight),
                "data": metadata_by_original[original_id],
            }
            if len(self._inputs) > 1:
                constraint_data["input_segments"] = [f"{s.construct_label}.{s.label}" for s in self._inputs]
                constraint_data["position_in_inputs"] = seg_idx
            original._constraints_metadata[self.label] = constraint_data

    def _resolve_metadata_recipient_position(self, sequence_idx: int, metadata_recipient: str | None) -> int | None:
        """Resolve a metadata target to an input position for one proposal."""
        if metadata_recipient is None:
            return None

        if self._input_slots:
            matches = [idx for idx, slot in enumerate(self._input_slots) if slot.label == metadata_recipient]
        else:
            matches = [
                idx
                for idx, segment in enumerate(self._inputs)
                if metadata_recipient == segment.label
                or (segment.construct_label and metadata_recipient == f"{segment.construct_label}.{segment.label}")
            ]

        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"'{self.label}' proposal {sequence_idx}: metadata_recipient {metadata_recipient!r} "
            "must match exactly one input label"
        )

    def _validate_constraint(self) -> None:
        """Validate constraint configuration.

        Checks:
            1. Non-empty: At least one segment must be provided.
            2. Distinct inputs: Warn (do not reject) if the same Segment instance appears more than once.
            3. Consistent proposals: All segments must have the same number of proposals.
            4. Supported types: Constraint callable must declare supported sequence types.
            5. Type compatibility: Each segment's sequence type must be supported by the constraint.
            6. Input count: Number of input segments must match input_labels length if specified.

        Raises:
            ValueError: If any validation check fails.
        """
        if not self._inputs:
            raise ValueError("At least one segment must be provided")

        # Aliased inputs are allowed but almost always a user error.
        if len({id(seg) for seg in self._inputs}) != len(self._inputs):
            logger.warning(
                "Constraint %r has duplicate Segment instances in `inputs`; pass N distinct Segments instead.",
                self.label,
            )

        # All segments must have same number of proposals
        proposal_sizes = [seg.num_proposals for seg in self._inputs]
        if not all(size == proposal_sizes[0] for size in proposal_sizes):
            raise ValueError(f"All segments must have the same number of proposal sequences. Found: {proposal_sizes}")

        # Use whichever callable is available for reading decorator-set attributes
        source_fn = self._function if self._function is not None else self._backward_fn

        # Check sequence types are supported
        supported_types = getattr(source_fn, "_constraint_supported_sequence_types", None)
        if supported_types is None:
            logger.warning(
                "Constraint %r missing supported_sequence_types attribute; allowing all sequence types as input",
                self.label,
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
            logger.warning(
                "Constraint %r does not specify input_labels; using %d input segment(s)",
                self.label,
                len(self._inputs),
            )
        else:
            num_inputs = len(self._inputs)
            if num_inputs != expected_inputs:
                raise ValueError(
                    f"Constraint '{self.label}' requires exactly {expected_inputs} input segment(s) "
                    f"(per input_labels), but {num_inputs} segment(s) were provided."
                )

    def compute_gradient(self, **kwargs: Any) -> list[GradientConstraintOutput]:
        """Compute gradients for all proposals as a batch, parallel with ``evaluate()``.

        Builds a batched input list from all proposals and passes it to the backward
        callable in a single batched call. Validates per-proposal slot requirements
        before the call, then validates per-result shape and propagates structures /
        metadata after.

        Args:
            **kwargs (Any): Forwarded to the backward callable (e.g. ``temperature``, ``soft``, ``hard``).

        Returns:
            list[GradientConstraintOutput]: One result per proposal. Raw gradient, loss, and
                metrics from the backward pass. Weight is NOT applied — the
                optimizer reads ``constraint.weight`` and handles weighting during
                gradient merging.

        Raises:
            RuntimeError: If this constraint has no backward callable, any declared slot's
                ``requires_logits`` / ``requires_structure`` is unmet, or (fallback) no input
                has logits on a proposal.
            TypeError: If a returned element is not ``GradientConstraintOutput``.
            ValueError: If result count, gradient shape, or structure arity is invalid.
        """
        if self._backward_fn is None:
            raise RuntimeError(f"Constraint '{self.label}' has no backward callable; use evaluate() instead.")

        num_proposals = self._inputs[0].num_proposals
        has_declared_logits_slot = any(s.requires_logits for s in self._input_slots)

        all_input_tuples: list[tuple[Sequence, ...]] = [
            tuple(seg.proposal_sequences[idx] for seg in self._inputs) for idx in range(num_proposals)
        ]

        for idx, inputs_tuple in enumerate(all_input_tuples):
            if self._input_slots:
                for slot_idx, (slot, seq) in enumerate(zip(self._input_slots, inputs_tuple, strict=True)):
                    if slot.requires_logits and seq.logits is None:
                        raise RuntimeError(f"'{self.label}' proposal {idx} slot {slot_idx} '{slot.label}': missing logits")  # fmt: skip
                    if slot.requires_structure and seq.structure is None:
                        raise RuntimeError(f"'{self.label}' proposal {idx} slot {slot_idx} '{slot.label}': missing structure")  # fmt: skip
            if not has_declared_logits_slot and all(seq.logits is None for seq in inputs_tuple):
                raise RuntimeError(f"'{self.label}' proposal {idx}: no input has logits")

        results = self._backward_fn(all_input_tuples, config=self._backward_config, **kwargs)

        if len(results) != num_proposals:
            raise ValueError(f"'{self.label}' returned {len(results)} gradient results, expected {num_proposals}")

        n_inputs = len(self._inputs)
        masked_results: list[GradientConstraintOutput] = []
        for idx, (result, inputs_tuple) in enumerate(zip(results, all_input_tuples, strict=True)):
            if not isinstance(result, GradientConstraintOutput):
                raise TypeError(f"'{self.label}': expected GradientConstraintOutput, got {type(result).__name__}")
            if len(result.gradient) != n_inputs:
                raise ValueError(f"'{self.label}': {len(result.gradient)} gradients, expected {n_inputs}")
            for seg_idx, (grad, seq) in enumerate(zip(result.gradient, inputs_tuple, strict=True)):
                if seq.logits is not None and grad.shape != seq.logits.shape:
                    raise ValueError(
                        f"'{self.label}' segment {seg_idx}: gradient shape {grad.shape} != logits shape {seq.logits.shape}"
                    )

            masked_result = self._apply_gradient_positions(result, inputs_tuple)

            if masked_result.structures:
                if len(masked_result.structures) != n_inputs:
                    raise ValueError(f"'{self.label}': {len(masked_result.structures)} structures, expected {n_inputs}")
                for seq, struct in zip(inputs_tuple, masked_result.structures, strict=True):
                    if struct is not None:
                        seq.structure = struct

            self._write_constraint_metadata(idx, masked_result.loss, masked_result.metrics)
            masked_results.append(masked_result)

        return masked_results

    def _apply_gradient_positions(
        self,
        result: GradientConstraintOutput,
        inputs_tuple: tuple[Sequence, ...],
    ) -> GradientConstraintOutput:
        """Apply the optional single-input gradient-position mask."""
        if self._gradient_positions is None:
            return result

        gradient_input_indices = [idx for idx, seq in enumerate(inputs_tuple) if seq.logits is not None]
        if len(gradient_input_indices) != 1:
            raise ValueError(
                f"'{self.label}' gradient_positions currently supports exactly one input sequence with logits; "
                f"found {len(gradient_input_indices)}."
            )

        target_idx = gradient_input_indices[0]
        gradient = result.gradient[target_idx]
        sequence_length = gradient.shape[0]
        out_of_range = [position for position in self._gradient_positions if position >= sequence_length]
        if out_of_range:
            raise ValueError(
                f"'{self.label}' gradient_positions {out_of_range} are >= gradient length ({sequence_length})."
            )

        mask = np.zeros(sequence_length, dtype=bool)
        mask[list(self._gradient_positions)] = True
        masked_gradient = gradient.copy()
        masked_gradient[~mask] = 0.0

        gradients = list(result.gradient)
        gradients[target_idx] = masked_gradient
        return result.model_copy(update={"gradient": tuple(gradients)})
