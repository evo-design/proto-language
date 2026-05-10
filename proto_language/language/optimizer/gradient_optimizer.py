"""Gradient-based optimizer for differentiable sequence design."""

import copy
import logging
from collections.abc import Callable
from typing import Any, Literal, final

import numpy as np
from pydantic import ValidationInfo, field_validator, model_validator

from proto_language.base_config import BaseConfig, BaseOptimizerConfig, ConfigField
from proto_language.language.core import Constraint, Construct, Generator, Optimizer, Segment
from proto_language.language.generator import PositionWeightGenerator
from proto_language.language.optimizer.constraint_compiler import (
    GradientProvider,
    GradientProviderOutput,
    compile_gradient_providers,
    constraint_supports_compiled_gradient,
)
from proto_language.language.optimizer.optimizer_registry import optimizer
from proto_language.utils import softmax
from proto_language.utils.gradients import MERGERS, GradientMergerName, align_norms, normalize_gradient
from proto_language.utils.ml_optimizers import ML_OPTIMIZERS, AdamConfig, MLOptimizerType
from proto_language.utils.scheduling import SCHEDULES, Schedule, Scheduler
from proto_language.utils.sequence_logit_bias import SequenceLogitBiasConfig, build_sequence_logit_bias_matrix

logger = logging.getLogger(__name__)


class ConstraintWeightSchedule(BaseConfig):
    """Per-step weight schedule keyed by ``Constraint.label``; overrides the static weight.

    Note:
        ``schedule="exponential"`` requires both ``start_weight`` and ``end_weight`` to be ``> 0``.
        ``schedule="constant"`` uses ``start_weight`` only — ``end_weight`` is ignored.
        Ramps use ``progress = step / num_steps`` with ``step`` starting at 1, so step 1 evaluates
        to ``start + (end - start) / num_steps`` (not exactly ``start``); step ``num_steps`` is
        exactly ``end``.

    Attributes:
        constraint_label (str): Label of the target constraint.
        start_weight (float): Weight at step 1.
        end_weight (float): Weight at final step (ignored when ``schedule="constant"``).
        schedule (Scheduler): Interpolation between start and end.
    """

    constraint_label: str = ConfigField(
        min_length=1,
        title="Constraint Label",
        description="Label of the constraint whose weight this schedule overrides.",
    )
    start_weight: float = ConfigField(
        title="Start Weight",
        ge=0.0,
        description="Weight at the first step.",
    )
    end_weight: float = ConfigField(
        title="End Weight",
        ge=0.0,
        description="Weight at the final step.",
    )
    schedule: Scheduler = ConfigField(
        default="linear",
        title="Schedule",
        description="Interpolation schedule between start_weight and end_weight.",
    )

    @model_validator(mode="after")
    def _check_schedule_endpoints(self) -> "ConstraintWeightSchedule":
        if self.schedule == "exponential" and min(self.start_weight, self.end_weight) <= 0.0:
            raise ValueError("schedule='exponential' requires start_weight > 0 and end_weight > 0.")
        if self.schedule == "hinge" and self.start_weight >= self.end_weight:
            raise ValueError("schedule='hinge' requires start_weight < end_weight.")
        return self


class GradientOptimizerConfig(BaseOptimizerConfig):
    """Configuration for gradient-based sequence optimization.

    Each GradientOptimizer runs one mode (fixed or ramping soft, with optional
    temperature annealing). Chain multiple in a ``Program`` for multi-phase
    pipelines (e.g. logit phase → softmax phase).

    Attributes:
        num_results (int | None): Parallel optimization trajectories.
        num_steps (int): Number of gradient steps.
        lr (float): Base learning rate.
        sequence_bias (SequenceLogitBiasConfig | None): Declarative per-position symbol bias resolved
            against the target segment vocabulary and added to initial logits.
        soft_start (float): Soft blending at step 1 (0=hard, 1=softmax).
        soft_end (float): Soft blending at final step.
        hard_start (float): Straight-through estimator at step 1 (0=relaxed, 1=argmax).
        hard_end (float): Straight-through estimator at final step.
        temperature_start (float): Temperature at step 1. Both schedules interpolate
            between this and ``temperature_end``; they differ only in curve shape.
        temperature_end (float): Temperature at final step.
        softmax_schedule (Scheduler): Softmax sharpening schedule for constraints.
        lr_schedule (Scheduler): Learning rate decay schedule.
        merger (GradientMergerName): Gradient merging strategy.
        ml_optimizer (MLOptimizerType): Gradient update algorithm (SGD, Adam).
        adam_config (AdamConfig): Adam hyperparameters. Only visible when ``ml_optimizer="adam"``.
        norm_alignment (Literal["none", "unit", "match_first"]): Per-constraint
            gradient normalization before merging.
        zero_norm_eps (float): In match_first mode, zero out gradients with norm below this.
        normalize_gradients (bool): Normalize merged gradient before update.
        normalize_mode (Literal["unit", "sqrt_length"]): Normalization formula.
            ``"unit"`` = L2 to magnitude 1. ``"sqrt_length"`` = ``g * sqrt(eff_L) / ||g||``.
        fixed_positions (list[int] | None): Positions to freeze during optimization.
        scale_lr_by_temperature (bool): Multiply LR by soft/temperature blending.
        min_lr_scale (float): Floor for effective LR scale.
        tracking_interval (int): Steps between progress snapshots.
        track_proposals (bool): Record per-proposal results in history.
        save_best (bool): Return the lowest-loss result instead of the last iteration.
        constraint_weight_schedules (list[ConstraintWeightSchedule] | None): Per-label
            weight schedules that override ``Constraint.weight`` each step.
        gumbel_logit_init (bool): If True, add Gumbel(0,1) noise per position on top of the
            resolved ``sequence_bias`` — gives parallel trajectories stochastic starts.
        gumbel_init_alpha (float): Divisor for Gumbel init noise. ``1.0`` = unscaled.
        initial_logits (list[list[float]] | None): Base logit matrix ``(L, |vocab|)``
            replacing default initialization.
        softmax_init_positions (list[int] | None): Positions receiving per-trajectory
            Gumbel noise + softmax over ``initial_logits``.

    Note:
        Ramps use ``progress = step / num_steps`` with ``step`` starting at 1,
        so step 1 evaluates to ``start + (end - start) / num_steps`` (not exactly
        ``start``); step ``num_steps`` evaluates exactly to ``end``.
    """

    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Candidate designs for this optimizer. Overrides program-level count.",
        advanced=True,
    )
    num_steps: int = ConfigField(
        default=1,
        ge=1,
        title="Num Steps",
        description="Number of gradient descent steps.",
    )
    lr: float = ConfigField(
        default=0.05,
        gt=0.0,
        title="Learning Rate",
        description="Base learning rate for gradient updates.",
    )
    sequence_bias: SequenceLogitBiasConfig | None = ConfigField(
        default=None,
        title="Sequence Bias",
        description="Declarative sequence-symbol bias resolved against the target segment vocabulary.",
        advanced=True,
    )
    soft_start: float = ConfigField(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="Soft Start",
        description="Soft blending at step 1 (0=hard logits, 1=full softmax).",
        advanced=True,
    )
    soft_end: float = ConfigField(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="Soft End",
        description="Soft blending at final step.",
        advanced=True,
    )
    hard_start: float = ConfigField(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Hard Start",
        description="Straight-through estimator blending at step 1 (0=relaxed, 1=argmax).",
        advanced=True,
    )
    hard_end: float = ConfigField(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Hard End",
        description="Straight-through estimator blending at final step.",
        advanced=True,
    )
    temperature_start: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature Start",
        description="Softmax temperature at step 1.",
    )
    temperature_end: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature End",
        description="Softmax temperature at final step.",
    )
    softmax_schedule: Scheduler = ConfigField(
        default="constant",
        title="Softmax Schedule",
        description="Softmax sharpening schedule for constraints.",
    )
    lr_schedule: Scheduler = ConfigField(
        default="constant",
        title="LR Schedule",
        description="Learning rate decay schedule.",
    )
    merger: GradientMergerName = ConfigField(
        default="weighted_sum",
        title="Gradient Merger",
        description="Strategy for merging gradients from multiple constraints.",
    )
    ml_optimizer: MLOptimizerType = ConfigField(
        default="sgd",
        title="ML Optimizer",
        description="Gradient update algorithm (SGD, Adam).",
    )
    adam_config: AdamConfig = ConfigField(
        default_factory=AdamConfig,
        title="Adam Config",
        description="Adam hyperparameters.",
        depends_on={"field": "ml_optimizer", "value": "adam"},
    )
    norm_alignment: Literal["none", "unit", "match_first"] = ConfigField(
        default="none",
        title="Norm Alignment",
        description="Per-constraint gradient normalization before merging.",
        advanced=True,
    )
    zero_norm_eps: float = ConfigField(
        default=0.0,
        ge=0.0,
        title="Zero Norm Epsilon",
        description="In match_first mode, zero out gradients with norm below this threshold.",
        advanced=True,
        depends_on={"field": "norm_alignment", "value": "match_first"},
    )
    normalize_gradients: bool = ConfigField(
        default=True,
        title="Normalize Gradients",
        description="Normalize the merged gradient before each update.",
        advanced=True,
    )
    normalize_mode: Literal["unit", "sqrt_length"] = ConfigField(
        default="unit",
        title="Normalize Mode",
        description="'unit' = L2 to 1.0. 'sqrt_length' = g*sqrt(eff_L)/||g||.",
        advanced=True,
    )
    fixed_positions: list[int] | None = ConfigField(
        default=None,
        title="Fixed Positions",
        description="Sequence positions to freeze. Pair with sequence_bias to anchor them at the desired AA.",
        advanced=True,
    )
    scale_lr_by_temperature: bool = ConfigField(
        default=False,
        title="Scale LR by Temperature",
        description="Multiply LR by soft/temperature blending factor.",
        advanced=True,
    )
    min_lr_scale: float = ConfigField(
        default=0.0,
        ge=0.0,
        title="Min LR Scale",
        description="Floor for effective LR scale when temperature scaling is enabled.",
        advanced=True,
    )
    save_best: bool = ConfigField(
        default=True,
        title="Save Best",
        description="Return the lowest-loss result instead of the last iteration.",
        advanced=True,
    )
    constraint_weight_schedules: list[ConstraintWeightSchedule] | None = ConfigField(
        default=None,
        title="Constraint Weight Schedules",
        description="Per-label weight schedules that override Constraint.weight each step.",
        advanced=True,
    )
    gumbel_logit_init: bool = ConfigField(
        default=False,
        title="Gumbel Logit Init",
        description="Gumbel(0,1) noise at init; zeroed at fixed_positions; additive with sequence_bias.",
        advanced=True,
    )
    gumbel_init_alpha: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Gumbel Init Alpha",
        description="Divisor for Gumbel init noise. 1.0 = unscaled.",
        advanced=True,
    )
    initial_logits: list[list[float]] | None = ConfigField(
        default=None,
        title="Initial Logits",
        description="Base logit matrix (L x |vocab|) replacing default initialization.",
        advanced=True,
        hidden=True,
    )
    softmax_init_positions: list[int] | None = ConfigField(
        default=None,
        title="Softmax Init Positions",
        description="Positions receiving Gumbel noise + softmax over initial_logits.",
        advanced=True,
        hidden=True,
    )

    @property
    def ml_optimizer_config(self) -> BaseConfig | None:
        """Return the active ML optimizer config, or None for stateless optimizers."""
        configs: dict[str, BaseConfig] = {"adam": self.adam_config}
        return configs.get(self.ml_optimizer)

    @field_validator("initial_logits")
    @classmethod
    def _validate_matrix(cls, v: list[list[float]] | None, info: ValidationInfo) -> list[list[float]] | None:
        """Reject non-rectangular / non-numeric input; shape-vs-segment is checked in ``GradientOptimizer._validate_optimizer``."""
        if v is not None:
            try:
                np.asarray(v, dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"{info.field_name} must be a rectangular 2-D matrix: {exc}") from exc
        return v

    @model_validator(mode="after")
    def _validate_initial_logits_config(self) -> "GradientOptimizerConfig":
        """Validate cross-field initialization settings."""
        if self.softmax_init_positions is not None and self.initial_logits is None:
            raise ValueError("softmax_init_positions requires initial_logits to be set")
        if self.softmax_init_positions and self.fixed_positions:
            overlap = set(self.softmax_init_positions) & set(self.fixed_positions)
            if overlap:
                raise ValueError(
                    f"positions {sorted(overlap)} appear in both softmax_init_positions and fixed_positions"
                )
        return self

    @classmethod
    def germinal_logit_preset(cls) -> "GradientOptimizerConfig":
        """Logit hallucination phase: 65 SGD steps with soft 0→1, naturalness weight 0.2→0.4.

        Ramps naturalness weight via hinge schedule ``max(0.4·t, 0.2)`` — flat at 0.2 until
        50% progress, then linear to 0.4. Constraint must be labeled ``"ablang"``.
        Initial logits get Gumbel(0,1) noise so parallel trajectories diverge.
        """
        return cls(
            num_steps=65,
            lr=0.1,
            soft_start=0.0,
            soft_end=1.0,
            temperature_start=1.0,
            temperature_end=1.0,
            softmax_schedule="constant",
            lr_schedule="constant",
            merger="pcgrad",
            norm_alignment="match_first",
            normalize_mode="sqrt_length",
            constraint_weight_schedules=[
                ConstraintWeightSchedule(constraint_label="ablang", start_weight=0.2, end_weight=0.4, schedule="hinge")
            ],
            gumbel_logit_init=True,
            gumbel_init_alpha=2.0,
        )

    @classmethod
    def germinal_softmax_preset(cls) -> "GradientOptimizerConfig":
        """Softmax refinement phase: 35 SGD steps with soft=1.0, temperature 1.0→0.01 quadratic.

        Naturalness weight is constant 0.4 — set ``Constraint(weight=0.4)`` directly.
        """
        return cls(
            num_steps=35,
            lr=0.1,
            soft_start=1.0,
            soft_end=1.0,
            temperature_start=1.0,
            temperature_end=0.01,
            softmax_schedule="quadratic",
            lr_schedule="quadratic",
            merger="pcgrad",
            norm_alignment="match_first",
            normalize_mode="sqrt_length",
            scale_lr_by_temperature=True,
            min_lr_scale=0.01,
        )


@optimizer(
    key="gradient",
    label="Gradient Optimizer",
    config=GradientOptimizerConfig,
    description="Gradient-based sequence optimization via differentiable constraints",
    compatible_generators=["position-weight"],
    required_constraint_mode="gradient",
    targets_single_segment=True,
)
@final
class GradientOptimizer(Optimizer):
    """Gradient-based optimizer for differentiable sequence design.

    Updates ``seq.logits`` directly on proposal sequences via gradient descent
    through differentiable constraints. Uses ``PositionWeightGenerator``
    to discretize logits into sequences for tracking and handoff.

    Chain multiple GradientOptimizers in a ``Program`` for multi-phase
    pipelines (e.g., logit phase → softmax phase).

    Attributes:
        config (GradientOptimizerConfig): Optimizer configuration.
    """

    # Class attribute required by OptimizerRegistry
    config_class = GradientOptimizerConfig
    config: GradientOptimizerConfig

    def __init__(
        self,
        target_segment: Segment,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: GradientOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the gradient optimizer.

        Args:
            target_segment (Segment): The specific Segment to optimize. Must belong to one of the constructs.
            constructs (list[Construct]): Constructs to optimize.
            generators (list[Generator]): Must contain exactly one PositionWeightGenerator.
            constraints (list[Constraint]): Must include at least one gradient-capable constraint.
            config (GradientOptimizerConfig): Configuration.
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked steps
                (governed by ``tracking_interval``).
            clear_tool_cache (int | bool | list[str]): (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If the generator count/type is wrong, no gradient-capable constraints,
                or a constraint's inputs do not include the target segment.
        """
        if len(generators) != 1:
            raise ValueError(f"GradientOptimizer requires exactly one generator, got {len(generators)}.")
        generator = generators[0]

        self.config = config
        self.target_segment: Segment = target_segment
        self.generator: Generator = generator
        unsupported = []
        for constraint in constraints:
            ok, reason = constraint_supports_compiled_gradient(constraint, target_segment)
            if not ok:
                unsupported.append(reason or f"Constraint '{constraint.label}' does not support gradient evaluation.")
        if unsupported:
            raise ValueError("GradientOptimizer requires differentiable constraints: " + "; ".join(unsupported))
        self._gradient_constraints = list(constraints)
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_results=config.num_results,
            proposals_per_result=1,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
            seed=config.seed,
        )

        self.num_steps: int = config.num_steps
        # Merger, ML optimizer, schedules
        self._merger = MERGERS[config.merger]()
        self._ml_optimizer = ML_OPTIMIZERS[config.ml_optimizer](config.ml_optimizer_config)
        self._softmax_schedule = SCHEDULES[config.softmax_schedule](config.temperature_start, config.temperature_end)
        self._lr_schedule = SCHEDULES[config.lr_schedule](config.temperature_start, config.temperature_end)

        # Missing labels warn (not error) so presets remain portable across constraint sets.
        known = {c.label for c in self._gradient_constraints}
        self._weight_schedules: dict[str, Schedule] = {}
        for e in config.constraint_weight_schedules or []:
            if e.constraint_label in known:
                self._weight_schedules[e.constraint_label] = SCHEDULES[e.schedule](e.start_weight, e.end_weight)
            else:
                logger.warning("Unknown weight-schedule label %r; ignored.", e.constraint_label)
        self._gradient_providers: list[GradientProvider] = compile_gradient_providers(
            self._gradient_constraints, self.target_segment
        )

        self._sequence_bias_matrix: np.ndarray | None = build_sequence_logit_bias_matrix(
            config.sequence_bias, self.target_segment
        )

    def _validate_optimizer(self) -> None:
        """Extend base validation with gradient-specific checks against the target segment."""
        super()._validate_optimizer()
        self._validate_target_segment(self.target_segment)

        if not isinstance(self.generator, PositionWeightGenerator):
            raise ValueError(
                f"GradientOptimizer requires a PositionWeightGenerator, got {self.generator.__class__.__name__}."
            )

        seq_len = self.target_segment.sequence_length
        if self.config.fixed_positions:
            out_of_bounds = [p for p in self.config.fixed_positions if p < 0 or p >= seq_len]
            if out_of_bounds:
                raise ValueError(f"fixed_positions {out_of_bounds} out of bounds for segment length {seq_len}.")

        if self.config.initial_logits is not None:
            expected = (seq_len, len(self.target_segment.ordered_vocab()))
            row0 = self.config.initial_logits[0] if self.config.initial_logits else ()
            actual = (len(self.config.initial_logits), len(row0))
            if actual != expected:
                raise ValueError(f"initial_logits shape {actual} does not match target segment {expected}.")
        if self.config.softmax_init_positions:
            out_of_bounds = [p for p in self.config.softmax_init_positions if p < 0 or p >= seq_len]
            if out_of_bounds:
                raise ValueError(f"softmax_init_positions {out_of_bounds} out of bounds for segment length {seq_len}.")

    def _validate_component_compatibility(self) -> None:
        """Validate dependencies while allowing compiler-backed gradient constraints."""
        from proto_language.language.constraint.constraint_registry import ConstraintRegistry
        from proto_language.language.generator.generator_registry import GeneratorRegistry
        from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry

        opt_key = OptimizerRegistry.find_key(self)
        opt = OptimizerRegistry.get(opt_key) if opt_key else None
        opt_label = opt.label if opt else self.__class__.__name__
        gen_keys = {k for gen in self.generators if (k := GeneratorRegistry.find_key(gen)) is not None}

        if opt and opt.compatible_generators is not None:
            for key in gen_keys:
                if key not in opt.compatible_generators:
                    raise ValueError(
                        f"Generator '{key}' is not compatible with {opt_label}. "
                        f"Compatible generators: {', '.join(opt.compatible_generators)}"
                    )

        if opt and opt.required_constraint_mode is not None:
            required = opt.required_constraint_mode
            ok_modes = {"gradient": ("gradient", "dual"), "discrete": ("discrete", "dual")}[required]
            for con in self.constraints:
                con_key = ConstraintRegistry.find_key(con)
                if con_key and ConstraintRegistry.get(con_key).mode in ok_modes:
                    continue
                ok, reason = constraint_supports_compiled_gradient(con, self.target_segment)
                if ok:
                    continue
                detail = f": {reason}" if reason else ""
                raise ValueError(
                    f"Constraint '{con.label}' does not support {required} evaluation, required by {opt_label}{detail}"
                )

        for con in self.constraints:
            con_key = ConstraintRegistry.find_key(con)
            spec = ConstraintRegistry.get(con_key) if con_key else None
            if not spec or not spec.requires_generators:
                continue
            missing = [r for r in spec.requires_generators if r not in gen_keys]
            if missing:
                raise ValueError(
                    f"Constraint '{con.label}' requires a {', '.join(missing)} generator in the same optimization stage"
                )

    def run(self) -> None:
        """Execute gradient optimization.

        Each step:
        1. Compute soft and temperature from linear/scheduled interpolation
        2. Compute gradients from all gradient-capable constraints
        3. Align norms, apply weights, merge (PCGrad/MGDA/weighted sum)
        4. Zero fixed positions, then normalize merged gradient
        5. Update ``seq.logits`` via the configured ML optimizer
        6. At tracked steps: discretize via generator, save snapshot
        """
        self._prepare_run()
        self._ml_optimizer.reset()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing
        target = self.target_segment

        vocab = target.ordered_vocab()
        needs_rng = self.config.initial_logits is not None or self.config.gumbel_logit_init
        init_rng = np.random.default_rng(self.seed) if needs_rng else None
        logit_bias = self._sequence_bias_matrix
        initial_logits_arr = (
            np.asarray(self.config.initial_logits, dtype=np.float64) if self.config.initial_logits is not None else None
        )
        for seq in target.proposal_sequences:
            if seq.logits is None:
                seq.logits = _init_logits(
                    target.sequence_length,
                    len(vocab),
                    initial_logits=initial_logits_arr,
                    logit_bias=logit_bias,
                    rng=init_rng,
                    fixed_positions=self.config.fixed_positions,
                    gumbel_alpha=self.config.gumbel_init_alpha,
                    softmax_init_positions=self.config.softmax_init_positions,
                )

        self.generator.sample()
        self._proposal_outcomes = ["accepted"] * self.num_proposals
        self._proposal_energy_scores = list(self.energy_scores)
        self._sync_proposals_to_results()
        initial_temp = self._softmax_schedule(0, self.config.num_steps)
        initial_lr = self._effective_lr(self._lr_schedule(0, self.config.num_steps), self.config.soft_start)
        self._save_progress_snapshot(
            time_step=0,
            optimizer_metadata={
                "type": "gradient",
                "num_steps": self.config.num_steps,
                "num_results": self.num_results,
                "temperature": initial_temp,
                "learning_rate": initial_lr,
                "soft": self.config.soft_start,
                "hard": self.config.hard_start,
                "proposal_count": len(self._proposal_outcomes),
                "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
            },
        )

        if self.config.save_best:
            best_energies = list(self.energy_scores)
            best_proposals = [copy.deepcopy(self.target_segment.proposal_sequences[k]) for k in range(self.num_results)]

        if self.verbose:
            logger.info(
                f"GradientOptimizer: {self.num_results} trajectories, {self.config.num_steps} steps, "
                f"soft {self.config.soft_start}→{self.config.soft_end}, "
                f"hard {self.config.hard_start}→{self.config.hard_end}, "
                f"temp {self.config.temperature_start}→{self.config.temperature_end}"
            )

        for step in range(1, self.config.num_steps + 1):
            # 1. Compute soft and temperature from linear/scheduled interpolation
            progress = step / self.config.num_steps
            soft = self.config.soft_start + (self.config.soft_end - self.config.soft_start) * progress
            hard = self.config.hard_start + (self.config.hard_end - self.config.hard_start) * progress
            temp = self._softmax_schedule(step, self.config.num_steps)
            lr_temp = self._lr_schedule(step, self.config.num_steps)
            lr = self._effective_lr(lr_temp, soft)

            # 2. Compute gradients from all gradient-capable constraints
            provider_outputs = [
                provider.compute(
                    temperature=temp,
                    soft=soft,
                    hard=hard,
                    step=step,
                    effective_weight=self._effective_weight,
                )
                for provider in self._gradient_providers
            ]
            for output in provider_outputs:
                for k, grad in enumerate(output.gradients):
                    if not np.isfinite(grad).all():
                        raise ValueError(f"Non-finite gradient from '{output.label}' at step {step} (proposal {k}).")

            # 3-5. Merge gradients and update logits for each trajectory
            for k in range(self.num_results):
                self._update_trajectory(k, provider_outputs, lr, target, step)

            # Report the same weighted objective that gradient descent is actually minimizing.
            self.energy_scores = [sum(output.losses[k] for output in provider_outputs) for k in range(self.num_results)]
            self._proposal_outcomes = ["accepted"] * self.num_proposals
            self._proposal_energy_scores = list(self.energy_scores)

            if self.config.save_best:
                for k in range(self.num_results):
                    if self.energy_scores[k] < best_energies[k]:
                        best_energies[k] = self.energy_scores[k]
                        best_proposals[k] = copy.deepcopy(self.target_segment.proposal_sequences[k])

            # 6. At tracked steps: discretize, sync proposals→results, snapshot
            if step % self.tracking_interval == 0 or step == self.config.num_steps:
                self.generator.sample()
                self._sync_proposals_to_results()
                self._save_progress_snapshot(
                    time_step=step,
                    optimizer_metadata={
                        "type": "gradient",
                        "num_steps": self.config.num_steps,
                        "num_results": self.num_results,
                        "temperature": temp,
                        "learning_rate": lr,
                        "soft": soft,
                        "hard": hard,
                        "proposal_count": len(self._proposal_outcomes),
                        "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
                    },
                )
                self._log_progress(step, temp, lr)

            self._clear_tool_cache()

        if self.config.save_best:
            for k in range(self.num_results):
                self.target_segment.proposal_sequences[k] = best_proposals[k]
            self.energy_scores = best_energies
            self.generator.sample()
            self._sync_proposals_to_results()

    # =============================================================================
    # Private helpers
    # =============================================================================

    def _log_progress(self, step: int, temperature: float, lr: float) -> None:
        """Log optimization progress at tracked steps."""
        if self.verbose:
            best = min(self.energy_scores)
            mean = float(np.mean(self.energy_scores))
            logger.info(
                f"Step {step:4d}/{self.config.num_steps} | "
                f"T={temperature:.4f} | lr={lr:.6f} | best={best:.6f} | mean={mean:.6f}"
            )

        if self.custom_logging:
            self.custom_logging(step, self.segments)

    def _update_trajectory(
        self, k: int, provider_outputs: list[GradientProviderOutput], lr: float, target: Segment, step: int
    ) -> None:
        """Align, merge, normalize, and apply one gradient step for trajectory *k*."""
        grads = [output.gradients[k] for output in provider_outputs]
        weights = [output.weight for output in provider_outputs]

        # Align norms first so ``match_first`` doesn't wash out weights.
        grads = align_norms(grads, self.config.norm_alignment, zero_norm_eps=self.config.zero_norm_eps)
        grads = [g * w for g, w in zip(grads, weights, strict=True)]
        merged = self._merger.merge(grads)

        if self.config.fixed_positions:
            merged[self.config.fixed_positions] = 0.0
        if self.config.normalize_gradients:
            merged = normalize_gradient(merged, self.config.normalize_mode)

        seq = target.proposal_sequences[k]
        assert seq.logits is not None  # noqa: S101 -- guaranteed by initialization
        seq.logits = self._ml_optimizer.step(seq.logits, merged, lr, trajectory=k, step=step)

    def _capture_initial_state(self) -> None:
        """Capture initial state preserving logits.

        Overrides the base implementation to pass ``include_logits=True`` so that
        on multi-stage re-run, ``_restore_initial_state`` can hand the continuous
        logits back to ``run()`` instead of silently re-initializing from scratch.
        """
        self._initial_state = {
            "segments": [
                {
                    "result": [seq.to_dict(include_logits=True) for seq in seg.result_sequences],
                    "proposals": [seq.to_dict(include_logits=True) for seq in seg.proposal_sequences],
                }
                for seg in self.segments
            ],
            "energy_scores": self.energy_scores.copy(),
        }

    def _effective_lr(self, temperature: float, soft: float) -> float:
        """Effective LR, optionally scaled: ``lr * ((1 - soft) + soft * temp)``."""
        lr = self.config.lr
        if self.config.scale_lr_by_temperature:
            lr_scale = (1.0 - soft) + (soft * temperature)
            lr_scale = max(lr_scale, self.config.min_lr_scale)
            lr *= lr_scale
        return lr

    def _effective_weight(self, constraint: Constraint, step: int) -> float:
        """Return the weight for *constraint* at *step*, using any configured schedule."""
        schedule = self._weight_schedules.get(constraint.label)
        return schedule(step, self.config.num_steps) if schedule else constraint.weight

    def _sync_proposals_to_results(self) -> None:
        """1:1 copy proposals → results across all segments so snapshots aren't stale."""
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        for seg in self.segments:
            for k in range(self.num_results):
                seg.result_sequences[k] = copy.deepcopy(seg.proposal_sequences[k])


def _init_logits(
    num_positions: int,
    vocab_size: int,
    *,
    initial_logits: np.ndarray | None = None,
    logit_bias: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    fixed_positions: list[int] | None = None,
    gumbel_alpha: float = 1.0,
    softmax_init_positions: list[int] | None = None,
) -> np.ndarray:
    """Build ``(num_positions, vocab_size)`` initial logits.

    With ``initial_logits``: copies the matrix, adds any bias, then applies
    Gumbel + softmax at ``softmax_init_positions`` only. Without:
    Gumbel/alpha + bias (original path).
    """
    shape = (num_positions, vocab_size)

    if initial_logits is not None:
        logits: np.ndarray = initial_logits.copy()
        if logit_bias is not None:
            logits = logits + logit_bias
        if softmax_init_positions and rng is not None:
            sp = sorted(softmax_init_positions)
            logits[sp] = softmax(logits[sp] + rng.gumbel(size=(len(sp), vocab_size)))
        return logits

    logits = rng.gumbel(size=shape) / gumbel_alpha if rng is not None else np.zeros(shape, dtype=np.float64)
    if fixed_positions:
        logits[fixed_positions] = 0.0
    if logit_bias is not None:
        logits = logits + logit_bias
    return logits
