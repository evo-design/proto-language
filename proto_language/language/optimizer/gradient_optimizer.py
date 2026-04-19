"""Gradient-based optimizer for differentiable sequence design."""

import copy
import logging
from collections.abc import Callable
from typing import Any, Literal, final

import numpy as np
from pydantic import model_validator

from proto_language.base_config import BaseConfig, BaseOptimizerConfig, ConfigField
from proto_language.language.core import Constraint, Construct, Generator, Optimizer, Segment
from proto_language.language.core.constraint import GradientResult
from proto_language.language.generator import PositionWeightGenerator
from proto_language.language.optimizer.optimizer_registry import optimizer
from proto_language.utils.gradients import MERGERS, GradientMergerName, align_norms, normalize_gradient
from proto_language.utils.scheduling import SCHEDULES, Schedule, ScheduleName

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
        schedule (ScheduleName): Interpolation between start and end.
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
    schedule: ScheduleName = ConfigField(
        default="linear",
        title="Schedule",
        description="Interpolation schedule between start_weight and end_weight.",
    )

    @model_validator(mode="after")
    def _check_exponential_endpoints(self) -> "ConstraintWeightSchedule":
        if self.schedule == "exponential" and min(self.start_weight, self.end_weight) <= 0.0:
            raise ValueError("schedule='exponential' requires start_weight > 0 and end_weight > 0.")
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
        initial_logit_bias (float): One-time logit bias at starting positions.
        soft_start (float): Soft blending at step 1 (0=hard, 1=softmax).
        soft_end (float): Soft blending at final step.
        temperature_start (float): Temperature at step 1.
        temperature_end (float): Temperature at final step.
        schedule (ScheduleName): Temperature decay schedule.
        merger (GradientMergerName): Gradient merging strategy.
        norm_alignment (Literal["none", "unit", "match_first"]): Per-constraint
            gradient normalization before merging.
        normalize_gradients (bool): Normalize merged gradient before update.
        normalize_mode (Literal["unit", "sqrt_length"]): Normalization formula.
            ``"unit"`` = L2 to magnitude 1. ``"sqrt_length"`` = ``g * sqrt(eff_L) / ||g||``.
        fixed_positions (list[int] | None): Positions to freeze during optimization.
        scale_lr_by_temperature (bool): Multiply LR by soft/temperature blending.
        min_lr_scale (float): Floor for effective LR scale.
        tracking_interval (int): Steps between progress snapshots.
        track_proposals (bool): Record per-proposal results in history.
        constraint_weight_schedules (list[ConstraintWeightSchedule] | None): Per-label
            weight schedules that override ``Constraint.weight`` each step.
        gumbel_logit_init (bool): If True, add Gumbel(0,1) noise per position on top of
            the bias — gives parallel trajectories stochastic starts.
        gumbel_init_alpha (float): Divisor for Gumbel init noise. ``1.0`` = unscaled.

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
    initial_logit_bias: float = ConfigField(
        default=0.0,
        ge=0.0,
        title="Initial Logit Bias",
        description="One-time logit bias at starting-sequence positions.",
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
    schedule: ScheduleName = ConfigField(
        default="constant",
        title="Temperature Schedule",
        description="Temperature decay schedule across steps.",
    )
    merger: GradientMergerName = ConfigField(
        default="weighted_sum",
        title="Gradient Merger",
        description="Strategy for merging gradients from multiple constraints.",
    )
    norm_alignment: Literal["none", "unit", "match_first"] = ConfigField(
        default="none",
        title="Norm Alignment",
        description="Per-constraint gradient normalization before merging.",
        advanced=True,
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
        description="Sequence positions to freeze. Pair with initial_logit_bias > 0 to anchor them.",
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
    constraint_weight_schedules: list[ConstraintWeightSchedule] | None = ConfigField(
        default=None,
        title="Constraint Weight Schedules",
        description="Per-label weight schedules that override Constraint.weight each step.",
        advanced=True,
    )
    gumbel_logit_init: bool = ConfigField(
        default=False,
        title="Gumbel Logit Init",
        description="Add Gumbel(0,1) noise per position on top of the bias at init.",
        advanced=True,
    )
    gumbel_init_alpha: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Gumbel Init Alpha",
        description="Divisor for Gumbel init noise. 1.0 = unscaled.",
        advanced=True,
    )

    @classmethod
    def germinal_logit_preset(cls) -> "GradientOptimizerConfig":
        """Logit hallucination phase: 65 SGD steps with soft 0→1, naturalness weight 0.2→0.4.

        Ramps naturalness weight linearly — constraint must be labeled ``"ablang"``.
        Initial logits get Gumbel(0,1) noise so parallel trajectories diverge.
        """
        return cls(
            num_steps=65,
            lr=0.1,
            soft_start=0.0,
            soft_end=1.0,
            temperature_start=1.0,
            temperature_end=1.0,
            schedule="constant",
            merger="pcgrad",
            norm_alignment="match_first",
            normalize_mode="sqrt_length",
            constraint_weight_schedules=[
                ConstraintWeightSchedule(constraint_label="ablang", start_weight=0.2, end_weight=0.4, schedule="linear")
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
            schedule="quadratic",
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

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: GradientOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the gradient optimizer.

        Args:
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
            ValueError: If no gradient-capable constraints, generator validation fails,
                or a constraint's inputs do not include the target segment.
        """
        self.config = config
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

        # Extract gradient-capable constraints
        self._gradient_constraints = [c for c in constraints if c.supports_gradient]
        if not self._gradient_constraints:
            raise ValueError("GradientOptimizer requires at least one gradient-capable constraint")

        # Validate generator
        pwg_generators = [g for g in generators if isinstance(g, PositionWeightGenerator)]
        if len(pwg_generators) != 1:
            raise ValueError(
                f"GradientOptimizer requires exactly one PositionWeightGenerator, got {len(pwg_generators)}"
            )
        self._generator = pwg_generators[0]

        # Validate fixed_positions are within the target segment's sequence length.
        if config.fixed_positions:
            seq_len = self._generator.segment.sequence_length
            out_of_bounds = [p for p in config.fixed_positions if p < 0 or p >= seq_len]
            if out_of_bounds:
                raise ValueError(f"fixed_positions {out_of_bounds} out of bounds for segment length {seq_len}")

        # Resolve which gradient index corresponds to the target segment in each constraint
        target_seg = self._generator.segment
        self._gradient_indices: list[int] = []
        for c in self._gradient_constraints:
            if target_seg not in c.inputs:
                raise ValueError(f"Constraint '{c.label}' inputs do not include the target segment")
            self._gradient_indices.append(c.inputs.index(target_seg))

        # Warn about non-gradient constraints that will be ignored
        skipped = [c.label for c in constraints if not c.supports_gradient]
        if skipped:
            logger.warning(f"GradientOptimizer ignoring non-gradient constraints: {skipped}")

        # Build merger and schedule
        self._merger = MERGERS[config.merger]()
        self._temperature_schedule = SCHEDULES[config.schedule](config.temperature_start, config.temperature_end)

        # Missing labels warn (not error) so presets remain portable across constraint sets.
        known = {c.label for c in self._gradient_constraints}
        self._weight_schedules: dict[str, Schedule] = {}
        for e in config.constraint_weight_schedules or []:
            if e.constraint_label in known:
                self._weight_schedules[e.constraint_label] = SCHEDULES[e.schedule](e.start_weight, e.end_weight)
            else:
                logger.warning(f"Unknown weight-schedule label '{e.constraint_label}'; ignored.")

    def run(self) -> None:
        """Execute gradient optimization.

        Each step:
        1. Compute soft and temperature from linear/scheduled interpolation
        2. Compute gradients from all gradient-capable constraints
        3. Align norms, apply weights, merge (PCGrad/MGDA/weighted sum)
        4. Normalize merged gradient, zero fixed positions
        5. Update ``seq.logits`` via SGD: ``logits -= lr * gradient``
        6. At tracked steps: discretize via generator, save snapshot
        """
        self._prepare_run()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing
        target = self._generator.segment

        vocab = target.ordered_vocab()
        init_rng = np.random.default_rng(self.config.seed) if self.config.gumbel_logit_init else None
        for seq in target.proposal_sequences:
            if seq.logits is None:
                seq.logits = _init_logits(
                    target.sequence_length,
                    seq.sequence,
                    self.config.initial_logit_bias,
                    vocab,
                    rng=init_rng,
                    fixed_positions=self.config.fixed_positions,
                    gumbel_alpha=self.config.gumbel_init_alpha,
                )

        self._generator.sample()
        self._proposal_outcomes = ["accepted"] * self.num_proposals
        self._proposal_energy_scores = list(self.energy_scores)
        self._sync_target_proposals_to_results(target)
        self._save_progress_snapshot(time_step=0)

        if self.verbose:
            logger.info(
                f"GradientOptimizer: {self.num_results} trajectories, {self.config.num_steps} steps, "
                f"soft {self.config.soft_start}→{self.config.soft_end}, "
                f"temp {self.config.temperature_start}→{self.config.temperature_end}"
            )

        all_results: list[list[GradientResult]]
        for step in range(1, self.config.num_steps + 1):
            # 1. Compute soft and temperature from linear/scheduled interpolation
            progress = step / self.config.num_steps
            soft = self.config.soft_start + (self.config.soft_end - self.config.soft_start) * progress
            temp = self._temperature_schedule(step, self.config.num_steps)
            lr = self._effective_lr(temp, soft)

            # 2. Compute gradients from all gradient-capable constraints
            all_results = [c.compute_gradient(temperature=temp, soft=soft) for c in self._gradient_constraints]
            for i, constraint in enumerate(self._gradient_constraints):
                for k, r in enumerate(all_results[i]):
                    if not np.isfinite(r.gradient[self._gradient_indices[i]]).all():
                        raise ValueError(
                            f"Non-finite gradient from '{constraint.label}' at step {step} (proposal {k})."
                        )

            # 3-5. Merge gradients and update logits for each trajectory
            for k in range(self.num_results):
                self._update_trajectory(k, all_results, lr, target, step)

            # Report the same weighted objective that gradient descent is actually minimizing.
            self.energy_scores = [
                sum(
                    self._effective_weight(c, step) * all_results[i][k].loss
                    for i, c in enumerate(self._gradient_constraints)
                )
                for k in range(self.num_results)
            ]
            self._proposal_outcomes = ["accepted"] * self.num_proposals
            self._proposal_energy_scores = list(self.energy_scores)

            # 6. At tracked steps: discretize, sync proposals→results, snapshot
            if step % self.tracking_interval == 0 or step == self.config.num_steps:
                self._generator.sample()
                self._sync_target_proposals_to_results(target)
                self._save_progress_snapshot(time_step=step)
                self._log_progress(step, temp, lr)

            self._clear_tool_cache()

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
        self, k: int, all_results: list[list[GradientResult]], lr: float, target: Segment, step: int
    ) -> None:
        """Align, merge, normalize, and apply one gradient step for trajectory *k*."""
        grads = [all_results[i][k].gradient[self._gradient_indices[i]] for i in range(len(self._gradient_constraints))]
        weights = [self._effective_weight(c, step) for c in self._gradient_constraints]

        # Align norms first so ``match_first`` doesn't wash out weights.
        grads = align_norms(grads, self.config.norm_alignment)
        grads = [g * w for g, w in zip(grads, weights, strict=True)]
        merged = self._merger.merge(grads)

        if self.config.normalize_gradients:
            merged = normalize_gradient(merged, self.config.normalize_mode)
        if self.config.fixed_positions:
            merged[self.config.fixed_positions] = 0.0

        seq = target.proposal_sequences[k]
        assert seq.logits is not None  # noqa: S101 -- guaranteed by initialization
        seq.logits = seq.logits - lr * merged

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

    def _sync_target_proposals_to_results(self, target: Segment) -> None:
        """Copy current proposals into ``result_sequences`` so snapshots aren't stale."""
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        for k in range(self.num_results):
            target.result_sequences[k] = copy.deepcopy(target.proposal_sequences[k])


def _init_logits(
    num_positions: int,
    sequence: str,
    bias: float,
    vocab: list[str],
    *,
    rng: np.random.Generator | None = None,
    fixed_positions: list[int] | None = None,
    gumbel_alpha: float = 1.0,
) -> np.ndarray:
    """Return ``(num_positions, len(vocab))`` logits; ``sequence`` may be empty and only drives the bias.

    When ``rng`` is provided, Gumbel(0,1) noise (divided by ``gumbel_alpha``) is added per position
    so parallel trajectories diverge. Noise is skipped at ``fixed_positions``.
    """
    shape = (num_positions, len(vocab))
    logits = rng.gumbel(size=shape) / gumbel_alpha if rng is not None else np.zeros(shape, dtype=np.float64)
    if fixed_positions:
        logits[fixed_positions] = 0.0
    for pos, char in enumerate(sequence[:num_positions]):
        logits[pos, vocab.index(char)] += bias
    return logits
