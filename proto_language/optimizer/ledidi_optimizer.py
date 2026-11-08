"""Ledidi optimizer for designing a small number of edits to a template sequence.

Provides ``LedidiOptimizer`` (registered under key ``"ledidi"``) and its
``LedidiOptimizerConfig``. This is a reimplementation of Ledidi (Schreiber et
al. 2025, doi:10.1101/2025.04.22.650035) on top of Proto's gradient providers.
The template is the target segment's current sequence. Each trajectory learns a
weight matrix ``W`` that is added to ``log(X + eps)``, where ``X`` is the one-hot
template. Every step draws ``batch_size`` hard sequences from the Gumbel-softmax
distribution over those logits, scores them with the differentiable constraints
through a straight-through estimator, and penalizes each sample by ``l`` times
its number of edits. ``W`` is updated with AdamW on the batch-mean loss.
Trajectories stop after ``early_stopping_iter`` steps without improving their
batch-mean loss, and each returns the batch of samples from its best step, so
the optimizer produces ``n_repeats * batch_size`` results.

Examples:
    >>> from proto_language.constraint import MalinoisActivityConfig, malinois_activity_constraint
    >>> from proto_language.core import Constraint, Construct, Program, Segment
    >>> from proto_language.generator import PositionWeightGenerator, PositionWeightGeneratorConfig
    >>> from proto_language.optimizer import LedidiOptimizer, LedidiOptimizerConfig
    >>> seg = Segment(sequence="ACGT" * 50, sequence_type="dna", label="enhancer")
    >>> gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    >>> gen.assign(seg)
    >>> con = Constraint(
    ...     inputs=[seg],
    ...     function=malinois_activity_constraint,
    ...     function_config=MalinoisActivityConfig(cell_type="K562", direction="max"),
    ...     label="malinois_k562_max",
    ... )
    >>> optimizer = LedidiOptimizer(
    ...     target_segment=seg,
    ...     constructs=[Construct([seg])],
    ...     generators=[gen],
    ...     constraints=[con],
    ...     config=LedidiOptimizerConfig(n_repeats=2, batch_size=16),
    ... )
    >>> optimizer.num_results  # n_repeats * batch_size
    32
    >>> program = Program([optimizer], num_results=32)
    >>> # program.run()  # needs GPU
"""

import copy
import logging
from collections.abc import Callable
from typing import Any, final

import numpy as np
from pydantic import ValidationInfo, field_validator, model_validator

from proto_language.core import Constraint, Construct, Generator, Optimizer, Segment
from proto_language.generator import PositionWeightGenerator
from proto_language.optimizer.constraint_compiler import (
    GradientProvider,
    compile_gradient_providers,
    constraint_supports_compiled_gradient,
)
from proto_language.optimizer.optimizer_registry import optimizer
from proto_language.utils.base import BaseOptimizerConfig, ConfigField
from proto_language.utils.ml_optimizers import Adam, AdamConfig

logger = logging.getLogger(__name__)

# Logit given to the non-template characters at fixed positions, so the template character is
# always drawn. Ledidi uses -inf; tool inputs travel as JSON, which has no infinity.
_FIXED_LOGIT = -1.0e4


class LedidiOptimizerConfig(BaseOptimizerConfig):
    """Configuration for Ledidi edit design.

    Field names and defaults follow the ``ledidi`` package. ``weight_decay`` is the
    ``torch.optim.AdamW`` default that ``ledidi`` uses without exposing it.

    Attributes:
        n_repeats (int): Independent trajectories, each editing its own template.
        batch_size (int): Sequences sampled per trajectory per step; the loss is averaged over them,
            and each trajectory returns this many sequences.
        num_results (int | None): Always ``n_repeats * batch_size``; set automatically.
        max_iter (int): Maximum optimization steps.
        early_stopping_iter (int): Steps without improvement in a trajectory's batch-mean loss before
            that trajectory stops.
        lr (float): AdamW learning rate.
        weight_decay (float): AdamW decoupled weight decay.
        adam_config (AdamConfig): Adam betas and epsilon.
        tau (float): Gumbel-softmax temperature.
        l (float): Weight on the number of edits relative to the constraint loss.
        eps (float): Added to the one-hot template before the log; smaller values make edits
            harder to propose.
        fixed_positions (list[int] | None): Zero-based positions that are never edited.
        initial_weights (list[list[float]] | None): Starting weight matrix ``(L, |vocab|)`` shared
            by every trajectory; positive values make an edit to that character more likely.
        tracking_interval (int): Steps between progress snapshots and log lines.
        track_proposals (bool): Unused; Ledidi records per-trajectory statistics in each snapshot instead.
    """

    n_repeats: int = ConfigField(
        default=1,
        ge=1,
        title="Repeats",
        description="Independent trajectories, each editing its own template sequence.",
    )
    batch_size: int = ConfigField(
        default=16,
        ge=1,
        title="Batch Size",
        description="Sequences sampled per trajectory per step; each trajectory returns this many.",
    )
    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Always n_repeats x batch_size; set automatically.",
    )
    max_iter: int = ConfigField(
        default=1000,
        ge=1,
        title="Max Iterations",
        description="Maximum number of optimization steps.",
    )
    early_stopping_iter: int = ConfigField(
        default=100,
        ge=1,
        title="Early Stopping Iterations",
        description="Steps without improvement in a trajectory's batch-mean loss before it stops.",
    )
    lr: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Learning Rate",
        description="AdamW learning rate for the edit weight matrix.",
    )
    weight_decay: float = ConfigField(
        default=0.01,
        ge=0.0,
        title="Weight Decay",
        description="AdamW decoupled weight decay (torch's AdamW default, which ledidi uses).",
    )
    adam_config: AdamConfig = ConfigField(
        default_factory=AdamConfig,
        title="Adam Hyperparameters",
        description="Beta and epsilon parameters of the AdamW update.",
    )
    tau: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Tau",
        description="Gumbel-softmax temperature; changes the straight-through gradient, not the drawn sequences.",
    )
    l: float = ConfigField(  # noqa: E741 -- ledidi's name for this parameter
        default=0.1,
        ge=0.0,
        title="Edit Penalty",
        description="Weight on the number of edits relative to the constraint loss. 0 disables it.",
    )
    eps: float = ConfigField(
        default=1e-4,
        gt=0.0,
        title="Epsilon",
        description="Added to the one-hot template before the log; smaller values make edits harder to propose.",
    )
    fixed_positions: list[int] | None = ConfigField(
        default=None,
        title="Fixed Positions",
        description="Zero-based positions that are never edited.",
    )
    initial_weights: list[list[float]] | None = ConfigField(
        default=None,
        title="Initial Weights",
        description="Starting weight matrix (rows=positions, cols=vocab); positive values favor that edit.",
    )

    @field_validator("num_results")
    @classmethod
    def _set_num_results(cls, value: int | None, info: ValidationInfo) -> int:
        """Derive ``num_results`` from ``n_repeats`` and ``batch_size``."""
        expected = int(info.data.get("n_repeats", 1)) * int(info.data.get("batch_size", 16))
        if value is not None and value != expected:
            raise ValueError(f"num_results={value} must equal n_repeats * batch_size = {expected}; leave it unset.")
        return expected

    @model_validator(mode="after")
    def _sync_num_results(self) -> "LedidiOptimizerConfig":
        """Keep ``num_results`` equal to ``n_repeats * batch_size`` when either is reassigned."""
        expected = self.n_repeats * self.batch_size
        if self.num_results != expected:
            # Written directly: assigning would re-run validation on this model and recurse.
            self.__dict__["num_results"] = expected
        return self

    @field_validator("initial_weights")
    @classmethod
    def _check_initial_weights(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        """Reject a ragged or non-finite weight matrix; its shape is checked against the segment at init."""
        if value is not None:
            try:
                matrix = np.asarray(value, dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"initial_weights must be a rectangular 2-D matrix: {exc}") from exc
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError("initial_weights must be a finite rectangular 2-D matrix.")
        return value


@optimizer(
    key="ledidi",
    label="Ledidi Optimizer",
    config=LedidiOptimizerConfig,
    description="Designs a small number of edits to a template sequence: learns per-position edit weights, samples hard edited sequences from a Gumbel-softmax, and descends the differentiable constraint loss plus a penalty on the number of edits (Ledidi).",
    compatible_generators=["position-weight"],
    required_constraint_mode="gradient",
    targets_single_segment=True,
)
@final
class LedidiOptimizer(Optimizer):
    """Ledidi edit design on a single segment.

    The template of trajectory ``r`` is the ``r``-th starting sequence of the target segment
    (the segment's sequence, or a previous stage's results, cycled). Proposals are laid out
    in blocks: trajectory ``r`` owns proposals and results ``r * batch_size`` through
    ``(r + 1) * batch_size - 1``.

    Each step: (1) draw Gumbel noise ``G`` and form ``z = log(X + eps) + W + G`` per sample;
    (2) write ``z`` into the proposal logits and decode ``argmax(z)`` into the proposal
    sequences; (3) call every gradient provider with ``temperature=tau, soft=1, hard=1``,
    which is the straight-through Gumbel-softmax estimator, to get each sample's weighted
    constraint loss and its gradient with respect to ``z``; (4) add the gradient of
    ``l * edits`` through the same estimator; (5) average over the batch and take one AdamW
    step on ``W``. A trajectory keeps the batch from its lowest batch-mean loss and stops after
    ``early_stopping_iter`` steps without improvement. The unedited template, scored once
    before the first step, is the starting best, so a trajectory that never improves returns
    copies of its template.
    Providers score the whole proposal pool, so a trajectory that has stopped is still sampled
    and scored until every trajectory stops; its weights and results no longer change. Other
    segments in the constructs keep, in every slot of trajectory ``r``, the sequence that
    arrived with template ``r``.

    The ``PositionWeightGenerator`` marks the target segment, as single-target optimizers
    require; its decoding settings are not used, because results are the sampled sequences.
    Every constraint must be differentiable, and providers must apply the standard relaxation
    (``softmax(logits / temperature)`` with a straight-through hard forward when ``hard=1``).

    Attributes:
        config (LedidiOptimizerConfig): Optimizer configuration.

    Examples:
        >>> from proto_language.constraint import MalinoisActivityConfig, malinois_activity_constraint
        >>> from proto_language.core import Constraint, Construct, Program, Segment
        >>> from proto_language.generator import PositionWeightGenerator, PositionWeightGeneratorConfig
        >>> from proto_language.optimizer import LedidiOptimizer, LedidiOptimizerConfig
        >>> seg = Segment(sequence="ACGT" * 50, sequence_type="dna", label="enhancer")
        >>> gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        >>> gen.assign(seg)
        >>> up = Constraint(
        ...     inputs=[seg],
        ...     function=malinois_activity_constraint,
        ...     function_config=MalinoisActivityConfig(cell_type="K562", direction="max"),
        ...     label="k562_max",
        ... )
        >>> down = Constraint(
        ...     inputs=[seg],
        ...     function=malinois_activity_constraint,
        ...     function_config=MalinoisActivityConfig(cell_type="HepG2", direction="min"),
        ...     label="hepg2_min",
        ... )
        >>> optimizer = LedidiOptimizer(
        ...     target_segment=seg,
        ...     constructs=[Construct([seg])],
        ...     generators=[gen],
        ...     constraints=[up, down],
        ...     config=LedidiOptimizerConfig(n_repeats=4, batch_size=16, l=0.1),
        ... )
        >>> program = Program([optimizer], num_results=64)
        >>> # program.run()  # needs GPU
    """

    # Class attribute required by OptimizerRegistry
    config_class = LedidiOptimizerConfig
    config: LedidiOptimizerConfig

    def __init__(
        self,
        target_segment: Segment,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: LedidiOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the Ledidi optimizer.

        Args:
            target_segment (Segment): The segment to edit. Must belong to one of the constructs.
            constructs (list[Construct]): Constructs to optimize.
            generators (list[Generator]): Exactly one PositionWeightGenerator assigned to the target.
            constraints (list[Constraint]): Differentiable constraints that form the output loss.
            config (LedidiOptimizerConfig): Configuration.
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked steps.
            clear_tool_cache (int | bool | list[str]): (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If the generator count/type is wrong, a constraint is not differentiable,
                or a fixed position or the initial weights do not fit the target segment.
        """
        if len(generators) != 1:
            raise ValueError(f"LedidiOptimizer requires exactly one generator, got {len(generators)}.")

        self.config = config
        self.target_segment: Segment = target_segment
        self.generator: Generator = generators[0]
        unsupported = []
        for constraint in constraints:
            if constraint.threshold is not None:
                unsupported.append(f"Constraint '{constraint.label}' is a filter; Ledidi has no filter step.")
                continue
            ok, reason = constraint_supports_compiled_gradient(constraint, target_segment)
            if not ok:
                unsupported.append(reason or f"Constraint '{constraint.label}' does not support gradient evaluation.")
        if unsupported:
            raise ValueError("LedidiOptimizer requires differentiable constraints: " + "; ".join(unsupported))

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

        self.num_steps: int = config.max_iter
        self._gradient_providers: list[GradientProvider] = compile_gradient_providers(constraints, target_segment)

    def _validate_optimizer(self) -> None:
        """Extend base validation with checks against the target segment."""
        super()._validate_optimizer()
        self._validate_target_segment(self.target_segment)

        if not isinstance(self.generator, PositionWeightGenerator):
            raise ValueError(
                f"LedidiOptimizer requires a PositionWeightGenerator, got {self.generator.__class__.__name__}."
            )
        self._check_segment_fit()

    def _check_segment_fit(self) -> None:
        """Check ``fixed_positions`` and ``initial_weights`` against the target segment."""
        seq_len = self.target_segment.sequence_length
        if self.config.fixed_positions:
            out_of_bounds = [p for p in self.config.fixed_positions if p < 0 or p >= seq_len]
            if out_of_bounds:
                raise ValueError(f"fixed_positions {out_of_bounds} out of bounds for segment length {seq_len}.")

        if self.config.initial_weights is not None:
            expected = (seq_len, len(self.target_segment.ordered_vocab()))
            actual = np.asarray(self.config.initial_weights).shape
            if actual != expected:
                raise ValueError(f"initial_weights shape {actual} does not match target segment {expected}.")

    def _validate_component_compatibility(self) -> None:
        """Validate dependencies while allowing compiler-backed gradient constraints."""
        from proto_language.constraint.constraint_registry import ConstraintRegistry
        from proto_language.generator.generator_registry import GeneratorRegistry
        from proto_language.optimizer.optimizer_registry import OptimizerRegistry

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

        for con in self.constraints:
            con_key = ConstraintRegistry.find_key(con)
            if con_key and ConstraintRegistry.get(con_key).mode in ("gradient", "dual"):
                continue
            ok, reason = constraint_supports_compiled_gradient(con, self.target_segment)
            if not ok:
                detail = f": {reason}" if reason else ""
                raise ValueError(
                    f"Constraint '{con.label}' does not support gradient evaluation, required by {opt_label}{detail}"
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
        """Execute Ledidi edit design.

        Raises:
            ValueError: If a template contains characters outside the segment vocabulary, or a
                provider returns a non-finite gradient.
        """
        self._prepare_run()
        cfg = self.config
        target = self.target_segment
        n_repeats, batch_size = cfg.n_repeats, cfg.batch_size
        if self.num_results != n_repeats * batch_size or len(target.proposal_sequences) != n_repeats * batch_size:
            raise ValueError(
                f"n_repeats * batch_size = {n_repeats * batch_size} no longer matches the {self.num_results} result "
                "slots this optimizer was built with; build a new LedidiOptimizer after changing either."
            )
        self._check_segment_fit()

        # Lay every segment's pool out in blocks, so that all slots of trajectory r carry the r-th
        # incoming sequence of every segment and partner segments stay paired with their template.
        for seg in self.segments:
            starts = seg.proposal_sequences[:n_repeats]
            seg.proposal_sequences = [copy.deepcopy(starts[k // batch_size]) for k in range(n_repeats * batch_size)]
            seg.result_sequences = [copy.deepcopy(seq) for seq in seg.proposal_sequences]
        vocab = target.ordered_vocab()
        index = {char: i for i, char in enumerate(vocab)}
        seq_len, n_chars = target.sequence_length, len(vocab)

        logger.info(
            f"LedidiOptimizer: {n_repeats} trajectories x {batch_size} samples, max_iter={cfg.max_iter}, "
            f"early_stopping_iter={cfg.early_stopping_iter}, l={cfg.l}, tau={cfg.tau}, lr={cfg.lr}, "
            f"{len(self.constraints)} constraints"
        )

        templates = np.zeros((n_repeats, seq_len), dtype=np.int64)
        for r in range(n_repeats):
            sequence = target.proposal_sequences[r * batch_size].sequence
            bad = sorted({char for char in sequence if char not in index})
            if bad or len(sequence) != seq_len:
                raise ValueError(
                    f"Template {r} for segment '{target.label or 'unlabeled'}' must be {seq_len} characters from "
                    f"{''.join(vocab)}; found invalid characters {bad}."
                )
            templates[r] = [index[char] for char in sequence]

        one_hot = np.eye(n_chars)[templates]  # (R, L, V)
        base = np.log(one_hot + cfg.eps)
        fixed = np.zeros(seq_len, dtype=bool)
        if cfg.fixed_positions:
            fixed[cfg.fixed_positions] = True
        fixed_logits = np.where(one_hot[:, fixed] > 0, 0.0, _FIXED_LOGIT)  # (R, n_fixed, V)

        weights = np.zeros((n_repeats, seq_len, n_chars))
        if cfg.initial_weights is not None:
            weights += np.asarray(cfg.initial_weights, dtype=np.float64)
        adam = Adam(cfg.adam_config)
        adam.reset()
        rng = np.random.default_rng(self.seed)

        # Step 0: the unedited templates are the starting best batches.
        z = np.broadcast_to(base[:, None], (n_repeats, batch_size, seq_len, n_chars)).copy()
        z[:, :, fixed] = fixed_logits[:, None]
        output_loss, _ = self._evaluate(z, 0)
        edits = np.zeros((n_repeats, batch_size))
        totals = output_loss + cfg.l * edits
        best_loss = totals.mean(axis=1)
        best_step = np.zeros(n_repeats, dtype=np.int64)
        best_totals = totals.copy()
        best_output = output_loss.copy()
        best_edits = edits.copy()
        best_proposals = [copy.deepcopy(seq) for seq in target.proposal_sequences]
        stalled = np.zeros(n_repeats, dtype=np.int64)
        active = np.ones(n_repeats, dtype=bool)
        self._set_results(best_proposals, best_totals, best_output, best_edits, best_step, templates, vocab)
        self._snapshot(0, totals, edits, active, active)

        step = 0
        for step in range(1, cfg.max_iter + 1):
            z = base[:, None] + weights[:, None] + rng.gumbel(size=(n_repeats, batch_size, seq_len, n_chars))
            z[:, :, fixed] = fixed_logits[:, None]
            output_loss, output_grad = self._evaluate(z, step)

            samples = z.argmax(axis=-1)  # (R, B, L)
            edits = (samples != templates[:, None]).sum(axis=-1).astype(np.float64)
            edit_grad = _edit_gradient(z, one_hot[:, None], cfg.l, cfg.tau)
            grad = (output_grad + edit_grad).mean(axis=1)
            grad[:, fixed] = 0.0
            totals = output_loss + cfg.l * edits
            batch_loss = totals.mean(axis=1)

            ran = active.copy()
            for r in np.flatnonzero(active).tolist():
                weights[r] *= 1.0 - cfg.lr * cfg.weight_decay
                weights[r] = adam.step(weights[r], grad[r], cfg.lr, trajectory=r, step=step)

                if batch_loss[r] < best_loss[r]:
                    best_loss[r] = batch_loss[r]
                    best_step[r] = step
                    best_totals[r], best_output[r], best_edits[r] = totals[r], output_loss[r], edits[r]
                    block = range(r * batch_size, (r + 1) * batch_size)
                    for k in block:
                        best_proposals[k] = copy.deepcopy(target.proposal_sequences[k])
                    stalled[r] = 0
                else:
                    stalled[r] += 1
                    if stalled[r] == cfg.early_stopping_iter:
                        active[r] = False

            last = not active.any() or step == cfg.max_iter
            if step % self.tracking_interval == 0 or last:
                self._set_results(best_proposals, best_totals, best_output, best_edits, best_step, templates, vocab)
                self._snapshot(step, totals, edits, ran, active)
                self._log_progress(step, totals, edits, best_loss, ran, active)

            self._clear_tool_cache()
            if not active.any():
                break

        logger.info(f"LedidiOptimizer: finished after {step} steps; best batch-mean losses {np.round(best_loss, 4)}")

    # =============================================================================
    # Private helpers
    # =============================================================================

    def _evaluate(self, z: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Score sampled logits with every provider.

        Writes ``z`` into the proposal logits and ``argmax(z)`` into the proposal sequences, then
        calls each provider with the straight-through Gumbel-softmax settings.

        Args:
            z (np.ndarray): Perturbed logits, shape ``(n_repeats, batch_size, L, |vocab|)``.
            step (int): Optimizer step, passed on to the providers.

        Returns:
            tuple[np.ndarray, np.ndarray]: Weighted output loss per sample ``(R, B)`` and its
                gradient with respect to ``z``.
        """
        n_repeats, batch_size = z.shape[:2]
        vocab = self.target_segment.ordered_vocab()
        flat = z.reshape(n_repeats * batch_size, *z.shape[2:])
        for k, seq in enumerate(self.target_segment.proposal_sequences):
            seq.logits = flat[k].copy()
            seq.sequence = "".join(vocab[i] for i in flat[k].argmax(axis=-1))

        loss = np.zeros(len(flat))
        grad = np.zeros_like(flat)
        for provider in self._gradient_providers:
            output = provider.compute(
                temperature=self.config.tau,
                soft=1.0,
                hard=1.0,
                step=step,
                effective_weight=lambda constraint, _step: constraint.weight,
            )
            for k, raw in enumerate(output.gradients):
                gradient = np.asarray(raw, dtype=np.float64)
                if not np.isfinite(gradient).all():
                    raise ValueError(f"Non-finite gradient from '{output.label}' (proposal {k}).")
                grad[k] += output.weight * gradient
            loss += np.asarray(output.losses, dtype=np.float64)
        return loss.reshape(n_repeats, batch_size), grad.reshape(z.shape)

    def _set_results(
        self,
        best_proposals: list[Any],
        best_totals: np.ndarray,
        best_output: np.ndarray,
        best_edits: np.ndarray,
        best_step: np.ndarray,
        templates: np.ndarray,
        vocab: list[str],
    ) -> None:
        """Write each trajectory's best batch into the result and proposal pools.

        Args:
            best_proposals (list[Any]): Best proposal ``Sequence`` per slot, in block order.
            best_totals (np.ndarray): Total loss per sample, shape ``(R, B)``.
            best_output (np.ndarray): Weighted constraint loss per sample, shape ``(R, B)``.
            best_edits (np.ndarray): Number of edits per sample, shape ``(R, B)``.
            best_step (np.ndarray): Step each trajectory's best batch was drawn at, shape ``(R,)``.
            templates (np.ndarray): Template character indices, shape ``(R, L)``.
            vocab (list[str]): Segment vocabulary.
        """
        batch_size = self.config.batch_size
        eye = np.eye(len(vocab))
        results = []
        for k, proposal in enumerate(best_proposals):
            r, b = divmod(k, batch_size)
            seq = copy.deepcopy(proposal)
            codes = np.array([vocab.index(char) for char in seq.sequence])
            seq.logits = np.log(eye[codes] + self.config.eps)
            edited = np.flatnonzero(codes != templates[r])
            seq._metadata["ledidi"] = {
                "repeat": r,
                "best_step": int(best_step[r]),
                "num_edits": int(best_edits[r, b]),
                "edits": [(int(p), vocab[templates[r, p]], vocab[codes[p]]) for p in edited],
                "output_loss": float(best_output[r, b]),
                "edit_loss": float(self.config.l * best_edits[r, b]),
                "total_loss": float(best_totals[r, b]),
            }
            results.append(seq)
        self.target_segment.result_sequences = results
        self.energy_scores = [float(x) for x in best_totals.ravel()]
        for seg in self.segments:
            if seg is not self.target_segment:
                seg.result_sequences = [copy.deepcopy(s) for s in seg.proposal_sequences]

    def _snapshot(self, step: int, totals: np.ndarray, edits: np.ndarray, ran: np.ndarray, active: np.ndarray) -> None:
        """Save the current best results with this step's statistics; ``None`` for stopped trajectories."""
        self._save_progress_snapshot(
            time_step=step,
            optimizer_metadata={
                "type": "ledidi",
                "max_iter": self.config.max_iter,
                "n_repeats": self.config.n_repeats,
                "batch_size": self.config.batch_size,
                "mean_edits": [float(x) if r else None for x, r in zip(edits.mean(axis=1), ran, strict=True)],
                "mean_total_loss": [float(x) if r else None for x, r in zip(totals.mean(axis=1), ran, strict=True)],
                "active_trajectories": int(active.sum()),
            },
        )

    def _log_progress(
        self,
        step: int,
        totals: np.ndarray,
        edits: np.ndarray,
        best_loss: np.ndarray,
        ran: np.ndarray,
        active: np.ndarray,
    ) -> None:
        """Log optimization progress over the trajectories that ran this step."""
        logger.info(
            f"Step {step}/{self.config.max_iter}: mean edits {edits[ran].mean():.2f}, "
            f"mean loss {totals[ran].mean():.4f}, "
            f"best batch-mean loss {best_loss.min():.4f}, {int(active.sum())}/{len(active)} trajectories active"
        )
        if self.custom_logging:
            self.custom_logging(step, self.segments)


def _edit_gradient(z: np.ndarray, template: np.ndarray, l: float, tau: float) -> np.ndarray:  # noqa: E741
    """Straight-through gradient of ``l * sum|onehot(argmax z) - X| / 2`` with respect to ``z``.

    The hard sample's L1 distance to the template is passed back through
    ``softmax(z / tau)``, as ``torch.nn.functional.gumbel_softmax(hard=True)`` does.

    Args:
        z (np.ndarray): Perturbed logits, ``(..., L, |vocab|)``.
        template (np.ndarray): One-hot template, broadcastable to ``z``.
        l (float): Edit penalty weight.
        tau (float): Gumbel-softmax temperature.

    Returns:
        np.ndarray: Gradient with the shape of ``z``.
    """
    y_hard = np.eye(z.shape[-1])[z.argmax(axis=-1)]
    y_soft = _softmax(z / tau)
    upstream = l * np.sign(y_hard - template) / 2.0
    grad: np.ndarray = y_soft * (upstream - (upstream * y_soft).sum(axis=-1, keepdims=True)) / tau
    return grad


def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax over the last axis."""
    shifted = np.exp(x - x.max(axis=-1, keepdims=True))
    probs: np.ndarray = shifted / shifted.sum(axis=-1, keepdims=True)
    return probs
