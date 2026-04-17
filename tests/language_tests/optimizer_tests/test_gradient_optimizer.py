"""Tests for the GradientOptimizer, multi-stage pipelines, and GPU integration."""

from collections.abc import Callable

import numpy as np
import pytest
from pydantic import BaseModel

from proto_language.language.core import Constraint, Construct, Program, Segment
from proto_language.language.core.constraint import GradientResult
from proto_language.language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
)
from proto_language.language.optimizer import (
    ConstraintWeightSchedule,
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)


class _Cfg(BaseModel):
    """Empty config for mocks."""


def _backward(inputs: tuple, *, config: BaseModel, temperature: float = 1.0, **kwargs: object) -> GradientResult:
    """Gradient pushes logits toward alanine (column 0)."""
    logits = inputs[0].logits
    target = np.zeros_like(logits)
    target[:, 0] = 1.0
    grad = logits - target
    return GradientResult(gradient=(grad,), loss=float(np.mean(grad**2)), metrics={"temperature": temperature})


def _backward_toward_C(inputs: tuple, *, config: BaseModel, **kwargs: object) -> GradientResult:
    """Gradient pushes logits toward cysteine (column 2) — conflicts with _backward."""
    logits = inputs[0].logits
    target = np.zeros_like(logits)
    target[:, 2] = 1.0
    grad = logits - target
    return GradientResult(gradient=(grad,), loss=float(np.mean(grad**2)), metrics={})


def _unit_grad_bwd(inputs: tuple, *, config: BaseModel, **kwargs: object) -> GradientResult:
    """Constant unit gradient — handy for measuring update magnitudes directly."""
    return GradientResult(gradient=(np.ones_like(inputs[0].logits),), loss=0.0, metrics={})


def _scorer(input_sequences: list[tuple], config: BaseModel) -> list[float]:
    return [sum(c != "A" for c in seq.sequence) / max(len(seq.sequence), 1) for (seq,) in input_sequences]


_scorer._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]
_scorer._constraint_num_input_sequences_per_tuple = 1  # type: ignore[attr-defined]


def _make(num_steps: int = 5, num_results: int = 1, seed: int = 42, **kw: object) -> tuple[GradientOptimizer, Segment]:
    """Default-shaped optimizer: ``EVQLV`` protein segment, ``_backward`` + ``_scorer``."""
    seg = Segment(sequence="EVQLV", sequence_type="protein")
    construct = Construct([seg])
    gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    gen.assign(seg)
    con = Constraint(
        inputs=[seg], backward=_backward, backward_config=_Cfg(), function=_scorer, function_config=_Cfg(), label="mock"
    )
    defaults: dict[str, object] = {"lr": 0.1, "beta1": 0.0, "beta2": 0.0}
    defaults.update(kw)
    cfg = GradientOptimizerConfig(num_results=num_results, num_steps=num_steps, seed=seed, **defaults)
    opt = GradientOptimizer(constructs=[construct], generators=[gen], constraints=[con], config=cfg)
    return opt, seg


def _make_optimizer(
    seg: Segment, backward: Callable[..., GradientResult], label: str = "t", weight: float = 1.0, **cfg: object
) -> GradientOptimizer:
    """Minimal single-constraint optimizer on a caller-supplied segment."""
    gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    gen.assign(seg)
    con = Constraint(inputs=[seg], backward=backward, backward_config=_Cfg(), label=label, weight=weight)
    return GradientOptimizer(
        constructs=[Construct([seg])],
        generators=[gen],
        constraints=[con],
        config=GradientOptimizerConfig(num_results=1, **cfg),  # type: ignore[arg-type]
    )


class TestConfig:
    def test_germinal_logit_preset_exact_values(self) -> None:
        """Lock Germinal Stage 0 hyperparameters; a future refactor that breaks parity must fail here."""
        cfg = GradientOptimizerConfig.germinal_logit_preset()
        assert cfg.num_steps == 65 and cfg.lr == 0.1
        assert cfg.beta1 == 0.0 and cfg.beta2 == 0.0  # SGD
        assert (cfg.soft_start, cfg.soft_end) == (0.0, 1.0)
        assert (cfg.temperature_start, cfg.temperature_end) == (1.0, 1.0)
        assert cfg.schedule == "constant"
        assert cfg.merger == "pcgrad"
        assert cfg.norm_alignment == "match_first"
        assert cfg.normalize_mode == "sqrt_length"
        assert cfg.gumbel_logit_init is True
        assert cfg.gumbel_init_alpha == 2.0
        assert cfg.constraint_weight_schedules == [
            ConstraintWeightSchedule(constraint_label="ablang", start_weight=0.2, end_weight=0.4, schedule="linear")
        ]

    def test_germinal_softmax_preset_exact_values(self) -> None:
        """Lock Germinal Stage 1 hyperparameters."""
        cfg = GradientOptimizerConfig.germinal_softmax_preset()
        assert cfg.num_steps == 35 and cfg.lr == 0.1
        assert cfg.beta1 == 0.0 and cfg.beta2 == 0.0
        assert (cfg.soft_start, cfg.soft_end) == (1.0, 1.0)
        assert (cfg.temperature_start, cfg.temperature_end) == (1.0, 0.01)
        assert cfg.schedule == "quadratic"
        assert cfg.merger == "pcgrad"
        assert cfg.norm_alignment == "match_first"
        assert cfg.normalize_mode == "sqrt_length"
        assert cfg.scale_lr_by_temperature is True
        assert cfg.min_lr_scale == 0.01


class TestRun:
    def test_produces_results_with_logits(self) -> None:
        opt, seg = _make()
        opt.run()
        assert seg.result_sequences[0].sequence != ""
        assert seg.result_sequences[0].logits is not None
        assert seg.result_sequences[0].logits.shape == (5, 20)

    def test_loss_decreases(self) -> None:
        opt, _ = _make(num_steps=20, lr=0.05, tracking_interval=5)
        opt.run()
        early = opt.history[1]["results"][0]["energy_score"]
        last = opt.history[-1]["results"][0]["energy_score"]
        assert last < early

    def test_multiple_trajectories(self) -> None:
        opt, seg = _make(num_results=3)
        opt.run()
        assert len(seg.result_sequences) == 3
        assert len(opt.energy_scores) == 3

    def test_rerun_determinism(self) -> None:
        opt, seg = _make()
        opt.run()
        first = seg.result_sequences[0].sequence
        opt.run()
        assert seg.result_sequences[0].sequence == first


class TestSchedules:
    def test_soft_and_temperature_forwarded(self) -> None:
        """Soft ramps linearly; temperature decays via schedule — both forwarded to backward."""
        received_soft: list[float] = []
        received_temp: list[float] = []

        def bwd(
            inputs: tuple, *, config: BaseModel, soft: float = 1.0, temperature: float = 1.0, **kwargs: object
        ) -> GradientResult:
            received_soft.append(soft)
            received_temp.append(temperature)
            return GradientResult(gradient=(np.zeros_like(inputs[0].logits),), loss=0.0, metrics={})

        seg = Segment(sequence="AA", sequence_type="protein")
        opt = _make_optimizer(
            seg,
            bwd,
            num_steps=4,
            soft_start=0.0,
            soft_end=1.0,
            temperature_start=1.0,
            temperature_end=0.1,
            schedule="linear",
        )
        opt.run()
        assert received_soft[0] == pytest.approx(0.25) and received_soft[-1] == pytest.approx(1.0)
        assert received_temp[0] > received_temp[-1]

    def test_temperature_scaling_shrinks_updates(self) -> None:
        """scale_lr_by_temperature=True must shrink the logit delta vs =False under a unit gradient."""

        def _abs_sum(scale: bool) -> float:
            seg = Segment(sequence="AA", sequence_type="protein")
            opt = _make_optimizer(
                seg,
                _unit_grad_bwd,
                num_steps=5,
                lr=1.0,
                beta1=0.0,
                beta2=0.0,
                temperature_end=0.01,
                schedule="quadratic",
                scale_lr_by_temperature=scale,
                normalize_gradients=False,
            )
            opt.run()
            logits = seg.result_sequences[0].logits
            assert logits is not None
            return float(np.abs(logits).sum())

        assert _abs_sum(scale=True) < _abs_sum(scale=False)


class TestFixedPositions:
    def test_logits_unchanged_at_fixed(self) -> None:
        opt, seg = _make(num_steps=20, lr=1.0, initial_logit_bias=5.0, fixed_positions=[0, 4], gumbel_logit_init=True)
        opt.run()
        from proto_language.language.core.sequence import PROTEIN_AMINO_ACIDS

        aa_idx = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
        result = seg.result_sequences[0].logits
        assert result is not None
        assert result[0, aa_idx["E"]] == pytest.approx(5.0)
        assert result[4, aa_idx["V"]] == pytest.approx(5.0)
        assert result[1, aa_idx["V"]] != pytest.approx(5.0)


class TestValidation:
    def test_no_gradient_constraints(self) -> None:
        seg = Segment(sequence="AA", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], function=_scorer, function_config=_Cfg())
        with pytest.raises(ValueError, match="gradient-capable"):
            GradientOptimizer(
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )

    def test_wrong_generator(self) -> None:
        seg = Segment(sequence="AA", sequence_type="protein")
        gen = RandomProteinGenerator(RandomProteinGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="m")
        with pytest.raises(ValueError, match="PositionWeightGenerator"):
            GradientOptimizer(
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )

    def test_nan_gradient_names_offending_constraint(self) -> None:
        """Non-finite gradients must raise with the constraint name so flaky backwards are easy to attribute."""

        def nan_bwd(inputs: tuple, *, config: BaseModel, **kwargs: object) -> GradientResult:
            return GradientResult(gradient=(np.full_like(inputs[0].logits, np.nan),), loss=0.0, metrics={})

        opt = _make_optimizer(Segment(sequence="AA", sequence_type="protein"), nan_bwd, label="naughty", num_steps=1)
        with pytest.raises(ValueError, match="naughty"):
            opt.run()


class TestHistory:
    def test_snapshot_count(self) -> None:
        opt, _ = _make(num_steps=10, tracking_interval=5)
        opt.run()
        assert len(opt.history) == 3  # step 0, 5, 10

    def test_snapshots_reflect_current_logits(self) -> None:
        """History entries must follow the trajectory, not report the initial state."""
        opt, _ = _make(num_steps=30, lr=1.0, tracking_interval=5, seed=1, initial_logit_bias=5.0)
        opt.run()

        def _seq(idx: int) -> str:
            return str(opt.history[idx]["results"][0]["constructs"][0]["segments"][0]["sequence"])

        assert _seq(0) == "EVQLV"  # initial snapshot reflects the bias-5 init
        assert _seq(-1).count("A") > _seq(0).count("A")  # gradient pushes toward alanine


class TestOptimizerChoice:
    def test_adam_converges_and_differs_from_sgd(self) -> None:
        """beta1/beta2>0 runs the Adam path and diverges from SGD on the same seed."""
        opt_adam, seg_adam = _make(num_steps=15, lr=0.05, beta1=0.9, beta2=0.999, seed=7)
        opt_sgd, seg_sgd = _make(num_steps=15, lr=0.05, beta1=0.0, beta2=0.0, seed=7)
        opt_adam.run()
        opt_sgd.run()
        adam, sgd = seg_adam.result_sequences[0].logits, seg_sgd.result_sequences[0].logits
        assert adam is not None and sgd is not None
        assert adam[:, 0].mean() > adam[:, 1:].mean()  # still converges toward alanine
        assert not np.allclose(adam, sgd)


class TestMergers:
    def test_pcgrad_differs_from_weighted_sum(self) -> None:
        """PCGrad conflict projection produces different logits than naive weighted sum."""
        results = {}
        for merger in ("pcgrad", "weighted_sum"):
            seg = Segment(sequence="GGGGG", sequence_type="protein")
            gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
            gen.assign(seg)
            con_a = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="A")
            con_c = Constraint(inputs=[seg], backward=_backward_toward_C, backward_config=_Cfg(), label="C")
            opt = GradientOptimizer(
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con_a, con_c],
                config=GradientOptimizerConfig(
                    num_results=1,
                    num_steps=10,
                    lr=0.1,
                    beta1=0.0,
                    beta2=0.0,
                    merger=merger,
                    normalize_gradients=False,
                    seed=42,
                ),
            )
            opt.run()
            results[merger] = seg.result_sequences[0].logits
        assert not np.allclose(results["pcgrad"], results["weighted_sum"])

    def test_constraint_weight_dominance(self) -> None:
        """High-weight constraint dominates in logit space."""
        seg = Segment(sequence="GGGGG", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con_a = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="A", weight=10.0)
        con_c = Constraint(inputs=[seg], backward=_backward_toward_C, backward_config=_Cfg(), label="C", weight=0.01)
        opt = GradientOptimizer(
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[con_a, con_c],
            config=GradientOptimizerConfig(
                num_results=1, num_steps=20, lr=0.1, beta1=0.0, beta2=0.0, merger="weighted_sum", seed=42
            ),
        )
        opt.run()
        logits = seg.result_sequences[0].logits
        assert logits is not None
        assert logits[:, 0].mean() > logits[:, 2].mean()  # A dominates C


class TestWeightSchedules:
    def test_schedule_overrides_static_weight(self) -> None:
        # Static w=99 * unit grad * 10 steps would push logits to ~-990; schedule avg ~0.1 -> ~-1.
        seg = Segment(sequence="AA", sequence_type="protein")
        opt = _make_optimizer(
            seg,
            _unit_grad_bwd,
            label="ablang",
            weight=99.0,
            num_steps=10,
            lr=1.0,
            beta1=0.0,
            beta2=0.0,
            normalize_gradients=False,
            constraint_weight_schedules=[
                ConstraintWeightSchedule(constraint_label="ablang", start_weight=0.0, end_weight=0.2)
            ],
        )
        opt.run()
        logits = seg.result_sequences[0].logits
        assert logits is not None
        assert logits.min() > -50.0

    def test_missing_label_warns_but_runs(self, caplog: pytest.LogCaptureFixture) -> None:
        seg = Segment(sequence="AA", sequence_type="protein")
        with caplog.at_level("WARNING"):
            opt = _make_optimizer(
                seg,
                _backward,
                label="actual",
                num_steps=1,
                constraint_weight_schedules=[
                    ConstraintWeightSchedule(constraint_label="missing", start_weight=0.0, end_weight=1.0)
                ],
            )
        assert any("missing" in r.message for r in caplog.records)
        opt.run()


class TestGumbelLogitInit:
    def test_gumbel_adds_noise(self) -> None:
        opt_d, seg_d = _make(num_steps=1, seed=42, initial_logit_bias=5.0, gumbel_logit_init=False)
        opt_g, seg_g = _make(num_steps=1, seed=42, initial_logit_bias=5.0, gumbel_logit_init=True)
        opt_d.run()
        opt_g.run()
        assert not np.allclose(seg_d.result_sequences[0].logits, seg_g.result_sequences[0].logits)

    def test_gumbel_reproducible_with_seed(self) -> None:
        opt_a, seg_a = _make(num_steps=5, seed=123, gumbel_logit_init=True)
        opt_b, seg_b = _make(num_steps=5, seed=123, gumbel_logit_init=True)
        opt_a.run()
        opt_b.run()
        assert np.allclose(seg_a.result_sequences[0].logits, seg_b.result_sequences[0].logits)


class TestMultiStage:
    """End-to-end multi-stage Program pipelines."""

    @staticmethod
    def _gradient_stage(seg: Segment, construct: Construct, label: str, **kw: object) -> GradientOptimizer:
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label=label)
        defaults: dict[str, object] = {"num_results": 1, "num_steps": 5, "lr": 0.1, "beta1": 0.0, "beta2": 0.0}
        defaults.update(kw)
        return GradientOptimizer(
            constructs=[construct], generators=[gen], constraints=[con], config=GradientOptimizerConfig(**defaults)
        )

    def test_logit_handoff_across_stages(self) -> None:
        """Stage 2 reads logits produced by stage 1, not re-initialized zeros."""
        seg = Segment(sequence="EVQLV", sequence_type="protein")
        construct = Construct([seg])
        opt1 = self._gradient_stage(seg, construct, "s1", soft_start=0.0, soft_end=1.0)
        opt2 = self._gradient_stage(seg, construct, "s2", temperature_start=1.0, temperature_end=0.1, schedule="linear")

        program = Program(optimizers=[opt1, opt2], num_results=1)
        program.run_stage(0)
        stage1 = seg.result_sequences[0].logits
        assert stage1 is not None and stage1[0, 0] > stage1[0, 5]  # optimized toward A

        program.run_stage(1)
        stage2 = seg.result_sequences[0].logits
        assert stage2 is not None and stage2[0, 0] > stage1[0, 0]  # further optimized

    def test_gradient_then_mcmc(self) -> None:
        """GradientOptimizer → MCMCOptimizer pipeline completes with valid scores."""
        seg = Segment(sequence="EVQLV", sequence_type="protein")
        construct = Construct([seg])
        opt1 = self._gradient_stage(seg, construct, "g", seed=42)

        gen2 = RandomProteinGenerator(RandomProteinGeneratorConfig())
        gen2.assign(seg)
        con2 = Constraint(inputs=[seg], function=_scorer, function_config=_Cfg(), label="m")
        opt2 = MCMCOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[con2],
            config=MCMCOptimizerConfig(num_results=1, num_steps=5),
        )

        Program(optimizers=[opt1, opt2], num_results=1).run()
        assert opt2.energy_scores[0] < float("inf")

    def test_germinal_presets_plug_into_program(self) -> None:
        """CPU proof that both Germinal presets chain into a Program and logits flow across stages."""
        seg = Segment(sequence="EVQLVESGGG", sequence_type="protein")
        construct = Construct([seg])

        def stage(cfg: GradientOptimizerConfig, weight: float = 1.0) -> GradientOptimizer:
            gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
            gen.assign(seg)
            con = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="ablang", weight=weight)
            return GradientOptimizer(constructs=[construct], generators=[gen], constraints=[con], config=cfg)

        cfg_logit = GradientOptimizerConfig.germinal_logit_preset()
        cfg_logit.num_steps = 3
        cfg_soft = GradientOptimizerConfig.germinal_softmax_preset()
        cfg_soft.num_steps = 3

        Program(optimizers=[stage(cfg_logit), stage(cfg_soft, weight=0.4)], num_results=1).run()
        assert seg.result_sequences[0].logits is not None
        assert seg.result_sequences[0].logits.shape == (10, 20)


class TestExport:
    def test_to_dataframe_and_fasta(self) -> None:
        opt, _ = _make()
        opt.run()
        df = opt.to_dataframe(table="sequences")
        assert len(df) > 0
        fasta = opt.to_fasta()
        assert fasta.startswith(">")


def _ablang_constraint(seg: Segment, label: str = "ablang") -> Constraint:
    from proto_language.language.constraint.differentiable import ablang_vhh_gradient_backward
    from proto_language.language.constraint.differentiable.ablang_naturalness_gradient_constraint import (
        AbLangGradientConstraintConfig,
    )

    return Constraint(
        inputs=[seg],
        backward=ablang_vhh_gradient_backward,
        backward_config=AbLangGradientConstraintConfig(),
        label=label,
    )


def _af2_constraint(binder: Segment, target: Segment, label: str = "af2") -> Constraint:
    from proto_language.language.constraint.differentiable.af2_binder_gradient_constraint import (
        AF2BinderGradientConfig,
        af2_binder_backward,
    )

    return Constraint(
        inputs=[binder, target],
        backward=af2_binder_backward,
        backward_config=AF2BinderGradientConfig(
            target_chain="A", binder_chain="B", num_recycles=1, loss_weights={"plddt": 1.0}
        ),
        label=label,
    )


def _target_with_pdl1() -> Segment:
    from proto_tools.entities.structures import Structure

    from tests.helpers.mock_structure import PDL1_PDB

    seg = Segment(sequence="A" * 10, sequence_type="protein", label="target")
    structure = Structure(structure=PDL1_PDB.read_text(), structure_format="pdb")
    seg.result_sequences[0].structure = structure
    seg.proposal_sequences[0].structure = structure
    return seg


@pytest.mark.uses_gpu
@pytest.mark.slow
class TestGradientOptimizerGPU:
    """GPU integration tests with real differentiable constraints."""

    def test_ablang_gradient_descent(self) -> None:
        """AbLang VHH gradient reduces naturalness loss over 10 steps."""
        seg = Segment(sequence="EVQLVESGGGLVQPGGSLRL", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        opt = GradientOptimizer(
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[_ablang_constraint(seg)],
            config=GradientOptimizerConfig(
                num_results=1, num_steps=10, lr=0.1, beta1=0.0, beta2=0.0, tracking_interval=5
            ),
        )
        opt.run()

        assert opt.history[-1]["results"][0]["energy_score"] < opt.history[1]["results"][0]["energy_score"]
        assert seg.result_sequences[0].logits is not None

    def test_af2_binder_gradient_descent(self) -> None:
        """AF2 binder gradient produces finite logit updates over 3 steps against a target."""
        binder = Segment(length=10, sequence_type="protein", label="binder")
        target = _target_with_pdl1()
        construct = Construct([binder, target])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(binder)

        opt = GradientOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[_af2_constraint(binder, target)],
            config=GradientOptimizerConfig(num_results=1, num_steps=3, lr=0.1, beta1=0.0, beta2=0.0, seed=7),
        )
        opt.run()
        logits = binder.result_sequences[0].logits
        assert logits is not None and logits.shape == (10, 20)
        assert np.isfinite(logits).all()

    def test_af2_plus_ablang_pcgrad(self) -> None:
        """AF2 + AbLang merged via PCGrad — both real gradients, finite logits, both losses recorded."""
        binder = Segment(sequence="EVQLVESGGG", sequence_type="protein", label="binder")
        target = _target_with_pdl1()
        construct = Construct([binder, target])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(binder)

        opt = GradientOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[_af2_constraint(binder, target), _ablang_constraint(binder)],
            config=GradientOptimizerConfig(
                num_results=1,
                num_steps=2,
                lr=0.1,
                beta1=0.0,
                beta2=0.0,
                merger="pcgrad",
                norm_alignment="match_first",
                normalize_mode="sqrt_length",
                seed=7,
            ),
        )
        opt.run()
        logits = binder.result_sequences[0].logits
        assert logits is not None and np.isfinite(logits).all()
        # Energy is the sum of AF2 + AbLang losses; both must be finite for it to be finite.
        assert np.isfinite(opt.energy_scores[0])

    def test_two_stage_germinal_preset(self) -> None:
        """Full Germinal preset chain (logit → softmax) with real AbLang gradients."""
        seg = Segment(sequence="EVQLVESGGGLVQPGGSLRL", sequence_type="protein")
        construct = Construct([seg])

        def stage(cfg: GradientOptimizerConfig, label: str) -> GradientOptimizer:
            gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
            gen.assign(seg)
            return GradientOptimizer(
                constructs=[construct], generators=[gen], constraints=[_ablang_constraint(seg, label)], config=cfg
            )

        cfg1 = GradientOptimizerConfig.germinal_logit_preset()
        cfg1.num_steps = 10
        cfg2 = GradientOptimizerConfig.germinal_softmax_preset()
        cfg2.num_steps = 5

        Program(optimizers=[stage(cfg1, "ablang"), stage(cfg2, "ablang_s2")], num_results=1).run()
        result = seg.result_sequences[0]
        assert result.logits is not None and result.sequence != "EVQLVESGGGLVQPGGSLRL"
