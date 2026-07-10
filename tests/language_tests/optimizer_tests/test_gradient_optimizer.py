"""Tests for the GradientOptimizer, multi-stage pipelines, and GPU integration."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from proto_tools import Structure
from pydantic import BaseModel

from proto_language import GradientConstraintOutput
from proto_language.core import Constraint, ConstraintOutput, Construct, Program, Segment
from proto_language.core.sequence import PROTEIN_AMINO_ACIDS
from proto_language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
)
from proto_language.optimizer import (
    ConstraintWeightSchedule,
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)
from proto_language.utils.scheduling import hinge_schedule
from proto_language.utils.sequence_matrices import SequenceLogitBiasConfig


class _Cfg(BaseModel):
    """Empty config for mocks."""


def _backward(
    input_sequences: list[tuple], *, config: BaseModel, temperature: float = 1.0, **kwargs: object
) -> list[GradientConstraintOutput]:
    """Gradient pushes logits toward alanine (column 0)."""
    results: list[GradientConstraintOutput] = []
    for (seq,) in input_sequences:
        logits = seq.logits
        target = np.zeros_like(logits)
        target[:, 0] = 1.0
        grad = logits - target
        results.append(
            GradientConstraintOutput(
                gradient=(grad,), loss=float(np.mean(grad**2)), metrics={"temperature": temperature}
            )
        )
    return results


def _backward_toward_C(
    input_sequences: list[tuple], *, config: BaseModel, **kwargs: object
) -> list[GradientConstraintOutput]:
    """Gradient pushes logits toward cysteine (column 2) — conflicts with _backward."""
    results: list[GradientConstraintOutput] = []
    for (seq,) in input_sequences:
        logits = seq.logits
        target = np.zeros_like(logits)
        target[:, 2] = 1.0
        grad = logits - target
        results.append(GradientConstraintOutput(gradient=(grad,), loss=float(np.mean(grad**2)), metrics={}))
    return results


def _unit_grad_bwd(
    input_sequences: list[tuple], *, config: BaseModel, **kwargs: object
) -> list[GradientConstraintOutput]:
    """Constant unit gradient — handy for measuring update magnitudes directly."""
    return [
        GradientConstraintOutput(gradient=(np.ones_like(seq.logits),), loss=0.0, metrics={})
        for (seq,) in input_sequences
    ]


def _scorer(input_sequences: list[tuple], config: BaseModel) -> list[ConstraintOutput]:
    return [
        ConstraintOutput(score=sum(c != "A" for c in seq.sequence) / max(len(seq.sequence), 1))
        for (seq,) in input_sequences
    ]


_scorer._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]
_scorer._constraint_num_input_sequences_per_tuple = 1  # type: ignore[attr-defined]


def _zero_scorer(input_sequences: list[tuple], config: BaseModel) -> list[ConstraintOutput]:
    return [ConstraintOutput(score=0.0) for _ in input_sequences]


_zero_scorer._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]
_zero_scorer._constraint_num_input_sequences_per_tuple = 1  # type: ignore[attr-defined]


def _make(num_steps: int = 5, num_results: int = 1, seed: int = 42, **kw: object) -> tuple[GradientOptimizer, Segment]:
    """Default-shaped optimizer: ``EVQLV`` protein segment, ``_backward`` + ``_scorer``."""
    seg = Segment(sequence="EVQLV", sequence_type="protein")
    construct = Construct([seg])
    gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    gen.assign(seg)
    con = Constraint(
        inputs=[seg], backward=_backward, backward_config=_Cfg(), function=_scorer, function_config=_Cfg(), label="mock"
    )
    defaults: dict[str, object] = {"lr": 0.1}
    defaults.update(kw)
    cfg = GradientOptimizerConfig(num_results=num_results, num_steps=num_steps, seed=seed, **defaults)
    opt = GradientOptimizer(target_segment=seg, constructs=[construct], generators=[gen], constraints=[con], config=cfg)
    return opt, seg


def _make_optimizer(
    seg: Segment,
    backward: Callable[..., list[GradientConstraintOutput]],
    label: str = "t",
    weight: float = 1.0,
    **cfg: object,
) -> GradientOptimizer:
    """Minimal single-constraint optimizer on a caller-supplied segment."""
    gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    gen.assign(seg)
    con = Constraint(inputs=[seg], backward=backward, backward_config=_Cfg(), label=label, weight=weight)
    return GradientOptimizer(
        target_segment=seg,
        constructs=[Construct([seg])],
        generators=[gen],
        constraints=[con],
        config=GradientOptimizerConfig(num_results=1, **cfg),  # type: ignore[arg-type]
    )


def _af2_binder_confidence_problem() -> tuple[Segment, Segment, Construct, list[Constraint]]:
    from proto_language import (
        AlphaFold2BinderStructureConfig,
        StructureBasedConstraintConfig,
        structure_ipae_constraint,
        structure_plddt_constraint,
    )
    from tests.helpers.mock_structure import PDL1_PDB

    binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
    target = Segment(sequence="A" * 10, sequence_type="protein", label="target")
    construct = Construct([binder, target])
    config = StructureBasedConstraintConfig(
        structure_tool="alphafold2_binder",
        alphafold2_binder_config=AlphaFold2BinderStructureConfig(
            target_pdb=PDL1_PDB.read_text(),
            binder_chain="B",
            target_chains=["A"],
        ),
    )
    constraints = [
        Constraint(
            inputs=[binder, target],
            function=structure_plddt_constraint,
            function_config=config,
            label="af2_plddt",
            weight=2.0,
        ),
        Constraint(
            inputs=[binder, target],
            function=structure_ipae_constraint,
            function_config=config,
            label="af2_ipae",
            weight=0.5,
        ),
    ]
    return binder, target, construct, constraints


def _esmfold_confidence_problem() -> tuple[Segment, Construct, list[Constraint]]:
    from proto_language import StructureBasedConstraintConfig, structure_plddt_constraint, structure_ptm_constraint

    binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
    construct = Construct([binder])
    config = StructureBasedConstraintConfig(structure_tool="esmfold")
    constraints = [
        Constraint(
            inputs=[binder],
            function=structure_plddt_constraint,
            function_config=config,
            label="esmfold_plddt",
            weight=2.0,
        ),
        Constraint(
            inputs=[binder],
            function=structure_ptm_constraint,
            function_config=config,
            label="esmfold_ptm",
            weight=0.5,
        ),
    ]
    return binder, construct, constraints


def _protenix_confidence_problem() -> tuple[Segment, Segment, Construct, list[Constraint]]:
    from proto_tools import ProtenixConfig

    from proto_language import (
        StructureBasedConstraintConfig,
        structure_iptm_constraint,
        structure_pae_constraint,
        structure_plddt_constraint,
        structure_ptm_constraint,
    )

    binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
    target = Segment(sequence="A" * 10, sequence_type="protein", label="target")
    construct = Construct([binder, target])
    config = StructureBasedConstraintConfig(
        structure_tool="protenix",
        protenix_config=ProtenixConfig(seed=0, use_msa=False),
    )
    constraints = [
        Constraint(
            inputs=[binder, target],
            function=structure_plddt_constraint,
            function_config=config,
            label="protenix_plddt",
            weight=2.0,
        ),
        Constraint(
            inputs=[binder, target],
            function=structure_ptm_constraint,
            function_config=config,
            label="protenix_ptm",
            weight=0.5,
        ),
        Constraint(
            inputs=[binder, target],
            function=structure_iptm_constraint,
            function_config=config,
            label="protenix_iptm",
            weight=3.0,
        ),
        Constraint(
            inputs=[binder, target],
            function=structure_pae_constraint,
            function_config=config,
            label="protenix_pae",
            weight=4.0,
        ),
    ]
    return binder, target, construct, constraints


def _malinois_activity_problem() -> tuple[Segment, Construct, list[Constraint]]:
    from proto_language import MalinoisActivityConfig, malinois_activity_constraint

    enhancer = Segment(sequence="ACGT" * 50, sequence_type="dna", label="enhancer")
    construct = Construct([enhancer])
    constraints = [
        Constraint(
            inputs=[enhancer],
            function=malinois_activity_constraint,
            function_config=MalinoisActivityConfig(cell_type="K562", direction="max", device="cpu"),
            label="malinois_k562_max",
            weight=2.0,
        ),
        Constraint(
            inputs=[enhancer],
            function=malinois_activity_constraint,
            function_config=MalinoisActivityConfig(cell_type="HepG2", direction="min", device="cpu"),
            label="malinois_hepg2_min",
            weight=0.5,
        ),
    ]
    return enhancer, construct, constraints


class TestConfig:
    def test_germinal_logit_preset_exact_values(self) -> None:
        """Lock Germinal Stage 0 hyperparameters; a future refactor that breaks parity must fail here."""
        cfg = GradientOptimizerConfig.germinal_logit_preset()
        assert cfg.num_steps == 65 and cfg.lr == 0.1
        assert (cfg.soft_start, cfg.soft_end) == (0.0, 1.0)
        assert (cfg.temperature_start, cfg.temperature_end) == (1.0, 1.0)
        assert cfg.softmax_schedule == "constant"
        assert cfg.lr_schedule == "constant"
        assert cfg.merger == "pcgrad"
        assert cfg.norm_alignment == "match_first"
        assert cfg.normalize_mode == "sqrt_length"
        assert cfg.gumbel_logit_init is True
        assert cfg.gumbel_init_alpha == 2.0
        assert cfg.constraint_weight_schedules == [
            ConstraintWeightSchedule(constraint_label="ablang", start_weight=0.2, end_weight=0.4, schedule="hinge")
        ]

    def test_germinal_softmax_preset_exact_values(self) -> None:
        """Lock Germinal Stage 1 hyperparameters."""
        cfg = GradientOptimizerConfig.germinal_softmax_preset()
        assert cfg.num_steps == 35 and cfg.lr == 0.1
        assert (cfg.soft_start, cfg.soft_end) == (1.0, 1.0)
        assert (cfg.temperature_start, cfg.temperature_end) == (1.0, 0.01)
        assert cfg.softmax_schedule == "quadratic"
        assert cfg.lr_schedule == "quadratic"
        assert cfg.merger == "pcgrad"
        assert cfg.norm_alignment == "match_first"
        assert cfg.normalize_mode == "sqrt_length"
        assert cfg.scale_lr_by_temperature is True
        assert cfg.min_lr_scale == 0.01


class TestInitLogitsTemplateSoft:
    """Tests for the initial_logits + softmax_init_positions initialization path."""

    def test_softmax_positions_are_probabilities_and_framework_is_template(self) -> None:
        from proto_language.optimizer.gradient_optimizer import _init_logits

        base = np.eye(5, 20, dtype=np.float64)
        original = base.copy()
        result = _init_logits(5, 20, initial_logits=base, rng=np.random.default_rng(42), softmax_init_positions=[1, 3])
        np.testing.assert_array_equal(base, original)
        for pos in [1, 3]:
            assert np.all(result[pos] >= 0.0)
            np.testing.assert_allclose(result[pos].sum(), 1.0, atol=1e-7)
        assert not np.allclose(result[[1, 3]], base[[1, 3]])
        for pos in [0, 2, 4]:
            np.testing.assert_array_equal(result[pos], base[pos])

    def test_backward_compat_none(self) -> None:
        from proto_language.optimizer.gradient_optimizer import _init_logits

        rng1, rng2 = np.random.default_rng(99), np.random.default_rng(99)
        np.testing.assert_array_equal(
            _init_logits(5, 20, initial_logits=None, rng=rng1, gumbel_alpha=2.0),
            rng2.gumbel(size=(5, 20)) / 2.0,
        )

    def test_rejects_invalid_initial_logits_config(self) -> None:
        with pytest.raises(ValueError, match="softmax_init_positions requires initial_logits"):
            GradientOptimizerConfig(softmax_init_positions=[0, 1])
        with pytest.raises(ValueError, match="initial_logits must be a rectangular"):
            GradientOptimizerConfig(initial_logits=[[1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="initial_logits shape"):
            _make(initial_logits=np.eye(3, 20).tolist())
        with pytest.raises(ValueError, match=r"softmax_init_positions .* out of bounds"):
            _make(initial_logits=np.eye(5, 20).tolist(), softmax_init_positions=[-1, 5])
        with pytest.raises(ValueError, match="appear in both softmax_init_positions and fixed_positions"):
            _make(
                initial_logits=np.eye(5, 20).tolist(),
                softmax_init_positions=[1],
                fixed_positions=[1],
            )


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

    def test_save_best_returns_logits_matching_reported_energy(self) -> None:
        """save_best returns logits whose recomputed loss matches the reported energy."""
        opt, seg = _make(num_steps=8, lr=0.3, save_best=True)
        opt.run()

        returned_logits = seg.result_sequences[0].logits
        assert returned_logits is not None
        target = np.zeros_like(returned_logits)
        target[:, 0] = 1.0
        recomputed_loss = float(np.mean((returned_logits - target) ** 2))

        assert recomputed_loss == pytest.approx(opt.energy_scores[0], abs=1e-9)


class TestSchedules:
    def test_soft_and_temperature_forwarded(self) -> None:
        """Soft ramps linearly; temperature decays via schedule — both forwarded to backward."""
        received_soft: list[float] = []
        received_temp: list[float] = []

        def bwd(
            input_sequences: list[tuple],
            *,
            config: BaseModel,
            soft: float = 1.0,
            temperature: float = 1.0,
            **kwargs: object,
        ) -> list[GradientConstraintOutput]:
            received_soft.append(soft)
            received_temp.append(temperature)
            return [
                GradientConstraintOutput(gradient=(np.zeros_like(seq.logits),), loss=0.0, metrics={})
                for (seq,) in input_sequences
            ]

        seg = Segment(sequence="AA", sequence_type="protein")
        opt = _make_optimizer(
            seg,
            bwd,
            num_steps=4,
            soft_start=0.0,
            soft_end=1.0,
            temperature_start=1.0,
            temperature_end=0.1,
            softmax_schedule="linear",
        )
        opt.run()
        assert received_soft[0] == pytest.approx(0.25) and received_soft[-1] == pytest.approx(1.0)
        assert received_temp[0] > received_temp[-1]

    def test_temperature_scaling_shrinks_updates(self) -> None:
        """scale_lr_by_temperature=True must shrink the logit delta vs =False under a unit gradient.

        ``save_best=False`` keeps the final post-update logits (what this test measures).
        """

        def _abs_sum(scale: bool) -> float:
            seg = Segment(sequence="AA", sequence_type="protein")
            opt = _make_optimizer(
                seg,
                _unit_grad_bwd,
                num_steps=5,
                lr=1.0,
                temperature_end=0.01,
                softmax_schedule="quadratic",
                lr_schedule="quadratic",
                scale_lr_by_temperature=scale,
                normalize_gradients=False,
                save_best=False,
            )
            opt.run()
            logits = seg.result_sequences[0].logits
            assert logits is not None
            return float(np.abs(logits).sum())

        assert _abs_sum(scale=True) < _abs_sum(scale=False)


class TestSequenceBias:
    def test_declarative_reference_bias_flows_to_init_logits(self) -> None:
        """Declarative reference bias anchors init logits to the WT at every position."""
        aa_idx = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
        opt, seg = _make(
            num_steps=1,
            seed=42,
            lr=1e-30,
            sequence_bias=SequenceLogitBiasConfig(reference_sequence="EVQLV", reference_bias=10.0),
        )
        opt.run()
        logits = seg.result_sequences[0].logits
        assert logits is not None
        for position, aa in enumerate("EVQLV"):
            assert int(logits[position].argmax()) == aa_idx[aa]

    def test_excluded_symbols_dominates_gradient(self) -> None:
        """``excluded_symbols`` -1e6 penalty survives gradient updates toward the banned AA."""
        aa_idx = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
        # _backward pushes toward A (column 0); the declarative penalty must still win.
        opt, seg = _make(
            num_steps=10,
            seed=42,
            lr=1.0,
            sequence_bias=SequenceLogitBiasConfig(excluded_symbols=["A"]),
        )
        opt.run()
        logits = seg.result_sequences[0].logits
        assert logits is not None
        assert np.all(logits[:, aa_idx["A"]] < -1e5)
        assert "A" not in seg.result_sequences[0].sequence

    def test_segment_mismatch_raises_at_init(self) -> None:
        """Reference-sequence length mismatch surfaces eagerly in __init__, not at run()."""
        with pytest.raises(ValueError, match="reference_sequence length"):
            _make(
                num_steps=1,
                sequence_bias=SequenceLogitBiasConfig(reference_sequence="EVQLVAA", reference_bias=1.0),
            )


class TestFixedPositions:
    def test_logits_unchanged_at_fixed(self) -> None:
        opt, seg = _make(
            num_steps=20,
            lr=1.0,
            sequence_bias=SequenceLogitBiasConfig(reference_sequence="EVQLV", reference_bias=5.0),
            fixed_positions=[0, 4],
            gumbel_logit_init=True,
        )
        opt.run()

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
        with pytest.raises(ValueError, match="gradient evaluation"):
            GradientOptimizer(
                target_segment=seg,
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
        with pytest.raises(ValueError, match="not compatible with"):
            GradientOptimizer(
                target_segment=seg,
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )

    def test_nan_gradient_names_offending_constraint(self) -> None:
        """Non-finite gradients must raise with the constraint name so flaky backwards are easy to attribute."""

        def nan_bwd(
            input_sequences: list[tuple], *, config: BaseModel, **kwargs: object
        ) -> list[GradientConstraintOutput]:
            return [
                GradientConstraintOutput(gradient=(np.full_like(seq.logits, np.nan),), loss=0.0, metrics={})
                for (seq,) in input_sequences
            ]

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
        opt, _ = _make(
            num_steps=30,
            lr=1.0,
            tracking_interval=5,
            seed=1,
            sequence_bias=SequenceLogitBiasConfig(reference_sequence="EVQLV", reference_bias=5.0),
        )
        opt.run()

        def _seq(idx: int) -> str:
            return str(opt.history[idx]["results"][0]["constructs"][0]["segments"][0]["sequence"])

        assert _seq(0) == "EVQLV"  # initial snapshot reflects the bias-5 init
        assert _seq(-1).count("A") > _seq(0).count("A")  # gradient pushes toward alanine


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
                target_segment=seg,
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con_a, con_c],
                config=GradientOptimizerConfig(
                    num_results=1,
                    num_steps=10,
                    lr=0.1,
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
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[con_a, con_c],
            config=GradientOptimizerConfig(num_results=1, num_steps=20, lr=0.1, merger="weighted_sum", seed=42),
        )
        opt.run()
        logits = seg.result_sequences[0].logits
        assert logits is not None
        assert logits[:, 0].mean() > logits[:, 2].mean()  # A dominates C


class TestMLOptimizer:
    def test_adam_produces_different_logits_than_sgd(self) -> None:
        opt_sgd, seg_sgd = _make(num_steps=10, seed=42, ml_optimizer="sgd")
        opt_adam, seg_adam = _make(num_steps=10, seed=42, ml_optimizer="adam")
        opt_sgd.run()
        opt_adam.run()
        logits_sgd = seg_sgd.result_sequences[0].logits
        logits_adam = seg_adam.result_sequences[0].logits
        assert logits_sgd is not None and logits_adam is not None
        assert not np.allclose(logits_sgd, logits_adam)


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


class TestCompiledConstraints:
    def test_groups_esmfold_same_objective_key_sums_weights(self) -> None:
        """Two constraints sharing one ESMFold objective key contribute summed weights (2.0+0.5=2.5)."""
        from proto_language import StructureBasedConstraintConfig, structure_plddt_constraint
        from tests.helpers.mock_structure import PDL1_PDB

        binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
        construct = Construct([binder])
        config = StructureBasedConstraintConfig(structure_tool="esmfold")
        constraints = [
            Constraint(
                inputs=[binder],
                function=structure_plddt_constraint,
                function_config=config,
                label="plddt_a",
                weight=2.0,
            ),
            Constraint(
                inputs=[binder],
                function=structure_plddt_constraint,
                function_config=config,
                label="plddt_b",
                weight=0.5,
            ),
        ]
        output = SimpleNamespace(
            gradient=[[0.1] * 20 for _ in range(5)],
            loss=3.0,
            metrics={"avg_plddt": 0.8, "loss_plddt": 0.2},
            structure=Structure(structure=PDL1_PDB.read_text(), structure_format="pdb"),
        )

        with patch("proto_language.optimizer.constraint_compiler.esmfold_provider.run_esmfold_gradient") as mock_esm:
            mock_esm.return_value = output
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            opt = GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=constraints,
                config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1, normalize_gradients=False),
            )
            assert len(opt._gradient_providers) == 1
            opt.run()

        assert mock_esm.call_count == 1
        assert mock_esm.call_args.args[1].loss_weights == {"plddt": pytest.approx(2.5)}

    def test_groups_af2_same_objective_key_sums_weights(self) -> None:
        """Two AF2 binder constraints sharing one objective key must contribute summed weights."""
        from proto_language import (
            AlphaFold2BinderStructureConfig,
            StructureBasedConstraintConfig,
            structure_plddt_constraint,
        )
        from tests.helpers.mock_structure import PDL1_PDB

        binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
        target = Segment(sequence="A" * 10, sequence_type="protein", label="target")
        construct = Construct([binder, target])
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold2_binder",
            alphafold2_binder_config=AlphaFold2BinderStructureConfig(
                target_pdb=PDL1_PDB.read_text(), binder_chain="B", target_chains=["A"]
            ),
        )
        constraints = [
            Constraint(
                inputs=[binder, target],
                function=structure_plddt_constraint,
                function_config=config,
                label="af2_plddt_a",
                weight=2.0,
            ),
            Constraint(
                inputs=[binder, target],
                function=structure_plddt_constraint,
                function_config=config,
                label="af2_plddt_b",
                weight=0.5,
            ),
        ]
        output = SimpleNamespace(
            gradient=[[0.1] * 20 for _ in range(5)],
            loss=3.0,
            metrics={"plddt": 1.0},
            structure=Structure(structure=PDL1_PDB.read_text(), structure_format="pdb"),
        )

        with patch(
            "proto_language.optimizer.constraint_compiler.alphafold2_binder_provider.run_alphafold2_gradient"
        ) as mock_af2:
            mock_af2.return_value = output
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            opt = GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=constraints,
                config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1, normalize_gradients=False),
            )
            assert len(opt._gradient_providers) == 1
            opt.run()

        assert mock_af2.call_count == 1
        assert mock_af2.call_args[0][1].loss_weights == {"plddt": pytest.approx(2.5)}

    def test_groups_esmfold_confidence_terms_into_one_tool_call(self) -> None:
        from tests.helpers.mock_structure import PDL1_PDB

        binder, construct, constraints = _esmfold_confidence_problem()
        output = SimpleNamespace(
            gradient=[[0.1] * 20 for _ in range(5)],
            loss=3.0,
            metrics={
                "avg_plddt": 0.8,
                "ptm": 0.4,
                "avg_pae": 6.0,
                "loss_plddt": 0.2,
                "loss_ptm": 0.6,
            },
            structure=Structure(structure=PDL1_PDB.read_text(), structure_format="pdb"),
        )

        with patch("proto_language.optimizer.constraint_compiler.esmfold_provider.run_esmfold_gradient") as mock_esm:
            mock_esm.return_value = output
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            opt = GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=constraints,
                config=GradientOptimizerConfig(
                    num_results=1,
                    num_steps=1,
                    lr=0.1,
                    normalize_gradients=False,
                ),
            )
            assert len(opt._gradient_providers) == 1
            opt.run()
            metadata = binder.result_sequences[0]._constraints_metadata

        assert opt.energy_scores == [pytest.approx(3.0)]
        assert mock_esm.call_count == 1
        assert mock_esm.call_args.args[0].target_chain_indices == [0]
        assert mock_esm.call_args.args[1].loss_weights == {"plddt": 2.0, "ptm": 0.5}
        assert {"esmfold_plddt", "esmfold_ptm"}.issubset(metadata)

    def test_groups_esmfold_scoring_terms_into_one_prediction_call(self) -> None:
        from proto_language.optimizer.constraint_compiler import evaluate_scoring_constraints
        from tests.helpers.mock_structure import PDL1_PDB

        binder, _construct, constraints = _esmfold_confidence_problem()
        structure = Structure(
            structure=PDL1_PDB.read_text(),
            structure_format="pdb",
            metrics={"avg_plddt": 0.8, "ptm": 0.4, "avg_pae": 6.0},
        )

        with patch("proto_language.optimizer.constraint_compiler.esmfold_provider.predict_structures") as mock_esm:
            mock_esm.return_value = SimpleNamespace(structures=[structure])
            scores = evaluate_scoring_constraints(constraints, mask=[True])

        assert scores == [[pytest.approx(0.7)]]
        assert mock_esm.call_count == 1
        assert mock_esm.call_args.args[1] == "esmfold"
        metadata = binder.proposal_sequences[0]._constraints_metadata
        assert {"esmfold_plddt", "esmfold_ptm"}.issubset(metadata)
        assert metadata["esmfold_plddt"]["data"]["group_score"] == pytest.approx(0.7)
        assert binder.proposal_sequences[0].structure is structure

    def test_groups_protenix_scoring_terms_into_one_prediction_call(self) -> None:
        from proto_language.optimizer.constraint_compiler import evaluate_scoring_constraints
        from tests.helpers.mock_structure import PDL1_PDB

        binder, _target, _construct, constraints = _protenix_confidence_problem()
        structure = Structure(
            structure=PDL1_PDB.read_text(),
            structure_format="pdb",
            metrics={"avg_plddt": 0.8, "ptm": 0.4, "iptm": 0.7, "avg_pae": 6.35},
        )

        with patch(
            "proto_language.optimizer.constraint_compiler.protenix_provider.predict_structures"
        ) as mock_protenix:
            mock_protenix.return_value = SimpleNamespace(structures=[structure])
            scores = evaluate_scoring_constraints(constraints, mask=[True])

        expected = (2.0 * 0.2) + (0.5 * 0.6) + (3.0 * 0.3) + (4.0 * 0.2)
        assert scores == [[pytest.approx(expected)]]
        assert mock_protenix.call_count == 1
        assert mock_protenix.call_args.args[1] == "protenix"
        assert mock_protenix.call_args.args[2].seed == 0
        metadata = binder.proposal_sequences[0]._constraints_metadata
        assert {"protenix_plddt", "protenix_ptm", "protenix_iptm", "protenix_pae"}.issubset(metadata)
        assert metadata["protenix_plddt"]["data"]["avg_plddt"] == pytest.approx(0.8)
        assert metadata["protenix_plddt"]["data"]["loss_key"] == "plddt"
        assert metadata["protenix_plddt"]["data"]["output_loss"] == pytest.approx(0.2)
        assert metadata["protenix_plddt"]["data"]["group_score"] == pytest.approx(expected)
        assert metadata["protenix_pae"]["data"]["avg_pae"] == pytest.approx(6.35)
        assert binder.proposal_sequences[0].structure is structure

    @pytest.mark.parametrize(("mode", "tool_loss"), [("gradient", 1.25), ("scoring", 0.672354)])
    def test_groups_malinois_cell_type_terms_into_one_tool_call(self, mode: str, tool_loss: float) -> None:
        from proto_language.optimizer.constraint_compiler import evaluate_scoring_constraints

        enhancer, construct, constraints = _malinois_activity_problem()
        if mode == "scoring":
            from proto_tools import MalinoisScoreOutput, MalinoisScoreResult

            score_output = MalinoisScoreOutput(
                results=[
                    MalinoisScoreResult(
                        sequence=enhancer.original_sequence.sequence,
                        sequence_length=200,
                        scores={"K562": 5.0, "HepG2": 3.0},
                    )
                ],
                cell_types=["K562", "HepG2"],
                seq_length=200,
            )
            with patch(
                "proto_language.optimizer.constraint_compiler.malinois_provider.run_malinois_score",
                return_value=score_output,
            ) as mock_malinois:
                scores = evaluate_scoring_constraints(constraints, mask=[True])
            assert scores == [[pytest.approx(tool_loss, rel=1e-5)]]
            assert mock_malinois.call_count == 1
            assert mock_malinois.call_args.args[1].cell_types == ["K562", "HepG2"]
            metadata = enhancer.proposal_sequences[0]._constraints_metadata
        else:
            gradient_output = SimpleNamespace(
                gradient=[[[0.1, 0.0, 0.0, -0.1]] * 200],
                loss=tool_loss,
                metrics={
                    "losses": [tool_loss],
                    "loss_terms": [
                        [
                            {
                                "cell_type": "K562",
                                "direction": "max",
                                "weight": 2.0,
                                "raw_score": 5.0,
                                "scaled_score": 1.0,
                                "sigmoid_value": 0.7310586,
                                "score": 0.2689414,
                                "weighted_score": 0.5378828,
                                "sigmoid_center": 4.0,
                                "sigmoid_scale": 1.0,
                            },
                            {
                                "cell_type": "HepG2",
                                "direction": "min",
                                "weight": 0.5,
                                "raw_score": 3.0,
                                "scaled_score": -1.0,
                                "sigmoid_value": 0.2689414,
                                "score": 0.2689414,
                                "weighted_score": 0.1344707,
                                "sigmoid_center": 4.0,
                                "sigmoid_scale": 1.0,
                            },
                        ],
                    ],
                },
                vocab=["A", "C", "G", "T"],
            )
            with patch(
                "proto_language.optimizer.constraint_compiler.malinois_provider.run_malinois_gradient",
                return_value=gradient_output,
            ) as mock_malinois:
                generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
                generator.assign(enhancer)
                opt = GradientOptimizer(
                    target_segment=enhancer,
                    constructs=[construct],
                    generators=[generator],
                    constraints=constraints,
                    config=GradientOptimizerConfig(
                        num_results=1,
                        num_steps=1,
                        lr=0.1,
                        normalize_gradients=False,
                    ),
                )
                opt.run()
                assert opt.energy_scores == [pytest.approx(tool_loss)]
                metadata = enhancer.result_sequences[0]._constraints_metadata
            assert mock_malinois.call_count == 1
            gradient_config = mock_malinois.call_args.args[1]
            assert [(term.cell_type, term.direction, term.weight) for term in gradient_config.loss_terms] == [
                ("K562", "max", 2.0),
                ("HepG2", "min", 0.5),
            ]

        assert {"malinois_k562_max", "malinois_hepg2_min"}.issubset(metadata)
        assert metadata["malinois_k562_max"]["data"]["malinois_direction"] == "max"
        assert metadata["malinois_hepg2_min"]["data"]["malinois_direction"] == "min"

    def test_esmfold_tool_failure_surfaces_captured_error(self) -> None:
        binder, construct, constraints = _esmfold_confidence_problem()
        failed_output = SimpleNamespace(
            success=False,
            errors=["RuntimeError: esmfold failed inside the remote worker"],
            gradient=None,
        )

        with patch(
            "proto_language.optimizer.constraint_compiler.esmfold_provider.run_esmfold_gradient",
            return_value=failed_output,
        ):
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            opt = GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=constraints,
                config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1),
            )
            with pytest.raises(RuntimeError, match="ESMFold gradient failed: RuntimeError: esmfold failed"):
                opt.run()

    def test_rejects_esmfold_interface_metric_gradient(self) -> None:
        from proto_language import StructureBasedConstraintConfig, structure_iptm_constraint

        binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
        constraint = Constraint(
            inputs=[binder],
            function=structure_iptm_constraint,
            function_config=StructureBasedConstraintConfig(structure_tool="esmfold"),
            label="esmfold_iptm",
        )
        with pytest.raises(ValueError, match="supported ESMFold confidence gradients"):
            GradientOptimizer(
                target_segment=binder,
                constructs=[Construct([binder])],
                generators=[PositionWeightGenerator(PositionWeightGeneratorConfig())],
                constraints=[constraint],
                config=GradientOptimizerConfig(num_steps=1),
            )

    @pytest.mark.parametrize(("mode", "tool_loss"), [("gradient", 3.0), ("scoring", 4.0)])
    def test_groups_af2_structure_terms_into_one_tool_call(self, mode: str, tool_loss: float) -> None:
        from proto_language.optimizer.constraint_compiler import evaluate_scoring_constraints
        from proto_language.utils.alphafold2_binder import AF2_BINDER_TOOL_LOSS_ALIASES
        from tests.helpers.mock_structure import PDL1_PDB

        binder, target, construct, constraints = _af2_binder_confidence_problem()
        if mode == "scoring":
            binder.proposal_sequences = [binder.original_sequence]
            target.proposal_sequences = [target.original_sequence]
        output = SimpleNamespace(
            gradient=[[0.1] * 20 for _ in range(5)] if mode == "gradient" else None,
            loss=tool_loss,
            metrics={"plddt": 1.0, "ipae": 2.0, "iptm": 0.8},
            structure=Structure(structure=PDL1_PDB.read_text(), structure_format="pdb"),
        )

        with patch(
            "proto_language.optimizer.constraint_compiler.alphafold2_binder_provider.run_alphafold2_gradient"
        ) as mock_af2:
            mock_af2.return_value = output
            if mode == "gradient":
                generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
                generator.assign(binder)
                opt = GradientOptimizer(
                    target_segment=binder,
                    constructs=[construct],
                    generators=[generator],
                    constraints=constraints,
                    config=GradientOptimizerConfig(
                        num_results=1,
                        num_steps=1,
                        lr=0.1,
                        normalize_gradients=False,
                    ),
                )
                opt.run()
                assert opt.energy_scores == [pytest.approx(tool_loss)]
                metadata = binder.result_sequences[0]._constraints_metadata
            else:
                scores = evaluate_scoring_constraints(constraints, mask=[True])
                assert scores == [[tool_loss]]
                metadata = binder.proposal_sequences[0]._constraints_metadata

        assert mock_af2.call_count == 1
        assert mock_af2.call_args[0][1].loss_weights == {
            "plddt": 2.0,
            AF2_BINDER_TOOL_LOSS_ALIASES.get("ipae", "ipae"): 0.5,
        }
        assert "af2_plddt" in metadata and "af2_ipae" in metadata

    @pytest.mark.parametrize("mode", ["gradient", "scoring"])
    def test_groups_af2_structure_terms_with_equivalent_target_pdb_files(self, tmp_path, mode: str) -> None:
        from proto_language import StructureBasedConstraintConfig
        from proto_language.optimizer.constraint_compiler import evaluate_scoring_constraints
        from proto_language.utils.alphafold2_binder import AF2_BINDER_TOOL_LOSS_ALIASES
        from tests.helpers.mock_structure import PDL1_PDB

        binder, target, construct, original_constraints = _af2_binder_confidence_problem()
        first_path = tmp_path / "upload_a.pdb"
        second_path = tmp_path / "upload_b.pdb"
        first_path.write_text(PDL1_PDB.read_text())
        second_path.write_text(PDL1_PDB.read_text())
        constraints: list[Constraint] = []
        for original, target_pdb in zip(original_constraints, [first_path, second_path], strict=True):
            assert original.function is not None
            assert isinstance(original.function_config, StructureBasedConstraintConfig)
            config = original.function_config.model_copy(deep=True)
            config.alphafold2_binder_config.target_pdb = str(target_pdb)
            constraints.append(
                Constraint(
                    inputs=[binder, target],
                    function=original.function,
                    function_config=config,
                    label=original.label,
                    weight=original.weight,
                )
            )

        if mode == "scoring":
            binder.proposal_sequences = [binder.original_sequence]
            target.proposal_sequences = [target.original_sequence]
        output = SimpleNamespace(
            gradient=[[0.1] * 20 for _ in range(5)] if mode == "gradient" else None,
            loss=3.5,
            metrics={"plddt": 1.0, "ipae": 2.0},
            structure=Structure(structure=PDL1_PDB.read_text(), structure_format="pdb"),
        )

        with patch(
            "proto_language.optimizer.constraint_compiler.alphafold2_binder_provider.run_alphafold2_gradient",
            return_value=output,
        ) as mock_af2:
            if mode == "gradient":
                generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
                generator.assign(binder)
                opt = GradientOptimizer(
                    target_segment=binder,
                    constructs=[construct],
                    generators=[generator],
                    constraints=constraints,
                    config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1, normalize_gradients=False),
                )
                opt.run()
            else:
                scores = evaluate_scoring_constraints(constraints, mask=[True])
                assert scores == [[pytest.approx(3.5)]]

        assert mock_af2.call_count == 1
        assert mock_af2.call_args[0][0].target_pdb.source == str(first_path)
        assert mock_af2.call_args[0][1].loss_weights == {
            "plddt": 2.0,
            AF2_BINDER_TOOL_LOSS_ALIASES.get("ipae", "ipae"): 0.5,
        }

    def test_af2_group_key_separates_different_target_pdb_files(self, tmp_path) -> None:
        from proto_language import StructureBasedConstraintConfig
        from proto_language.optimizer.constraint_compiler import alphafold2_binder_provider as af2m
        from tests.helpers.mock_structure import PDL1_PDB

        _binder, _target, _construct, constraints = _af2_binder_confidence_problem()
        first_path = tmp_path / "upload_a.pdb"
        second_path = tmp_path / "upload_b.pdb"
        first_path.write_text(PDL1_PDB.read_text())
        second_path.write_text(f"{PDL1_PDB.read_text()}\nREMARK different uploaded file\n")
        group_keys = []
        for original, target_pdb in zip(constraints, [first_path, second_path], strict=True):
            assert isinstance(original.function_config, StructureBasedConstraintConfig)
            config = original.function_config.model_copy(deep=True)
            config.alphafold2_binder_config.target_pdb = str(target_pdb)
            group_keys.append(af2m.group_key(original, config))

        assert group_keys[0] != group_keys[1]

    def test_score_energy_uses_grouped_af2_scoring(self) -> None:
        from proto_language.utils.alphafold2_binder import AF2_BINDER_TOOL_LOSS_ALIASES
        from tests.helpers.mock_structure import PDL1_PDB

        binder, _target, construct, constraints = _af2_binder_confidence_problem()
        output = SimpleNamespace(
            gradient=None,
            loss=4.0,
            metrics={"plddt": 1.0, "ipae": 2.0},
            structure=Structure(structure=PDL1_PDB.read_text(), structure_format="pdb"),
        )

        with patch(
            "proto_language.optimizer.constraint_compiler.alphafold2_binder_provider.run_alphafold2_gradient",
            return_value=output,
        ) as mock_af2:
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            opt = GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=constraints,
                config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1, seed=7),
            )
            opt._prepare_run()
            opt.score_energy()

        assert opt.energy_scores == [pytest.approx(4.0)]
        assert mock_af2.call_count == 1
        assert mock_af2.call_args[0][1].loss_weights == {
            "plddt": 2.0,
            AF2_BINDER_TOOL_LOSS_ALIASES.get("ipae", "ipae"): 0.5,
        }
        assert "af2_plddt" in binder.proposal_sequences[0]._constraints_metadata
        assert "af2_ipae" in binder.proposal_sequences[0]._constraints_metadata

    def test_af2_binder_tool_failure_surfaces_captured_error(self) -> None:
        binder, _target, construct, constraints = _af2_binder_confidence_problem()
        failed_output = SimpleNamespace(
            success=False,
            errors=["RuntimeError: alphafold2 failed inside the remote worker"],
            gradient=None,
        )

        with patch(
            "proto_language.optimizer.constraint_compiler.alphafold2_binder_provider.run_alphafold2_gradient",
            return_value=failed_output,
        ):
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            opt = GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=constraints,
                config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1),
            )
            with pytest.raises(RuntimeError, match="AF2 binder gradient failed: RuntimeError: alphafold2 failed"):
                opt.run()

    def test_scoring_compiler_evaluates_non_af2_dict_config_directly(self) -> None:
        """Non-AF2 constraints with dict configs should not be parsed as structure configs."""
        from proto_language.optimizer.constraint_compiler import evaluate_scoring_constraints

        seg = Segment(sequence="AA", sequence_type="protein")
        constraint = Constraint(inputs=[seg], function=_scorer, function_config={"unused": True}, label="plain")

        scores = evaluate_scoring_constraints([constraint], mask=[True])

        assert scores == [[0.0]]

    def test_af2_config_parse_is_strict_only_when_requested(self) -> None:
        """Compiler probes can be lenient, while execution paths preserve validation errors."""
        from pydantic import ValidationError

        from proto_language import (
            AlphaFold2BinderStructureConfig,
            StructureBasedConstraintConfig,
            structure_plddt_constraint,
        )
        from proto_language.optimizer.constraint_compiler.alphafold2_binder_provider import (
            config_for_constraint,
        )

        binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
        target = Segment(sequence="A" * 10, sequence_type="protein", label="target")
        constraint = Constraint(
            inputs=[binder, target],
            function=structure_plddt_constraint,
            function_config=StructureBasedConstraintConfig(
                structure_tool="alphafold2_binder",
                alphafold2_binder_config=AlphaFold2BinderStructureConfig(target_pdb="x"),
            ),
            label="af2_plddt",
        )
        constraint._function_config = {  # type: ignore[attr-defined]
            "structure_tool": "alphafold2_binder",
            "alphafold2_binder_config": {"target_pdb": "x", "target_input_indices": "not-an-int"},
        }

        assert config_for_constraint(constraint) is None
        with pytest.raises(ValidationError, match="target_input_indices"):
            config_for_constraint(constraint, strict=True)

    def test_af2_term_score_missing_metric_warns_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        from proto_language.optimizer.constraint_compiler.alphafold2_binder_provider import _term_score

        with caplog.at_level(
            "WARNING", logger="proto_language.optimizer.constraint_compiler.alphafold2_binder_provider"
        ):
            score, fallback_used = _term_score({"iptm": 0.8}, "plddt", 4.0)

        assert score == 4.0
        assert fallback_used is True
        assert any("Using grouped loss" in record.message for record in caplog.records)

    def test_rejects_discrete_constraint_without_compiled_gradient(self) -> None:
        seg = Segment(sequence="AA", sequence_type="protein")
        con = Constraint(inputs=[seg], function=_scorer, function_config=_Cfg(), label="plain")
        with pytest.raises(ValueError, match="does not support gradient evaluation"):
            GradientOptimizer(
                target_segment=seg,
                constructs=[Construct([seg])],
                generators=[PositionWeightGenerator(PositionWeightGeneratorConfig())],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )

    def test_rejects_af2_ptm_gradient_even_though_forward_is_supported(self) -> None:
        from proto_language import (
            AlphaFold2BinderStructureConfig,
            StructureBasedConstraintConfig,
            structure_ptm_constraint,
        )
        from tests.helpers.mock_structure import PDL1_PDB

        binder = Segment(sequence="EVQLV", sequence_type="protein", label="binder")
        target = Segment(sequence="A" * 10, sequence_type="protein", label="target")
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold2_binder",
            alphafold2_binder_config=AlphaFold2BinderStructureConfig(
                target_pdb=PDL1_PDB.read_text(),
                binder_chain="B",
                target_chains=["A"],
            ),
        )
        con = Constraint(
            inputs=[binder, target],
            function=structure_ptm_constraint,
            function_config=config,
            label="af2_ptm",
        )

        with pytest.raises(ValueError, match="structure-iptm"):
            GradientOptimizer(
                target_segment=binder,
                constructs=[Construct([binder, target])],
                generators=[PositionWeightGenerator(PositionWeightGeneratorConfig())],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )


class TestHingeSchedule:
    def test_flat_then_ramp(self) -> None:
        s = hinge_schedule(0.2, 0.4)
        assert s(0, 65) == 0.2
        assert s(32, 65) == pytest.approx(0.2)
        assert s(33, 65) > 0.2
        assert s(49, 65) == pytest.approx(0.4 * 49 / 65)
        assert s(65, 65) == 0.4

    def test_start_ge_end_rejected(self) -> None:
        with pytest.raises(ValueError, match="start_weight < end_weight"):
            ConstraintWeightSchedule(constraint_label="x", start_weight=0.4, end_weight=0.2, schedule="hinge")


class TestGumbelLogitInit:
    def test_gumbel_adds_noise(self) -> None:
        bias = SequenceLogitBiasConfig(reference_sequence="EVQLV", reference_bias=5.0)
        opt_d, seg_d = _make(num_steps=1, seed=42, sequence_bias=bias, gumbel_logit_init=False)
        opt_g, seg_g = _make(num_steps=1, seed=42, sequence_bias=bias, gumbel_logit_init=True)
        opt_d.run()
        opt_g.run()
        assert not np.allclose(seg_d.result_sequences[0].logits, seg_g.result_sequences[0].logits)

    def test_gumbel_reproducible_with_seed(self) -> None:
        opt_a, seg_a = _make(num_steps=5, seed=123, gumbel_logit_init=True)
        opt_b, seg_b = _make(num_steps=5, seed=123, gumbel_logit_init=True)
        opt_a.run()
        opt_b.run()
        assert np.allclose(seg_a.result_sequences[0].logits, seg_b.result_sequences[0].logits)

    def test_program_seed_overrides_gumbel_config_seed(self) -> None:
        def run(config_seed: int, program_seed: int) -> tuple[np.ndarray, int | None]:
            opt, seg = _make(num_steps=1, seed=config_seed, gumbel_logit_init=True)
            Program(optimizers=[opt], num_results=1, seed=program_seed).run()
            assert seg.result_sequences[0].logits is not None
            return seg.result_sequences[0].logits.copy(), opt.config.seed

        logits_a, effective_a = run(config_seed=100, program_seed=42)
        logits_b, effective_b = run(config_seed=200, program_seed=42)
        logits_c, _ = run(config_seed=100, program_seed=99)

        assert effective_a == effective_b
        assert effective_a not in (100, 200)
        assert np.allclose(logits_a, logits_b)
        assert not np.allclose(logits_a, logits_c)


class TestMultiStage:
    """End-to-end multi-stage Program pipelines."""

    @staticmethod
    def _gradient_stage(seg: Segment, construct: Construct, label: str, **kw: object) -> GradientOptimizer:
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label=label)
        defaults: dict[str, object] = {"num_results": 1, "num_steps": 5, "lr": 0.1}
        defaults.update(kw)
        return GradientOptimizer(
            target_segment=seg,
            constructs=[construct],
            generators=[gen],
            constraints=[con],
            config=GradientOptimizerConfig(**defaults),
        )

    def test_logit_handoff_across_stages(self) -> None:
        """Stage 2 reads logits produced by stage 1, not re-initialized zeros."""
        seg = Segment(sequence="EVQLV", sequence_type="protein")
        construct = Construct([seg])
        opt1 = self._gradient_stage(seg, construct, "s1", soft_start=0.0, soft_end=1.0)
        opt2 = self._gradient_stage(
            seg, construct, "s2", temperature_start=1.0, temperature_end=0.1, softmax_schedule="linear"
        )

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

    def test_gradient_then_semigreedy_mcmc_preserves_logits(self) -> None:
        """Gradient logits stay available across accepted semigreedy MCMC proposal steps."""
        seg = Segment(sequence="EVQLV", sequence_type="protein")
        construct = Construct([seg])
        opt1 = self._gradient_stage(seg, construct, "g", seed=42)

        gen2 = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig())
        gen2.assign(seg)
        con2 = Constraint(inputs=[seg], function=_zero_scorer, function_config=_Cfg(), label="flat")
        opt2 = MCMCOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[con2],
            config=MCMCOptimizerConfig(num_results=1, proposals_per_result=2, num_steps=3, seed=7),
        )

        Program(optimizers=[opt1, opt2], num_results=1).run()
        assert seg.result_sequences[0].logits is not None

    def test_germinal_presets_plug_into_program(self) -> None:
        """CPU proof that both Germinal presets chain into a Program and logits flow across stages."""
        seg = Segment(sequence="EVQLVESGGG", sequence_type="protein")
        construct = Construct([seg])

        def stage(cfg: GradientOptimizerConfig, weight: float = 1.0) -> GradientOptimizer:
            gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
            gen.assign(seg)
            con = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="ablang", weight=weight)
            return GradientOptimizer(
                target_segment=seg, constructs=[construct], generators=[gen], constraints=[con], config=cfg
            )

        cfg_logit = GradientOptimizerConfig.germinal_logit_preset()
        cfg_logit.num_steps = 3
        cfg_soft = GradientOptimizerConfig.germinal_softmax_preset()
        cfg_soft.num_steps = 3

        Program(optimizers=[stage(cfg_logit), stage(cfg_soft, weight=0.4)], num_results=1).run()
        assert seg.result_sequences[0].logits is not None
        assert seg.result_sequences[0].logits.shape == (10, 20)


class TestSaveBest:
    """Tests for the save_best config option."""

    @staticmethod
    def _make_v_shaped_backward() -> Callable[..., list[GradientConstraintOutput]]:
        """Create a backward that produces V-shaped loss: 5, 3, 1, 2, 4."""
        state: dict[str, list[float]] = {"losses": [5.0, 3.0, 1.0, 2.0, 4.0], "call": [0]}

        def bwd(
            input_sequences: list[tuple],
            *,
            config: BaseModel,
            **kwargs: object,
        ) -> list[GradientConstraintOutput]:
            idx = min(state["call"][0], len(state["losses"]) - 1)
            loss = state["losses"][idx]
            state["call"][0] += 1
            results: list[GradientConstraintOutput] = []
            for (seq,) in input_sequences:
                grad = np.zeros_like(seq.logits)
                grad[:, idx % seq.logits.shape[1]] = 1.0
                results.append(GradientConstraintOutput(gradient=(grad,), loss=loss, metrics={}))
            return results

        return bwd

    def test_save_best_returns_lowest_loss(self) -> None:
        """V-shaped loss (5,3,1,2,4): save_best=True returns energy 1, False returns 4."""
        for save_best, expected in [(True, 1.0), (False, 4.0)]:
            bwd = self._make_v_shaped_backward()
            seg = Segment(sequence="AA", sequence_type="protein")
            opt = _make_optimizer(seg, bwd, num_steps=5, lr=0.1, save_best=save_best, normalize_gradients=False)
            opt.run()
            assert opt.energy_scores[0] == pytest.approx(expected), f"save_best={save_best}"


class TestExport:
    def test_to_dataframe_and_fasta(self) -> None:
        opt, _ = _make()
        opt.run()
        df = opt.to_dataframe(table="sequences")
        assert len(df) > 0
        fasta = opt.to_fasta()
        assert fasta.startswith(">")


def _ablang_constraint(seg: Segment, label: str = "ablang") -> Constraint:
    from proto_language import AbLangPerplexityConfig, ablang_perplexity_gradient_backward

    return Constraint(
        inputs=[seg],
        backward=ablang_perplexity_gradient_backward,
        backward_config=AbLangPerplexityConfig(temperature=0.6),
        label=label,
    )


def _mpnn_constraint(seg: Segment, structure_pdb: str, label: str = "mpnn") -> Constraint:
    from proto_language import ConstraintRegistry

    return ConstraintRegistry.create(
        key="mpnn-perplexity",
        segments=[seg],
        config_dict={
            "structure_input": {"structure": structure_pdb, "chains_to_redesign": ["A"]},
            "temperature": 0.7,
            "score_mode": "nll",
            "seed": 7,
            # STE off so a short uniform-init descent is observable: with STE,
            # forward = argmax(softmax(uniform)) stays poly-A across a few small
            # steps and loss is bit-identical despite valid gradients.
            "use_ste": False,
        },
        label=label,
    )


def _af2_constraint(
    binder: Segment, target: Segment, label: str = "af2", function: Callable | None = None
) -> Constraint:
    from proto_language import (
        AlphaFold2BinderStructureConfig,
        StructureBasedConstraintConfig,
        structure_plddt_constraint,
    )
    from tests.helpers.mock_structure import PDL1_PDB

    af2_config = AlphaFold2BinderStructureConfig(
        target_pdb=PDL1_PDB.read_text(),
        target_chains="A",
        binder_chain="B",
        num_recycles=1,
    )
    return Constraint(
        inputs=[binder, target],
        function=function or structure_plddt_constraint,
        function_config=StructureBasedConstraintConfig(
            structure_tool="alphafold2_binder",
            alphafold2_binder_config=af2_config,
        ),
        label=label,
    )


def _esmfold_constraint(binder: Segment, label: str = "esmfold", function: Callable | None = None) -> Constraint:
    from proto_tools import ESMFoldConfig

    from proto_language import StructureBasedConstraintConfig, structure_plddt_constraint

    return Constraint(
        inputs=[binder],
        function=function or structure_plddt_constraint,
        function_config=StructureBasedConstraintConfig(
            structure_tool="esmfold",
            esmfold_config=ESMFoldConfig(num_recycles=1),
        ),
        label=label,
    )


def _target_segment() -> Segment:
    """Target Segment for AF2 binder design — slot is pure output, no pre-population needed."""
    return Segment(sequence="A" * 10, sequence_type="protein", label="target")


@pytest.mark.uses_gpu
@pytest.mark.slow
class TestGradientOptimizerGPU:
    """GPU integration tests with real differentiable constraints."""

    def test_ablang_gradient_descent(self) -> None:
        """AbLang naturalness gradient (VHH mode) reduces loss over 10 steps."""
        seg = Segment(sequence="EVQLVESGGGLVQPGGSLRL", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        opt = GradientOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[_ablang_constraint(seg)],
            config=GradientOptimizerConfig(num_results=1, num_steps=10, lr=0.1, tracking_interval=1),
        )
        opt.run()

        # SGD at lr=0.1 can overshoot at the tail; assert any tracked step beats the first.
        energies = [h["results"][0]["energy_score"] for h in opt.history[1:]]
        assert min(energies) < energies[0]
        assert seg.result_sequences[0].logits is not None

    def test_proteinmpnn_gradient_descent(self, sample_pdb_content: str) -> None:
        """ProteinMPNN perplexity gradients drive descent on the AGSVL backbone over 5 steps."""
        seg = Segment(sequence="AGSVL", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        opt = GradientOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[_mpnn_constraint(seg, sample_pdb_content)],
            config=GradientOptimizerConfig(num_results=1, num_steps=5, lr=0.1, tracking_interval=1, seed=7),
        )
        opt.run()

        # SGD can overshoot; assert any tracked step beats the first, mirroring ablang_gradient_descent.
        energies = [h["results"][0]["energy_score"] for h in opt.history[1:]]
        assert min(energies) < energies[0]

        logits = seg.result_sequences[0].logits
        assert logits is not None and logits.shape == (5, 20)
        assert np.isfinite(logits).all()

    def test_af2_binder_gradient_descent(self) -> None:
        """AF2 binder gradient produces finite logit updates over 3 steps against a target."""
        binder = Segment(length=10, sequence_type="protein", label="binder")
        target = _target_segment()
        construct = Construct([binder, target])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(binder)

        opt = GradientOptimizer(
            target_segment=binder,
            constructs=[construct],
            generators=[gen],
            constraints=[_af2_constraint(binder, target)],
            config=GradientOptimizerConfig(num_results=1, num_steps=3, lr=0.1, seed=7),
        )
        opt.run()
        logits = binder.result_sequences[0].logits
        assert logits is not None and logits.shape == (10, 20)
        assert np.isfinite(logits).all()

    def test_af2_binder_grouped_confidence_terms_end_to_end(self) -> None:
        """Real AF2 binder run with grouped pLDDT+iPAE compiler objectives."""
        from proto_language import structure_ipae_constraint

        binder = Segment(length=10, sequence_type="protein", label="binder")
        target = _target_segment()
        construct = Construct([binder, target])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(binder)
        constraints = [
            _af2_constraint(binder, target, "af2_plddt"),
            _af2_constraint(binder, target, "af2_ipae", function=structure_ipae_constraint),
        ]

        opt = GradientOptimizer(
            target_segment=binder,
            constructs=[construct],
            generators=[gen],
            constraints=constraints,
            config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1, seed=7),
        )
        assert len(opt._gradient_providers) == 1

        opt.run()

        result = binder.result_sequences[0]
        assert result.logits is not None and np.isfinite(result.logits).all()
        assert np.isfinite(opt.energy_scores[0])
        assert {"af2_plddt", "af2_ipae"}.issubset(result._constraints_metadata)

    def test_esmfold_grouped_confidence_terms_end_to_end(self) -> None:
        """Real ESMFold run with grouped pLDDT+pTM compiler objectives."""
        from proto_language import structure_ptm_constraint

        binder = Segment(length=6, sequence_type="protein", label="binder")
        construct = Construct([binder])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(binder)
        constraints = [
            _esmfold_constraint(binder, "esmfold_plddt"),
            _esmfold_constraint(binder, "esmfold_ptm", function=structure_ptm_constraint),
        ]

        opt = GradientOptimizer(
            target_segment=binder,
            constructs=[construct],
            generators=[gen],
            constraints=constraints,
            config=GradientOptimizerConfig(num_results=1, num_steps=1, lr=0.1, seed=7),
        )
        assert len(opt._gradient_providers) == 1

        opt.run()

        result = binder.result_sequences[0]
        assert result.logits is not None and np.isfinite(result.logits).all()
        assert np.isfinite(opt.energy_scores[0])
        assert {"esmfold_plddt", "esmfold_ptm"}.issubset(result._constraints_metadata)

    def test_af2_plus_ablang_pcgrad(self) -> None:
        """AF2 + AbLang merged via PCGrad — both real gradients, finite logits, both losses recorded."""
        binder = Segment(sequence="EVQLVESGGG", sequence_type="protein", label="binder")
        target = _target_segment()
        construct = Construct([binder, target])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(binder)

        opt = GradientOptimizer(
            target_segment=binder,
            constructs=[construct],
            generators=[gen],
            constraints=[_af2_constraint(binder, target), _ablang_constraint(binder)],
            config=GradientOptimizerConfig(
                num_results=1,
                num_steps=2,
                lr=0.1,
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
                target_segment=seg,
                constructs=[construct],
                generators=[gen],
                constraints=[_ablang_constraint(seg, label)],
                config=cfg,
            )

        cfg1 = GradientOptimizerConfig.germinal_logit_preset()
        cfg1.num_steps = 10
        cfg2 = GradientOptimizerConfig.germinal_softmax_preset()
        cfg2.num_steps = 5

        Program(optimizers=[stage(cfg1, "ablang"), stage(cfg2, "ablang_s2")], num_results=1).run()
        result = seg.result_sequences[0]
        assert result.logits is not None and result.sequence != "EVQLVESGGGLVQPGGSLRL"
