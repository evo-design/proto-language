"""Tests for the LedidiOptimizer."""

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from proto_language import GradientConstraintOutput
from proto_language.core import Constraint, Construct, Program, Segment, Sequence
from proto_language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.optimizer import LedidiOptimizer, LedidiOptimizerConfig, OptimizerRegistry
from proto_language.optimizer.ledidi_optimizer import _edit_gradient, _softmax

TEMPLATE = "ACGTACGTACGTACGTACGT"
G = 2  # column of G in the ACGT vocabulary
# Per-position value of a G, mean 1, so one edit gains between 0.1 / L and 1.9 / L.
POSITION_WEIGHTS = np.linspace(0.1, 1.9, len(TEMPLATE))


class _Cfg(BaseModel):
    """Empty config for mocks."""


def _relaxed(logits: np.ndarray, temperature: float, soft: float, hard: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The relaxation proto-tools gradient tools apply: forward value, softmax probabilities, d(value)/d(probs)."""
    probs = _softmax(logits / temperature)
    hard_probs = np.eye(logits.shape[-1])[probs.argmax(axis=-1)]
    relaxed = (1.0 - soft) * hard_probs + soft * probs
    scale = soft
    if hard > 0.0:
        relaxed = hard * hard_probs + (1.0 - hard) * relaxed
        scale = hard + (1.0 - hard) * soft
    return relaxed, probs, scale


def _g_fraction_backward(
    input_sequences: list[tuple], *, config: BaseModel, temperature: float = 1.0, soft: float = 1.0, hard: float = 0.0
) -> list[GradientConstraintOutput]:
    """Oracle loss = 1 - position-weighted fraction of G, differentiated through the standard relaxation."""
    results = []
    for (seq,) in input_sequences:
        relaxed, probs, scale = _relaxed(seq.logits, temperature, soft, hard)
        upstream = np.zeros_like(seq.logits)
        upstream[:, G] = -scale * POSITION_WEIGHTS / seq.logits.shape[0]
        grad = probs * (upstream - (upstream * probs).sum(-1, keepdims=True)) / temperature
        results.append(
            GradientConstraintOutput(
                gradient=(grad,), loss=1.0 - float((relaxed[:, G] * POSITION_WEIGHTS).mean()), metrics={}
            )
        )
    return results


def _constant_backward(
    input_sequences: list[tuple], *, config: BaseModel, **kwargs: object
) -> list[GradientConstraintOutput]:
    """Oracle with a constant loss and no gradient."""
    return [
        GradientConstraintOutput(gradient=(np.zeros_like(seq.logits),), loss=0.5, metrics={})
        for (seq,) in input_sequences
    ]


def _make(
    backward=_g_fraction_backward, template: str = TEMPLATE, threshold: float | None = None, **cfg: object
) -> tuple[LedidiOptimizer, Segment]:
    seg = Segment(sequence=template, sequence_type="dna", label="design")
    gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    gen.assign(seg)
    con = Constraint(inputs=[seg], backward=backward, backward_config=_Cfg(), label="oracle", threshold=threshold)
    defaults: dict[str, object] = {"batch_size": 4, "max_iter": 30, "seed": 0}
    defaults.update(cfg)
    config = LedidiOptimizerConfig(**defaults)  # type: ignore[arg-type]
    opt = LedidiOptimizer(
        target_segment=seg, constructs=[Construct([seg])], generators=[gen], constraints=[con], config=config
    )
    return opt, seg


def _results(seg: Segment) -> list[str]:
    return [seq.sequence for seq in seg.result_sequences]


# --- configuration ---------------------------------------------------------------------------


def test_registered() -> None:
    assert OptimizerRegistry.get("ledidi").label == "Ledidi Optimizer"


def test_defaults_match_ledidi() -> None:
    cfg = LedidiOptimizerConfig()
    assert (cfg.batch_size, cfg.max_iter, cfg.early_stopping_iter) == (16, 1000, 100)
    assert (cfg.lr, cfg.tau, cfg.l, cfg.eps, cfg.weight_decay) == (1.0, 1.0, 0.1, 1e-4, 0.01)


def test_num_results_is_derived() -> None:
    assert LedidiOptimizerConfig(n_repeats=3, batch_size=5).num_results == 15
    assert LedidiOptimizerConfig(n_repeats=3, batch_size=5, num_results=15).num_results == 15
    with pytest.raises(ValidationError, match="n_repeats \\* batch_size"):
        LedidiOptimizerConfig(n_repeats=3, batch_size=5, num_results=4)


@pytest.mark.parametrize(
    "field,value", [("tau", 0.0), ("l", -1.0), ("batch_size", 0), ("lr", 0.0), ("eps", 0.0), ("n_repeats", 0)]
)
def test_invalid_config(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        LedidiOptimizerConfig(**{field: value})


def test_initial_weights_must_be_rectangular_and_fit() -> None:
    with pytest.raises(ValidationError):
        LedidiOptimizerConfig(initial_weights=[[0.0, 0.0], [0.0]])
    with pytest.raises(ValueError, match="initial_weights shape"):
        _make(initial_weights=[[0.0] * 4] * 3)


def test_fixed_positions_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        _make(fixed_positions=[len(TEMPLATE)])


def test_filter_constraints_rejected() -> None:
    with pytest.raises(ValueError, match="filter"):
        _make(threshold=0.5)


def test_requires_position_weight_generator() -> None:
    seg = Segment(sequence=TEMPLATE, sequence_type="dna")
    gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    gen.assign(seg)
    con = Constraint(inputs=[seg], backward=_g_fraction_backward, backward_config=_Cfg(), label="oracle")
    with pytest.raises(ValueError):
        LedidiOptimizer(
            target_segment=seg,
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[con],
            config=LedidiOptimizerConfig(),
        )


# --- the edit-penalty gradient ---------------------------------------------------------------


def test_edit_gradient_matches_finite_differences() -> None:
    """With the hard sample held fixed, the gradient is that of l * sum(sign(y_hard - X) * softmax(z / tau)) / 2."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(3, 6, 4)) * 2
    template = np.eye(4)[rng.integers(0, 4, size=(3, 6))]
    penalty, tau = 0.3, 0.7
    sign = np.sign(np.eye(4)[z.argmax(-1)] - template)

    def surrogate(x: np.ndarray) -> float:
        return float(penalty * (sign * _softmax(x / tau)).sum() / 2.0)

    numeric = np.zeros_like(z)
    h = 1e-6
    for idx in np.ndindex(z.shape):
        step = np.zeros_like(z)
        step[idx] = h
        numeric[idx] = (surrogate(z + step) - surrogate(z - step)) / (2 * h)
    np.testing.assert_allclose(_edit_gradient(z, template, penalty, tau), numeric, atol=1e-8)


def test_edit_gradient_is_zero_without_edits() -> None:
    template = np.eye(4)[[0, 1, 2, 3]]
    z = np.log(template + 1e-4)
    np.testing.assert_array_equal(_edit_gradient(z, template, 0.1, 1.0), 0.0)


# --- runs ------------------------------------------------------------------------------------


def test_result_layout_and_metadata() -> None:
    opt, seg = _make(n_repeats=2, batch_size=3, l=0.0)
    opt.run()
    assert len(seg.result_sequences) == 6
    assert len(opt.energy_scores) == 6
    for k, seq in enumerate(seg.result_sequences):
        meta = seq.metadata["ledidi"]
        assert meta["repeat"] == k // 3
        edited = [i for i, (a, b) in enumerate(zip(TEMPLATE, seq.sequence, strict=True)) if a != b]
        assert [p for p, _, _ in meta["edits"]] == edited
        assert meta["num_edits"] == len(edited)
        assert all(TEMPLATE[p] == ref and seq.sequence[p] == alt for p, ref, alt in meta["edits"])
        assert meta["total_loss"] == pytest.approx(meta["output_loss"] + meta["edit_loss"])
        assert opt.energy_scores[k] == pytest.approx(meta["total_loss"])
        is_g = np.array([c == "G" for c in seq.sequence])
        assert meta["output_loss"] == pytest.approx(1.0 - float((is_g * POSITION_WEIGHTS).mean()))
        assert seq.logits is not None and "".join("ACGT"[i] for i in seq.logits.argmax(-1)) == seq.sequence


def test_without_penalty_edits_toward_the_oracle() -> None:
    opt, seg = _make(l=0.0, max_iter=200)
    opt.run()
    template_g = TEMPLATE.count("G")
    assert all(seq.count("G") > template_g for seq in _results(seg))


def test_large_penalty_keeps_the_template() -> None:
    opt, seg = _make(l=100.0, max_iter=50)
    opt.run()
    assert _results(seg) == [TEMPLATE] * 4
    assert all(seq.metadata["ledidi"]["best_step"] == 0 for seq in seg.result_sequences)
    is_g = np.array([c == "G" for c in TEMPLATE])
    assert opt.energy_scores == pytest.approx([1.0 - float((is_g * POSITION_WEIGHTS).mean())] * 4)


def test_penalty_selects_the_valuable_edits() -> None:
    """With l between the smallest and largest per-edit gain, only the more valuable positions are edited."""
    opt, seg = _make(l=0.05, max_iter=300, n_repeats=2)
    opt.run()
    for seq in seg.result_sequences:
        positions = [p for p, _, _ in seq.metadata["ledidi"]["edits"]]
        assert 0 < len(positions) < len(TEMPLATE) - TEMPLATE.count("G")
        assert np.mean(POSITION_WEIGHTS[positions]) > 1.0


def test_fixed_positions_are_never_edited() -> None:
    fixed = [0, 1, 4, 5, 19]
    opt, seg = _make(l=0.0, max_iter=200, n_repeats=3, fixed_positions=fixed)
    opt.run()
    for seq in _results(seg):
        assert all(seq[p] == TEMPLATE[p] for p in fixed)
        assert seq != TEMPLATE


def test_seed_reproducibility() -> None:
    runs = []
    for seed in (7, 7, 8):
        opt, seg = _make(l=0.05, seed=seed, eps=0.5, max_iter=5)
        opt.run()
        runs.append((_results(seg), opt.energy_scores, [h["optimizer"]["mean_total_loss"] for h in opt.history]))
    assert runs[0] == runs[1]
    assert runs[0][2] != runs[2][2]


def test_rerun_reproduces() -> None:
    opt, seg = _make(l=0.0, seed=3)
    opt.run()
    before = _results(seg)
    opt.run()
    assert _results(seg) == before


def test_early_stopping() -> None:
    """A flat oracle never beats the unedited template, so every trajectory stops after early_stopping_iter."""
    opt, seg = _make(backward=_constant_backward, early_stopping_iter=7, max_iter=100, n_repeats=2)
    opt.run()
    assert opt.history[-1]["time_step"] == 7
    assert _results(seg) == [TEMPLATE] * 8


def test_max_iter_caps_the_run() -> None:
    opt, _ = _make(l=0.0, max_iter=5, early_stopping_iter=100)
    opt.run()
    assert opt.history[-1]["time_step"] == 5


def test_history_snapshots() -> None:
    opt, _ = _make(l=0.0, max_iter=10, early_stopping_iter=100, tracking_interval=4)
    opt.run()
    assert [h["time_step"] for h in opt.history] == [0, 4, 8, 10]
    meta = opt.history[-1]["optimizer"]
    assert meta["type"] == "ledidi"
    assert len(meta["mean_edits"]) == 1


def test_each_trajectory_edits_its_own_template() -> None:
    other = "TTTTTTTTTTAAAAAAAAAA"
    opt, seg = _make(l=0.0, n_repeats=2, batch_size=2, max_iter=50)
    seg.result_sequences = [Sequence(TEMPLATE, sequence_type="dna"), Sequence(other, sequence_type="dna")]
    opt._initialize_sequence_pools()
    opt.run()
    for k, seq in enumerate(seg.result_sequences):
        template = TEMPLATE if k < 2 else other
        assert all(template[p] == ref for p, ref, _ in seq.metadata["ledidi"]["edits"])
        assert seq.metadata["ledidi"]["num_edits"] == sum(a != b for a, b in zip(template, seq.sequence, strict=True))


def test_runs_in_a_program() -> None:
    opt, seg = _make(l=0.0, n_repeats=2, batch_size=2, max_iter=20)
    program = Program([opt], num_results=4, seed=11)
    program.run()
    assert len(seg.result_sequences) == 4
