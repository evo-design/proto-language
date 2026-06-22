"""tests/language_tests/constraint_tests/test_protein_structure/test_dbp_design_metrics_constraint.py.

Config validation plus pure scoring / threshold / H-bond-weighting tests for the
``dbp-design-metrics`` constraint. These exercise the PyRosetta-free helpers
(``_weighted_sum``, the penalty functions, ``_compute_penalties``, and
``_composite_score``) on synthetic metric dicts; the heavy PyRosetta + dbp_design
evaluation path (``_compute_dbp_metrics_for_pdb``) is never invoked here and is
covered only by an env-gated smoke test.
"""

import math

import pytest
from pydantic import ValidationError

from proto_language.constraint.protein_structure.dbp_design_metrics_constraint import (
    _BASE_SCORE_WEIGHTS,
    _BIDENTATE_SCORE_WEIGHTS,
    _DEFAULT_PREFILTER_CUT_PATH,
    _DEFAULT_PREFILTER_EQ_PATH,
    _PHOSPHATE_SCORE_WEIGHTS,
    DBPDesignMetricsConfig,
    _component_weights,
    _composite_score,
    _compute_penalties,
    _evaluate_prefilter,
    _extract_candidate_mpnn_score,
    _failed_dbp_metrics,
    _failed_penalties,
    _heavy_penalty,
    _heavy_penalty_max,
    _heavy_penalty_min,
    _load_dbp_modules,
    _penalty_max,
    _penalty_min,
    _prefilter_logp,
    _read_first_line,
    _resolve_repo_root,
    _weighted_sum,
)


def _good_metrics() -> dict[str, float]:
    """A metric dict that satisfies every default threshold (all penalties 0)."""
    return {
        "base_score": 20.0,
        "phosphate_score": 5.0,
        "bidentate_score": 2.0,
        "total_hbonds": 12.0,
        "n_backbone_phosphate_contacts": 3.0,
        "min_compactness_contacts": 5.0,
        "max_loop_length": 4.0,
        "motif_in_rec_helix": 1.0,
        "rifres_in_rec_helix": 1.0,
        "ddg": -30.0,
        "contact_molecular_surface": 400.0,
        "net_charge": 0.0,
        "net_charge_over_sasa": 0.0,
        "ddg_over_cms": -0.1,
        "shape_complementarity": 0.8,
        "packstat": 0.7,
        "buried_unsats": 0.0,
        "max_rboltz_rkqe": 0.3,
        "avg_top_two_rboltz": 0.2,
        "mpnn_score": 1.0,
    }


class TestConfigValidation:
    """Config field validation for DBPDesignMetricsConfig."""

    def test_defaults(self):
        config = DBPDesignMetricsConfig()
        # Path defaults are empty: the constraint requires an explicit dbp_design checkout.
        assert config.dbp_design_repo_path == ""
        assert config.pyrosetta_site_packages == ""
        assert config.count_hbond_script == "2b_design_mpnn/count_hbond_types.py"
        assert config.compactness_script == "2b_design_mpnn/compactness_filter.py"
        assert config.structure_tool == "alphafold3"
        assert config.min_base_score == 10.0
        assert config.max_ddg == -15.0
        assert config.ddg_weight == 2.0
        assert config.fail_hard is False
        assert config.failure_score == 1.0

    def test_overrides(self):
        config = DBPDesignMetricsConfig(
            structure_tool="alphafold3",
            min_base_score=5.0,
            max_ddg=-20.0,
            dbp_design_repo_path="/opt/custom/dbp_design",
        )
        assert config.structure_tool == "alphafold3"
        assert config.min_base_score == 5.0
        assert config.max_ddg == -20.0
        assert config.dbp_design_repo_path == "/opt/custom/dbp_design"

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            DBPDesignMetricsConfig(base_weight=-1.0)

    def test_failure_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            DBPDesignMetricsConfig(failure_score=1.5)

    def test_missing_mpnn_penalty_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            DBPDesignMetricsConfig(missing_mpnn_score_penalty=2.0)

    def test_substantial_divergence_must_be_positive(self):
        with pytest.raises(ValidationError):
            DBPDesignMetricsConfig(substantial_divergence_ratio=0.0)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            DBPDesignMetricsConfig(not_a_field=3)


class TestRegistration:
    """The constraint registers under its kebab-case key once imported."""

    def test_registered(self):
        from proto_language.constraint.constraint_registry import ConstraintRegistry

        spec = ConstraintRegistry.get("dbp-design-metrics")
        assert spec is not None
        assert spec.config_model is DBPDesignMetricsConfig
        assert spec.uses_gpu is True
        assert set(spec.supported_sequence_types) >= {"protein", "dna"}


class TestHBondWeighting:
    """Pure H-bond / base / bidentate weighting helpers."""

    def test_weighted_sum_picks_weighted_keys(self):
        metrics = {"ARG_g_hbonds": 2.0, "ASP_g_hbonds": 1.0, "irrelevant": 99.0}
        # 2*5.0 + 1*(-3.0) = 7.0
        assert _weighted_sum(metrics, _BASE_SCORE_WEIGHTS) == pytest.approx(7.0)

    def test_weighted_sum_missing_keys_default_zero(self):
        assert _weighted_sum({}, _BASE_SCORE_WEIGHTS) == 0.0

    def test_phosphate_weighting(self):
        metrics = {"GLN_phosphate_hbonds": 1.0, "ARG_phosphate_hbonds": 1.0}
        # 10.0 + 7.0
        assert _weighted_sum(metrics, _PHOSPHATE_SCORE_WEIGHTS) == pytest.approx(17.0)

    def test_bidentate_weighting(self):
        metrics = {"ARG_g_bidentates": 2.0, "ASN_a_bidentates": 1.0}
        assert _weighted_sum(metrics, _BIDENTATE_SCORE_WEIGHTS) == pytest.approx(3.0)

    def test_weight_table_values_preserved(self):
        # Spot-check that key weight values match the dbp_design source exactly.
        assert _BASE_SCORE_WEIGHTS["ASP_c_hbonds"] == 7.0
        assert _BASE_SCORE_WEIGHTS["ARG_g_hbonds"] == 5.0
        assert _PHOSPHATE_SCORE_WEIGHTS["GLN_phosphate_hbonds"] == 10.0
        assert _BIDENTATE_SCORE_WEIGHTS["GLN_a_bidentates"] == 1.0


class TestPenaltyHelpers:
    """Pure threshold-penalty helpers."""

    def test_penalty_min_satisfied(self):
        assert _penalty_min(10.0, 5.0) == 0.0

    def test_penalty_min_deficit(self):
        # threshold=10, value=5 -> (10-5)/10 = 0.5
        assert _penalty_min(5.0, 10.0) == pytest.approx(0.5)

    def test_penalty_min_clamped(self):
        assert _penalty_min(-100.0, 1.0) == 1.0

    def test_penalty_max_satisfied(self):
        assert _penalty_max(3.0, 5.0) == 0.0

    def test_penalty_max_excess(self):
        # threshold=10, value=15 -> (15-10)/10 = 0.5
        assert _penalty_max(15.0, 10.0) == pytest.approx(0.5)

    def test_penalty_denominator_floor(self):
        # threshold below 1.0 in abs => denom floored at 1.0
        assert _penalty_min(0.0, 0.5) == pytest.approx(0.5)

    def test_heavy_penalty_quadratic_and_saturation(self):
        assert _heavy_penalty(0.0, 0.5) == 0.0
        # normalized_delta == saturation_ratio -> scaled == 1 -> penalty 1
        assert _heavy_penalty(0.5, 0.5) == pytest.approx(1.0)
        # half-saturation -> 0.25
        assert _heavy_penalty(0.25, 0.5) == pytest.approx(0.25)
        # beyond saturation clamps to 1
        assert _heavy_penalty(1.0, 0.5) == 1.0

    def test_heavy_penalty_min_satisfied(self):
        assert _heavy_penalty_min(400.0, 225.0, 0.5) == 0.0

    def test_heavy_penalty_max_satisfied(self):
        assert _heavy_penalty_max(-30.0, -15.0, 0.5) == 0.0

    def test_heavy_penalty_max_violation(self):
        # ddg threshold -15, value 0 -> normalized_delta = (0 - -15)/15 = 1 -> >saturation -> 1
        assert _heavy_penalty_max(0.0, -15.0, 0.5) == 1.0

    def test_non_finite_value_is_worst_penalty(self):
        # NaN/inf must collapse to the worst penalty instead of leaking through np.clip.
        nan = float("nan")
        assert _penalty_min(nan, 5.0) == 1.0
        assert _penalty_max(nan, 5.0) == 1.0
        assert _heavy_penalty_min(nan, 225.0, 0.5) == 1.0
        assert _heavy_penalty_max(nan, -15.0, 0.5) == 1.0
        assert _penalty_min(float("inf"), 5.0) == 1.0
        assert _heavy_penalty_max(float("inf"), -15.0, 0.5) == 1.0


class TestCompositeScoring:
    """End-to-end pure scoring from synthetic metric dicts."""

    def test_perfect_metrics_score_zero(self):
        config = DBPDesignMetricsConfig()
        penalties = _compute_penalties(_good_metrics(), config)
        assert all(v == 0.0 for v in penalties.values())
        assert _composite_score(penalties, config) == 0.0

    def test_violations_increase_score(self):
        config = DBPDesignMetricsConfig()
        metrics = _good_metrics()
        metrics["base_score"] = -10.0  # well below min_base_score=10
        metrics["ddg"] = 0.0  # above max_ddg=-15
        penalties = _compute_penalties(metrics, config)
        assert penalties["base"] > 0.0
        assert penalties["ddg"] > 0.0
        score = _composite_score(penalties, config)
        assert 0.0 < score <= 1.0

    def test_motif_requirement_penalty(self):
        config = DBPDesignMetricsConfig(require_motif_in_rec_helix=True)
        metrics = _good_metrics()
        metrics["motif_in_rec_helix"] = 0.0
        penalties = _compute_penalties(metrics, config)
        assert penalties["motif"] == 1.0
        # When not required, no penalty.
        config2 = DBPDesignMetricsConfig(require_motif_in_rec_helix=False)
        assert _compute_penalties(metrics, config2)["motif"] == 0.0

    def test_rifres_requirement_penalty(self):
        config = DBPDesignMetricsConfig(require_rifres_in_rec_helix=True)
        metrics = _good_metrics()
        metrics["rifres_in_rec_helix"] = 0.0
        assert _compute_penalties(metrics, config)["rifres"] == 1.0

    def test_missing_mpnn_uses_configured_penalty(self):
        config = DBPDesignMetricsConfig(missing_mpnn_score_penalty=0.25)
        metrics = _good_metrics()
        metrics["mpnn_score"] = float("nan")
        assert _compute_penalties(metrics, config)["mpnn_score"] == pytest.approx(0.25)

    def test_net_charge_over_sasa_two_sided(self):
        config = DBPDesignMetricsConfig()
        metrics = _good_metrics()
        # Above the max bound triggers a penalty.
        metrics["net_charge_over_sasa"] = 100.0
        assert _compute_penalties(metrics, config)["net_charge_over_sasa"] > 0.0
        # Below the min bound also triggers a penalty.
        metrics["net_charge_over_sasa"] = -100.0
        assert _compute_penalties(metrics, config)["net_charge_over_sasa"] > 0.0

    def test_zero_total_weight_scores_zero(self):
        # All weights zero => composite short-circuits to 0.0.
        config = DBPDesignMetricsConfig(
            base_weight=0.0,
            phosphate_weight=0.0,
            bidentate_weight=0.0,
            backbone_weight=0.0,
            compactness_weight=0.0,
            loop_weight=0.0,
            motif_weight=0.0,
            rifres_weight=0.0,
            buried_unsat_weight=0.0,
            shape_complementarity_weight=0.0,
            packstat_weight=0.0,
            rboltz_weight=0.0,
            avg_top_two_rboltz_weight=0.0,
            ddg_weight=0.0,
            contact_molecular_surface_weight=0.0,
            net_charge_over_sasa_weight=0.0,
            ddg_over_cms_weight=0.0,
            mpnn_score_weight=0.0,
        )
        # Even with a fully-violating penalty set, total weight 0 -> score 0.
        assert _composite_score(_failed_penalties(), config) == 0.0

    def test_all_violations_clamps_to_one(self):
        config = DBPDesignMetricsConfig()
        assert _composite_score(_failed_penalties(), config) == pytest.approx(1.0)

    def test_rboltz_families_weighted_independently(self):
        # rboltz and avg_top_two_rboltz must draw from distinct weight fields.
        config = DBPDesignMetricsConfig(rboltz_weight=0.0, avg_top_two_rboltz_weight=3.0)
        weights = _component_weights(config)
        assert weights["rboltz"] == 0.0
        assert weights["avg_top_two_rboltz"] == 3.0


class TestFailureSentinels:
    """Fail-soft metric / penalty sentinels."""

    def test_failed_metrics_nan_pattern(self):
        metrics = _failed_dbp_metrics()
        assert math.isnan(metrics["base_score"])
        assert metrics["motif_in_rec_helix"] == 0.0
        assert metrics["rifres_in_rec_helix"] == 0.0

    def test_failed_penalties_all_one(self):
        penalties = _failed_penalties()
        assert all(v == 1.0 for v in penalties.values())
        # Penalty keys must align with the weight keys for composite scoring.
        assert set(penalties) == set(_component_weights(DBPDesignMetricsConfig()))


class _FakeSeq:
    """Minimal Sequence stand-in carrying only a ``_metadata`` dict."""

    def __init__(self, metadata: dict):
        self._metadata = metadata


class TestSharedSetupResolution:
    """Shared dbp_design setup must raise (not soft-fail) when unresolvable."""

    def test_empty_repo_path_raises(self):
        with pytest.raises(ValueError, match="dbp_design_repo_path"):
            _resolve_repo_root(DBPDesignMetricsConfig())

    def test_missing_repo_path_raises(self):
        config = DBPDesignMetricsConfig(dbp_design_repo_path="/nonexistent/dbp_design")
        with pytest.raises(FileNotFoundError, match="dbp_design repo not found"):
            _resolve_repo_root(config)

    def test_loader_raises_on_empty_repo_path(self):
        with pytest.raises(ValueError, match="dbp_design_repo_path"):
            _load_dbp_modules(DBPDesignMetricsConfig())


class TestMpnnScoreExtraction:
    """Candidate MPNN-score extraction keeps only proto-tools-emitted fields."""

    def test_ligandmpnn_perplexity_extracted(self):
        candidate = (_FakeSeq({"ligandmpnn_metrics": {"perplexity": 5.1}}),)
        assert _extract_candidate_mpnn_score(candidate) == pytest.approx(5.1)

    def test_overall_confidence_ignored(self):
        # overall_confidence is higher-is-better and not emitted by proto-tools; it must
        # not be picked up as a (lower-is-better) MPNN score.
        candidate = (_FakeSeq({"ligandmpnn_metrics": {"overall_confidence": 0.9}}),)
        assert _extract_candidate_mpnn_score(candidate) is None

    def test_no_metadata_returns_none(self):
        assert _extract_candidate_mpnn_score((_FakeSeq({}),)) is None


class TestPrefilter:
    """Pure ddG/CMS maximum-likelihood prefilter evaluation (PyRosetta-free)."""

    def test_default_assets_present(self):
        config = DBPDesignMetricsConfig()
        assert config.prefilter_eq_path == _DEFAULT_PREFILTER_EQ_PATH
        assert config.prefilter_cut_path == _DEFAULT_PREFILTER_CUT_PATH
        assert config.prefilter_cut is None
        # Shipped equation asset exists and parses with the truncated feature names.
        eq_text = _read_first_line(config.prefilter_eq_path)
        assert "ddg_ove" in eq_text
        assert "contact_molecular_su" in eq_text

    def test_strong_design_passes(self):
        # Very negative ddg_over_cms and high CMS sit on the passing side of both
        # calibrated sigmoids -> log_prob ~ 0 >= cut.
        config = DBPDesignMetricsConfig()
        passed, log_prob, cut = _evaluate_prefilter(config, -0.1, 400.0)
        assert passed is True
        assert log_prob >= cut
        assert log_prob == pytest.approx(0.0, abs=1e-6)

    def test_weak_design_fails(self):
        # ddg_over_cms above the sharp ddg_ove threshold (-0.07957) collapses the
        # first sigmoid -> log_prob far below the cut.
        config = DBPDesignMetricsConfig()
        passed, log_prob, cut = _evaluate_prefilter(config, -0.05, 200.0)
        assert passed is False
        assert log_prob < cut

    def test_cutoff_boundary_pass_fail(self):
        config = DBPDesignMetricsConfig()
        eq_text = _read_first_line(config.prefilter_eq_path)
        cut = float(_read_first_line(config.prefilter_cut_path).strip())
        # Just inside the passing region for ddg_ove (< -0.07957) -> passes.
        good_lp = _prefilter_logp(eq_text, -0.2, 400.0)
        assert good_lp >= cut
        # Just outside (> -0.07957) -> fails.
        bad_lp = _prefilter_logp(eq_text, 0.0, 400.0)
        assert bad_lp < cut

    def test_explicit_cut_overrides_file(self):
        # An absurdly high cut makes even a strong design fail.
        config = DBPDesignMetricsConfig(prefilter_cut=10.0)
        passed, _log_prob, cut = _evaluate_prefilter(config, -0.1, 400.0)
        assert cut == 10.0
        assert passed is False

    def test_overflow_falls_back_to_sentinel(self):
        # Degenerate eval (divide-by-zero / overflow) collapses to the -100 sentinel.
        config = DBPDesignMetricsConfig()
        eq_text = _read_first_line(config.prefilter_eq_path)
        log_prob = _prefilter_logp(eq_text, 0.0, 100.0)
        assert log_prob == -100.0

    def test_non_finite_features_fail_closed(self):
        # NaN/inf features can't be substituted into the equation; fail closed to -100.
        config = DBPDesignMetricsConfig()
        eq_text = _read_first_line(config.prefilter_eq_path)
        assert _prefilter_logp(eq_text, float("nan"), 400.0) == -100.0
        assert _prefilter_logp(eq_text, -0.1, float("inf")) == -100.0

    def test_prefilter_skippable_when_eq_none(self):
        config = DBPDesignMetricsConfig(prefilter_eq_path=None)
        assert config.prefilter_eq_path is None


@pytest.mark.uses_gpu
@pytest.mark.skip(reason="Requires provisioned PyRosetta + local dbp_design checkout.")
def test_compute_dbp_metrics_requires_pyrosetta():
    """Smoke marker for the heavy PyRosetta path; skipped without that env."""
    from proto_language.constraint.protein_structure.dbp_design_metrics_constraint import (
        _compute_dbp_metrics_for_pdb,
    )

    _compute_dbp_metrics_for_pdb("nonexistent.pdb", DBPDesignMetricsConfig())
