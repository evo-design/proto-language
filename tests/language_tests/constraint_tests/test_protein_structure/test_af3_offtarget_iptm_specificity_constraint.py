"""Tests for the AF3 off-target ipTM specificity constraint.

Covers config validation, the pure motif-substitution / margin-scoring helpers,
and the constraint entry point with ``predict_structures`` mocked out (no GPU/AF3).
"""

from unittest.mock import MagicMock, patch

import pytest
from proto_tools import StructurePredictionOutput

from proto_language.constraint.protein_structure.af3_offtarget_iptm_specificity_constraint import (
    AF3OffTargetIPTMSpecificityConfig,
    _clean_dna,
    _margin_score,
    _replace_motif_at_indices,
    _reverse_complement,
    af3_offtarget_iptm_specificity_constraint,
)
from proto_language.core import Sequence
from tests.helpers.mock_structure import MockStructure

_PREDICT = "proto_language.constraint.protein_structure.af3_offtarget_iptm_specificity_constraint.predict_structures"


def _make_output(structures: list[MockStructure]) -> StructurePredictionOutput:
    """Mock a StructurePredictionOutput whose structures align with the fold order."""
    output = MagicMock(spec=StructurePredictionOutput)
    output.structures = structures
    return output


# ============================================================================
# Pure helpers
# ============================================================================


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ("ACGT", "ACGT"),
        ("  acgt ", "ACGT"),
        ("aAcC", "AACC"),
    ],
)
def test_clean_dna_normalizes(sequence, expected):
    assert _clean_dna(sequence) == expected


@pytest.mark.parametrize("bad", ["", "   ", "ACGU", "ACGTN", "123"])
def test_clean_dna_rejects_invalid(bad):
    with pytest.raises(ValueError):
        _clean_dna(bad)


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ("ACGT", "ACGT"),
        ("AAAA", "TTTT"),
        ("ATGC", "GCAT"),
        ("GGGGCCCC", "GGGGCCCC"),
    ],
)
def test_reverse_complement(sequence, expected):
    assert _reverse_complement(sequence) == expected


def test_replace_motif_at_indices_basic():
    # Substitute a 2-base motif at indices 1 and 2 in a 4-base scaffold.
    assert _replace_motif_at_indices("AAAA", "GC", [1, 2]) == "AGCA"


def test_replace_motif_at_indices_non_contiguous():
    # Indices need not be contiguous or sorted.
    assert _replace_motif_at_indices("AAAAAA", "GC", [0, 5]) == "GAAAAC"


def test_replace_motif_preserves_other_positions():
    scaffold = "ACGTACGT"
    out = _replace_motif_at_indices(scaffold, "TTTT", [2, 3, 4, 5])
    assert out == "ACTTTTGT"
    assert len(out) == len(scaffold)


# ============================================================================
# Margin scoring
# ============================================================================


@pytest.mark.parametrize(
    "target, best_off, margin, expected",
    [
        # Advantage exactly meets the margin -> best score 0.0.
        (0.9, 0.85, 0.05, 0.0),
        # Advantage exceeds margin -> clamped to 0.0.
        (0.9, 0.5, 0.05, 0.0),
        # No advantage -> worst score 1.0.
        (0.7, 0.7, 0.05, 1.0),
        # Off-target beats target -> clamped to 1.0.
        (0.5, 0.9, 0.05, 1.0),
        # Half the margin achieved -> 0.5.
        (0.85, 0.825, 0.05, 0.5),
    ],
)
def test_margin_score(target, best_off, margin, expected):
    assert _margin_score(target, best_off, margin) == pytest.approx(expected)


def test_margin_score_in_unit_interval():
    for target in (0.0, 0.3, 0.6, 1.0):
        for best_off in (0.0, 0.3, 0.6, 1.0):
            score = _margin_score(target, best_off, 0.1)
            assert 0.0 <= score <= 1.0


# ============================================================================
# Config validation
# ============================================================================


def _valid_config_kwargs():
    return {
        "target_dna_sequence": "ACGTACGTACGT",
        "target_motif": "GTAC",
        "off_target_motifs": ["AAAA", "TTTT"],
        "dna_indices": [2, 3, 4, 5],
        "structure_tool": "alphafold3",
    }


def test_config_valid():
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())
    assert config.target_dna_sequence == "ACGTACGTACGT"
    assert config.off_target_motifs == ["AAAA", "TTTT"]
    assert config.desired_margin == 0.05
    assert config.include_reverse_complement is True


def test_config_normalizes_lowercase():
    kwargs = _valid_config_kwargs()
    kwargs["target_motif"] = "gtac"
    kwargs["off_target_motifs"] = ["aaaa", "tttt"]
    config = AF3OffTargetIPTMSpecificityConfig(**kwargs)
    assert config.target_motif == "GTAC"
    assert config.off_target_motifs == ["AAAA", "TTTT"]


def test_config_rejects_motif_length_mismatch_with_indices():
    kwargs = _valid_config_kwargs()
    kwargs["dna_indices"] = [2, 3, 4]  # length 3 vs motif length 4
    with pytest.raises(ValueError, match="dna_indices length"):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_config_rejects_off_target_length_mismatch():
    kwargs = _valid_config_kwargs()
    kwargs["off_target_motifs"] = ["AAA"]  # length 3 vs target motif length 4
    with pytest.raises(ValueError, match="off_target_motifs must match"):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_config_rejects_out_of_bounds_indices():
    kwargs = _valid_config_kwargs()
    kwargs["target_dna_sequence"] = "ACGT"  # length 4
    kwargs["dna_indices"] = [2, 3, 4, 5]  # 4, 5 out of bounds
    with pytest.raises(ValueError, match="exceed target_dna_sequence bounds"):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_config_rejects_negative_indices():
    kwargs = _valid_config_kwargs()
    kwargs["dna_indices"] = [-1, 3, 4, 5]
    with pytest.raises(ValueError, match="non-negative"):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_config_rejects_duplicate_indices():
    kwargs = _valid_config_kwargs()
    kwargs["dna_indices"] = [2, 2, 4, 5]
    with pytest.raises(ValueError, match="unique"):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_config_rejects_empty_off_targets():
    kwargs = _valid_config_kwargs()
    kwargs["off_target_motifs"] = []
    with pytest.raises(ValueError, match="at least one motif"):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_config_rejects_invalid_dna_alphabet():
    kwargs = _valid_config_kwargs()
    kwargs["target_motif"] = "GTAX"
    with pytest.raises(ValueError, match="A, C, G, T"):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_config_rejects_non_positive_margin():
    kwargs = _valid_config_kwargs()
    kwargs["desired_margin"] = 0.0
    with pytest.raises(ValueError):
        AF3OffTargetIPTMSpecificityConfig(**kwargs)


def test_off_target_construction_matches_indices():
    """End-to-end check of the substitution used to build off-target operators."""
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())
    off = _replace_motif_at_indices(config.target_dna_sequence, config.off_target_motifs[0], config.dna_indices)
    # Positions outside dna_indices are unchanged from the scaffold.
    for i, base in enumerate(off):
        if i not in config.dna_indices:
            assert base == config.target_dna_sequence[i]
    # Positions at dna_indices carry the off-target motif.
    for idx, base in zip(config.dna_indices, config.off_target_motifs[0], strict=True):
        assert off[idx] == base


def test_config_defaults_to_dna_capable_tool():
    """Default structure_tool is a DNA-capable predictor, not the DNA-incapable esmfold."""
    config = AF3OffTargetIPTMSpecificityConfig(
        target_dna_sequence="ACGTACGTACGT",
        target_motif="GTAC",
        off_target_motifs=["AAAA", "TTTT"],
        dna_indices=[2, 3, 4, 5],
    )
    assert config.structure_tool == "alphafold3"


# ============================================================================
# Constraint entry point (predict_structures mocked)
# ============================================================================


@pytest.fixture
def protein():
    """Single protein chain candidate."""
    return Sequence("MKTAYIAKQRQISFVK", "protein")


def test_empty_input_returns_empty():
    """No candidates short-circuits to an empty result list, never calling the predictor."""
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())
    with patch(_PREDICT) as mock_predict:
        assert af3_offtarget_iptm_specificity_constraint([], config) == []
        mock_predict.assert_not_called()


def test_candidate_without_protein_raises():
    """A candidate lacking any protein chain is a config-level error and must raise."""
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())
    candidate = (Sequence("ACGTACGT", "dna"),)
    with patch(_PREDICT):
        with pytest.raises(ValueError, match="at least one protein"):
            af3_offtarget_iptm_specificity_constraint([candidate], config)


def test_specific_binder_scores_best(protein):
    """Target ipTM beating both off-targets by >= desired_margin scores ~0 (fully specific)."""
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())  # desired_margin=0.05
    # Fold order per candidate: [target, off_0, off_1].
    structures = [
        MockStructure(metrics={"iptm": 0.90}),  # target
        MockStructure(metrics={"iptm": 0.40}),  # off-target AAAA
        MockStructure(metrics={"iptm": 0.50}),  # off-target TTTT
    ]
    with patch(_PREDICT) as mock_predict:
        mock_predict.return_value = _make_output(structures)
        [result] = af3_offtarget_iptm_specificity_constraint([(protein,)], config)

    assert result.score == pytest.approx(0.0)
    assert result.metadata["target_iptm"] == pytest.approx(0.90)
    assert result.metadata["best_off_target_iptm"] == pytest.approx(0.50)


def test_non_specific_binder_scores_worst(protein):
    """An off-target matching the target (no advantage) scores ~1 (non-specific)."""
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())
    structures = [
        MockStructure(metrics={"iptm": 0.70}),  # target
        MockStructure(metrics={"iptm": 0.40}),  # off-target AAAA
        MockStructure(metrics={"iptm": 0.70}),  # off-target TTTT ties the target
    ]
    with patch(_PREDICT) as mock_predict:
        mock_predict.return_value = _make_output(structures)
        [result] = af3_offtarget_iptm_specificity_constraint([(protein,)], config)

    assert result.score == pytest.approx(1.0)
    assert result.metadata["iptm_advantage"] == pytest.approx(0.0)


def test_missing_iptm_soft_falls_and_warns(protein, caplog):
    """A missing ipTM flags metadata, logs a warning, and soft-fails to the worst score."""
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())
    structures = [
        MockStructure(metrics={}),  # target missing iptm
        MockStructure(metrics={"iptm": 0.40}),
        MockStructure(metrics={"iptm": 0.50}),
    ]
    logger_name = "proto_language.constraint.protein_structure.af3_offtarget_iptm_specificity_constraint"
    with patch(_PREDICT) as mock_predict:
        mock_predict.return_value = _make_output(structures)
        with caplog.at_level("WARNING", logger=logger_name):
            [result] = af3_offtarget_iptm_specificity_constraint([(protein,)], config)

    assert result.score == pytest.approx(1.0)
    assert result.metadata["target_iptm"] == pytest.approx(0.0)
    assert "iptm_error" in result.metadata
    assert "iptm" in caplog.text.lower()


def test_missing_off_target_iptm_is_not_rewarded(protein):
    """A missing OFF-target ipTM must not inflate specificity into a better (lower) score."""
    config = AF3OffTargetIPTMSpecificityConfig(**_valid_config_kwargs())
    # Target binds strongly; one off-target is missing ipTM. Coercing it to 0.0 would
    # deflate best_off and make the design look fully specific — guard against that.
    structures = [
        MockStructure(metrics={"iptm": 0.90}),  # target
        MockStructure(metrics={"iptm": 0.40}),  # off-target AAAA
        MockStructure(metrics={}),  # off-target TTTT missing iptm
    ]
    with patch(_PREDICT) as mock_predict:
        mock_predict.return_value = _make_output(structures)
        [result] = af3_offtarget_iptm_specificity_constraint([(protein,)], config)

    assert result.score == pytest.approx(1.0)  # soft-failed, not rewarded
    assert "iptm_error" in result.metadata
