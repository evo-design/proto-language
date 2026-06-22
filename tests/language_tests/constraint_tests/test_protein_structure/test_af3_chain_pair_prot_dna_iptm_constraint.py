"""Tests for the AF3 chain-pair protein-DNA ipTM constraint.

Covers config validation, the pure matrix-selection / aggregation / chain-layout
helpers on synthetic ``chain_pair_iptm`` matrices, and the constraint entry point
with ``predict_structures`` monkeypatched (no GPU or real structure prediction).
"""

from unittest.mock import MagicMock, patch

import pytest

from proto_language.constraint.protein_structure.af3_chain_pair_prot_dna_iptm_constraint import (
    AF3ChainPairProtDNAIPTMConfig,
    _build_chain_layout,
    _reverse_complement,
    _select_chain_pair_iptm,
    af3_chain_pair_prot_dna_iptm_constraint,
)
from proto_language.core import Sequence
from tests.helpers.mock_structure import MockStructure

# ============================================================================
# Config validation
# ============================================================================


def test_config_defaults():
    config = AF3ChainPairProtDNAIPTMConfig()
    assert config.structure_tool == "alphafold3"  # DNA-capable default (not inherited esmfold)
    assert config.num_protein_copies == 2
    assert config.desired_iptm == 0.7
    assert config.aggregation == "max"
    assert config.include_reverse_complement is False


@pytest.mark.parametrize("aggregation", ["max", "mean"])
def test_config_accepts_valid_aggregation(aggregation):
    config = AF3ChainPairProtDNAIPTMConfig(aggregation=aggregation)
    assert config.aggregation == aggregation


def test_config_rejects_invalid_aggregation():
    with pytest.raises(ValueError):
        AF3ChainPairProtDNAIPTMConfig(aggregation="median")


def test_config_rejects_zero_protein_copies():
    with pytest.raises(ValueError):
        AF3ChainPairProtDNAIPTMConfig(num_protein_copies=0)


@pytest.mark.parametrize("desired", [0.0, 1.5])
def test_config_rejects_out_of_range_desired_iptm(desired):
    with pytest.raises(ValueError):
        AF3ChainPairProtDNAIPTMConfig(desired_iptm=desired)


# ============================================================================
# _reverse_complement
# ============================================================================


def test_reverse_complement():
    assert _reverse_complement("ACGT") == "ACGT"
    assert _reverse_complement("AAAA") == "TTTT"
    assert _reverse_complement("atcg") == "CGAT"


# ============================================================================
# _build_chain_layout
# ============================================================================


def test_build_chain_layout_duplicates_protein_to_copies():
    config = AF3ChainPairProtDNAIPTMConfig(num_protein_copies=2)
    candidate = (Sequence("MKTAYIAK", "protein"), Sequence("ACGTACGT", "dna"))
    chains, protein_indices, dna_indices = _build_chain_layout(candidate, config, 0)

    assert protein_indices == [0, 1]
    assert dna_indices == [2]
    assert len(chains) == 3
    assert chains[0]["sequence"] == chains[1]["sequence"] == "MKTAYIAK"
    assert chains[0]["entity_type"] == "protein"
    assert chains[2]["entity_type"] == "dna"


def test_build_chain_layout_adds_reverse_complement():
    config = AF3ChainPairProtDNAIPTMConfig(num_protein_copies=1, include_reverse_complement=True)
    candidate = (Sequence("MKTAYIAK", "protein"), Sequence("ACGT", "dna"))
    chains, protein_indices, dna_indices = _build_chain_layout(candidate, config, 0)

    assert protein_indices == [0]
    assert dna_indices == [1, 2]
    assert chains[2]["sequence"] == _reverse_complement("ACGT")


def test_build_chain_layout_no_reverse_complement_when_two_dna():
    config = AF3ChainPairProtDNAIPTMConfig(num_protein_copies=1, include_reverse_complement=True)
    candidate = (
        Sequence("MKTAYIAK", "protein"),
        Sequence("ACGT", "dna"),
        Sequence("TTTT", "dna"),
    )
    chains, _protein_indices, dna_indices = _build_chain_layout(candidate, config, 0)
    assert dna_indices == [1, 2]
    assert len(chains) == 3


def test_build_chain_layout_requires_protein():
    config = AF3ChainPairProtDNAIPTMConfig()
    with pytest.raises(ValueError, match="at least one protein"):
        _build_chain_layout((Sequence("ACGT", "dna"),), config, 0)


def test_build_chain_layout_requires_dna():
    config = AF3ChainPairProtDNAIPTMConfig()
    with pytest.raises(ValueError, match="at least one DNA"):
        _build_chain_layout((Sequence("MKTAYIAK", "protein"),), config, 0)


# ============================================================================
# _select_chain_pair_iptm
# ============================================================================


def _matrix():
    # 3 chains: protein 0, protein 1, dna 2.
    # prot-dna pairs: (0,2)=0.8, (1,2)=0.4; prot-prot pair: (0,1)=0.95.
    return [
        [1.0, 0.95, 0.8],
        [0.95, 1.0, 0.4],
        [0.8, 0.4, 1.0],
    ]


def test_select_max_aggregation():
    prot_dna, prot_prot = _select_chain_pair_iptm(_matrix(), [0, 1], [2], "max")
    assert prot_dna == pytest.approx(0.8)
    assert prot_prot == pytest.approx(0.95)


def test_select_mean_aggregation():
    prot_dna, prot_prot = _select_chain_pair_iptm(_matrix(), [0, 1], [2], "mean")
    assert prot_dna == pytest.approx((0.8 + 0.4) / 2)
    assert prot_prot == pytest.approx(0.95)


def test_select_single_protein_has_no_prot_prot():
    prot_dna, prot_prot = _select_chain_pair_iptm(_matrix(), [0], [2], "max")
    assert prot_dna == pytest.approx(0.8)
    assert prot_prot is None


def test_select_returns_none_when_matrix_missing():
    prot_dna, prot_prot = _select_chain_pair_iptm(None, [0, 1], [2], "max")
    assert prot_dna is None
    assert prot_prot is None


def test_select_returns_none_when_matrix_too_small():
    small = [[1.0, 0.5], [0.5, 1.0]]  # only 2 chains but layout needs 3
    prot_dna, prot_prot = _select_chain_pair_iptm(small, [0, 1], [2], "max")
    assert prot_dna is None
    assert prot_prot is None


# ============================================================================
# Constraint entry point (predict_structures monkeypatched)
# ============================================================================

_PATCH_TARGET = "proto_language.constraint.protein_structure.af3_chain_pair_prot_dna_iptm_constraint.predict_structures"

# Homodimer + 1 DNA chain: prot-dna pairs (0,2)=0.8, (1,2)=0.4; prot-prot (0,1)=0.95.
_CHAIN_PAIR_MATRIX = [
    [1.0, 0.95, 0.8],
    [0.95, 1.0, 0.4],
    [0.8, 0.4, 1.0],
]


def _mock_output(structures: list) -> MagicMock:
    output = MagicMock()
    output.structures = structures
    return output


def _candidate() -> tuple[Sequence, Sequence]:
    return (Sequence("MKTAYIAK", "protein"), Sequence("ACGTACGT", "dna"))


@pytest.mark.parametrize(
    "aggregation,expected_iptm,expected_score",
    [
        ("max", 0.8, 0.0),  # 0.8 >= desired 0.7 -> clamped best score
        ("mean", (0.8 + 0.4) / 2, (0.7 - 0.6) / 0.7),
    ],
)
def test_entry_point_selects_prot_dna_iptm(aggregation, expected_iptm, expected_score):
    """Constraint extracts protein-DNA chain-pair ipTM and scores it (lower is better)."""
    config = AF3ChainPairProtDNAIPTMConfig(aggregation=aggregation)
    structure = MockStructure(metrics={"chain_pair_iptm": _CHAIN_PAIR_MATRIX, "iptm": 0.99})

    with patch(_PATCH_TARGET) as mock_predict:
        mock_predict.return_value = _mock_output([structure])
        [result] = af3_chain_pair_prot_dna_iptm_constraint([_candidate()], config)

    assert result.metadata["prot_dna_iptm"] == pytest.approx(expected_iptm)
    assert result.metadata["prot_prot_iptm"] == pytest.approx(0.95)
    assert result.metadata["overall_iptm"] == pytest.approx(0.99)
    # Score: 0 (best) when iptm >= desired (0.7), rises toward 1 as iptm -> 0.
    assert result.score == pytest.approx(expected_score)


def test_entry_point_reads_boltz2_matrix_key():
    """Boltz-2 exposes the same matrix under ``pair_chains_iptm``."""
    config = AF3ChainPairProtDNAIPTMConfig(structure_tool="boltz2")
    structure = MockStructure(metrics={"pair_chains_iptm": _CHAIN_PAIR_MATRIX, "iptm": 0.99})

    with patch(_PATCH_TARGET) as mock_predict:
        mock_predict.return_value = _mock_output([structure])
        [result] = af3_chain_pair_prot_dna_iptm_constraint([_candidate()], config)

    assert result.metadata["prot_dna_iptm"] == pytest.approx(0.8)


def test_entry_point_scores_protein_protein_pair():
    """``pair_type='protein-protein'`` scores the homodimer interface ipTM (0.95 >= desired -> best score)."""
    config = AF3ChainPairProtDNAIPTMConfig(pair_type="protein-protein")
    structure = MockStructure(metrics={"chain_pair_iptm": _CHAIN_PAIR_MATRIX, "iptm": 0.5})

    with patch(_PATCH_TARGET) as mock_predict:
        mock_predict.return_value = _mock_output([structure])
        [result] = af3_chain_pair_prot_dna_iptm_constraint([_candidate()], config)

    assert result.metadata["prot_prot_iptm"] == pytest.approx(0.95)
    assert result.score == 0.0  # clamped: 0.95 >= desired 0.7


def test_entry_point_raises_when_matrix_absent():
    """A predictor that omits the per-chain-pair matrix is a hard error, not a fallback."""
    config = AF3ChainPairProtDNAIPTMConfig()
    structure = MockStructure(metrics={"iptm": 0.99})  # overall ipTM only, no matrix

    with patch(_PATCH_TARGET) as mock_predict:
        mock_predict.return_value = _mock_output([structure])
        with pytest.raises(RuntimeError, match="per-chain-pair ipTM matrix"):
            af3_chain_pair_prot_dna_iptm_constraint([_candidate()], config)


def test_entry_point_empty_proposals_returns_empty():
    config = AF3ChainPairProtDNAIPTMConfig()
    with patch(_PATCH_TARGET) as mock_predict:
        assert af3_chain_pair_prot_dna_iptm_constraint([], config) == []
        mock_predict.assert_not_called()
