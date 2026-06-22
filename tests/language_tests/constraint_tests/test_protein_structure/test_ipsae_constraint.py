"""Tests for the protein-DNA ipSAE interface constraint.

Covers config validation and the pure chain-layout / score-mapping helpers, the
active-tool-config PAE toggle, and the batched entry point with the structure
predictor and ipSAE tool mocked (no GPU or real prediction).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from proto_tools import BFactorType, Structure

from proto_language.constraint.protein_structure.ipsae_constraint import (
    ProteinDNAIpsaeConfig,
    _build_chain_layout,
    _ipsae_score,
    _reverse_complement,
    _tool_config_with_pae,
    protein_dna_ipsae_constraint,
)
from proto_language.core import Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY

# ============================================================================
# Config validation
# ============================================================================


def test_config_defaults():
    config = ProteinDNAIpsaeConfig()
    assert config.num_protein_copies == 2
    assert config.desired_ipsae == 0.5
    assert config.include_reverse_complement is True
    assert config.pae_cutoff == 10.0
    assert config.distance_cutoff == 10.0


def test_config_rejects_zero_protein_copies():
    with pytest.raises(ValueError):
        ProteinDNAIpsaeConfig(num_protein_copies=0)


@pytest.mark.parametrize("desired", [0.0, -0.1])
def test_config_rejects_non_positive_desired_ipsae(desired):
    with pytest.raises(ValueError):
        ProteinDNAIpsaeConfig(desired_ipsae=desired)


@pytest.mark.parametrize("field", ["pae_cutoff", "distance_cutoff"])
def test_config_rejects_non_positive_cutoffs(field):
    with pytest.raises(ValueError):
        ProteinDNAIpsaeConfig(**{field: 0.0})


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
    config = ProteinDNAIpsaeConfig(num_protein_copies=2, include_reverse_complement=False)
    candidate = (Sequence("MKTAYIAK", "protein"), Sequence("ACGTACGT", "dna"))
    chains, protein_indices, dna_indices = _build_chain_layout(candidate, config, 0)

    assert protein_indices == [0, 1]
    assert dna_indices == [2]
    assert len(chains) == 3
    assert chains[0]["sequence"] == chains[1]["sequence"] == "MKTAYIAK"
    assert chains[0]["entity_type"] == "protein"
    assert chains[2]["entity_type"] == "dna"


def test_build_chain_layout_adds_reverse_complement():
    config = ProteinDNAIpsaeConfig(num_protein_copies=1, include_reverse_complement=True)
    candidate = (Sequence("MKTAYIAK", "protein"), Sequence("ACGT", "dna"))
    chains, protein_indices, dna_indices = _build_chain_layout(candidate, config, 0)

    assert protein_indices == [0]
    assert dna_indices == [1, 2]
    assert chains[2]["sequence"] == _reverse_complement("ACGT")


def test_build_chain_layout_no_reverse_complement_when_two_dna():
    config = ProteinDNAIpsaeConfig(num_protein_copies=1, include_reverse_complement=True)
    candidate = (
        Sequence("MKTAYIAK", "protein"),
        Sequence("ACGT", "dna"),
        Sequence("TTTT", "dna"),
    )
    chains, _protein_indices, dna_indices = _build_chain_layout(candidate, config, 0)
    assert dna_indices == [1, 2]
    assert len(chains) == 3


def test_build_chain_layout_requires_protein():
    config = ProteinDNAIpsaeConfig()
    with pytest.raises(ValueError, match="at least one protein"):
        _build_chain_layout((Sequence("ACGT", "dna"),), config, 0)


def test_build_chain_layout_requires_dna():
    config = ProteinDNAIpsaeConfig()
    with pytest.raises(ValueError, match="at least one DNA"):
        _build_chain_layout((Sequence("MKTAYIAK", "protein"),), config, 0)


# ============================================================================
# _ipsae_score
# ============================================================================


def test_ipsae_score_best_when_at_or_above_desired():
    assert _ipsae_score(0.5, 0.5) == pytest.approx(MIN_ENERGY)
    assert _ipsae_score(0.8, 0.5) == pytest.approx(MIN_ENERGY)


def test_ipsae_score_worst_when_zero():
    assert _ipsae_score(0.0, 0.5) == pytest.approx(MAX_ENERGY)


def test_ipsae_score_linear_interpolation():
    # halfway to desired -> halfway cost.
    assert _ipsae_score(0.25, 0.5) == pytest.approx(0.5)


# ============================================================================
# _tool_config_with_pae
# ============================================================================


def test_tool_config_with_pae_enables_matrix_on_boltz2():
    config = ProteinDNAIpsaeConfig(structure_tool="boltz2")
    tool_config = _tool_config_with_pae(config)
    assert tool_config.include_pae_matrix is True
    # The original config field is left untouched (model_copy returns a new object).
    assert config.boltz2_config.include_pae_matrix is False


def test_tool_config_with_pae_enables_matrix_on_alphafold3():
    config = ProteinDNAIpsaeConfig(structure_tool="alphafold3")
    tool_config = _tool_config_with_pae(config)
    assert tool_config.include_pae_matrix is True


# ============================================================================
# Config tool default (must be DNA-capable; esmfold cannot fold DNA)
# ============================================================================


def test_config_default_tool_is_dna_capable():
    assert ProteinDNAIpsaeConfig().structure_tool == "alphafold3"


# ============================================================================
# protein_dna_ipsae_constraint entry point (predictor + ipSAE tool mocked)
# ============================================================================

_PREDICT = "proto_language.constraint.protein_structure.ipsae_constraint.predict_structures"
_SCORE = "proto_language.constraint.protein_structure.ipsae_constraint.run_ipsae_scoring"


def _two_chain_structure(*, with_pae: bool) -> Structure:
    """Build a CA-only protein(A)+DNA(B) complex with pLDDT B-factors and a PAE matrix."""
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C  ",
        "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00 90.00           C  ",
        "ATOM      3  P    DA B   1      10.000   0.000   0.000  1.00 90.00           P  ",
        "ATOM      4  P    DA B   2      13.800   0.000   0.000  1.00 90.00           P  ",
        "END",
    ]
    metrics = {"pae": [[0.0] * 4 for _ in range(4)]} if with_pae else {}
    return Structure(
        structure="\n".join(lines),
        structure_format="pdb",
        b_factor_type=BFactorType.PLDDT,
        metrics=metrics,
    )


def _ipsae_output(value):
    """Mimic an IPSAEScoringOutput exposing ``metrics.primary_value``."""
    return SimpleNamespace(metrics=SimpleNamespace(primary_value=value))


@pytest.mark.parametrize(
    "ipsae,expected",
    [(0.8, MIN_ENERGY), (0.25, 0.5), (0.0, MAX_ENERGY)],
)
def test_constraint_scores_ipsae_direction(ipsae, expected):
    """Higher ipSAE yields a lower (better) cost; desired_ipsae=0.5 anchors the mapping."""
    config = ProteinDNAIpsaeConfig(num_protein_copies=1, include_reverse_complement=False)
    proposals = [(Sequence("MKTAYIAK", "protein"), Sequence("ACGT", "dna"))]

    with patch(_PREDICT) as mock_predict, patch(_SCORE) as mock_score:
        mock_predict.return_value = SimpleNamespace(structures=[_two_chain_structure(with_pae=True)])
        mock_score.return_value = _ipsae_output(ipsae)
        [result] = protein_dna_ipsae_constraint(proposals, config)

    assert result.score == pytest.approx(expected)
    assert result.metadata["ipsae"] == ipsae
    assert mock_predict.call_args[0][1] == "alphafold3"


def test_constraint_pae_missing_soft_fails(caplog):
    """A predictor that omits the PAE matrix soft-fails that candidate at MAX_ENERGY."""
    config = ProteinDNAIpsaeConfig(num_protein_copies=1, include_reverse_complement=False)
    proposals = [(Sequence("MKTAYIAK", "protein"), Sequence("ACGT", "dna"))]

    with patch(_PREDICT) as mock_predict, patch(_SCORE) as mock_score:
        mock_predict.return_value = SimpleNamespace(structures=[_two_chain_structure(with_pae=False)])
        with caplog.at_level("WARNING", logger="proto_language.constraint.protein_structure.ipsae_constraint"):
            [result] = protein_dna_ipsae_constraint(proposals, config)

    assert result.score == MAX_ENERGY
    assert result.metadata["ipsae_error"] == "pae matrix unavailable"
    assert "PAE matrix unavailable" in caplog.text
    mock_score.assert_not_called()
