"""Tests for structure confidence constraints across all metrics and prediction tools."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from proto_tools import Structure, StructurePredictionOutput

from proto_language import AlphaFold2MultimerStructureConfig, structure_contact_constraint
from proto_language.constraint.protein_structure.structure_confidence_constraint import (
    PAE_MAXIMUM,
    TOOL_AVAILABLE_METRICS,
    StructureBasedConstraintConfig,
    structure_composite_constraint,
    structure_ipae_constraint,
    structure_iplddt_constraint,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.core import Sequence
from proto_language.utils.alphafold2_multimer import (
    AF2_MULTIMER_CONFIDENCE_LOSS_BY_METRIC,
    AF2_MULTIMER_PAE_MAXIMUM,
    AF2_MULTIMER_TOOL_LOSS_ALIASES,
    af2_multimer_confidence_loss_weights,
)
from tests.helpers.mock_structure import PDL1_PDB, MockStructure

# ============================================================================
# Fixtures
# ============================================================================


def make_mock_structure(**metrics) -> MockStructure:
    """Create a real Structure subclass with specified metrics."""
    return MockStructure(metrics=metrics)


def make_mock_output(structures: list) -> StructurePredictionOutput:
    """Create a mock StructurePredictionOutput."""
    output = MagicMock(spec=StructurePredictionOutput)
    output.structures = structures
    return output


@pytest.fixture
def protein_sequence():
    """Single protein sequence."""
    return Sequence("MKTAYIAKQRQISFVK", "protein")


@pytest.fixture
def protein_sequence_b():
    """Second protein sequence for heteromers."""
    return Sequence("GVQVETISPGDGRTFPK", "protein")


@pytest.fixture
def dna_sequence():
    """Single DNA sequence."""
    return Sequence("ACGTACGTACGT", "dna")


@pytest.fixture
def rna_sequence():
    """Single RNA sequence."""
    return Sequence("ACGUACGUACGU", "rna")


# ============================================================================
# Test Score Calculations
# ============================================================================


class TestScoreCalculations:
    """Test that constraint scores are calculated correctly."""

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (1.0, 0.0),  # Perfect confidence
            (0.9, 0.1),
            (0.75, 0.25),
            (0.5, 0.5),
            (0.0, 1.0),  # No confidence
        ],
    )
    def test_plddt_scoring_esmfold(self, protein_sequence, metric_value, expected_score):
        """Test that pLDDT score = 1.0 - avg_plddt."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=metric_value, ptm=0.8)])

            results = structure_plddt_constraint(proposals, config)
            assert abs(results[0].score - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (100.0, 0.0),  # Perfect confidence
            (90.0, 0.1),
            (75.0, 0.25),
            (50.0, 0.5),
            (0.0, 1.0),  # No confidence
        ],
    )
    def test_plddt_scoring_af3(self, protein_sequence, metric_value, expected_score):
        """Test that pLDDT score = 1.0 - **normalized** avg_plddt."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=metric_value, ptm=0.8)])

            results = structure_plddt_constraint(proposals, config)
            assert abs(results[0].score - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (1.0, 0.0),
            (0.85, 0.15),
            (0.5, 0.5),
            (0.0, 1.0),
        ],
    )
    def test_ptm_scoring(self, protein_sequence, metric_value, expected_score):
        """Test that pTM score = 1.0 - ptm."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.9, ptm=metric_value)])

            results = structure_ptm_constraint(proposals, config)
            assert abs(results[0].score - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (1.0, 0.0),
            (0.7, 0.3),
            (0.5, 0.5),
            (0.0, 1.0),
        ],
    )
    def test_iptm_scoring(self, protein_sequence, protein_sequence_b, metric_value, expected_score):
        """Test that ipTM score = 1.0 - iptm."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=metric_value, avg_pae=0.85)]
            )

            results = structure_iptm_constraint(proposals, config)
            assert abs(results[0].score - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (1.0, 0.0),
            (0.7, 0.3),
            (0.5, 0.5),
            (0.0, 1.0),
        ],
    )
    def test_iptm_scoring_dna(self, protein_sequence, dna_sequence, metric_value, expected_score):
        """Test that ipTM score = 1.0 - iptm."""
        proposals = [(protein_sequence, dna_sequence)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=metric_value, avg_pae=0.85)]
            )

            results = structure_iptm_constraint(proposals, config)
            assert abs(results[0].score - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (1.0, 0.0),
            (0.7, 0.3),
            (0.5, 0.5),
            (0.0, 1.0),
        ],
    )
    def test_iptm_scoring_rna(self, protein_sequence, rna_sequence, metric_value, expected_score):
        """Test that ipTM score = 1.0 - iptm."""
        proposals = [(protein_sequence, rna_sequence)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=metric_value, avg_pae=0.85)]
            )

            results = structure_iptm_constraint(proposals, config)
            assert abs(results[0].score - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (0.0, 0.0),  # Perfect (low error)
            (15.875, 0.5),
            (31.75, 1.0),  # High error
        ],
    )
    def test_pae_scoring(self, protein_sequence, metric_value, expected_score):
        """Test that pAE score = avg_pae / 31.75."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=metric_value)]
            )

            results = structure_pae_constraint(proposals, config)
            assert abs(results[0].score - expected_score) < 1e-9


# ============================================================================
# Test Tool Dispatching
# ============================================================================


class TestToolDispatching:
    """Test that constraints correctly dispatch to different tools."""

    @pytest.mark.parametrize("tool_name", ["esmfold", "alphafold3", "boltz2", "chai1"])
    def test_plddt_dispatches_to_correct_tool(self, protein_sequence, tool_name):
        """Test that pLDDT constraint dispatches to the specified tool."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool=tool_name)

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            if tool_name == "alphafold3":
                mock_predict.return_value = make_mock_output(
                    [make_mock_structure(avg_plddt=90.0, ptm=0.8, iptm=0.7, avg_pae=5.0)]
                )
            else:
                mock_predict.return_value = make_mock_output(
                    [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.0)]
                )

            structure_plddt_constraint(proposals, config)

            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0]
            assert call_args[1] == tool_name

    @pytest.mark.parametrize(
        "constraint_fn,expected_loss,expected_tool_loss,tool_metrics",
        [
            (
                structure_plddt_constraint,
                "plddt",
                "plddt",
                {"avg_plddt": 0.75, "plddt": 0.25, "iptm": 0.8},
            ),
            (structure_iplddt_constraint, "iplddt", AF2_MULTIMER_TOOL_LOSS_ALIASES["iplddt"], None),
            (structure_ipae_constraint, "ipae", AF2_MULTIMER_TOOL_LOSS_ALIASES["ipae"], None),
            (structure_iptm_constraint, "iptm", AF2_MULTIMER_TOOL_LOSS_ALIASES["iptm"], None),
            (structure_contact_constraint, "con", "con", None),
        ],
    )
    def test_first_class_af2_terms(
        self,
        protein_sequence,
        constraint_fn,
        expected_loss,
        expected_tool_loss,
        tool_metrics,
    ):
        """First-class AF2 structure terms expose canonical names and adapt tool-layer keys."""
        binder = Sequence("EVQLVESG", "protein")
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold2_multimer",
            alphafold2_multimer_config=AlphaFold2MultimerStructureConfig(
                target_pdb=PDL1_PDB.read_text(),
                binder_chain="B",
                target_chains=["A"],
            ),
        )
        structure = Structure(structure=PDL1_PDB.read_text(), structure_format="pdb")
        output = SimpleNamespace(
            loss=0.5,
            metrics=tool_metrics or {expected_tool_loss: 0.5},
            structure=structure,
        )

        with patch("proto_language.utils.alphafold2_multimer.run_alphafold2_binder") as mock_af2:
            mock_af2.return_value = output
            (result,) = constraint_fn([(binder, protein_sequence)], config)

        if expected_loss == "plddt":
            assert result.score == 0.25
            assert result.metadata["avg_plddt"] == 0.75
            assert result.metadata["loss_plddt"] == 0.25
        elif expected_loss == "ipae":
            expected_score = (0.5 * AF2_MULTIMER_PAE_MAXIMUM) / PAE_MAXIMUM
            assert result.score == pytest.approx(expected_score)
            assert result.metadata[expected_loss] == pytest.approx(0.5 * AF2_MULTIMER_PAE_MAXIMUM)
        else:
            assert result.score == 0.5
            if expected_loss in result.metadata:
                assert result.metadata[expected_loss] == 0.5
        tool_input, tool_config = mock_af2.call_args[0]
        assert tool_input.binder_chain == "B"
        assert tool_input.target_chain == "A"
        assert tool_config.compute_gradient is False
        assert result.metadata["af2_loss_key"] == expected_loss
        assert result.metadata[f"loss_{expected_loss}"] == (0.25 if expected_loss == "plddt" else 0.5)
        assert tool_config.loss_weights == {expected_tool_loss: 1.0}
        assert len(result.structures) == 2

    def test_af2_ptm_forward_scores_metric(self, protein_sequence):
        """AF2 pTM is available for forward scoring even though it is not compiler-backed."""
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold2_multimer",
            alphafold2_multimer_config=AlphaFold2MultimerStructureConfig(
                target_pdb=PDL1_PDB.read_text(),
                binder_chain="B",
                target_chains=["A"],
            ),
        )
        structure = Structure(structure=PDL1_PDB.read_text(), structure_format="pdb")
        output = SimpleNamespace(loss=0.0, metrics={"ptm": 0.8}, structure=structure)

        with patch("proto_language.utils.alphafold2_multimer.run_alphafold2_binder") as mock_af2:
            mock_af2.return_value = output
            (result,) = structure_ptm_constraint([(protein_sequence, Sequence("A" * 10, "protein"))], config)

        assert result.score == pytest.approx(0.2)
        assert result.metadata["ptm"] == 0.8
        assert "af2_loss_key" not in result.metadata
        assert AF2_MULTIMER_CONFIDENCE_LOSS_BY_METRIC["ptm"] is None
        assert af2_multimer_confidence_loss_weights("ptm") == {}
        assert mock_af2.call_args[0][1].loss_weights == {}

    @pytest.mark.parametrize("tool_name", ["alphafold3", "boltz2", "chai1"])
    def test_iptm_dispatches_to_correct_tool(self, protein_sequence, protein_sequence_b, tool_name):
        """Test that ipTM constraint dispatches to supported tools."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool=tool_name)

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            if tool_name == "alphafold3":
                mock_predict.return_value = make_mock_output(
                    [make_mock_structure(avg_plddt=90.0, ptm=0.8, iptm=0.7, avg_pae=5.0)]
                )
            else:
                mock_predict.return_value = make_mock_output(
                    [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.0)]
                )

            structure_iptm_constraint(proposals, config)

            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0]
            assert call_args[1] == tool_name

    def test_af3_alias_works(self, protein_sequence):
        """Test that 'alphafold3' tool works."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=90.0, ptm=0.8, iptm=0.7, avg_pae=8.5)]
            )

            results = structure_plddt_constraint(proposals, config)

            mock_predict.assert_called_once()
            assert results[0].score == pytest.approx(0.1)


# ============================================================================
# Test Metric Availability Validation
# ============================================================================


class TestMetricAvailability:
    """Test that metrics are validated against tool capabilities."""

    def test_esmfold_supports_plddt_and_ptm(self, protein_sequence):
        """Test that ESMFold supports avg_plddt and ptm."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.9, ptm=0.8)])

            # Should not raise
            structure_plddt_constraint(proposals, config)
            structure_ptm_constraint(proposals, config)

    def test_esmfold_does_not_support_iptm(self, protein_sequence):
        """Test that ESMFold raises error for ipTM."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with pytest.raises(ValueError, match="Metric 'iptm' is not available for tool 'esmfold'"):
            structure_iptm_constraint(proposals, config)

    def test_alphafold3_supports_all_metrics(self, protein_sequence):
        """Test that AlphaFold3 supports all metrics."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=90.0, ptm=0.8, iptm=0.7, avg_pae=5.0)]
            )

            # All should work without raising
            structure_plddt_constraint(proposals, config)
            structure_ptm_constraint(proposals, config)
            structure_iptm_constraint(proposals, config)
            structure_pae_constraint(proposals, config)

    def test_chai_supports_all_metrics(self, protein_sequence):
        """Test that Chai1 supports all metrics."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="chai1")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.0)]
            )

            # All should work without raising
            structure_plddt_constraint(proposals, config)
            structure_ptm_constraint(proposals, config)
            structure_iptm_constraint(proposals, config)
            structure_pae_constraint(proposals, config)

    def test_boltz_supports_all_metrics(self, protein_sequence):
        """Test that Boltz supports all metrics."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="boltz2")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.0)]
            )

            # All should work without raising
            structure_plddt_constraint(proposals, config)
            structure_ptm_constraint(proposals, config)
            structure_iptm_constraint(proposals, config)
            structure_pae_constraint(proposals, config)

    @pytest.mark.parametrize("structure_tool", ["alphafold3", "boltz2", "chai1"])
    @pytest.mark.parametrize(
        "constraint_fn,metric",
        [(structure_iplddt_constraint, "iplddt"), (structure_ipae_constraint, "ipae")],
    )
    def test_interface_confidence_metrics_are_af2m_only(self, protein_sequence, structure_tool, constraint_fn, metric):
        """Interface-local confidence constraints are only wired for AF2 multimer."""
        config = StructureBasedConstraintConfig(structure_tool=structure_tool)

        with pytest.raises(ValueError, match=f"Metric '{metric}' is not available for tool '{structure_tool}'"):
            constraint_fn([(protein_sequence,)], config)

    def test_tool_available_metrics_constant(self):
        """Test that TOOL_AVAILABLE_METRICS has expected structure."""
        assert "esmfold" in TOOL_AVAILABLE_METRICS
        assert "alphafold3" in TOOL_AVAILABLE_METRICS
        assert "boltz2" in TOOL_AVAILABLE_METRICS
        assert "chai1" in TOOL_AVAILABLE_METRICS

        # ESMFold has limited metrics
        assert TOOL_AVAILABLE_METRICS["esmfold"] == {"avg_plddt", "ptm", "avg_pae"}
        assert TOOL_AVAILABLE_METRICS["alphafold3"] == {"avg_plddt", "ptm", "iptm", "avg_pae"}
        assert TOOL_AVAILABLE_METRICS["boltz2"] == {"avg_plddt", "ptm", "iptm", "avg_pae"}
        assert TOOL_AVAILABLE_METRICS["chai1"] == {"avg_plddt", "ptm", "iptm", "avg_pae"}
        assert TOOL_AVAILABLE_METRICS["alphafold2_multimer"] == {
            "avg_plddt",
            "ptm",
            "iptm",
            "avg_pae",
            "iplddt",
            "ipae",
        }


# ============================================================================
# Test Multimer Support
# ============================================================================


class TestMultimerSupport:
    """Test support for monomers, homodimers, and heteromultimers."""

    def test_monomer_single_chain(self, protein_sequence):
        """Test monomer prediction (single chain tuple)."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.9, ptm=0.8)])

            structure_plddt_constraint(proposals, config)

            # Verify single chain complex was created
            call_args = mock_predict.call_args
            complexes = call_args[0][0]  # First positional arg
            assert len(complexes) == 1
            assert len(complexes[0].chains) == 1
            assert complexes[0].chains[0].sequence == "MKTAYIAKQRQISFVK"

    def test_homodimer_two_identical_chains(self, protein_sequence):
        """Test homodimer prediction (same sequence twice)."""
        proposals = [(protein_sequence, protein_sequence)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.85, ptm=0.75)])

            structure_plddt_constraint(proposals, config)

            # Verify two-chain complex was created
            call_args = mock_predict.call_args
            complexes = call_args[0][0]
            assert len(complexes[0].chains) == 2
            assert complexes[0].chains[0].sequence == complexes[0].chains[1].sequence

    def test_heterodimer_two_different_chains(self, protein_sequence, protein_sequence_b):
        """Test heterodimer prediction (two different sequences)."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=88.0, ptm=0.78, iptm=0.72, avg_pae=5.0)]
            )

            structure_iptm_constraint(proposals, config)

            # Verify heterodimer complex was created
            call_args = mock_predict.call_args
            complexes = call_args[0][0]
            assert len(complexes[0].chains) == 2
            assert complexes[0].chains[0].sequence == "MKTAYIAKQRQISFVK"
            assert complexes[0].chains[1].sequence == "GVQVETISPGDGRTFPK"

    def test_homotrimer_three_chains(self, protein_sequence):
        """Test homotrimer prediction."""
        proposals = [(protein_sequence, protein_sequence, protein_sequence)]
        config = StructureBasedConstraintConfig(structure_tool="boltz2")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.82, ptm=0.7, iptm=0.65, avg_pae=7.5)]
            )

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            complexes = call_args[0][0]
            assert len(complexes[0].chains) == 3

    def test_batch_of_multiple_complexes(self, protein_sequence, protein_sequence_b):
        """Test batch processing of multiple complexes."""
        proposals = [
            (protein_sequence,),  # Monomer
            (protein_sequence, protein_sequence),  # Homodimer
            (protein_sequence, protein_sequence_b),  # Heterodimer
        ]
        config = StructureBasedConstraintConfig(structure_tool="chai1")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [
                    make_mock_structure(avg_plddt=0.9, ptm=0.85, iptm=0.8, avg_pae=8.8),
                    make_mock_structure(avg_plddt=0.85, ptm=0.8, iptm=0.75, avg_pae=8.2),
                    make_mock_structure(avg_plddt=0.88, ptm=0.82, iptm=0.78, avg_pae=8.5),
                ]
            )

            results = structure_plddt_constraint(proposals, config)

            assert len(results) == 3
            assert results[0].score == pytest.approx(0.1)  # 1 - 0.9
            assert results[1].score == pytest.approx(0.15)  # 1 - 0.85
            assert results[2].score == pytest.approx(0.12)  # 1 - 0.88

    def test_entity_types_correctly_set(self, protein_sequence):
        """Test that entity types are correctly inferred from sequences."""
        proposals = [(protein_sequence, protein_sequence)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.9, ptm=0.8)])

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            complexes = call_args[0][0]
            assert complexes[0].entity_types == ["protein", "protein"]


# ============================================================================
# Test Tool Configuration Passthrough
# ============================================================================


class TestToolConfigPassthrough:
    """Test that tool-specific configuration is passed correctly."""

    def test_esmfold_config_passthrough(self, protein_sequence):
        """Test that ESMFold-specific config is passed through."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(
            structure_tool="esmfold",
            esmfold_config={
                "verbose": True,
                "residue_idx_offset": 256,
                "chain_linker": "GGGGG",
            },
        )

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.9, ptm=0.8)])

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            passed_tool_config = call_args[0][2]  # Third positional arg
            # Config is now a typed ESMFoldConfig object (converted from dict)
            from proto_tools import ESMFoldConfig

            assert isinstance(passed_tool_config, ESMFoldConfig)
            # BaseConfig.verbose is int (0/1/2/...) not bool — truthy means verbose enabled.
            assert passed_tool_config.verbose
            assert passed_tool_config.residue_idx_offset == 256
            assert passed_tool_config.chain_linker == "GGGGG"

    def test_alphafold3_config_passthrough(self, protein_sequence):
        """Test that AlphaFold3-specific config is passed through."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold3",
            alphafold3_config={
                "seeds": [0, 1, 2],
                "use_msa": False,
                "verbose": True,
            },
        )

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=90.0, ptm=0.8, iptm=0.7, avg_pae=8.5)]
            )

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            passed_tool_config = call_args[0][2]
            # Config is now a typed AlphaFold3Config object (converted from dict)
            from proto_tools import AlphaFold3Config

            assert isinstance(passed_tool_config, AlphaFold3Config)
            assert passed_tool_config.seeds == [0, 1, 2]
            assert passed_tool_config.use_msa is False

    def test_empty_tool_config_default(self, protein_sequence):
        """Test that empty tool config works (uses defaults)."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.9, ptm=0.8)])

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            passed_tool_config = call_args[0][2]
            # Config is now a typed ESMFoldConfig object with default values
            from proto_tools import ESMFoldConfig

            assert isinstance(passed_tool_config, ESMFoldConfig)
            # Verify it has default values
            assert passed_tool_config.device == "cuda"
            assert not passed_tool_config.verbose


# ============================================================================
# Test Metadata Storage
# ============================================================================


class TestMetadataStorage:
    """Test that results are correctly stored in sequence metadata."""

    def test_plddt_metadata_storage(self, protein_sequence):
        """Test that pLDDT and related metadata is carried on the result."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_struct = make_mock_structure(avg_plddt=0.92, ptm=0.88)
            mock_predict.return_value = make_mock_output([mock_struct])

            metadata = structure_plddt_constraint(proposals, config)[0].metadata
            assert metadata["avg_plddt"] == 0.92
            assert metadata["pdb_output"] == mock_struct.structure_pdb
            assert metadata["structure_tool"] == "esmfold"

    def test_ptm_metadata_storage(self, protein_sequence):
        """Test that pTM constraint carries ptm in result metadata."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(avg_plddt=0.9, ptm=0.85)])

            metadata = structure_ptm_constraint(proposals, config)[0].metadata
            assert metadata["ptm"] == 0.85
            assert metadata["structure_tool"] == "esmfold"

    def test_iptm_metadata_storage(self, protein_sequence, protein_sequence_b):
        """Test that ipTM constraint carries iptm in result metadata."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=90.0, ptm=0.85, iptm=0.78, avg_pae=8.2)]
            )

            metadata = structure_iptm_constraint(proposals, config)[0].metadata
            assert metadata["iptm"] == 0.78
            assert metadata["structure_tool"] == "alphafold3"

    def test_pae_metadata_storage(self, protein_sequence):
        """Test that pAE constraint carries avg_pae in result metadata."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=90.0, ptm=0.85, iptm=0.78, avg_pae=8.8)]
            )

            metadata = structure_pae_constraint(proposals, config)[0].metadata
            assert metadata["avg_pae"] == 8.8

    def test_structure_attached_to_first_input_only(self, protein_sequence, protein_sequence_b):
        """Test that predicted structure is attached to slot 0 only via Constraint.evaluate()."""
        from proto_language.core import Constraint, Segment

        seg_a = Segment(sequence=protein_sequence.sequence, sequence_type="protein")
        seg_b = Segment(sequence=protein_sequence_b.sequence, sequence_type="protein")
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=90.0, ptm=0.85, iptm=0.78, avg_pae=8.2)]
            )
            Constraint(
                inputs=[seg_a, seg_b],
                function=structure_plddt_constraint,
                function_config=config,
            ).evaluate()

            # Slot 0 got the predicted structure; slot 1 did not.
            assert seg_a.proposal_sequences[0].structure is not None
            assert seg_b.proposal_sequences[0].structure is None


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unknown_tool_raises_error(self, protein_sequence):
        """Test that unknown tool raises ValidationError at config time."""
        from pydantic import ValidationError

        # Pydantic's Literal validation catches invalid tools at construction.
        with pytest.raises(ValidationError):
            StructureBasedConstraintConfig(structure_tool="unknown_tool")

    def test_prediction_failure_raises_error(self, protein_sequence):
        """Test that prediction failure raises an error."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.side_effect = RuntimeError("GPU out of memory")

            with pytest.raises(RuntimeError, match="GPU out of memory"):
                structure_plddt_constraint(proposals, config)

    def test_missing_metric_returns_worst_score(self, protein_sequence, caplog):
        """Test that missing metric in output returns score of 1.0."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            # Return structure without the expected metric
            mock_predict.return_value = make_mock_output(
                [
                    make_mock_structure(ptm=0.8)  # No avg_plddt
                ]
            )

            with caplog.at_level(
                "WARNING", logger="proto_language.constraint.protein_structure.structure_confidence_constraint"
            ):
                results = structure_plddt_constraint(proposals, config)

            assert results[0].score == 1.0
            assert "Metric 'avg_plddt' not found in structure output" in caplog.text

    def test_empty_proposals_returns_empty_scores(self):
        """Test that empty proposals list returns empty scores."""
        proposals = []
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([])

            results = structure_plddt_constraint(proposals, config)

            assert results == []

    def test_batch_with_partial_failure(self, protein_sequence, protein_sequence_b, caplog):
        """Test batch where some predictions have missing metrics."""
        proposals = [
            (protein_sequence,),
            (protein_sequence_b,),
        ]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [
                    make_mock_structure(avg_plddt=0.9, ptm=0.8),  # Good
                    make_mock_structure(ptm=0.7),  # Missing avg_plddt
                ]
            )

            with caplog.at_level(
                "WARNING", logger="proto_language.constraint.protein_structure.structure_confidence_constraint"
            ):
                results = structure_plddt_constraint(proposals, config)

            assert len(results) == 2
            assert results[0].score == pytest.approx(0.1)  # Good result
            assert results[1].score == 1.0  # Missing metric
            assert "Metric 'avg_plddt' not found in structure output" in caplog.text


# ============================================================================
# Test Configuration Defaults
# ============================================================================


class TestConfigurationDefaults:
    """Test default configuration values."""

    def test_default_tool_is_esmfold(self):
        """Test that default structure_tool is 'esmfold'."""
        config = StructureBasedConstraintConfig()
        assert config.structure_tool == "esmfold"

    def test_default_tool_config_is_esmfold(self):
        """Test that default tool_config returns a default ESMFoldConfig."""
        config = StructureBasedConstraintConfig()
        from proto_tools import ESMFoldConfig

        assert isinstance(config.tool_config, ESMFoldConfig)
        assert isinstance(config.esmfold_config, ESMFoldConfig)
        assert config.esmfold_config.device == "cuda"
        assert not config.esmfold_config.verbose

    def test_tool_name_strict(self, protein_sequence):
        """Test that tool names must be exact (case-sensitive, no whitespace)."""
        from pydantic import ValidationError

        # Only exact lowercase names should work
        config = StructureBasedConstraintConfig(structure_tool="esmfold")
        assert config.structure_tool == "esmfold"

        # Case variations and whitespace should fail
        for tool_variant in ["ESMFold", "ESMFOLD", " esmfold ", "EsmFold", "af3"]:
            with pytest.raises(ValidationError):
                StructureBasedConstraintConfig(structure_tool=tool_variant)


# ============================================================================
# Integration-Style Tests
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_heterodimer_interface_assessment(self, protein_sequence, protein_sequence_b):
        """Test assessing a heterodimer interface with ipTM."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold3",
            alphafold3_config={"seeds": [0]},
        )

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=88.0, ptm=0.82, iptm=0.75, avg_pae=8.0)]
            )

            # Get ipTM score for interface quality
            iptm_results = structure_iptm_constraint(proposals, config)

            assert iptm_results[0].score == pytest.approx(0.25)  # 1 - 0.75
            assert iptm_results[0].metadata["iptm"] == 0.75

    def test_compare_multiple_tools_same_sequence(self, protein_sequence):
        """Test comparing predictions from different tools."""
        proposals = [(protein_sequence,)]

        scores_by_tool: dict[str, float] = {}
        for tool in ["esmfold", "alphafold3", "boltz2", "chai1"]:
            config = StructureBasedConstraintConfig(structure_tool=tool)

            with patch(
                "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
            ) as mock_predict:
                # Simulate slightly different results per tool
                plddt = {"esmfold": 0.85, "alphafold3": 92.0, "boltz2": 0.88, "chai1": 0.90}[tool]
                mock_predict.return_value = make_mock_output(
                    [make_mock_structure(avg_plddt=plddt, ptm=0.8, iptm=0.7, avg_pae=8.2)]
                )

                results = structure_plddt_constraint(proposals, config)
                scores_by_tool[tool] = results[0].score

        # Verify different tools give different scores
        assert scores_by_tool["esmfold"] == pytest.approx(0.15)
        assert scores_by_tool["alphafold3"] == pytest.approx(0.08)
        assert scores_by_tool["boltz2"] == pytest.approx(0.12)
        assert scores_by_tool["chai1"] == pytest.approx(0.10)

    def test_screening_multiple_proposals(self, protein_sequence, protein_sequence_b):
        """Test screening multiple proposal complexes."""
        # Simulate screening 5 proposal dimers
        seq_c = Sequence("MAEGEITTFTALTEKFNLPPGN", "protein")
        seq_d = Sequence("MGSSHHHHHHSSGLVPRGSH", "protein")
        seq_e = Sequence("MKFLILLFNILCLFPVLAAD", "protein")

        proposals = [
            (protein_sequence, protein_sequence_b),
            (protein_sequence, seq_c),
            (protein_sequence_b, seq_d),
            (seq_c, seq_e),
            (protein_sequence, seq_e),
        ]

        config = StructureBasedConstraintConfig(
            structure_tool="chai1",
            chai1_config={"verbose": False},
        )

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            # Simulate varying quality predictions
            mock_predict.return_value = make_mock_output(
                [
                    make_mock_structure(avg_plddt=0.92, ptm=0.88, iptm=0.85),
                    make_mock_structure(avg_plddt=0.78, ptm=0.72, iptm=0.65),
                    make_mock_structure(avg_plddt=0.85, ptm=0.80, iptm=0.75),
                    make_mock_structure(avg_plddt=0.60, ptm=0.55, iptm=0.45),
                    make_mock_structure(avg_plddt=0.88, ptm=0.82, iptm=0.78),
                ]
            )

            results = structure_iptm_constraint(proposals, config)

            assert len(results) == 5
            # Best proposal (highest ipTM = lowest score)
            best_idx = [r.score for r in results].index(min(r.score for r in results))
            assert best_idx == 0  # First proposal had ipTM=0.85


# ============================================================================
# Test structure-composite (one-call, all-metrics, composite score)
# ============================================================================


class TestStructureComposite:
    """Test the composite confidence constraint."""

    @pytest.mark.parametrize(
        "tool,plddt,iptm,ptm,pae,expected",
        [
            # Boundaries.
            ("chai1", 1.0, 1.0, 1.0, 0.0, 0.0),
            ("chai1", 0.0, 0.0, 0.0, PAE_MAXIMUM, 1.0),
            # Interior for each supported tool: (0.1 + 0.2 + 0.3 + 0.1) / 4 = 0.175.
            ("chai1", 0.9, 0.8, 0.7, 3.175, 0.175),
            ("boltz2", 0.9, 0.8, 0.7, 3.175, 0.175),
            # AlphaFold3 reports pLDDT on 0-100 scale; composite must normalize for scoring.
            ("alphafold3", 90.0, 0.8, 0.7, 3.175, 0.175),
        ],
    )
    def test_composite_scoring(self, protein_sequence, protein_sequence_b, tool, plddt, iptm, ptm, pae, expected):
        """Composite = mean of (1-plddt_norm, 1-iptm, 1-ptm, pae/PAE_MAX); AF3 plddt is 0-100 scale."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool=tool)

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=plddt, iptm=iptm, ptm=ptm, avg_pae=pae)]
            )
            [result] = structure_composite_constraint(proposals, config)
            assert abs(result.score - expected) < 1e-9

    def test_composite_missing_metric_returns_worst(self, protein_sequence, protein_sequence_b):
        """If the tool omits any of the four metrics (e.g. degenerate single-chain input, missing ``iptm``)."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool="chai1")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, ptm=0.7, avg_pae=3.0)]  # iptm absent
            )
            [result] = structure_composite_constraint(proposals, config)
            assert result.score == 1.0

    def test_composite_exposes_normalized_metrics_for_post_hoc_thresholding(self, protein_sequence, protein_sequence_b):
        """All four metadata metrics are normalized to ``[0, 1]`` so downstream threshold code is tool-agnostic.

        AF3 pLDDT is divided by 100 (0-100 scale → 0-1); pAE is divided by ``PAE_MAXIMUM`` (31.75 Å).
        """
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=90.0, iptm=0.8, ptm=0.7, avg_pae=3.175)]
            )
            [result] = structure_composite_constraint(proposals, config)

            meta = result.metadata
            # AF3 plddt 90.0 -> 0.9; pae 3.175 Å -> 0.1; others already in [0, 1].
            assert meta["composite_avg_plddt"] == pytest.approx(0.9)
            assert meta["composite_iptm"] == pytest.approx(0.8)
            assert meta["composite_ptm"] == pytest.approx(0.7)
            assert meta["composite_avg_pae"] == pytest.approx(0.1)
            assert meta["structure_tool"] == "alphafold3"
            assert "ATOM" in meta["pdb_output"]

    def test_composite_rejects_single_chain_tools(self, protein_sequence):
        """ESMFold is missing ``iptm`` - reject at config time via TOOL_AVAILABLE_METRICS rather than silently degrade."""
        with pytest.raises(ValueError, match=r"missing.*iptm"):
            structure_composite_constraint(
                [(protein_sequence,)], StructureBasedConstraintConfig(structure_tool="esmfold")
            )

    def test_composite_batches_to_one_predict_call(self, protein_sequence, protein_sequence_b):
        """Design-intent assert: N proposals → 1 ``predict_structures`` call (vs 4N for stacked single-metric constraints)."""
        n = 5
        proposals = [(protein_sequence, protein_sequence_b) for _ in range(n)]
        config = StructureBasedConstraintConfig(structure_tool="chai1")

        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(avg_plddt=0.9, iptm=0.8, ptm=0.7, avg_pae=3.0) for _ in range(n)]
            )
            results = structure_composite_constraint(proposals, config)
            assert len(results) == n and mock_predict.call_count == 1


# ============================================================================
# Test ESMFold2 (native ``plddt`` key; complex-capable with ipTM)
# ============================================================================


class TestESMFold2:
    """ESMFold2 uses a native ``plddt`` key and is complex-capable.

    Unlike ESMFold v1 it yields ``iptm`` and folds complexes, so the native-name bridge
    must resolve ``plddt`` or confidence constraints silently return the worst score.
    """

    def test_available_metrics(self):
        """ESMFold2 advertises the complex-capable canonical metric set."""
        assert TOOL_AVAILABLE_METRICS["esmfold2"] == {"avg_plddt", "ptm", "iptm", "avg_pae"}

    def test_plddt_bridges_native_key(self, protein_sequence):
        """Regression: ESMFold2 reports ``plddt``; structure-plddt must score it, not return the worst score."""
        config = StructureBasedConstraintConfig(structure_tool="esmfold2")
        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([make_mock_structure(plddt=0.9, ptm=0.8, avg_pae=5.0)])
            [result] = structure_plddt_constraint([(protein_sequence,)], config)
            assert result.score == pytest.approx(0.1)
            assert mock_predict.call_args[0][1] == "esmfold2"

    def test_iptm_supported_for_complexes(self, protein_sequence, protein_sequence_b):
        """ESMFold2 produces ipTM for multi-chain complexes (ESMFold v1 does not)."""
        config = StructureBasedConstraintConfig(structure_tool="esmfold2")
        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(plddt=0.9, ptm=0.8, iptm=0.75, avg_pae=5.0)]
            )
            [result] = structure_iptm_constraint([(protein_sequence, protein_sequence_b)], config)
            assert result.score == pytest.approx(0.25)

    def test_composite_supported(self, protein_sequence, protein_sequence_b):
        """ESMFold2 has all four metrics and folds complexes, so structure-composite accepts it."""
        config = StructureBasedConstraintConfig(structure_tool="esmfold2")
        with patch(
            "proto_language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output(
                [make_mock_structure(plddt=0.9, iptm=0.8, ptm=0.7, avg_pae=3.175)]
            )
            [result] = structure_composite_constraint([(protein_sequence, protein_sequence_b)], config)
            assert result.score == pytest.approx(0.175)
