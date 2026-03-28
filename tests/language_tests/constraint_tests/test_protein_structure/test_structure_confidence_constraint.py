"""
tests/language_tests/constraint_tests/test_protein_structure/test_structure_confidence_constraint.py

Tests cover:
1. Score calculation for all metrics (pLDDT, pTM, ipTM, pAE)
2. Tool dispatching (ESMFold, AlphaFold3, Boltz, Chai1)
3. Metric availability validation per tool
4. Multimer support (monomers, homodimers, heteromultimers)
5. Tool configuration passthrough
6. Metadata storage
7. Error handling
"""

from unittest.mock import MagicMock, patch

import pytest
from proto_tools import Structure, StructurePredictionOutput

from proto_language.language.constraint.protein_structure.structure_confidence_constraint import (
    TOOL_AVAILABLE_METRICS,
    StructureBasedConstraintConfig,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.language.core import Sequence
from proto_language.storage import get_file_content, is_file_reference

# ============================================================================
# Fixtures
# ============================================================================

MOCK_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.90           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.90           C
END
"""


def make_mock_structure(**metrics) -> Structure:
    """Create a mock Structure with specified metrics."""
    structure = MagicMock(spec=Structure)
    structure.metrics = metrics
    structure.structure_pdb = MOCK_PDB
    return structure


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
            (1.0, 0.0),   # Perfect confidence
            (0.9, 0.1),
            (0.75, 0.25),
            (0.5, 0.5),
            (0.0, 1.0),   # No confidence
        ],
    )
    def test_plddt_scoring_esmfold(self, protein_sequence, metric_value, expected_score):
        """Test that pLDDT score = 1.0 - avg_plddt."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=metric_value, ptm=0.8)
            ])

            scores = structure_plddt_constraint(proposals, config)
            assert abs(scores[0] - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (100., 0.0),   # Perfect confidence
            (90., 0.1),
            (75., 0.25),
            (50., 0.5),
            (0., 1.0),   # No confidence
        ],
    )
    def test_plddt_scoring_af3(self, protein_sequence, metric_value, expected_score):
        """Test that pLDDT score = 1.0 - **normalized** avg_plddt."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=metric_value, ptm=0.8)
            ])

            scores = structure_plddt_constraint(proposals, config)
            assert abs(scores[0] - expected_score) < 1e-9

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=metric_value)
            ])

            scores = structure_ptm_constraint(proposals, config)
            assert abs(scores[0] - expected_score) < 1e-9

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=metric_value, avg_pae=0.85)
            ])

            scores = structure_iptm_constraint(proposals, config)
            assert abs(scores[0] - expected_score) < 1e-9

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=metric_value, avg_pae=0.85)
            ])

            scores = structure_iptm_constraint(proposals, config)
            assert abs(scores[0] - expected_score) < 1e-9

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=metric_value, avg_pae=0.85)
            ])

            scores = structure_iptm_constraint(proposals, config)
            assert abs(scores[0] - expected_score) < 1e-9

    @pytest.mark.parametrize(
        "metric_value,expected_score",
        [
            (0.0, 0.0),    # Perfect (low error)
            (15.875, 0.5),
            (31.75, 1.0),    # High error
        ],
    )
    def test_pae_scoring(self, protein_sequence, metric_value, expected_score):
        """Test that pAE score = avg_pae / 31.75."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=metric_value)
            ])

            scores = structure_pae_constraint(proposals, config)
            assert abs(scores[0] - expected_score) < 1e-9


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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            if tool_name == "alphafold3":
                mock_predict.return_value = make_mock_output([
                    make_mock_structure(avg_plddt=90., ptm=0.8, iptm=0.7, avg_pae=5.)
                ])
            else:
                mock_predict.return_value = make_mock_output([
                    make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.)
                ])

            structure_plddt_constraint(proposals, config)

            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0]
            assert call_args[1] == tool_name

    @pytest.mark.parametrize("tool_name", ["alphafold3", "boltz2", "chai1"])
    def test_iptm_dispatches_to_correct_tool(self, protein_sequence, protein_sequence_b, tool_name):
        """Test that ipTM constraint dispatches to supported tools."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool=tool_name)

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            if tool_name == "alphafold3":
                mock_predict.return_value = make_mock_output([
                    make_mock_structure(avg_plddt=90., ptm=0.8, iptm=0.7, avg_pae=5.)
                ])
            else:
                mock_predict.return_value = make_mock_output([
                    make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.)
                ])

            structure_iptm_constraint(proposals, config)

            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0]
            assert call_args[1] == tool_name

    def test_af3_alias_works(self, protein_sequence):
        """Test that 'alphafold3' tool works."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=90., ptm=0.8, iptm=0.7, avg_pae=8.5)
            ])

            scores = structure_plddt_constraint(proposals, config)

            mock_predict.assert_called_once()
            assert scores[0] == pytest.approx(0.1)


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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8)
            ])

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=90., ptm=0.8, iptm=0.7, avg_pae=5.)
            ])

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.)
            ])

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8, iptm=0.7, avg_pae=5.)
            ])

            # All should work without raising
            structure_plddt_constraint(proposals, config)
            structure_ptm_constraint(proposals, config)
            structure_iptm_constraint(proposals, config)
            structure_pae_constraint(proposals, config)

    def test_tool_available_metrics_constant(self):
        """Test that TOOL_AVAILABLE_METRICS has expected structure."""
        assert "esmfold" in TOOL_AVAILABLE_METRICS
        assert "alphafold3" in TOOL_AVAILABLE_METRICS
        assert "boltz2" in TOOL_AVAILABLE_METRICS
        assert "chai1" in TOOL_AVAILABLE_METRICS

        # ESMFold has limited metrics
        assert TOOL_AVAILABLE_METRICS["esmfold"] == {"avg_plddt", "ptm", "avg_pae"}


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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8)
            ])

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.85, ptm=0.75)
            ])

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=88., ptm=0.78, iptm=0.72, avg_pae=5.)
            ])

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.82, ptm=0.7, iptm=0.65, avg_pae=7.5)
            ])

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            complexes = call_args[0][0]
            assert len(complexes[0].chains) == 3

    def test_batch_of_multiple_complexes(self, protein_sequence, protein_sequence_b):
        """Test batch processing of multiple complexes."""
        proposals = [
            (protein_sequence,),                           # Monomer
            (protein_sequence, protein_sequence),          # Homodimer
            (protein_sequence, protein_sequence_b),        # Heterodimer
        ]
        config = StructureBasedConstraintConfig(structure_tool="chai1")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.85, iptm=0.8, avg_pae=8.8),
                make_mock_structure(avg_plddt=0.85, ptm=0.8, iptm=0.75, avg_pae=8.2),
                make_mock_structure(avg_plddt=0.88, ptm=0.82, iptm=0.78, avg_pae=8.5),
            ])

            scores = structure_plddt_constraint(proposals, config)

            assert len(scores) == 3
            assert scores[0] == pytest.approx(0.1)   # 1 - 0.9
            assert scores[1] == pytest.approx(0.15)  # 1 - 0.85
            assert scores[2] == pytest.approx(0.12)  # 1 - 0.88

    def test_entity_types_correctly_set(self, protein_sequence):
        """Test that entity types are correctly inferred from sequences."""
        proposals = [(protein_sequence, protein_sequence)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8)
            ])

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
            tool_config={
                "verbose": True,
                "residue_idx_offset": 256,
                "chain_linker": "GGGGG",
            },
        )

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8)
            ])

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            passed_tool_config = call_args[0][2]  # Third positional arg
            # Config is now a typed ESMFoldConfig object (converted from dict)
            from proto_tools import ESMFoldConfig
            assert isinstance(passed_tool_config, ESMFoldConfig)
            assert passed_tool_config.verbose is True
            assert passed_tool_config.residue_idx_offset == 256
            assert passed_tool_config.chain_linker == "GGGGG"

    def test_alphafold3_config_passthrough(self, protein_sequence):
        """Test that AlphaFold3-specific config is passed through."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold3",
            tool_config={
                "seeds": [0, 1, 2],
                "use_msa": False,
                "verbose": True,
            },
        )

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=90., ptm=0.8, iptm=0.7, avg_pae=8.5)
            ])

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
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8)
            ])

            structure_plddt_constraint(proposals, config)

            call_args = mock_predict.call_args
            passed_tool_config = call_args[0][2]
            # Config is now a typed ESMFoldConfig object with default values
            from proto_tools import ESMFoldConfig
            assert isinstance(passed_tool_config, ESMFoldConfig)
            # Verify it has default values
            assert passed_tool_config.device == "cuda"
            assert passed_tool_config.verbose is False


# ============================================================================
# Test Metadata Storage
# ============================================================================

class TestMetadataStorage:
    """Test that results are correctly stored in sequence metadata."""

    def test_plddt_metadata_storage(self, protein_sequence):
        """Test that pLDDT and related metadata is stored."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.92, ptm=0.88)
            ])

            structure_plddt_constraint(proposals, config)

            metadata = protein_sequence._metadata
            assert metadata["avg_plddt"] == 0.92
            # pdb_output is now a file reference
            assert is_file_reference(metadata["pdb_output"])
            assert get_file_content(metadata["pdb_output"]) == MOCK_PDB
            assert metadata["structure_tool"] == "esmfold"

    def test_ptm_metadata_storage(self, protein_sequence):
        """Test that pTM constraint stores ptm in metadata."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.85)
            ])

            structure_ptm_constraint(proposals, config)

            metadata = protein_sequence._metadata
            assert metadata["ptm"] == 0.85
            assert metadata["structure_tool"] == "esmfold"

    def test_iptm_metadata_storage(self, protein_sequence, protein_sequence_b):
        """Test that ipTM constraint stores iptm in metadata."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=90., ptm=0.85, iptm=0.78, avg_pae=8.2)
            ])

            structure_iptm_constraint(proposals, config)

            # Metadata should be on first sequence in tuple
            metadata = protein_sequence._metadata
            assert metadata["iptm"] == 0.78
            assert metadata["structure_tool"] == "alphafold3"

    def test_pae_metadata_storage(self, protein_sequence):
        """Test that pAE constraint stores avg_pae in metadata."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=90., ptm=0.85, iptm=0.78, avg_pae=8.8)
            ])

            structure_pae_constraint(proposals, config)

            metadata = protein_sequence._metadata
            assert metadata["avg_pae"] == 8.8

    def test_metadata_on_first_sequence_in_tuple(self, protein_sequence, protein_sequence_b):
        """Test that metadata is attached to first sequence in tuple only."""
        proposals = [(protein_sequence, protein_sequence_b)]
        config = StructureBasedConstraintConfig(structure_tool="alphafold3")

        # Clear any existing metadata
        protein_sequence._metadata = {}
        protein_sequence_b._metadata = {}

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=90., ptm=0.85, iptm=0.78, avg_pae=8.2)
            ])

            structure_plddt_constraint(proposals, config)

            # First sequence should have metadata
            assert "avg_plddt" in protein_sequence._metadata
            assert "pdb_output" in protein_sequence._metadata

            # Second sequence should NOT have metadata (or have empty)
            assert "avg_plddt" not in protein_sequence_b._metadata


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_unknown_tool_raises_error(self, protein_sequence):
        """Test that unknown tool raises ValidationError at config time."""
        from pydantic import ValidationError

        # Pydantic's Literal validation catches invalid tools before our validator runs
        with pytest.raises(ValidationError):
            config = StructureBasedConstraintConfig(structure_tool="unknown_tool")
            # If this passes, check the error message contains expected info
            assert "esmfold" in str(config) or "alphafold3" in str(config)

    def test_prediction_failure_raises_error(self, protein_sequence):
        """Test that prediction failure raises an error."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.side_effect = RuntimeError("GPU out of memory")

            with pytest.raises(RuntimeError, match="GPU out of memory"):
                structure_plddt_constraint(proposals, config)

    def test_missing_metric_returns_worst_score(self, protein_sequence, caplog):
        """Test that missing metric in output returns score of 1.0."""
        proposals = [(protein_sequence,)]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            # Return structure without the expected metric
            mock_predict.return_value = make_mock_output([
                make_mock_structure(ptm=0.8)  # No avg_plddt
            ])

            with caplog.at_level("WARNING", logger="proto_language.language.constraint.protein_structure.structure_confidence_constraint"):
                scores = structure_plddt_constraint(proposals, config)

            assert scores[0] == 1.0
            assert "Metric 'avg_plddt' not found in structure output" in caplog.text

    def test_empty_proposals_returns_empty_scores(self):
        """Test that empty proposals list returns empty scores."""
        proposals = []
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([])

            scores = structure_plddt_constraint(proposals, config)

            assert scores == []

    def test_batch_with_partial_failure(self, protein_sequence, protein_sequence_b, caplog):
        """Test batch where some predictions have missing metrics."""
        proposals = [
            (protein_sequence,),
            (protein_sequence_b,),
        ]
        config = StructureBasedConstraintConfig(structure_tool="esmfold")

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.9, ptm=0.8),  # Good
                make_mock_structure(ptm=0.7),  # Missing avg_plddt
            ])

            with caplog.at_level("WARNING", logger="proto_language.language.constraint.protein_structure.structure_confidence_constraint"):
                scores = structure_plddt_constraint(proposals, config)

            assert len(scores) == 2
            assert scores[0] == pytest.approx(0.1)  # Good result
            assert scores[1] == 1.0  # Missing metric
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

    def test_default_tool_config_is_empty(self):
        """Test that default tool_config creates a default ESMFoldConfig."""
        config = StructureBasedConstraintConfig()
        # New behavior: tool_config is automatically converted to typed config
        from proto_tools import ESMFoldConfig
        assert isinstance(config.tool_config, ESMFoldConfig)
        # Verify it uses default values
        assert config.tool_config.device == "cuda"
        assert config.tool_config.verbose is False

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
            tool_config={"seeds": [0]},
        )

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=88., ptm=0.82, iptm=0.75, avg_pae=8.)
            ])

            # Get ipTM score for interface quality
            iptm_scores = structure_iptm_constraint(proposals, config)

            assert iptm_scores[0] == pytest.approx(0.25)  # 1 - 0.75
            assert protein_sequence._metadata["iptm"] == 0.75

    def test_compare_multiple_tools_same_sequence(self, protein_sequence):
        """Test comparing predictions from different tools."""
        proposals = [(protein_sequence,)]

        results = {}
        for tool in ["esmfold", "alphafold3", "boltz2", "chai1"]:
            config = StructureBasedConstraintConfig(structure_tool=tool)

            with patch(
                "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
            ) as mock_predict:
                # Simulate slightly different results per tool
                plddt = {"esmfold": 0.85, "alphafold3": 92., "boltz2": 0.88, "chai1": 0.90}[tool]
                mock_predict.return_value = make_mock_output([
                    make_mock_structure(avg_plddt=plddt, ptm=0.8, iptm=0.7, avg_pae=8.2)
                ])

                scores = structure_plddt_constraint(proposals, config)
                results[tool] = scores[0]

        # Verify different tools give different scores
        assert results["esmfold"] == pytest.approx(0.15)
        assert results["alphafold3"] == pytest.approx(0.08)
        assert results["boltz2"] == pytest.approx(0.12)
        assert results["chai1"] == pytest.approx(0.10)

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
            tool_config={"verbose": False},
        )

        with patch(
            "proto_language.language.constraint.protein_structure.structure_confidence_constraint.predict_structures"
        ) as mock_predict:
            # Simulate varying quality predictions
            mock_predict.return_value = make_mock_output([
                make_mock_structure(avg_plddt=0.92, ptm=0.88, iptm=0.85),
                make_mock_structure(avg_plddt=0.78, ptm=0.72, iptm=0.65),
                make_mock_structure(avg_plddt=0.85, ptm=0.80, iptm=0.75),
                make_mock_structure(avg_plddt=0.60, ptm=0.55, iptm=0.45),
                make_mock_structure(avg_plddt=0.88, ptm=0.82, iptm=0.78),
            ])

            scores = structure_iptm_constraint(proposals, config)

            assert len(scores) == 5
            # Best proposal (highest ipTM = lowest score)
            best_idx = scores.index(min(scores))
            assert best_idx == 0  # First proposal had ipTM=0.85
