"""tests/language_tests/generator_tests/test_ligandmpnn_generator.py."""

import copy
from pathlib import Path

import pytest
from proto_tools import InverseFoldingStructureInput

from proto_language.language.core import Segment
from proto_language.language.generator import (
    LigandMPNNGenerator,
    LigandMPNNGeneratorConfig,
)

DEFAULT_CHECKPOINT = Path.home() / ".foundry" / "checkpoints" / "ligandmpnn_v_32_010_25.pt"


@pytest.mark.uses_gpu
class TestLigandMPNNGenerator:
    """Integration tests for LigandMPNN generator (require GPU and checkpoint)."""

    def test_basic_sampling(self, temp_pdb_file):
        """Test basic sequence generation from structure."""
        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=temp_pdb_file,
                temperature=0.1,
            )
        )

        segment = Segment(length=5, sequence_type="protein")
        generator.assign(segment)

        assert generator._assigned_segments == (segment,)

        generator.sample()

        assert segment.proposal_sequences[0].sequence is not None
        assert len(segment.proposal_sequences[0].sequence) == 5
        assert segment.proposal_sequences[0].sequence_type == "protein"

    def test_fixed_positions(self, temp_pdb_file):
        """Test that fixed positions are preserved in generated sequences."""
        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(
                    structure=temp_pdb_file,
                    fixed_positions={"A": [1, 2]},
                ),
                temperature=0.1,
            )
        )

        segment = Segment(length=5, sequence_type="protein")
        generator.assign(segment)
        generator.sample()

        # Fixed positions should match original PDB residues
        assert segment.proposal_sequences[0].sequence[0] == "A"  # Position 1 is ALA
        assert segment.proposal_sequences[0].sequence[1] == "G"  # Position 2 is GLY

    def test_batch_sampling(self, temp_pdb_file):
        """Test generating multiple sequences from single structure."""
        num_proposals = 3
        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=temp_pdb_file,
                temperature=0.1,
            )
        )

        segment = Segment(sequence="AGSVL", sequence_type="protein")
        generator.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_proposals)]

        generator.sample()

        for i in range(num_proposals):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) == 5

    def test_batch_size_sampling(self, temp_pdb_file):
        """Test LigandMPNN generator with batch_size>1 for GPU memory management."""
        num_proposals = 4
        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=temp_pdb_file,
                temperature=0.1,
                batch_size=2,
            )
        )

        segment = Segment(sequence="AGSVL", sequence_type="protein")
        generator.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_proposals)]

        assert generator.batch_size == 2

        generator.sample()

        for i in range(num_proposals):
            assert segment.proposal_sequences[i].sequence is not None
            assert len(segment.proposal_sequences[i].sequence) == 5
            assert segment.proposal_sequences[i].sequence_type == "protein"


class TestLigandMPNNGeneratorValidation:
    """Unit tests for LigandMPNN configuration and validation (no GPU required)."""

    @pytest.mark.parametrize("sequence_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, temp_pdb_file, sequence_type):
        """LigandMPNN should reject non-protein segments."""
        generator = LigandMPNNGenerator(LigandMPNNGeneratorConfig(structure_inputs=temp_pdb_file))
        segment = Segment(length=50, sequence_type=sequence_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        assert "does not support sequence type" in str(exc_info.value)

    def test_rejects_ligand_segment(self, temp_pdb_file):
        """LigandMPNN should reject ligand segments (ligands cannot be mutated)."""
        generator = LigandMPNNGenerator(LigandMPNNGeneratorConfig(structure_inputs=temp_pdb_file))
        segment = Segment(sequence="CCC", sequence_type="ligand")

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            generator.assign(segment)

    def test_pdb_content_string(self, sample_pdb_content):
        """Should accept PDB content as a string (not just file path)."""
        generator = LigandMPNNGenerator(LigandMPNNGeneratorConfig(structure_inputs=sample_pdb_content))

        # Structure should be resolved automatically
        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].structure is not None

    def test_structure_without_chains_to_redesign_defaults_to_all(self, temp_pdb_file):
        """When chains_to_redesign is not specified, chain_ids_to_redesign should resolve to all chains."""
        generator = LigandMPNNGenerator(LigandMPNNGeneratorConfig(structure_inputs=temp_pdb_file))

        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].chain_ids_to_redesign == ["A"]  # Only chain A in sample PDB

    def test_structure_input_with_chains_to_redesign(self, temp_pdb_file):
        """Should accept InverseFoldingStructureInput with chains_to_redesign."""
        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(
                    structure=temp_pdb_file,
                    chains_to_redesign=["A"],
                )
            )
        )

        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].chain_ids_to_redesign == ["A"]

    def test_multiple_structure_inputs(self, sample_pdb_content):
        """Should accept multiple InverseFoldingStructureInput objects."""
        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=[
                    InverseFoldingStructureInput(
                        structure=sample_pdb_content,
                        chains_to_redesign=["A"],
                        fixed_positions={"A": [1, 2]},
                    ),
                    InverseFoldingStructureInput(
                        structure=sample_pdb_content,
                        chains_to_redesign=["A"],
                    ),
                ]
            )
        )

        assert len(generator.structure_inputs) == 2
        assert generator.structure_inputs[0].fixed_positions.chains == {"A": [1, 2]}
        assert generator.structure_inputs[1].fixed_positions is None
