import copy
import os
import tempfile

import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import (
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)
from proto_language.tools.inverse_folding.shared_data_models import InverseFoldingStructureInput

# Sample PDB content for testing (minimal valid structure)
SAMPLE_PDB_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.326   1.562   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.941   2.877   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.449   2.831   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.074   1.772   0.000  1.00  0.00           O
ATOM      9  N   SER A   3       6.032   4.027   0.000  1.00  0.00           N
ATOM     10  CA  SER A   3       7.476   4.180   0.000  1.00  0.00           C
ATOM     11  C   SER A   3       8.064   5.572   0.000  1.00  0.00           C
ATOM     12  O   SER A   3       7.337   6.562   0.000  1.00  0.00           O
ATOM     13  OG  SER A   3       7.929   3.453   1.135  1.00  0.00           O
ATOM     14  N   VAL A   4       9.377   5.660   0.000  1.00  0.00           N
ATOM     15  CA  VAL A   4      10.044   6.955   0.000  1.00  0.00           C
ATOM     16  C   VAL A   4      11.548   6.820   0.000  1.00  0.00           C
ATOM     17  O   VAL A   4      12.101   5.720   0.000  1.00  0.00           O
ATOM     18  CB  VAL A   4       9.566   7.867  -1.140  1.00  0.00           C
ATOM     19  CG1 VAL A   4      10.238   9.235  -1.043  1.00  0.00           C
ATOM     20  CG2 VAL A   4       8.050   8.008  -1.071  1.00  0.00           C
ATOM     21  N   LEU A   5      12.207   7.978   0.000  1.00  0.00           N
ATOM     22  CA  LEU A   5      13.655   8.068   0.000  1.00  0.00           C
ATOM     23  C   LEU A   5      14.195   9.485   0.000  1.00  0.00           C
ATOM     24  O   LEU A   5      13.424  10.440   0.000  1.00  0.00           O
ATOM     25  CB  LEU A   5      14.232   7.264  -1.171  1.00  0.00           C
ATOM     26  CG  LEU A   5      13.781   7.730  -2.561  1.00  0.00           C
ATOM     27  CD1 LEU A   5      14.329   6.786  -3.630  1.00  0.00           C
ATOM     28  CD2 LEU A   5      14.248   9.152  -2.857  1.00  0.00           C
END
"""


@pytest.fixture
def temp_pdb_file():
    """Create a temporary PDB file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(SAMPLE_PDB_CONTENT)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.mark.uses_gpu
class TestProteinMPNNGenerator:
    """Integration tests for ProteinMPNN generator (require GPU)."""

    def test_basic_sampling(self, temp_pdb_file):
        """Test basic sequence generation from structure."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=temp_pdb_file,
                temperature=0.1,
            )
        )

        segment = Segment(length=5, sequence_type="protein")
        generator.assign(segment)

        assert generator._assigned_segment is segment

        generator.sample()

        assert segment.candidate_sequences[0].sequence is not None
        assert len(segment.candidate_sequences[0].sequence) == 5
        assert segment.candidate_sequences[0].sequence_type == "protein"

    def test_fixed_positions(self, temp_pdb_file):
        """Test that fixed positions are preserved in generated sequences."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
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
        assert segment.candidate_sequences[0].sequence[0] == "A"  # Position 1 is ALA
        assert segment.candidate_sequences[0].sequence[1] == "G"  # Position 2 is GLY

    def test_batch_sampling(self, temp_pdb_file):
        """Test generating multiple sequences from single structure."""
        num_candidates = 3
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=temp_pdb_file,
                temperature=0.1,
            )
        )

        segment = Segment(sequence="AGSVL", sequence_type="protein")
        generator.assign(segment)
        segment.candidate_sequences = [
            copy.deepcopy(segment.original_sequence) for _ in range(num_candidates)
        ]

        generator.sample()

        for i in range(num_candidates):
            assert segment.candidate_sequences[i].sequence is not None
            assert len(segment.candidate_sequences[i].sequence) == 5


class TestProteinMPNNGeneratorValidation:
    """Unit tests for ProteinMPNN configuration and validation (no GPU required)."""

    def test_rejects_non_protein_segment(self, temp_pdb_file):
        """ProteinMPNN should reject non-protein segments."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=temp_pdb_file)
        )
        segment = Segment(length=50, sequence_type="dna")

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        assert "does not support sequence type" in str(exc_info.value)

    def test_rejects_ligand_segment(self, temp_pdb_file):
        """ProteinMPNN should reject ligand segments (ligands cannot be mutated)."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=temp_pdb_file)
        )
        segment = Segment(sequence="CCC", sequence_type="ligand")

        with pytest.raises(
            ValueError, match="Cannot assign generator to ligand segment"
        ):
            generator.assign(segment)

    def test_pdb_content_string(self):
        """Should accept PDB content as a string (not just file path)."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=SAMPLE_PDB_CONTENT)
        )

        # Structure should be resolved automatically
        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].structure is not None

    def test_structure_without_chain_ids_defaults_to_all(self, temp_pdb_file):
        """When chain_ids is not specified, should default to all chains."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=temp_pdb_file)
        )

        # chain_ids should be populated with all available chains
        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].chain_ids is not None
        assert generator.structure_inputs[0].chain_ids == [
            "A"
        ]  # Only chain A in sample PDB

    def test_structure_input_with_chain_ids(self, temp_pdb_file):
        """Should accept InverseFoldingStructureInput with chain_ids."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(
                    structure=temp_pdb_file,
                    chain_ids=["A"],
                )
            )
        )

        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].chain_ids == ["A"]

    def test_multiple_structure_inputs(self):
        """Should accept multiple InverseFoldingStructureInput objects."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=[
                    InverseFoldingStructureInput(
                        structure=SAMPLE_PDB_CONTENT,
                        chain_ids=["A"],
                        fixed_positions={"A": [1, 2]},
                    ),
                    InverseFoldingStructureInput(
                        structure=SAMPLE_PDB_CONTENT,
                        chain_ids=["A"],
                    ),
                ]
            )
        )

        assert len(generator.structure_inputs) == 2
        assert generator.structure_inputs[0].fixed_positions == {"A": [1, 2]}
        assert generator.structure_inputs[1].fixed_positions is None
