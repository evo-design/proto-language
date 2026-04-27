"""tests/language_tests/generator_tests/test_proteinmpnn_generator.py."""

import copy
from importlib import import_module
from types import SimpleNamespace

import pytest
from proto_tools import InverseFoldingStructureInput

from proto_language.language.core import Segment
from proto_language.language.generator import (
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)


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

        assert segment.proposal_sequences[0].sequence is not None
        assert len(segment.proposal_sequences[0].sequence) == 5
        assert segment.proposal_sequences[0].sequence_type == "protein"

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
        assert segment.proposal_sequences[0].sequence[0] == "A"  # Position 1 is ALA
        assert segment.proposal_sequences[0].sequence[1] == "G"  # Position 2 is GLY

    def test_batch_sampling(self, temp_pdb_file):
        """Test generating multiple sequences from single structure."""
        num_proposals = 3
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
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
        """Test ProteinMPNN generator with batch_size>1 for GPU memory management."""
        num_proposals = 4
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
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


class TestProteinMPNNGeneratorValidation:
    """Unit tests for ProteinMPNN configuration and validation (no GPU required)."""

    def test_output_chain_id_selects_chain_and_preserves_generator_metadata(
        self, monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str
    ) -> None:
        """Multi-chain sampling can write one selected chain to the target segment."""
        chain_b = "\n".join(
            f"{line[:21]}B{line[22:]}" if line.startswith("ATOM") else line
            for line in sample_pdb_content.splitlines()
            if line != "END"
        )
        two_chain_pdb = sample_pdb_content.replace("END\n", "") + chain_b + "\nEND\n"

        def fake_run_proteinmpnn_sample(*, inputs, config):
            assert inputs.inputs[0].chain_ids == ["A", "B"]
            assert config.num_sequences_per_structure == 1
            designed = SimpleNamespace(sequences=["AAAAA/CCCCC"], perplexity=[2.5], sequence_recovery=[0.4])
            return SimpleNamespace(designed_sequences=[designed])

        proteinmpnn_module = import_module("proto_language.language.generator.proteinmpnn_generator")
        monkeypatch.setattr(proteinmpnn_module, "run_proteinmpnn_sample", fake_run_proteinmpnn_sample)
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(structure=two_chain_pdb, chain_ids=["A", "B"]),
                output_chain_id="B",
            )
        )
        segment = Segment(sequence="CCCCC", sequence_type="protein")
        generator.assign(segment)

        generator.sample()

        proposal = segment.proposal_sequences[0]
        assert proposal.sequence == "CCCCC"
        assert proposal._generator_metadata["proteinmpnn"] == {
            "perplexity": 2.5,
            "sequence_recovery": 0.4,
            "full_sequence": "AAAAA/CCCCC",
        }

    @pytest.mark.parametrize("sequence_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, temp_pdb_file, sequence_type):
        """ProteinMPNN should reject non-protein segments."""
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(structure_inputs=temp_pdb_file))
        segment = Segment(length=50, sequence_type=sequence_type)

        with pytest.raises(ValueError) as exc_info:
            generator.assign(segment)

        assert "does not support sequence type" in str(exc_info.value)

    def test_rejects_ligand_segment(self, temp_pdb_file):
        """ProteinMPNN should reject ligand segments (ligands cannot be mutated)."""
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(structure_inputs=temp_pdb_file))
        segment = Segment(sequence="CCC", sequence_type="ligand")

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            generator.assign(segment)

    def test_pdb_content_string(self, sample_pdb_content):
        """Should accept PDB content as a string (not just file path)."""
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(structure_inputs=sample_pdb_content))

        # Structure should be resolved automatically
        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].structure is not None

    def test_structure_without_chain_ids_defaults_to_all(self, temp_pdb_file):
        """When chain_ids is not specified, should default to all chains."""
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(structure_inputs=temp_pdb_file))

        # chain_ids should be populated with all available chains
        assert len(generator.structure_inputs) == 1
        assert generator.structure_inputs[0].chain_ids is not None
        assert generator.structure_inputs[0].chain_ids == ["A"]  # Only chain A in sample PDB

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

    def test_multiple_structure_inputs(self, sample_pdb_content):
        """Should accept multiple InverseFoldingStructureInput objects."""
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=[
                    InverseFoldingStructureInput(
                        structure=sample_pdb_content,
                        chain_ids=["A"],
                        fixed_positions={"A": [1, 2]},
                    ),
                    InverseFoldingStructureInput(
                        structure=sample_pdb_content,
                        chain_ids=["A"],
                    ),
                ]
            )
        )

        assert len(generator.structure_inputs) == 2
        assert generator.structure_inputs[0].fixed_positions == {"A": [1, 2]}
        assert generator.structure_inputs[1].fixed_positions is None


class TestProteinMPNNStructureFallback:
    """Unit tests for structure fallback in ProteinMPNN (no GPU required)."""

    def test_no_structure_anywhere_raises(self):
        """sample() raises when no structure available from any source."""
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig())
        segment = Segment(sequence="AGSVL", sequence_type="protein")
        generator.assign(segment)

        with pytest.raises(ValueError, match="No structure_inputs"):
            generator.sample()
