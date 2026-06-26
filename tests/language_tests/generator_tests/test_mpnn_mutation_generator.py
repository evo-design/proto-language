"""Tests for the MPNN mutation generator."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from proto_tools import InverseFoldingStructureInput

from proto_language.core import Segment
from proto_language.core.sequence import PROTEIN_AMINO_ACIDS
from proto_language.generator import MPNNMutationGenerator, MPNNMutationGeneratorConfig


class FakeScore(SimpleNamespace):
    """Minimal score object matching proto-tools Metrics access used by the generator."""

    def model_dump(self, exclude=None):
        return {"perplexity": self.perplexity}


def test_ligandmpnn_mutation_uses_mutable_positions_and_logits(monkeypatch: pytest.MonkeyPatch, temp_pdb_file):
    """A single mutable position is selected and replaced from MPNN logits."""
    vocab = list(PROTEIN_AMINO_ACIDS)
    logits = np.zeros((5, len(vocab)))
    logits[1, vocab.index("G")] = 1.0
    logits[1, vocab.index("W")] = 5.0

    def fake_run_ligandmpnn_score(*, inputs, config):
        pair = inputs.sequence_structure_pairs[0]
        assert pair.sequence == "AGSVL"
        assert pair.fixed_positions.chains == {"A": [1, 3, 4, 5]}
        assert config.return_logits is True
        assert config.scoring_mode == "single_aa"
        return SimpleNamespace(scores=[FakeScore(logits=logits, vocab=vocab, perplexity=2.0)])

    module = import_module("proto_language.generator.mpnn_mutation_generator")
    monkeypatch.setattr(module, "run_ligandmpnn_score", fake_run_ligandmpnn_score)

    generator = MPNNMutationGenerator(
        MPNNMutationGeneratorConfig(
            model="ligandmpnn",
            structure_inputs=InverseFoldingStructureInput(structure=temp_pdb_file, chains_to_redesign=["A"]),
            output_chain_id="A",
            mutable_positions={"A": [2]},
            num_mutations=1,
            replacement_strategy="argmax",
            device="cpu",
        )
    )
    segment = Segment(sequence="AGSVL", sequence_type="protein")
    generator.assign(segment)

    generator.sample()

    proposal = segment.proposal_sequences[0]
    assert proposal.sequence == "AWSVL"
    assert proposal._generator_metadata["mpnn-mutation"]["mutations"] == [{"position": 2, "from": "G", "to": "W"}]


def test_proteinmpnn_model_dispatches_to_proteinmpnn_score(monkeypatch: pytest.MonkeyPatch, temp_pdb_file):
    """The model switch selects ProteinMPNN scoring config and run function."""
    vocab = list(PROTEIN_AMINO_ACIDS)
    logits = np.zeros((5, len(vocab)))
    logits[0, vocab.index("A")] = 1.0
    logits[0, vocab.index("Y")] = 5.0

    def fail_ligandmpnn_score(*args, **kwargs):
        raise AssertionError("ligandmpnn score should not be called")

    def fake_run_proteinmpnn_score(*, inputs, config):
        assert inputs.sequence_structure_pairs[0].sequence == "AGSVL"
        assert config.model_choice == "soluble"
        assert config.return_logits is True
        return SimpleNamespace(scores=[FakeScore(logits=logits, vocab=vocab, perplexity=1.5)])

    module = import_module("proto_language.generator.mpnn_mutation_generator")
    monkeypatch.setattr(module, "run_ligandmpnn_score", fail_ligandmpnn_score)
    monkeypatch.setattr(module, "run_proteinmpnn_score", fake_run_proteinmpnn_score)

    generator = MPNNMutationGenerator(
        MPNNMutationGeneratorConfig(
            model="proteinmpnn",
            proteinmpnn_model_choice="soluble",
            structure_inputs=temp_pdb_file,
            output_chain_id="A",
            mutable_positions={"A": [1]},
            num_mutations=1,
            replacement_strategy="argmax",
            device="cpu",
        )
    )
    segment = Segment(sequence="AGSVL", sequence_type="protein")
    generator.assign(segment)

    generator.sample()

    assert segment.proposal_sequences[0].sequence == "YGSVL"


def test_requires_mutable_candidates_after_fixed_positions(monkeypatch: pytest.MonkeyPatch, temp_pdb_file):
    """Fixed positions remove candidates even when mutable_positions includes them."""
    vocab = list(PROTEIN_AMINO_ACIDS)

    def fake_run_ligandmpnn_score(*, inputs, config):
        return SimpleNamespace(scores=[FakeScore(logits=np.zeros((5, len(vocab))), vocab=vocab, perplexity=2.0)])

    module = import_module("proto_language.generator.mpnn_mutation_generator")
    monkeypatch.setattr(module, "run_ligandmpnn_score", fake_run_ligandmpnn_score)

    generator = MPNNMutationGenerator(
        MPNNMutationGeneratorConfig(
            structure_inputs=InverseFoldingStructureInput(
                structure=temp_pdb_file,
                chains_to_redesign=["A"],
                fixed_positions={"A": [2]},
            ),
            output_chain_id="A",
            mutable_positions={"A": [2]},
            num_mutations=1,
            device="cpu",
        )
    )
    segment = Segment(sequence="AGSVL", sequence_type="protein")
    generator.assign(segment)

    with pytest.raises(ValueError, match="only 0 mutable positions"):
        generator.sample()


def test_crossover_positions_match_mutable_positions(temp_pdb_file):
    """GA crossover can reuse the MPNN mutation generator's mutable residue scope."""
    generator = MPNNMutationGenerator(
        MPNNMutationGeneratorConfig(
            structure_inputs=InverseFoldingStructureInput(
                structure=temp_pdb_file,
                chains_to_redesign=["A"],
                fixed_positions={"A": [4]},
            ),
            output_chain_id="A",
            mutable_positions={"A": [2, 4]},
            num_mutations=1,
            device="cpu",
        )
    )
    segment = Segment(sequence="AGSVL", sequence_type="protein")
    generator.assign(segment)

    assert generator.crossover_position_indices(segment) == {1}


def test_scoring_pdb_sanitizer_selects_single_altloc_and_preserves_ligand_context():
    pdb = """\
ATOM      1  N   SER A   1       0.000   0.000   0.000  0.50  0.00           N
ATOM      2  CA  SER A   1       1.000   0.000   0.000  0.50  0.00           C
ATOM      3  C   SER A   1       1.000   1.000   0.000  0.50  0.00           C
ATOM      4  O   SER A   1       1.000   1.000   1.000  0.50  0.00           O
ATOM      5  CB ASER A   1       1.000  -1.000   0.000  0.50  0.00           C
ATOM      6  CB BSER A   1       1.100  -1.100   0.000  0.50  0.00           C
HETATM    7 ZN    ZN A 100       2.000   2.000   2.000  0.50  0.00          ZN
END
"""

    sanitized = MPNNMutationGenerator._sanitize_pdb_for_scoring(pdb, {"A"})
    lines = sanitized.splitlines()

    assert all("BSER" not in line for line in lines)
    assert all(line[16] == " " for line in lines if line.startswith("ATOM"))
    assert all(line[54:60] == "  1.00" for line in lines if line.startswith(("ATOM", "HETATM")))
    assert [line[21] for line in lines if line.startswith("HETATM")] == ["B"]
