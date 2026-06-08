"""tests/language_tests/generator_tests/test_rfdiffusion_proteinmpnn_binder_generator.py."""

import copy
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
from proto_tools import ProteinMPNNSampleConfig, Structure

from proto_language.core import Segment
from proto_language.generator import (
    RFdiffusionProteinMPNNBinderGenerator,
    RFdiffusionProteinMPNNBinderGeneratorConfig,
)

KEY = "rfdiffusion-proteinmpnn-binder"
_BINDER_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _relabel_chain(pdb: str, chain_id: str) -> str:
    """Return the ATOM records of ``pdb`` with their chain column set to ``chain_id``."""
    return "\n".join(
        (line[:21] + chain_id + line[22:]) if line.startswith("ATOM") else line
        for line in pdb.splitlines()
        if line not in ("END", "")
    )


def _multi_chain_pdb(base_pdb: str, chain_ids: list[str]) -> str:
    """Build a PDB whose chains are copies of ``base_pdb`` relabeled to ``chain_ids``."""
    return "\n".join(_relabel_chain(base_pdb, cid) for cid in chain_ids) + "\nEND\n"


def _make_rfd_mock(
    captured: dict[str, Any], *, output_chain_ids: list[str], base_pdb: str, n_backbones: int | None = None
):
    """Fake ``run_rfdiffusion3`` returning one bundle of multi-chain backbones."""

    def fake_rfd(*, inputs, config):
        captured["rfd_inputs"] = inputs
        captured["rfd_config"] = config
        count = n_backbones if n_backbones is not None else config.n_batches * config.diffusion_batch_size
        backbones = [
            SimpleNamespace(structure=Structure(structure=_multi_chain_pdb(base_pdb, output_chain_ids)))
            for _ in range(count)
        ]
        return SimpleNamespace(designed_structures=[backbones])

    return fake_rfd


def _make_mpnn_mock(captured: dict[str, Any]):
    """Fake ``run_proteinmpnn_sample`` designing a distinct binder per backbone/sequence."""

    def fake_mpnn(*, inputs, config):
        captured["mpnn_inputs"] = inputs
        captured["mpnn_config"] = config
        counter = 0
        design_sets = []
        for struct_input in inputs.inputs:
            chain_ids = struct_input.structure.get_chain_ids()
            binder_id = struct_input.chain_ids_to_redesign[0]
            complexes = []
            for _ in range(config.num_sequences_per_structure):
                binder_seq = _BINDER_AAS[counter % len(_BINDER_AAS)] * 5
                counter += 1
                chains = [
                    SimpleNamespace(id=cid, sequence=binder_seq if cid == binder_id else "AGSVL") for cid in chain_ids
                ]
                complexes.append(
                    SimpleNamespace(
                        chains=chains,
                        designed=[cid == binder_id for cid in chain_ids],
                        metrics=SimpleNamespace(perplexity=2.5, sequence_recovery=0.4),
                    )
                )
            design_sets.append(SimpleNamespace(complexes=complexes))
        return SimpleNamespace(design_sets=design_sets)

    return fake_mpnn


def _patch_tools(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    *,
    output_chain_ids: list[str],
    base_pdb: str,
    n_backbones: int | None = None,
) -> None:
    module = import_module("proto_language.generator.rfdiffusion_proteinmpnn_binder_generator")
    monkeypatch.setattr(
        module,
        "run_rfdiffusion3",
        _make_rfd_mock(captured, output_chain_ids=output_chain_ids, base_pdb=base_pdb, n_backbones=n_backbones),
    )
    monkeypatch.setattr(module, "run_proteinmpnn_sample", _make_mpnn_mock(captured))


def _binder_segment(length: int = 5, num_proposals: int = 1) -> Segment:
    """Length-only protein segment with the requested number of empty proposals."""
    segment = Segment(length=length, sequence_type="protein")
    if num_proposals > 1:
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_proposals)]
    return segment


class TestRFdiffusionProteinMPNNBinderGeneratorConfig:
    """Config-level construction and validation (no tool calls)."""

    def test_init_stores_config(self, sample_pdb_content: str) -> None:
        config = RFdiffusionProteinMPNNBinderGeneratorConfig(
            target_structure=sample_pdb_content,
            target_chains=["A"],
            hotspots=["A3"],
            proteinmpnn_config=ProteinMPNNSampleConfig(temperature=0.2, num_sequences_per_structure=2),
        )
        gen = RFdiffusionProteinMPNNBinderGenerator(config)
        assert gen.target_chains == ["A"]
        assert gen.hotspots == ["A3"]
        assert gen.proteinmpnn_config.temperature == 0.2
        assert gen.proteinmpnn_config.num_sequences_per_structure == 2

    def test_bad_hotspot_format_rejected(self, sample_pdb_content: str) -> None:
        with pytest.raises(ValueError, match="Hotspots must be"):
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, hotspots=["3"])

    def test_hotspot_chain_not_in_target_rejected(self, sample_pdb_content: str) -> None:
        with pytest.raises(ValueError, match="not in target_chains"):
            RFdiffusionProteinMPNNBinderGeneratorConfig(
                target_structure=sample_pdb_content, target_chains=["A"], hotspots=["B3"]
            )

    def test_contig_override_not_accepted(self, sample_pdb_content: str) -> None:
        # The contig is always auto-built (binder last) so the binder is reliably the last
        # output chain; an explicit override could put the binder first and silently redesign
        # the target, so it is rejected.
        with pytest.raises(ValueError, match="contig"):
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, contig="50-120,/0,A17-131")


class TestRFdiffusionProteinMPNNBinderGeneratorAssign:
    """Segment-assignment validation."""

    @pytest.mark.parametrize("sequence_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, sample_pdb_content: str, sequence_type: str) -> None:
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        with pytest.raises(ValueError, match="does not support sequence type"):
            gen.assign(Segment(length=50, sequence_type=sequence_type))

    def test_rejects_ligand_segment(self, sample_pdb_content: str) -> None:
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            gen.assign(Segment(sequence="CCC", sequence_type="ligand"))


class TestRFdiffusionProteinMPNNBinderGeneratorSample:
    """End-to-end sampling behavior with both tools mocked."""

    def test_contig_auto_build_single_target(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, target_chains=["A"])
        )
        gen.assign(_binder_segment(length=5))
        gen.sample()
        assert captured["rfd_inputs"].design_specs[0].contig == "A1-5,/0,5"

    def test_contig_auto_build_multi_target(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        target_pdb = _multi_chain_pdb(sample_pdb_content, ["A", "B"])
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B", "C"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=target_pdb, target_chains=["A", "B"])
        )
        gen.assign(_binder_segment(length=5))
        gen.sample()
        assert captured["rfd_inputs"].design_specs[0].contig == "A1-5,/0,B1-5,/0,5"
        assert captured["mpnn_inputs"].inputs[0].chain_ids_to_redesign == ["C"]

    def test_binder_chain_is_last_and_only_redesigned(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        assert captured["mpnn_inputs"].inputs[0].chain_ids_to_redesign == ["B"]
        assert segment.proposal_sequences[0].sequence != "AGSVL"
        assert len(segment.proposal_sequences[0].sequence) == 5

    def test_hotspots_forwarded_to_rfdiffusion(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(
                target_structure=sample_pdb_content, hotspots=["A3", "A5"], infer_ori_strategy="hotspots"
            )
        )
        gen.assign(_binder_segment(length=5))
        gen.sample()
        spec = captured["rfd_inputs"].design_specs[0]
        assert spec.select_hotspots == "A3,A5"
        assert spec.infer_ori_strategy == "hotspots"

    def test_num_proposals_drives_backbone_count(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        segment = _binder_segment(length=5, num_proposals=3)
        gen.assign(segment)
        gen.sample()
        # 3 proposals / 1 seq-per-backbone -> 3 backbones designed by ProteinMPNN.
        assert len(captured["mpnn_inputs"].inputs) == 3
        assert captured["mpnn_config"].num_sequences_per_structure == 1
        assert len({seq.sequence for seq in segment.proposal_sequences}) == 3

    def test_sequences_per_backbone_truncates_to_num_proposals(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(
                target_structure=sample_pdb_content,
                proteinmpnn_config=ProteinMPNNSampleConfig(num_sequences_per_structure=2),
            )
        )
        segment = _binder_segment(length=5, num_proposals=3)
        gen.assign(segment)
        gen.sample()
        # ceil(3 / 2) = 2 backbones x 2 seqs = 4 designs, truncated to the 3 proposals.
        assert len(captured["mpnn_inputs"].inputs) == 2
        assert captured["mpnn_config"].num_sequences_per_structure == 2
        assert all(len(seq.sequence) == 5 for seq in segment.proposal_sequences)
        assert segment.num_proposals == 3

    def test_structure_preserved_after_sample(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        proposal = segment.proposal_sequences[0]
        assert proposal.structure is not None
        assert proposal.structure.get_chain_ids() == ["A", "B"]

    def test_metadata_written(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        metadata = segment.proposal_sequences[0]._generator_metadata[KEY]
        assert metadata["perplexity"] == 2.5
        assert metadata["sequence_recovery"] == 0.4
        assert metadata["contig"] == "A1-5,/0,5"
        assert metadata["full_complex_sequence"] == f"AGSVL/{segment.proposal_sequences[0].sequence}"

    def test_de_novo_empty_segment_runs(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        segment = _binder_segment(length=5)
        assert not segment.proposals_populated
        gen.assign(segment)
        gen.sample()
        sequence = segment.proposal_sequences[0].sequence
        assert len(sequence) == 5
        assert sequence != "XXXXX"

    def test_no_backbones_raises(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content, n_backbones=0)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        gen.assign(_binder_segment(length=5))
        with pytest.raises(RuntimeError, match="no binder backbones"):
            gen.sample()

    def test_fewer_designs_than_requested_raises(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content, n_backbones=1)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        gen.assign(_binder_segment(length=5, num_proposals=3))
        with pytest.raises(RuntimeError, match="fewer than"):
            gen.sample()

    def test_seed_advances_across_tools(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionProteinMPNNBinderGenerator(
            RFdiffusionProteinMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        )
        gen._set_program_seed(123)
        gen.assign(_binder_segment(length=5))
        gen.sample()
        rfd_seed = captured["rfd_config"].seed
        mpnn_seed = captured["mpnn_config"].seed
        assert isinstance(rfd_seed, int)
        assert isinstance(mpnn_seed, int)
        assert rfd_seed != mpnn_seed
