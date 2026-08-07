"""Tests for the CodonFM (Encodon) coding-sequence mutation generator."""

import copy
from types import SimpleNamespace

import pytest

from proto_language.core import GeneratorInputType, Segment
from proto_tools.transforms.masking import RandomMaskingStrategy

from proto_language.generator.codonfm_generator import CodonFMGenerator, CodonFMGeneratorConfig
from proto_language.generator.generator_registry import GeneratorRegistry

_MODULE = "proto_language.generator.codonfm_generator"
_CDS = "ATGGTGAGCAAGGGC"  # 15 nt, 5 codons


class TestSample:
    def test_sample_writes_resampled_sequences_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def fake_run(inputs, config):
            captured["inputs"] = inputs
            captured["config"] = config
            # Echo a deterministic single-codon edit per proposal (length preserved).
            return SimpleNamespace(sequences=["AAA" + s[3:] for s in inputs.sequences])

        monkeypatch.setattr(_MODULE + ".run_codonfm_sample", fake_run)

        gen = CodonFMGenerator(CodonFMGeneratorConfig(masking_strategy=RandomMaskingStrategy(num_mutations=2), temperature=1.1, device="cpu"))
        segment = Segment(sequence=_CDS, sequence_type="dna")
        gen.assign(segment)
        gen.sample()

        assert segment.proposal_sequences[0].sequence == "AAA" + _CDS[3:]
        assert len(segment.proposal_sequences[0].sequence) == len(_CDS)
        # Config forwarded to the tool.
        assert captured["config"].masking_strategy.num_mutations == 2
        assert captured["config"].temperature == 1.1
        assert captured["config"].model_checkpoint == "encodon_80m"
        assert captured["inputs"].sequences == [_CDS]

    def test_batch_of_proposals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _MODULE + ".run_codonfm_sample",
            lambda inputs, config: SimpleNamespace(sequences=list(inputs.sequences)),
        )
        gen = CodonFMGenerator(CodonFMGeneratorConfig(masking_strategy=RandomMaskingStrategy(mask_fraction=0.2), device="cpu"))
        segment = Segment(sequence=_CDS, sequence_type="dna")
        gen.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(4)]
        gen.sample()
        assert len(segment.proposal_sequences) == 4
        for proposal in segment.proposal_sequences:
            assert len(proposal.sequence) == len(_CDS)
            assert proposal.sequence_type == "dna"

    def test_seed_is_forwarded_and_advances(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seeds: list = []
        monkeypatch.setattr(
            _MODULE + ".run_codonfm_sample",
            lambda inputs, config: seeds.append(config.seed) or SimpleNamespace(sequences=list(inputs.sequences)),
        )
        gen = CodonFMGenerator(CodonFMGeneratorConfig(device="cpu"))
        segment = Segment(sequence=_CDS, sequence_type="dna")
        gen.assign(segment)
        gen.sample()
        gen.sample()
        assert len(seeds) == 2  # a seed was forwarded on each sample


class TestConfigAndRegistry:
    def test_num_mutations_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError):
            CodonFMGeneratorConfig(masking_strategy=RandomMaskingStrategy(num_mutations=0))

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            CodonFMGeneratorConfig(temperature=0.0)

    def test_registration(self) -> None:
        spec = GeneratorRegistry.get("codonfm")
        assert spec.category == "mutation"
        assert spec.tools_called == ["codonfm-sample"]
        assert spec.supported_sequence_types == ["dna", "rna"]
        assert CodonFMGenerator.input_type == GeneratorInputType.STARTING_SEQUENCE
