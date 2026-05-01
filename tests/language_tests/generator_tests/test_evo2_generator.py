"""Tests for Evo2Generator."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from proto_tools import Evo2KVCacheRef

from proto_language.language.core import Segment
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig


def _segment_with_proposals(length: int, count: int) -> Segment:
    segment = Segment(length=length, sequence_type="dna")
    segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(count)]
    return segment


class TestEvo2Generator:
    @patch("proto_language.language.generator.evo2_generator.run_evo2_sample")
    def test_sample_dispatches_batched_prompts(self, mock_run):
        cache_refs = [
            {"type": "evo2_kv_cache", "cache_id": "cache-a"},
            {"type": "evo2_kv_cache", "cache_id": "cache-b"},
        ]
        mock_run.return_value = SimpleNamespace(
            sequences=["ATCGAAAA", "GGCCTTTT"],
            kv_caches=cache_refs,
        )
        generator = Evo2Generator(
            Evo2GeneratorConfig(
                prompts=["ATCG", "GGCC"],
                prepend_prompt=True,
                store_kv_cache=True,
                batch_size=2,
                device="cuda:1",
                top_k=8,
                top_p=0.9,
                temperature=0.7,
                stop_at_eos=False,
            )
        )
        segment = _segment_with_proposals(length=8, count=2)
        generator.assign(segment)

        generator.sample()

        call = mock_run.call_args.kwargs
        assert call["inputs"].prompts == ["ATCG", "GGCC"]
        config = call["config"]
        assert config.num_tokens == 4
        assert config.prepend_prompt is True
        assert config.return_kv_cache is True
        assert config.batch_size == 2
        assert config.device == "cuda:1"
        assert config.top_k == 8
        assert config.top_p == 0.9
        assert config.temperature == 0.7
        assert config.stop_at_eos is False
        assert config.old_kv_cache is None
        assert [seq.sequence for seq in segment.proposal_sequences] == ["ATCGAAAA", "GGCCTTTT"]
        assert generator.kv_caches == cache_refs

    @patch("proto_language.language.generator.evo2_generator.run_evo2_sample")
    def test_single_prompt_replicates_across_proposals(self, mock_run):
        cache_ref = Evo2KVCacheRef(cache_id="prefix")
        mock_run.return_value = SimpleNamespace(sequences=["AA", "CC", "GG"], kv_caches=None)
        generator = Evo2Generator(Evo2GeneratorConfig(prompts="ATCG", prepend_prompt=False))
        segment = _segment_with_proposals(length=6, count=3)
        generator.assign(segment)

        generator.sample(prepend_prompt=False, num_tokens=2, old_kv_cache=cache_ref)

        call = mock_run.call_args.kwargs
        assert call["inputs"].prompts == ["ATCG", "ATCG", "ATCG"]
        assert call["config"].num_tokens == 2
        assert call["config"].prepend_prompt is False
        assert call["config"].old_kv_cache == cache_ref
        assert call["config"].return_kv_cache is False
        assert [seq.sequence for seq in segment.proposal_sequences] == ["AA", "CC", "GG"]
        assert generator.kv_caches == []

    @patch("proto_language.language.generator.evo2_generator.run_evo2_sample")
    def test_prompt_count_must_match_proposal_count(self, mock_run):
        generator = Evo2Generator(Evo2GeneratorConfig(prompts=["ATCG", "GGCC", "TTAA"]))
        segment = _segment_with_proposals(length=8, count=2)
        generator.assign(segment)

        with pytest.raises(ValueError, match="Expected 1 or 2 prompts"):
            generator.sample()

        mock_run.assert_not_called()

    @patch("proto_language.language.generator.evo2_generator.run_evo2_sample")
    def test_num_tokens_uses_prepend_prompt_override(self, mock_run):
        mock_run.return_value = SimpleNamespace(sequences=["A" * 100], kv_caches=[])
        generator = Evo2Generator(Evo2GeneratorConfig(prompts="ATCG", prepend_prompt=False))
        segment = _segment_with_proposals(length=100, count=1)
        generator.assign(segment)

        generator.sample(prepend_prompt=True)

        assert mock_run.call_args.kwargs["config"].num_tokens == 96

    def test_valid_dna_assignment(self):
        generator = Evo2Generator(Evo2GeneratorConfig(prompts="ATGC"))
        segment = Segment(length=100, sequence_type="dna")

        generator.assign(segment)

        assert generator._assigned_segment is segment

    @pytest.mark.parametrize("seq_type", ["protein", "rna"])
    def test_rejects_non_dna_segment(self, seq_type):
        generator = Evo2Generator(Evo2GeneratorConfig(prompts="ATGC"))
        segment = Segment(length=100, sequence_type=seq_type)

        with pytest.raises(ValueError, match="does not support sequence type"):
            generator.assign(segment)

    @patch("proto_language.language.generator.evo2_generator.release_evo2_kv_caches")
    def test_release_kv_cache_delegates_to_proto_tools(self, mock_release):
        generator = Evo2Generator(Evo2GeneratorConfig(prompts="ATGC"))
        cache_ref = {"type": "evo2_kv_cache", "cache_id": "cache-a"}

        generator.release_kv_cache(cache_ref)

        mock_release.assert_called_once_with(cache_ref)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_evo2_generator_gpu_smoke():
    generator = Evo2Generator(Evo2GeneratorConfig(prompts="ATCG", prepend_prompt=True, stop_at_eos=False))
    segment = Segment(length=12, sequence_type="dna")
    generator.assign(segment)

    generator.sample()

    assert segment.proposal_sequences[0].sequence is not None
    assert len(segment.proposal_sequences[0].sequence) == 12
