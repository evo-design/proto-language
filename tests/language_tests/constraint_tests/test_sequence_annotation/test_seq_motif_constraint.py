"""Tests for the sequence motif constraint.

Covers config validation, wanted/unwanted motif logic, aggregation, exclusive mode,
and metadata propagation. The ``meme-fimo-scan`` tool call is mocked (no MEME install
or GPU needed); the real FIMO parity is covered in the proto-tools tool's own tests.
"""

from types import SimpleNamespace
from unittest.mock import mock_open, patch

import pytest

from proto_language.constraint import seq_motif_constraint
from proto_language.constraint.sequence_annotation.seq_motif_constraint import SeqMotifConfig
from proto_language.core import Constraint, Segment, Sequence

_MODULE = "proto_language.constraint.sequence_annotation.seq_motif_constraint"


def _scan(*per_sequence: list[tuple]) -> SimpleNamespace:
    """Fake ``run_meme_fimo_scan``: each arg is one input sequence's ``(motif_id, pvalue)`` matches."""
    results = [
        SimpleNamespace(matches=[SimpleNamespace(motif_id=mid, pvalue=pv) for mid, pv in seq_matches])
        for seq_matches in per_sequence
    ]
    return SimpleNamespace(results=results)


class TestSeqMotifConstraint:
    """Tests for the Sequence Motif constraint."""

    def test_motifs_path_is_required(self):
        """motifs_path is the only required field; fimo_config defaults are supplied."""
        with pytest.raises(Exception):  # Pydantic ValidationError — motifs_path missing
            SeqMotifConfig()
        # fimo_config defaults to FIMO's standard behavior without being passed.
        config = SeqMotifConfig(motifs_path="/path/to/motifs.meme")
        assert config.fimo_config.threshold == pytest.approx(1e-4)
        assert config.fimo_config.both_strands is True

    def test_invalid_percentile(self):
        """Out-of-range percentile values are rejected (constraint-specific validation)."""
        with pytest.raises(Exception):
            SeqMotifConfig(motifs_path="/path/to/motifs.meme", percentile_value=150.0)
        with pytest.raises(Exception):
            SeqMotifConfig(motifs_path="/path/to/motifs.meme", percentile_value=-10.0)

    def test_no_motifs_wanted_or_unwanted(self):
        """No wanted/unwanted motifs and no matches -> zero penalty."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = SeqMotifConfig(motifs_path="/mock/motifs.meme")

        with (
            patch("builtins.open", mock_open(read_data="MOTIF motif1\nMOTIF motif2")),
            patch(f"{_MODULE}.run_meme_fimo_scan", return_value=_scan([])),
        ):
            constraint = Constraint(inputs=[segment], function=seq_motif_constraint, function_config=config)
            scores = constraint.evaluate()

        assert scores == [0.0]

    def test_wanted_motif_found(self):
        """A found wanted motif surfaces its p-value in metadata and scores in range."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = SeqMotifConfig(motifs_path="/mock/motifs.meme", wanted=["motif1"], exclusive=False)

        with (
            patch("builtins.open", mock_open(read_data="MOTIF motif1")),
            patch(f"{_MODULE}.run_meme_fimo_scan", return_value=_scan([("motif1", 1e-5)])),
        ):
            constraint = Constraint(inputs=[segment], function=seq_motif_constraint, function_config=config)
            scores = constraint.evaluate()

        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0
        meta = segment.proposal_sequences[0]._constraints_metadata["seq_motif_constraint"]["data"]["motif_constraint"]
        assert meta["found"]["motif1"] == pytest.approx(1e-5)
        assert meta["details"]["motif1"]["p_value"] == pytest.approx(1e-5)
        assert meta["wanted"] == ["motif1"]
        assert isinstance(meta["not_wanted"], list)

    def test_per_sequence_results_and_lowest_pvalue(self):
        """Each input sequence gets its own result (1:1); lowest p-value per motif is kept."""
        seqs = [
            (Sequence("ATCGATCGATCG", "dna"),),
            (Sequence("GGGGCCCCATAT", "dna"),),
            (Sequence("TTTTAAAACGCG", "dna"),),
        ]
        config = SeqMotifConfig(motifs_path="/mock/motifs.meme", not_wanted=["m0", "m1", "m2"], exclusive=False)
        scan = _scan(
            [("m0", 1e-6)],  # sequence 0
            [("m1", 1e-3), ("m1", 1e-8)],  # sequence 1: lowest p-value wins
            [],  # sequence 2: no matches
        )

        with (
            patch("builtins.open", mock_open(read_data="MOTIF m0\nMOTIF m1\nMOTIF m2")),
            patch(f"{_MODULE}.run_meme_fimo_scan", return_value=scan),
        ):
            results = seq_motif_constraint(seqs, config)

        assert len(results) == 3
        found = [r.metadata["motif_constraint"]["found"] for r in results]
        assert found[0] == {"m0": pytest.approx(1e-6)}
        assert found[1] == {"m1": pytest.approx(1e-8)}
        assert found[2] == {}

    def test_constraint_specific_config_options(self):
        """Constraint-specific config knobs validate and store correctly."""
        assert SeqMotifConfig(motifs_path="/mock/motifs.meme", wanted="all").wanted == ["all"]
        assert SeqMotifConfig(motifs_path="/mock/motifs.meme", wanted="none").wanted == ["none"]
        assert SeqMotifConfig(motifs_path="/mock/motifs.meme", wanted=["motif1"], exclusive=True).exclusive
        for agg in ["smart", "average", "max", "percentile"]:
            assert SeqMotifConfig(motifs_path="/mock/motifs.meme", aggregation=agg).aggregation == agg
