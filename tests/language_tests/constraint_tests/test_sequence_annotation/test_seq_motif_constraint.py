"""Tests cover:.

1. Configuration validation
2. Wanted/unwanted motif logic
3. Aggregation strategies
4. Exclusive mode
5. Registry integration
6. Metadata propagation

Note: Actual MEME/FIMO execution is mocked to avoid dependencies.
"""

from unittest.mock import Mock, mock_open, patch

import pytest

from proto_language.constraint import seq_motif_constraint
from proto_language.constraint.sequence_annotation.seq_motif_constraint import (
    SeqMotifConfig,
)
from proto_language.core import Constraint, Segment


class TestSeqMotifConstraint:
    """Tests for Sequence Motif constraint."""

    def test_config_required_fields(self):
        """Test that required config fields must be provided (constraint-specific validation)."""
        # motifs_path is required
        with pytest.raises(Exception):  # Pydantic ValidationError
            SeqMotifConfig(meme_bin_path="/usr/bin")

        # meme_bin_path is required
        with pytest.raises(Exception):  # Pydantic ValidationError
            SeqMotifConfig(motifs_path="/path/to/motifs.meme")

    def test_invalid_percentile(self):
        """Test that invalid percentile values raise errors (constraint-specific validation)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            SeqMotifConfig(
                motifs_path="/path/to/motifs.meme",
                meme_bin_path="/usr/bin",
                percentile_value=150.0,  # > 100
            )

        with pytest.raises(Exception):  # Pydantic ValidationError
            SeqMotifConfig(
                motifs_path="/path/to/motifs.meme",
                meme_bin_path="/usr/bin",
                percentile_value=-10.0,  # < 0
            )

    def test_no_motifs_wanted_or_unwanted(self):
        """Test scoring when no wanted/unwanted motifs specified."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = SeqMotifConfig(
            motifs_path="/mock/motifs.meme",
            meme_bin_path="/usr/bin",
        )

        # Mock motif file reading
        motif_file_content = "MOTIF motif1\nMOTIF motif2"

        with (
            patch("builtins.open", mock_open(read_data=motif_file_content)),
            patch("proto_language.constraint.sequence_annotation.seq_motif_constraint.subprocess.run") as _,
            patch(
                "proto_language.constraint.sequence_annotation.seq_motif_constraint.tempfile.TemporaryDirectory"
            ) as mock_temp,
        ):
            # Setup mock temp directory
            mock_temp_dir = "/mock/test_temp"
            mock_temp_inst = Mock()
            mock_temp_inst.__enter__ = Mock(return_value=mock_temp_dir)
            mock_temp_inst.__exit__ = Mock(return_value=False)
            mock_temp.return_value = mock_temp_inst

            # Mock FIMO output (no hits)
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = False

                constraint = Constraint(
                    inputs=[segment],
                    function=seq_motif_constraint,
                    function_config=config,
                )

                scores = constraint.evaluate()
                assert len(scores) == 1
                assert scores[0] == 0.0  # No wanted/unwanted -> penalty = 0.0

    def test_wanted_motif_found(self):
        """Wanted motif found: p-value parsed from the real FIMO column order surfaces as ``p_value`` (M9)."""
        segment = Segment(sequence="ATCGATCGATCG", sequence_type="dna")
        config = SeqMotifConfig(
            motifs_path="/mock/motifs.meme",
            meme_bin_path="/usr/bin",
            wanted=["motif1"],
            exclusive=False,
        )

        # Mock motif file and FIMO results, in the real FIMO column order:
        # motif_id, motif_alt_id, sequence_name, start, stop, strand, score, p-value, q-value, matched_sequence
        motif_file_content = "MOTIF motif1"
        fimo_results = (
            "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence\n"
        )
        fimo_results += "motif1\t\tquery\t1\t10\t+\t10.0\t1e-5\t0.001\tATCGATCGAT\n"

        with (
            patch("builtins.open", mock_open(read_data=motif_file_content)),
            patch("proto_language.constraint.sequence_annotation.seq_motif_constraint.subprocess.run") as _,
            patch(
                "proto_language.constraint.sequence_annotation.seq_motif_constraint.tempfile.TemporaryDirectory"
            ) as mock_temp,
        ):
            mock_temp_dir = "/mock/test_temp"
            mock_temp_inst = Mock()
            mock_temp_inst.__enter__ = Mock(return_value=mock_temp_dir)
            mock_temp_inst.__exit__ = Mock(return_value=False)
            mock_temp.return_value = mock_temp_inst

            # Mock FIMO output file
            with patch("os.path.exists") as mock_exists, patch("builtins.open", mock_open(read_data=fimo_results)):
                mock_exists.return_value = True

                constraint = Constraint(
                    inputs=[segment],
                    function=seq_motif_constraint,
                    function_config=config,
                )

                scores = constraint.evaluate()
                assert len(scores) == 1
                # Wanted motif found with good p-value -> low penalty.
                assert 0.0 <= scores[0] <= 1.0

                meta = segment.proposal_sequences[0]._constraints_metadata["seq_motif_constraint"]["data"][
                    "motif_constraint"
                ]
                # Parser must read the p-value column (1e-5), not the score column (10.0).
                assert meta["found"]["motif1"] == pytest.approx(1e-5)
                assert meta["details"]["motif1"]["p_value"] == pytest.approx(1e-5)

    def test_constraint_specific_config_options(self):
        """Test constraint-specific config options (wanted, exclusive, aggregation)."""
        # Test 'all' keyword for wanted motifs
        config_all = SeqMotifConfig(
            motifs_path="/mock/motifs.meme",
            meme_bin_path="/usr/bin",
            wanted="all",
        )
        assert config_all.wanted == ["all"]

        # Test 'none' keyword for wanted motifs
        config_none = SeqMotifConfig(
            motifs_path="/mock/motifs.meme",
            meme_bin_path="/usr/bin",
            wanted="none",
        )
        assert config_none.wanted == ["none"]

        # Test exclusive mode
        config_exclusive = SeqMotifConfig(
            motifs_path="/mock/motifs.meme",
            meme_bin_path="/usr/bin",
            wanted=["motif1"],
            exclusive=True,
        )
        assert config_exclusive.exclusive

        # Test different aggregation strategies
        for agg in ["smart", "average", "max", "percentile"]:
            config = SeqMotifConfig(
                motifs_path="/mock/motifs.meme",
                meme_bin_path="/usr/bin",
                aggregation=agg,
            )
            assert config.aggregation == agg
