"""
tests/language_tests/constraint_tests/test_protein_structure/test_gyration_radius_constraint.py

Tests for Gyration Radius constraint.
"""

from unittest.mock import patch

from proto_tools import StructureMetrics, StructureMetricsOutput

from proto_language.language.constraint.protein_structure.gyration_radius_constraint import (
    GyrationRadiusConfig,
    gyration_radius_constraint,
)
from proto_language.language.core import Sequence
from proto_language.utils import MAX_ENERGY


def _make_sequence(sequence="MKTAYIAK", pdb_path="/tmp/test.pdb"):
    """Helper to create a Sequence with pdb_path metadata."""
    return Sequence(
        sequence=sequence,
        sequence_type="protein",
        metadata={"pdb_path": pdb_path} if pdb_path else None,
    )


def _mock_metrics_output(gyration_radius, longest_alpha_helix=5):
    """Helper to create a mock StructureMetricsOutput."""
    return StructureMetricsOutput(
        tool_id="structure-metrics",
        execution_time=0.0,
        success=True,
        metrics=[
            StructureMetrics(
                pdb_path="/tmp/test.pdb",
                gyration_radius=gyration_radius,
                longest_alpha_helix=longest_alpha_helix,
            )
        ],
        warnings=[],
        metadata={},
    )


PATCH_TARGET = (
    "proto_language.language.constraint.protein_structure"
    ".gyration_radius_constraint.run_structure_metrics"
)


class TestGyrationRadiusConstraint:
    """tests/language_tests/constraint_tests/test_protein_structure/test_gyration_radius_constraint.py

Tests for Gyration Radius constraint."""

    def test_within_threshold_scores_zero(self):
        """Structures within the max gyration radius score 0.0."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=50.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=30.0)
            scores = gyration_radius_constraint([(seq,)], config)

        assert len(scores) == 1
        assert scores[0] == 0.0

    def test_above_threshold_penalized(self):
        """Structures above threshold get a proportional penalty."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=40.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=60.0)
            scores = gyration_radius_constraint([(seq,)], config)

        assert len(scores) == 1
        # deviation = (60 - 40) / 40 = 0.5
        assert scores[0] == 0.5

    def test_penalty_clamped_to_one(self):
        """Penalty is clamped to 1.0 for very large radii."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=20.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=60.0)
            # deviation = (60 - 20) / 20 = 2.0, clamped to 1.0
            scores = gyration_radius_constraint([(seq,)], config)

        assert len(scores) == 1
        assert scores[0] == 1.0

    def test_missing_pdb_path_gets_max_energy(self):
        """Sequences without pdb_path metadata get MAX_ENERGY score."""
        seq = _make_sequence(pdb_path=None)
        config = GyrationRadiusConfig()

        scores = gyration_radius_constraint([(seq,)], config)

        assert len(scores) == 1
        assert scores[0] == MAX_ENERGY

    def test_explicit_pdb_paths_config(self):
        """Config-provided pdb_paths override sequence metadata."""
        seq = _make_sequence(pdb_path=None)
        config = GyrationRadiusConfig(
            max_gyration_radius=50.0,
            pdb_paths=["/tmp/explicit.pdb"],
        )

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=25.0)
            scores = gyration_radius_constraint([(seq,)], config)

        assert len(scores) == 1
        assert scores[0] == 0.0
        # Verify the explicit path was used
        call_input = mock_run.call_args[0][0]
        assert call_input.pdb_paths == ["/tmp/explicit.pdb"]

    def test_stores_metadata(self):
        """Constraint stores gyration_radius and longest_alpha_helix in metadata."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=50.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(
                gyration_radius=30.0, longest_alpha_helix=12
            )
            gyration_radius_constraint([(seq,)], config)

        assert seq._metadata["gyration_radius"] == 30.0
        assert seq._metadata["longest_alpha_helix"] == 12

    def test_multiple_sequences(self):
        """Constraint handles multiple sequences correctly."""
        seq1 = _make_sequence(pdb_path="/tmp/a.pdb")
        seq2 = _make_sequence(pdb_path="/tmp/b.pdb")
        config = GyrationRadiusConfig(max_gyration_radius=40.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = StructureMetricsOutput(
                tool_id="structure-metrics",
                execution_time=0.0,
                success=True,
                metrics=[
                    StructureMetrics(
                        pdb_path="/tmp/a.pdb",
                        gyration_radius=30.0,
                        longest_alpha_helix=5,
                    ),
                    StructureMetrics(
                        pdb_path="/tmp/b.pdb",
                        gyration_radius=50.0,
                        longest_alpha_helix=8,
                    ),
                ],
                warnings=[],
                metadata={},
            )
            scores = gyration_radius_constraint([(seq1,), (seq2,)], config)

        assert len(scores) == 2
        assert scores[0] == 0.0  # 30 <= 40
        assert scores[1] == 0.25  # (50 - 40) / 40 = 0.25
