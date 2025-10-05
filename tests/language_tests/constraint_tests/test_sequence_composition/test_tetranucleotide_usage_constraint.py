import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import tetranucleotide_usage_constraint, ConstraintRegistry
from proto_language.language.constraint.sequence_composition.tetranucleotide_usage_constraint import TetranucleotideUsageConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for tetranucleotide_usage_constraint
class TestTetranucleotideUsageConstraint:
    def test_tud_scoring(self):
        tetranuc = "GATC"
        tud_range = (0.8, 1.2)
        # From old tests: seq with one GATC, TUD is ~3.16, outside range.
        seq_balanced = create_segment("AGCT" * 10 + "GATC" + "AGCT" * 10)
        seq_no_gatc = create_segment("A" * 25)  # TUD is 0, outside range.

        config = TetranucleotideUsageConfig(
            tetranucleotide=tetranuc,
            min_tud=tud_range[0],
            max_tud=tud_range[1],
        )
        constraint_bal = Constraint(
            inputs=[seq_balanced],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config=config,
        )
        # TUD is high, deviation is (3.16-1.2)/1.2 -> capped at 1.0
        assert abs(constraint_bal.evaluate()[0] - 1.0) < 1e-9
        assert (
            "segment_0.tetranucleotide_usage_constraint.GATC_tud"
            in seq_balanced[0]._metadata
        )
        assert (
            seq_balanced[0]._metadata[
                "segment_0.tetranucleotide_usage_constraint.GATC_tud"
            ]
            > 3.0
        )

        constraint_no_gatc = Constraint(
            inputs=[seq_no_gatc],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config=config,
        )
        # TUD is 0, deviation is (0.8-0)/0.8 = 1.0
        assert abs(constraint_no_gatc.evaluate()[0] - 1.0) < 1e-9
        assert (
            seq_no_gatc[0]._metadata[
                "segment_0.tetranucleotide_usage_constraint.GATC_tud"
            ]
            == 0.0
        )

    def test_edge_cases(self):
        """Test constraint-specific edge cases (short and empty sequences)."""
        # Sequence too short
        seq_short = create_segment("GAT")
        config = TetranucleotideUsageConfig(
            tetranucleotide="GATC",
            min_tud=0.8,
            max_tud=1.2,
        )
        constraint_short = Constraint(
            inputs=[seq_short],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config=config,
        )
        assert constraint_short.evaluate()[0] == 0.0
        assert (
            seq_short[0]._metadata[
                "segment_0.tetranucleotide_usage_constraint.GATC_tud"
            ]
            == 0.0
        )

        # Empty sequence
        seq_empty = create_segment("")
        constraint_empty = Constraint(
            inputs=[seq_empty],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config=config,
        )
        assert constraint_empty.evaluate()[0] == 0.0

    def test_all_same_tetranucleotide(self):
        """Tests when the sequence is composed of the target tetranucleotide (constraint-specific scoring)."""
        # TUD for AAAA in AAAAAAAAAAAAAAAA should be 1.0
        seq_all_a = create_segment("A" * 16)
        config = TetranucleotideUsageConfig(
            tetranucleotide="AAAA",
            min_tud=0.8,
            max_tud=1.2,
        )
        constraint = Constraint(
            inputs=[seq_all_a],
            scoring_function=tetranucleotide_usage_constraint,
            scoring_function_config=config,
        )
        assert constraint.evaluate()[0] == 0.0
        # Check constraint-specific metadata
        assert (
            abs(
                seq_all_a[0]._metadata[
                    "segment_0.tetranucleotide_usage_constraint.AAAA_tud"
                ]
                - 1.0
            )
            < 1e-9
        )