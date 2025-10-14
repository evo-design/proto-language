import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
)
from proto_language.language.constraint import dinucleotide_frequency_constraint, ConstraintRegistry
from proto_language.language.constraint.sequence_composition.dinucleotide_frequency_constraint import DinucleotideFrequencyConfig
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for dinucleotide_frequency_constraint
class TestDinucleotideFrequencyConstraint:
    def test_dna_sequences(self):
        # Sequence "ATCGATCG" has freqs: AT=0.286, TC=0.286, CG=0.286, GA=0.143
        # But also has 0.0 for all other dinucleotides (AA, TT, CC, GG, etc.)
        seq_ok = create_segment("ATCGATCG", SequenceType.DNA)
        # Sequence with only AT dinucleotides (freq 1.0)
        seq_violate = create_segment("ATATATAT", SequenceType.DNA)

        # Range that includes 0.0 frequency (for dinucleotides that don't appear)
        config_ok = DinucleotideFrequencyConfig(min_freq=0.0, max_freq=0.3)
        constraint_ok = Constraint(
            inputs=[seq_ok],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config=config_ok,
        )
        assert constraint_ok.evaluate()[0] == 0.0

        # Range that excludes 0.0 frequency, should fail
        config_fail = DinucleotideFrequencyConfig(min_freq=0.1, max_freq=0.3)
        constraint_fail = Constraint(
            inputs=[seq_ok],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config=config_fail,
        )
        assert constraint_fail.evaluate()[0] > 0.0

        # Repetitive sequence, should fail narrow range
        config_violate = DinucleotideFrequencyConfig(min_freq=0.0, max_freq=0.5)
        constraint_violate = Constraint(
            inputs=[seq_violate],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config=config_violate,
        )
        assert constraint_violate.evaluate()[0] > 0.0
        assert (
            "segment_0.dinucleotide_frequency_constraint.dinucleotide_freqs"
            in seq_violate[0]._metadata
        )
        # ATATATAT has AT freq ~0.57 and TA freq ~0.43
        assert (
            abs(
                seq_violate[0]._metadata[
                    "segment_0.dinucleotide_frequency_constraint.dinucleotide_freqs"
                ]["AT"]
                - 4 / 7
            )
            < 1e-9
        )

    @pytest.mark.parametrize("sequence", ["", "A"])
    def test_edge_cases(self, sequence):
        """Test with sequences too short to have dinucleotides (constraint-specific edge case)."""
        segment = create_segment(sequence)
        config = DinucleotideFrequencyConfig(min_freq=0.1, max_freq=0.9)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=dinucleotide_frequency_constraint,
            scoring_function_config=config,
        )
        assert constraint.evaluate()[0] == 1.0  # MAX_ENERGY