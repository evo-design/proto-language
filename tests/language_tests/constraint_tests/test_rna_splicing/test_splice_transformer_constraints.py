"""
tests/language_tests/constraint_tests/test_rna_splicing/test_splice_transformer_constraints.py

Both constraints accept three-part input tuples (left_flank, intron_core,
right_flank) and concatenate them into a single target sequence for scoring.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from proto_tools import CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH
from proto_tools import TARGET_LENGTH as SPLICE_TRANSFORMER_TARGET_LENGTH
from proto_tools import SpliceTransformerConfig

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.constraint.rna_splicing.splice_transformer_intron_boundary import (
    SpliceTransformerIntronBoundaryConfig,
    splice_transformer_intron_boundary,
)
from proto_language.language.constraint.rna_splicing.splice_transformer_specificity import (
    SpliceTransformerSpecificityConfig,
    splice_transformer_specificity,
)
from proto_language.language.core import Segment, Sequence

# --- Registration ---


def test_boundary_registered():
    spec = ConstraintRegistry.get("splice-transformer-intron-boundary")
    assert spec.num_input_sequences_per_tuple == 3
    assert spec.key == "splice-transformer-intron-boundary"


def test_specificity_registered():
    spec = ConstraintRegistry.get("splice-transformer-specificity")
    assert spec.num_input_sequences_per_tuple == 3
    assert spec.key == "splice-transformer-specificity"


# --- Wrong sequence type ---


def test_boundary_wrong_sequence_type():
    with pytest.raises(TypeError, match="does not support sequence type"):
        ConstraintRegistry.create(
            key="splice-transformer-intron-boundary",
            segments=[
                Segment(sequence="MKTAY", sequence_type="protein"),
                Segment(sequence="ACDEF", sequence_type="protein"),
                Segment(sequence="GHIKL", sequence_type="protein"),
            ],
            config_dict={
                "left_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "right_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "donor_pos": [10],
                "acceptor_pos": [20],
            },
        )


def test_specificity_wrong_sequence_type():
    with pytest.raises(TypeError, match="does not support sequence type"):
        ConstraintRegistry.create(
            key="splice-transformer-specificity",
            segments=[
                Segment(sequence="MKTAY", sequence_type="protein"),
                Segment(sequence="ACDEF", sequence_type="protein"),
                Segment(sequence="GHIKL", sequence_type="protein"),
            ],
            config_dict={
                "left_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "right_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "splice_pos": [10],
                "tissue": "BRAIN",
                "direction": "max",
            },
        )


# --- Scoring ---


@pytest.mark.skip_ci
def test_splice_transformer_tissue_specificity():
    """Test that tissue specificity can be computed correctly."""
    left_flank = Sequence("A" * 200, sequence_type="dna")
    intron_core = Sequence("A" * 600, sequence_type="dna")
    right_flank = Sequence("A" * 200, sequence_type="dna")

    specificity_config = SpliceTransformerSpecificityConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        splice_pos=[0, 100, -1],
        tissue="BRAIN",
        direction="max",
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    scores = splice_transformer_specificity(
        [(left_flank, intron_core, right_flank)], specificity_config
    )

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"


@pytest.mark.uses_gpu
def test_splice_transformer_all_tissues():
    """Test that average tissue specificity can be computed correctly."""
    left_flank = Sequence("A" * 200, sequence_type="dna")
    intron_core = Sequence("A" * 600, sequence_type="dna")
    right_flank = Sequence("A" * 200, sequence_type="dna")

    specificity_config = SpliceTransformerSpecificityConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        splice_pos=50,
        tissue="AVERAGE",
        direction="min",
        splice_transformer_config=SpliceTransformerConfig(device="cuda"),
    )

    scores = splice_transformer_specificity(
        [(left_flank, intron_core, right_flank)], specificity_config
    )

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"


@pytest.mark.skip_ci
def test_splice_transformer_intron_boundary_cpu():
    """Test that intron boundary computation can be computed correctly."""
    left_flank = Sequence("A" * 200, sequence_type="dna")
    intron_core = Sequence("A" * 600, sequence_type="dna")
    right_flank = Sequence("A" * 200, sequence_type="dna")

    boundary_config = SpliceTransformerIntronBoundaryConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        donor_pos=[0, 100, -1],
        acceptor_pos=[0, 100, -1],
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    scores = splice_transformer_intron_boundary(
        [(left_flank, intron_core, right_flank)], boundary_config
    )

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"


@pytest.mark.uses_gpu
def test_splice_transformer_intron_boundary_gpu():
    """Test that intron boundary computation can be computed correctly."""
    left_flank = Sequence("A" * 200, sequence_type="dna")
    intron_core = Sequence("A" * 600, sequence_type="dna")
    right_flank = Sequence("A" * 200, sequence_type="dna")

    boundary_config = SpliceTransformerIntronBoundaryConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        donor_pos=50,
        acceptor_pos=60,
        splice_transformer_config=SpliceTransformerConfig(device="cuda"),
    )

    scores = splice_transformer_intron_boundary(
        [(left_flank, intron_core, right_flank)], boundary_config
    )

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"


# --- Metadata propagation ---


@pytest.mark.skip_ci
def test_boundary_metadata_propagation():
    """Metadata should be propagated to all three input sequences."""
    left_flank = Sequence("A" * 200, sequence_type="dna")
    intron_core = Sequence("C" * 600, sequence_type="dna")
    right_flank = Sequence("G" * 200, sequence_type="dna")

    config = SpliceTransformerIntronBoundaryConfig(
        left_context="T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        donor_pos=[199],
        acceptor_pos=[800],
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    splice_transformer_intron_boundary(
        [(left_flank, intron_core, right_flank)], config
    )

    for seq in (left_flank, intron_core, right_flank):
        assert "donor_score" in seq._metadata
        assert "acceptor_score" in seq._metadata
        assert "total_splice_score" in seq._metadata


@pytest.mark.skip_ci
def test_specificity_metadata_propagation():
    """Metadata should be propagated to all three input sequences."""
    left_flank = Sequence("A" * 200, sequence_type="dna")
    intron_core = Sequence("C" * 600, sequence_type="dna")
    right_flank = Sequence("G" * 200, sequence_type="dna")

    config = SpliceTransformerSpecificityConfig(
        left_context="T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        splice_pos=[199, 800],
        tissue="BRAIN",
        direction="max",
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    splice_transformer_specificity(
        [(left_flank, intron_core, right_flank)], config
    )

    for seq in (left_flank, intron_core, right_flank):
        assert "specificity_direction_BRAIN" in seq._metadata
        assert "specificity_score_BRAIN" in seq._metadata


# --- Batched inference ---


def test_splice_transformer_specificity_batches():
    left_a = Sequence("A" * 200, sequence_type="dna")
    intron_a = Sequence("A" * 600, sequence_type="dna")
    right_a = Sequence("A" * 200, sequence_type="dna")
    left_b = Sequence("C" * 200, sequence_type="dna")
    intron_b = Sequence("C" * 600, sequence_type="dna")
    right_b = Sequence("C" * 200, sequence_type="dna")

    predictions = np.zeros((2, SPLICE_TRANSFORMER_TARGET_LENGTH, 18), dtype=float)
    brain_channel = 6  # SPLICE_TISSUE_CHANNEL_INDEX["BRAIN"]
    predictions[0, 10, brain_channel] = 0.2
    predictions[0, 20, brain_channel] = 0.4
    predictions[1, 10, brain_channel] = 0.8
    predictions[1, 20, brain_channel] = 0.6

    specificity_config = SpliceTransformerSpecificityConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        splice_pos=[10, 20],
        tissue="BRAIN",
        direction="max",
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    with patch(
        "proto_language.language.constraint.rna_splicing.splice_transformer_specificity.run_splice_transformer",
        return_value=SimpleNamespace(prediction=predictions),
    ) as mock_run:
        scores = splice_transformer_specificity(
            [(left_a, intron_a, right_a), (left_b, intron_b, right_b)],
            specificity_config,
        )

    assert scores == pytest.approx([0.7, 0.3], abs=1e-6)
    mock_run.assert_called_once()
    call_input = mock_run.call_args[0][0]
    assert len(call_input.target_seqs) == 2


def test_splice_transformer_intron_boundary_batches():
    left_a = Sequence("A" * 200, sequence_type="dna")
    intron_a = Sequence("A" * 600, sequence_type="dna")
    right_a = Sequence("A" * 200, sequence_type="dna")
    left_b = Sequence("C" * 200, sequence_type="dna")
    intron_b = Sequence("C" * 600, sequence_type="dna")
    right_b = Sequence("C" * 200, sequence_type="dna")

    predictions = np.zeros((2, SPLICE_TRANSFORMER_TARGET_LENGTH, 18), dtype=float)
    predictions[0, 50, 2] = 0.9  # donor
    predictions[0, 60, 1] = 0.5  # acceptor
    predictions[1, 50, 2] = 0.2
    predictions[1, 60, 1] = 0.4

    boundary_config = SpliceTransformerIntronBoundaryConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        donor_pos=[50],
        acceptor_pos=[60],
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    with patch(
        "proto_language.language.constraint.rna_splicing.splice_transformer_intron_boundary.run_splice_transformer",
        return_value=SimpleNamespace(prediction=predictions),
    ) as mock_run:
        scores = splice_transformer_intron_boundary(
            [(left_a, intron_a, right_a), (left_b, intron_b, right_b)],
            boundary_config,
        )

    assert scores == pytest.approx([0.3, 0.7], abs=1e-6)
    mock_run.assert_called_once()
    call_input = mock_run.call_args[0][0]
    assert len(call_input.target_seqs) == 2
