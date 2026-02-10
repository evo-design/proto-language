"""
Tests for SpliceTransformer constraints.
"""

import pytest

from proto_language.language.core import Sequence
from proto_language.language.constraint.rna_splicing.splice_transformer_specificity import (
    splice_transformer_specificity,
    SpliceTransformerSpecificityConfig,
)
from proto_language.language.constraint.rna_splicing.splice_transformer_intron_boundary import (
    splice_transformer_intron_boundary,
    SpliceTransformerIntronBoundaryConfig,
)
from proto_language.bio_tools.tools.rna_splicing.splice_transformer import (
    SpliceTransformerConfig,
    CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH,
    TARGET_LENGTH as SPLICE_TRANSFORMER_TARGET_LENGTH,
)


@pytest.mark.skip_ci
def test_splice_transformer_tissue_specificity():
    """
    Test that tissue specificity can be computed correctly.
    """
    input_sequence = Sequence("A" * SPLICE_TRANSFORMER_TARGET_LENGTH)

    specificity_config = SpliceTransformerSpecificityConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        splice_pos=[0, 100, -1],
        tissue="BRAIN",
        direction="max",
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    scores = splice_transformer_specificity([(input_sequence,)], specificity_config)

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"


@pytest.mark.uses_gpu
def test_splice_transformer_all_tissues():
    """
    Test that average tissue specificity can be computed correctly.
    """
    input_sequence = Sequence("A" * SPLICE_TRANSFORMER_TARGET_LENGTH)

    specificity_config = SpliceTransformerSpecificityConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        splice_pos=50,
        tissue="AVERAGE",
        direction="min",
        splice_transformer_config=SpliceTransformerConfig(device="cuda"),
    )

    scores = splice_transformer_specificity([(input_sequence,)], specificity_config)

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"


@pytest.mark.skip_ci
def test_splice_transformer_intron_boundary_cpu():
    """
    Test that intron boundary computation can be computed correctly.
    """
    input_sequence = Sequence("A" * SPLICE_TRANSFORMER_TARGET_LENGTH)

    boundary_config = SpliceTransformerIntronBoundaryConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        donor_pos=[0, 100, -1],
        acceptor_pos=[0, 100, -1],
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    scores = splice_transformer_intron_boundary([(input_sequence,)], boundary_config)

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"


@pytest.mark.uses_gpu
def test_splice_transformer_intron_boundary_gpu():
    """
    Test that intron boundary computation can be computed correctly.
    """
    input_sequence = Sequence("A" * SPLICE_TRANSFORMER_TARGET_LENGTH)

    boundary_config = SpliceTransformerIntronBoundaryConfig(
        left_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        right_context="A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
        donor_pos=50,
        acceptor_pos=60,
        splice_transformer_config=SpliceTransformerConfig(device="cuda"),
    )

    scores = splice_transformer_intron_boundary([(input_sequence,)], boundary_config)

    assert len(scores) == 1
    assert 0. <= scores[0] <= 1., "Score must be between 0 and 1, inclusive"
