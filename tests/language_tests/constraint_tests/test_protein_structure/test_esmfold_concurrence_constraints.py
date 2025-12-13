"""
Comprehensive tests for ESMFold constraints (pLDDT and pTM).

Tests cover:
1. Basic scoring functionality
2. Sequence replication for multimers
3. ESMFold configuration parameters
4. Caching behavior
5. Metadata storage
"""

import pytest

from proto_language.language.core import Sequence
from proto_language.language.constraint import esmfold_rmsd_constraint
from proto_language.language.constraint.protein_structure.esmfold_concurrence_constraint import (
    ESMFoldRMSDConfig,
)


CRO_SEQ = "MRKKLDLKKFVEDKNQEYAARALGLSQKLIEEVLKRGLPVYVETNKDGNIKVYITQDGITQPFPP"
TOP7_SEQ = "MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLEGGSLEHHHHHH"
UNCONFIDENT_SEQ = "EASGTYPGREACGGHEASGTYPGREACGGHEASGTYPGREACGGH"


@pytest.mark.uses_gpu
class TestESMFoldRMSDConstraint:
    """Tests for ESMFold RMSD constraint."""

    def test_perfect_match(self):
        """Test that comparing a sequence with itself gives 0 RMSD."""
        config = ESMFoldRMSDConfig(target_sequence=CRO_SEQ)
        rmsd = esmfold_rmsd_constraint([Sequence(CRO_SEQ, 'protein')], config)[0]
        assert rmsd < 0.01  # For some reason there is some imprecision.

    def test_imperfect_match(self):
        """Test comparing different sequences."""
        config = ESMFoldRMSDConfig(target_sequence=TOP7_SEQ)
        rmsd = esmfold_rmsd_constraint([Sequence(CRO_SEQ, 'protein')], config)[0]
        assert rmsd > 0.

    def test_unconfident_match(self):
        """Test that using an unconfident target sequence results in an RMSD of 1."""
        config = ESMFoldRMSDConfig(target_sequence=UNCONFIDENT_SEQ)
        rmsd = esmfold_rmsd_constraint([Sequence(CRO_SEQ, 'protein')], config)[0]
        assert rmsd == 1.
