"""
Tests for structure prediction similarity constraints.
"""

import pytest

from proto_language.language.core import Sequence
from proto_language.language.constraint import structure_rmsd_constraint, structure_tmscore_constraint
from proto_language.language.constraint.protein_structure.structure_similarity_constraint import (
    StructureRMSDConfig,
    StructureTMScoreConfig,
)


CRO_SEQ = "MRKKLDLKKFVEDKNQEYAARALGLSQKLIEEVLKRGLPVYVETNKDGNIKVYITQDGITQPFPP"
TOP7_SEQ = "MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLEGGSLEHHHHHH"
UNCONFIDENT_SEQ = "EASGTYPGREACGGHEASGTYPGREACGGHEASGTYPGREACGGH"
ROP_SEQ = "MTKQEKTALNMARFIRSQTLTLLEKLNELDADEQADICESLHDHADELYRSCLARFGDDGENL"
EPSILON = 0.05


def _match(
    constraint: str,
    structure_tool: str,
    candidate_seq: str,
    target_seq: str,
) -> float:
    """
    Compute similarity between the candidate and target sequences using the
    specified constraint and structure prediction tool.
    """
    if constraint == "rmsd":
        config = StructureRMSDConfig(
            target_chains=[target_seq],
            structure_tool=structure_tool,
        )
        score = structure_rmsd_constraint(
            [(Sequence(candidate_seq, 'protein'),)],
            config,
        )[0]

    elif constraint == "tmscore":
        config = StructureTMScoreConfig(
            target_chains=[target_seq],
            structure_tool=structure_tool,
        )
        score = structure_tmscore_constraint(
            [(Sequence(candidate_seq, 'protein'),)],
            config,
        )[0]

    return score


def _perfect_match(constraint: str, structure_tool: str) -> float:
    """Compute similarity of what should be a perfect match."""
    return _match(constraint, structure_tool, CRO_SEQ, CRO_SEQ)


def _imperfect_match(constraint: str, structure_tool: str) -> float:
    """Compute similarity of two different proteins."""
    return _match(constraint, structure_tool, CRO_SEQ, TOP7_SEQ)


def _unconfident_match(constraint: str, structure_tool: str) -> float:
    """Compute similarity with an unconfident target."""
    return _match(constraint, structure_tool, CRO_SEQ, UNCONFIDENT_SEQ)


@pytest.mark.uses_gpu
class TestESMFoldRMSDConstraint:
    """Tests for ESMFold RMSD constraint."""

    def test_perfect_match(self):
        assert _perfect_match("rmsd", "esmfold") < EPSILON  # For some reason there is some imprecision.

    def test_imperfect_match(self):
        assert _imperfect_match("rmsd", "esmfold") > 0.

    def test_unconfident_match(self):
        assert _unconfident_match("rmsd", "esmfold") == 1.

    def test_pdb_file_target(self):
        """
        Test loading from PDB file. Underlying implementation logic is same
        so only need to test once.
        """
        config = StructureRMSDConfig(
            target_pdb_file="tests/dummy_data/test_structure_similarity.pdb",
            structure_tool="esmfold",
        )
        rmsd = structure_rmsd_constraint(
            [(Sequence(CRO_SEQ, 'protein'),)],
            config,
        )[0]
        assert rmsd < EPSILON

    def test_pdb_content_target(self):
        """
        Test comparison with PDB file content. Underlying implementation logic
        is same so only need to test once.
        """
        with open("tests/dummy_data/test_structure_similarity.pdb", "r") as f:
            pdb_content = f.read().rstrip()

        config = StructureRMSDConfig(
            target_pdb_content=pdb_content,
            structure_tool="esmfold",
        )
        rmsd = structure_rmsd_constraint(
            [(Sequence(CRO_SEQ, 'protein'),)],
            config,
        )[0]
        assert rmsd < EPSILON

    def test_multichain(self):
        """Test multichain comparison."""
        config = StructureRMSDConfig(
            target_chains=(ROP_SEQ, ROP_SEQ),
            structure_tool="esmfold",
        )
        rmsd = structure_rmsd_constraint(
            [(
                Sequence(ROP_SEQ, 'protein'),
                Sequence(ROP_SEQ, 'protein'),
            )],
            config,
        )[0]
        assert rmsd < EPSILON


@pytest.mark.uses_gpu
class TestESMFoldTMscoreConstraint:
    """Tests for ESMFold TMscore constraint."""

    def test_perfect_match(self):
        assert _perfect_match("tmscore", "esmfold") == 0.

    def test_imperfect_match(self):
        assert _imperfect_match("tmscore", "esmfold") > 0.

    def test_unconfident_match(self):
        assert _unconfident_match("tmscore", "esmfold") == 1.

    def test_plddt_threshold_filtering(self):
        """
        Test that setting a pLDDT threshold affects the TM-score calculation.
        """
        # Test standard calculation (no threshold).
        config_raw = StructureTMScoreConfig(
            target_chains=[CRO_SEQ],
            structure_tool="esmfold",
            plddt_threshold=None,  # Default behavior.
        )
        score_raw = structure_tmscore_constraint(
            [(Sequence(CRO_SEQ, 'protein'),)],
            config_raw,
        )[0]

        assert score_raw < EPSILON

        # Test extreme threshold.
        # This should filter out ALL atoms, resulting in a TM-score of 0.0, so we
        # expect a score of 1.0.
        config_strict = StructureTMScoreConfig(
            target_chains=[CRO_SEQ],
            structure_tool="esmfold",
            plddt_threshold=0.999, # ESMFold normalizes by 100, so pLDDT is 0-1.
        )
        score_strict = structure_tmscore_constraint(
            [(Sequence(CRO_SEQ, 'protein'),)],
            config_strict,
        )[0]

        assert score_strict == 1.0


@pytest.mark.slow
@pytest.mark.uses_gpu
class TestSlowStructurePredictorSimilarityConstraint:
    """Tests for AlphaFold3/Chai/Boltz RMSD and TMScore constraints."""

    def test_perfect_match_af3(self):
        assert _perfect_match("rmsd", "alphafold3") < EPSILON

    def test_perfect_match_chai(self):
        assert _perfect_match("rmsd", "chai") < EPSILON

    def test_perfect_match_boltz(self):
        assert _perfect_match("tmscore", "boltz") < EPSILON

    def test_imperfect_match(self):
        assert _imperfect_match("tmscore", "alphafold3") > 0.

    def test_multichain(self):
        """Test multichain comparison."""
        config = StructureRMSDConfig(
            target_chains=(ROP_SEQ, ROP_SEQ),
            structure_tool="alphafold3",
        )
        rmsd = structure_rmsd_constraint(
            [(
                Sequence(ROP_SEQ, 'protein'),
                Sequence(ROP_SEQ, 'protein'),
            )],
            config,
        )[0]
        assert rmsd < EPSILON
