"""Tests for structure prediction similarity constraints."""

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import pytest
from proto_tools import AlphaFold3Config, Boltz2Config, Chai1Config
from proto_tools.utils.standalone_helpers_source.standalone_helpers import resolve_weights_dir

from proto_language.constraint import (
    structure_rmsd_constraint,
    structure_tmscore_constraint,
)
from proto_language.constraint.protein_structure.structure_similarity_constraint import (
    StructureRMSDConfig,
    StructureTMScoreConfig,
)
from proto_language.core import Sequence
from tests.helpers.mock_structure import MockStructure

CRO_SEQ = "MRKKLDLKKFVEDKNQEYAARALGLSQKLIEEVLKRGLPVYVETNKDGNIKVYITQDGITQPFPP"
TOP7_SEQ = "MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLEGGSLEHHHHHH"
UNCONFIDENT_SEQ = "EASGTYPGREACGGHEASGTYPGREACGGHEASGTYPGREACGGH"
ROP_SEQ = "MTKQEKTALNMARFIRSQTLTLLEKLNELDADEQADICESLHDHADELYRSCLARFGDDGENL"
EPSILON = 0.05


def _alphafold3_weights_skip_reason() -> str | None:
    """Return a skip reason if AlphaFold3 weights aren't resolvable on this host, else None.

    AlphaFold3 weights are gated under DeepMind's ToU and must be obtained
    separately. Mirrors the same gate proto-tools uses in
    ``tests/structure_prediction_tests/test_af3.py``.
    """
    weights_dir = resolve_weights_dir("alphafold3")
    if weights_dir is None:
        return (
            "AlphaFold3 weights dir could not be resolved "
            "(set PROTO_ALPHAFOLD3_WEIGHTS_DIR or PROTO_MODEL_CACHE/PROTO_HOME)"
        )
    if not any(Path(weights_dir).glob("*.bin*")):
        return f"AlphaFold3 weights dir resolved to {weights_dir} but contains no .bin* files"
    return None


def _seeded_tool_config(structure_tool: str) -> dict:
    """Pin the predictor seed so a self-match folds the *same* structure twice.

    Boltz-2/Chai-1/AlphaFold3 are stochastic (diffusion sampling); without a fixed seed the
    target fold and the proposal fold of an identical sequence diverge enough to push the
    self-similarity above ``EPSILON``. Deterministic predictors (ESMFold, AlphaFold2) need nothing.
    """
    if structure_tool == "boltz2":
        return {"boltz2_config": Boltz2Config(seed=0)}
    if structure_tool == "chai1":
        return {"chai1_config": Chai1Config(seed=0)}
    if structure_tool == "alphafold3":
        return {"alphafold3_config": AlphaFold3Config(seeds=[0])}
    return {}


class MockResult(NamedTuple):
    structures: list


def _match(
    constraint: str,
    structure_tool: str,
    proposal_seq: str,
    target_seq: str,
) -> float:
    """Compute similarity between the proposal and target sequences using the.

    specified constraint and structure prediction tool.
    """
    if constraint == "rmsd":
        config = StructureRMSDConfig(
            target_chains=[target_seq],
            structure_tool=structure_tool,
            **_seeded_tool_config(structure_tool),
        )
        score = structure_rmsd_constraint(
            [(Sequence(proposal_seq, "protein"),)],
            config,
        )[0].score

    elif constraint == "tmscore":
        config = StructureTMScoreConfig(
            target_chains=[target_seq],
            structure_tool=structure_tool,
            **_seeded_tool_config(structure_tool),
        )
        score = structure_tmscore_constraint(
            [(Sequence(proposal_seq, "protein"),)],
            config,
        )[0].score

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
        assert _imperfect_match("rmsd", "esmfold") > 0.0

    def test_unconfident_match(self):
        assert _unconfident_match("rmsd", "esmfold") == 1.0

    def test_pdb_file_target(self):
        """Test loading from PDB file. Underlying implementation logic is same.

        so only need to test once.
        """
        config = StructureRMSDConfig(
            target_structure="tests/dummy_data/test_structure_similarity.pdb",
            structure_tool="esmfold",
        )
        rmsd = structure_rmsd_constraint(
            [(Sequence(CRO_SEQ, "protein"),)],
            config,
        )[0].score
        assert rmsd < EPSILON

    def test_pdb_content_target(self):
        """Test comparison with PDB file content. Underlying implementation logic.

        is same so only need to test once.
        """
        with open("tests/dummy_data/test_structure_similarity.pdb") as f:
            pdb_content = f.read().rstrip()

        config = StructureRMSDConfig(
            target_structure=pdb_content,
            structure_tool="esmfold",
        )
        rmsd = structure_rmsd_constraint(
            [(Sequence(CRO_SEQ, "protein"),)],
            config,
        )[0].score
        assert rmsd < EPSILON

    def test_multichain(self):
        """Test multichain comparison."""
        config = StructureRMSDConfig(
            target_chains=(ROP_SEQ, ROP_SEQ),
            structure_tool="esmfold",
        )
        rmsd = structure_rmsd_constraint(
            [
                (
                    Sequence(ROP_SEQ, "protein"),
                    Sequence(ROP_SEQ, "protein"),
                )
            ],
            config,
        )[0].score
        assert rmsd < EPSILON


class TestStructureRMSDToolRouting:
    """Tests for RMSD alignment tool routing that do not run structure predictors."""

    def test_pymol_alignment_method_schema_is_dropdown_enum(self):
        schema = StructureRMSDConfig.model_json_schema()
        assert schema["properties"]["pymol_alignment_method"]["enum"] == ["cealign", "align"]

    def test_pymol_alignment_method_passed_to_tool(self):
        with (
            patch(
                "proto_language.constraint.protein_structure.structure_similarity_constraint._prepare_target_structure"
            ) as mock_target,
            patch(
                "proto_language.constraint.protein_structure.structure_similarity_constraint.predict_structures"
            ) as mock_predict,
            patch(
                "proto_language.constraint.protein_structure.structure_similarity_constraint.run_pymol_rmsd_alignment"
            ) as mock_pymol,
        ):
            mock_target.return_value = MockStructure().structure_pdb
            mock_predict.return_value = MockResult(structures=[MockStructure()])
            mock_pymol.return_value = SimpleNamespace(rmsd=1.25)

            config = StructureRMSDConfig(
                target_structure="ignored by mocked target prep",
                structure_tool="esmfold",
                pymol_alignment_method="align",
            )

            result = structure_rmsd_constraint([(Sequence("AAA", "protein"),)], config)[0]

        assert result.metadata["rmsd_alignment_method"] == "align"
        assert result.metadata["rmsd_val"] == 1.25
        pymol_input, pymol_config = mock_pymol.call_args.args
        assert pymol_input.target_structure.structure_pdb == MockStructure().structure_pdb
        assert pymol_input.mobile_structure.structure_pdb == MockStructure().structure_pdb
        assert pymol_config.method == "align"


class TestESMFoldTMscoreConstraint:
    """Tests for ESMFold TMscore constraint."""

    @pytest.fixture
    def mock_predict(self):
        """Mocks the heavy folding function."""
        with patch(
            "proto_language.constraint.protein_structure.structure_similarity_constraint.predict_structures"
        ) as m:
            # Return a valid structure so the code proceeds
            m.return_value = MockResult(structures=[MockStructure()])
            yield m

    @pytest.fixture
    def mock_target_prep(self):
        """Mocks target preparation to avoid folding the target."""
        with patch(
            "proto_language.constraint.protein_structure.structure_similarity_constraint._prepare_target_structure"
        ) as m:
            m.return_value = MockStructure().structure_pdb
            yield m

    @pytest.fixture
    def mock_tmalign(self):
        """Mocks the TMalign tool wrapper."""
        with patch("proto_language.constraint.protein_structure.structure_similarity_constraint.run_tmalign") as m:
            from proto_tools import TMalignMetrics, TMalignOutput

            m.return_value = TMalignOutput(
                metrics=TMalignMetrics(
                    tm_score_chain_1=0.5,
                    tm_score_chain_2=0.5,
                ),
            )
            yield m

    @pytest.mark.uses_gpu
    def test_perfect_match(self):
        assert _perfect_match("tmscore", "esmfold") == 0.0

    @pytest.mark.uses_gpu
    def test_imperfect_match(self):
        assert _imperfect_match("tmscore", "esmfold") > 0.0

    @pytest.mark.uses_gpu
    def test_unconfident_match(self):
        assert _unconfident_match("tmscore", "esmfold") == 1.0

    @pytest.mark.uses_gpu
    def test_plddt_threshold_filtering(self):
        """Test that setting a pLDDT threshold affects the TM-score calculation."""
        # Test standard calculation (no threshold).
        config_raw = StructureTMScoreConfig(
            target_chains=[CRO_SEQ],
            structure_tool="esmfold",
            plddt_threshold=None,  # Default behavior.
        )
        score_raw = structure_tmscore_constraint(
            [(Sequence(CRO_SEQ, "protein"),)],
            config_raw,
        )[0].score

        assert score_raw < EPSILON

        # Test extreme threshold.
        # This should filter out ALL atoms, resulting in a TM-score of 0.0, so we
        # expect a score of 1.0.
        config_strict = StructureTMScoreConfig(
            target_chains=[CRO_SEQ],
            structure_tool="esmfold",
            plddt_threshold=0.999,  # ESMFold normalizes by 100, so pLDDT is 0-1.
        )
        score_strict = structure_tmscore_constraint(
            [(Sequence(CRO_SEQ, "protein"),)],
            config_strict,
        )[0].score

        assert score_strict == 1.0

    @pytest.mark.uses_gpu
    def test_multimer_perfect_match(self):
        """Test we can compare a multimer to a multimer."""
        config = StructureTMScoreConfig(
            target_chains=(ROP_SEQ, ROP_SEQ),
            structure_tool="esmfold",
        )

        score = structure_tmscore_constraint(
            [
                (
                    Sequence(ROP_SEQ, "protein"),
                    Sequence(ROP_SEQ, "protein"),
                )
            ],
            config,
        )[0].score

        assert score < EPSILON

    @pytest.mark.uses_gpu
    def test_monomer_to_multimer_subunit_match(self):
        """Test we can compare a monomer to a multimer.

        The TM-score is normalized by the Target (Dimer) length.
        Since the monomer covers exactly 50% of the homodimer, the max TM-score
        is approximately 0.5.
        """
        config = StructureTMScoreConfig(
            target_chains=(ROP_SEQ, ROP_SEQ),
            structure_tool="esmfold",
            tm_score_normalization="structure2",
        )

        score = structure_tmscore_constraint(
            [(Sequence(ROP_SEQ, "protein"),)],
            config,
        )[0].score

        assert 0.5 - EPSILON < score < 0.5 + EPSILON

    def test_tm_score_normalization_logic(
        self,
        mock_predict,
        mock_target_prep,
        mock_tmalign,
    ):
        """Verify the math for structure1 vs structure2 vs mean/max/min."""
        # Setup the mock to return distinct scores
        # Structure 1 (Proposal) Norm = 0.8  (Good match)
        # Structure 2 (Target) Norm    = 0.4  (Bad match, maybe target is huge)
        from proto_tools import TMalignMetrics, TMalignOutput

        mock_tmalign.return_value = TMalignOutput(
            metrics=TMalignMetrics(
                tm_score_chain_1=0.8,
                tm_score_chain_2=0.4,
            ),
        )

        cases = [
            ("structure1", 0.8),  # Score = 1.0 - 0.8 = 0.2
            ("structure2", 0.4),  # Score = 1.0 - 0.4 = 0.6
            ("max", 0.8),  # Score = 1.0 - 0.8 = 0.2
            ("min", 0.4),  # Score = 1.0 - 0.4 = 0.6
            ("mean", 0.6),  # Score = 1.0 - 0.6 = 0.4
        ]

        for mode, expected_tm in cases:
            config = StructureTMScoreConfig(
                target_structure="FAKE",  # Content doesn't matter, mocked
                structure_tool="esmfold",
                tm_score_normalization=mode,
            )

            results = structure_tmscore_constraint([(Sequence("AAA", "protein"),)], config)

            # Constraint returns 1.0 - TMscore
            expected_constraint_score = 1.0 - expected_tm
            assert results[0].score == pytest.approx(expected_constraint_score), f"Failed for mode: {mode}"


@pytest.mark.slow
@pytest.mark.uses_gpu
class TestSlowStructurePredictorSimilarityConstraint:
    """Tests for AlphaFold3/Chai1/Boltz RMSD and TMScore constraints."""

    @pytest.mark.skipif(_alphafold3_weights_skip_reason() is not None, reason=_alphafold3_weights_skip_reason() or "")
    def test_perfect_match_af3(self):
        assert _perfect_match("rmsd", "alphafold3") < EPSILON

    def test_perfect_match_chai(self):
        assert _perfect_match("rmsd", "chai1") < EPSILON

    def test_perfect_match_boltz(self):
        assert _perfect_match("tmscore", "boltz2") < EPSILON

    @pytest.mark.skipif(_alphafold3_weights_skip_reason() is not None, reason=_alphafold3_weights_skip_reason() or "")
    def test_imperfect_match(self):
        assert _imperfect_match("tmscore", "alphafold3") > 0.0

    @pytest.mark.skip(
        reason="AF3 multichain (homodimer) RMSD self-match is not deterministic even with a pinned seed "
        "(~0.25 RMSD); needs chain-correspondence-aware comparison."
    )
    def test_multichain(self):
        """Test multichain comparison."""
        config = StructureRMSDConfig(
            target_chains=(ROP_SEQ, ROP_SEQ),
            structure_tool="alphafold3",
        )
        rmsd = structure_rmsd_constraint(
            [
                (
                    Sequence(ROP_SEQ, "protein"),
                    Sequence(ROP_SEQ, "protein"),
                )
            ],
            config,
        )[0].score
        assert rmsd < EPSILON


class TestESMFold2SimilarityBridge:
    """ESMFold2 folds similarity targets with a native ``plddt`` key.

    ``_prepare_target_structure`` must bridge ``plddt`` → ``avg_plddt`` or
    structure-rmsd / -tmscore silently return MAX_ENERGY for every proposal.
    """

    def test_rmsd_esmfold2_target_uses_plddt(self):
        """Regression: an esmfold2 target fold (native ``plddt``) is accepted, not scored worst."""
        config = StructureRMSDConfig(target_chains=[CRO_SEQ], structure_tool="esmfold2")
        folded = MockResult(structures=[MockStructure(metrics={"plddt": 0.9})])
        sim = "proto_language.constraint.protein_structure.structure_similarity_constraint"
        with (
            patch(f"{sim}.predict_structures", return_value=folded),
            patch(f"{sim}.run_pymol_rmsd_alignment", return_value=SimpleNamespace(rmsd=1.0)),
        ):
            [result] = structure_rmsd_constraint([(Sequence(CRO_SEQ, "protein"),)], config)
        assert result.metadata.get("reason") != "unconfident_target"
        assert "rmsd_val" in result.metadata


class TestMinTargetPlddtNormalization:
    """``min_target_plddt`` (0-1 scale) must be compared against a normalized pLDDT.

    AlphaFold3 emits pLDDT on a 0-100 scale; without normalizing, the gate never
    fires (e.g. ``40.0 < 0.6`` is always False), silently accepting low-confidence
    targets. The fix mirrors structure_plddt: ``x / 100 if x > 1 else x``.
    """

    SIM = "proto_language.constraint.protein_structure.structure_similarity_constraint"

    def test_low_confidence_0_100_target_rejected(self):
        """A 0-100 target pLDDT of 40.0 (norm 0.4) is below 0.6 and must be rejected."""
        config = StructureRMSDConfig(target_chains=[CRO_SEQ], structure_tool="alphafold3", min_target_plddt=0.6)
        folded = MockResult(structures=[MockStructure(metrics={"avg_plddt": 40.0})])
        with (
            patch(f"{self.SIM}.predict_structures", return_value=folded),
            patch(f"{self.SIM}.run_pymol_rmsd_alignment", return_value=SimpleNamespace(rmsd=1.0)),
        ):
            [result] = structure_rmsd_constraint([(Sequence(CRO_SEQ, "protein"),)], config)
        assert result.metadata.get("reason") == "unconfident_target"
