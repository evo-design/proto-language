"""Tests for the AF2 binder-design constraint (dual-mode: forward + backward)."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from proto_tools.entities.structures import BFactorType, Structure

from proto_language.language.constraint.constraint_registry import ConstraintRegistry, InputSlot
from proto_language.language.constraint.differentiable.af2_binder_constraint import (
    AF2BinderConstraintConfig,
    af2_binder_backward,
    af2_binder_forward,
)
from proto_language.language.core import Segment, Sequence
from tests.helpers.mock_structure import PDL1_PDB

_TOOL_MODULE = "proto_language.language.constraint.differentiable.af2_binder_constraint"


def _binder_with_logits(logits: np.ndarray) -> Sequence:
    seq = Sequence("A" * logits.shape[0], "protein")
    seq.logits = logits
    return seq


def _target_sequence() -> Sequence:
    """Target template Sequence — structure is now on config.target_pdb, not on the segment."""
    return Sequence("A" * 10, "protein")


_PDL1_PDB_TEXT = PDL1_PDB.read_text()


_DEFAULT_METRICS = {"avg_plddt": 0.8, "ptm": 0.7, "iptm": 0.6, "avg_pae": 2.0, "i_pae": 1.5}


def _mock_tool_output(
    *,
    gradient: list[list[float]] | None,
    loss: float = 0.5,
    metrics: dict[str, float] | None = None,
    residues: list[tuple[str, int, float]] | None = None,
) -> SimpleNamespace:
    """Build a mock AlphaFold2BinderOutput-shaped object with a PLDDT structure."""
    if residues is None:
        residues = [("A", 1, 95.0), ("A", 2, 80.0), ("B", 1, 70.0), ("B", 2, 65.0)]
    lines = [
        f"ATOM  {i:5d}  CA  ALA {chain}{resseq:4d}    {(i - 1) * 3.8:8.3f}   0.000   0.000  1.00{bf:6.2f}           C  "
        for i, (chain, resseq, bf) in enumerate(residues, start=1)
    ]
    lines.append("END")
    structure = Structure(structure="\n".join(lines), b_factor_type=BFactorType.PLDDT)
    return SimpleNamespace(
        gradient=gradient,
        loss=loss,
        metrics=metrics if metrics is not None else _DEFAULT_METRICS,
        vocab=[],
        structure=structure,
    )


class TestConfig:
    def test_defaults(self) -> None:
        config = AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT)
        assert config.target_chain == "A"
        assert config.binder_chain == "H"
        assert config.backend == "base"

    def test_empty_target_pdb_rejected(self) -> None:
        """Fail fast at config-time; AF2 can't run on an empty template."""
        with pytest.raises(ValueError, match="non-empty PDB"):
            AF2BinderConstraintConfig()

    def test_germinal_vhh_preset(self) -> None:
        config = AF2BinderConstraintConfig.germinal_vhh_preset(target_pdb=_PDL1_PDB_TEXT)
        assert config.backend == "germinal"
        assert config.bias_redesign == 10.0
        assert config.loss_weights["i_plddt"] == 1.0
        assert config.omit_aas == "C"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bias_redesign": 10.0},
            {"design_positions": [0, 1, 2]},
            {"starting_binder_seq": "AAAAA"},
            *({"loss_weights": {"plddt": 1.0, k: 0.1}} for k in ("rg", "i_ptm", "NC", "helix", "beta_strand")),
        ],
    )
    def test_germinal_only_fields_rejected_on_base(self, kwargs: dict[str, object]) -> None:
        """Each germinal-only field / extension loss key fails validation under backend='base'."""
        with pytest.raises(ValueError, match="require backend='germinal'"):
            AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT, **kwargs)


class TestBackward:
    @patch(f"{_TOOL_MODULE}.run_alphafold2_binder")
    def test_dispatches_with_config_and_returns_sized_gradients(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=[[0.1] * 20] * 5, loss=0.5)
        binder = _binder_with_logits(np.ones((5, 20)) / 20.0)
        target = _target_sequence()
        config = AF2BinderConstraintConfig(
            target_pdb=_PDL1_PDB_TEXT,
            binder_chain="B",
            loss_weights={"plddt": 2.0},
            bias_redesign=5.0,
            backend="germinal",
        )

        result = af2_binder_backward((binder, target), temperature=1.0, soft=1.0, config=config)

        tool_input, tool_config = mock_run.call_args[0]
        assert tool_input.binder_chain == "B"
        assert tool_input.target_pdb == _PDL1_PDB_TEXT  # PDB comes from config, not segment slot.
        assert tool_config.bias_redesign == 5.0
        assert tool_config.backend == "germinal"
        assert tool_config.compute_gradient is True
        assert result.gradient[0].shape == (5, 20)
        assert result.gradient[1].shape == (10, 20) and np.all(result.gradient[1] == 0.0)
        # Each segment slot holds its own chain (both sliced from the same AF2 output).
        assert result.structures[0].get_chain_ids() == ["B"]
        assert result.structures[1].get_chain_ids() == ["A"]

    @patch(f"{_TOOL_MODULE}.run_alphafold2_binder")
    def test_preserves_multi_chain_target_structure(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(
            gradient=[[0.1] * 20] * 2,
            residues=[
                ("A", 1, 95.0),
                ("A", 2, 90.0),
                ("B", 1, 80.0),
                ("B", 2, 75.0),
                ("H", 1, 70.0),
                ("H", 2, 65.0),
            ],
        )
        binder = _binder_with_logits(np.ones((2, 20)) / 20.0)
        target = Sequence("A" * 4, "protein")
        config = AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT, target_chain="A,B", binder_chain="H")

        result = af2_binder_backward((binder, target), temperature=1.0, soft=1.0, config=config)

        assert result.structures[0].get_chain_ids() == ["H"]
        assert result.structures[1].get_chain_ids() == ["A", "B"]
        assert result.structures[1].per_residue_plddt == pytest.approx([0.95, 0.90, 0.80, 0.75])

    @patch(f"{_TOOL_MODULE}.run_alphafold2_binder")
    def test_soft_kwarg_forwards_to_tool_config(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(gradient=[[0.0] * 20] * 3, loss=0.0)
        binder = _binder_with_logits(np.zeros((3, 20)))
        target = _target_sequence()
        cfg = AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT, binder_chain="B")

        af2_binder_backward((binder, target), temperature=1.0, config=cfg, soft=0.5)
        assert mock_run.call_args[0][1].soft == 0.5


class TestForward:
    @patch(f"{_TOOL_MODULE}.run_alphafold2_binder")
    def test_returns_raw_loss(self, mock_run: object) -> None:
        cfg = AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT, binder_chain="B")

        def score_for(loss: float) -> float:
            mock_run.return_value = _mock_tool_output(gradient=None, loss=loss)
            return af2_binder_forward([(Sequence("EVQLV", "protein"), _target_sequence())], config=cfg)[0]

        assert score_for(0.0) == pytest.approx(0.0)
        assert score_for(-2.0) < score_for(-1.0) < score_for(0.0) < score_for(1.0) < score_for(2.0)

    @patch(f"{_TOOL_MODULE}.run_alphafold2_binder")
    def test_writes_per_chain_structures_and_metadata(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(
            gradient=None,
            loss=0.75,
            metrics={"avg_plddt": 0.82, "ptm": 0.65, "iptm": 0.55, "avg_pae": 2.1, "i_pae": 1.3},
        )
        binder = Sequence("EV", "protein")
        target = _target_sequence()
        af2_binder_forward(
            [(binder, target)], config=AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT, binder_chain="B")
        )

        assert mock_run.call_args[0][1].compute_gradient is False
        # Binder slot: single-chain structure with (binder_len,) pLDDT — SemigreedyMutationGenerator invariant.
        assert binder.structure.get_chain_ids() == ["B"]
        assert binder.structure.per_residue_plddt == pytest.approx([0.70, 0.65])  # Chain-B bfactors 70/65 → [0,1].
        assert len(binder.structure.per_residue_plddt) == len(binder.sequence)
        # Target slot: the predicted target chain only (complex-building is a consumer concern via concat).
        assert target.structure.get_chain_ids() == ["A"]
        assert binder._metadata["complex_pdb"] == mock_run.return_value.structure.structure_pdb
        assert binder._metadata["avg_plddt"] == 0.82
        assert binder._metadata["i_pae"] == 1.3
        assert binder._metadata["loss"] == 0.75

    @patch(f"{_TOOL_MODULE}.run_alphafold2_binder")
    def test_forward_preserves_multi_chain_target_structure(self, mock_run: object) -> None:
        mock_run.return_value = _mock_tool_output(
            gradient=None,
            residues=[
                ("A", 1, 95.0),
                ("A", 2, 90.0),
                ("B", 1, 80.0),
                ("B", 2, 75.0),
                ("H", 1, 70.0),
                ("H", 2, 65.0),
            ],
        )
        binder = Sequence("EV", "protein")
        target = Sequence("A" * 4, "protein")

        af2_binder_forward(
            [(binder, target)],
            config=AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT, target_chain="A,B", binder_chain="H"),
        )

        assert binder.structure.get_chain_ids() == ["H"]
        assert target.structure.get_chain_ids() == ["A", "B"]
        assert target.structure.per_residue_plddt == pytest.approx([0.95, 0.90, 0.80, 0.75])

    @patch(f"{_TOOL_MODULE}.run_alphafold2_binder")
    def test_forward_sends_true_one_hot_with_hard_ste(self, mock_run: object) -> None:
        """Forward scoring always sends a true one-hot matrix with ColabDesign ``hard=1, soft=0``."""
        mock_run.return_value = _mock_tool_output(gradient=None, loss=0.5)
        af2_binder_forward(
            [(Sequence("EVQ", "protein"), _target_sequence())],
            config=AF2BinderConstraintConfig(target_pdb=_PDL1_PDB_TEXT, binder_chain="B"),
        )
        tool_input, tool_config = mock_run.call_args[0]
        assert tool_config.hard == 1.0
        assert tool_config.soft == 0.0
        # Exact one-hot: every row has a single 1.0 and all other entries 0.0.
        for row in tool_input.logits:
            assert sum(row) == 1.0
            assert max(row) == 1.0


class TestRegistry:
    def test_registers_as_dual_mode(self) -> None:
        spec = ConstraintRegistry.get("af2-binder")
        assert spec.mode == "dual"
        assert spec.function is af2_binder_forward
        assert spec.backward is af2_binder_backward
        assert spec.input_labels == [
            InputSlot(label="Binder Chain", requires_logits=True),
            InputSlot(label="Target"),
        ]

    def test_factory_builds_dual_capable_constraint(self) -> None:
        binder_seg = Segment(sequence="EVQLVESG", sequence_type="protein")
        target_seg = Segment(sequence="MKTAYIAK", sequence_type="protein")
        c = ConstraintRegistry.create("af2-binder", [binder_seg, target_seg], {"target_pdb": _PDL1_PDB_TEXT})
        assert c.supports_gradient and c.supports_discrete


@pytest.mark.uses_gpu
@pytest.mark.slow
class TestGPU:
    def test_different_inputs_produce_different_gradients(self) -> None:
        config = AF2BinderConstraintConfig(
            target_pdb=_PDL1_PDB_TEXT, target_chain="A", binder_chain="B", num_recycles=1, loss_weights={"plddt": 1.0}
        )
        target = _target_sequence()
        uniform = _binder_with_logits(np.zeros((10, 20), dtype=np.float64))
        biased = _binder_with_logits(np.zeros((10, 20), dtype=np.float64))
        biased.logits[:, 0] = 5.0

        r1 = af2_binder_backward((uniform, target), temperature=1.0, soft=1.0, config=config)
        r2 = af2_binder_backward((biased, target), temperature=1.0, soft=1.0, config=config)

        assert np.isfinite(r1.gradient[0]).all() and np.isfinite(r2.gradient[0]).all()
        assert r1.loss != r2.loss

    def test_loss_weights_change_gradient(self) -> None:
        target = _target_sequence()
        binder = _binder_with_logits(np.random.RandomState(42).randn(10, 20).astype(np.float64))
        base = {"target_pdb": _PDL1_PDB_TEXT, "target_chain": "A", "binder_chain": "B", "num_recycles": 1}

        r_plddt = af2_binder_backward(
            (binder, target),
            temperature=1.0,
            soft=1.0,
            config=AF2BinderConstraintConfig(**base, loss_weights={"plddt": 1.0}),
        )
        r_con = af2_binder_backward(
            (binder, target),
            temperature=1.0,
            soft=1.0,
            config=AF2BinderConstraintConfig(**base, loss_weights={"con": 1.0}),
        )
        assert not np.allclose(r_plddt.gradient[0], r_con.gradient[0])

    def test_metrics_populated(self) -> None:
        config = AF2BinderConstraintConfig(
            target_pdb=_PDL1_PDB_TEXT, target_chain="A", binder_chain="B", num_recycles=1, loss_weights={"plddt": 1.0}
        )
        result = af2_binder_backward(
            (_binder_with_logits(np.zeros((10, 20), dtype=np.float64)), _target_sequence()),
            temperature=1.0,
            soft=1.0,
            config=config,
        )
        assert np.isfinite(result.loss) and result.loss != 0.0
        assert "avg_plddt" in result.metrics
