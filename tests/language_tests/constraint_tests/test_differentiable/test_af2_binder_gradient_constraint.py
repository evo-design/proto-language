"""Tests for the AF2 binder-design gradient constraint."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from proto_tools.entities.structures import Structure

from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.language.constraint.differentiable.af2_binder_gradient_constraint import (
    AF2BinderGradientConfig,
    af2_binder_backward,
)
from proto_language.language.core import Segment, Sequence
from tests.helpers.mock_structure import PDL1_PDB

_TOOL_MODULE = "proto_language.language.constraint.differentiable.af2_binder_gradient_constraint"


def _binder_with_logits(logits: np.ndarray) -> Sequence:
    seq = Sequence("A" * logits.shape[0], "protein")
    seq.logits = logits
    return seq


def _target_with_structure() -> Sequence:
    seq = Sequence("A" * 10, "protein")
    seq.structure = Structure(structure=PDL1_PDB.read_text(), structure_format="pdb")
    return seq


class TestConfig:
    def test_defaults(self) -> None:
        config = AF2BinderGradientConfig()
        assert config.target_chain == "A"
        assert config.binder_chain == "H"
        assert config.backend == "base"

    def test_germinal_vhh_preset(self) -> None:
        config = AF2BinderGradientConfig.germinal_vhh_preset()
        assert config.backend == "germinal"
        assert config.bias_redesign == 10.0
        assert config.loss_weights["i_plddt"] == 1.0
        assert config.omit_aas == "C"


class TestBackward:
    @patch(f"{_TOOL_MODULE}.run_alphafold2_gradient")
    def test_reads_structure_from_target_segment(self, mock_run: object) -> None:
        mock_run.return_value = SimpleNamespace(gradient=[[0.1] * 20] * 5, loss=0.5, metrics={"plddt": 0.8})
        binder = _binder_with_logits(np.ones((5, 20)) / 20.0)
        target = _target_with_structure()

        result = af2_binder_backward((binder, target), temperature=1.0, config=AF2BinderGradientConfig())

        assert len(result.gradient) == 2
        assert result.gradient[0].shape == (5, 20)  # binder gradient
        assert result.gradient[1].shape == (10, 20)  # target zero gradient
        assert np.all(result.gradient[1] == 0.0)
        # Verify PDB string (not path) was passed to tool input
        tool_input = mock_run.call_args[0][0]
        assert "\n" in tool_input.target_pdb  # PDB string, not file path

    @patch(f"{_TOOL_MODULE}.run_alphafold2_gradient")
    def test_forwards_config_to_tool(self, mock_run: object) -> None:
        mock_run.return_value = SimpleNamespace(gradient=[[0.1] * 20] * 5, loss=0.5, metrics={})
        binder = _binder_with_logits(np.ones((5, 20)) / 20.0)
        target = _target_with_structure()
        config = AF2BinderGradientConfig(
            binder_chain="L", loss_weights={"plddt": 2.0}, bias_redesign=5.0, backend="germinal"
        )

        af2_binder_backward((binder, target), temperature=1.0, config=config)

        tool_input = mock_run.call_args[0][0]
        tool_config = mock_run.call_args[0][1]
        assert tool_input.binder_chain == "L"
        assert tool_config.bias_redesign == 5.0
        assert tool_config.backend == "germinal"

    @patch(f"{_TOOL_MODULE}.run_alphafold2_gradient")
    def test_soft_kwarg_defaults_to_one(self, mock_run: object) -> None:
        mock_run.return_value = SimpleNamespace(gradient=[[0.0] * 20] * 3, loss=0.0, metrics={})
        binder = _binder_with_logits(np.zeros((3, 20)))
        target = _target_with_structure()
        config = AF2BinderGradientConfig()

        af2_binder_backward((binder, target), temperature=1.0, config=config, soft=0.5)
        assert mock_run.call_args[0][1].soft == 0.5

        af2_binder_backward((binder, target), temperature=1.0, config=config)
        assert mock_run.call_args[0][1].soft == 1.0


class TestRegistry:
    def test_registers_as_gradient_only(self) -> None:
        spec = ConstraintRegistry.get("af2-binder-gradient")
        assert spec.function is None and spec.backward is af2_binder_backward
        assert spec.input_labels == ["Binder Chain", "Target Structure"]

    def test_create(self) -> None:
        binder_seg = Segment(sequence="EVQLVESG", sequence_type="protein")
        target_seg = Segment(sequence="MKTAYIAK", sequence_type="protein")
        c = ConstraintRegistry.create("af2-binder-gradient", [binder_seg, target_seg], {})
        assert c.supports_gradient and not c.supports_discrete


@pytest.mark.uses_gpu
@pytest.mark.slow
class TestGPU:
    def test_different_inputs_different_gradients(self) -> None:
        config = AF2BinderGradientConfig(
            target_chain="A", binder_chain="B", num_recycles=1, loss_weights={"plddt": 1.0}
        )
        target = _target_with_structure()
        uniform = _binder_with_logits(np.zeros((10, 20), dtype=np.float64))
        biased = _binder_with_logits(np.zeros((10, 20), dtype=np.float64))
        biased.logits[:, 0] = 5.0

        r1 = af2_binder_backward((uniform, target), temperature=1.0, config=config)
        r2 = af2_binder_backward((biased, target), temperature=1.0, config=config)

        assert np.isfinite(r1.gradient[0]).all() and np.isfinite(r2.gradient[0]).all()
        assert r1.loss != r2.loss

    def test_loss_weights_change_gradient(self) -> None:
        target = _target_with_structure()
        binder = _binder_with_logits(np.random.RandomState(42).randn(10, 20).astype(np.float64))
        base = {"target_chain": "A", "binder_chain": "B", "num_recycles": 1}

        r_plddt = af2_binder_backward(
            (binder, target), temperature=1.0, config=AF2BinderGradientConfig(**base, loss_weights={"plddt": 1.0})
        )
        r_con = af2_binder_backward(
            (binder, target), temperature=1.0, config=AF2BinderGradientConfig(**base, loss_weights={"con": 1.0})
        )

        assert not np.allclose(r_plddt.gradient[0], r_con.gradient[0])

    def test_metrics_are_valid(self) -> None:
        config = AF2BinderGradientConfig(
            target_chain="A", binder_chain="B", num_recycles=1, loss_weights={"plddt": 1.0}
        )
        target = _target_with_structure()
        binder = _binder_with_logits(np.zeros((10, 20), dtype=np.float64))

        result = af2_binder_backward((binder, target), temperature=1.0, config=config)

        assert np.isfinite(result.loss) and result.loss != 0.0
        assert "avg_plddt" in result.metrics
