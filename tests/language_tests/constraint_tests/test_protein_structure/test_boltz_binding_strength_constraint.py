"""
Comprehensive tests for Boltz Binding Strength constraint.

Tests cover:
1. Configuration validation
2. Registry integration
3. Basic parameter handling
4. Return component selection

Note: Actual Boltz predictions are not tested here as they require
heavy computation. These tests focus on configuration and structure.
"""

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
from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.constraint.protein_structure.boltz_binding_strength_constraint import BoltzBindingStrengthConfig
from ..test_utils import create_segment


class TestBoltzBindingStrengthConstraint:
    """Tests for Boltz Binding Strength constraint."""

    def test_config_with_custom_return_component(self):
        """Test config with custom return component."""
        config = BoltzBindingStrengthConfig(return_component="iptm")
        assert config.return_component == "iptm"

    def test_config_with_custom_params(self):
        """Test config with custom parameters."""
        config = BoltzBindingStrengthConfig(
            desired_higher={"iptm": 0.95},
            weights={"iptm": 0.5},
            on_error="raise",
            return_component="iptm"
        )
        assert config.desired_higher["iptm"] == 0.95
        assert config.weights["iptm"] == 0.5
        assert config.on_error == "raise"
        assert config.return_component == "iptm"

    def test_via_registry_minimal(self):
        """Test constraint creation via registry with minimal config."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)

        constraint = ConstraintRegistry.create(
            key="boltz-binding-strength",
            segments=[segment],
            config_dict={}
        )

        assert constraint.scoring_function_config.return_component == "total_penalty"
        assert constraint.scoring_function_config.boltz_config is None

    def test_via_registry_with_return_component(self):
        """Test registry with custom return component."""
        segment = create_segment("MVLSPADK", SequenceType.PROTEIN)

        constraint = ConstraintRegistry.create(
            key="boltz-binding-strength",
            segments=[segment],
            config_dict={"return_component": "ligand_iptm"}
        )

        assert constraint.scoring_function_config.return_component == "ligand_iptm"

    def test_via_registry_with_full_config(self):
        """Test registry with full config parameters."""
        segment = create_segment("MKTAYIAKQRQISFVK", SequenceType.PROTEIN)

        constraint = ConstraintRegistry.create(
            key="boltz-binding-strength",
            segments=[segment],
            config_dict={
                "desired_higher": {"iptm": 0.90, "ptm": 0.70},
                "desired_lower": {"complex_ipde": 2.0},
                "tol_higher": {"iptm": 0.05},
                "tol_lower": {"complex_ipde": 2.0},
                "weights": {"iptm": 0.50, "complex_iplddt": 0.30},
                "include_confidence_score": True,
                "on_error": "penalize",
                "return_component": "total_penalty"
            }
        )

        assert constraint.scoring_function_config.desired_higher["iptm"] == 0.90
        assert constraint.scoring_function_config.weights["iptm"] == 0.50

    def test_config_with_batch_size(self):
        """Test config with batch_size."""
        config = BoltzBindingStrengthConfig(
            batch_size=4,
            on_error="penalize",
        )
        assert config.batch_size == 4
        assert config.on_error == "penalize"

    def test_config_with_boltz_params(self):
        """Test config with Boltz prediction parameters."""
        from proto_language.tools.models.structure_prediction.boltz import BoltzConfig

        boltz_cfg = BoltzConfig(
            recycling_steps=3,
            diffusion_samples=1,
        )
        config = BoltzBindingStrengthConfig(boltz_config=boltz_cfg)
        assert config.boltz_config.recycling_steps == 3
        assert config.boltz_config.diffusion_samples == 1

    def test_constraint_spec_not_vectorized(self):
        """Test that constraint is registered as not vectorized."""
        spec = ConstraintRegistry.get("boltz-binding-strength")
        assert spec.vectorized == False

    def test_constraint_spec_not_concatenate(self):
        """Test that constraint is registered with concatenate=False."""
        spec = ConstraintRegistry.get("boltz-binding-strength")
        assert spec.concatenate == False

    def test_constraint_description(self):
        """Test that constraint has a meaningful description."""
        spec = ConstraintRegistry.get("boltz-binding-strength")
        assert len(spec.description) > 20
        assert "boltz" in spec.description.lower() or "binding" in spec.description.lower()
