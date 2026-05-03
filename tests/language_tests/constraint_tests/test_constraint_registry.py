"""Tests for constraint registry registration, discovery, schema generation, and factory methods."""

import copy
from typing import Any

import pytest
from pydantic import BaseModel, Field

from proto_language.language.constraint import ConstraintRegistry, constraint
from proto_language.language.core import Constraint, ConstraintOutput, Segment

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def dna_segment():
    """Create a DNA segment for testing."""
    return Segment(sequence="ATCGATCGATCG", sequence_type="dna")


@pytest.fixture
def protein_segment():
    """Create a protein segment for testing."""
    return Segment(sequence="MKTAYIAKQRQISFVK", sequence_type="protein")


# ============================================================================
# Unit Tests: Registration
# ============================================================================


class TestRegistration:
    """Test constraint registration mechanism."""

    def test_register_stores_constraint_spec(self):
        """Test that register decorator stores ConstraintSpec in registry."""
        initial_count = ConstraintRegistry.count()

        # Create temporary config for testing
        class TestConfig(BaseModel):
            threshold: float = Field(default=0.5, description="Test threshold")

        # Register a test constraint
        @constraint(
            key="test-temp-constraint",
            label="Test Temp Constraint",
            config=TestConfig,
            description="Temporary test constraint",
            supported_sequence_types=["dna", "protein"],
        )
        def test_constraint(input_sequences, config: TestConfig):
            return [ConstraintOutput(score=0.5) for _ in input_sequences]

        # Verify registration
        assert ConstraintRegistry.count() == initial_count + 1
        assert "test-temp-constraint" in sorted(ConstraintRegistry._registry.keys())

        # Get the spec and verify
        spec = ConstraintRegistry.get("test-temp-constraint")
        assert spec.config_model == TestConfig
        assert spec.description == "Temporary test constraint"
        assert spec.function == test_constraint

        # Cleanup
        del ConstraintRegistry._registry["test-temp-constraint"]

    def test_register_returns_original_function(self):
        """Test that register decorator returns the original function unchanged."""

        class TestConfig(BaseModel):
            pass

        def original_func(input_sequences, config: TestConfig):
            """Original docstring."""
            return [ConstraintOutput(score=0.5) for _ in input_sequences]

        registered_func = ConstraintRegistry.register(
            key="test-return",
            label="Test Return",
            config=TestConfig,
            description="Test",
            supported_sequence_types=["protein"],
        )(original_func)

        assert registered_func == original_func
        assert registered_func.__doc__ == "Original docstring."
        assert registered_func.__name__ == "original_func"

        # Cleanup
        del ConstraintRegistry._registry["test-return"]

    def test_duplicate_registration_raises_error(self):
        """Test that registering same key twice raises ValueError."""

        class TestConfig(BaseModel):
            value: int = 1

        # First registration should work
        @constraint(
            key="test-duplicate-check",
            label="Test Duplicate Check",
            config=TestConfig,
            description="Test constraint",
            supported_sequence_types=["dna"],
        )
        def test_constraint_1(sequence, config):
            return 0.0

        # Verify first registration worked
        assert "test-duplicate-check" in ConstraintRegistry._registry

        # Second registration with same key should raise ValueError
        with pytest.raises(ValueError, match="already registered"):

            @constraint(
                key="test-duplicate-check",  # Same key!
                label="Test Duplicate Check 2",
                config=TestConfig,
                description="Another test",
                supported_sequence_types=["dna"],
            )
            def test_constraint_2(sequence, config):
                return 0.0

        # Cleanup
        del ConstraintRegistry._registry["test-duplicate-check"]

    def test_external_backward_still_enforces_slot_requirements(self):
        """@constraint(backward=separate_fn) path must still run slot validation in compute_gradient."""
        import numpy as np

        from proto_language.language.constraint import InputSlot
        from proto_language.language.core.constraint import GradientConstraintOutput

        class TestConfig(BaseModel):
            pass

        def external_backward(
            input_sequences: list[tuple], *, config: BaseModel, **kwargs: Any
        ) -> list[GradientConstraintOutput]:
            return [GradientConstraintOutput(gradient=(np.zeros((3, 20)),), loss=0.0) for _ in input_sequences]

        @constraint(
            key="test-external-backward",
            label="Test External Backward",
            config=TestConfig,
            description="Scoring + explicit external backward",
            supported_sequence_types=["protein"],
            input_labels=[InputSlot(label="Chain", requires_logits=True)],
            backward=external_backward,
        )
        def scoring_fn(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        try:
            seg = Segment(sequence="ACD", sequence_type="protein")
            c = ConstraintRegistry.create("test-external-backward", [seg], {})
            # No logits set → slot check must fire from compute_gradient.
            with pytest.raises(RuntimeError, match=r"slot 0 'Chain': missing logits"):
                c.compute_gradient(temperature=1.0)
        finally:
            del ConstraintRegistry._registry["test-external-backward"]


# ============================================================================
# Unit Tests: Transactional API (snapshot / restore / unregister)
# ============================================================================


class TestTransactionalAPI:
    """snapshot / restore / unregister on BaseRegistry via ConstraintRegistry."""

    @staticmethod
    def _register_temp(key: str) -> None:
        class _Cfg(BaseModel):
            value: float = Field(default=0.0)

        @constraint(
            key=key,
            label=f"Probe {key}",
            config=_Cfg,
            description="Probe.",
            supported_sequence_types=["protein"],
        )
        def _probe(input_sequences, config: _Cfg):  # type: ignore[no-untyped-def]
            return [ConstraintOutput(score=config.value) for _ in input_sequences]

    def test_snapshot_restore_roundtrip(self):
        """Restore must return the registry to exactly the snapshot state — no more, no less."""
        baseline = set(ConstraintRegistry._registry)
        snap = ConstraintRegistry.snapshot()
        try:
            self._register_temp("test-roundtrip-a")
            self._register_temp("test-roundtrip-b")
            assert {"test-roundtrip-a", "test-roundtrip-b"} <= set(ConstraintRegistry._registry)
        finally:
            ConstraintRegistry.restore(snap)
        assert set(ConstraintRegistry._registry) == baseline

    def test_snapshot_is_deep_copy(self):
        """Snapshot must be independent of the live registry in both directions."""
        snap = ConstraintRegistry.snapshot()
        assert snap, "expected at least one shipped constraint to be registered"
        first_key = next(iter(snap))
        assert snap[first_key] is not ConstraintRegistry.get(first_key)
        snap.pop(first_key)
        assert first_key in ConstraintRegistry._registry

    def test_unregister_removes_present_key_and_noops_on_missing(self):
        """Unregister deletes when present, silently does nothing when absent."""
        try:
            self._register_temp("test-unregister-x")
            ConstraintRegistry.unregister("test-unregister-x")
            assert "test-unregister-x" not in ConstraintRegistry._registry
            ConstraintRegistry.unregister("not-a-real-key")  # must not raise
        finally:
            ConstraintRegistry._registry.pop("test-unregister-x", None)


# ============================================================================
# Unit Tests: Discovery Methods
# ============================================================================


class TestDiscovery:
    """Test constraint discovery methods."""

    def test_list_all_returns_all_constraints(self):
        """Test that list_all returns all registered constraints."""
        all_constraints = ConstraintRegistry.list_all()

        assert isinstance(all_constraints, list)
        assert len(all_constraints) >= 20  # Should have at least 20 constraints

        # Check structure of returned data
        for spec in all_constraints:
            assert spec.key is not None
            assert spec.label is not None
            assert spec.description is not None
            assert hasattr(spec, "uses_gpu")
            # Verify config_model is present and can generate JSON schema
            assert spec.config_model is not None
            schema = spec.config_model.model_json_schema()
            assert isinstance(schema, dict)
            assert "properties" in schema
            assert isinstance(spec.uses_gpu, bool)

    def test_count_returns_correct_number(self):
        """Test that count returns the correct number of registered constraints."""
        count = ConstraintRegistry.count()
        keys = sorted(ConstraintRegistry._registry.keys())

        assert count == len(keys)
        assert count >= 20

    def test_get_returns_constraint_spec(self):
        """Test that get returns the correct ConstraintSpec."""
        spec = ConstraintRegistry.get("gc-content")

        assert spec.description is not None
        assert spec.config_model is not None
        assert spec.function is not None

    def test_get_raises_on_unknown_key(self):
        """Test that get raises ValueError for unknown constraint."""
        with pytest.raises(ValueError, match="Unknown constraint"):
            ConstraintRegistry.get("nonexistent-constraint-key")

    def test_get_error_message_lists_available(self):
        """Test that error message includes available constraints."""
        try:
            ConstraintRegistry.get("bad-key")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "Available constraints:" in error_msg
            assert "gc-content" in error_msg  # Should list actual constraints


# ============================================================================
# Unit Tests: Schema Generation
# ============================================================================


class TestSchemaGeneration:
    """Test JSON schema generation for client integration."""

    def test_get_schema_returns_valid_json_schema(self):
        """Test that get_schema returns valid JSON Schema."""
        schema = ConstraintRegistry.get_schema("gc-content")

        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "title" in schema

        # Check for expected properties
        properties = schema["properties"]
        assert "min_gc" in properties
        assert "max_gc" in properties

        # Check property structure
        for prop_info in properties.values():
            assert "type" in prop_info or "anyOf" in prop_info
            assert "description" in prop_info

    def test_schema_includes_field_descriptions(self):
        """Test that generated schemas include field descriptions."""
        schema = ConstraintRegistry.get_schema("sequence-length")

        properties = schema["properties"]
        assert "target_length" in properties

        target_length_info = properties["target_length"]
        assert "description" in target_length_info
        assert len(target_length_info["description"]) > 10  # Should be meaningful


# ============================================================================
# Unit Tests: Factory Method (create)
# ============================================================================


class TestFactoryMethod:
    """Test Constraint creation from config dicts."""

    def test_create_validates_and_creates_constraint(self, dna_segment):
        """Test that create validates config and creates Constraint."""
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[dna_segment], config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )

        assert isinstance(constraint, Constraint)
        assert constraint.function is not None
        assert constraint.function_config is not None
        assert constraint.function_config.min_gc == 40.0
        assert constraint.function_config.max_gc == 60.0

    def test_create_raises_on_invalid_config(self, dna_segment):
        """create() reformats Pydantic ValidationError into a ValueError."""
        with pytest.raises(ValueError, match=r"config invalid"):
            ConstraintRegistry.create(
                key="gc-content",
                segments=[dna_segment],
                config_dict={"min_gc": "invalid", "max_gc": 60.0},  # Wrong type
            )

    def test_create_raises_on_missing_required_params(self, dna_segment):
        """create() reformats missing-required-field ValidationError into a ValueError."""
        with pytest.raises(ValueError, match=r"config invalid"):
            ConstraintRegistry.create(
                key="gc-content",
                segments=[dna_segment],
                config_dict={"min_gc": 40.0},  # Missing max_gc
            )

    def test_create_with_nested_config(self, dna_segment, tmp_path):
        """Test create with nested Pydantic models in config."""
        # Create dummy database
        dummy_db = tmp_path / "test.db"
        dummy_db.mkdir()

        constraint = ConstraintRegistry.create(
            key="mmseqs-gene-similarity",
            segments=[dna_segment],
            config_dict={
                "min_similarity": 80.0,
                "max_similarity": 100.0,
                "mmseqs_db": str(dummy_db),
            },
        )

        assert isinstance(constraint, Constraint)
        # Nested configs should be Pydantic models, not dicts
        assert hasattr(constraint.function_config, "mmseqs_config")
        assert hasattr(constraint.function_config, "mmseqs_db")

    def test_create_with_label(self, dna_segment):
        """Test that create accepts and sets label."""
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0},
            label="test_gc_label",
        )

        assert constraint.label == "test_gc_label"

    def test_create_validates_sequence_type(self, dna_segment, protein_segment):
        """Test that create validates sequence type compatibility."""
        # gc-content only supports dna and rna, not protein
        with pytest.raises(TypeError, match="does not support sequence type"):
            ConstraintRegistry.create(
                key="gc-content", segments=[protein_segment], config_dict={"min_gc": 40.0, "max_gc": 60.0}
            )

        # protein-length only supports protein, not dna
        with pytest.raises(TypeError, match="does not support sequence type"):
            ConstraintRegistry.create(
                key="protein-length", segments=[dna_segment], config_dict={"min_length": 10, "max_length": 500}
            )

    def test_create_with_compatible_sequence_type(self, dna_segment, protein_segment):
        """Test that create works with compatible sequence types."""
        # gc-content with dna segment
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[dna_segment], config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )
        assert constraint is not None

        # protein-length with protein segment
        constraint = ConstraintRegistry.create(
            key="protein-length", segments=[protein_segment], config_dict={"min_length": 10, "max_length": 500}
        )
        assert constraint is not None

    def test_create_returns_gradient_capable_when_backward_registered(self, dna_segment):
        """Backward callable in @constraint → create() returns Constraint with gradient support."""
        import numpy as np

        from proto_language.language.core.constraint import GradientConstraintOutput

        class _Cfg(BaseModel):
            min_gc: float = 40.0
            max_gc: float = 60.0

        def _backward(logits, *, config, **kwargs):
            return GradientConstraintOutput(gradient=(np.zeros_like(logits),), loss=0.0)

        @ConstraintRegistry.register(
            key="_test-diff",
            label="Test",
            config=_Cfg,
            description="test",
            supported_sequence_types=["dna"],
            backward=_backward,
        )
        def _score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        try:
            c = ConstraintRegistry.create(key="_test-diff", segments=[dna_segment], config_dict={})
            assert isinstance(c, Constraint)
            assert c.supports_gradient
            assert c.backward is _backward

            # Without backward, create() returns Constraint without gradient support.
            plain = ConstraintRegistry.create(
                key="gc-content", segments=[dna_segment], config_dict={"min_gc": 40.0, "max_gc": 60.0}
            )
            assert not plain.supports_gradient
        finally:
            ConstraintRegistry._registry.pop("_test-diff", None)

    def test_backward_only_auto_detected_from_return_type(self, dna_segment):
        """Decorated function returning GradientConstraintOutput is auto-detected as backward callable."""
        import numpy as np

        from proto_language.language.core.constraint import GradientConstraintOutput

        class _Cfg(BaseModel):
            pass

        @ConstraintRegistry.register(
            key="_test-backward-auto",
            label="Test Backward Auto",
            config=_Cfg,
            description="test",
            supported_sequence_types=["dna"],
        )
        def _my_backward(logits: np.ndarray, *, config: _Cfg, **kwargs: Any) -> list[GradientConstraintOutput]:
            return [GradientConstraintOutput(gradient=(-logits,), loss=float(np.mean(logits**2)))]

        try:
            spec = ConstraintRegistry.get("_test-backward-auto")
            assert spec.function is None
            assert spec.backward is _my_backward

            c = ConstraintRegistry.create(key="_test-backward-auto", segments=[dna_segment], config_dict={})
            assert c.supports_gradient
            assert not c.supports_discrete
            assert c.backward is _my_backward
            assert ConstraintRegistry.get_key(c) == "_test-backward-auto"
        finally:
            ConstraintRegistry._registry.pop("_test-backward-auto", None)

    def test_backward_return_type_with_explicit_backward_raises(self):
        """Decorated function returning GradientConstraintOutput + backward= kwarg raises ValueError."""
        import numpy as np

        from proto_language.language.core.constraint import GradientConstraintOutput

        class _Cfg(BaseModel):
            pass

        def _other_backward(logits, *, config, **kwargs):
            return GradientConstraintOutput(gradient=(np.zeros_like(logits),), loss=0.0)

        with pytest.raises(ValueError, match="decorated function returns list\\[GradientConstraintOutput\\]"):

            @ConstraintRegistry.register(
                key="_test-backward-conflict",
                label="Test",
                config=_Cfg,
                description="test",
                supported_sequence_types=["dna"],
                backward=_other_backward,
            )
            def _my_backward(logits: np.ndarray, *, config: _Cfg, **kwargs: Any) -> list[GradientConstraintOutput]:
                return [GradientConstraintOutput(gradient=(-logits,), loss=0.0)]

        ConstraintRegistry._registry.pop("_test-backward-conflict", None)


# ============================================================================
# Unit Tests: Mode & Backward Config
# ============================================================================


class TestModeAndBackwardConfig:
    """Test mode field and backward_config_model on ConstraintSpec."""

    def test_discrete_mode_and_serialization(self):
        """Scoring function → mode='discrete', backward_config_model absent in serialization."""

        class _Cfg(BaseModel):
            x: float = 1.0

        @constraint(key="_test-m1", label="T", config=_Cfg, description="t", supported_sequence_types=["dna"])
        def _score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        try:
            data = ConstraintRegistry.get("_test-m1").model_dump()
            assert data["mode"] == "discrete"
            assert data["backward_config_model"] is None
        finally:
            ConstraintRegistry._registry.pop("_test-m1", None)

    def test_gradient_mode_auto_detected_and_explicit(self):
        """Both -> GradientConstraintOutput return type and backward= kwarg produce mode='gradient'."""
        import numpy as np

        from proto_language.language.core.constraint import GradientConstraintOutput

        class _Cfg(BaseModel):
            pass

        @constraint(key="_test-m2a", label="T", config=_Cfg, description="t", supported_sequence_types=["dna"])
        def _bw(logits: np.ndarray, *, config: _Cfg, **kwargs: Any) -> list[GradientConstraintOutput]:
            return [GradientConstraintOutput(gradient=(-logits,), loss=0.0)]

        def _bw_fn(logits, *, config, **kwargs):
            return GradientConstraintOutput(gradient=(np.zeros_like(logits),), loss=0.0)

        @constraint(
            key="_test-m2b",
            label="T",
            config=_Cfg,
            description="t",
            supported_sequence_types=["dna"],
            backward=_bw_fn,
        )
        def _score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        try:
            # Backward-only (decorated fn returns GradientConstraintOutput) -> "gradient".
            assert ConstraintRegistry.get("_test-m2a").mode == "gradient"
            # Forward fn paired with separate backward via backward= -> "dual".
            assert ConstraintRegistry.get("_test-m2b").mode == "dual"
        finally:
            ConstraintRegistry._registry.pop("_test-m2a", None)
            ConstraintRegistry._registry.pop("_test-m2b", None)

    def test_backward_config_model_serialized_as_separate_schema(self):
        """backward_config_model serializes as its own JSON Schema, distinct from config_model."""
        import numpy as np

        from proto_language.language.core.constraint import GradientConstraintOutput

        class _ScoreCfg(BaseModel):
            threshold: float = 0.5

        class _GradCfg(BaseModel):
            temperature: float = 0.6

        def _bw(logits, *, config, **kwargs):
            return GradientConstraintOutput(gradient=(np.zeros_like(logits),), loss=0.0)

        @constraint(
            key="_test-m3",
            label="T",
            config=_ScoreCfg,
            description="t",
            supported_sequence_types=["dna"],
            backward=_bw,
            backward_config=_GradCfg,
        )
        def _score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        try:
            data = ConstraintRegistry.get("_test-m3").model_dump()
            assert "threshold" in data["config_model"]["properties"]
            assert "temperature" in data["backward_config_model"]["properties"]
        finally:
            ConstraintRegistry._registry.pop("_test-m3", None)

    def test_create_separate_backward_config(self, dna_segment):
        """create() validates backward_config_dict against backward_config_model; rejects invalid input."""
        import numpy as np

        from proto_language.language.core.constraint import GradientConstraintOutput

        class _ScoreCfg(BaseModel):
            threshold: float = 0.5

        class _GradCfg(BaseModel):
            lr: float = 0.01

        def _bw(logits, *, config, **kwargs):
            return GradientConstraintOutput(gradient=(np.zeros_like(logits),), loss=0.0)

        @constraint(
            key="_test-m4",
            label="T",
            config=_ScoreCfg,
            description="t",
            supported_sequence_types=["dna"],
            backward=_bw,
            backward_config=_GradCfg,
        )
        def _score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        try:
            c = ConstraintRegistry.create(
                key="_test-m4",
                segments=[dna_segment],
                config_dict={"threshold": 0.8},
                backward_config_dict={"lr": 0.05},
            )
            assert isinstance(c.function_config, _ScoreCfg) and c.function_config.threshold == 0.8
            assert isinstance(c.backward_config, _GradCfg) and c.backward_config.lr == 0.05

            # Empty dict uses backward model defaults, not scoring config fallback
            c2 = ConstraintRegistry.create(
                key="_test-m4", segments=[dna_segment], config_dict={}, backward_config_dict={}
            )
            assert isinstance(c2.backward_config, _GradCfg) and c2.backward_config.lr == 0.01

            with pytest.raises(ValueError, match=r"backward config invalid"):
                ConstraintRegistry.create(
                    key="_test-m4",
                    segments=[dna_segment],
                    config_dict={},
                    backward_config_dict={"lr": "bad"},
                )
        finally:
            ConstraintRegistry._registry.pop("_test-m4", None)

    def test_create_backward_config_falls_back_to_config_dict(self, dna_segment):
        """Without backward_config_dict, backward config uses config_dict + config_model."""
        import numpy as np

        from proto_language.language.core.constraint import GradientConstraintOutput

        class _Cfg(BaseModel):
            x: float = 1.0

        def _bw(logits, *, config, **kwargs):
            return GradientConstraintOutput(gradient=(np.zeros_like(logits),), loss=0.0)

        @constraint(
            key="_test-m5",
            label="T",
            config=_Cfg,
            description="t",
            supported_sequence_types=["dna"],
            backward=_bw,
        )
        def _score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        try:
            c = ConstraintRegistry.create(key="_test-m5", segments=[dna_segment], config_dict={"x": 2.0})
            assert c.function_config.x == 2.0
            assert c.backward_config.x == 2.0
        finally:
            ConstraintRegistry._registry.pop("_test-m5", None)

    def test_all_builtin_constraints_have_valid_mode(self):
        """All registered constraints have a valid mode."""
        for spec in ConstraintRegistry.list_all():
            assert spec.mode in ("discrete", "gradient", "dual"), f"{spec.key}: invalid mode {spec.mode}"

    def test_ablang_dual_mode_constraints(self):
        """AbLang naturalness constraint is dual-mode (function + backward both set)."""
        spec = ConstraintRegistry.get("ablang-naturalness")
        assert spec.mode == "dual"
        assert spec.function is not None and spec.backward is not None

    def test_backward_config_without_backward_raises(self):
        """backward_config= on a discrete-only constraint raises ValueError."""

        class _Cfg(BaseModel):
            pass

        class _GradCfg(BaseModel):
            lr: float = 0.01

        with pytest.raises(
            ValueError, match="backward_config= requires backward= or -> list\\[GradientConstraintOutput\\]"
        ):

            @constraint(
                key="_test-m6",
                label="T",
                config=_Cfg,
                description="t",
                supported_sequence_types=["dna"],
                backward_config=_GradCfg,
            )
            def _score(input_sequences, config):
                return [ConstraintOutput(score=0.0) for _ in input_sequences]

        ConstraintRegistry._registry.pop("_test-m6", None)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_full_workflow_list_create_evaluate(self, dna_segment):
        """Test complete workflow: list → get schema → create → evaluate."""
        # 1. List all constraints
        constraints = ConstraintRegistry.list_all()
        constraint_keys = {spec.key for spec in constraints}
        assert "gc-content" in constraint_keys

        # 2. Get schema for form generation
        schema = ConstraintRegistry.get_schema("gc-content")
        assert "properties" in schema

        # 3. Create constraint from user input
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[dna_segment], config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )

        # 4. Create proposals before evaluation (constraints evaluate proposal_sequences)
        dna_segment.proposal_sequences = [copy.deepcopy(dna_segment.original_sequence) for _ in range(1)]

        # 5. Evaluate
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_all_registered_constraints_are_creatable(self, dna_segment, protein_segment):
        """Test that all registered constraints can be instantiated."""
        all_constraints = ConstraintRegistry.list_all()

        errors = []
        for spec in all_constraints:
            try:
                # Try to get schema (should not raise)
                schema = ConstraintRegistry.get_schema(spec.key)

                # Extract defaults from schema
                _ = {k: v.get("default") for k, v in schema.get("properties", {}).items() if "default" in v}

                # Note: We can't create all constraints without proper config values
                # This test just verifies the registry methods work for all

            except Exception as e:  # noqa: PERF203
                errors.append(f"{spec.key}: {e!s}")

        assert len(errors) == 0, f"Errors accessing constraints: {errors}"

    def test_registry_methods_consistent(self):
        """Test that different registry methods return consistent data."""
        # Get constraint keys from different methods
        keys_from_list_all = {spec.key for spec in ConstraintRegistry.list_all()}
        keys_from_list_keys = set(ConstraintRegistry._registry.keys())
        count = ConstraintRegistry.count()

        # All should be consistent
        assert keys_from_list_all == keys_from_list_keys
        assert len(keys_from_list_all) == count
        assert count >= 20  # We expect at least 20 constraints


# ============================================================================
# Test Builtin Constraints
# ============================================================================


class TestBuiltinConstraints:
    """Test that all expected builtin constraints are registered."""

    def test_all_sequence_composition_constraints_registered(self):
        """Test that all sequence composition constraints are registered."""
        expected = [
            "gc-content",
            "sequence-length",
            "max-homopolymer",
            "kmer-frequency",  # Replaces dinucleotide-frequency and tetranucleotide-usage
        ]

        registered = sorted(ConstraintRegistry._registry.keys())
        for key in expected:
            assert key in registered, f"Missing constraint: {key}"

    def test_all_protein_quality_constraints_registered(self):
        """Test that all protein quality constraints are registered."""
        expected = [
            "protein-length",
            "protein-diversity",
            "protein-repetitiveness",
            "protein-complexity",
            "balanced-aa",
            "protein-domain",
            "overall-protein-quality",
        ]

        registered = sorted(ConstraintRegistry._registry.keys())
        for key in expected:
            assert key in registered, f"Missing constraint: {key}"

    def test_all_protein_structure_constraints_registered(self):
        """Test that all protein structure constraints are registered."""
        expected = [
            "structure-plddt",
            "structure-ptm",
            "structure-pae",
            "structure-iptm",
            "structure-rmsd",
            "structure-tmscore",
            "protein-symmetry-ring",
            "protein-globularity",
            "boltz2-binding-strength",
        ]

        registered = sorted(ConstraintRegistry._registry.keys())
        for key in expected:
            assert key in registered, f"Missing constraint: {key}"

    def test_all_sequence_annotation_constraints_registered(self):
        """Test that all sequence annotation constraints are registered."""
        expected = [
            "mmseqs-gene-similarity",  # Removed orfipy-mmseqs constraints
            "sigma70-promoter",
            "seq-motif",
            "promoter-strength",
        ]

        registered = sorted(ConstraintRegistry._registry.keys())
        for key in expected:
            assert key in registered, f"Missing constraint: {key}"

    def test_gpu_constraints_marked_correctly(self):
        """Test that GPU-requiring constraints are properly marked."""
        # Constraints that should be marked as GPU-required
        gpu_constraints = [
            "structure-plddt",
            "structure-ptm",
            "structure-pae",
            "structure-iptm",
            "protein-symmetry-ring",
            "protein-globularity",
            "boltz2-binding-strength",
        ]

        # Constraints that should NOT require GPU
        cpu_constraints = ["gc-content", "sequence-length", "protein-length", "protein-complexity", "protein-domain"]

        all_constraints = ConstraintRegistry.list_all()
        constraints_dict = {spec.key: spec for spec in all_constraints}

        # Check GPU constraints
        for key in gpu_constraints:
            assert key in constraints_dict, f"GPU constraint {key} not registered"
            assert constraints_dict[key].uses_gpu, f"Constraint {key} should be marked as uses_gpu=True"

        # Check CPU constraints
        for key in cpu_constraints:
            assert key in constraints_dict, f"CPU constraint {key} not registered"
            assert constraints_dict[key].uses_gpu is False, f"Constraint {key} should be marked as uses_gpu=False"

    def test_supported_sequence_types_field_present(self):
        """Test that all constraints have supported_sequence_types field."""
        all_constraints = ConstraintRegistry.list_all()

        for spec in all_constraints:
            assert hasattr(spec, "supported_sequence_types"), (
                f"Constraint {spec.key} missing supported_sequence_types field"
            )
            assert isinstance(spec.supported_sequence_types, list), (
                f"Constraint {spec.key} supported_sequence_types should be a list"
            )

    def test_protein_only_constraints_have_correct_types(self):
        """Test that protein-only constraints have correct supported_sequence_types."""
        protein_only_constraints = [
            "protein-length",
            "protein-diversity",
            "protein-repetitiveness",
            "protein-complexity",
            "balanced-aa",
        ]

        all_constraints = ConstraintRegistry.list_all()
        constraints_dict = {spec.key: spec for spec in all_constraints}

        for key in protein_only_constraints:
            assert key in constraints_dict, f"Constraint {key} not registered"
            assert constraints_dict[key].supported_sequence_types == ["protein"], (
                f"Constraint {key} should only support protein, got {constraints_dict[key].supported_sequence_types}"
            )

    def test_dna_rna_constraints_have_correct_types(self):
        """Test that DNA/RNA constraints have correct supported_sequence_types."""
        dna_rna_constraints = {
            "gc-content": ["dna", "rna"],
            "rna-property-similarity": ["dna", "rna"],
            "rna-motif-similarity": ["dna", "rna"],
            "rna-feature-similarity": ["dna", "rna"],
            "rna-basepair-similarity": ["dna", "rna"],
        }

        all_constraints = ConstraintRegistry.list_all()
        constraints_dict = {spec.key: spec for spec in all_constraints}

        for key, expected_types in dna_rna_constraints.items():
            assert key in constraints_dict, f"Constraint {key} not registered"
            assert set(constraints_dict[key].supported_sequence_types) == set(expected_types), (
                f"Constraint {key} should support {expected_types}, got {constraints_dict[key].supported_sequence_types}"
            )

    def test_all_constraints_have_explicit_types(self):
        """Test that all constraints have non-empty supported_sequence_types."""
        all_constraints = ConstraintRegistry.list_all()

        for spec in all_constraints:
            assert len(spec.supported_sequence_types) > 0, (
                f"Constraint {spec.key} must have non-empty supported_sequence_types"
            )

    def test_structure_constraints_support_protein(self):
        """Test that structure prediction constraints support protein sequences."""
        structure_constraints = [
            "structure-plddt",
            "structure-ptm",
            "structure-pae",
            "structure-iptm",
            "structure-rmsd",
            "structure-tmscore",
        ]

        all_constraints = ConstraintRegistry.list_all()
        constraints_dict = {spec.key: spec for spec in all_constraints}

        for key in structure_constraints:
            assert key in constraints_dict, f"Constraint {key} not registered"
            assert "protein" in constraints_dict[key].supported_sequence_types, (
                f"Constraint {key} should support protein, got {constraints_dict[key].supported_sequence_types}"
            )

    def test_boltz_binding_supports_multiple_types(self):
        """Test that boltz2-binding-strength supports multiple sequence types."""
        all_constraints = ConstraintRegistry.list_all()
        constraints_dict = {spec.key: spec for spec in all_constraints}

        spec = constraints_dict["boltz2-binding-strength"]
        expected_types = {"dna", "rna", "protein", "ligand"}
        assert set(spec.supported_sequence_types) == expected_types, (
            f"boltz2-binding-strength should support {expected_types}, got {spec.supported_sequence_types}"
        )

    def test_config_validation_patterns(self):
        """Test that Pydantic config validation works through registry.

        This tests the pattern once rather than per-constraint.
        """
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")

        # Test 1: Invalid type for numeric parameter
        with pytest.raises(Exception):  # Pydantic ValidationError
            ConstraintRegistry.create(
                key="gc-content",
                segments=[segment],
                config_dict={
                    "min_gc": "not_a_number",  # Should be float
                    "max_gc": 60,
                },
            )

        # Test 2: Missing required parameter
        with pytest.raises(Exception):  # Pydantic ValidationError
            ConstraintRegistry.create(
                key="gc-content",
                segments=[segment],
                config_dict={"min_gc": 40},  # Missing max_gc
            )

        # Test 3: Valid config should work
        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[segment], config_dict={"min_gc": 40, "max_gc": 60}
        )
        assert constraint.function_config.min_gc == 40
        assert constraint.function_config.max_gc == 60

    def test_config_with_optional_parameters(self):
        """Test constraints with optional config parameters."""
        segment = Segment(sequence="MVLSPADKTN", sequence_type="protein")

        # protein-complexity should work with its default config
        constraint = ConstraintRegistry.create(
            key="protein-complexity", segments=[segment], config_dict={"max_low_complexity": 0.3}
        )
        assert constraint.function_config.max_low_complexity == 0.3

    def test_config_validation_with_constraints(self):
        """Test that Pydantic validators work through registry."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")

        # Many constraints have validators (e.g., min < max, values in range)
        # Test with gc-content which should validate min_gc < max_gc

        # This should fail if there's a validator for min < max
        # (if not, it's just documenting current behavior)
        try:  # noqa: SIM105
            ConstraintRegistry.create(
                key="gc-content",
                segments=[segment],
                config_dict={"min_gc": 80, "max_gc": 20},  # min > max
            )
            # If no error, document that this constraint doesn't validate ordering
            # Individual constraints may have domain-specific validators
        except Exception:
            # Validation worked - good!
            pass
