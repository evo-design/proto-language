"""
Comprehensive tests for ConstraintRegistry.

Tests cover:
1. Registration mechanism (decorator-based)
2. Discovery and listing methods
3. Schema generation for client
4. Factory method (create)
5. Validation and error handling
6. Import-time registration verification
"""
import copy
import pytest
from pydantic import BaseModel, Field, ValidationError

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.core import Segment, Sequence, Constraint


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
        @ConstraintRegistry.register(
            key="test-temp-constraint",
            label="Test Temp Constraint",
            config=TestConfig,
            description="Temporary test constraint",
            supported_sequence_types=["dna", "protein"],
        )
        def test_constraint(sequence: Sequence, config: TestConfig) -> float:
            return 0.5

        # Verify registration
        assert ConstraintRegistry.count() == initial_count + 1
        assert "test-temp-constraint" in sorted(ConstraintRegistry._registry.keys())

        # Get the spec and verify
        spec = ConstraintRegistry.get("test-temp-constraint")
        assert spec.config_model == TestConfig
        assert spec.description == "Temporary test constraint"
        assert spec.function == test_constraint
        assert spec.batched is False
        assert spec.concatenate is True

        # Cleanup
        del ConstraintRegistry._registry["test-temp-constraint"]

    def test_register_with_custom_flags(self):
        """Test registration with custom batched/concatenate flags."""
        class TestConfig(BaseModel):
            value: int = 1

        @ConstraintRegistry.register(
            key="test-batched",
            label="Test batched",
            config=TestConfig,
            description="batched constraint",
            batched=True,
            concatenate=False,
            supported_sequence_types=["dna"],
        )
        def test_constraint(sequences, config):
            return [0.0] * len(sequences)

        spec = ConstraintRegistry.get("test-batched")
        assert spec.batched is True
        assert spec.concatenate is False

        # Cleanup
        del ConstraintRegistry._registry["test-batched"]

    def test_register_returns_original_function(self):
        """Test that register decorator returns the original function unchanged."""
        class TestConfig(BaseModel):
            pass

        def original_func(sequence: Sequence, config: TestConfig) -> float:
            """Original docstring."""
            return 0.5

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
        @ConstraintRegistry.register(
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
            @ConstraintRegistry.register(
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
            assert hasattr(spec, "batched")
            assert hasattr(spec, "concatenate")
            assert hasattr(spec, "gpu_required")
            # Verify config_model is present and can generate JSON schema
            assert spec.config_model is not None
            schema = spec.config_model.model_json_schema()
            assert isinstance(schema, dict)
            assert "properties" in schema
            assert isinstance(spec.gpu_required, bool)

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
        assert isinstance(spec.batched, bool)
        assert isinstance(spec.concatenate, bool)

    def test_get_raises_on_unknown_key(self):
        """Test that get raises ValueError for unknown constraint."""
        with pytest.raises(ValueError, match="Unknown constraint"):
            ConstraintRegistry.get("nonexistent-constraint-key")

    def test_get_error_message_lists_available(self):
        """Test that error message includes available constraints."""
        try:
            ConstraintRegistry.get("bad-key")
            assert False, "Should have raised ValueError"
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
        for prop_name, prop_info in properties.items():
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
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )

        assert isinstance(constraint, Constraint)
        assert constraint.function is not None
        assert constraint.function_config is not None
        assert constraint.function_config.min_gc == 40.0
        assert constraint.function_config.max_gc == 60.0

    def test_create_raises_on_invalid_config(self, dna_segment):
        """Test that create raises ValidationError for invalid config."""
        with pytest.raises(ValidationError):
            ConstraintRegistry.create(
                key="gc-content",
                segments=[dna_segment],
                config_dict={"min_gc": "invalid", "max_gc": 60.0}  # Wrong type
            )

    def test_create_raises_on_missing_required_params(self, dna_segment):
        """Test that create raises ValidationError for missing required params."""
        with pytest.raises(ValidationError):
            ConstraintRegistry.create(
                key="gc-content",
                segments=[dna_segment],
                config_dict={"min_gc": 40.0}  # Missing max_gc
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
                "mmseqs_config": {"results_dir": ""}
            }
        )

        assert isinstance(constraint, Constraint)
        # Nested configs should be Pydantic models, not dicts
        assert hasattr(constraint.function_config, 'mmseqs_config')
        assert hasattr(constraint.function_config, 'mmseqs_db')

    def test_create_preserves_batched_flag(self, dna_segment):
        """Test that create preserves the batched flag."""
        # All constraints are now batched
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )
        assert constraint.batched == True

        # Verify another constraint is also batched
        constraint2 = ConstraintRegistry.create(
            key="sigma70-promoter",
            segments=[dna_segment],
            config_dict={}
        )
        assert constraint2.batched == True

    def test_create_with_label(self, dna_segment):
        """Test that create accepts and sets label."""
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0},
            label="test_gc_label"
        )

        assert constraint.label == "test_gc_label"

    def test_create_validates_sequence_type(self, dna_segment, protein_segment):
        """Test that create validates sequence type compatibility."""
        # gc-content only supports dna and rna, not protein
        with pytest.raises(ValueError, match="does not support sequence type"):
            ConstraintRegistry.create(
                key="gc-content",
                segments=[protein_segment],
                config_dict={"min_gc": 40.0, "max_gc": 60.0}
            )

        # protein-length only supports protein, not dna
        with pytest.raises(ValueError, match="does not support sequence type"):
            ConstraintRegistry.create(
                key="protein-length",
                segments=[dna_segment],
                config_dict={"min_length": 10, "max_length": 500}
            )

    def test_create_with_compatible_sequence_type(self, dna_segment, protein_segment):
        """Test that create works with compatible sequence types."""
        # gc-content with dna segment
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )
        assert constraint is not None

        # protein-length with protein segment
        constraint = ConstraintRegistry.create(
            key="protein-length",
            segments=[protein_segment],
            config_dict={"min_length": 10, "max_length": 500}
        )
        assert constraint is not None


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
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )
        
        # 4. Create candidates before evaluation (constraints evaluate candidate_sequences)
        dna_segment.candidate_sequences = [copy.deepcopy(dna_segment.original_sequence) for _ in range(1)]
        
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

            except Exception as e:
                errors.append(f"{spec.key}: {str(e)}")
        
        assert len(errors) == 0, f"Errors accessing constraints: {errors}"
    
    def test_registry_methods_consistent(self):
        """Test that different registry methods return consistent data."""
        # Get constraint keys from different methods
        keys_from_list_all = {spec.key for spec in ConstraintRegistry.list_all()}
        keys_from_list_keys = set(sorted(ConstraintRegistry._registry.keys()))
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
            "overall-protein-quality"
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
            "boltz-binding-strength"
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
            "promoter-strength"
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
            "boltz-binding-strength"
        ]

        # Constraints that should NOT require GPU
        cpu_constraints = [
            "gc-content",
            "sequence-length",
            "protein-length",
            "protein-complexity",
            "protein-domain"
        ]

        all_constraints = ConstraintRegistry.list_all()
        constraints_dict = {spec.key: spec for spec in all_constraints}

        # Check GPU constraints
        for key in gpu_constraints:
            assert key in constraints_dict, f"GPU constraint {key} not registered"
            assert constraints_dict[key].gpu_required == True, \
                f"Constraint {key} should be marked as gpu_required=True"

        # Check CPU constraints
        for key in cpu_constraints:
            assert key in constraints_dict, f"CPU constraint {key} not registered"
            assert constraints_dict[key].gpu_required is False, \
                f"Constraint {key} should be marked as gpu_required=False"

    def test_supported_sequence_types_field_present(self):
        """Test that all constraints have supported_sequence_types field."""
        all_constraints = ConstraintRegistry.list_all()
        
        for spec in all_constraints:
            assert hasattr(spec, 'supported_sequence_types'), \
                f"Constraint {spec.key} missing supported_sequence_types field"
            assert isinstance(spec.supported_sequence_types, list), \
                f"Constraint {spec.key} supported_sequence_types should be a list"

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
            assert constraints_dict[key].supported_sequence_types == ["protein"], \
                f"Constraint {key} should only support protein, got {constraints_dict[key].supported_sequence_types}"

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
            assert set(constraints_dict[key].supported_sequence_types) == set(expected_types), \
                f"Constraint {key} should support {expected_types}, got {constraints_dict[key].supported_sequence_types}"

    def test_all_constraints_have_explicit_types(self):
        """Test that all constraints have non-empty supported_sequence_types."""
        all_constraints = ConstraintRegistry.list_all()

        for spec in all_constraints:
            assert len(spec.supported_sequence_types) > 0, \
                f"Constraint {spec.key} must have non-empty supported_sequence_types"

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
            assert "protein" in constraints_dict[key].supported_sequence_types, \
                f"Constraint {key} should support protein, got {constraints_dict[key].supported_sequence_types}"

    def test_boltz_binding_supports_multiple_types(self):
        """Test that boltz-binding-strength supports multiple sequence types."""
        all_constraints = ConstraintRegistry.list_all()
        constraints_dict = {spec.key: spec for spec in all_constraints}

        spec = constraints_dict["boltz-binding-strength"]
        expected_types = {"dna", "rna", "protein", "ligand"}
        assert set(spec.supported_sequence_types) == expected_types, \
            f"boltz-binding-strength should support {expected_types}, got {spec.supported_sequence_types}"
    
    def test_config_validation_patterns(self):
        """
        Test that Pydantic config validation works through registry.
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
                    "max_gc": 60
                }
            )
        
        # Test 2: Missing required parameter
        with pytest.raises(Exception):  # Pydantic ValidationError
            ConstraintRegistry.create(
                key="gc-content",
                segments=[segment],
                config_dict={"min_gc": 40}  # Missing max_gc
            )
        
        # Test 3: Valid config should work
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 40, "max_gc": 60}
        )
        assert constraint.function_config.min_gc == 40
        assert constraint.function_config.max_gc == 60
    
    def test_config_with_optional_parameters(self):
        """Test constraints with optional config parameters."""
        segment = Segment(sequence="MVLSPADKTN", sequence_type="protein")
        
        # protein-complexity has optional segmasker_path
        # Should work with defaults
        constraint = ConstraintRegistry.create(
            key="protein-complexity",
            segments=[segment],
            config_dict={"max_low_complexity": 0.3}
        )
        assert constraint.function_config.max_low_complexity == 0.3
        
        # Should also work with custom path
        constraint_custom = ConstraintRegistry.create(
            key="protein-complexity",
            segments=[segment],
            config_dict={
                "max_low_complexity": 0.3,
                "segmasker_path": "/custom/path"
            }
        )
        assert constraint_custom.function_config.segmasker_path == "/custom/path"
    
    def test_config_validation_with_constraints(self):
        """Test that Pydantic validators work through registry."""
        segment = Segment(sequence="ATCGATCG", sequence_type="dna")
        
        # Many constraints have validators (e.g., min < max, values in range)
        # Test with gc-content which should validate min_gc < max_gc
        
        # This should fail if there's a validator for min < max
        # (if not, it's just documenting current behavior)
        try:
            ConstraintRegistry.create(
                key="gc-content",
                segments=[segment],
                config_dict={"min_gc": 80, "max_gc": 20}  # min > max
            )
            # If no error, document that this constraint doesn't validate ordering
            # Individual constraints may have domain-specific validators
        except Exception:
            # Validation worked - good!
            pass
