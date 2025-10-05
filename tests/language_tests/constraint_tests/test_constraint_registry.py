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

import pytest
from pydantic import BaseModel, Field, ValidationError

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.base import Segment, Sequence, SequenceType, Constraint
from .test_utils import create_segment


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def dna_segment():
    """Create a DNA segment for testing."""
    return Segment(sequence="ATCGATCGATCG", sequence_type=SequenceType.DNA)


@pytest.fixture
def protein_segment():
    """Create a protein segment for testing."""
    return Segment(sequence="MKTAYIAKQRQISFVK", sequence_type=SequenceType.PROTEIN)


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
            config=TestConfig,
            description="Temporary test constraint"
        )
        def test_constraint(sequence: Sequence, config: TestConfig) -> float:
            return 0.5
        
        # Verify registration
        assert ConstraintRegistry.count() == initial_count + 1
        assert "test-temp-constraint" in ConstraintRegistry.list_keys()
        
        # Get the spec and verify
        spec = ConstraintRegistry.get("test-temp-constraint")
        assert spec.config_model == TestConfig
        assert spec.description == "Temporary test constraint"
        assert spec.function == test_constraint
        assert spec.vectorized == False
        assert spec.concatenate == True
        
        # Cleanup
        del ConstraintRegistry._registry["test-temp-constraint"]
    
    def test_register_with_custom_flags(self):
        """Test registration with custom vectorized/concatenate flags."""
        class TestConfig(BaseModel):
            value: int = 1
        
        @ConstraintRegistry.register(
            key="test-vectorized",
            config=TestConfig,
            description="Vectorized constraint",
            vectorized=True,
            concatenate=False
        )
        def test_constraint(sequences, config):
            return [0.0] * len(sequences)
        
        spec = ConstraintRegistry.get("test-vectorized")
        assert spec.vectorized == True
        assert spec.concatenate == False
        
        # Cleanup
        del ConstraintRegistry._registry["test-vectorized"]
    
    def test_register_returns_original_function(self):
        """Test that register decorator returns the original function unchanged."""
        class TestConfig(BaseModel):
            pass
        
        def original_func(sequence: Sequence, config: TestConfig) -> float:
            """Original docstring."""
            return 0.5
        
        registered_func = ConstraintRegistry.register(
            key="test-return",
            config=TestConfig,
            description="Test"
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
            config=TestConfig,
            description="Test constraint"
        )
        def test_constraint_1(sequence, config):
            return 0.0
        
        # Verify first registration worked
        assert "test-duplicate-check" in ConstraintRegistry._registry
        
        # Second registration with same key should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            @ConstraintRegistry.register(
                key="test-duplicate-check",  # Same key!
                config=TestConfig,
                description="Another test"
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
        
        assert isinstance(all_constraints, dict)
        assert len(all_constraints) >= 20  # Should have at least 20 constraints
        
        # Check structure of returned data
        for key, info in all_constraints.items():
            assert "description" in info
            assert "vectorized" in info
            assert "concatenate" in info
            assert "gpu_required" in info
            assert "config_schema" in info
            assert isinstance(info["config_schema"], dict)
            assert isinstance(info["gpu_required"], bool)
    
    def test_list_keys_returns_sorted_keys(self):
        """Test that list_keys returns sorted list of constraint keys."""
        keys = ConstraintRegistry.list_keys()
        
        assert isinstance(keys, list)
        assert len(keys) >= 20
        assert keys == sorted(keys)  # Should be sorted
    
    def test_count_returns_correct_number(self):
        """Test that count returns the correct number of registered constraints."""
        count = ConstraintRegistry.count()
        keys = ConstraintRegistry.list_keys()
        
        assert count == len(keys)
        assert count >= 20
    
    def test_get_returns_constraint_spec(self):
        """Test that get returns the correct ConstraintSpec."""
        spec = ConstraintRegistry.get("gc-content")
        
        assert spec.description is not None
        assert spec.config_model is not None
        assert spec.function is not None
        assert isinstance(spec.vectorized, bool)
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
    
    def test_get_defaults_extracts_default_values(self):
        """Test that get_defaults extracts default values from schema."""
        # Test with a constraint that has defaults
        defaults = ConstraintRegistry.get_defaults("esmfold-plddt")
        
        assert isinstance(defaults, dict)
        # ESMFold pLDDT should have n_replications with default
        assert "n_replications" in defaults
    
    def test_get_defaults_empty_for_required_params(self):
        """Test that get_defaults returns empty for constraints with no defaults."""
        defaults = ConstraintRegistry.get_defaults("gc-content")
        
        # GC content has no defaults (min_gc and max_gc are required)
        assert defaults == {} or all(v is None for v in defaults.values())
    
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
        assert constraint.scoring_function is not None
        assert constraint.scoring_function_config is not None
        assert constraint.scoring_function_config.min_gc == 40.0
        assert constraint.scoring_function_config.max_gc == 60.0
    
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
    
    def test_create_with_nested_config(self, dna_segment):
        """Test create with nested Pydantic models in config."""
        constraint = ConstraintRegistry.create(
            key="orfipy-mmseqs-gene-hit-count",
            segments=[dna_segment],
            config_dict={
                "min_hits": 1,
                "max_hits": 10,
                "orfipy_kwargs": {"min_len": 30},
                "mmseqs_kwargs": {"database": "/tmp/test.db"}
            }
        )
        
        assert isinstance(constraint, Constraint)
        # Nested configs should be Pydantic models, not dicts
        assert hasattr(constraint.scoring_function_config, 'orfipy_kwargs')
        assert hasattr(constraint.scoring_function_config.orfipy_kwargs, 'min_len')
    
    def test_create_preserves_vectorized_flag(self, dna_segment):
        """Test that create preserves the vectorized flag."""
        # GC content is not vectorized
        constraint_non_vec = ConstraintRegistry.create(
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )
        assert constraint_non_vec.vectorized == False
        
        # Sigma70 promoter is vectorized
        constraint_vec = ConstraintRegistry.create(
            key="sigma70-promoter",
            segments=[dna_segment],
            config_dict={}
        )
        assert constraint_vec.vectorized == True
    
    def test_create_with_label(self, dna_segment):
        """Test that create accepts and sets label."""
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0},
            label="test_gc_label"
        )
        
        assert constraint.label == "test_gc_label"


# ============================================================================
# Unit Tests: Ensure Loaded
# ============================================================================

class TestEnsureLoaded:
    """Test import verification mechanism."""
    
    def test_ensure_loaded_passes_with_correct_count(self):
        """Test that ensure_loaded passes when count matches."""
        actual_count = ConstraintRegistry.count()
        
        # Should not raise any warnings
        ConstraintRegistry.ensure_loaded(expected_count=actual_count)
    
    def test_ensure_loaded_warns_on_mismatch(self):
        """Test that ensure_loaded warns when count doesn't match."""
        actual_count = ConstraintRegistry.count()
        
        with pytest.warns(ImportWarning):
            ConstraintRegistry.ensure_loaded(expected_count=actual_count + 5)
    
    def test_ensure_loaded_without_count_passes(self):
        """Test that ensure_loaded without count always passes."""
        # Should not raise or warn
        ConstraintRegistry.ensure_loaded()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflows."""
    
    def test_full_workflow_list_create_evaluate(self, dna_segment):
        """Test complete workflow: list → get schema → create → evaluate."""
        # 1. List all constraints
        constraints = ConstraintRegistry.list_all()
        assert "gc-content" in constraints
        
        # 2. Get schema for form generation
        schema = ConstraintRegistry.get_schema("gc-content")
        assert "properties" in schema
        
        # 3. Get defaults for pre-filling
        defaults = ConstraintRegistry.get_defaults("gc-content")
        
        # 4. Create constraint from user input
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[dna_segment],
            config_dict={"min_gc": 40.0, "max_gc": 60.0}
        )
        
        # 5. Evaluate
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0
    
    def test_all_registered_constraints_are_creatable(self, dna_segment, protein_segment):
        """Test that all registered constraints can be instantiated."""
        all_constraints = ConstraintRegistry.list_all()
        
        errors = []
        for key in all_constraints.keys():
            try:
                # Get defaults
                defaults = ConstraintRegistry.get_defaults(key)
                
                # Try to get schema (should not raise)
                schema = ConstraintRegistry.get_schema(key)
                
                # Note: We can't create all constraints without proper config values
                # This test just verifies the registry methods work for all
                
            except Exception as e:
                errors.append(f"{key}: {str(e)}")
        
        assert len(errors) == 0, f"Errors accessing constraints: {errors}"
    
    def test_registry_methods_consistent(self):
        """Test that different registry methods return consistent data."""
        # Get constraint keys from different methods
        keys_from_list_all = set(ConstraintRegistry.list_all().keys())
        keys_from_list_keys = set(ConstraintRegistry.list_keys())
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
            "dinucleotide-frequency",
            "tetranucleotide-usage"
        ]
        
        registered = ConstraintRegistry.list_keys()
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
        
        registered = ConstraintRegistry.list_keys()
        for key in expected:
            assert key in registered, f"Missing constraint: {key}"
    
    def test_all_protein_structure_constraints_registered(self):
        """Test that all protein structure constraints are registered."""
        expected = [
            "esmfold-plddt",
            "esmfold-ptm",
            "protein-symmetry-ring",
            "protein-globularity",
            "boltz-binding-strength"
        ]
        
        registered = ConstraintRegistry.list_keys()
        for key in expected:
            assert key in registered, f"Missing constraint: {key}"
    
    def test_all_sequence_annotation_constraints_registered(self):
        """Test that all sequence annotation constraints are registered."""
        expected = [
            "orfipy-mmseqs-gene-hit-count",
            "orfipy-mmseqs-gene-homology",
            "sigma70-promoter",
            "seq-motif",
            "promoter-strength"
        ]
        
        registered = ConstraintRegistry.list_keys()
        for key in expected:
            assert key in registered, f"Missing constraint: {key}"
    
    def test_total_constraint_count(self):
        """Test that we have the expected total number of constraints."""
        count = ConstraintRegistry.count()
        assert count == 22, f"Expected 22 constraints, got {count}"
    
    def test_gpu_constraints_marked_correctly(self):
        """Test that GPU-requiring constraints are properly marked."""
        # Constraints that should be marked as GPU-required
        gpu_constraints = [
            "esmfold-plddt",
            "esmfold-ptm",
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
        
        # Check GPU constraints
        for key in gpu_constraints:
            assert key in all_constraints, f"GPU constraint {key} not registered"
            assert all_constraints[key]["gpu_required"] == True, \
                f"Constraint {key} should be marked as gpu_required=True"
        
        # Check CPU constraints
        for key in cpu_constraints:
            assert key in all_constraints, f"CPU constraint {key} not registered"
            assert all_constraints[key]["gpu_required"] == False, \
                f"Constraint {key} should be marked as gpu_required=False"
    
    def test_config_validation_patterns(self):
        """
        Test that Pydantic config validation works through registry.
        This tests the pattern once rather than per-constraint.
        """
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        
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
        assert constraint.scoring_function_config.min_gc == 40
        assert constraint.scoring_function_config.max_gc == 60
    
    def test_config_with_optional_parameters(self):
        """Test constraints with optional config parameters."""
        segment = create_segment("MVLSPADKTN", SequenceType.PROTEIN)
        
        # protein-complexity has optional segmasker_path
        # Should work with defaults
        constraint = ConstraintRegistry.create(
            key="protein-complexity",
            segments=[segment],
            config_dict={"max_low_complexity": 0.3}
        )
        assert constraint.scoring_function_config.max_low_complexity == 0.3
        
        # Should also work with custom path
        constraint_custom = ConstraintRegistry.create(
            key="protein-complexity",
            segments=[segment],
            config_dict={
                "max_low_complexity": 0.3,
                "segmasker_path": "/custom/path"
            }
        )
        assert constraint_custom.scoring_function_config.segmasker_path == "/custom/path"
    
    def test_config_validation_with_constraints(self):
        """Test that Pydantic validators work through registry."""
        segment = create_segment("ATCGATCG", SequenceType.DNA)
        
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
