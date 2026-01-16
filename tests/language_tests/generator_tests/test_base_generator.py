from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proto_language.language.core import Generator, Segment
from proto_language.language.generator.generator_registry import (
    GeneratorRegistry,
    GeneratorSpec,
)


# Concrete implementation for testing the abstract base class
class ConcreteGenerator(Generator):
    """Concrete implementation of Generator for testing purposes."""

    def __init__(self) -> None:
        """Minimal implementation of abstract __init__."""
        super().__init__()

    def sample(self) -> None:
        """Dummy sample implementation that does nothing."""
        pass


class TestGeneratorBase:
    """Tests for the base Generator class functionality."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Tests that Generator cannot be directly instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Generator()

    def test_concrete_implementation_initializes(self):
        """Tests that a concrete implementation initializes correctly."""
        gen = ConcreteGenerator()
        assert gen._assigned_segment is None

    def test_assign_sets_segment(self):
        """Tests that assign correctly sets the assigned segment."""
        gen = ConcreteGenerator()
        segment = Segment(sequence="ATCG", sequence_type="dna")

        # Mock the registry lookup to return a spec that allows DNA
        mock_spec = MagicMock(spec=GeneratorSpec)
        mock_spec.supported_sequence_types = []  # Empty means all types supported
        mock_spec.category = "mutation"

        with patch.object(GeneratorRegistry, "get", return_value=mock_spec):
            with patch.object(
                GeneratorRegistry, "get_key", return_value="concrete-generator"
            ):
                gen.assign(segment)

        assert gen._assigned_segment is segment

    def test_assign_rejects_ligand_segment(self):
        """Tests that assign raises error for ligand segments."""
        gen = ConcreteGenerator()
        segment = Segment(sequence="CCC", sequence_type="ligand")

        with pytest.raises(
            ValueError, match="Cannot assign generator to ligand segment"
        ):
            gen.assign(segment)

    def test_assign_validates_sequence_type(self):
        """Tests that assign validates sequence type against supported types."""
        gen = ConcreteGenerator()
        segment = Segment(sequence="ATCG", sequence_type="dna")

        # Mock the registry to return a spec that only supports protein
        mock_spec = MagicMock(spec=GeneratorSpec)
        mock_spec.supported_sequence_types = ["protein"]
        mock_spec.category = "mutation"

        with patch.object(GeneratorRegistry, "get", return_value=mock_spec):
            with patch.object(
                GeneratorRegistry, "get_key", return_value="concrete-generator"
            ):
                with pytest.raises(ValueError, match="does not support sequence type"):
                    gen.assign(segment)

    def test_assign_allows_all_types_when_supported_empty(self):
        """Tests that assign allows any sequence type when supported_sequence_types is empty."""
        gen = ConcreteGenerator()

        mock_spec = MagicMock(spec=GeneratorSpec)
        mock_spec.supported_sequence_types = []  # Empty means all types supported
        mock_spec.category = "mutation"

        with patch.object(GeneratorRegistry, "get", return_value=mock_spec):
            with patch.object(
                GeneratorRegistry, "get_key", return_value="concrete-generator"
            ):
                # Should work for DNA
                segment_dna = Segment(sequence="ATCG", sequence_type="dna")
                gen.assign(segment_dna)
                assert gen._assigned_segment is segment_dna

                # Should work for RNA
                segment_rna = Segment(sequence="ACGU", sequence_type="rna")
                gen.assign(segment_rna)
                assert gen._assigned_segment is segment_rna

                # Should work for protein
                segment_protein = Segment(sequence="MKKL", sequence_type="protein")
                gen.assign(segment_protein)
                assert gen._assigned_segment is segment_protein

    def test_mutation_generator_initializes_random_sequence_on_sample(self):
        """Tests that mutation generators initialize a random sequence on first sample() if none provided."""
        from proto_language.language.generator import (
            UniformMutationGenerator,
            UniformMutationGeneratorConfig,
        )

        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        seq_len = 20
        segment = Segment(length=seq_len, sequence_type="dna")

        # Ensure segment starts with empty sequence
        assert segment.original_sequence.sequence == ""
        # Segment was created with length, not sequence
        assert not segment.has_original_sequence

        gen.assign(segment)

        # After assign, segment should still have empty sequence (lazy initialization)
        assert segment.original_sequence.sequence == ""

        # Call sample() to trigger lazy initialization
        gen.sample()

        # candidate_sequences should now have a random sequence of the correct length
        assert len(segment.candidate_sequences[0].sequence) == seq_len
        # All characters should be valid DNA nucleotides
        assert all(c in "ACGT" for c in segment.candidate_sequences[0].sequence)
        # has_sequence flag should still be False (segment was created with length)
        # This ensures serialization outputs "length" not "sequence"
        assert not segment.has_original_sequence

    def test_mutation_generator_preserves_existing_sequence(self):
        """Tests that mutation generators preserve existing sequences."""
        from proto_language.language.generator import (
            UniformMutationGenerator,
            UniformMutationGeneratorConfig,
        )

        config = UniformMutationGeneratorConfig(num_mutations=1)
        gen = UniformMutationGenerator(config)
        predefined_seq = "ATCGATCG"
        segment = Segment(sequence=predefined_seq, sequence_type="dna")

        gen.assign(segment)
        gen.sample()

        # Should preserve the original sequence (has_sequence is True)
        assert segment.has_original_sequence
        # The sequence may be mutated but should still have same length
        assert len(segment.candidate_sequences[0].sequence) == len(predefined_seq)

    def test_assign_autoregressive_generator_no_random_init(self):
        """Tests that autoregressive generators don't initialize random sequences."""
        gen = ConcreteGenerator()
        seq_len = 20
        segment = Segment(length=seq_len, sequence_type="dna")

        # Ensure segment starts with empty sequence
        assert segment.original_sequence.sequence == ""

        mock_spec = MagicMock(spec=GeneratorSpec)
        mock_spec.supported_sequence_types = []
        mock_spec.category = (
            "autoregressive"  # Autoregressive generators don't init random
        )

        with patch.object(GeneratorRegistry, "get", return_value=mock_spec):
            with patch.object(
                GeneratorRegistry, "get_key", return_value="concrete-generator"
            ):
                gen.assign(segment)

        # Should still have empty sequence (autoregressive generates from scratch)
        assert segment.original_sequence.sequence == ""


class TestGeneratorRegistry:
    """Tests for GeneratorRegistry functionality."""

    def test_registry_exists(self):
        """Tests that the GeneratorRegistry is available."""
        assert GeneratorRegistry is not None
        assert hasattr(GeneratorRegistry, "_registry")

    def test_registry_has_generators(self):
        """Tests that some generators are registered."""
        # After imports, registry should have generators
        all_generators = GeneratorRegistry.list_all()
        assert len(all_generators) > 0

    def test_get_key_for_unknown_generator_raises(self):
        """Tests that get_key raises for unregistered generators."""
        gen = ConcreteGenerator()  # Not registered

        with pytest.raises(ValueError, match="is not registered"):
            GeneratorRegistry.get_key(gen)

    def test_create_with_invalid_key_raises(self):
        """Tests that create raises for invalid generator key."""
        with pytest.raises(ValueError, match="Unknown generator"):
            GeneratorRegistry.create("nonexistent-generator", {})

    def test_list_all_returns_specs(self):
        """Tests that list_all returns GeneratorSpec instances."""
        all_specs = GeneratorRegistry.list_all()

        for spec in all_specs:
            assert isinstance(spec, GeneratorSpec)
            assert hasattr(spec, "key")
            assert hasattr(spec, "label")
            assert hasattr(spec, "description")
            assert hasattr(spec, "category")
            assert hasattr(spec, "requires_gpu")
            assert hasattr(spec, "supported_sequence_types")
