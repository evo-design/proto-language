"""
Tests for core BeamSearchOptimizer functionality.

These tests verify the behavior of the BeamSearchOptimizer without API dependencies.
"""

from unittest.mock import patch, Mock

# Mock all problematic imports at module level before any other imports
with patch.dict('sys.modules', {
    'google.cloud': Mock(),
    'google.cloud.storage': Mock(),
    'proto_language.file_resolver': Mock(),
    'proto_language.file_utils': Mock(),
    'torch': Mock(),
    'proto_language.tools': Mock(),
    'proto_language.tools.tool_cache': Mock(),
    'proto_language.tools.structure_prediction': Mock(),
    'proto_language.tools.orf_prediction': Mock(),
    'proto_language.tools.orf_prediction.orfipy': Mock(),
    'proto_language.tools.orf_prediction.prodigal': Mock(),
    'proto_language.tools.orf_prediction.genemark': Mock(),
    'proto_language.tools.gene_annotation': Mock(),
    'proto_language.tools.gene_annotation.blast': Mock(),
    'proto_language.tools.gene_annotation.hmmer': Mock(),
    'proto_language.tools.gene_annotation.mmseqs': Mock(),
    'proto_language.tools.blast': Mock(),
    'proto_language.tools.hmmer': Mock(),
    'proto_language.tools.mmseqs': Mock(),
    'proto_language.tools.structure_prediction.esmfold': Mock(),
    'proto_language.tools.structure_prediction.esm2': Mock(),
    'proto_language.tools.structure_prediction.esm3': Mock(),
    'proto_language.tools.structure_prediction.chai': Mock(),
    'proto_language.tools.structure_prediction.boltz': Mock(),
    'proto_language.tools.structure_prediction.io': Mock(),
    'proto_language.tools.structure_prediction.utils': Mock(),
    'proto_language.tools.structure_prediction.visualize': Mock(),
    'deployment.cloud_functions': Mock()
}):
    # Now import the modules safely
    from proto_language.language.core import (
        Construct, Segment, Constraint, SequenceType)
    from proto_language.language.generator import (
        UniformMutationGenerator,
        UniformMutationGeneratorConfig
    )
    from proto_language.language.optimizer import BeamSearchOptimizer, BeamSearchOptimizerConfig
    from examples.scripts.beam_search_utils import MockAutoregressiveGenerator, gc_content_constraint

class TestBeamSearchOptimizer:
    """Test core BeamSearchOptimizer functionality."""

    def test_beam_search_generator_initialization(self):
        """Test basic BeamSearchOptimizer initialization."""
        # Create test segments
        segment1 = Segment(sequence="AAAA", sequence_type=SequenceType.DNA)
        segment2 = Segment(sequence="TTTT", sequence_type=SequenceType.DNA)
        segment3 = Segment(sequence="GGGG", sequence_type=SequenceType.DNA)
        
        # Create construct
        construct = Construct([segment1, segment2, segment3])
        
        # Create generators
        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=4, batch_size=1, num_mutations=1)
        )
        gen1.assign(segment1)
        
        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=4, batch_size=1, num_mutations=1)
        )
        gen2.assign(segment2)
        
        gen3 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=4, batch_size=1, num_mutations=1)
        )
        gen3.assign(segment3)
        
        gen4 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=4, batch_size=1, num_mutations=1)
        )
        gen4.assign(segment3)
        
        # Create constraint
        def gc_content_constraint(sequence):
            gc_count = sequence.sequence.count('G') + sequence.sequence.count('C')
            total_length = len(sequence.sequence)
            if total_length == 0:
                return 100.0
            gc_ratio = (gc_count / total_length) * 100
            return max(0.0, 100.0 - gc_ratio)
        
        constraint = Constraint(
            inputs=[segment1, segment2, segment3],
            scoring_function=gc_content_constraint,
            scoring_function_config={}
        )
        
        # Create beam search optimizer
        beam_gen = BeamSearchOptimizer(
            constructs=[construct],
            generators=[gen1, gen2, gen3, gen4],
            constraints=[constraint],
            config=BeamSearchOptimizerConfig(beam_width=2, num_candidates=3, verbose=False),
        )
        
        # Verify initialization
        assert beam_gen.beam_width == 2
        assert beam_gen.num_candidates == 3
        assert len(beam_gen.constraints) == 1
        assert len(beam_gen.generators) == 4
        
        # Verify segments have correct batch size
        for segment in construct.segments:
            assert len(segment.batch_sequences) == 2  # beam_width

    def test_extension_based_generator_detection(self):
        """Test the _is_extension_based_generator method."""
        # Create a dummy beam search optimizer for testing the method
        # We need at least one construct to initialize BeamSearchOptimizer
        segment = Segment(sequence="AAAA", sequence_type=SequenceType.DNA)
        construct = Construct([segment])
        
        beam_gen = BeamSearchOptimizer(
            constructs=[construct],
            generators=[],
            constraints=[],
            config=BeamSearchOptimizerConfig(beam_width=2, num_candidates=3, verbose=False),
        )
        
        # Test mutation-based generator (no prepend_prompt)
        class MockMutationGenerator:
            def __init__(self):
                pass  # No prepend_prompt attribute
        
        mutation_gen = MockMutationGenerator()
        assert beam_gen._is_extension_based_generator(mutation_gen) == False
        
        # Test extension-based generator (has prepend_prompt)
        extension_gen = MockAutoregressiveGenerator(sequence_length=4, prepend_prompt=True)
        assert beam_gen._is_extension_based_generator(extension_gen) == True
        
        # Test extension-based generator with prepend_prompt=False
        extension_gen_false = MockAutoregressiveGenerator(sequence_length=4, prepend_prompt=False)
        assert beam_gen._is_extension_based_generator(extension_gen_false) == False 
    
    def test_demo_beam_search_scenario(self):
        """Test the exact same scenario as demo_beam_search.py and assert expected final sequences."""
        # Create the same segments as in the demo
        segment1 = Segment(sequence="A", sequence_type=SequenceType.DNA)
        segment2 = Segment(sequence="T", sequence_type=SequenceType.DNA)
        segment3 = Segment(sequence="", sequence_type=SequenceType.DNA)  # No initial prompt
        
        # Create construct with multiple segments
        construct = Construct([segment1, segment2, segment3])
        
        # Create the same generators as in the demo (with same random seeds for reproducibility)
        mock_gen1 = MockAutoregressiveGenerator(sequence_length=1, random_seed=69, prepend_prompt=True)
        mock_gen2 = MockAutoregressiveGenerator(sequence_length=1, random_seed=42, prepend_prompt=True)
        mock_gen3 = MockAutoregressiveGenerator(sequence_length=1, random_seed=123, prepend_prompt=True)
        mock_gen4 = MockAutoregressiveGenerator(sequence_length=1, random_seed=456, prepend_prompt=True)
        
        generators = [mock_gen1, mock_gen2, mock_gen3, mock_gen4]
        
        # Assign generators to segments (same as demo)
        generators[0].assign(segment1)
        generators[1].assign(segment2)
        generators[2].assign(segment3)
        generators[3].assign(segment3)  # Assign to the same segment
        
        # Create the same constraint as in the demo
        constraint = Constraint(
            inputs=[segment1, segment2, segment3],
            scoring_function=gc_content_constraint,
            scoring_function_config={}
        )
        
        # Create beam search optimizer with same parameters
        beam_gen = BeamSearchOptimizer(
            constructs=[construct],
            generators=generators,
            constraints=[constraint],
            config=BeamSearchOptimizerConfig(beam_width=2, num_candidates=3, verbose=True),
        )
        
        # Run the same number of samples as demo default
        num_samples = 2
        for _ in range(num_samples):
            print("BOINK")
            beam_gen.sample()
        
        # Now assert the expected final sequences based on the demo behavior
        # The sequences should be deterministic due to fixed random seeds
        
        # Get final sequences from each segment
        final_segment1_seq = segment1.batch_sequences[0].sequence
        final_segment2_seq = segment2.batch_sequences[0].sequence
        final_segment3_seq = segment3.batch_sequences[0].sequence
        
        # Assert expected sequences based on the demo's deterministic behavior
        # These are the expected sequences after 2 samples with the given random seeds
        
        # Segment 1: Starts with "A", generates 1 token per sample, 2 samples total
        # Expected: "A" + 2 generated tokens
        assert len(final_segment1_seq) == 3, f"Segment 1 should have length 3, got {len(final_segment1_seq)}"
        assert final_segment1_seq.startswith("A"), f"Segment 1 should start with 'A', got '{final_segment1_seq}'"
        
        # Segment 2: Starts with "T", generates 1 token per sample, 2 samples total  
        # Expected: "T" + 2 generated tokens
        assert len(final_segment2_seq) == 3, f"Segment 2 should have length 3, got {len(final_segment2_seq)}"
        assert final_segment2_seq.startswith("T"), f"Segment 2 should start with 'T', got '{final_segment2_seq}'"
        
        # Segment 3: Starts empty, has 2 generators, generates 1 token per sample, 2 samples total
        # Expected: 4 generated tokens (from 2 generators × 2 samples)
        assert len(final_segment3_seq) == 4, f"Segment 3 should have length 4, got {len(final_segment3_seq)}"
        
        # Verify that all sequences contain only valid DNA characters
        valid_dna_chars = set("ATCG")
        for seq in [final_segment1_seq, final_segment2_seq, final_segment3_seq]:
            assert all(char in valid_dna_chars for char in seq), f"Sequence '{seq}' contains invalid DNA characters"
        
        # Verify that the beam search maintained the correct batch size
        assert len(segment1.batch_sequences) == 2, f"Segment 1 should have beam_width=2 sequences, got {len(segment1.batch_sequences)}"
        assert len(segment2.batch_sequences) == 2, f"Segment 2 should have beam_width=2 sequences, got {len(segment2.batch_sequences)}"
        assert len(segment3.batch_sequences) == 2, f"Segment 3 should have beam_width=2 sequences, got {len(segment3.batch_sequences)}"
        
        # Verify that all sequences in the beam have the same length
        for segment in [segment1, segment2, segment3]:
            seq_lengths = [len(seq.sequence) for seq in segment.batch_sequences]
            assert len(set(seq_lengths)) == 1, f"All sequences in segment should have same length, got lengths: {seq_lengths}"
        
        # Verify the final top 2 concatenated sequences from the beam search
        # These should match the demo output after 2 samples
        beam_candidates = beam_gen._beam_candidates
        assert len(beam_candidates) == 2, f"Should have 2 beam candidates, got {len(beam_candidates)}"
        
        # Get the final concatenated sequences from the beam
        final_beam_sequence_0 = beam_candidates[0]
        final_beam_sequence_1 = beam_candidates[1]
        
        # Verify the sequences have the expected total length
        # Total length = sum of all segment lengths
        expected_total_length = len(final_segment1_seq) + len(final_segment2_seq) + len(final_segment3_seq)
        assert len(final_beam_sequence_0) == expected_total_length, f"Beam sequence 0 should have total length {expected_total_length}, got {len(final_beam_sequence_0)}"
        assert len(final_beam_sequence_1) == expected_total_length, f"Beam sequence 1 should have total length {expected_total_length}, got {len(final_beam_sequence_1)}"
        
        # Verify the sequences start with the expected segment sequences
        assert final_beam_sequence_0.startswith(final_segment1_seq), f"Beam sequence 0 should start with segment 1 sequence '{final_segment1_seq}', got '{final_beam_sequence_0}'"
        assert final_beam_sequence_1.startswith(final_segment1_seq), f"Beam sequence 1 should start with segment 1 sequence '{final_segment1_seq}', got '{final_beam_sequence_1}'"
        
        # Verify the sequences contain the expected segment sequences in order
        # The concatenated sequence should be: segment1 + segment2 + segment3
        expected_concatenated = final_segment1_seq + final_segment2_seq + final_segment3_seq
        assert final_beam_sequence_0 == expected_concatenated or final_beam_sequence_1 == expected_concatenated, f"One beam sequence should match expected concatenated sequence '{expected_concatenated}'"
        
        # NOTE: The following exact string matches could be brittle if the MockAutoregressiveGenerator 
        # implementation changes or if there are other sources of randomness in the beam search process.
        # These are based on the current test run with fixed random seeds (69, 42, 123, 456).
        # If the test becomes flaky, consider removing these exact matches and keeping only the structural assertions above.
        expected_beam_0 = "ACGTCGGCCG"
        expected_beam_1 = "ACGTCGGCGG"
        
        assert final_beam_sequence_0 == expected_beam_0, f"Beam 0 should be '{expected_beam_0}', got '{final_beam_sequence_0}'"
        assert final_beam_sequence_1 == expected_beam_1, f"Beam 1 should be '{expected_beam_1}', got '{final_beam_sequence_1}'"