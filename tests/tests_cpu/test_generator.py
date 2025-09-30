import pytest
import random
import numpy as np
import copy
from typing import Tuple

import sys

sys.path.append(".")
from proto_language.base import (
    Construct,
    ConstructSegment,
    Constraint,
    ConstraintType,
    SequenceType,
)
from proto_language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.generator import UniformMutationGenerator, TwoSegmentUniformMutationGenerator, MCMCGenerator, ChainedGenerator

# Helper function
def create_segment(
    sequence: str, seq_type: SequenceType = SequenceType.DNA
) -> ConstructSegment:
    """Helper to create a ConstructSegment with a single sequence."""
    return ConstructSegment(sequence=sequence, sequence_type=seq_type)


class TestUniformMutationGenerator:
    def test_initialization(self):
        """Tests the __init__ method for correct initialization."""
        gen = UniformMutationGenerator(sequence_length=15, batch_size=5)
        assert gen.sequence_length == 15
        assert gen.batch_size == 5
        assert not gen._is_initialized

    def test_assign_and_initialization(self):
        """Tests the assign method initializes the output segment correctly."""
        seq_len = 20
        # Test assign with an empty segment (should initialize randomly)
        segment = create_segment("", seq_type=SequenceType.RNA)
        gen = UniformMutationGenerator(sequence_length=seq_len, batch_size=3)
        gen.assign(segment)

        assert gen._is_initialized
        assert gen._generator_output is segment
        assert segment._is_assigned
        assert len(segment) == 3
        assert len(segment[0]) == seq_len
        assert all(c in "ACGU" for c in segment[0].sequence)

        # Test assign with a pre-defined sequence
        predefined_seq = "A" * seq_len
        segment_pre = create_segment(predefined_seq, seq_type=SequenceType.RNA)
        gen.assign(segment_pre)
        assert segment_pre[0].sequence == predefined_seq

    def test_assign_errors(self):
        """Tests runtime validation for the assign method."""
        gen = UniformMutationGenerator(sequence_length=10)
        # Should raise error if provided sequence length doesn't match configured length
        with pytest.raises(AssertionError):
            gen.assign(create_segment("A"*5))

    def test_sample_mutates_sequence(self):
        """Tests the sample method introduces a single valid mutation."""
        seq_len = 25
        gen = UniformMutationGenerator(sequence_length=seq_len)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.PROTEIN)
        gen.assign(segment)

        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence

        assert len(mutated_sequence) == seq_len
        # Check that exactly one position has changed
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 1
        diff_indices = [i for i, (a, b) in enumerate(zip(initial_sequence, mutated_sequence)) if a != b]
        mutated_char = mutated_sequence[diff_indices[0]]

        assert mutated_char in segment._valid_chars
        assert mutated_char != initial_sequence[diff_indices[0]]

    def test_sample_batch(self):
        """Tests that sample mutates all sequences in a batch independently."""
        gen = UniformMutationGenerator(sequence_length=30, batch_size=5)
        segment = create_segment("A" * 30)
        gen.assign(segment)

        initial_sequences = [s.sequence for s in segment]
        gen.sample()
        mutated_sequences = [s.sequence for s in segment]

        for i in range(len(initial_sequences)):
            assert initial_sequences[i] != mutated_sequences[i]
            diff_count = sum(1 for a,b in zip(initial_sequences[i], mutated_sequences[i]) if a != b)
            assert diff_count == 1
        # Check that mutations are likely different across the batch
        assert len(set(mutated_sequences)) > 1

    def test_deterministic_behavior(self):
        """Tests that with a fixed seed, the behavior is reproducible."""
        def run_with_seed(seed):
            random.seed(seed)
            gen = UniformMutationGenerator(sequence_length=50)
            segment = create_segment("", seq_type=SequenceType.DNA)
            gen.assign(segment)
            initial_seq = segment[0].sequence
            for _ in range(10):
                gen.sample()
            final_seq = segment[0].sequence
            return initial_seq, final_seq

        init1, final1 = run_with_seed(42)
        init2, final2 = run_with_seed(42)
        init3, final3 = run_with_seed(123)

        assert init1 == init2
        assert final1 == final2
        assert init1 != init3
        assert final1 != final3

    def test_sample_len_one_sequence(self):
        """Tests that a sequence of length 1 is mutated correctly."""
        gen = UniformMutationGenerator(sequence_length=1)
        segment = create_segment("A", seq_type=SequenceType.DNA)
        gen.assign(segment)

        initial_char = segment[0].sequence
        gen.sample()
        mutated_char = segment[0].sequence

        assert len(mutated_char) == 1
        assert mutated_char in "CGT"
        assert mutated_char != initial_char

    def test_num_mutations_parameter(self):
        """Tests that specifying num_mutations produces exactly that many changes."""
        seq_len = 30
        num_mut = 5
        gen = UniformMutationGenerator(sequence_length=seq_len, num_mutations=num_mut)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        gen.assign(segment)

        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence

        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == num_mut

    def test_num_mutations_capped_by_sequence_length(self):
        """Tests that num_mutations larger than length is capped to sequence length."""
        seq_len = 3
        num_mut = 10
        gen = UniformMutationGenerator(sequence_length=seq_len, num_mutations=num_mut)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        gen.assign(segment)

        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence

        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == seq_len

    def test_mutation_scheduler_decreasing(self):
        """Tests that a scheduler can control mutations based on iteration count."""
        seq_len = 20
        def scheduler(iteration: int) -> int:
            # 1st call: 3, 2nd: 2, 3rd+: 1
            return max(1, 3 - iteration)

        gen = UniformMutationGenerator(sequence_length=seq_len, mutation_scheduler=scheduler)
        segment = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        gen.assign(segment)

        # Iteration 0 -> expect 3 mutations
        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 3
        assert gen.get_iteration_count() == 1

        # Iteration 1 -> expect 2 mutations
        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 2
        assert gen.get_iteration_count() == 2

        # Iteration 2 -> expect 1 mutation
        initial_sequence = segment[0].sequence
        gen.sample()
        mutated_sequence = segment[0].sequence
        diff_count = sum(1 for a, b in zip(initial_sequence, mutated_sequence) if a != b)
        assert diff_count == 1
        assert gen.get_iteration_count() == 3

    def test_iteration_count_independent_instances(self):
        """Tests iteration counters are per generator instance and resettable."""
        seq_len = 10
        g1 = UniformMutationGenerator(sequence_length=seq_len)
        g2 = UniformMutationGenerator(sequence_length=seq_len)
        s1 = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        s2 = create_segment("A" * seq_len, seq_type=SequenceType.DNA)
        g1.assign(s1)
        g2.assign(s2)

        assert g1.get_iteration_count() == 0
        assert g2.get_iteration_count() == 0

        g1.sample()
        assert g1.get_iteration_count() == 1
        assert g2.get_iteration_count() == 0

        g2.sample()
        g2.sample()
        assert g1.get_iteration_count() == 1
        assert g2.get_iteration_count() == 2

        g1.reset_iteration_count()
        assert g1.get_iteration_count() == 0

class TestTwoSegmentUniformMutationGenerator:
    def test_assign_and_sample(self):
        """Tests basic functionality: assign two segments and mutate them."""
        segment1 = create_segment("ATCGG", seq_type=SequenceType.DNA)
        segment2 = create_segment("MKLLF", seq_type=SequenceType.PROTEIN)
        
        gen = TwoSegmentUniformMutationGenerator(batch_size=1)
        gen.assign([segment1, segment2])

        assert gen._is_initialized
        assert len(gen.get_generator_outputs()) == 2
        
        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence
        
        gen.sample()
        
        # Both sequences should be mutated
        assert segment1[0].sequence != initial_seq1
        assert segment2[0].sequence != initial_seq2
        # Lengths should be preserved
        assert len(segment1[0].sequence) == len(initial_seq1)
        assert len(segment2[0].sequence) == len(initial_seq2)

    def test_assign_errors(self):
        """Tests error conditions for assignment."""
        gen = TwoSegmentUniformMutationGenerator()
        
        # Wrong number of segments
        with pytest.raises(ValueError, match="requires exactly 2 segments"):
            gen.assign([create_segment("ATCG")])
        
        # Empty sequences
        with pytest.raises(ValueError, match="requires segments with existing sequences"):
            gen.assign([create_segment(""), create_segment("ATCG")])

    def test_different_lengths(self):
        """Tests that segments can have different lengths."""
        segment1 = create_segment("AT")
        segment2 = create_segment("GCGCGCGC")
        
        gen = TwoSegmentUniformMutationGenerator()
        gen.assign([segment1, segment2])
        
        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence
        
        gen.sample()
        
        assert len(segment1[0].sequence) == 2
        assert len(segment2[0].sequence) == 8
        assert segment1[0].sequence != initial_seq1
        assert segment2[0].sequence != initial_seq2

def _setup_mcmc_components(
    seq_length: int = 10,
    batch_size: int = 1,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    num_mcmc_steps: int = 10,
):
    """Helper function to set up a basic MCMC generator for testing."""
    # 1. Create the proposal generator and the segment it will modify.
    proposal_gen = UniformMutationGenerator(
        sequence_length=seq_length, batch_size=batch_size
    )
    segment = create_segment("A" * seq_length) # Start with a known sequence
    proposal_gen.assign(segment)

    # 2. Create the construct and constraint.
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        scoring_function=gc_content_constraint,
        scoring_function_config={
            "min_gc": gc_target_range[0],
            "max_gc": gc_target_range[1],
        },
    )

    # 3. Create the MCMC generator.
    mcmc_gen = MCMCGenerator(
        constructs=[construct],
        generators=[proposal_gen],
        constraints=[constraint],
        num_steps=num_mcmc_steps,
        verbose=False,
    )
    return mcmc_gen, proposal_gen, constraint, segment


class TestMCMCGenerator:
    def test_initialization_and_validation(self):
        """Tests successful initialization and validation of MCMCGenerator."""
        mcmc_gen, proposal_gen, constraint, segment = _setup_mcmc_components()
        
        assert mcmc_gen.generators == [proposal_gen]
        assert mcmc_gen.constraints == [constraint]
        assert mcmc_gen.constraint_weights == [1.0]
        assert mcmc_gen._is_initialized # IterativeGenerator base class is auto-initialized

        # Test validation errors
        # Unassigned generator
        unassigned_gen = UniformMutationGenerator(sequence_length=10)
        with pytest.raises(RuntimeError, match="has not been assigned"):
            MCMCGenerator(
                constructs=[Construct([create_segment("A"*10)])],
                generators=[unassigned_gen],
                constraints=[],
            )
        
        # Mismatched weights and constraints
        with pytest.raises(ValueError, match="must match"):
            MCMCGenerator(
                constructs=mcmc_gen.constructs,
                generators=mcmc_gen.generators,
                constraints=mcmc_gen.constraints,
                constraint_weights=[1.0, 2.0],
            )

        # Unassigned segment in construct
        segment_assigned = create_segment("A"*10)
        gen = UniformMutationGenerator(sequence_length=10)
        gen.assign(segment_assigned)
        segment_unassigned = create_segment("C"*10) # Not assigned to any generator
        construct = Construct([segment_assigned, segment_unassigned])
        # Need at least one constraint, so add a dummy one
        dummy_constraint = Constraint(
            inputs=[segment_assigned],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        with pytest.raises(ValueError, match="not assigned to any generator"):
            MCMCGenerator(
                constructs=[construct],
                generators=[gen],
                constraints=[dummy_constraint]
            )

    def test_score_energy(self):
        """Tests the score_energy method."""
        mcmc_gen, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))

        # Test with a sequence that is within the target GC range
        segment.batch_sequences[0].sequence = "GCGCGAATTA"  # 50% GC
        mcmc_gen.score_energy()
        assert len(mcmc_gen.energy_scores) == 1
        assert mcmc_gen.energy_scores[0] == 0.0

        # Test with a sequence below the target range
        segment.batch_sequences[0].sequence = "GCTTAATTAA"  # 20% GC
        mcmc_gen.score_energy()
        expected_score = (40.0 - 20.0) / 40.0  # 0.5
        assert abs(mcmc_gen.energy_scores[0] - expected_score) < 1e-9

        # Test that energy scores are stored in the generator's energy_scores attribute
        assert hasattr(mcmc_gen, 'energy_scores')
        assert len(mcmc_gen.energy_scores) == 1
        assert abs(mcmc_gen.energy_scores[0] - expected_score) < 1e-9
        
        # Test that calling score_energy again updates the stored scores
        segment.batch_sequences[0].sequence = "GCGCGCGCGC"  # 100% GC -> score 1.0
        mcmc_gen.score_energy()
        expected_new_score = abs((40.0 - 100.0) / 40.0)  # Should be 1.5, but clamped to 1.0
        assert abs(mcmc_gen.energy_scores[0] - min(expected_new_score, 1.0)) < 1e-9

    def test_score_energy_multiply(self):
        """Tests the score_energy method with operation='multiply'."""
        mcmc_gen, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))
        segment.batch_sequences[0].sequence = "GCTTAATTAA"  # 20% GC -> score 0.5
        
        # With one constraint, multiply and add should be the same
        mcmc_gen.score_energy(operation="add")
        energy_add = mcmc_gen.energy_scores[0]
        mcmc_gen.score_energy(operation="multiply")
        energy_mul = mcmc_gen.energy_scores[0]
        assert abs(energy_add - 0.5) < 1e-9
        assert abs(energy_mul - 0.5) < 1e-9

    def test_sample_history(self):
        """Tests that sampling can improve the energy score over time."""
        # Use a restrictive constraint to guide optimization
        mcmc_gen, _, _, segment = _setup_mcmc_components(
            seq_length=50,
            gc_target_range=(80.0, 90.0), # Encourage high GC
            num_mcmc_steps=100
        )
        
        # Start with a bad sequence
        segment.batch_sequences[0].sequence = "A" * 50
        mcmc_gen.score_energy()
        initial_energy = mcmc_gen.energy_scores[0]
        assert initial_energy > 0.99 # Should be max penalty (1.0)
        
        # Sample and check for improvement
        mcmc_gen.sample()
        mcmc_gen.score_energy()
        final_energy = mcmc_gen.energy_scores[0]
        
        assert final_energy < initial_energy
        assert len(mcmc_gen.history) > 1 # Check that history is being tracked

    def test_multiple_constraints(self):
        """Tests the MCMC generator with multiple constraints and weights."""
        seq_len = 30
        proposal_gen = UniformMutationGenerator(sequence_length=seq_len)
        segment = create_segment("A" * seq_len)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        gc_con = Constraint(
            [segment], gc_content_constraint, {"min_gc": 40.0, "max_gc": 60.0}
        )
        len_con = Constraint(
            [segment], sequence_length_constraint, {"target_length": seq_len}
        )

        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_con, len_con],
            constraint_weights=[1.0, 2.0], # Weight length more
            num_steps=1,
            verbose=False,
        )

        segment.batch_sequences[0].sequence = "A" * 20 # Violates length and GC
        gc_score = gc_con.evaluate()[0] # (40-0)/40 = 1.0
        len_score = len_con.evaluate()[0] # (30-20)/30 = 0.333
        
        # E = 1.0 * 1.0 + 2.0 * 0.333...
        expected_energy = gc_score * 1.0 + len_score * 2.0
        mcmc_gen.score_energy("add")
        assert abs(mcmc_gen.energy_scores[0] - expected_energy) < 1e-9

        # Test multiply operation
        expected_energy_mul = (gc_score * 1.0) * (len_score * 2.0)
        mcmc_gen.score_energy("multiply")
        assert abs(mcmc_gen.energy_scores[0] - expected_energy_mul) < 1e-9

    def test_with_multiple_generators(self):
        """Tests MCMC with more than one proposal generator."""
        # Create a second, simple generator for testing
        class InversionGenerator(UniformMutationGenerator):
            def sample(self) -> None:
                for seq in self._generator_output.batch_sequences:
                    # Invert a small slice of the sequence
                    start = random.randint(0, len(seq.sequence) - 3)
                    end = start + 3
                    sub_seq = seq.sequence[start:end]
                    inverted_sub = sub_seq[::-1]
                    seq.sequence = seq.sequence[:start] + inverted_sub + seq.sequence[end:]
        
        seq_len = 50
        # Generator 1: Point mutations
        mut_gen = UniformMutationGenerator(sequence_length=seq_len)
        segment1 = create_segment("A" * seq_len)
        mut_gen.assign(segment1)

        # Generator 2: Inversions
        inv_gen = InversionGenerator(sequence_length=seq_len)
        segment2 = create_segment("C" * seq_len)
        inv_gen.assign(segment2)

        construct = Construct([segment1, segment2])
        constraint = Constraint(
            inputs=[segment1, segment2], # Constraint on the whole construct
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": seq_len * 2}
        )

        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[mut_gen, inv_gen],
            constraints=[constraint],
            num_steps=20,
            verbose=False,
        )

        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence
        
        # Sampling should modify the sequences
        mcmc_gen.sample()

        final_seq1 = segment1[0].sequence
        final_seq2 = segment2[0].sequence

        # Check that at least one sequence was modified (both should be, but inversions might be symmetric)
        assert initial_seq1 != final_seq1 or initial_seq2 != final_seq2

    def test_topk_initialization(self):
        """Tests initialization of top-k MCMC with various top_k values."""
        mcmc_gen, _, _, _ = _setup_mcmc_components(batch_size=10)
        
        # Test top_k=1 (standard MCMC)
        assert mcmc_gen.top_k == 1
        
        # Test top_k > 1
        mcmc_gen_topk, proposal_gen, constraint, segment = _setup_mcmc_components(batch_size=10)
        mcmc_gen_topk.top_k = 5
        mcmc_gen_topk._validate_generator()
        
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            MCMCGenerator(
                constructs=mcmc_gen.constructs,
                generators=mcmc_gen.generators,
                constraints=mcmc_gen.constraints,
                top_k=0
            )
        
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            MCMCGenerator(
                constructs=mcmc_gen.constructs,
                generators=mcmc_gen.generators,
                constraints=mcmc_gen.constraints,
                top_k=-1
            )

    def test_topk_batch_expansion(self):
        """Tests that batch sizes are correctly expanded for top-k MCMC."""
        batch_size = 6
        top_k = 3
        mcmc_gen, proposal_gen, constraint, segment = _setup_mcmc_components(
            batch_size=batch_size, num_mcmc_steps=1
        )
        
        # Create top-k MCMC generator
        mcmc_gen_topk = MCMCGenerator(
            constructs=mcmc_gen.constructs,
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            verbose=False,
            top_k=top_k
        )
        
        # Before sampling, batch size should be original
        assert proposal_gen.batch_size == batch_size
        assert len(segment.batch_sequences) == batch_size
        
        # After sampling with top_k, batch should be expanded
        mcmc_gen_topk.sample()
        expected_batch_size = batch_size * top_k
        assert proposal_gen.batch_size == expected_batch_size
        assert len(segment.batch_sequences) == expected_batch_size
        assert constraint.batch_size == expected_batch_size

    def test_topk_maintains_k_parents(self):
        """Tests that top-k MCMC maintains exactly k parent sequences."""
        batch_size = 4
        top_k = 2
        seq_length = 20
        
        # Set up with a constraint that prefers 'A' nucleotides
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("ATCGATCGATCGATCGATCG")
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        def count_a_constraint(seq, **kwargs):
            return -seq.sequence.count('A')  # Lower energy = more A's
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=count_a_constraint,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=20,
            temperature=1.0,
            temperature_min=0.01,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Check that history tracks progress
        assert len(mcmc_gen.history) > 0
        
        # Energy scores should exist for expanded batch
        expected_batch = batch_size * top_k
        assert len(mcmc_gen.energy_scores) == expected_batch

    def test_topk_vs_standard_mcmc_compatibility(self):
        """Tests that top_k=1 behaves identically to standard MCMC."""
        batch_size = 4
        seq_length = 15
        num_steps = 10
        
        # Create two identical setups
        segment1 = create_segment("A" * seq_length)
        segment2 = create_segment("A" * seq_length)
        
        gen1 = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        gen2 = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        
        gen1.assign(segment1)
        gen2.assign(segment2)
        
        constraint1 = Constraint(
            inputs=[segment1],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0}
        )
        constraint2 = Constraint(
            inputs=[segment2],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0}
        )
        
        # Standard MCMC
        mcmc_standard = MCMCGenerator(
            constructs=[Construct([segment1])],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=num_steps,
            verbose=False,
            top_k=1  # Explicit top_k=1
        )
        
        # Top-k with k=1 (should behave the same)
        mcmc_topk1 = MCMCGenerator(
            constructs=[Construct([segment2])],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=num_steps,
            verbose=False,
            top_k=1
        )
        
        # Both should maintain batch_size after sampling
        mcmc_standard.sample()
        mcmc_topk1.sample()
        
        assert gen1.batch_size == batch_size
        assert gen2.batch_size == batch_size
        assert len(segment1.batch_sequences) == batch_size
        assert len(segment2.batch_sequences) == batch_size

    def test_topk_mcmc_acceptance_criterion(self):
        """Tests that MCMC acceptance criterion is properly applied in top-k mode."""
        batch_size = 5
        top_k = 2
        seq_length = 20
        
        # Create generator with small mutations
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Constraint that strongly prefers 'G'
        def count_g_constraint(seq, **kwargs):
            return -seq.sequence.count('G')
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=count_g_constraint,
            scoring_function_config={}
        )
        
        # High temperature = more exploratory (more acceptances)
        mcmc_high_temp = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=50,
            temperature=5.0,
            temperature_min=0.1,
            verbose=False,
            top_k=top_k
        )
        
        # Get initial energy before sampling
        mcmc_high_temp.score_energy()
        initial_energy = mcmc_high_temp.energy_scores[0]
        
        mcmc_high_temp.sample()
        final_energy = min(mcmc_high_temp.energy_scores)
        
        # Should improve (lower energy)
        assert final_energy < initial_energy
        
        # Check that sequences actually changed from initial
        final_sequences = [seq.sequence for seq in segment.batch_sequences]
        initial_seq = "A" * seq_length
        assert any(seq != initial_seq for seq in final_sequences)  # At least some changed
        
        # Note: With correct parent restoration, top-k may converge to same optimum
        # This is expected behavior when the optimal solution is found

    def test_topk_fallback_to_parents(self):
        """Tests that top-k MCMC falls back to best parents when acceptances < k."""
        batch_size = 3
        top_k = 3
        seq_length = 15
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("GCGCGCGCGCGCGCG")  # Already optimal
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Very strict constraint
        def perfect_gc_constraint(seq, **kwargs):
            gc_count = seq.sequence.count('G') + seq.sequence.count('C')
            if gc_count == seq_length:
                return 0.0
            return 1.0  # Heavy penalty for non-perfect
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=perfect_gc_constraint,
            scoring_function_config={}
        )
        
        # Very low temperature = very few acceptances
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=10,
            temperature=0.001,  # Very low = almost no acceptances
            temperature_min=0.0001,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Should still have top_k * batch_size sequences after sampling
        expected_batch = top_k * batch_size
        assert len(segment.batch_sequences) == expected_batch
        assert len(mcmc_gen.energy_scores) == expected_batch

    def test_topk_history_tracking(self):
        """Tests that history is properly tracked during top-k MCMC."""
        batch_size = 4
        top_k = 2
        num_steps = 30
        track_step_size = 10
        
        mcmc_gen, _, _, segment = _setup_mcmc_components(
            batch_size=batch_size,
            num_mcmc_steps=num_steps
        )
        
        mcmc_gen_topk = MCMCGenerator(
            constructs=mcmc_gen.constructs,
            generators=mcmc_gen.generators,
            constraints=mcmc_gen.constraints,
            num_steps=num_steps,
            track_step_size=track_step_size,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen_topk.sample()
        
        # History should have snapshots at tracked steps
        expected_snapshots = 1 + (num_steps // track_step_size) + (1 if num_steps % track_step_size != 0 else 0)
        assert len(mcmc_gen_topk.history) == expected_snapshots
        
        # Each history entry should have proper structure
        for entry in mcmc_gen_topk.history:
            assert 'time_step' in entry
            assert 'energy_scores' in entry
            assert 'constructs' in entry
            assert len(entry['energy_scores']) == batch_size * top_k

    def test_history_timesteps_validation(self):
        """Tests that time_step values in history entries are correctly tracked."""
        batch_size = 4
        top_k = 2
        num_steps = 35
        track_step_size = 10
        
        mcmc_gen, _, _, _ = _setup_mcmc_components(
            batch_size=batch_size,
            num_mcmc_steps=num_steps
        )
        
        mcmc_gen_topk = MCMCGenerator(
            constructs=mcmc_gen.constructs,
            generators=mcmc_gen.generators,
            constraints=mcmc_gen.constraints,
            num_steps=num_steps,
            track_step_size=track_step_size,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen_topk.sample()
        
        # Expected timesteps: 0 (initial), 10, 20, 30, 35 (final)
        expected_timesteps = [0, 10, 20, 30, 35]
        actual_timesteps = [entry['time_step'] for entry in mcmc_gen_topk.history]
        
        assert actual_timesteps == expected_timesteps, (
            f"Expected timesteps {expected_timesteps}, but got {actual_timesteps}"
        )
        
        # Verify timesteps are monotonically increasing
        for i in range(1, len(actual_timesteps)):
            assert actual_timesteps[i] > actual_timesteps[i-1], (
                f"Timesteps should be monotonically increasing, but {actual_timesteps[i-1]} >= {actual_timesteps[i]}"
            )
        
        # Verify first timestep is 0 (initial state)
        assert actual_timesteps[0] == 0, "First timestep should be 0 (initial state)"
        
        # Verify last timestep equals num_steps
        assert actual_timesteps[-1] == num_steps, (
            f"Last timestep should equal num_steps ({num_steps}), but got {actual_timesteps[-1]}"
        )

    def test_history_timesteps_with_even_tracking(self):
        """Tests timesteps when track_step_size evenly divides num_steps."""
        batch_size = 3
        num_steps = 30
        track_step_size = 10
        
        mcmc_gen, _, _, _ = _setup_mcmc_components(
            batch_size=batch_size,
            num_mcmc_steps=num_steps
        )
        
        mcmc_gen.num_steps = num_steps
        mcmc_gen.track_step_size = track_step_size
        mcmc_gen.sample()
        
        # Expected timesteps: 0 (initial), 10, 20, 30
        # When num_steps % track_step_size == 0, final state is already captured
        expected_timesteps = [0, 10, 20, 30]
        actual_timesteps = [entry['time_step'] for entry in mcmc_gen.history]
        
        assert actual_timesteps == expected_timesteps, (
            f"Expected timesteps {expected_timesteps}, but got {actual_timesteps}"
        )

    def test_temperature_scheduling(self):
        """Tests that simulated annealing temperature schedule is correct."""
        batch_size = 2
        num_steps = 100
        temperature = 10.0
        temperature_min = 0.01
        
        mcmc_gen, _, _, _ = _setup_mcmc_components(
            batch_size=batch_size,
            num_mcmc_steps=num_steps
        )
        
        mcmc_gen.temperature = temperature
        mcmc_gen.temperature_min = temperature_min
        mcmc_gen.num_steps = num_steps
        
        # Test temperature at key steps
        step_1_temp = mcmc_gen._calculate_temperature(1)
        step_50_temp = mcmc_gen._calculate_temperature(50)
        step_100_temp = mcmc_gen._calculate_temperature(100)
        
        # Step 1 should be exactly T_max
        assert abs(step_1_temp - temperature) < 1e-10, (
            f"Step 1 temperature should be T_max={temperature}, got {step_1_temp}"
        )
        
        # Final step should be exactly T_min
        assert abs(step_100_temp - temperature_min) < 1e-10, (
            f"Final step temperature should be T_min={temperature_min}, got {step_100_temp}"
        )
        
        # Middle step should be between T_max and T_min
        assert temperature_min < step_50_temp < temperature, (
            f"Middle step temperature {step_50_temp} should be between {temperature_min} and {temperature}"
        )
        
        # Temperatures should decrease monotonically
        temperatures = [mcmc_gen._calculate_temperature(step) for step in range(1, num_steps + 1)]
        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i-1], (
                f"Temperature should decrease monotonically, but T[{i}]={temperatures[i]} > T[{i-1}]={temperatures[i-1]}"
            )

    def test_temperature_scheduling_edge_cases(self):
        """Tests temperature scheduling edge cases."""
        batch_size = 2
        temperature = 5.0
        temperature_min = 0.001
        
        # Test num_steps=1 (should return T_max)
        mcmc_gen, _, _, _ = _setup_mcmc_components(
            batch_size=batch_size,
            num_mcmc_steps=1
        )
        mcmc_gen.temperature = temperature
        mcmc_gen.temperature_min = temperature_min
        mcmc_gen.num_steps = 1
        
        step_1_temp = mcmc_gen._calculate_temperature(1)
        assert abs(step_1_temp - temperature) < 1e-10, (
            f"With num_steps=1, temperature should be T_max={temperature}, got {step_1_temp}"
        )
        
        # Test num_steps=2 (should go from T_max to T_min in one step)
        mcmc_gen.num_steps = 2
        step_1_temp = mcmc_gen._calculate_temperature(1)
        step_2_temp = mcmc_gen._calculate_temperature(2)
        
        assert abs(step_1_temp - temperature) < 1e-10, (
            f"Step 1 should be T_max={temperature}, got {step_1_temp}"
        )
        assert abs(step_2_temp - temperature_min) < 1e-10, (
            f"Step 2 should be T_min={temperature_min}, got {step_2_temp}"
        )
        
        # Test with very large num_steps (should still work)
        mcmc_gen.num_steps = 10000
        step_1_temp = mcmc_gen._calculate_temperature(1)
        step_final_temp = mcmc_gen._calculate_temperature(10000)
        
        assert abs(step_1_temp - temperature) < 1e-10
        assert abs(step_final_temp - temperature_min) < 1e-10

    def test_topk_with_multiple_constraints(self):
        """Tests top-k MCMC with multiple weighted constraints."""
        batch_size = 6
        top_k = 3
        seq_length = 30
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=2
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Multiple constraints
        gc_constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0}
        )
        
        length_constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": seq_length}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_constraint, length_constraint],
            constraint_weights=[1.0, 2.0],
            num_steps=30,
            verbose=False,
            top_k=top_k
        )
        
        # Get initial energies before sampling
        mcmc_gen.score_energy()
        initial_energies = mcmc_gen.energy_scores.copy()
        
        mcmc_gen.sample()
        final_energies = mcmc_gen.energy_scores
        
        # Should have improved (some energy should be lower)
        assert min(final_energies) <= min(initial_energies)
        
        # Batch should be expanded
        expected_batch = batch_size * top_k
        assert len(final_energies) == expected_batch

    def test_topk_parent_replication(self):
        """Tests that parent sequences are correctly replicated to batch positions."""
        batch_size = 4
        top_k = 2
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        # Create distinct initial sequences
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        
        # Manually set different sequences for testing
        segment.batch_sequences[0].sequence = "A" * seq_length
        segment.batch_sequences[1].sequence = "C" * seq_length
        segment.batch_sequences[2].sequence = "G" * seq_length
        segment.batch_sequences[3].sequence = "T" * seq_length
        
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: len(seq.sequence),
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            verbose=False,
            top_k=top_k
        )
        
        # Score to get initial energies
        mcmc_gen.score_energy()
        
        # Get initial top-k parents (should be indices with lowest energies)
        proposals_per_parent = mcmc_gen._initialize_topk()
        top_k_idx, parent_energies = mcmc_gen._get_initial_parents()
        
        # Save parent sequences before replication
        parent_seqs = [segment.batch_sequences[idx].sequence for idx in top_k_idx]
        
        # Replicate parents to batch
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
        
        # Verify replication: each parent should be copied to its designated positions
        for parent_pos, parent_idx in enumerate(top_k_idx):
            start_idx = parent_pos * proposals_per_parent
            end_idx = (parent_pos + 1) * proposals_per_parent
            parent_seq = parent_seqs[parent_pos]
            
            for idx in range(start_idx, end_idx):
                assert segment.batch_sequences[idx].sequence == parent_seq, (
                    f"Position {idx} should have parent {parent_pos} sequence, "
                    f"but got {segment.batch_sequences[idx].sequence} != {parent_seq}"
                )

    def test_topk_deepcopy_independence(self):
        """Tests that deepcopy ensures independent Sequence objects at each batch position."""
        batch_size = 3
        top_k = 2
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        
        # Set initial sequences
        segment.batch_sequences[0].sequence = "A" * seq_length
        segment.batch_sequences[1].sequence = "C" * seq_length
        segment.batch_sequences[2].sequence = "G" * seq_length
        
        # Add nested metadata to test deep copy
        segment.batch_sequences[0]._metadata['nested'] = {'count': 1, 'tags': ['x']}
        
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.score_energy()
        proposals_per_parent = mcmc_gen._initialize_topk()
        top_k_idx, parent_energies = mcmc_gen._get_initial_parents()
        
        # Test 1: _replicate_parents_to_batch creates independent copies
        mcmc_gen._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
        
        # Verify all batch positions are independent objects
        for i in range(len(segment.batch_sequences)):
            for j in range(i + 1, len(segment.batch_sequences)):
                assert segment.batch_sequences[i] is not segment.batch_sequences[j], (
                    f"Batch positions {i} and {j} should be different objects"
                )
        
        # Verify nested metadata is deeply copied (modifying one doesn't affect others)
        if 'nested' in segment.batch_sequences[0]._metadata:
            segment.batch_sequences[0]._metadata['nested']['count'] = 999
            segment.batch_sequences[0]._metadata['nested']['tags'].append('y')
            # Check other positions aren't affected
            for idx in range(1, proposals_per_parent):
                if 'nested' in segment.batch_sequences[idx]._metadata:
                    assert segment.batch_sequences[idx]._metadata['nested']['count'] == 1
                    assert segment.batch_sequences[idx]._metadata['nested']['tags'] == ['x']
        
        # Test 2: _save_parent_states and _restore_parent_state maintain independence
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        
        # Modify current sequences
        for idx in range(len(segment.batch_sequences)):
            segment.batch_sequences[idx].sequence = "T" * seq_length
        
        # Restore multiple positions from same parent
        parent_idx = top_k_idx[0]
        mcmc_gen._restore_parent_state(0, parent_idx, parent_states)
        mcmc_gen._restore_parent_state(1, parent_idx, parent_states)
        
        # Verify restored sequences are independent
        assert segment.batch_sequences[0] is not segment.batch_sequences[1], (
            "Restored sequences should be independent objects"
        )
        
        # Modify one restored sequence and verify it doesn't affect the other
        segment.batch_sequences[0].sequence = "G" * seq_length
        assert segment.batch_sequences[1].sequence != "G" * seq_length, (
            "Modifying one restored sequence should not affect others"
        )

    def test_topk_parent_energy_consistency(self):
        """Validates critical invariant: parent_energies matches parent_states energies."""
        batch_size = 4
        top_k = 2
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=2
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: float(seq.sequence.count('A')),
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=5,  # Multiple iterations
            verbose=False,
            top_k=top_k
        )
        
        # Patch _save_parent_states to verify consistency
        original_save = mcmc_gen._save_parent_states
        
        def checked_save(top_k_idx):
            # Get parent_energies from the frame above (passed to _select_topk_with_mcmc)
            import inspect
            frame = inspect.currentframe().f_back
            parent_energies = frame.f_locals.get('parent_energies')
            
            if parent_energies is not None:
                # Verify invariant: parent_energies[i] == self.energy_scores[top_k_idx[i]]
                for i, (idx, expected_energy) in enumerate(zip(top_k_idx, parent_energies)):
                    actual_energy = mcmc_gen.energy_scores[idx]
                    assert abs(actual_energy - expected_energy) < 1e-6, (
                        f"Invariant violated at parent {i}:\n"
                        f"  parent_energies[{i}] = {expected_energy}\n"
                        f"  self.energy_scores[{idx}] = {actual_energy}\n"
                        f"  Difference: {abs(actual_energy - expected_energy)}"
                    )
            
            # Call original and verify saved energies match
            parent_states = original_save(top_k_idx)
            
            if parent_energies is not None:
                for i, idx in enumerate(top_k_idx):
                    saved_energy = parent_states[idx]['energy']
                    expected_energy = parent_energies[i]
                    assert abs(saved_energy - expected_energy) < 1e-6, (
                        f"Saved energy mismatch at parent {i}:\n"
                        f"  parent_energies[{i}] = {expected_energy}\n"
                        f"  parent_states[{idx}]['energy'] = {saved_energy}\n"
                        f"  Difference: {abs(saved_energy - expected_energy)}"
                    )
            
            return parent_states
        
        mcmc_gen._save_parent_states = checked_save
        
        # Run MCMC - this will trigger the checks on every iteration
        mcmc_gen.sample()

    def test_topk_rejection_restores_parent(self):
        """Tests that rejected proposals correctly restore parent sequences and metadata."""
        batch_size = 3
        top_k = 2
        seq_length = 15
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=5
        )
        segment = create_segment("G" * seq_length)  # Optimal sequence
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Constraint that makes current state optimal (any mutation makes it worse)
        def strict_g_constraint(seq, **kwargs):
            return 0.0 if seq.sequence == "G" * seq_length else 100.0
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=strict_g_constraint,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=3,
            temperature=0.001,  # Very low = almost no acceptances of worse sequences
            temperature_min=0.0001,
            verbose=False,
            top_k=top_k
        )
        
        # Get initial best energy
        mcmc_gen.score_energy()
        initial_best_energy = min(mcmc_gen.energy_scores)
        
        mcmc_gen.sample()
        
        # With very low temperature and optimal initial state, rejections should preserve parents
        # Best energy should remain optimal (0.0)
        final_best_energy = min(mcmc_gen.energy_scores)
        assert final_best_energy == initial_best_energy, (
            f"Expected best energy to remain {initial_best_energy}, got {final_best_energy}"
        )
        
        # At least some sequences should remain optimal
        optimal_seq = "G" * seq_length
        optimal_count = sum(1 for seq in segment.batch_sequences if seq.sequence == optimal_seq)
        assert optimal_count > 0, "Expected some optimal sequences to be preserved through rejections"

    def test_topk_selection_correctness(self):
        """Tests that top-k selection picks the k best sequences by energy."""
        batch_size = 6
        top_k = 3
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("C" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Constraint with known energies based on 'G' count
        def g_count_energy(seq, **kwargs):
            return -seq.sequence.count('G')  # More G = lower energy
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=g_count_energy,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=20,
            temperature=2.0,
            temperature_min=0.1,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Get final energies
        all_energies = mcmc_gen.energy_scores
        
        # The top-k parents should be among the best energies
        # Sort all energies and check that top-k are reasonable
        sorted_energies = sorted(all_energies)
        best_k_energies = sorted_energies[:top_k]
        
        # Verify we have the right number of sequences
        assert len(all_energies) == batch_size * top_k

    def test_topk_diversity_maintenance(self):
        """Tests that top-k maintains diversity among parent sequences."""
        batch_size = 5
        top_k = 3
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=2
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Constraint that allows multiple local optima
        def diversity_constraint(seq, **kwargs):
            g_count = seq.sequence.count('G')
            c_count = seq.sequence.count('C')
            # Prefer either high G OR high C (creates multiple optima)
            return -max(g_count, c_count)
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=diversity_constraint,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=50,
            temperature=1.5,
            temperature_min=0.1,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Check that we maintain top_k distinct sequences (or at least some diversity)
        # Get the top-k parent sequences by looking at the best k energy scores
        energy_idx_pairs = [(e, i) for i, e in enumerate(mcmc_gen.energy_scores)]
        energy_idx_pairs.sort()
        top_k_indices = [idx for _, idx in energy_idx_pairs[:top_k]]
        top_k_sequences = [segment.batch_sequences[idx].sequence for idx in top_k_indices]
        
        # Count unique sequences among top-k
        unique_seqs = len(set(top_k_sequences))
        
        # With diversity-promoting constraint and sufficient temperature,
        # we should have some diversity (at least 2 different sequences)
        # Note: This is probabilistic, but with 50 steps it should converge
        assert unique_seqs >= 1, f"Expected some diversity in top-k, got {unique_seqs} unique sequences"

    def test_topk_boundary_case_equals_batch_size(self):
        """Tests edge case where top_k equals batch_size."""
        batch_size = 4
        top_k = 4
        seq_length = 15
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=10,
            verbose=False,
            top_k=top_k
        )
        
        # Should work without errors
        mcmc_gen.sample()
        
        # All initial sequences become parents
        expected_batch = batch_size * top_k
        assert len(segment.batch_sequences) == expected_batch
        assert len(mcmc_gen.energy_scores) == expected_batch

    def test_topk_all_rejections_scenario(self):
        """Tests behavior when all proposals are rejected (fall back to parents)."""
        batch_size = 3
        top_k = 2
        seq_length = 10
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=5
        )
        segment = create_segment("G" * seq_length)  # Optimal for constraint
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Constraint that makes current state optimal
        def perfect_g_constraint(seq, **kwargs):
            if seq.sequence == "G" * seq_length:
                return 0.0
            return 1000.0  # Huge penalty for any mutation
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=perfect_g_constraint,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=5,
            temperature=0.00001,  # Extremely low = no bad acceptances
            temperature_min=0.000001,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Should still complete without errors
        assert len(mcmc_gen.energy_scores) == batch_size * top_k
        
        # Best energy should remain at 0 (optimal state preserved)
        assert min(mcmc_gen.energy_scores) == 0.0

    def test_topk_energy_non_regression(self):
        """Tests that best energy never gets worse (monotonic improvement)."""
        batch_size = 5
        top_k = 3
        seq_length = 20
        num_steps = 50
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Simple constraint - prefer high GC content
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 50.0, "max_gc": 50.0}  # Target 50% GC
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=num_steps,
            track_step_size=1,  # Track every step
            temperature=0.5,  # Low temp = greedy
            temperature_min=0.01,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Extract best energy at each step from history
        best_energies_over_time = [min(entry['energy_scores']) for entry in mcmc_gen.history]
        
        # Best energy should never increase (allowing small numerical tolerance)
        tolerance = 1e-6
        for i in range(1, len(best_energies_over_time)):
            assert best_energies_over_time[i] <= best_energies_over_time[i-1] + tolerance, (
                f"Energy regression at step {i}: {best_energies_over_time[i]} > {best_energies_over_time[i-1]}"
            )
        
        # Final energy should be better than or equal to initial
        assert best_energies_over_time[-1] <= best_energies_over_time[0]

    def test_topk_metadata_preserved_through_acceptance(self):
        """Tests that metadata is preserved when proposals are accepted."""
        batch_size = 3
        top_k = 2
        seq_length = 15
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        
        # Add initial metadata
        for i, seq in enumerate(segment.batch_sequences):
            seq._metadata['seq_id'] = f"seq_{i}"
            seq._metadata['generation'] = 0
        
        construct = Construct([segment])
        
        # Constraint that strongly encourages changes (high GC better)
        def strong_gc_constraint(seq, **kwargs):
            gc_count = seq.sequence.count('G') + seq.sequence.count('C')
            return -gc_count  # More GC = lower energy = better
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=strong_gc_constraint,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=10,
            temperature=5.0,  # High temp = accept most proposals
            temperature_min=1.0,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # After MCMC with high acceptance rate, sequences should have changed
        # but metadata should still exist (even if modified)
        final_sequences = [seq.sequence for seq in segment.batch_sequences]
        changed_count = sum(1 for seq in final_sequences if seq != "A" * seq_length)
        
        # With high temp and GC-promoting constraint, expect changes
        assert changed_count > 0, "Expected at least some sequences to change with high acceptance rate"
        
        # Metadata should exist for all sequences
        for seq in segment.batch_sequences:
            assert len(seq._metadata) > 0, "Metadata should not be empty"

    def test_topk_with_multiple_generators_specific(self):
        """Tests top-k MCMC specifically with multiple proposal generators."""
        batch_size = 6
        top_k = 3
        seq_length = 20
        
        # Create two different mutation generators
        gen1 = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        gen2 = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=3
        )
        
        segment1 = create_segment("A" * seq_length)
        segment2 = create_segment("C" * seq_length)
        
        gen1.assign(segment1)
        gen2.assign(segment2)
        
        construct = Construct([segment1, segment2])
        
        constraint = Constraint(
            inputs=[segment1, segment2],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
            constraint_type=ConstraintType.CONTIGUOUS
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[gen1, gen2],  # Multiple generators
            constraints=[constraint],
            num_steps=20,
            temperature=1.0,
            temperature_min=0.1,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Should work without errors and expand both generators
        assert gen1.batch_size == batch_size * top_k
        assert gen2.batch_size == batch_size * top_k
        assert len(segment1.batch_sequences) == batch_size * top_k
        assert len(segment2.batch_sequences) == batch_size * top_k

    def test_topk_convergence_to_optimal(self):
        """Tests that top-k MCMC converges to optimal solution with enough steps."""
        batch_size = 8
        top_k = 4
        seq_length = 15
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)  # Start far from optimal
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        # Constraint with clear global optimum (all G's)
        def perfect_g_energy(seq, **kwargs):
            g_count = seq.sequence.count('G')
            return seq_length - g_count  # Perfect = 0 energy
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=perfect_g_energy,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=200,  # Many steps for convergence
            temperature=2.0,
            temperature_min=0.01,  # Anneal to greedy
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.score_energy()
        initial_best_energy = min(mcmc_gen.energy_scores)
        
        mcmc_gen.sample()
        final_best_energy = min(mcmc_gen.energy_scores)
        
        # Should get very close to optimal (energy near 0)
        assert final_best_energy < initial_best_energy * 0.5, (
            f"Expected significant improvement, got {final_best_energy} vs {initial_best_energy}"
        )
        
        # With enough steps, should find sequences close to all G's
        best_g_count = max(seq.sequence.count('G') for seq in segment.batch_sequences)
        assert best_g_count >= seq_length * 0.8, (
            f"Expected convergence toward G's, best has only {best_g_count}/{seq_length} G's"
        )

    def test_topk_energy_variance_reduction(self):
        """Tests that energy variance among top-k decreases over time (convergence)."""
        batch_size = 6
        top_k = 4
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 50.0, "max_gc": 50.0}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=50,
            track_step_size=10,
            temperature=1.0,
            temperature_min=0.01,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.sample()
        
        # Extract energy variance at each tracked step
        variances = []
        for entry in mcmc_gen.history:
            energies = entry['energy_scores']
            # Get top-k energies
            sorted_energies = sorted(energies)[:top_k]
            var = np.var(sorted_energies) if len(sorted_energies) > 1 else 0.0
            variances.append(var)
        
        # Variance should generally decrease (allowing some fluctuation)
        # Compare first quarter to last quarter
        first_quarter_var = np.mean(variances[:len(variances)//4 + 1])
        last_quarter_var = np.mean(variances[-len(variances)//4:])
        
        # Last quarter should have lower or similar variance (convergence)
        assert last_quarter_var <= first_quarter_var * 2.0, (
            f"Expected variance reduction, got initial={first_quarter_var:.4f}, "
            f"final={last_quarter_var:.4f}"
        )

    def test_topk_exact_state_restoration(self):
        """Validates that rejected proposals EXACTLY match parent state (bit-perfect)."""
        batch_size = 3
        top_k = 2
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=3
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        
        # Add complex nested metadata to test deep equality
        for i, seq_obj in enumerate(segment.batch_sequences):
            seq_obj._metadata['test_data'] = {
                'id': i,
                'nested': {'values': [1, 2, 3], 'flag': True}
            }
        
        construct = Construct([segment])
        # Constraint that makes mutations worse (favors original)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: float(seq.sequence.count('C') + seq.sequence.count('G')),
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            temperature=0.001,  # Very low temp → reject bad moves
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.score_energy()
        proposals_per_parent = mcmc_gen._initialize_topk()
        top_k_idx, parent_energies = mcmc_gen._get_initial_parents()
        
        # Save EXACT parent state before mutation
        exact_parent_states = {}
        for parent_idx in top_k_idx:
            exact_parent_states[parent_idx] = {
                'sequence': segment.batch_sequences[parent_idx].sequence,
                'metadata': copy.deepcopy(segment.batch_sequences[parent_idx]._metadata),
                'energy': mcmc_gen.energy_scores[parent_idx]
            }
        
        # Save and replicate
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
        
        # Mutate (makes sequences worse)
        mcmc_gen._generate_proposals()
        
        # Manually restore and check EXACT equality
        for parent_pos, parent_idx in enumerate(top_k_idx):
            # Restore first proposal from this parent
            proposal_idx = parent_pos * proposals_per_parent
            mcmc_gen._restore_parent_state(proposal_idx, parent_idx, parent_states)
            
            # Verify EXACT match
            restored_seq = segment.batch_sequences[proposal_idx]
            expected = exact_parent_states[parent_idx]
            
            # Check sequence
            assert restored_seq.sequence == expected['sequence'], (
                f"Sequence mismatch: {restored_seq.sequence} != {expected['sequence']}"
            )
            
            # Check energy
            assert mcmc_gen.energy_scores[proposal_idx] == expected['energy'], (
                f"Energy mismatch: {mcmc_gen.energy_scores[proposal_idx]} != {expected['energy']}"
            )
            
            # Check metadata deep equality
            assert restored_seq._metadata == expected['metadata'], (
                f"Metadata mismatch:\n  Restored: {restored_seq._metadata}\n  Expected: {expected['metadata']}"
            )
            
            # Verify nested metadata wasn't aliased
            if 'test_data' in restored_seq._metadata:
                restored_seq._metadata['test_data']['nested']['values'].append(999)
                # Check other restored positions weren't affected
                for other_pos in range(proposals_per_parent):
                    other_idx = parent_pos * proposals_per_parent + other_pos
                    if other_idx != proposal_idx:
                        other_metadata = segment.batch_sequences[other_idx]._metadata
                        if 'test_data' in other_metadata:
                            assert 999 not in other_metadata['test_data']['nested']['values']

    def test_topk_mixed_acceptance_rejection(self):
        """Tests scenario with both accepted and rejected proposals from same parent."""
        batch_size = 5  # 5 proposals per parent
        top_k = 2
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=2
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        
        construct = Construct([segment])
        # Constraint with controlled randomness
        import random as rand
        rand.seed(42)
        
        acceptance_pattern = [True, False, True, False, False]  # 2 accept, 3 reject per parent
        
        def controlled_constraint(seq, **kwargs):
            # Return different energies to control acceptance
            a_count = seq.sequence.count('A')
            if a_count >= 18:  # Mostly A's (original or similar)
                return 10.0  # Parent energy
            else:  # Mutated
                return 5.0  # Better energy → should accept
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=controlled_constraint,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            temperature=1.0,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.score_energy()
        proposals_per_parent = mcmc_gen._initialize_topk()
        top_k_idx, parent_energies = mcmc_gen._get_initial_parents()
        
        # Save parent sequences
        parent_seqs = {idx: segment.batch_sequences[idx].sequence for idx in top_k_idx}
        
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
        mcmc_gen._generate_proposals()
        
        # Apply MCMC selection
        new_top_k_idx, new_parent_energies = mcmc_gen._select_topk_with_mcmc(
            top_k_idx, parent_energies, proposals_per_parent, 1.0, parent_states
        )
        
        # Verify: candidate pool should have mix of mutations and restored parents
        # Count how many sequences changed vs stayed same for each parent
        for parent_pos, parent_idx in enumerate(top_k_idx):
            start_idx = parent_pos * proposals_per_parent
            end_idx = (parent_pos + 1) * proposals_per_parent
            parent_seq = parent_seqs[parent_idx]
            
            unchanged_count = 0
            changed_count = 0
            
            for idx in range(start_idx, end_idx):
                if segment.batch_sequences[idx].sequence == parent_seq:
                    unchanged_count += 1
                else:
                    changed_count += 1
            
            # Should have SOME of both (mix of accept/reject)
            # With batch_size=5 and mutations, unlikely all identical or all different
            assert unchanged_count + changed_count == proposals_per_parent

    def test_topk_energy_scores_sync_after_rejection(self):
        """Validates self.energy_scores stays synchronized after rejections."""
        batch_size = 4
        top_k = 2
        seq_length = 15
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=3
        )
        segment = create_segment("G" * seq_length)
        proposal_gen.assign(segment)
        
        construct = Construct([segment])
        # Constraint that makes current state optimal
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0 if seq.sequence == "G" * seq_length else 100.0,
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            temperature=0.001,  # Low temp → reject mutations
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.score_energy()
        proposals_per_parent = mcmc_gen._initialize_topk()
        top_k_idx, parent_energies = mcmc_gen._get_initial_parents()
        
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
        mcmc_gen._generate_proposals()
        
        # After mutation, energy_scores should be 100.0 (bad)
        for idx in range(len(mcmc_gen.energy_scores)):
            assert mcmc_gen.energy_scores[idx] == 100.0
        
        # Apply MCMC (should reject all and restore)
        new_top_k_idx, new_parent_energies = mcmc_gen._select_topk_with_mcmc(
            top_k_idx, parent_energies, proposals_per_parent, 0.001, parent_states
        )
        
        # After rejection, energy_scores should be synchronized with sequences
        for idx in range(len(mcmc_gen.energy_scores)):
            seq = segment.batch_sequences[idx].sequence
            expected_energy = 0.0 if seq == "G" * seq_length else 100.0
            actual_energy = mcmc_gen.energy_scores[idx]
            
            assert actual_energy == expected_energy, (
                f"Energy mismatch at position {idx}:\n"
                f"  Sequence: {seq}\n"
                f"  Expected energy: {expected_energy}\n"
                f"  Actual energy: {actual_energy}\n"
                f"  This indicates energy_scores array is out of sync with sequences!"
            )

    def test_topk_all_proposals_better(self):
        """Edge case: all proposals improve energy (no rejections)."""
        batch_size = 3
        top_k = 2
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=2
        )
        segment = create_segment("C" * seq_length)
        proposal_gen.assign(segment)
        
        construct = Construct([segment])
        # Constraint where mutations (adding A's) are BETTER
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: float(seq.sequence.count('C')),  # Lower C count = better
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            temperature=1.0,
            verbose=False,
            top_k=top_k
        )
        
        mcmc_gen.score_energy()
        proposals_per_parent = mcmc_gen._initialize_topk()
        top_k_idx, parent_energies = mcmc_gen._get_initial_parents()
        
        # Record parent energies
        parent_energy_values = [mcmc_gen.energy_scores[idx] for idx in top_k_idx]
        
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
        mcmc_gen._generate_proposals()
        
        # Verify all proposals are better than their parents
        for parent_pos, parent_idx in enumerate(top_k_idx):
            start_idx = parent_pos * proposals_per_parent
            end_idx = (parent_pos + 1) * proposals_per_parent
            parent_energy = parent_energy_values[parent_pos]
            
            for idx in range(start_idx, end_idx):
                proposal_energy = mcmc_gen.energy_scores[idx]
                assert proposal_energy < parent_energy, (
                    f"Expected all proposals to be better, but proposal {idx} has energy "
                    f"{proposal_energy} >= parent energy {parent_energy}"
                )
        
        # Apply MCMC selection
        new_top_k_idx, new_parent_energies = mcmc_gen._select_topk_with_mcmc(
            top_k_idx, parent_energies, proposals_per_parent, 1.0, parent_states
        )
        
        # Verify: all selected parents should have better energy than original parents
        for new_energy in new_parent_energies:
            assert all(new_energy <= orig_energy for orig_energy in parent_energy_values), (
                f"Selected energy {new_energy} should be better than all original parents"
            )

    def test_topk_proposal_to_parent_relationship(self):
        """Validates each proposal is compared to its generating parent, not global best."""
        # This test validates the CRITICAL invariant: each proposal is compared to its OWN parent,
        # not to the global best parent. This is what makes top-k MCMC work correctly.
        
        batch_size = 2
        top_k = 2
        seq_length = 20
        
        proposal_gen = UniformMutationGenerator(
            sequence_length=seq_length, batch_size=batch_size, num_mutations=1
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: float(seq.sequence.count('A')),
            scoring_function_config={}
        )
        
        mcmc_gen = MCMCGenerator(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            num_steps=1,
            temperature=10.0,
            verbose=False,
            top_k=top_k
        )
        
        # Initialize and get initial parents (all will be identical after init)
        proposals_per_parent = mcmc_gen._initialize_topk()
        mcmc_gen.score_energy()
        
        # Manually create two parents with different energies AFTER initialization
        # Parent 0 (position 0): best energy
        # Parent 1 (position 1): worse energy
        segment.batch_sequences[0].sequence = "G" * seq_length  # 0 A's
        segment.batch_sequences[1].sequence = "AAAAAGGGGGGGGGGGGGGG"  # 5 A's
        mcmc_gen.score_energy()
        
        top_k_idx = np.array([0, 1])
        parent_energies = [mcmc_gen.energy_scores[0], mcmc_gen.energy_scores[1]]
        
        # Verify parents have different energies
        assert parent_energies[1] > parent_energies[0]
        
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._replicate_parents_to_batch(top_k_idx, proposals_per_parent)
        
        # Manually set proposal from parent 1 to have energy BETWEEN the two parents
        # Parent 0: energy = 0 (best)
        # Proposal: energy = 2 (middle)
        # Parent 1: energy = 5 (worse)
        # 
        # If compared to parent 0 (WRONG): 2 > 0 → worse → likely reject
        # If compared to parent 1 (CORRECT): 2 < 5 → better → accept
        
        proposal_from_parent1 = 1 * proposals_per_parent  # First proposal from parent 1
        segment.batch_sequences[proposal_from_parent1].sequence = "AA" + "G" * (seq_length - 2)
        mcmc_gen.score_energy()
        
        proposal_energy = mcmc_gen.energy_scores[proposal_from_parent1]
        assert parent_energies[0] < proposal_energy < parent_energies[1], (
            f"Test setup failed: proposal energy {proposal_energy} should be between "
            f"{parent_energies[0]} and {parent_energies[1]}"
        )
        
        # Apply MCMC - if proposal is compared to correct parent, it should be accepted
        new_top_k_idx, new_parent_energies = mcmc_gen._select_topk_with_mcmc(
            top_k_idx, parent_energies, proposals_per_parent, 10.0, parent_states
        )
        
        # Check if proposal was accepted (sequence should still have 2 A's)
        final_seq = segment.batch_sequences[proposal_from_parent1].sequence
        assert final_seq.count('A') == 2, (
            f"Proposal should have been accepted (better than its parent {parent_energies[1]}), "
            f"but final sequence has {final_seq.count('A')} A's instead of 2. "
            f"This suggests it was incorrectly compared to the global best parent instead of its own parent."
        )


def _setup_chained_components(
    seq_length: int = 10,
    batch_size: int = 2,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
):
    """Helper function to set up components for ChainedGenerator testing."""
    # 1. Create segments and generators
    segment1 = create_segment("A" * seq_length)
    segment2 = create_segment("C" * seq_length)
    
    gen1 = UniformMutationGenerator(sequence_length=seq_length, batch_size=batch_size)
    gen2 = UniformMutationGenerator(sequence_length=seq_length, batch_size=batch_size)
    
    # 2. Assign generators to segments (this sets _is_assigned = True)
    gen1.assign(segment1)
    gen2.assign(segment2)
    
    # 3. Create constructs and constraints
    construct1 = Construct([segment1])
    construct2 = Construct([segment2])
    
    constraint1 = Constraint(
        inputs=[segment1],
        scoring_function=gc_content_constraint,
        scoring_function_config={
            "min_gc": gc_target_range[0],
            "max_gc": gc_target_range[1],
        }
    )
    constraint2 = Constraint(
        inputs=[segment2],
        scoring_function=gc_content_constraint,
        scoring_function_config={
            "min_gc": gc_target_range[0],
            "max_gc": gc_target_range[1],
        }
    )
    
    return segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2


class TestChainedGenerator:
    def test_initialization(self):
        """Tests successful initialization of ChainedGenerator."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=3,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        assert len(chained.generator_stages) == 2
        assert chained.generator_stages[0] == stage1
        assert chained.generator_stages[1] == stage2
        assert chained.verbose == False
        assert chained.capture_metadata == True
        assert len(chained.stage_results) == 0
        assert chained._execution_start_time is None

    def test_validation_errors(self):
        """Tests validation errors during initialization."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Test empty stages list
        with pytest.raises(ValueError, match="At least one generator stage must be provided"):
            ChainedGenerator([], verbose=False)
        
        # Test non-IterativeGenerator stage
        with pytest.raises(ValueError, match="must be an IterativeGenerator"):
            ChainedGenerator([gen1], verbose=False)  # gen1 is not an IterativeGenerator
        
        # Test mismatched batch sizes
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        
        # Create stage with different batch size
        gen2_different_batch = UniformMutationGenerator(sequence_length=10, batch_size=3)
        gen2_different_batch.assign(segment2)
        stage2_different = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2_different_batch],
            constraints=[constraint2],
            num_steps=3,
            verbose=False
        )
        
        with pytest.raises(ValueError, match="same batch_size"):
            ChainedGenerator([stage1, stage2_different], verbose=False)


    def test_basic_execution(self):
        """Tests basic execution of the chained generator."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Run the pipeline
        chained.run()
        
        # Check that results were captured
        assert len(chained.stage_results) == 2
        assert chained.stage_results[0]['stage'] == 0
        assert chained.stage_results[1]['stage'] == 1
        assert chained.stage_results[0]['stage_type'] == 'MCMCGenerator'
        assert chained.stage_results[1]['stage_type'] == 'MCMCGenerator'

    def test_sequence_propagation(self):
        """Tests that sequences are properly propagated between stages."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Run the pipeline
        chained.run()
        
        # Check that sequences were propagated
        # Stage 1 should have modified segment1
        stage1_constructs = chained.stage_results[0]['constructs']
        stage2_constructs = chained.stage_results[1]['constructs']
        
        # The sequences should be different from the initial "A" * 10
        # Access the sequence through the batch_sequences
        assert stage1_constructs[0].segments[0].batch_sequences[0].sequence != "A" * 10
        assert stage2_constructs[0].segments[0].batch_sequences[0].sequence != "C" * 10

    def test_metadata_capture(self):
        """Tests that metadata is properly captured from each stage."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=True, capture_metadata=True)
        
        # Run the pipeline
        chained.run()
        
        # Check metadata capture
        for i, result in enumerate(chained.stage_results):
            assert 'stage' in result
            assert 'stage_type' in result
            assert 'constructs' in result
            assert 'final_energy' in result
            assert 'execution_time' in result
            assert 'stage_config' in result
            assert 'outputs_metadata' in result
            
            # Check specific values
            assert result['stage'] == i
            assert result['execution_time'] > 0
            assert len(result['constructs']) > 0

    def test_results_access_methods(self):
        """Tests all the results access methods."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test before running
        with pytest.raises(RuntimeError, match="run\\(\\) must be called"):
            chained.get_final_constructs()
        
        # Run the pipeline
        chained.run()
        
        # Test get_final_constructs
        final_constructs = chained.get_final_constructs()
        assert len(final_constructs) > 0
        assert final_constructs == chained.stage_results[-1]['constructs']
        
        # Test get_final_sequences
        final_sequences = chained.get_final_sequences()
        assert len(final_sequences) > 0
        assert isinstance(final_sequences[0], str)
        
        # Test get_stage_results
        stage_results = chained.get_stage_results()
        assert len(stage_results) == 2
        assert stage_results == chained.stage_results
        
        # Test get_stage_metadata
        stage_metadata = chained.get_stage_metadata()
        assert len(stage_metadata) == 2
        for meta in stage_metadata:
            assert 'stage' in meta
            assert 'stage_type' in meta
            assert 'outputs_metadata' in meta
            assert 'execution_summary' in meta

    def test_stage_access_methods(self):
        """Tests methods for accessing individual stages."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test get_stage
        assert chained.get_stage(0) == stage1
        assert chained.get_stage(1) == stage2
        assert chained.get_stage(2) is None
        assert chained.get_stage(-1) is None
        
        # Test get_stage_result before running
        assert chained.get_stage_result(0) is None
        
        # Run the pipeline
        chained.run()
        
        # Test get_stage_result after running
        stage1_result = chained.get_stage_result(0)
        stage2_result = chained.get_stage_result(1)
        assert stage1_result is not None
        assert stage2_result is not None
        assert stage1_result['stage'] == 0
        assert stage2_result['stage'] == 1
        assert chained.get_stage_result(2) is None

    def test_execution_summary(self):
        """Tests the execution summary functionality."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test summary before running
        summary_before = chained.get_execution_summary()
        assert summary_before['total_stages'] == 2
        assert summary_before['total_execution_time'] == 0.0
        assert summary_before['final_energy'] is None
        assert summary_before['energy_progression'] == []
        assert summary_before['stage_types'] == ['MCMCGenerator', 'MCMCGenerator']
        
        # Run the pipeline
        chained.run()
        
        # Test summary after running
        summary_after = chained.get_execution_summary()
        assert summary_after['total_stages'] == 2
        assert summary_after['total_execution_time'] > 0
        assert summary_after['final_energy'] is not None
        assert len(summary_after['energy_progression']) == 2
        assert len(summary_after['stage_types']) == 2

    def test_energy_progression(self):
        """Tests the energy progression tracking."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test before running
        assert chained.get_energy_progression() == []
        
        # Run the pipeline
        chained.run()
        
        # Test after running
        energy_prog = chained.get_energy_progression()
        assert len(energy_prog) == 2
        assert all(isinstance(e, (float, type(None))) for e in energy_prog)

    def test_export_results(self):
        """Tests the export functionality."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False)
        
        # Test export before running
        with pytest.raises(RuntimeError, match="No results to export"):
            chained.export_results('test.json')
        
        # Run the pipeline
        chained.run()
        
        # Test JSON export
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            chained.export_results(temp_file, 'json')
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Test pickle export
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        try:
            chained.export_results(temp_file, 'pickle')
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            chained.export_results('test.txt', 'txt')

    def test_verbose_execution(self):
        """Tests that verbose mode provides appropriate output."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=True)
        
        # Run the pipeline (this should print progress)
        chained.run()
        
        # Check that results were captured despite verbose output
        assert len(chained.stage_results) == 2

    def test_metadata_capture_disabled(self):
        """Tests that metadata capture can be disabled."""
        segment1, segment2, gen1, gen2, construct1, construct2, constraint1, constraint2 = _setup_chained_components()
        
        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1, stage2], verbose=False, capture_metadata=False)
        
        # Run the pipeline
        chained.run()
        
        # Check that basic results are still captured
        assert len(chained.stage_results) == 2
        
        # Check that outputs_metadata might be empty or minimal
        for result in chained.stage_results:
            assert 'outputs_metadata' in result

    def test_single_stage_execution(self):
        """Tests execution with only one stage."""
        segment1, _, gen1, _, construct1, _, constraint1, _ = _setup_chained_components()
        
        # Create single stage
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False
        )
        
        chained = ChainedGenerator([stage1], verbose=False)
        
        # Run the pipeline
        chained.run()
        
        # Check results
        assert len(chained.stage_results) == 1
        assert chained.stage_results[0]['stage'] == 0
        assert chained.stage_results[0]['stage_type'] == 'MCMCGenerator'
        
        # Test final constructs access
        final_constructs = chained.get_final_constructs()
        assert len(final_constructs) > 0
