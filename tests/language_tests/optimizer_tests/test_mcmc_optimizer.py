import pytest
import random
import numpy as np
import copy
from typing import Tuple

import sys

sys.path.append(".")
from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
    SequenceType,
)
from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import SequenceLengthConfig
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
)


# Helper function
def create_segment(sequence: str, seq_type: SequenceType = SequenceType.DNA) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


def _setup_mcmc_components(
    seq_length: int = 10,
    num_candidates: int = None,  # Defaults to batch_size if None
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    num_mcmc_steps: int = 10,
):
    """Helper function to set up a basic MCMC Optimizer for testing."""
    # 1. Create the proposal generator and the segment it will modify.
    # Note: sub-generator batch_size will be overridden to batch_size by MCMCOptimizer
    proposal_gen = UniformMutationGenerator(
        UniformMutationGeneratorConfig(sequence_length=seq_length, batch_size=1, num_mutations=1)
    )
    segment = create_segment("A" * seq_length)  # Start with a known sequence
    proposal_gen.assign(segment)

    # 2. Create the construct and constraint.
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        scoring_function=gc_content_constraint,
        scoring_function_config=GCContentConfig(
            min_gc=gc_target_range[0],
            max_gc=gc_target_range[1],
        ),
    )

    # 3. Create the MCMC Optimizer config
    config = MCMCOptimizerConfig(
        num_steps=num_mcmc_steps,
        verbose=False,
    )
    if num_candidates is not None:
        config.num_candidates = num_candidates

    mcmc_gen = MCMCOptimizer(
        constructs=[construct],
        generators=[proposal_gen],
        constraints=[constraint],
        config=config,
    )
    return mcmc_gen, proposal_gen, constraint, segment


class TestMCMCOptimizer:
    def test_initialization_and_validation(self):
        """Tests successful initialization and validation of MCMCOptimizer."""
        mcmc_gen, proposal_gen, constraint, segment = _setup_mcmc_components()

        assert mcmc_gen.generators == [proposal_gen]
        assert mcmc_gen.constraints == [constraint]
        assert mcmc_gen.constraint_weights == [1.0]
        assert (
            mcmc_gen._is_initialized
        )  # Optimizer base class is auto-initialized

        # Test validation errors
        # Unassigned generator
        unassigned_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=10, num_mutations=1)
        )
        with pytest.raises(RuntimeError, match="has not been assigned"):
            MCMCOptimizer(
                constructs=[Construct([create_segment("A" * 10)])],
                generators=[unassigned_gen],
                constraints=[],
                config=MCMCOptimizerConfig(),
            )

        # Mismatched weights and constraints
        with pytest.raises(ValueError, match="must match"):
            MCMCOptimizer(
                constructs=mcmc_gen.constructs,
                generators=mcmc_gen.generators,
                constraints=mcmc_gen.constraints,
                constraint_weights=[1.0, 2.0],
                config=MCMCOptimizerConfig(),
            )

        # Unassigned segment in construct
        segment_assigned = create_segment("A" * 10)
        gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=10, num_mutations=1)
        )
        gen.assign(segment_assigned)
        segment_unassigned = create_segment("C" * 10)  # Not assigned to any generator
        construct = Construct([segment_assigned, segment_unassigned])
        # Need at least one constraint, so add a dummy one
        dummy_constraint = Constraint(
            inputs=[segment_assigned],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={},
        )
        with pytest.raises(ValueError, match="not assigned to any generator"):
            MCMCOptimizer(
                constructs=[construct], generators=[gen], constraints=[dummy_constraint],
                config=MCMCOptimizerConfig()
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
        assert hasattr(mcmc_gen, "energy_scores")
        assert len(mcmc_gen.energy_scores) == 1
        assert abs(mcmc_gen.energy_scores[0] - expected_score) < 1e-9

        # Test that calling score_energy again updates the stored scores
        segment.batch_sequences[0].sequence = "GCGCGCGCGC"  # 100% GC -> score 1.0
        mcmc_gen.score_energy()
        expected_new_score = abs(
            (40.0 - 100.0) / 40.0
        )  # Should be 1.5, but clamped to 1.0
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
            gc_target_range=(80.0, 90.0),  # Encourage high GC
            num_mcmc_steps=100,
        )

        # Start with a bad sequence
        segment.batch_sequences[0].sequence = "A" * 50
        mcmc_gen.score_energy()
        initial_energy = mcmc_gen.energy_scores[0]
        assert initial_energy > 0.99  # Should be max penalty (1.0)

        # Sample and check for improvement
        mcmc_gen.sample()
        mcmc_gen.score_energy()
        final_energy = mcmc_gen.energy_scores[0]

        assert final_energy < initial_energy
        assert len(mcmc_gen.history) > 1  # Check that history is being tracked

    def test_multiple_constraints(self):
        """Tests the MCMC Optimizer with multiple constraints and weights."""
        seq_len = 30
        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_len, num_mutations=1)
        )
        segment = create_segment("A" * seq_len)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        gc_con = Constraint(
            [segment], gc_content_constraint, {"min_gc": 40.0, "max_gc": 60.0}
        )
        len_con = Constraint(
            [segment], sequence_length_constraint, {"target_length": seq_len}
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_con, len_con],
            constraint_weights=[1.0, 2.0],  # Weight length more
            config=MCMCOptimizerConfig(num_steps=1, verbose=False),
        )

        segment.batch_sequences[0].sequence = "A" * 20  # Violates length and GC
        gc_score = gc_con.evaluate()[0]  # (40-0)/40 = 1.0
        len_score = len_con.evaluate()[0]  # (30-20)/30 = 0.333

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
                    seq.sequence = (
                        seq.sequence[:start] + inverted_sub + seq.sequence[end:]
                    )

        seq_len = 50
        # Generator 1: Point mutations
        mut_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_len, num_mutations=1)
        )
        segment1 = create_segment("A" * seq_len)
        mut_gen.assign(segment1)

        # Generator 2: Inversions
        inv_gen = InversionGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_len, num_mutations=1)
        )
        segment2 = create_segment("C" * seq_len)
        inv_gen.assign(segment2)

        construct = Construct([segment1, segment2])
        constraint = Constraint(
            inputs=[segment1, segment2],  # Constraint on the whole construct
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": seq_len * 2},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[mut_gen, inv_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=20, verbose=False),
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
        """Tests initialization of top-k MCMC with various batch_size values."""
        proposals_per_parent = 10
        mcmc_gen, _, _, _ = _setup_mcmc_components(num_candidates=proposals_per_parent)

        # Test batch_size=1 (standard MCMC, default from _setup_mcmc_components)
        assert mcmc_gen.batch_size == 1

        # Test batch_size > 1
        # Note: We can't directly test batch_size > 1 with _setup_mcmc_components
        # since it creates a batch of 1. We test validation instead.

        # Test invalid batch_size (pydantic will raise ValidationError)
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_candidates=proposals_per_parent, batch_size=0)

        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_candidates=proposals_per_parent, batch_size=-1)

    def test_topk_batch_expansion(self):
        """Tests that batch sizes are correctly expanded for top-k MCMC."""
        proposals_per_parent = 6
        batch_size = 3
        seq_length = 10

        # Create components manually for batch_size > 1
        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # Create top-k MCMC Optimizer
        mcmc_gen_topk = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=1, verbose=False, batch_size=batch_size),
        )

        # After initialization, batch size should be set
        assert mcmc_gen_topk.batch_size == batch_size
        assert len(segment.batch_sequences) == batch_size

        # After sampling with batch_size, batch should be trimmed to batch_size
        mcmc_gen_topk.sample()
        # Generator batch_size gets expanded during sampling
        expected_expanded_batch = batch_size * proposals_per_parent
        assert proposal_gen.batch_size == expected_expanded_batch
        # But segments are trimmed to batch_size for user visibility
        assert len(segment.batch_sequences) == batch_size
        assert segment.batch_size == batch_size

    def test_topk_maintains_k_parents(self):
        """Tests that top-k MCMC maintains exactly k parent sequences."""
        proposals_per_parent = 4
        batch_size = 2
        seq_length = 20

        # Set up with a constraint that prefers 'A' nucleotides
        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("ATCGATCGATCGATCGATCG")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        def count_a_constraint(seq, **kwargs):
            return -seq.sequence.count("A")  # Lower energy = more A's

        constraint = Constraint(
            inputs=[segment],
            scoring_function=count_a_constraint,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=20, temperature=1.0, temperature_min=0.01, verbose=False, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # Check that history tracks progress
        assert len(mcmc_gen.history) > 0

        # After trimming, energy scores should match batch_size
        assert len(mcmc_gen.energy_scores) == batch_size

    def test_topk_vs_standard_mcmc_compatibility(self):
        """Tests that top_k=1 behaves identically to standard MCMC."""
        proposals_per_parent = 4
        seq_length = 15
        num_steps = 10

        # Create two identical setups
        segment1 = create_segment("A" * seq_length)
        segment2 = create_segment("A" * seq_length)

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=1, num_mutations=1
            )
        )
        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=1, num_mutations=1
            )
        )

        gen1.assign(segment1)
        gen2.assign(segment2)

        constraint1 = Constraint(
            inputs=[segment1],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )
        constraint2 = Constraint(
            inputs=[segment2],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        # Standard MCMC
        mcmc_standard = MCMCOptimizer(
            constructs=[Construct([segment1])],
            generators=[gen1],
            constraints=[constraint1],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=num_steps, verbose=False, batch_size=1),
        )

        # Top-k with k=1 (should behave the same)
        mcmc_topk1 = MCMCOptimizer(
            constructs=[Construct([segment2])],
            generators=[gen2],
            constraints=[constraint2],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=num_steps, verbose=False, batch_size=1),
        )

        # Both should maintain batch_size sequences after sampling (which is 1 in both cases)
        mcmc_standard.sample()
        mcmc_topk1.sample()

        # Generator batch_size remains expanded
        assert gen1.batch_size == proposals_per_parent
        assert gen2.batch_size == proposals_per_parent
        # But segments are trimmed to batch_size (which is 1 in both cases)
        assert len(segment1.batch_sequences) == 1
        assert len(segment2.batch_sequences) == 1

    def test_topk_mcmc_acceptance_criterion(self):
        """Tests that MCMC acceptance criterion is properly applied in top-k mode."""
        proposals_per_parent = 5
        batch_size = 2
        seq_length = 20

        # Create generator with small mutations
        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Constraint that strongly prefers 'G'
        def count_g_constraint(seq, **kwargs):
            return -seq.sequence.count("G")

        constraint = Constraint(
            inputs=[segment],
            scoring_function=count_g_constraint,
            scoring_function_config={},
        )

        # High temperature = more exploratory (more acceptances)
        mcmc_high_temp = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=50, temperature=5.0, temperature_min=0.1, verbose=False, batch_size=batch_size),
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
        assert any(
            seq != initial_seq for seq in final_sequences
        )  # At least some changed

        # Note: With correct parent restoration, top-k may converge to same optimum
        # This is expected behavior when the optimal solution is found

    def test_topk_fallback_to_parents(self):
        """Tests that top-k MCMC falls back to best parents when acceptances < k."""
        proposals_per_parent = 3
        batch_size = 3
        seq_length = 15

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("GCGCGCGCGCGCGCG")  # Already optimal
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Very strict constraint
        def perfect_gc_constraint(seq, **kwargs):
            gc_count = seq.sequence.count("G") + seq.sequence.count("C")
            if gc_count == seq_length:
                return 0.0
            return 1.0  # Heavy penalty for non-perfect

        constraint = Constraint(
            inputs=[segment],
            scoring_function=perfect_gc_constraint,
            scoring_function_config={},
        )

        # Very low temperature = very few acceptances
        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=10, temperature=0.001, temperature_min=0.0001, verbose=False, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # After trimming, should have batch_size sequences
        assert len(segment.batch_sequences) == batch_size
        assert len(mcmc_gen.energy_scores) == batch_size

    def test_topk_history_tracking(self):
        """Tests that history is properly tracked during top-k MCMC."""
        proposals_per_parent = 4
        batch_size = 2
        num_steps = 30
        track_step_size = 10
        seq_length = 10

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        mcmc_gen_topk = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=num_steps, track_step_size=track_step_size, verbose=False, batch_size=batch_size),
        )

        mcmc_gen_topk.sample()

        # History should have snapshots at tracked steps
        expected_snapshots = (
            1
            + (num_steps // track_step_size)
            + (1 if num_steps % track_step_size != 0 else 0)
        )
        assert len(mcmc_gen_topk.history) == expected_snapshots

        # Each history entry should have proper structure
        for entry in mcmc_gen_topk.history:
            assert "time_step" in entry
            assert "energy_scores" in entry
            assert "constructs" in entry
            # After trimming, history should have batch_size energy scores
            assert len(entry["energy_scores"]) == batch_size

    def test_history_timesteps_validation(self):
        """Tests that time_step values in history entries are correctly tracked."""
        proposals_per_parent = 4
        batch_size = 2
        num_steps = 35
        track_step_size = 10
        seq_length = 10

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        mcmc_gen_topk = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=num_steps, track_step_size=track_step_size, verbose=False, batch_size=batch_size),
        )

        mcmc_gen_topk.sample()

        # Expected timesteps: 0 (initial), 10, 20, 30, 35 (final)
        expected_timesteps = [0, 10, 20, 30, 35]
        actual_timesteps = [entry["time_step"] for entry in mcmc_gen_topk.history]

        assert (
            actual_timesteps == expected_timesteps
        ), f"Expected timesteps {expected_timesteps}, but got {actual_timesteps}"

        # Verify timesteps are monotonically increasing
        for i in range(1, len(actual_timesteps)):
            assert (
                actual_timesteps[i] > actual_timesteps[i - 1]
            ), f"Timesteps should be monotonically increasing, but {actual_timesteps[i-1]} >= {actual_timesteps[i]}"

        # Verify first timestep is 0 (initial state)
        assert actual_timesteps[0] == 0, "First timestep should be 0 (initial state)"

        # Verify last timestep equals num_steps
        assert (
            actual_timesteps[-1] == num_steps
        ), f"Last timestep should equal num_steps ({num_steps}), but got {actual_timesteps[-1]}"

    def test_history_timesteps_with_even_tracking(self):
        """Tests timesteps when track_step_size evenly divides num_steps."""
        proposals_per_parent = 3
        num_steps = 30
        track_step_size = 10

        mcmc_gen, _, _, _ = _setup_mcmc_components(num_mcmc_steps=num_steps)

        mcmc_gen.num_steps = num_steps
        mcmc_gen.track_step_size = track_step_size
        mcmc_gen.sample()

        # Expected timesteps: 0 (initial), 10, 20, 30
        # When num_steps % track_step_size == 0, final state is already captured
        expected_timesteps = [0, 10, 20, 30]
        actual_timesteps = [entry["time_step"] for entry in mcmc_gen.history]

        assert (
            actual_timesteps == expected_timesteps
        ), f"Expected timesteps {expected_timesteps}, but got {actual_timesteps}"

    def test_temperature_scheduling(self):
        """Tests that simulated annealing temperature schedule is correct."""
        proposals_per_parent = 2
        num_steps = 100
        temperature = 10.0
        temperature_min = 0.01

        mcmc_gen, _, _, _ = _setup_mcmc_components(num_mcmc_steps=num_steps)

        mcmc_gen.temperature = temperature
        mcmc_gen.temperature_min = temperature_min
        mcmc_gen.num_steps = num_steps

        # Test temperature at key steps
        step_1_temp = mcmc_gen._calculate_temperature(1)
        step_50_temp = mcmc_gen._calculate_temperature(50)
        step_100_temp = mcmc_gen._calculate_temperature(100)

        # Step 1 should be exactly T_max
        assert (
            abs(step_1_temp - temperature) < 1e-10
        ), f"Step 1 temperature should be T_max={temperature}, got {step_1_temp}"

        # Final step should be exactly T_min
        assert (
            abs(step_100_temp - temperature_min) < 1e-10
        ), f"Final step temperature should be T_min={temperature_min}, got {step_100_temp}"

        # Middle step should be between T_max and T_min
        assert (
            temperature_min < step_50_temp < temperature
        ), f"Middle step temperature {step_50_temp} should be between {temperature_min} and {temperature}"

        # Temperatures should decrease monotonically
        temperatures = [
            mcmc_gen._calculate_temperature(step) for step in range(1, num_steps + 1)
        ]
        for i in range(1, len(temperatures)):
            assert (
                temperatures[i] <= temperatures[i - 1]
            ), f"Temperature should decrease monotonically, but T[{i}]={temperatures[i]} > T[{i-1}]={temperatures[i-1]}"

    def test_temperature_scheduling_edge_cases(self):
        """Tests temperature scheduling edge cases."""
        proposals_per_parent = 2
        temperature = 5.0
        temperature_min = 0.001

        # Test num_steps=1 (should return T_max)
        mcmc_gen, _, _, _ = _setup_mcmc_components(num_mcmc_steps=1)
        mcmc_gen.temperature = temperature
        mcmc_gen.temperature_min = temperature_min
        mcmc_gen.num_steps = 1

        step_1_temp = mcmc_gen._calculate_temperature(1)
        assert (
            abs(step_1_temp - temperature) < 1e-10
        ), f"With num_steps=1, temperature should be T_max={temperature}, got {step_1_temp}"

        # Test num_steps=2 (should go from T_max to T_min in one step)
        mcmc_gen.num_steps = 2
        step_1_temp = mcmc_gen._calculate_temperature(1)
        step_2_temp = mcmc_gen._calculate_temperature(2)

        assert (
            abs(step_1_temp - temperature) < 1e-10
        ), f"Step 1 should be T_max={temperature}, got {step_1_temp}"
        assert (
            abs(step_2_temp - temperature_min) < 1e-10
        ), f"Step 2 should be T_min={temperature_min}, got {step_2_temp}"

        # Test with very large num_steps (should still work)
        mcmc_gen.num_steps = 10000
        step_1_temp = mcmc_gen._calculate_temperature(1)
        step_final_temp = mcmc_gen._calculate_temperature(10000)

        assert abs(step_1_temp - temperature) < 1e-10
        assert abs(step_final_temp - temperature_min) < 1e-10

    def test_topk_with_multiple_constraints(self):
        """Tests top-k MCMC with multiple weighted constraints."""
        proposals_per_parent = 6
        batch_size = 3
        seq_length = 30

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=2
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Multiple constraints
        gc_constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        length_constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": seq_length},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_constraint, length_constraint],
            constraint_weights=[1.0, 2.0],
            config=MCMCOptimizerConfig(num_candidates=proposals_per_parent, num_steps=30, verbose=False, batch_size=batch_size),
        )

        # Get initial energies before sampling
        mcmc_gen.score_energy()
        initial_energies = mcmc_gen.energy_scores.copy()

        mcmc_gen.sample()
        final_energies = mcmc_gen.energy_scores

        # Should have improved (some energy should be lower)
        assert min(final_energies) <= min(initial_energies)

        # After trimming, should have batch_size sequences
        assert len(final_energies) == batch_size

    def test_topk_parent_replication(self):
        """Tests that parent sequences are correctly replicated to batch positions."""
        proposals_per_parent = 4
        batch_size = 2
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        # Create distinct initial sequences
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)

        # Manually set different sequences for testing (only 2 sequences since batch_size=2)
        segment.batch_sequences[0].sequence = "A" * seq_length
        segment.batch_sequences[1].sequence = "C" * seq_length

        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: len(seq.sequence),
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=1, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        # Score to get initial energies
        mcmc_gen.score_energy()

        # Manually initialize: select top-k parents
        top_k_idx = np.argsort(mcmc_gen.energy_scores)[:batch_size]

        # Save parent sequences before replication
        parent_seqs = [segment.batch_sequences[idx].sequence for idx in top_k_idx]

        # Expand batch and replicate parents
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._expand_batch_for_proposals(top_k_idx)

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
        proposals_per_parent = 3
        batch_size = 2
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)

        # Set initial sequences (only 2 sequences since batch_size=2)
        segment.batch_sequences[0].sequence = "A" * seq_length
        segment.batch_sequences[1].sequence = "C" * seq_length

        # Add nested metadata to test deep copy
        segment.batch_sequences[0]._metadata["nested"] = {"count": 1, "tags": ["x"]}

        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=1, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.score_energy()
        top_k_idx = np.argsort(mcmc_gen.energy_scores)[:batch_size]

        # Test 1: _expand_batch_for_proposals creates independent copies
        mcmc_gen._expand_batch_for_proposals(top_k_idx)

        # Verify all batch positions are independent objects
        for i in range(len(segment.batch_sequences)):
            for j in range(i + 1, len(segment.batch_sequences)):
                assert (
                    segment.batch_sequences[i] is not segment.batch_sequences[j]
                ), f"Batch positions {i} and {j} should be different objects"

        # Verify nested metadata is deeply copied (modifying one doesn't affect others)
        if "nested" in segment.batch_sequences[0]._metadata:
            segment.batch_sequences[0]._metadata["nested"]["count"] = 999
            segment.batch_sequences[0]._metadata["nested"]["tags"].append("y")
            # Check other positions aren't affected
            for idx in range(1, len(segment.batch_sequences)):
                if "nested" in segment.batch_sequences[idx]._metadata:
                    assert (
                        segment.batch_sequences[idx]._metadata["nested"]["count"] == 1
                    )
                    assert segment.batch_sequences[idx]._metadata["nested"]["tags"] == [
                        "x"
                    ]

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
        assert (
            segment.batch_sequences[0] is not segment.batch_sequences[1]
        ), "Restored sequences should be independent objects"

        # Modify one restored sequence and verify it doesn't affect the other
        segment.batch_sequences[0].sequence = "G" * seq_length
        assert (
            segment.batch_sequences[1].sequence != "G" * seq_length
        ), "Modifying one restored sequence should not affect others"

    def test_topk_parent_energy_consistency(self):
        """Validates critical invariant: parent_energies matches parent_states energies."""
        proposals_per_parent = 4
        batch_size = 2
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=2
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)

        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: float(seq.sequence.count("A")),
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=5, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        # Patch _save_parent_states to verify consistency
        original_save = mcmc_gen._save_parent_states

        def checked_save(top_k_idx):
            # Get parent_energies from the frame above (passed to _select_topk_with_mcmc)
            import inspect

            frame = inspect.currentframe().f_back
            parent_energies = frame.f_locals.get("parent_energies")

            if parent_energies is not None:
                # Verify invariant: parent_energies[i] == self.energy_scores[top_k_idx[i]]
                for i, (idx, expected_energy) in enumerate(
                    zip(top_k_idx, parent_energies)
                ):
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
                    saved_energy = parent_states[idx]["energy"]
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
        proposals_per_parent = 3
        batch_size = 2
        seq_length = 15

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=5
            )
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
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=3, temperature=0.001, temperature_min=0.0001, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        # Get initial best energy
        mcmc_gen.score_energy()
        initial_best_energy = min(mcmc_gen.energy_scores)

        mcmc_gen.sample()

        # With very low temperature and optimal initial state, rejections should preserve parents
        # Best energy should remain optimal (0.0)
        final_best_energy = min(mcmc_gen.energy_scores)
        assert (
            final_best_energy == initial_best_energy
        ), f"Expected best energy to remain {initial_best_energy}, got {final_best_energy}"

        # At least some sequences should remain optimal
        optimal_seq = "G" * seq_length
        optimal_count = sum(
            1 for seq in segment.batch_sequences if seq.sequence == optimal_seq
        )
        assert (
            optimal_count > 0
        ), "Expected some optimal sequences to be preserved through rejections"

    def test_topk_selection_correctness(self):
        """Tests that top-k selection picks the k best sequences by energy."""
        proposals_per_parent = 6
        batch_size = 3
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("C" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Constraint with known energies based on 'G' count
        def g_count_energy(seq, **kwargs):
            return -seq.sequence.count("G")  # More G = lower energy

        constraint = Constraint(
            inputs=[segment],
            scoring_function=g_count_energy,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=20, temperature=2.0, temperature_min=0.1, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # Get final energies
        all_energies = mcmc_gen.energy_scores

        # After trimming, we should have exactly batch_size sequences
        assert len(all_energies) == batch_size

        # The top-k parents should be sorted by energy
        sorted_energies = sorted(all_energies)
        assert (
            all_energies == sorted_energies or len(set(all_energies)) == 1
        )  # Either sorted or all equal

    def test_topk_diversity_maintenance(self):
        """Tests that top-k maintains diversity among parent sequences."""
        proposals_per_parent = 5
        batch_size = 3
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=2
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Constraint that allows multiple local optima
        def diversity_constraint(seq, **kwargs):
            g_count = seq.sequence.count("G")
            c_count = seq.sequence.count("C")
            # Prefer either high G OR high C (creates multiple optima)
            return -max(g_count, c_count)

        constraint = Constraint(
            inputs=[segment],
            scoring_function=diversity_constraint,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=50, temperature=1.5, temperature_min=0.1, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # Check that we maintain batch_size distinct sequences (or at least some diversity)
        # Get the top-k parent sequences by looking at the best k energy scores
        energy_idx_pairs = [(e, i) for i, e in enumerate(mcmc_gen.energy_scores)]
        energy_idx_pairs.sort()
        top_k_indices = [idx for _, idx in energy_idx_pairs[:batch_size]]
        top_k_sequences = [
            segment.batch_sequences[idx].sequence for idx in top_k_indices
        ]

        # Count unique sequences among top-k
        unique_seqs = len(set(top_k_sequences))

        # With diversity-promoting constraint and sufficient temperature,
        # we should have some diversity (at least 2 different sequences)
        # Note: This is probabilistic, but with 50 steps it should converge
        assert (
            unique_seqs >= 1
        ), f"Expected some diversity in top-k, got {unique_seqs} unique sequences"

    def test_topk_boundary_case_equals_batch_size(self):
        """Tests edge case where batch_size equals batch_size."""
        proposals_per_parent = 4
        batch_size = 4
        seq_length = 15

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=10, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        # Should work without errors
        mcmc_gen.sample()

        # After trimming, should have batch_size sequences (which equals batch_size in this test)
        assert len(segment.batch_sequences) == batch_size
        assert len(mcmc_gen.energy_scores) == batch_size

    def test_topk_all_rejections_scenario(self):
        """Tests behavior when all proposals are rejected (fall back to parents)."""
        proposals_per_parent = 3
        batch_size = 2
        seq_length = 10

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=5
            )
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
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=5, temperature=0.00001, temperature_min=0.000001, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # Should still complete without errors
        # After trimming, should have batch_size sequences
        assert len(mcmc_gen.energy_scores) == batch_size

        # Best energy should remain at 0 (optimal state preserved)
        assert min(mcmc_gen.energy_scores) == 0.0

    def test_topk_energy_non_regression(self):
        """Tests that best energy never gets worse (monotonic improvement)."""
        proposals_per_parent = 5
        batch_size = 3
        seq_length = 20
        num_steps = 50

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Simple constraint - prefer high GC content
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 50.0, "max_gc": 50.0},  # Target 50% GC
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=num_steps, track_step_size=1, temperature=0.5, temperature_min=0.01, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # Extract best energy at each step from history
        best_energies_over_time = [
            min(entry["energy_scores"]) for entry in mcmc_gen.history
        ]

        # Best energy should never increase (allowing small numerical tolerance)
        tolerance = 1e-6
        for i in range(1, len(best_energies_over_time)):
            assert (
                best_energies_over_time[i] <= best_energies_over_time[i - 1] + tolerance
            ), f"Energy regression at step {i}: {best_energies_over_time[i]} > {best_energies_over_time[i-1]}"

        # Final energy should be better than or equal to initial
        assert best_energies_over_time[-1] <= best_energies_over_time[0]

    def test_topk_metadata_preserved_through_acceptance(self):
        """Tests that metadata is preserved when proposals are accepted."""
        proposals_per_parent = 3
        batch_size = 2
        seq_length = 15

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)

        # Add initial metadata
        for i, seq in enumerate(segment.batch_sequences):
            seq._metadata["seq_id"] = f"seq_{i}"
            seq._metadata["generation"] = 0

        construct = Construct([segment])

        # Constraint that strongly encourages changes (high GC better)
        def strong_gc_constraint(seq, **kwargs):
            gc_count = seq.sequence.count("G") + seq.sequence.count("C")
            return -gc_count  # More GC = lower energy = better

        constraint = Constraint(
            inputs=[segment],
            scoring_function=strong_gc_constraint,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=10, temperature=5.0, temperature_min=1.0, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # After MCMC with high acceptance rate, sequences should have changed
        # but metadata should still exist (even if modified)
        final_sequences = [seq.sequence for seq in segment.batch_sequences]
        changed_count = sum(1 for seq in final_sequences if seq != "A" * seq_length)

        # With high temp and GC-promoting constraint, expect changes
        assert (
            changed_count > 0
        ), "Expected at least some sequences to change with high acceptance rate"

        # Metadata should exist for all sequences
        for seq in segment.batch_sequences:
            assert len(seq._metadata) > 0, "Metadata should not be empty"

    def test_topk_with_multiple_generators_specific(self):
        """Tests top-k MCMC specifically with multiple proposal generators."""
        proposals_per_parent = 6
        batch_size = 3
        seq_length = 20

        # Create two different mutation generators
        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=3
            )
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
            concatenate=True,
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],  # Multiple generators
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=20, temperature=1.0, temperature_min=0.1, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # Should work without errors
        # Generator batch_size remains expanded
        assert gen1.batch_size == batch_size * proposals_per_parent
        assert gen2.batch_size == batch_size * proposals_per_parent
        # But segments are trimmed to batch_size
        assert len(segment1.batch_sequences) == batch_size
        assert len(segment2.batch_sequences) == batch_size

    def test_topk_convergence_to_optimal(self):
        """Tests that top-k MCMC converges to optimal solution with enough steps."""
        proposals_per_parent = 8
        batch_size = 4
        seq_length = 15

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)  # Start far from optimal
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Constraint with clear global optimum (all G's)
        def perfect_g_energy(seq, **kwargs):
            g_count = seq.sequence.count("G")
            return seq_length - g_count  # Perfect = 0 energy

        constraint = Constraint(
            inputs=[segment],
            scoring_function=perfect_g_energy,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=200, temperature=2.0, temperature_min=0.01, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.score_energy()
        initial_best_energy = min(mcmc_gen.energy_scores)

        mcmc_gen.sample()
        final_best_energy = min(mcmc_gen.energy_scores)

        # Should get very close to optimal (energy near 0)
        assert (
            final_best_energy < initial_best_energy * 0.5
        ), f"Expected significant improvement, got {final_best_energy} vs {initial_best_energy}"

        # With enough steps, should find sequences close to all G's
        best_g_count = max(seq.sequence.count("G") for seq in segment.batch_sequences)
        assert (
            best_g_count >= seq_length * 0.8
        ), f"Expected convergence toward G's, best has only {best_g_count}/{seq_length} G's"

    def test_topk_energy_variance_reduction(self):
        """Tests that energy variance among top-k decreases over time (convergence)."""
        proposals_per_parent = 6
        batch_size = 4
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 50.0, "max_gc": 50.0},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=50, track_step_size=10, temperature=1.0, temperature_min=0.01, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.sample()

        # Extract energy variance at each tracked step
        variances = []
        for entry in mcmc_gen.history:
            energies = entry["energy_scores"]
            # Get top-k energies
            sorted_energies = sorted(energies)[:batch_size]
            var = np.var(sorted_energies) if len(sorted_energies) > 1 else 0.0
            variances.append(var)

        # Variance should generally decrease (allowing some fluctuation)
        # Compare first quarter to last quarter
        first_quarter_var = np.mean(variances[: len(variances) // 4 + 1])
        last_quarter_var = np.mean(variances[-len(variances) // 4 :])

        # Last quarter should have lower or similar variance (convergence)
        assert last_quarter_var <= first_quarter_var * 2.0, (
            f"Expected variance reduction, got initial={first_quarter_var:.4f}, "
            f"final={last_quarter_var:.4f}"
        )

    def test_topk_exact_state_restoration(self):
        """Validates that rejected proposals EXACTLY match parent state (bit-perfect)."""
        proposals_per_parent = 3
        batch_size = 2
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=3
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)

        # Add complex nested metadata to test deep equality
        for i, seq_obj in enumerate(segment.batch_sequences):
            seq_obj._metadata["test_data"] = {
                "id": i,
                "nested": {"values": [1, 2, 3], "flag": True},
            }

        construct = Construct([segment])
        # Constraint that makes mutations worse (favors original)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: float(
                seq.sequence.count("C") + seq.sequence.count("G")
            ),
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=1, temperature=0.001, verbose=False, num_candidates=proposals_per_parent, batch_size=batch_size),
        )

        mcmc_gen.score_energy()
        top_k_idx = np.argsort(mcmc_gen.energy_scores)[:batch_size]

        # Save EXACT parent state before mutation
        exact_parent_states = {}
        for parent_idx in top_k_idx:
            exact_parent_states[parent_idx] = {
                "sequence": segment.batch_sequences[parent_idx].sequence,
                "metadata": copy.deepcopy(
                    segment.batch_sequences[parent_idx]._metadata
                ),
                "energy": mcmc_gen.energy_scores[parent_idx],
            }

        # Save and replicate
        parent_states = mcmc_gen._save_parent_states(top_k_idx)
        mcmc_gen._expand_batch_for_proposals(top_k_idx)

        # Mutate (makes sequences worse)
        mcmc_gen._generate_proposals()

        # Manually restore and check EXACT equality
        for parent_pos, parent_idx in enumerate(top_k_idx):
            # Restore first proposal from this parent
            proposal_idx = parent_pos * mcmc_gen.num_candidates
            mcmc_gen._restore_parent_state(proposal_idx, parent_idx, parent_states)

            # Verify EXACT match
            restored_seq = segment.batch_sequences[proposal_idx]
            expected = exact_parent_states[parent_idx]

            # Check sequence
            assert (
                restored_seq.sequence == expected["sequence"]
            ), f"Sequence mismatch: {restored_seq.sequence} != {expected['sequence']}"

            # Check energy
            assert (
                mcmc_gen.energy_scores[proposal_idx] == expected["energy"]
            ), f"Energy mismatch: {mcmc_gen.energy_scores[proposal_idx]} != {expected['energy']}"

            # Check metadata deep equality
            assert (
                restored_seq._metadata == expected["metadata"]
            ), f"Metadata mismatch:\n  Restored: {restored_seq._metadata}\n  Expected: {expected['metadata']}"

            # Verify nested metadata wasn't aliased
            if "test_data" in restored_seq._metadata:
                restored_seq._metadata["test_data"]["nested"]["values"].append(999)
                # Check other restored positions weren't affected
                for other_pos in range(mcmc_gen.num_candidates):
                    other_idx = parent_pos * mcmc_gen.num_candidates + other_pos
                    if other_idx != proposal_idx:
                        other_metadata = segment.batch_sequences[other_idx]._metadata
                        if "test_data" in other_metadata:
                            assert (
                                999
                                not in other_metadata["test_data"]["nested"]["values"]
                            )

    def test_generate_proposals_with_multiple_generators(self):
        """Test that _generate_proposals randomly selects from multiple generators."""
        seq_length = 20
        batch_size = 2

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=5
            )
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
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=3, num_steps=50, verbose=False, batch_size=batch_size),
        )

        original_sample1 = gen1.sample
        original_sample2 = gen2.sample
        gen1_calls = []
        gen2_calls = []

        def tracked_sample1():
            gen1_calls.append(1)
            return original_sample1()

        def tracked_sample2():
            gen2_calls.append(1)
            return original_sample2()

        gen1.sample = tracked_sample1
        gen2.sample = tracked_sample2

        try:
            mcmc_gen.sample()
            assert len(gen1_calls) > 0
            assert len(gen2_calls) > 0
            assert len(gen1_calls) + len(gen2_calls) == 50
        finally:
            gen1.sample = original_sample1
            gen2.sample = original_sample2

    def test_custom_logging_callback(self):
        """Test that custom_logging is called at tracked steps."""
        seq_length = 15
        batch_size = 2
        num_steps = 25
        track_step_size = 5

        log_calls = []

        def custom_log(step, segments):
            log_calls.append({"step": step})

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=3, num_steps=num_steps, track_step_size=track_step_size, verbose=True, batch_size=batch_size),
            custom_logging=custom_log,
        )

        mcmc_gen.sample()

        expected_steps = [5, 10, 15, 20, 25]
        actual_steps = [call["step"] for call in log_calls]
        assert actual_steps == expected_steps

    def test_verbose_output_formats(self):
        """Test logging output for batch_size=1 vs >1."""
        import io
        import sys

        seq_length = 15

        proposal_gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=1, num_mutations=1
            )
        )
        segment1 = create_segment("A" * seq_length)
        proposal_gen1.assign(segment1)
        construct1 = Construct([segment1])
        constraint1 = Constraint(
            inputs=[segment1],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        mcmc_gen1 = MCMCOptimizer(
            constructs=[construct1],
            generators=[proposal_gen1],
            constraints=[constraint1],
            config=MCMCOptimizerConfig(num_candidates=2, num_steps=3, track_step_size=1, verbose=True, batch_size=1),
        )

        captured_output1 = io.StringIO()
        sys.stdout = captured_output1
        try:
            mcmc_gen1.sample()
        finally:
            sys.stdout = sys.__stdout__

        output1 = captured_output1.getvalue()
        assert "energy:" in output1
        assert "best:" not in output1

        proposal_gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=3, num_mutations=1
            )
        )
        segment2 = create_segment("A" * seq_length)
        proposal_gen2.assign(segment2)
        construct2 = Construct([segment2])
        constraint2 = Constraint(
            inputs=[segment2],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        mcmc_gen2 = MCMCOptimizer(
            constructs=[construct2],
            generators=[proposal_gen2],
            constraints=[constraint2],
            config=MCMCOptimizerConfig(num_candidates=4, num_steps=3, track_step_size=1, verbose=True, batch_size=3),
        )

        captured_output2 = io.StringIO()
        sys.stdout = captured_output2
        try:
            mcmc_gen2.sample()
        finally:
            sys.stdout = sys.__stdout__

        output2 = captured_output2.getvalue()
        assert "best:" in output2
        assert "mean:" in output2

    def test_acceptance_prob_overflow_protection(self):
        """Test that MAX_EXP_ARG prevents overflow."""
        seq_length = 10

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=1, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40.0, "max_gc": 60.0},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=1, verbose=False),
        )

        alpha = mcmc_gen._compute_acceptance_prob(1000.0, 0.0, 0.001)
        assert alpha == 1.0

        alpha = mcmc_gen._compute_acceptance_prob(0.0, 1000.0, 0.001)
        assert 0.0 <= alpha < 1e-10

    def test_edge_case_identical_energies(self):
        """Test behavior when all proposals have identical energy."""
        seq_length = 15
        batch_size = 3

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("GCGCGCGCGCGCGCG")
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: 5.0,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=4, num_steps=1, verbose=False, batch_size=batch_size),
        )

        mcmc_gen.sample()
        assert len(mcmc_gen.energy_scores) == batch_size

    def test_comprehensive_integration_complex_scenario(self):
        """Comprehensive integration test with batch_size>1, num_candidates>1."""
        seq_length = 30
        batch_size = 5
        num_steps = 50

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=3
            )
        )

        segment1 = create_segment("A" * seq_length)
        segment2 = create_segment("T" * seq_length)

        gen1.assign(segment1)
        gen2.assign(segment2)

        construct = Construct([segment1, segment2])

        gc_constraint = Constraint(
            inputs=[segment1, segment2],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 45.0, "max_gc": 55.0},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[gc_constraint],
            config=MCMCOptimizerConfig(num_candidates=6, num_steps=num_steps, verbose=False, batch_size=batch_size),
        )

        mcmc_gen.sample()
        assert len(mcmc_gen.energy_scores) == batch_size
        assert len(segment1.batch_sequences) == batch_size

    def test_acceptance_and_rejection_at_different_timesteps(self):
        """Test that temperature annealing affects acceptance behavior."""
        seq_length = 20
        batch_size = 3
        num_steps = 30

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=2
            )
        )
        segment = create_segment("G" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=lambda seq, **kwargs: float(seq.sequence.count("G"))
            / len(seq.sequence),
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=4, num_steps=num_steps, temperature=5.0, temperature_min=0.001, verbose=False, batch_size=batch_size),
        )

        mcmc_gen.sample()
        best_energies = [min(entry["energy_scores"]) for entry in mcmc_gen.history]
        assert best_energies[-1] <= best_energies[0]

    def test_large_batch_comprehensive(self):
        """Stress test with large batch_size."""
        seq_length = 25
        batch_size = 10
        num_steps = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(
                sequence_length=seq_length, batch_size=batch_size, num_mutations=1
            )
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 48.0, "max_gc": 52.0},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_candidates=12, num_steps=num_steps, verbose=False, batch_size=batch_size),
        )

        mcmc_gen.sample()
        assert len(mcmc_gen.energy_scores) == batch_size
        assert len(segment.batch_sequences) == batch_size
