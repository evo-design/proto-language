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
    num_selected: int = 1,
    num_candidates: int = None,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    num_mcmc_steps: int = 10,
):
    """Helper function to set up a basic MCMC Optimizer for testing."""
    # 1. Create the proposal generator and the segment it will modify
    proposal_gen = UniformMutationGenerator(
        UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
    )
    segment = create_segment("A" * seq_length)
    proposal_gen.assign(segment)

    # 2. Create the construct and constraint
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
        num_selected=num_selected,
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
        assert mcmc_gen.num_selected == 1
        assert mcmc_gen.num_candidates == 1  # Defaults to num_selected

        # Test validation errors - unassigned generator
        unassigned_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=10, num_mutations=1)
        )
        test_segment = create_segment("A" * 10)
        dummy_constraint = Constraint(
            inputs=[test_segment],
            scoring_function=lambda seq, **kwargs: 0.0,
            scoring_function_config={},
        )
        with pytest.raises(RuntimeError, match="has not been assigned"):
            MCMCOptimizer(
                constructs=[Construct([test_segment])],
                generators=[unassigned_gen],
                constraints=[dummy_constraint],
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

    def test_config_validation(self):
        """Tests MCMCOptimizerConfig validation."""
        from pydantic import ValidationError

        # Valid configs
        config = MCMCOptimizerConfig(num_selected=5, num_candidates=10)
        assert config.num_selected == 5
        assert config.num_candidates == 10

        # num_selected > num_candidates should fail
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_selected=10, num_candidates=5)

        # temperature_min >= temperature should fail
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(temperature=1.0, temperature_min=1.0)

        # Negative values should fail
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_selected=-1)

    def test_score_energy(self):
        """Tests the score_energy method."""
        mcmc_gen, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))

        # Test with a sequence within target GC range
        segment.candidate_sequences[0].sequence = "GCGCGAATTA"  # 50% GC
        mcmc_gen.score_energy()
        assert len(mcmc_gen.energy_scores) == 1
        assert mcmc_gen.energy_scores[0] == 0.0

        # Test with a sequence below target range
        segment.candidate_sequences[0].sequence = "GCTTAATTAA"  # 20% GC
        mcmc_gen.score_energy()
        expected_score = (40.0 - 20.0) / 40.0  # 0.5
        assert abs(mcmc_gen.energy_scores[0] - expected_score) < 1e-9

    def test_score_energy_multiply(self):
        """Tests the score_energy method with operation='multiply'."""
        mcmc_gen, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))
        segment.candidate_sequences[0].sequence = "GCTTAATTAA"  # 20% GC -> score 0.5

        # With one constraint, multiply and add should be the same
        mcmc_gen.score_energy(operation="add")
        energy_add = mcmc_gen.energy_scores[0]
        mcmc_gen.score_energy(operation="multiply")
        energy_mul = mcmc_gen.energy_scores[0]
        assert abs(energy_add - 0.5) < 1e-9
        assert abs(energy_mul - 0.5) < 1e-9

    def test_sample_improves_energy(self):
        """Tests that sampling can improve the energy score over time."""
        mcmc_gen, _, _, segment = _setup_mcmc_components(
            seq_length=50,
            gc_target_range=(80.0, 90.0),  # Encourage high GC
            num_mcmc_steps=100,
        )

        # Start with a bad sequence
        initial_seq = "A" * 50
        for seq in segment.selected_sequences:
            seq.sequence = initial_seq
        
        # Score initial state
        for i in range(mcmc_gen.num_selected):
            mcmc_gen.segments[0].candidate_sequences[i] = copy.deepcopy(
                mcmc_gen.segments[0].selected_sequences[i]
            )
        mcmc_gen.score_energy()
        initial_energy = mcmc_gen.energy_scores[0]
        assert initial_energy > 0.99  # Should be max penalty

        # Sample and check for improvement
        mcmc_gen.sample()
        final_energy = min(mcmc_gen.energy_scores)

        assert final_energy < initial_energy
        assert len(mcmc_gen.history) > 1  # Check history is tracked

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
            [segment], gc_content_constraint, GCContentConfig(min_gc=40.0, max_gc=60.0)
        )
        len_con = Constraint(
            [segment], sequence_length_constraint, SequenceLengthConfig(target_length=seq_len)
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_con, len_con],
            constraint_weights=[1.0, 2.0],  # Weight length more
            config=MCMCOptimizerConfig(num_steps=1, verbose=False),
        )

        segment.candidate_sequences[0].sequence = "A" * 20  # Violates length and GC
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
        seq_len = 50
        mut_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_len, num_mutations=1)
        )
        segment1 = create_segment("A" * seq_len)
        mut_gen.assign(segment1)

        inv_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_len, num_mutations=3)
        )
        segment2 = create_segment("C" * seq_len)
        inv_gen.assign(segment2)

        construct = Construct([segment1, segment2])
        constraint = Constraint(
            inputs=[segment1, segment2],
            scoring_function=sequence_length_constraint,
            scoring_function_config=SequenceLengthConfig(target_length=seq_len * 2),
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[mut_gen, inv_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_steps=20, verbose=False),
        )

        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence

        mcmc_gen.sample()

        final_seq1 = segment1[0].sequence
        final_seq2 = segment2[0].sequence

        # At least one sequence should be modified
        assert initial_seq1 != final_seq1 or initial_seq2 != final_seq2

    def test_topk_initialization(self):
        """Tests initialization of top-k MCMC with num_selected > 1."""
        mcmc_gen, _, _, _ = _setup_mcmc_components(
            num_selected=3, num_candidates=10
        )

        assert mcmc_gen.num_selected == 3
        # mcmc_width is the number of proposals per selected sequence
        assert mcmc_gen.mcmc_width == 10
        # num_candidates is the total pool size (num_selected * mcmc_width)
        assert mcmc_gen.num_candidates == 30

    def test_topk_maintains_k_sequences(self):
        """Tests that top-k MCMC maintains exactly k sequences."""
        num_selected = 3
        num_candidates = 4
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        segment = create_segment("ATCGATCGATCGATCGATCG")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_candidates=num_candidates,
                num_steps=20,
                temperature=1.0,
                temperature_min=0.01,
                verbose=False,
            ),
        )

        mcmc_gen.sample()

        # Should maintain num_selected sequences
        assert len(mcmc_gen.energy_scores) == num_selected
        assert len(segment.selected_sequences) == num_selected

    def test_history_tracking(self):
        """Tests that history is properly tracked during MCMC."""
        num_selected = 2
        num_steps = 30
        track_step_size = 10
        seq_length = 10

        mcmc_gen, _, _, _ = _setup_mcmc_components(
            seq_length=seq_length,
            num_selected=num_selected,
            num_mcmc_steps=num_steps,
        )
        mcmc_gen.track_step_size = track_step_size

        mcmc_gen.sample()

        # History should have snapshots: 0 (initial), 10, 20, 30 (final)
        expected_snapshots = 4
        assert len(mcmc_gen.history) == expected_snapshots

        # Each history entry should have proper structure
        for entry in mcmc_gen.history:
            assert "time_step" in entry
            assert "energy_scores" in entry
            assert "constructs" in entry
            assert len(entry["energy_scores"]) == num_selected

    def test_history_timesteps_validation(self):
        """Tests that time_step values in history entries are correctly tracked."""
        num_selected = 2
        num_steps = 35
        track_step_size = 10

        mcmc_gen, _, _, _ = _setup_mcmc_components(
            num_selected=num_selected, num_mcmc_steps=num_steps
        )
        mcmc_gen.track_step_size = track_step_size

        mcmc_gen.sample()

        # Expected timesteps: 0 (initial), 10, 20, 30, 35 (final)
        expected_timesteps = [0, 10, 20, 30, 35]
        actual_timesteps = [entry["time_step"] for entry in mcmc_gen.history]

        assert actual_timesteps == expected_timesteps

        # Verify timesteps are monotonically increasing
        for i in range(1, len(actual_timesteps)):
            assert actual_timesteps[i] > actual_timesteps[i - 1]

    def test_temperature_scheduling(self):
        """Tests that simulated annealing temperature schedule is correct."""
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
        assert abs(step_1_temp - temperature) < 1e-10

        # Final step should be exactly T_min
        assert abs(step_100_temp - temperature_min) < 1e-10

        # Middle step should be between T_max and T_min
        assert temperature_min < step_50_temp < temperature

        # Temperatures should decrease monotonically
        temperatures = [
            mcmc_gen._calculate_temperature(step) for step in range(1, num_steps + 1)
        ]
        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i - 1]

    def test_temperature_scheduling_edge_cases(self):
        """Tests temperature scheduling edge cases."""
        temperature = 5.0
        temperature_min = 0.001

        # Test num_steps=1 (should return T_max)
        mcmc_gen, _, _, _ = _setup_mcmc_components(num_mcmc_steps=1)
        mcmc_gen.temperature = temperature
        mcmc_gen.temperature_min = temperature_min
        mcmc_gen.num_steps = 1

        step_1_temp = mcmc_gen._calculate_temperature(1)
        assert abs(step_1_temp - temperature) < 1e-10

        # Test num_steps=2 (should go from T_max to T_min)
        mcmc_gen.num_steps = 2
        step_1_temp = mcmc_gen._calculate_temperature(1)
        step_2_temp = mcmc_gen._calculate_temperature(2)

        assert abs(step_1_temp - temperature) < 1e-10
        assert abs(step_2_temp - temperature_min) < 1e-10

    def test_mcmc_acceptance_probability(self):
        """Tests Metropolis-Hastings acceptance probability computation."""
        mcmc_gen, _, _, _ = _setup_mcmc_components()

        # Better proposal (lower energy) should always be accepted
        alpha = mcmc_gen._compute_mcmc_acceptance_prob(1.0, 0.5, 1)
        assert alpha == 1.0

        # Equal energy should be accepted
        alpha = mcmc_gen._compute_mcmc_acceptance_prob(0.5, 0.5, 1)
        assert alpha == 1.0

        # Worse proposal should have probability < 1
        alpha = mcmc_gen._compute_mcmc_acceptance_prob(0.5, 1.0, 1)
        assert 0.0 < alpha < 1.0

    def test_overflow_protection(self):
        """Test that MAX_EXP_ARG prevents overflow in acceptance computation."""
        mcmc_gen, _, _, _ = _setup_mcmc_components()

        # Very large energy improvement should be clamped to 1.0
        alpha = mcmc_gen._compute_mcmc_acceptance_prob(1000.0, 0.0, 1)
        assert alpha == 1.0

        # Very large energy increase should give very small probability
        alpha = mcmc_gen._compute_mcmc_acceptance_prob(0.0, 1000.0, 1)
        assert 0.0 <= alpha < 1e-10

    def test_energy_non_regression(self):
        """Tests that best energy never gets worse (monotonic improvement)."""
        num_selected = 3
        num_candidates = 5
        seq_length = 20
        num_steps = 50

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=50.0, max_gc=50.0),
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_candidates=num_candidates,
                num_steps=num_steps,
                track_step_size=1,
                temperature=0.5,
                temperature_min=0.01,
                verbose=False,
            ),
        )

        mcmc_gen.sample()

        # Extract best energy at each step from history
        best_energies_over_time = [
            min(entry["energy_scores"]) for entry in mcmc_gen.history
        ]

        # Best energy should never increase
        tolerance = 1e-6
        for i in range(1, len(best_energies_over_time)):
            assert best_energies_over_time[i] <= best_energies_over_time[i - 1] + tolerance

    def test_convergence_to_optimal(self):
        """Tests that MCMC converges toward optimal solution with enough steps."""
        num_selected = 3
        num_candidates = 8
        seq_length = 15

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Constraint with clear optimum (all G's)
        def perfect_g_energy(seq, config=None):
            g_count = seq.sequence.count("G")
            return seq_length - g_count

        constraint = Constraint(
            inputs=[segment],
            scoring_function=perfect_g_energy,
            scoring_function_config={},
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_candidates=num_candidates,
                num_steps=200,
                temperature=2.0,
                temperature_min=0.01,
                verbose=False,
            ),
        )

        # Score initial state
        for i in range(mcmc_gen.num_selected):
            mcmc_gen.segments[0].candidate_sequences[i] = copy.deepcopy(
                mcmc_gen.segments[0].selected_sequences[i]
            )
        mcmc_gen.score_energy()
        initial_best_energy = min(mcmc_gen.energy_scores[:num_selected])

        mcmc_gen.sample()
        final_best_energy = min(mcmc_gen.energy_scores)

        # Should get significant improvement
        assert final_best_energy < initial_best_energy * 0.5

    def test_custom_logging_callback(self):
        """Test that custom_logging is called at tracked steps."""
        seq_length = 15
        num_steps = 25
        track_step_size = 5

        log_calls = []

        def custom_log(step, segments):
            log_calls.append({"step": step})

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_steps=num_steps,
                track_step_size=track_step_size,
                verbose=True,
            ),
            custom_logging=custom_log,
        )

        mcmc_gen.sample()

        expected_steps = [5, 10, 15, 20, 25]
        actual_steps = [call["step"] for call in log_calls]
        assert actual_steps == expected_steps

    def test_verbose_output_formats(self):
        """Test logging output for num_selected=1 vs >1."""
        import io
        import sys

        seq_length = 15

        # Test num_selected=1 (should show "energy:")
        proposal_gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        segment1 = create_segment("A" * seq_length)
        proposal_gen1.assign(segment1)
        construct1 = Construct([segment1])
        constraint1 = Constraint(
            inputs=[segment1],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        mcmc_gen1 = MCMCOptimizer(
            constructs=[construct1],
            generators=[proposal_gen1],
            constraints=[constraint1],
            config=MCMCOptimizerConfig(
                num_selected=1, num_steps=3, track_step_size=1, verbose=True
            ),
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

        # Test num_selected>1 (should show "best:", "mean:", etc.)
        proposal_gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        segment2 = create_segment("A" * seq_length)
        proposal_gen2.assign(segment2)
        construct2 = Construct([segment2])
        constraint2 = Constraint(
            inputs=[segment2],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        mcmc_gen2 = MCMCOptimizer(
            constructs=[construct2],
            generators=[proposal_gen2],
            constraints=[constraint2],
            config=MCMCOptimizerConfig(
                num_selected=3, num_steps=3, track_step_size=1, verbose=True
            ),
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

    def test_deepcopy_independence(self):
        """Tests that deepcopy ensures independent Sequence objects."""
        num_selected = 2
        num_candidates = 3
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        segment = create_segment("A" * seq_length)
        proposal_gen.assign(segment)

        # Add metadata to test deep copy
        for i, seq in enumerate(segment.selected_sequences):
            seq._metadata["seq_id"] = f"seq_{i}"
            seq._metadata["nested"] = {"count": i, "tags": [f"tag_{i}"]}

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
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_candidates=num_candidates,
                num_steps=1,
                verbose=False,
            ),
        )

        mcmc_gen.sample()

        # Verify sequences are independent objects
        for i in range(len(segment.selected_sequences)):
            for j in range(i + 1, len(segment.selected_sequences)):
                assert segment.selected_sequences[i] is not segment.selected_sequences[j]

    def test_comprehensive_integration(self):
        """Comprehensive integration test with num_selected>1, num_candidates>1."""
        seq_length = 30
        num_selected = 5
        num_candidates = 6
        num_steps = 50

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=1)
        )
        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(sequence_length=seq_length, num_mutations=3)
        )

        segment1 = create_segment("A" * seq_length)
        segment2 = create_segment("T" * seq_length)

        gen1.assign(segment1)
        gen2.assign(segment2)

        construct = Construct([segment1, segment2])

        gc_constraint = Constraint(
            inputs=[segment1, segment2],
            scoring_function=gc_content_constraint,
            scoring_function_config=GCContentConfig(min_gc=45.0, max_gc=55.0),
        )

        mcmc_gen = MCMCOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[gc_constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_candidates=num_candidates,
                num_steps=num_steps,
                verbose=False,
            ),
        )

        mcmc_gen.sample()
        
        assert len(mcmc_gen.energy_scores) == num_selected
        assert len(segment1.selected_sequences) == num_selected
        assert len(segment2.selected_sequences) == num_selected
