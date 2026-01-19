from __future__ import annotations
import pytest
import copy
from typing import Tuple

from pydantic import BaseModel
from proto_language.language.core import (
    Construct,
    Segment,
    Constraint,
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


# Empty config for test constraints
class EmptyConfig(BaseModel):
    pass


def _setup_mcmc_components(
    seq_length: int = 10,
    num_selected: int = 1,
    mcmc_width: int = None,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    num_mcmc_steps: int = 10,
):
    """Helper function to set up a basic MCMC Optimizer for testing."""
    # 1. Create the proposal generator and the segment it will modify
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    proposal_gen = UniformMutationGenerator(
        UniformMutationGeneratorConfig(num_mutations=1)
    )
    proposal_gen.assign(segment)

    # 2. Create the construct and constraint
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(
            min_gc=gc_target_range[0],
            max_gc=gc_target_range[1],
        ),
    )

    # 3. Create the MCMC Optimizer config (mcmc_width defaults to 1)
    config_kwargs = {
        "num_selected": num_selected,
        "num_steps": num_mcmc_steps,
        "verbose": False,
    }
    if mcmc_width is not None:
        config_kwargs["mcmc_width"] = mcmc_width
    config = MCMCOptimizerConfig(**config_kwargs)

    optimizer = MCMCOptimizer(
        constructs=[construct],
        generators=[proposal_gen],
        constraints=[constraint],
        config=config,
    )
    return optimizer, proposal_gen, constraint, segment


class TestMCMCOptimizer:
    def test_initialization_and_validation(self):
        """Tests successful initialization and validation of MCMCOptimizer."""
        optimizer, proposal_gen, constraint, segment = _setup_mcmc_components()

        assert optimizer.generators == [proposal_gen]
        assert optimizer.constraints == [constraint]
        assert optimizer.num_selected == 1
        assert optimizer.num_candidates == 1  # Defaults to num_selected

        # Test validation errors - unassigned generator
        test_segment = Segment(sequence="A" * 10, sequence_type="dna")
        unassigned_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )

        # Create a dummy scoring function with required attributes
        def dummy_scoring_func(seq, config=None):
            return 0.0
        dummy_scoring_func._constraint_batched = False
        dummy_scoring_func._constraint_multi_input = False
        dummy_scoring_func._constraint_config_class = EmptyConfig
        dummy_scoring_func._constraint_supported_sequence_types = ["dna"]

        dummy_constraint = Constraint(
            inputs=[test_segment],
            function=dummy_scoring_func,
            function_config=EmptyConfig(),
        )
        with pytest.raises(RuntimeError, match="has no segment assigned"):
            MCMCOptimizer(
                constructs=[Construct([test_segment])],
                generators=[unassigned_gen],
                constraints=[dummy_constraint],
                config=MCMCOptimizerConfig(num_selected=1, num_steps=1),
            )

    def test_config_validation(self):
        """Tests MCMCOptimizerConfig validation."""
        from pydantic import ValidationError

        # Valid configs
        config = MCMCOptimizerConfig(num_selected=5, mcmc_width=10, num_steps=1)
        assert config.num_selected == 5
        assert config.mcmc_width == 10

        # min_temperature >= max_temperature should fail
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_selected=1, num_steps=1, max_temperature=1.0, min_temperature=1.0)

        # Negative values should fail
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_selected=-1, num_steps=1)

    def test_score_energy(self):
        """Tests the score_energy method."""
        optimizer, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))

        # Test with a sequence within target GC range
        segment.candidate_sequences[0].sequence = "GCGCGAATTA"  # 50% GC
        optimizer.score_energy()
        assert len(optimizer.energy_scores) == optimizer.num_candidates
        assert optimizer.energy_scores[0] == 0.0

        # Test with a sequence below target range
        segment.candidate_sequences[0].sequence = "GCTTAATTAA"  # 20% GC
        optimizer.score_energy()
        expected_score = (40.0 - 20.0) / 40.0  # 0.5
        assert abs(optimizer.energy_scores[0] - expected_score) < 1e-9

    def test_score_energy_multiply(self):
        """Tests the score_energy method with operation='multiply'."""
        optimizer, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))
        segment.candidate_sequences[0].sequence = "GCTTAATTAA"  # 20% GC -> score 0.5

        # With one constraint, multiply and add should be the same
        optimizer.score_energy(operation="add")
        energy_add = optimizer.energy_scores[0]
        optimizer.score_energy(operation="multiply")
        energy_mul = optimizer.energy_scores[0]
        assert abs(energy_add - 0.5) < 1e-9
        assert abs(energy_mul - 0.5) < 1e-9

    def test_reject_inf_nan_energies(self):
        """Tests that inf and nan energy proposals are always rejected."""
        optimizer, _, _, segment = _setup_mcmc_components(
            seq_length=10,
            num_selected=1,
            mcmc_width=3,
            num_mcmc_steps=1,
        )

        # Set up initial state
        initial_seq = "GCGCGAATTA"  # 50% GC, energy = 0
        segment.selected_sequences[0].sequence = initial_seq
        for i in range(optimizer.num_candidates):
            segment.candidate_sequences[i] = copy.deepcopy(segment.selected_sequences[0])

        optimizer.score_energy()
        old_selected_sequences = optimizer._save_sequence_state()

        # Manually set ALL energy scores to inf and nan (so no valid proposals)
        optimizer.energy_scores[0] = float('inf')
        optimizer.energy_scores[1] = float('inf')
        optimizer.energy_scores[2] = float('nan')

        # Mutate candidate sequences so we can detect if they get rejected
        segment.candidate_sequences[0].sequence = "AAAAAAAAAA"
        segment.candidate_sequences[1].sequence = "TTTTTTTTTT"
        segment.candidate_sequences[2].sequence = "CCCCCCCCCC"

        # Run acceptance step
        optimizer._select_topk_with_mcmc_acceptance(step=1, old_selected_sequences=old_selected_sequences)

        # After rejection of all inf/nan proposals, selected_sequences should be restored
        # to the initial sequence (trajectory keeps old state when all proposals rejected)
        assert segment.selected_sequences[0].sequence == initial_seq, (
            f"selected_sequences[0] was not restored: got {segment.selected_sequences[0].sequence}, expected {initial_seq}"
        )

        # Energy for the trajectory should be the old energy (0.0)
        assert optimizer.energy_scores[0] == 0.0

    def test_sample_improves_energy(self):
        """Tests that sampling can improve the energy score over time."""
        optimizer, _, _, segment = _setup_mcmc_components(
            seq_length=50,
            gc_target_range=(80.0, 90.0),  # Encourage high GC
            num_mcmc_steps=100,
        )

        # Start with a bad sequence
        initial_seq = "A" * 50
        for seq in segment.selected_sequences:
            seq.sequence = initial_seq

        # Score initial state
        for i in range(optimizer.num_selected):
            optimizer.segments[0].candidate_sequences[i] = copy.deepcopy(
                optimizer.segments[0].selected_sequences[i]
            )
        optimizer.score_energy()
        initial_energy = optimizer.energy_scores[0]
        assert initial_energy > 0.99  # Should be max penalty

        # Sample and check for improvement
        optimizer.run()
        final_energy = min(optimizer.energy_scores)

        assert final_energy < initial_energy
        assert len(optimizer.history) > 1  # Check history is tracked

    def test_multiple_constraints(self):
        """Tests the MCMC Optimizer with multiple constraints and weights."""
        seq_len = 30
        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment = Segment(sequence="A" * seq_len, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        gc_con = Constraint(
            [segment], gc_content_constraint, GCContentConfig(min_gc=40.0, max_gc=60.0),
        )
        len_con = Constraint(
            [segment], sequence_length_constraint, SequenceLengthConfig(target_length=seq_len),
            weight=2.0,
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_con, len_con],
            config=MCMCOptimizerConfig(num_selected=1, num_steps=1, verbose=False),
        )

        assert optimizer.constraint_weights == [1.0, 2.0]

        segment.candidate_sequences[0].sequence = "A" * 20  # Violates length and GC
        expected_gc_score = (40 - 0) / 40  # = 1.0
        expected_len_score = (30 - 20) / 30 # = 0.333

        # E = 1.0 * 1.0 + 2.0 * 0.333...
        expected_energy = expected_gc_score * 1.0 + expected_len_score * 2.0
        optimizer.score_energy("add")
        assert abs(optimizer.energy_scores[0] - expected_energy) < 1e-9

        # Test multiply operation
        expected_energy_mul = (expected_gc_score * 1.0) * (expected_len_score * 2.0)
        optimizer.score_energy("multiply")
        assert abs(optimizer.energy_scores[0] - expected_energy_mul) < 1e-9

    def test_with_multiple_generators(self):
        """Tests MCMC with more than one proposal generator."""
        seq_len = 50
        mut_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment1 = Segment(sequence="A" * seq_len, sequence_type="dna")
        mut_gen.assign(segment1)

        inv_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=3)
        )
        segment2 = Segment(sequence="C" * seq_len, sequence_type="dna")
        inv_gen.assign(segment2)

        construct = Construct([segment1, segment2])
        # Use separate constraints for each segment (single-input constraints)
        constraint1 = Constraint(
            inputs=[segment1],
            function=sequence_length_constraint,
            function_config=SequenceLengthConfig(target_length=seq_len),
        )
        constraint2 = Constraint(
            inputs=[segment2],
            function=sequence_length_constraint,
            function_config=SequenceLengthConfig(target_length=seq_len),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[mut_gen, inv_gen],
            constraints=[constraint1, constraint2],
            config=MCMCOptimizerConfig(num_selected=1, num_steps=20, verbose=False),
        )

        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence

        optimizer.run()

        final_seq1 = segment1[0].sequence
        final_seq2 = segment2[0].sequence

        # At least one sequence should be modified
        assert initial_seq1 != final_seq1 or initial_seq2 != final_seq2

    def test_topk_initialization(self):
        """Tests initialization of top-k MCMC with num_selected > 1."""
        optimizer, _, _, _ = _setup_mcmc_components(
            num_selected=3, mcmc_width=10
        )

        assert optimizer.num_selected == 3
        # mcmc_width is the number of proposals per selected sequence
        assert optimizer.mcmc_width == 10
        # num_candidates is the total pool size (num_selected * mcmc_width)
        assert optimizer.num_candidates == 30

    def test_topk_maintains_k_sequences(self):
        """Tests that top-k MCMC maintains exactly k sequences."""
        num_selected = 3
        num_candidates = 4

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment = Segment(sequence="ATCGATCGATCGATCGATCG", sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                mcmc_width=num_candidates,
                num_steps=20,
                max_temperature=1.0,
                min_temperature=0.01,
                verbose=False,
            ),
        )

        optimizer.run()

        # Should maintain num_selected sequences
        # energy_scores is truncated to num_selected after each selection step
        assert len(optimizer.energy_scores) == num_selected
        assert len(segment.selected_sequences) == num_selected

    def test_history_tracking(self):
        """Tests that history is properly tracked during MCMC."""
        num_selected = 2
        num_steps = 30
        track_step_size = 10
        seq_length = 10

        optimizer, _, _, _ = _setup_mcmc_components(
            seq_length=seq_length,
            num_selected=num_selected,
            num_mcmc_steps=num_steps,
        )
        optimizer.track_step_size = track_step_size

        optimizer.run()

        # History should have snapshots: 0 (initial), 10, 20, 30 (final)
        expected_snapshots = 4
        assert len(optimizer.history) == expected_snapshots

        # Each history entry should have proper structure
        for entry in optimizer.history:
            assert "time_step" in entry
            assert "energy_scores" in entry
            assert "constructs" in entry
            assert len(entry["energy_scores"]) == num_selected

    def test_history_timesteps_validation(self):
        """Tests that time_step values in history entries are correctly tracked."""
        num_selected = 2
        num_steps = 35
        track_step_size = 10

        optimizer, _, _, _ = _setup_mcmc_components(
            num_selected=num_selected, num_mcmc_steps=num_steps
        )
        optimizer.track_step_size = track_step_size

        optimizer.run()

        # Expected timesteps: 0 (initial), 10, 20, 30, 35 (final)
        expected_timesteps = [0, 10, 20, 30, 35]
        actual_timesteps = [entry["time_step"] for entry in optimizer.history]

        assert actual_timesteps == expected_timesteps

        # Verify timesteps are monotonically increasing
        for i in range(1, len(actual_timesteps)):
            assert actual_timesteps[i] > actual_timesteps[i - 1]

    def test_temperature_scheduling(self):
        """Tests that simulated annealing temperature schedule is correct."""
        num_steps = 100
        max_temperature = 10.0
        min_temperature = 0.01

        optimizer, _, _, _ = _setup_mcmc_components(num_mcmc_steps=num_steps)
        optimizer.max_temperature = max_temperature
        optimizer.min_temperature = min_temperature
        optimizer.num_steps = num_steps

        # Test temperature at key steps
        step_1_temp = optimizer._compute_temperature(1)
        step_50_temp = optimizer._compute_temperature(50)
        step_100_temp = optimizer._compute_temperature(100)

        # Step 1 should be exactly T_max
        assert abs(step_1_temp - max_temperature) < 1e-10

        # Final step should be exactly T_min
        assert abs(step_100_temp - min_temperature) < 1e-10

        # Middle step should be between T_max and T_min
        assert min_temperature < step_50_temp < max_temperature

        # Temperatures should decrease monotonically
        temperatures = [
            optimizer._compute_temperature(step) for step in range(1, num_steps + 1)
        ]
        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i - 1]

    def test_temperature_scheduling_edge_cases(self):
        """Tests temperature scheduling edge cases."""
        max_temperature = 5.0
        min_temperature = 0.001

        # Test num_steps=1 (should return T_max)
        optimizer, _, _, _ = _setup_mcmc_components(num_mcmc_steps=1)
        optimizer.max_temperature = max_temperature
        optimizer.min_temperature = min_temperature
        optimizer.num_steps = 1

        step_1_temp = optimizer._compute_temperature(1)
        assert abs(step_1_temp - max_temperature) < 1e-10

        # Test num_steps=2 (should go from T_max to T_min)
        optimizer.num_steps = 2
        step_1_temp = optimizer._compute_temperature(1)
        step_2_temp = optimizer._compute_temperature(2)

        assert abs(step_1_temp - max_temperature) < 1e-10
        assert abs(step_2_temp - min_temperature) < 1e-10

    def test_mcmc_acceptance_probability(self):
        """Tests Metropolis-Hastings acceptance probability computation."""
        optimizer, _, _, _ = _setup_mcmc_components()

        # Better proposal (lower energy) should always be accepted
        alpha = optimizer._compute_mcmc_alpha(1.0, 0.5, 1)
        assert alpha == 1.0

        # Equal energy should be accepted
        alpha = optimizer._compute_mcmc_alpha(0.5, 0.5, 1)
        assert alpha == 1.0

        # Worse proposal should have probability < 1
        alpha = optimizer._compute_mcmc_alpha(0.5, 1.0, 1)
        assert 0.0 < alpha < 1.0

    def test_overflow_protection(self):
        """Test that MAX_EXP_ARG prevents overflow in acceptance computation."""
        optimizer, _, _, _ = _setup_mcmc_components()

        # Very large energy improvement should be clamped to 1.0
        alpha = optimizer._compute_mcmc_alpha(1000.0, 0.0, 1)
        assert alpha == 1.0

        # Very large energy increase should give very small probability
        alpha = optimizer._compute_mcmc_alpha(0.0, 1000.0, 1)
        assert 0.0 <= alpha < 1e-10

    def test_convergence_to_optimal(self):
        """Tests that MCMC converges toward optimal solution with enough steps."""
        num_selected = 3
        seq_length = 15

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Constraint with clear optimum (all G's)
        def perfect_g_energy(seq, config=None):
            g_count = seq.sequence.count("G")
            return seq_length - g_count

        # Add required attributes for the scoring function
        perfect_g_energy._constraint_batched = False
        perfect_g_energy._constraint_multi_input = False
        perfect_g_energy._constraint_config_class = EmptyConfig
        perfect_g_energy._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment],
            function=perfect_g_energy,
            function_config=EmptyConfig(),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_steps=200,
                max_temperature=2.0,
                min_temperature=0.01,
                verbose=False,
            ),
        )

        # Score initial state
        for i in range(optimizer.num_selected):
            optimizer.segments[0].candidate_sequences[i] = copy.deepcopy(
                optimizer.segments[0].selected_sequences[i]
            )
        optimizer.score_energy()
        initial_best_energy = min(optimizer.energy_scores[:num_selected])

        optimizer.run()
        final_best_energy = min(optimizer.energy_scores)

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
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=1,
                num_steps=num_steps,
                track_step_size=track_step_size,
                verbose=True,
            ),
            custom_logging=custom_log,
        )

        optimizer.run()

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
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment1 = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen1.assign(segment1)
        construct1 = Construct([segment1])
        constraint1 = Constraint(
            inputs=[segment1],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        optimizer1 = MCMCOptimizer(
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
            optimizer1.run()
        finally:
            sys.stdout = sys.__stdout__

        output1 = captured_output1.getvalue()
        assert "energy:" in output1
        assert "best:" not in output1

        # Test num_selected>1 (should show "best:", "mean:", etc.)
        proposal_gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment2 = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen2.assign(segment2)
        construct2 = Construct([segment2])
        constraint2 = Constraint(
            inputs=[segment2],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40.0, max_gc=60.0),
        )

        optimizer2 = MCMCOptimizer(
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
            optimizer2.run()
        finally:
            sys.stdout = sys.__stdout__

        output2 = captured_output2.getvalue()
        assert "best:" in output2
        assert "mean:" in output2

    def test_deepcopy_independence(self):
        """Tests that deepcopy ensures independent Sequence objects."""
        num_selected = 2
        seq_length = 20

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)

        # Add metadata to test deep copy
        for i, seq in enumerate(segment.selected_sequences):
            seq._metadata["seq_id"] = f"seq_{i}"
            seq._metadata["nested"] = {"count": i, "tags": [f"tag_{i}"]}

        # Create a dummy scoring function with required attributes
        def dummy_scoring_func(seq, config=None):
            return 0.0
        dummy_scoring_func._constraint_batched = False
        dummy_scoring_func._constraint_multi_input = False
        dummy_scoring_func._constraint_config_class = EmptyConfig
        dummy_scoring_func._constraint_supported_sequence_types = ["dna"]

        construct = Construct([segment])
        constraint = Constraint(
            inputs=[segment],
            function=dummy_scoring_func,
            function_config=EmptyConfig(),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_steps=1,
                verbose=False,
            ),
        )

        optimizer.run()

        # Verify sequences are independent objects
        for i in range(len(segment.selected_sequences)):
            for j in range(i + 1, len(segment.selected_sequences)):
                assert segment.selected_sequences[i] is not segment.selected_sequences[j]

    def test_comprehensive_integration(self):
        """Comprehensive integration test with num_selected>1."""
        seq_length = 30
        num_selected = 5
        num_steps = 50

        gen1 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        gen2 = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=3)
        )

        segment1 = Segment(sequence="A" * seq_length, sequence_type="dna")
        segment2 = Segment(sequence="T" * seq_length, sequence_type="dna")

        gen1.assign(segment1)
        gen2.assign(segment2)

        construct = Construct([segment1, segment2])

        # Separate constraints for each segment (single-input constraints)
        gc_constraint1 = Constraint(
            inputs=[segment1],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=45.0, max_gc=55.0),
        )
        gc_constraint2 = Constraint(
            inputs=[segment2],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=45.0, max_gc=55.0),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[gc_constraint1, gc_constraint2],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                num_steps=num_steps,
                verbose=False,
            ),
        )

        optimizer.run()

        # energy_scores is truncated to num_selected after each selection step
        assert len(optimizer.energy_scores) == num_selected
        assert len(segment1.selected_sequences) == num_selected
        assert len(segment2.selected_sequences) == num_selected

    def test_run_restarts_from_initial_state(self):
        """Tests that calling run() twice restarts from initial state."""
        optimizer, _, _, segment = _setup_mcmc_components(
            seq_length=20, num_mcmc_steps=5
        )

        # Capture original state before any run
        original_seq = segment.selected_sequences[0].sequence
        assert original_seq == "A" * 20  # Initial sequence

        # First run
        optimizer.run()

        # Verify state was captured with correct content
        assert optimizer._initial_state is not None
        assert len(optimizer._initial_state['segments']) == 1
        
        # Verify captured state contains original sequence (using index 0)
        captured_selected = optimizer._initial_state['segments'][0]['selected']
        assert len(captured_selected) == 1
        assert captured_selected[0].sequence == original_seq
        
        # Verify energy scores were captured (initial state captures full num_candidates before first run)
        assert 'energy_scores' in optimizer._initial_state
        assert len(optimizer._initial_state['energy_scores']) == optimizer.num_candidates

        # Manually modify the sequence to verify restore works
        segment.selected_sequences[0].sequence = "G" * 20
        segment.candidate_sequences[0].sequence = "G" * 20

        # Second run should restore from initial state (original "AAAA...")
        optimizer.run()
        second_run_final_seq = segment.selected_sequences[0].sequence

        # Verify sequences were restored (not all G's, optimization ran from restored state)
        # The restored state was "A" * 20, then mutations were applied
        assert second_run_final_seq != "G" * 20
        
        # Both runs should have started from original state
        # History should be fresh (cleared on restart)
        assert len(optimizer.history) > 0

    def test_independent_trajectories_no_crossover(self):
        """Tests that each batch index is an independent trajectory with no crossover.
        
        Each trajectory should only select from its own proposal pool, not mix with
        proposals from other trajectories.
        """
        num_selected = 3
        mcmc_width = 4
        seq_length = 10

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Custom constraint that returns energy = number of G's (so more G's = higher energy)
        def count_g_energy(seq, config=None):
            return seq.sequence.count("G")
        count_g_energy._constraint_batched = False
        count_g_energy._constraint_multi_input = False
        count_g_energy._constraint_config_class = EmptyConfig
        count_g_energy._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment],
            function=count_g_energy,
            function_config=EmptyConfig(),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                mcmc_width=mcmc_width,
                num_steps=1,
                max_temperature=0.002,  # Very low temp = greedy
                min_temperature=0.001,
                verbose=False,
            ),
        )

        # Set up distinct initial sequences for each trajectory
        # Trajectory 0: starts with "AAAAAAAAAA" (energy=0, best)
        # Trajectory 1: starts with "GGGGGGGGGG" (energy=10, worst)
        # Trajectory 2: starts with "GGGGGAAAAA" (energy=5, middle)
        segment.selected_sequences[0].sequence = "A" * seq_length  # energy=0
        segment.selected_sequences[1].sequence = "G" * seq_length  # energy=10
        segment.selected_sequences[2].sequence = "G" * 5 + "A" * 5  # energy=5

        # Set the energy_scores for selected sequences (indices 0, 1, 2)
        # This mimics the state after previous iteration where only first num_selected
        # entries are the selected energies
        optimizer.energy_scores[0] = 0   # Trajectory 0
        optimizer.energy_scores[1] = 10  # Trajectory 1
        optimizer.energy_scores[2] = 5   # Trajectory 2

        # Save state BEFORE populating candidates (this is how the real loop works)
        old_selected_sequences = optimizer._save_sequence_state()

        # Populate candidate_sequences by replicating each selected_sequence
        optimizer._populate_candidate_sequences()

        # Score all candidates
        optimizer.score_energy()

        # Verify the layout: each trajectory's proposals are in its own range
        # Trajectory 0: indices [0, 4), Trajectory 1: indices [4, 8), Trajectory 2: indices [8, 12)
        for traj_idx in range(num_selected):
            start_idx = traj_idx * mcmc_width
            end_idx = (traj_idx + 1) * mcmc_width
            for cand_idx in range(start_idx, end_idx):
                assert segment.candidate_sequences[cand_idx].sequence == segment.selected_sequences[traj_idx].sequence

        # Now manually set up a scenario where crossover would be visible:
        # Give trajectory 1's candidates very good energies (better than trajectory 0's old energy)
        # If there's crossover, trajectory 0 would steal from trajectory 1's pool
        
        # Make trajectory 1's first candidate have energy=0 (best possible)
        segment.candidate_sequences[4].sequence = "A" * seq_length  # energy=0
        optimizer.energy_scores[4] = 0.0
        
        # Keep trajectory 0's candidates at their original (energy=0)
        # Keep trajectory 2's candidates at their original (energy=5)

        # Run acceptance step
        optimizer._select_topk_with_mcmc_acceptance(step=1, old_selected_sequences=old_selected_sequences)

        # Verify NO CROSSOVER with the new "best first, then MH" logic:
        # - Trajectory 1 finds its best proposal (energy=0 at index 4)
        # - MH acceptance is applied: old_energy=10, new_energy=0, so alpha=1.0 (always accept improvement)
        # - Trajectory 1 should now have "AAAAAAAAAA"
        # - Trajectory 0 should NOT have stolen from trajectory 1's pool
        
        # The key test: trajectory 1's selected sequence should now be "AAAAAAAAAA"
        assert segment.selected_sequences[1].sequence == "A" * seq_length, (
            f"Trajectory 1 should have selected from its own pool. "
            f"Got: {segment.selected_sequences[1].sequence}"
        )

        # Trajectory 0 should still be "AAAAAAAAAA" (no change since its proposals
        # were deepcopies of the same sequence)
        assert segment.selected_sequences[0].sequence == "A" * seq_length

        # Verify energies are updated correctly per trajectory
        assert optimizer.energy_scores[0] == 0.0  # Trajectory 0's energy
        assert optimizer.energy_scores[1] == 0.0  # Trajectory 1's energy (improved)

    def test_trajectory_isolation_with_different_starting_points(self):
        """Tests that trajectories starting from different sequences remain isolated."""
        num_selected = 2
        mcmc_width = 5
        seq_length = 20
        num_steps = 10

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=2)
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=45.0, max_gc=55.0),
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                mcmc_width=mcmc_width,
                num_steps=num_steps,
                track_step_size=1,
                max_temperature=1.0,
                min_temperature=0.01,
                verbose=False,
            ),
        )

        # Give each trajectory a very different starting sequence
        segment.selected_sequences[0].sequence = "A" * seq_length  # 0% GC
        segment.selected_sequences[1].sequence = "G" * seq_length  # 100% GC

        optimizer.run()

        # Each trajectory's energy should be tracked independently
        # Check that history shows each trajectory improving independently
        for entry in optimizer.history:
            assert len(entry["energy_scores"]) == num_selected

        # Final sequences should be different (each evolved from different start)
        # With high probability, they won't converge to identical sequences
        # in just 10 steps with such different starting points
        final_seq_0 = segment.selected_sequences[0].sequence
        final_seq_1 = segment.selected_sequences[1].sequence

        # Both should have improved from initial (moved toward 50% GC)
        gc_0 = (final_seq_0.count("G") + final_seq_0.count("C")) / seq_length * 100
        gc_1 = (final_seq_1.count("G") + final_seq_1.count("C")) / seq_length * 100

        # Trajectory 0 started at 0% GC, should have increased
        assert gc_0 > 0, "Trajectory 0 should have evolved from 0% GC"
        # Trajectory 1 started at 100% GC, should have decreased
        assert gc_1 < 100, "Trajectory 1 should have evolved from 100% GC"

    def test_best_first_then_mh_selection(self):
        """Tests that selection picks best proposal first, then applies single MH decision.
        
        The new selection logic:
        1. Find the best proposal by energy (lowest)
        2. Apply MH acceptance to that single best proposal
        3. If rejected, keep old state
        """
        num_selected = 1
        mcmc_width = 3
        seq_length = 10

        proposal_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Custom constraint: energy = count of non-A characters
        def count_non_a_energy(seq, config=None):
            return seq_length - seq.sequence.count("A")
        count_non_a_energy._constraint_batched = False
        count_non_a_energy._constraint_multi_input = False
        count_non_a_energy._constraint_config_class = EmptyConfig
        count_non_a_energy._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment],
            function=count_non_a_energy,
            function_config=EmptyConfig(),
        )

        # Scenario 1: Low temperature - best proposal improves energy -> should be accepted
        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                mcmc_width=mcmc_width,
                num_steps=1,
                max_temperature=0.002,  # Very low temp = greedy
                min_temperature=0.001,
                verbose=False,
            ),
        )

        # Set initial sequence with energy=5
        segment.selected_sequences[0].sequence = "AAAAAGGGGG"  # 5 non-A = energy 5
        optimizer.energy_scores[0] = 5.0

        old_selected_sequences = optimizer._save_sequence_state()
        optimizer._populate_candidate_sequences()

        # Set up proposals with different energies: [0.8, 0.3, 0.9]
        # Best is at index 1 with energy 0.3
        segment.candidate_sequences[0].sequence = "AAAAAAAGGG"  # 3 non-A
        segment.candidate_sequences[1].sequence = "AAAAAAAAAT"  # 1 non-A (best)
        segment.candidate_sequences[2].sequence = "AAAAAAGGGG"  # 4 non-A
        optimizer.energy_scores = [3.0, 1.0, 4.0]

        optimizer._select_topk_with_mcmc_acceptance(step=1, old_selected_sequences=old_selected_sequences)

        # With low temperature, the best proposal (energy=1.0) should be accepted
        # because it improves from energy=5.0
        assert segment.selected_sequences[0].sequence == "AAAAAAAAAT", (
            f"Expected best proposal to be accepted. Got: {segment.selected_sequences[0].sequence}"
        )
        assert optimizer.energy_scores[0] == 1.0

        # Scenario 2: Test that only the best proposal is considered for MH
        # Set up a case where the best proposal worsens energy
        optimizer2 = MCMCOptimizer(
            constructs=[Construct([segment])],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_selected=num_selected,
                mcmc_width=mcmc_width,
                num_steps=1,
                max_temperature=0.002,  # Very low temp
                min_temperature=0.001,
                verbose=False,
            ),
        )

        # Start with a very good sequence (energy=0)
        segment.selected_sequences[0].sequence = "A" * seq_length  # 0 non-A = energy 0
        optimizer2.energy_scores[0] = 0.0

        old_selected_sequences2 = optimizer2._save_sequence_state()
        optimizer2._populate_candidate_sequences()

        # All proposals are worse than current (energy 0)
        segment.candidate_sequences[0].sequence = "AAAAAAAAAT"  # 1 non-A
        segment.candidate_sequences[1].sequence = "AAAAAAGGGG"  # 4 non-A
        segment.candidate_sequences[2].sequence = "AAAAAAAAAC"  # 1 non-A
        optimizer2.energy_scores = [1.0, 4.0, 1.0]

        optimizer2._select_topk_with_mcmc_acceptance(step=1, old_selected_sequences=old_selected_sequences2)

        # At very low temperature, worse proposals should be rejected
        # The old state should be kept
        assert segment.selected_sequences[0].sequence == "A" * seq_length, (
            f"Expected rejection - old state should be kept. Got: {segment.selected_sequences[0].sequence}"
        )
        assert optimizer2.energy_scores[0] == 0.0
