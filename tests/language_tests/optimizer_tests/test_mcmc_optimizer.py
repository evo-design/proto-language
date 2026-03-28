"""tests/language_tests/optimizer_tests/test_mcmc_optimizer.py"""
from __future__ import annotations

import copy
import random
from typing import Tuple

import numpy as np
import pytest
from proto_tools.tools.masked_models.masking import MaskingStrategy
from pydantic import BaseModel

from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.constraint.sequence_composition.gc_content_constraint import (
    GCContentConfig,
)
from proto_language.language.constraint.sequence_composition.sequence_length_constraint import (
    SequenceLengthConfig,
)
from proto_language.language.core import Constraint, Construct, Segment
from proto_language.language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig


# Empty config for test constraints
class EmptyConfig(BaseModel):
    pass


def _setup_mcmc_components(
    seq_length: int = 10,
    num_results: int = 1,
    proposals_per_result: int = None,
    gc_target_range: Tuple[float, float] = (40.0, 60.0),
    num_mcmc_steps: int = 10,
):
    """Helper function to set up a basic MCMC Optimizer for testing."""
    # 1. Create the proposal generator and the segment it will modify
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    proposal_gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
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

    # 3. Create the MCMC Optimizer config (proposals_per_result defaults to 1)
    config_kwargs = {
        "num_results": num_results,
        "num_steps": num_mcmc_steps,
        "verbose": False,
    }
    if proposals_per_result is not None:
        config_kwargs["proposals_per_result"] = proposals_per_result
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
        assert optimizer.num_results == 1
        assert optimizer.num_proposals == 1  # Defaults to num_trajectories

        # Test validation errors - unassigned generator
        test_segment = Segment(sequence="A" * 10, sequence_type="dna")
        unassigned_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )

        # Create a dummy scoring function with required attributes
        def dummy_scoring_func(input_sequences, config=None):
            return [0.0 for _ in input_sequences]
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
                config=MCMCOptimizerConfig(num_results=1, num_steps=1),
            )

    def test_config_validation(self):
        """Tests MCMCOptimizerConfig validation."""
        from pydantic import ValidationError

        # Valid configs
        config = MCMCOptimizerConfig(num_results=5, proposals_per_result=10, num_steps=1)
        assert config.num_results == 5
        assert config.proposals_per_result == 10

        # min_temperature >= max_temperature should fail
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_results=1, num_steps=1, max_temperature=1.0, min_temperature=1.0)

        # Negative values should fail
        with pytest.raises(ValidationError):
            MCMCOptimizerConfig(num_results=-1, num_steps=1)

    def test_score_energy(self):
        """Tests the score_energy method."""
        optimizer, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))

        # Test with a sequence within target GC range
        segment.proposal_sequences[0].sequence = "GCGCGAATTA"  # 50% GC
        optimizer.score_energy()
        assert len(optimizer.energy_scores) == optimizer.num_proposals
        assert optimizer.energy_scores[0] == 0.0

        # Test with a sequence below target range
        segment.proposal_sequences[0].sequence = "GCTTAATTAA"  # 20% GC
        optimizer.score_energy()
        expected_score = (40.0 - 20.0) / 40.0  # 0.5
        assert abs(optimizer.energy_scores[0] - expected_score) < 1e-9

    def test_score_energy_multiply(self):
        """Tests the score_energy method with operation='multiply'."""
        optimizer, _, _, segment = _setup_mcmc_components(gc_target_range=(40.0, 60.0))
        segment.proposal_sequences[0].sequence = "GCTTAATTAA"  # 20% GC -> score 0.5

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
            num_results=1,
            proposals_per_result=3,
            num_mcmc_steps=1,
        )

        # Set up initial state
        initial_seq = "GCGCGAATTA"  # 50% GC, energy = 0
        segment.result_sequences[0].sequence = initial_seq
        for i in range(optimizer.num_proposals):
            segment.proposal_sequences[i] = copy.deepcopy(segment.result_sequences[0])

        optimizer.score_energy()
        old_result_sequences = optimizer._save_sequence_state()

        # Manually set ALL energy scores to inf and nan (so no valid proposals)
        optimizer.energy_scores[0] = float('inf')
        optimizer.energy_scores[1] = float('inf')
        optimizer.energy_scores[2] = float('nan')

        # Mutate proposal sequences so we can detect if they get rejected
        segment.proposal_sequences[0].sequence = "AAAAAAAAAA"
        segment.proposal_sequences[1].sequence = "TTTTTTTTTT"
        segment.proposal_sequences[2].sequence = "CCCCCCCCCC"

        # All proposals are "rejected" since energies are inf/nan
        optimizer._proposal_outcomes = ["inf/nan energy"] * optimizer.num_proposals
        optimizer._select_topk_with_mcmc_acceptance(step=1, old_result_sequences=old_result_sequences)

        # After rejection of all inf/nan proposals, result_sequences should be restored
        # to the initial sequence (trajectory keeps old state when all proposals rejected)
        assert segment.result_sequences[0].sequence == initial_seq, (
            f"result_sequences[0] was not restored: got {segment.result_sequences[0].sequence}, expected {initial_seq}"
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
        for seq in segment.result_sequences:
            seq.sequence = initial_seq

        # Score initial state
        for i in range(optimizer.num_results):
            optimizer.segments[0].proposal_sequences[i] = copy.deepcopy(
                optimizer.segments[0].result_sequences[i]
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
        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
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
            config=MCMCOptimizerConfig(num_results=1, num_steps=1, verbose=False),
        )

        assert optimizer.constraint_weights == [1.0, 2.0]

        segment.proposal_sequences[0].sequence = "A" * 20  # Violates length and GC
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
        mut_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        segment1 = Segment(sequence="A" * seq_len, sequence_type="dna")
        mut_gen.assign(segment1)

        inv_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3))
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
            config=MCMCOptimizerConfig(num_results=1, num_steps=20, verbose=False),
        )

        initial_seq1 = segment1[0].sequence
        initial_seq2 = segment2[0].sequence

        optimizer.run()

        final_seq1 = segment1[0].sequence
        final_seq2 = segment2[0].sequence

        # At least one sequence should be modified
        assert initial_seq1 != final_seq1 or initial_seq2 != final_seq2

    def test_topk_initialization(self):
        """Tests initialization of top-k MCMC with num_trajectories > 1."""
        optimizer, _, _, _ = _setup_mcmc_components(
            num_results=3, proposals_per_result=10
        )

        assert optimizer.num_results == 3
        # _proposals_per_result is the number of proposals per result sequence
        assert optimizer._proposals_per_result == 10
        # num_proposals is the total pool size (num_results * _proposals_per_result)
        assert optimizer.num_proposals == 30

    def test_topk_maintains_k_sequences(self):
        """Tests that top-k MCMC maintains exactly k sequences."""
        num_trajectories = 3
        num_proposals = 4

        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
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
                num_results=num_trajectories,
                proposals_per_result=num_proposals,
                num_steps=20,
                max_temperature=1.0,
                min_temperature=0.01,
                verbose=False,
            ),
        )

        optimizer.run()

        # Should maintain num_trajectories sequences
        # energy_scores is truncated to num_trajectories after each selection step
        assert len(optimizer.energy_scores) == num_trajectories
        assert len(segment.result_sequences) == num_trajectories

    def test_history_tracking(self):
        """Tests that history is properly tracked during MCMC — every step saved."""
        num_trajectories = 2
        num_steps = 10
        seq_length = 10

        optimizer, _, _, _ = _setup_mcmc_components(
            seq_length=seq_length,
            num_results=num_trajectories,
            num_mcmc_steps=num_steps,
        )

        optimizer.run()

        # History should have snapshots: 0 (initial) + 1..num_steps = num_steps + 1
        expected_snapshots = num_steps + 1
        assert len(optimizer.history) == expected_snapshots

        # Each history entry should have proper structure
        for entry in optimizer.history:
            assert "time_step" in entry
            assert "results" in entry
            assert len(entry["results"]) == num_trajectories

    def test_history_timesteps_validation(self):
        """Tests that time_step values in history entries are correctly tracked."""
        num_trajectories = 2
        num_steps = 5

        optimizer, _, _, _ = _setup_mcmc_components(
            num_results=num_trajectories, num_mcmc_steps=num_steps
        )

        optimizer.run()

        # Every step is saved: 0 (initial), 1, 2, 3, 4, 5
        expected_timesteps = list(range(num_steps + 1))
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
        num_trajectories = 3
        seq_length = 15

        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Constraint with clear optimum (all G's)
        def perfect_g_energy(input_sequences, config=None):
            return [(seq_length - seq.sequence.count("G")) / seq_length for (seq,) in input_sequences]

        # Add required attributes for the scoring function
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
                num_results=num_trajectories,
                num_steps=200,
                max_temperature=2.0,
                min_temperature=0.01,
                verbose=False,
            ),
        )

        # Score initial state
        for i in range(optimizer.num_results):
            optimizer.segments[0].proposal_sequences[i] = copy.deepcopy(
                optimizer.segments[0].result_sequences[i]
            )
        optimizer.score_energy()
        initial_best_energy = min(optimizer.energy_scores[:num_trajectories])

        optimizer.run()
        final_best_energy = min(optimizer.energy_scores)

        # Should get significant improvement
        assert final_best_energy < initial_best_energy * 0.5

    def test_custom_logging_callback(self):
        """Test that custom_logging is called at every step."""
        seq_length = 15
        num_steps = 5

        log_calls = []

        def custom_log(step, segments):
            log_calls.append({"step": step})

        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
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
                num_results=1,
                num_steps=num_steps,
                verbose=True,
            ),
            custom_logging=custom_log,
        )

        optimizer.run()

        # Every step should trigger custom logging
        expected_steps = list(range(1, num_steps + 1))
        actual_steps = [call["step"] for call in log_calls]
        assert actual_steps == expected_steps

    def test_verbose_output_formats(self, caplog):
        """Test logging output for num_trajectories=1 vs >1."""
        import logging

        seq_length = 15

        # Test num_trajectories=1 (should show "energy:")
        proposal_gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
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
                num_results=1, num_steps=3, verbose=True
            ),
        )

        with caplog.at_level(logging.DEBUG):
            optimizer1.run()

        output1 = caplog.text
        assert "energy:" in output1
        assert "best:" not in output1

        caplog.clear()

        # Test num_trajectories>1 (should show "best:", "mean:", etc.)
        proposal_gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
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
                num_results=3, num_steps=3, verbose=True
            ),
        )

        with caplog.at_level(logging.DEBUG):
            optimizer2.run()

        output2 = caplog.text
        assert "best:" in output2
        assert "mean:" in output2

    def test_deepcopy_independence(self):
        """Tests that deepcopy ensures independent Sequence objects."""
        num_trajectories = 2
        seq_length = 20

        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)

        # Add metadata to test deep copy
        for i, seq in enumerate(segment.result_sequences):
            seq._metadata["seq_id"] = f"seq_{i}"
            seq._metadata["nested"] = {"count": i, "tags": [f"tag_{i}"]}

        # Create a dummy scoring function with required attributes
        def dummy_scoring_func(input_sequences, config=None):
            return [0.0 for _ in input_sequences]
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
                num_results=num_trajectories,
                num_steps=1,
                verbose=False,
            ),
        )

        optimizer.run()

        # Verify sequences are independent objects
        for i in range(len(segment.result_sequences)):
            for j in range(i + 1, len(segment.result_sequences)):
                assert segment.result_sequences[i] is not segment.result_sequences[j]

    def test_comprehensive_integration(self):
        """Comprehensive integration test with num_trajectories>1."""
        seq_length = 30
        num_trajectories = 5
        num_steps = 50

        gen1 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        gen2 = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3))
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
                num_results=num_trajectories,
                num_steps=num_steps,
                verbose=False,
            ),
        )

        optimizer.run()

        # energy_scores is truncated to num_trajectories after each selection step
        assert len(optimizer.energy_scores) == num_trajectories
        assert len(segment1.result_sequences) == num_trajectories
        assert len(segment2.result_sequences) == num_trajectories

    def test_run_restarts_from_initial_state(self):
        """Tests that calling run() twice restarts from initial state."""
        optimizer, _, _, segment = _setup_mcmc_components(
            seq_length=20, num_mcmc_steps=5
        )

        # Capture original state before any run
        original_seq = segment.result_sequences[0].sequence
        assert original_seq == "A" * 20  # Initial sequence

        # First run
        optimizer.run()

        # Verify state was captured with correct content
        assert optimizer._initial_state is not None
        assert len(optimizer._initial_state['segments']) == 1

        # Verify captured state contains original sequence
        captured_result = optimizer._initial_state['segments'][0]['result']
        assert len(captured_result) == 1
        assert captured_result[0]['sequence'] == original_seq

        # Verify energy scores were captured (initial state captures full num_proposals before first run)
        assert 'energy_scores' in optimizer._initial_state
        assert len(optimizer._initial_state['energy_scores']) == optimizer.num_proposals

        # Manually modify the sequence to verify restore works
        segment.result_sequences[0].sequence = "G" * 20
        segment.proposal_sequences[0].sequence = "G" * 20

        # Second run should restore from initial state (original "AAAA...")
        optimizer.run()
        second_run_final_seq = segment.result_sequences[0].sequence

        # Verify sequences were restored (not all G's, optimization ran from restored state)
        # The restored state was "A" * 20, then mutations were applied
        assert second_run_final_seq != "G" * 20

        # Both runs should have started from original state
        # History should be fresh (cleared on restart)
        assert len(optimizer.history) > 0

    def test_independent_trajectories_no_crossover(self):
        """Tests that each result index is an independent trajectory with no crossover.

        Each trajectory should only select from its own proposal pool, not mix with
        proposals from other trajectories.
        """
        num_trajectories = 3
        proposals_per_result = 4
        seq_length = 10

        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Custom constraint that returns energy = number of G's (so more G's = higher energy)
        def count_g_energy(input_sequences, config=None):
            return [seq.sequence.count("G") / seq_length for (seq,) in input_sequences]
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
                num_results=num_trajectories,
                proposals_per_result=proposals_per_result,
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
        segment.result_sequences[0].sequence = "A" * seq_length  # energy=0
        segment.result_sequences[1].sequence = "G" * seq_length  # energy=10
        segment.result_sequences[2].sequence = "G" * 5 + "A" * 5  # energy=5

        # Set the energy_scores for result sequences (indices 0, 1, 2)
        # This mimics the state after previous iteration where only first num_trajectories
        # entries are the result energies
        optimizer.energy_scores[0] = 0   # Trajectory 0
        optimizer.energy_scores[1] = 10  # Trajectory 1
        optimizer.energy_scores[2] = 5   # Trajectory 2

        # Save state BEFORE populating proposals (this is how the real loop works)
        old_result_sequences = optimizer._save_sequence_state()

        # Populate proposal_sequences by replicating each result_sequence
        optimizer._populate_proposal_sequences()

        # Score all proposals
        optimizer.score_energy()

        # Verify the layout: each trajectory's proposals are in its own range
        # Trajectory 0: indices [0, 4), Trajectory 1: indices [4, 8), Trajectory 2: indices [8, 12)
        for traj_idx in range(num_trajectories):
            start_idx = traj_idx * proposals_per_result
            end_idx = (traj_idx + 1) * proposals_per_result
            for prop_idx in range(start_idx, end_idx):
                assert segment.proposal_sequences[prop_idx].sequence == segment.result_sequences[traj_idx].sequence

        # Now manually set up a scenario where crossover would be visible:
        # Give trajectory 1's proposals very good energies (better than trajectory 0's old energy)
        # If there's crossover, trajectory 0 would steal from trajectory 1's pool

        # Make trajectory 1's first proposal have energy=0 (best possible)
        segment.proposal_sequences[4].sequence = "A" * seq_length  # energy=0
        optimizer.energy_scores[4] = 0.0

        # Keep trajectory 0's proposals at their original (energy=0)
        # Keep trajectory 2's proposals at their original (energy=5)

        # Run acceptance step
        optimizer._proposal_outcomes = ["accepted"] * optimizer.num_proposals
        optimizer._select_topk_with_mcmc_acceptance(step=1, old_result_sequences=old_result_sequences)

        # Verify NO CROSSOVER with the new "best first, then MH" logic:
        # - Trajectory 1 finds its best proposal (energy=0 at index 4)
        # - MH acceptance is applied: old_energy=10, new_energy=0, so alpha=1.0 (always accept improvement)
        # - Trajectory 1 should now have "AAAAAAAAAA"
        # - Trajectory 0 should NOT have stolen from trajectory 1's pool

        # The key test: trajectory 1's result sequence should now be "AAAAAAAAAA"
        assert segment.result_sequences[1].sequence == "A" * seq_length, (
            f"Trajectory 1 should have accepted from its own pool. "
            f"Got: {segment.result_sequences[1].sequence}"
        )

        # Trajectory 0 should still be "AAAAAAAAAA" (no change since its proposals
        # were deepcopies of the same sequence)
        assert segment.result_sequences[0].sequence == "A" * seq_length

        # Verify energies are updated correctly per trajectory
        assert optimizer.energy_scores[0] == 0.0  # Trajectory 0's energy
        assert optimizer.energy_scores[1] == 0.0  # Trajectory 1's energy (improved)

    def test_trajectory_isolation_with_different_starting_points(self):
        """Tests that trajectories starting from different sequences remain isolated."""
        num_trajectories = 2
        proposals_per_result = 5
        seq_length = 20
        num_steps = 10

        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=2))
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
                num_results=num_trajectories,
                proposals_per_result=proposals_per_result,
                num_steps=num_steps,
                max_temperature=1.0,
                min_temperature=0.01,
                verbose=False,
            ),
        )

        # Give each trajectory a very different starting sequence
        segment.result_sequences[0].sequence = "A" * seq_length  # 0% GC
        segment.result_sequences[1].sequence = "G" * seq_length  # 100% GC

        optimizer.run()

        # Each trajectory's energy should be tracked independently
        # Check that history shows each trajectory improving independently
        for entry in optimizer.history:
            assert len(entry["results"]) == num_trajectories

        # Final sequences should be different (each evolved from different start)
        # With high probability, they won't converge to identical sequences
        # in just 10 steps with such different starting points
        final_seq_0 = segment.result_sequences[0].sequence
        final_seq_1 = segment.result_sequences[1].sequence

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
        num_trajectories = 1
        proposals_per_result = 3
        seq_length = 10

        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen.assign(segment)
        construct = Construct([segment])

        # Custom constraint: energy = count of non-A characters
        def count_non_a_energy(input_sequences, config=None):
            return [(seq_length - seq.sequence.count("A")) / seq_length for (seq,) in input_sequences]
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
                num_results=num_trajectories,
                proposals_per_result=proposals_per_result,
                num_steps=1,
                max_temperature=0.002,  # Very low temp = greedy
                min_temperature=0.001,
                verbose=False,
            ),
        )

        # Set initial sequence with energy=5
        segment.result_sequences[0].sequence = "AAAAAGGGGG"  # 5 non-A = energy 5
        optimizer.energy_scores[0] = 5.0

        old_result_sequences = optimizer._save_sequence_state()
        optimizer._populate_proposal_sequences()

        # Set up proposals with different energies: [0.8, 0.3, 0.9]
        # Best is at index 1 with energy 0.3
        segment.proposal_sequences[0].sequence = "AAAAAAAGGG"  # 3 non-A
        segment.proposal_sequences[1].sequence = "AAAAAAAAAT"  # 1 non-A (best)
        segment.proposal_sequences[2].sequence = "AAAAAAGGGG"  # 4 non-A
        optimizer.energy_scores = [3.0, 1.0, 4.0]

        optimizer._proposal_outcomes = ["accepted"] * optimizer.num_proposals
        optimizer._select_topk_with_mcmc_acceptance(step=1, old_result_sequences=old_result_sequences)

        # With low temperature, the best proposal (energy=1.0) should be accepted
        # because it improves from energy=5.0
        assert segment.result_sequences[0].sequence == "AAAAAAAAAT", (
            f"Expected best proposal to be accepted. Got: {segment.result_sequences[0].sequence}"
        )
        assert optimizer.energy_scores[0] == 1.0

        # Scenario 2: Test that only the best proposal is considered for MH
        # Set up a case where the best proposal worsens energy
        optimizer2 = MCMCOptimizer(
            constructs=[Construct([segment])],
            generators=[proposal_gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(
                num_results=num_trajectories,
                proposals_per_result=proposals_per_result,
                num_steps=1,
                max_temperature=0.002,  # Very low temp
                min_temperature=0.001,
                verbose=False,
            ),
        )

        # Start with a very good sequence (energy=0)
        segment.result_sequences[0].sequence = "A" * seq_length  # 0 non-A = energy 0
        optimizer2.energy_scores[0] = 0.0

        old_result_sequences2 = optimizer2._save_sequence_state()
        optimizer2._populate_proposal_sequences()

        # All proposals are worse than current (energy 0)
        segment.proposal_sequences[0].sequence = "AAAAAAAAAT"  # 1 non-A
        segment.proposal_sequences[1].sequence = "AAAAAAGGGG"  # 4 non-A
        segment.proposal_sequences[2].sequence = "AAAAAAAAAC"  # 1 non-A
        optimizer2.energy_scores = [1.0, 4.0, 1.0]

        optimizer2._proposal_outcomes = ["accepted"] * optimizer2.num_proposals
        optimizer2._select_topk_with_mcmc_acceptance(step=1, old_result_sequences=old_result_sequences2)

        # At very low temperature, worse proposals should be rejected
        # The old state should be kept
        assert segment.result_sequences[0].sequence == "A" * seq_length, (
            f"Expected rejection - old state should be kept. Got: {segment.result_sequences[0].sequence}"
        )
        assert optimizer2.energy_scores[0] == 0.0

    def test_proposal_tracking(self):
        """Tests that history has proposal_results with unified rejection reasons."""
        seq_length = 10
        segment = Segment(sequence="A" * seq_length, sequence_type="dna")
        proposal_gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3))
        )
        proposal_gen.assign(segment)
        construct = Construct([segment])

        gc_filter = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=45.0, max_gc=55.0),
            threshold=0.1,
        )

        optimizer = MCMCOptimizer(
            constructs=[construct],
            generators=[proposal_gen],
            constraints=[gc_filter],
            config=MCMCOptimizerConfig(
                num_results=2,
                proposals_per_result=3,
                num_steps=10,
                verbose=False,
                track_proposals=True,
            ),
        )

        optimizer.run()

        # Every step has proposal_results with correct structure
        valid_rejectors = {
            "gc_content_constraint",
            "Not best in proposal pool",
            "Metropolis-Hastings rejection",
        }
        all_rejectors = set()
        for entry in optimizer.history:
            assert "proposal_results" in entry
            for cand in entry["proposal_results"]:
                assert isinstance(cand["accepted"], bool)
                assert "rejected_by" in cand
                assert "constructs" in cand
                if cand["accepted"]:
                    assert cand["rejected_by"] is None
                else:
                    all_rejectors.add(cand["rejected_by"])

        assert all_rejectors.issubset(valid_rejectors)

    def test_tracking_interval(self):
        """tracking_interval=3 saves only steps {0, 3, 6, 9, 10}."""
        optimizer, _, _, _ = _setup_mcmc_components(
            seq_length=10,
            num_results=1,
            num_mcmc_steps=10,
        )
        optimizer.tracking_interval = 3

        optimizer.run()

        saved_steps = {entry["time_step"] for entry in optimizer.history}
        assert saved_steps == {0, 3, 6, 9, 10}

    def test_track_proposals_default_false(self):
        """track_proposals defaults to False — no proposal_results in snapshots."""
        optimizer, _, _, _ = _setup_mcmc_components(
            seq_length=10,
            num_results=1,
            num_mcmc_steps=3,
        )
        # Don't set track_proposals — should default to False
        optimizer.run()

        for entry in optimizer.history:
            assert "proposal_results" not in entry

    def test_mcmc_alpha_inf_inf_returns_zero(self):
        """inf vs inf should return 0.0 (reject, no improvement) instead of NaN."""
        optimizer, _, _, _ = _setup_mcmc_components()
        alpha = optimizer._compute_mcmc_alpha(float('inf'), float('inf'), 1)
        assert alpha == 0.0

    def test_mcmc_alpha_inf_current_accepts_finite(self):
        """inf current with finite proposed should return 1.0 (always accept)."""
        optimizer, _, _, _ = _setup_mcmc_components()
        alpha = optimizer._compute_mcmc_alpha(float('inf'), 0.5, 1)
        assert alpha == 1.0
        alpha = optimizer._compute_mcmc_alpha(float('inf'), 100.0, 1)
        assert alpha == 1.0

    def test_mcmc_alpha_finite_current_rejects_inf_proposed(self):
        """Finite current with inf proposed should return 0.0 (always reject)."""
        optimizer, _, _, _ = _setup_mcmc_components()
        alpha = optimizer._compute_mcmc_alpha(0.5, float('inf'), 1)
        assert alpha == 0.0
        alpha = optimizer._compute_mcmc_alpha(0.0, float('inf'), 1)
        assert alpha == 0.0

    def test_mcmc_alpha_negative_inf_proposed(self):
        """Negative inf proposed should be rejected (non-finite)."""
        optimizer, _, _, _ = _setup_mcmc_components()
        alpha = optimizer._compute_mcmc_alpha(0.5, float('-inf'), 1)
        assert alpha == 0.0

    def test_mcmc_alpha_negative_inf_current(self):
        """Negative inf current with finite proposed should accept."""
        optimizer, _, _, _ = _setup_mcmc_components()
        alpha = optimizer._compute_mcmc_alpha(float('-inf'), 0.5, 1)
        assert alpha == 1.0

    def test_checkpoint_resume_matches_continuous_run(self, tmp_path):
        """Checkpoint + resume run should match uninterrupted run."""
        seq_length = 30
        total_steps = 20
        split_steps = 8
        checkpoint_path = tmp_path / "mcmc_checkpoint.pkl"

        random.seed(1234)
        np.random.seed(1234)
        baseline_optimizer, _, _, baseline_segment = _setup_mcmc_components(
            seq_length=seq_length,
            num_results=1,
            num_mcmc_steps=total_steps,
        )
        baseline_optimizer.run()
        baseline_final_sequence = baseline_segment.result_sequences[0].sequence
        baseline_final_energy = baseline_optimizer.energy_scores[0]

        random.seed(1234)
        np.random.seed(1234)
        phase1_optimizer, _, _, _ = _setup_mcmc_components(
            seq_length=seq_length,
            num_results=1,
            num_mcmc_steps=split_steps,
        )
        phase1_optimizer.configure_checkpointing(
            checkpoint_path=checkpoint_path,
            save_interval_steps=1,
            resume=False,
        )
        phase1_optimizer.run()
        assert checkpoint_path.exists()

        # Use different seeds to ensure restored RNG state comes from checkpoint.
        random.seed(9999)
        np.random.seed(9999)
        resumed_optimizer, _, _, resumed_segment = _setup_mcmc_components(
            seq_length=seq_length,
            num_results=1,
            num_mcmc_steps=total_steps,
        )
        resumed_optimizer.configure_checkpointing(
            checkpoint_path=checkpoint_path,
            save_interval_steps=1,
            resume=True,
        )
        resumed_optimizer.run()

        assert resumed_segment.result_sequences[0].sequence == baseline_final_sequence
        assert resumed_optimizer.energy_scores[0] == pytest.approx(baseline_final_energy)

    def test_checkpoint_resume_rejects_num_results_mismatch(self, tmp_path):
        """Resume should fail if checkpoint and optimizer num_results mismatch."""
        checkpoint_path = tmp_path / "mcmc_checkpoint.pkl"

        optimizer1, _, _, _ = _setup_mcmc_components(
            seq_length=20,
            num_results=1,
            num_mcmc_steps=3,
        )
        optimizer1.configure_checkpointing(
            checkpoint_path=checkpoint_path,
            save_interval_steps=1,
            resume=False,
        )
        optimizer1.run()
        assert checkpoint_path.exists()

        optimizer2, _, _, _ = _setup_mcmc_components(
            seq_length=20,
            num_results=2,
            num_mcmc_steps=3,
        )
        optimizer2.configure_checkpointing(
            checkpoint_path=checkpoint_path,
            save_interval_steps=1,
            resume=True,
        )
        with pytest.raises(ValueError, match="Checkpoint num_results mismatch"):
            optimizer2.run()
