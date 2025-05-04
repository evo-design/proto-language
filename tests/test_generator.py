import pytest
import random
from typing import Tuple

import sys
sys.path.append(".")
from language.constraint import GCContentConstraint
from language.generator import ProgramMCMCGenerator, UniformMutationGenerator
from language.sequence import (
    ProgramDNASequence,
    ProgramRNASequence,
    ProgramProteinSequence,
)


##############################
## UniformMutationGenerator ##
##############################


def test_uniform_mutation_generator_init():
    """
    Tests the __init__ method for correct initialization of attributes
    based on sequence type (using 'dna' as an example).
    """
    seq_len = 15
    seq_type = "dna"
    gen = UniformMutationGenerator(sequence_length=seq_len, sequence_type=seq_type)

    assert gen.sequence_length == seq_len
    assert gen.sequence_type == seq_type
    assert gen.vocab == "ACGT"
    assert gen.sequence_class == ProgramDNASequence
    assert not gen._is_initialized
    assert gen.outputs is None
    # Check that invalid type raises ValueError.
    with pytest.raises(ValueError):
        UniformMutationGenerator(sequence_length=10, sequence_type="invalid_type")


def test_uniform_mutation_generator_register():
    """
    Tests the register method initializes the output sequence correctly
    (using 'rna' as an example).
    """
    seq_len = 20
    seq_type = "rna"
    gen = UniformMutationGenerator(sequence_length=seq_len, sequence_type=seq_type)

    # Test register without providing outputs.
    outputs = gen.register()
    assert gen._is_initialized
    assert len(outputs) == 1
    output_seq = outputs[0]
    assert isinstance(output_seq, ProgramRNASequence)
    assert output_seq.generator is gen
    assert output_seq.generator_output_idx == 0
    assert len(output_seq) == seq_len
    assert all(c in gen.vocab for c in output_seq.sequence)
    assert gen.get_outputs() == outputs

    # Reset and test register with valid provided outputs.
    gen = UniformMutationGenerator(sequence_length=seq_len, sequence_type=seq_type)
    predefined_seq = ProgramRNASequence(gen, 0, "A" * seq_len)
    outputs_pre = (predefined_seq,)
    registered_outputs = gen.register(outputs=outputs_pre)
    assert gen.outputs == outputs_pre
    assert registered_outputs == outputs_pre


def test_uniform_mutation_generator_sample():
    """
    Tests the sample method introduces a single valid mutation
    (using 'protein' as an example).
    """
    seq_len = 25
    seq_type = "protein"
    gen = UniformMutationGenerator(sequence_length=seq_len, sequence_type=seq_type)

    # Sample should implicitly call register if not initialized.
    assert not gen._is_initialized
    gen.sample()
    assert gen._is_initialized
    assert gen.outputs is not None
    initial_sequence = gen.outputs[0].sequence
    assert len(initial_sequence) == seq_len

    # Perform another sample.
    gen.sample()
    mutated_sequence = gen.outputs[0].sequence

    # Check length is maintained.
    assert len(mutated_sequence) == seq_len
    # Check exactly one position changed.
    diff_indices = [
        i for i, (a, b) in enumerate(zip(initial_sequence, mutated_sequence)) if a != b
    ]
    assert len(diff_indices) == 1
    mutated_index = diff_indices[0]
    # Check the new character is valid and different from the original.
    assert mutated_sequence[mutated_index] in gen.vocab
    assert mutated_sequence[mutated_index] != initial_sequence[mutated_index]


##########################
## ProgramMCMCGenerator ##
##########################


def _setup_mcmc_components(
    seq_length: int =10,
    gc_target_range: Tuple[int, int] = (40.0, 60.0),
    num_mcmc_steps: int = 10,
):
    """Helper function to set up a basic MCMC generator for testing."""
    # 1. Create and register the underlying proposal generator.
    proposal_gen = UniformMutationGenerator(
        sequence_length=seq_length, sequence_type="dna"
    )
    sequence_var = proposal_gen.register()[0]  # Register and get the sequence variable.

    # 2. Create the constraint linked to the sequence variable.
    constraint = GCContentConstraint(inputs=sequence_var, target_range=gc_target_range)

    # 3. Create the MCMC generator.
    mcmc_gen = ProgramMCMCGenerator(
        generators=[proposal_gen],
        constraints=[constraint],
        num_steps=num_mcmc_steps,  # Number of MCMC steps per sample() call.
    )
    return mcmc_gen, proposal_gen, constraint, sequence_var


def test_program_mcmc_generator_init_and_register():
    """
    Tests successful initialization and registration of ProgramMCMCGenerator.
    Also checks error handling for unregistered generators.
    """
    seq_len = 10
    # Setup with registered generator
    mcmc_gen, proposal_gen, constraint, sequence_var = _setup_mcmc_components(
        seq_length=seq_len
    )

    # Check initialization attributes.
    assert mcmc_gen.generators == [proposal_gen]
    assert mcmc_gen.constraints == [constraint]
    assert mcmc_gen.constraint_weights == [1.0]  # Default weight.
    assert mcmc_gen.num_steps == 10
    assert not mcmc_gen._is_initialized  # Should not be initialized yet.

    # Test registration.
    mcmc_outputs = mcmc_gen.register()
    assert mcmc_gen._is_initialized
    assert len(mcmc_outputs) == 1
    assert mcmc_outputs[0] is sequence_var  # Should be the same object.
    assert mcmc_gen.get_outputs() == (sequence_var,)

    # Test initialization fails if proposal generator isn't registered.
    unregistered_gen = UniformMutationGenerator(
        sequence_length=seq_len, sequence_type="dna"
    )
    with pytest.raises(ValueError) as excinfo:
        ProgramMCMCGenerator(
            generators=[unregistered_gen], constraints=[]
        )  # No constraint needed here.
    assert "Not all generators have been registered" in str(excinfo.value)

    # Test initialization fails if weights do not match constraints.
    mcmc_gen, proposal_gen, constraint, sequence_var = _setup_mcmc_components(
        seq_length=seq_len
    )
    with pytest.raises(ValueError) as excinfo:
        ProgramMCMCGenerator(
            generators=[proposal_gen],
            constraints=[constraint],
            constraint_weights=[1.0, 0.5],  # 2 weights, 1 constraint.
        )
    assert "Constraint weights must match number of constraints" in str(excinfo.value)


def test_program_mcmc_generator_sample():
    """
    Tests that the sample method runs and potentially modifies the sequence.
    """
    seq_len = 20
    # Use a constraint that encourages change (e.g., very high GC).
    mcmc_gen, _, _, sequence_var = _setup_mcmc_components(
        seq_length=seq_len,
        gc_target_range=(80.0, 90.0),  # Encourage high GC.
        num_mcmc_steps=50,  # More steps increase chance of seeing a change.
    )
    mcmc_gen.register()  # Register the MCMC generator itself.

    # Set a starting sequence that does not meet the high GC target.
    start_sequence = "A" * seq_len
    sequence_var.sequence = start_sequence
    assert sequence_var.sequence == start_sequence

    # Run sample (which executes num_mcmc_steps internally).
    mcmc_gen.sample()

    final_sequence = sequence_var.sequence

    # Assert basic properties.
    assert isinstance(final_sequence, str)
    assert len(final_sequence) == seq_len


def test_program_mcmc_generator_score_energy():
    """
    Tests the score_energy and score_energy_additive methods.
    """
    seq_len = 10
    gc_min, gc_max = 40.0, 60.0
    mcmc_gen, _, constraint, sequence_var = _setup_mcmc_components(
        seq_length=seq_len, gc_target_range=(gc_min, gc_max)
    )
    mcmc_gen.register()

    # Test case 1: Sequence within target GC range (score should be 0.0).
    # GC = 5/10 = 50% -> within [40, 60],
    test_seq_good = "GCGCGAATTA"
    sequence_var.sequence = test_seq_good
    expected_score_good = 0.0  # GCContentConstraint returns 0 if in range.
    assert constraint.evaluate() == expected_score_good
    assert mcmc_gen.score_energy() == expected_score_good  # Multiplicative (1.0 * 0.0).
    assert (
        mcmc_gen.score_energy_additive() == expected_score_good
    )  # Additive (1.0 * 0.0).

    # Test case 2: Sequence below target GC range (score > 0.0).
    # GC = 2/10 = 20% -> below 40%.
    test_seq_low_gc = "GCTTAATTAA"
    sequence_var.sequence = test_seq_low_gc
    gc_content_low = 20.0
    # Calculate expected deviation score for GCContentConstraint.
    # deviation = (min_gc - gc_content) / min_gc = (40 - 20) / 40 = 0.5.
    expected_score_low = 0.5
    assert (
        abs(constraint.evaluate() - expected_score_low) < 1e-9
    )  # Use tolerance for float compare.
    assert abs(mcmc_gen.score_energy() - expected_score_low) < 1e-9  # Multiplicative.
    assert abs(mcmc_gen.score_energy_additive() - expected_score_low) < 1e-9  # Additive.

    # Test case 3: Sequence above target GC range (score > 0.0).
    # GC = 8/10 = 80% -> above 60%.
    test_seq_high_gc = "GCGCGCGCAT"
    sequence_var.sequence = test_seq_high_gc
    gc_content_high = 80.0
    # Calculate expected deviation score for GCContentConstraint.
    # deviation = (gc_content - max_gc) / (100 - max_gc) = (80 - 60) / (100 - 60) = 20 / 40 = 0.5.
    expected_score_high = 0.5
    assert abs(constraint.evaluate() - expected_score_high) < 1e-9
    assert abs(mcmc_gen.score_energy() - expected_score_high) < 1e-9  # Multiplicative.
    assert (
        abs(mcmc_gen.score_energy_additive() - expected_score_high) < 1e-9
    )  # Additive.

    # Test with different weights.
    mcmc_gen.constraint_weights = [0.5]
    assert (
        abs(mcmc_gen.score_energy() - (0.5 * expected_score_high)) < 1e-9
    )  # Multiplicative.
    assert (
        abs(mcmc_gen.score_energy_additive() - (0.5 * expected_score_high)) < 1e-9
    )  # Additive.
