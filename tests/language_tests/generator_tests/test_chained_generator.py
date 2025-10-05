import pytest
import random
import numpy as np
import copy
from typing import Tuple

import sys

sys.path.append(".")
from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    SequenceType,
)
from proto_language.language.constraint import (
    gc_content_constraint,
    sequence_length_constraint,
)
from proto_language.language.generator import (
    UniformMutationGenerator,
    ChainedGenerator,
    MCMCGenerator,
)


# Helper function
def create_segment(sequence: str, seq_type: SequenceType = SequenceType.DNA) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


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
        },
    )
    constraint2 = Constraint(
        inputs=[segment2],
        scoring_function=gc_content_constraint,
        scoring_function_config={
            "min_gc": gc_target_range[0],
            "max_gc": gc_target_range[1],
        },
    )

    return (
        segment1,
        segment2,
        gen1,
        gen2,
        construct1,
        construct2,
        constraint1,
        constraint2,
    )


class TestChainedGenerator:
    def test_initialization(self):
        """Tests successful initialization of ChainedGenerator."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=3,
            verbose=False,
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
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Test empty stages list
        with pytest.raises(
            ValueError, match="At least one generator stage must be provided"
        ):
            ChainedGenerator([], verbose=False)

        # Test non-IterativeGenerator stage
        with pytest.raises(ValueError, match="must be an IterativeGenerator"):
            ChainedGenerator([gen1], verbose=False)  # gen1 is not an IterativeGenerator

        # Test mismatched batch sizes between stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )

        # Create stage with different batch size
        gen2_different_batch = UniformMutationGenerator(
            sequence_length=10, batch_size=3
        )
        gen2_different_batch.assign(segment2)
        stage2_different = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2_different_batch],
            constraints=[constraint2],
            num_steps=3,
            batch_size=3,  # Explicitly set different batch_size for stage2
            verbose=False,
        )

        with pytest.raises(ValueError, match="same batch_size"):
            ChainedGenerator([stage1, stage2_different], verbose=False)

    def test_basic_execution(self):
        """Tests basic execution of the chained generator."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator([stage1, stage2], verbose=False)

        # Run the pipeline
        chained.run()

        # Check that results were captured
        assert len(chained.stage_results) == 2
        assert chained.stage_results[0]["stage"] == 0
        assert chained.stage_results[1]["stage"] == 1
        assert chained.stage_results[0]["stage_type"] == "MCMCGenerator"
        assert chained.stage_results[1]["stage_type"] == "MCMCGenerator"

    def test_sequence_propagation(self):
        """Tests that sequences are properly propagated between stages."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator([stage1, stage2], verbose=False)

        # Run the pipeline
        chained.run()

        # Check that sequences were propagated
        # Stage 1 should have modified segment1
        stage1_constructs = chained.stage_results[0]["constructs"]
        stage2_constructs = chained.stage_results[1]["constructs"]

        # The sequences should be different from the initial "A" * 10
        # Access the sequence through the batch_sequences
        assert stage1_constructs[0].segments[0].batch_sequences[0].sequence != "A" * 10
        assert stage2_constructs[0].segments[0].batch_sequences[0].sequence != "C" * 10

    def test_metadata_capture(self):
        """Tests that metadata is properly captured from each stage."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator(
            [stage1, stage2], verbose=True, capture_metadata=True
        )

        # Run the pipeline
        chained.run()

        # Check metadata capture
        for i, result in enumerate(chained.stage_results):
            assert "stage" in result
            assert "stage_type" in result
            assert "constructs" in result
            assert "final_energy" in result
            assert "execution_time" in result
            assert "stage_config" in result
            assert "outputs_metadata" in result

            # Check specific values
            assert result["stage"] == i
            assert result["execution_time"] > 0
            assert len(result["constructs"]) > 0

    def test_results_access_methods(self):
        """Tests all the results access methods."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
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
        assert final_constructs == chained.stage_results[-1]["constructs"]

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
            assert "stage" in meta
            assert "stage_type" in meta
            assert "outputs_metadata" in meta
            assert "execution_summary" in meta

    def test_stage_access_methods(self):
        """Tests methods for accessing individual stages."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
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
        assert stage1_result["stage"] == 0
        assert stage2_result["stage"] == 1
        assert chained.get_stage_result(2) is None

    def test_execution_summary(self):
        """Tests the execution summary functionality."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator([stage1, stage2], verbose=False)

        # Test summary before running
        summary_before = chained.get_execution_summary()
        assert summary_before["total_stages"] == 2
        assert summary_before["total_execution_time"] == 0.0
        assert summary_before["final_energy"] is None
        assert summary_before["energy_progression"] == []
        assert summary_before["stage_types"] == ["MCMCGenerator", "MCMCGenerator"]

        # Run the pipeline
        chained.run()

        # Test summary after running
        summary_after = chained.get_execution_summary()
        assert summary_after["total_stages"] == 2
        assert summary_after["total_execution_time"] > 0
        assert summary_after["final_energy"] is not None
        assert len(summary_after["energy_progression"]) == 2
        assert len(summary_after["stage_types"]) == 2

    def test_energy_progression(self):
        """Tests the energy progression tracking."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
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
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator([stage1, stage2], verbose=False)

        # Test export before running
        with pytest.raises(RuntimeError, match="No results to export"):
            chained.export_results("test.json")

        # Run the pipeline
        chained.run()

        # Test JSON export
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            chained.export_results(temp_file, "json")
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        # Test pickle export
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            temp_file = f.name

        try:
            chained.export_results(temp_file, "pickle")
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            chained.export_results("test.txt", "txt")

    def test_verbose_execution(self):
        """Tests that verbose mode provides appropriate output."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator([stage1, stage2], verbose=True)

        # Run the pipeline (this should print progress)
        chained.run()

        # Check that results were captured despite verbose output
        assert len(chained.stage_results) == 2

    def test_metadata_capture_disabled(self):
        """Tests that metadata capture can be disabled."""
        (
            segment1,
            segment2,
            gen1,
            gen2,
            construct1,
            construct2,
            constraint1,
            constraint2,
        ) = _setup_chained_components()

        # Create stages
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )
        stage2 = MCMCGenerator(
            constructs=[construct2],
            generators=[gen2],
            constraints=[constraint2],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator(
            [stage1, stage2], verbose=False, capture_metadata=False
        )

        # Run the pipeline
        chained.run()

        # Check that basic results are still captured
        assert len(chained.stage_results) == 2

        # Check that outputs_metadata might be empty or minimal
        for result in chained.stage_results:
            assert "outputs_metadata" in result

    def test_single_stage_execution(self):
        """Tests execution with only one stage."""
        segment1, _, gen1, _, construct1, _, constraint1, _ = (
            _setup_chained_components()
        )

        # Create single stage
        stage1 = MCMCGenerator(
            constructs=[construct1],
            generators=[gen1],
            constraints=[constraint1],
            num_steps=2,
            verbose=False,
        )

        chained = ChainedGenerator([stage1], verbose=False)

        # Run the pipeline
        chained.run()

        # Check results
        assert len(chained.stage_results) == 1
        assert chained.stage_results[0]["stage"] == 0
        assert chained.stage_results[0]["stage_type"] == "MCMCGenerator"

        # Test final constructs access
        final_constructs = chained.get_final_constructs()
        assert len(final_constructs) > 0
