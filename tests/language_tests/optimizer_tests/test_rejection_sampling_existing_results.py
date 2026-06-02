"""Tests for rejection-sampling over existing upstream candidates."""

import pytest

from proto_language.constraint import gc_content_constraint
from proto_language.core import Constraint, Construct, Program, Segment, Sequence
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig


def test_rejection_sampling_scores_existing_results_without_generators():
    segment = Segment(sequence="AAAA", sequence_type="dna", label="candidate")
    segment.result_sequences = [
        Sequence("AAAA", "dna"),
        Sequence("AAGG", "dna"),
        Sequence("GGGG", "dna"),
    ]
    construct = Construct([segment], label="dna")
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 40, "max_gc": 60},
    )
    optimizer = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[],
        constraints=[constraint],
        config=RejectionSamplingOptimizerConfig(
            proposal_source="existing_results",
            num_samples=3,
            num_results=2,
            track_proposals=True,
        ),
    )

    program = Program(optimizers=[optimizer], num_results=2)
    program.run()

    assert [seq.sequence for seq in segment.result_sequences] == ["AAGG", "AAAA"]
    assert optimizer.energy_scores == [0.0, 1.0]
    assert optimizer.history
    assert optimizer.history[-1]["optimizer"]["type"] == "rejection-sampling"
    assert optimizer.history[-1]["optimizer"]["proposal_source"] == "existing_results"
    assert optimizer.history[-1]["optimizer"]["proposal_number"] == 3
    assert len(optimizer.history) == 3
    assert len(optimizer.history[-1]["proposal_results"]) == 1


def test_rejection_sampling_existing_results_allows_later_stage_generated_inputs():
    segment = Segment(length=10, sequence_type="dna", label="generated-upstream")
    construct = Construct([segment], label="dna")
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 40, "max_gc": 60},
    )

    optimizer = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[],
        constraints=[constraint],
        config=RejectionSamplingOptimizerConfig(proposal_source="existing_results", num_samples=1, num_results=1),
    )

    assert optimizer.generators == []


def test_rejection_sampling_existing_results_rejects_generators():
    segment = Segment(sequence="AAAA", sequence_type="dna", label="candidate")
    construct = Construct([segment], label="dna")
    generator = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    generator.assign(segment)
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 40, "max_gc": 60},
    )

    with pytest.raises(ValueError, match="does not accept generators"):
        RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=RejectionSamplingOptimizerConfig(proposal_source="existing_results", num_samples=1, num_results=1),
        )


def test_rejection_sampling_generated_mode_rejects_missing_generators():
    segment = Segment(sequence="AAAA", sequence_type="dna", label="candidate")
    construct = Construct([segment], label="dna")
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 40, "max_gc": 60},
    )

    with pytest.raises(ValueError, match="requires at least one generator"):
        RejectionSamplingOptimizer(
            constructs=[construct],
            generators=[],
            constraints=[constraint],
            config=RejectionSamplingOptimizerConfig(num_samples=1, num_results=1),
        )
