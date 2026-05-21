"""Tests for CyclingOptimizer."""

import copy
from pathlib import Path
from typing import Any

import pytest
from proto_tools import InverseFoldingStructureInput, Structure
from pydantic import BaseModel, ValidationError

from proto_language.core import Constraint, ConstraintOutput, Construct, Segment, Sequence
from proto_language.generator import (
    LigandMPNNGenerator,
    LigandMPNNGeneratorConfig,
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)
from proto_language.optimizer import CyclingOptimizer, CyclingOptimizerConfig

# =============================================================================
# Helpers
# =============================================================================


class EmptyConfig(BaseModel):
    pass


def make_mock_structure() -> Structure:
    """Create a minimal mock Structure."""
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
END
"""
    return Structure(structure=pdb_content)


def make_mock_conditioning_fn(num_proposals: int):
    """Create a mock conditioning function that returns structures."""
    structures = [make_mock_structure() for _ in range(num_proposals)]

    def conditioning_fn(sequences: list[Sequence]) -> list[Structure]:
        for seq, struct in zip(sequences, structures, strict=True):
            seq.structure = struct
        return structures

    return conditioning_fn, structures


def _setup_cycling_components(
    num_steps: int = 2,
    num_results: int = 2,
    include_constraint: bool = False,
    constraint_passes: bool = True,
):
    """Helper to set up CyclingOptimizer with mocked components."""
    target_segment = Segment(sequence="ACDEFGHIKLMNPQRSTVWY", sequence_type="protein")
    construct = Construct([target_segment])

    mock_structure = make_mock_structure()
    generator = ProteinMPNNGenerator(
        ProteinMPNNGeneratorConfig(structure_inputs=InverseFoldingStructureInput(structure=mock_structure))
    )
    generator.assign(target_segment)

    constraints = []
    if include_constraint:

        def filter_func(input_sequences, config=None):
            return [ConstraintOutput(score=(0.0 if constraint_passes else 1.0)) for _ in input_sequences]

        filter_func._constraint_config_class = EmptyConfig
        filter_func._constraint_supported_sequence_types = ["protein"]
        constraints.append(
            Constraint(
                inputs=[target_segment],
                function=filter_func,
                function_config=EmptyConfig(),
                threshold=0.5,
            )
        )

    config = CyclingOptimizerConfig(
        num_steps=num_steps,
        num_results=num_results,
    )

    conditioning_fn, _ = make_mock_conditioning_fn(num_results)

    return {
        "target_segment": target_segment,
        "construct": construct,
        "generator": generator,
        "constraints": constraints,
        "config": config,
        "conditioning_fn": conditioning_fn,
    }


# =============================================================================
# Config Tests
# =============================================================================


class TestCyclingOptimizerConfig:
    """Tests for CyclingOptimizerConfig validation."""

    def test_valid_config(self):
        """Test valid configuration and defaults."""
        config = CyclingOptimizerConfig(num_steps=5, num_results=4)
        assert config.num_steps == 5
        assert config.num_results == 4
        assert config.verbose is False

    def test_invalid_config_values(self):
        """Test that invalid config values are rejected."""
        with pytest.raises(ValidationError):
            CyclingOptimizerConfig(num_steps=0, num_results=1)
        with pytest.raises(ValidationError):
            CyclingOptimizerConfig(num_steps=1, num_results=0)

    def test_protein_hunter_rejects_alphafold2(self):
        """AF2 is deterministic in our codepath; keep it out of cycling pipelines."""
        from proto_language.optimizer.cycling_optimizer import ProteinHunterPipelineConfig

        with pytest.raises(ValidationError):
            ProteinHunterPipelineConfig(structure_tool="alphafold2")


# =============================================================================
# Validation Tests
# =============================================================================


class TestCyclingOptimizerValidation:
    """Tests for CyclingOptimizer initialization validation."""

    def test_requires_exactly_one_generator(self):
        """Test that exactly one generator is required."""
        components = _setup_cycling_components()

        with pytest.raises(ValueError, match="requires exactly one generator"):
            CyclingOptimizer(
                target_segment=components["target_segment"],
                constructs=[components["construct"]],
                generators=[],
                constraints=[],
                config=components["config"],
                conditioning_fn=components["conditioning_fn"],
            )

    def test_target_segment_must_be_in_constructs(self):
        """Test that target_segment must belong to a construct."""
        components = _setup_cycling_components()
        orphan_segment = Segment(sequence="ACDEFGHIK", sequence_type="protein")

        with pytest.raises(ValueError, match="is not in any of the provided constructs"):
            CyclingOptimizer(
                target_segment=orphan_segment,
                constructs=[components["construct"]],
                generators=[components["generator"]],
                constraints=[],
                config=components["config"],
                conditioning_fn=components["conditioning_fn"],
            )

    def test_constraints_must_be_filters(self):
        """Test that constraints must have threshold set (filter mode)."""
        components = _setup_cycling_components()

        def scoring_func(input_sequences, config=None):
            return [ConstraintOutput(score=0.5) for _ in input_sequences]

        scoring_func._constraint_config_class = EmptyConfig
        scoring_func._constraint_supported_sequence_types = ["protein"]

        scoring_constraint = Constraint(
            inputs=[components["target_segment"]],
            function=scoring_func,
            function_config=EmptyConfig(),
        )

        with pytest.raises(ValueError, match="only supports filter constraints"):
            CyclingOptimizer(
                target_segment=components["target_segment"],
                constructs=[components["construct"]],
                generators=[components["generator"]],
                constraints=[scoring_constraint],
                config=components["config"],
                conditioning_fn=components["conditioning_fn"],
            )

    def test_constraints_on_non_target_segment_rejected(self):
        """Constraints referencing only non-target segments are rejected."""
        components = _setup_cycling_components()
        context_segment = Segment(sequence="M" * 20, sequence_type="protein")
        construct = Construct([components["target_segment"], context_segment])

        def filter_func(input_sequences, config=None):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        filter_func._constraint_config_class = EmptyConfig
        filter_func._constraint_supported_sequence_types = ["protein"]

        non_target_constraint = Constraint(
            inputs=[context_segment],
            function=filter_func,
            function_config=EmptyConfig(),
            threshold=0.5,
        )

        with pytest.raises(ValueError, match="does not include the target segment"):
            CyclingOptimizer(
                target_segment=components["target_segment"],
                constructs=[construct],
                generators=[components["generator"]],
                constraints=[non_target_constraint],
                config=components["config"],
                conditioning_fn=components["conditioning_fn"],
            )

    def test_duplicate_constraint_instance_fails(self):
        """Same constraint instance cannot be passed twice."""
        components = _setup_cycling_components()

        def filter_func(input_sequences, config=None):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        filter_func._constraint_config_class = EmptyConfig
        filter_func._constraint_supported_sequence_types = ["protein"]

        constraint = Constraint(
            inputs=[components["target_segment"]],
            function=filter_func,
            function_config=EmptyConfig(),
            threshold=0.5,
        )

        with pytest.raises(ValueError, match="appears multiple times"):
            CyclingOptimizer(
                target_segment=components["target_segment"],
                constructs=[components["construct"]],
                generators=[components["generator"]],
                constraints=[constraint, constraint],
                config=components["config"],
                conditioning_fn=components["conditioning_fn"],
            )


# =============================================================================
# Run Method Tests
# =============================================================================


class TestCyclingOptimizerRun:
    """Tests for the CyclingOptimizer run method."""

    def test_run_completes_and_tracks_history(self):
        """Test run completes, calls conditioning_fn, and tracks history."""
        num_steps, num_proposals = 3, 2
        components = _setup_cycling_components(num_steps=num_steps, num_results=num_proposals)

        # Track conditioning function calls
        call_count = [0]
        original_fn = components["conditioning_fn"]

        def tracked_conditioning_fn(sequences):
            call_count[0] += 1
            return original_fn(sequences)

        # Mock the generator.sample to update sequences
        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].proposal_sequences:
                c.sequence = "MKTAYIAKQRQISFVKSHFS"

        components["generator"].sample = mock_sample

        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=components["config"],
            conditioning_fn=tracked_conditioning_fn,
        )
        optimizer.run()

        assert call_count[0] == num_steps
        # +1 for the always-saved initial (time_step=0) snapshot.
        assert len(optimizer.history) == num_steps + 1
        for entry in optimizer.history:
            assert "time_step" in entry and "results" in entry

    def test_filter_constraint_rejection_preserves_result(self):
        """Test that result_sequences stay unchanged when all proposals fail."""
        components = _setup_cycling_components(
            num_steps=1,
            num_results=2,
            include_constraint=True,
            constraint_passes=False,
        )

        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].proposal_sequences:
                c.sequence = "NEWSEQENCEAAAAAAAAAAA"

        components["generator"].sample = mock_sample

        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=components["constraints"],
            config=components["config"],
            conditioning_fn=components["conditioning_fn"],
        )

        original_seqs = [copy.deepcopy(s.sequence) for s in components["target_segment"].result_sequences]
        optimizer.run()

        # All should stay unchanged in result_sequences since constraint fails all
        for i, result_seq in enumerate(components["target_segment"].result_sequences):
            assert result_seq.sequence == original_seqs[i]

    def test_partial_filter_acceptance(self):
        """Test that only passing proposals update result_sequences."""
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=InverseFoldingStructureInput(structure=make_mock_structure()))
        )
        generator.assign(target_segment)

        def partial_filter(input_sequences, config=None):
            return [ConstraintOutput(score=(1.0 if "FAIL" in seq.sequence else 0.0)) for (seq,) in input_sequences]

        partial_filter._constraint_config_class = EmptyConfig
        partial_filter._constraint_supported_sequence_types = ["protein"]

        constraint = Constraint(
            inputs=[target_segment],
            function=partial_filter,
            function_config=EmptyConfig(),
            threshold=0.5,
        )

        conditioning_fn, _ = make_mock_conditioning_fn(3)

        pass_seq, fail_seq = "MKTAYIAKQRQISFVKSHFS", "FAILAYIAKQRQISFVKSHF"

        def mock_sample(structure_inputs=None):
            target_segment.proposal_sequences[0].sequence = pass_seq
            target_segment.proposal_sequences[1].sequence = fail_seq
            target_segment.proposal_sequences[2].sequence = fail_seq

        generator.sample = mock_sample

        config = CyclingOptimizerConfig(
            num_steps=1,
            num_results=3,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            conditioning_fn=conditioning_fn,
        )

        original_seqs = [copy.deepcopy(s.sequence) for s in target_segment.result_sequences]
        optimizer.run()

        # Proposal 0 passed → result updated
        assert target_segment.result_sequences[0].sequence == pass_seq
        # Proposals 1, 2 failed → result unchanged
        assert target_segment.result_sequences[1].sequence == original_seqs[1]
        assert target_segment.result_sequences[2].sequence == original_seqs[2]

    def test_conditioning_fn_wrong_length_raises(self):
        """Test that conditioning_fn returning wrong number of items raises ValueError.

        This is a regression test for a bug where a mismatched conditioning_fn return
        length could cause silent failures or unexpected behavior in the generator.
        """
        components = _setup_cycling_components(num_steps=1, num_results=3)

        # Create a conditioning function that returns wrong number of items
        def wrong_length_conditioning_fn(sequences: list[Sequence]):
            # Returns only 1 item instead of num_proposals (3)
            return [make_mock_structure()]

        # Mock the generator.sample to not actually run
        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].proposal_sequences:
                c.sequence = "MKTAYIAKQRQISFVKSHFS"

        components["generator"].sample = mock_sample

        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=components["config"],
            conditioning_fn=wrong_length_conditioning_fn,
        )

        # Should raise ValueError with informative message
        with pytest.raises(ValueError, match="conditioning_fn returned 1 items, expected 3"):
            optimizer.run()

    def test_conditioning_fn_too_many_items_raises(self):
        """Test that conditioning_fn returning too many items also raises ValueError."""
        components = _setup_cycling_components(num_steps=1, num_results=2)

        # Create a conditioning function that returns too many items
        def too_many_conditioning_fn(sequences: list[Sequence]):
            # Returns 5 items instead of num_proposals (2)
            return [make_mock_structure() for _ in range(5)]

        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].proposal_sequences:
                c.sequence = "MKTAYIAKQRQISFVKSHFS"

        components["generator"].sample = mock_sample

        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=components["config"],
            conditioning_fn=too_many_conditioning_fn,
        )

        with pytest.raises(ValueError, match="conditioning_fn returned 5 items, expected 2"):
            optimizer.run()


# =============================================================================
# Accept Pattern Behavior Tests
# =============================================================================


class TestAcceptPatternBehavior:
    """Tests for the accept pattern: passing proposals update result, failed stay unchanged."""

    def test_all_rejected_result_unchanged(self):
        """Test that all-fail → result stays at initial state, energy stays inf."""
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=InverseFoldingStructureInput(structure=make_mock_structure()))
        )
        generator.assign(target_segment)

        def always_fail_filter(input_sequences, config=None):
            return [ConstraintOutput(score=1.0) for _ in input_sequences]

        always_fail_filter._constraint_config_class = EmptyConfig
        always_fail_filter._constraint_supported_sequence_types = ["protein"]

        constraint = Constraint(
            inputs=[target_segment],
            function=always_fail_filter,
            function_config=EmptyConfig(),
            threshold=0.5,
            label="test_filter",
        )

        conditioning_fn, _ = make_mock_conditioning_fn(2)

        new_seq = "MKTAYIAKQRQISFVKSHFS"

        def mock_sample(structure_inputs=None):
            for c in target_segment.proposal_sequences:
                c.sequence = new_seq

        generator.sample = mock_sample

        config = CyclingOptimizerConfig(
            num_steps=1,
            num_results=2,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            conditioning_fn=conditioning_fn,
        )

        original_seqs = [copy.deepcopy(s.sequence) for s in target_segment.result_sequences]
        optimizer.run()

        # Result sequences should be unchanged (all rejected)
        for i, result_seq in enumerate(target_segment.result_sequences):
            assert result_seq.sequence == original_seqs[i]

        # Energy scores should stay at inf (never accepted)
        assert all(e == float("inf") for e in optimizer.energy_scores)

    def test_energy_tracks_accepted_values(self):
        """Test that partial acceptance → correct energies in snapshot."""
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=InverseFoldingStructureInput(structure=make_mock_structure()))
        )
        generator.assign(target_segment)

        # Filter that passes proposal 0, rejects proposal 1
        def partial_filter(input_sequences, config=None):
            return [ConstraintOutput(score=(0.0 if idx == 0 else 1.0)) for idx, _ in enumerate(input_sequences)]

        partial_filter._constraint_config_class = EmptyConfig
        partial_filter._constraint_supported_sequence_types = ["protein"]

        constraint = Constraint(
            inputs=[target_segment],
            function=partial_filter,
            function_config=EmptyConfig(),
            threshold=0.5,
            label="partial_filter",
        )

        conditioning_fn, _ = make_mock_conditioning_fn(2)

        pass_seq = "MKTAYIAKQRQISFVKSHFS"
        fail_seq = "GPLAFVTNLTGLRSQNEEIR"

        def mock_sample(structure_inputs=None):
            target_segment.proposal_sequences[0].sequence = pass_seq
            target_segment.proposal_sequences[1].sequence = fail_seq

        generator.sample = mock_sample

        config = CyclingOptimizerConfig(
            num_steps=1,
            num_results=2,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            conditioning_fn=conditioning_fn,
        )

        optimizer.run()

        # Proposal 0 passed → result updated
        assert target_segment.result_sequences[0].sequence == pass_seq
        # Proposal 1 rejected → result unchanged
        assert target_segment.result_sequences[1].sequence == "A" * 20

        # Energy for proposal 0: accepted score (0.0 from filter-only)
        assert optimizer.energy_scores[0] == 0.0
        # Energy for proposal 1: still inf (never accepted)
        assert optimizer.energy_scores[1] == float("inf")

    def test_conditioning_reads_from_result_sequences(self):
        """Test that conditioning fn receives result_sequences, not proposals."""
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=InverseFoldingStructureInput(structure=make_mock_structure()))
        )
        generator.assign(target_segment)

        received_sequences = []

        def tracking_conditioning_fn(sequences):
            received_sequences.append([s.sequence for s in sequences])
            return [make_mock_structure() for _ in sequences]

        def mock_sample(structure_inputs=None):
            for c in target_segment.proposal_sequences:
                c.sequence = "MKTAYIAKQRQISFVKSHFS"

        generator.sample = mock_sample

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[],
            config=config,
            conditioning_fn=tracking_conditioning_fn,
        )

        optimizer.run()

        # Step 1: conditioning fn receives initial result_sequences (all "A" * 20)
        assert all(s == "A" * 20 for s in received_sequences[0])
        # Step 2: conditioning fn receives updated result (from step 1 acceptance)
        assert all(s == "MKTAYIAKQRQISFVKSHFS" for s in received_sequences[1])

    def test_multi_step_rejection_preserves_previous_accepted(self):
        """Test: step 1 accepts, step 2 rejects → result retains step 1 state."""
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=InverseFoldingStructureInput(structure=make_mock_structure()))
        )
        generator.assign(target_segment)

        filter_call_count = [0]

        # Filter call 1 (step 1): pass all. Filter call 2 (step 2): fail all.
        def step_dependent_filter(input_sequences, config=None):
            filter_call_count[0] += 1
            if filter_call_count[0] == 1:
                return [ConstraintOutput(score=0.0) for _ in input_sequences]  # Pass
            return [ConstraintOutput(score=1.0) for _ in input_sequences]  # Fail

        step_dependent_filter._constraint_config_class = EmptyConfig
        step_dependent_filter._constraint_supported_sequence_types = ["protein"]

        constraint = Constraint(
            inputs=[target_segment],
            function=step_dependent_filter,
            function_config=EmptyConfig(),
            threshold=0.5,
        )

        conditioning_fn, _ = make_mock_conditioning_fn(2)

        step1_seq = "MKTAYIAKQRQISFVKSHFS"
        step2_seq = "GPLAFVTNLTGLRSQNEEIR"

        sample_call_count = [0]

        def mock_sample(structure_inputs=None):
            sample_call_count[0] += 1
            seq = step1_seq if sample_call_count[0] == 1 else step2_seq
            for c in target_segment.proposal_sequences:
                c.sequence = seq

        generator.sample = mock_sample

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            conditioning_fn=conditioning_fn,
        )

        optimizer.run()

        # Step 2 rejected → result retains step 1's accepted sequence
        for result_seq in target_segment.result_sequences:
            assert result_seq.sequence == step1_seq


# =============================================================================
# Structure-on-Sequence Tests
# =============================================================================


class TestStructureOnSequence:
    """Tests for structure data flowing through Sequence objects."""

    def test_structure_preserved_through_accept_cycle(self):
        """Structure set on proposals survives the accept→deepcopy→result cycle."""
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])

        mock_struct = make_mock_structure()
        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(structure_inputs=InverseFoldingStructureInput(structure=mock_struct))
        )
        generator.assign(target_segment)

        conditioning_fn, _ = make_mock_conditioning_fn(2)

        def mock_sample_with_structure(structure_inputs=None):
            for c in target_segment.proposal_sequences:
                c.sequence = "MKTAYIAKQRQISFVKSHFS"
                c.structure = mock_struct

        generator.sample = mock_sample_with_structure

        config = CyclingOptimizerConfig(
            num_steps=1,
            num_results=2,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[],
            config=config,
            conditioning_fn=conditioning_fn,
        )

        optimizer.run()

        # No constraints → all proposals accepted → deepcopied to result_sequences
        # Structure should survive the deepcopy (shared reference)
        for result_seq in target_segment.result_sequences:
            assert result_seq.structure is mock_struct


# =============================================================================
# GPU Integration Tests
# =============================================================================


TEST_PDB_FILE = Path(__file__).parent.parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.fixture(scope="module")
def pdb_structure():
    """Load test PDB structure."""
    return Structure(structure=TEST_PDB_FILE)


@pytest.mark.uses_gpu
class TestCyclingOptimizerGPU:
    """Integration tests with real models (require GPU)."""

    @pytest.mark.slow
    def test_full_cycle_with_proteinmpnn(self, pdb_structure):
        """Test complete optimization cycle with LigandMPNN."""
        from proto_tools import StructurePredictionComplex, predict_structures

        chain_seq = pdb_structure.get_chain_sequence("A")
        seq_length = len(chain_seq)

        # Inverse folding generators auto-initialize to "X" when no sequence provided
        target_segment = Segment(sequence="X" * seq_length, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(structure=pdb_structure, chains_to_redesign=["A"]),
                temperature=0.1,
            )
        )
        generator.assign(target_segment)

        def structure_conditioning_fn(sequences):
            complexes = [StructurePredictionComplex(chains=[seq.sequence]) for seq in sequences]
            return predict_structures(complexes, "chai1", {}).structures

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
            verbose=True,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[],
            config=config,
            conditioning_fn=structure_conditioning_fn,
        )
        optimizer.run()

        assert len(target_segment.result_sequences) == 2
        for seq in target_segment.result_sequences:
            assert len(seq.sequence) == seq_length
            assert seq.sequence != "X" * seq_length

    @pytest.mark.slow
    def test_with_filter_constraint(self, pdb_structure):
        """Test with filter constraint using real models."""
        from proto_tools import StructurePredictionComplex, predict_structures

        chain_seq = pdb_structure.get_chain_sequence("A")
        seq_length = len(chain_seq)

        # Inverse folding generators auto-initialize to "X" when no sequence provided
        target_segment = Segment(sequence="X" * seq_length, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(structure=pdb_structure, chains_to_redesign=["A"]),
                temperature=0.1,
            )
        )
        generator.assign(target_segment)

        def length_filter(input_sequences, config=None):
            return [ConstraintOutput(score=(0.0 if len(seq.sequence) > 10 else 1.0)) for (seq,) in input_sequences]

        length_filter._constraint_config_class = EmptyConfig
        length_filter._constraint_supported_sequence_types = ["protein"]

        constraint = Constraint(
            inputs=[target_segment],
            function=length_filter,
            function_config=EmptyConfig(),
            threshold=0.5,
        )

        def structure_conditioning_fn(sequences):
            complexes = [StructurePredictionComplex(chains=[seq.sequence]) for seq in sequences]
            return predict_structures(complexes, "chai1", {}).structures

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
            verbose=True,
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            conditioning_fn=structure_conditioning_fn,
        )
        optimizer.run()

        assert len(target_segment.result_sequences) == 2
        for seq in target_segment.result_sequences:
            assert len(seq.sequence) > 10


# =============================================================================
# State Restart Tests
# =============================================================================


class TestCyclingOptimizerRestart:
    """Tests for CyclingOptimizer state restart behavior."""

    def test_run_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        components = _setup_cycling_components(num_steps=2, num_results=2)

        # Mock the generator.sample to track calls and modify sequences
        call_count = [0]
        # Valid 20-char protein sequences for each call
        protein_seqs = ["MKTAYIAKQRQISFVKSHFS", "GPLAFVTNLTGLRSQNEEIR", "YWDEIKNPLGRAVTYDKWFP", "HCLQMNSGVEATRIDFWYKP"]

        def mock_sample(structure_inputs=None):
            call_count[0] += 1
            for c in components["target_segment"].proposal_sequences:
                c.sequence = protein_seqs[call_count[0] % len(protein_seqs)]

        components["generator"].sample = mock_sample

        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=components["config"],
            conditioning_fn=components["conditioning_fn"],
        )

        # First run
        optimizer.run()
        assert optimizer._initial_state is not None
        first_run_calls = call_count[0]
        first_run_seqs = [s.sequence for s in components["target_segment"].result_sequences]

        # Second run should restart - call count continues but state is fresh
        optimizer.run()
        assert call_count[0] > first_run_calls
        second_run_seqs = [s.sequence for s in components["target_segment"].result_sequences]

        # Verify sequences were modified from original (mock changes them to valid protein seqs)
        original_seq = "ACDEFGHIKLMNPQRSTVWY"
        assert all(seq != original_seq for seq in first_run_seqs)
        assert all(seq != original_seq for seq in second_run_seqs)
        # History from second run only (cleared on restart): step 0 (initial) + steps 1, 2.
        assert len(optimizer.history) == 3

    def test_initial_state_captured_correctly(self):
        """Test that initial state captures segment state with actual sequence content."""
        components = _setup_cycling_components(num_steps=1, num_results=2)

        def mock_sample(structure_inputs=None):
            pass  # Don't modify sequences

        components["generator"].sample = mock_sample

        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=components["config"],
            conditioning_fn=components["conditioning_fn"],
        )

        # Capture original sequences before run
        original_result = [copy.deepcopy(s) for s in components["target_segment"].result_sequences]
        original_proposals = [copy.deepcopy(s) for s in components["target_segment"].proposal_sequences]

        optimizer.run()

        # Verify state was captured
        assert optimizer._initial_state is not None
        assert len(optimizer._initial_state["segments"]) == 1

        # Verify captured state contains actual sequence content (using index 0)
        captured_result = optimizer._initial_state["segments"][0]["result"]
        captured_proposals = optimizer._initial_state["segments"][0]["proposals"]

        assert len(captured_result) == len(original_result)
        assert len(captured_proposals) == len(original_proposals)

        # Verify sequences match
        for orig, captured in zip(original_result, captured_result, strict=False):
            assert orig.sequence == captured["sequence"]
            assert orig.sequence_type == captured["sequence_type"]

        for orig, captured in zip(original_proposals, captured_proposals, strict=False):
            assert orig.sequence == captured["sequence"]
            assert orig.sequence_type == captured["sequence_type"]


# =============================================================================
# Pipeline Resolution Tests
# =============================================================================


class TestCyclingOptimizerPipelineResolution:
    """Tests for pipeline ↔ conditioning_fn mutex and generator-category validation."""

    def test_cannot_specify_both_pipeline_and_conditioning_fn(self):
        """Test that specifying both pipeline and conditioning_fn raises error."""
        target_segment = Segment(sequence="A" * 100, sequence_type="protein")
        construct = Construct([target_segment])
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(temperature=0.1))

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
            pipeline="protein-hunter",
        )

        with pytest.raises(ValueError, match="Specify exactly one"):
            CyclingOptimizer(
                target_segment=target_segment,
                constructs=[construct],
                generators=[generator],
                constraints=[],
                config=config,
                conditioning_fn=lambda x: x,  # Both pipeline and conditioning_fn
            )

    def test_must_specify_pipeline_or_conditioning_fn(self):
        """Test that neither pipeline nor conditioning_fn raises error."""
        target_segment = Segment(sequence="A" * 100, sequence_type="protein")
        construct = Construct([target_segment])
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(temperature=0.1))

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
            # No pipeline specified
        )

        with pytest.raises(ValueError, match="Specify exactly one"):
            CyclingOptimizer(
                target_segment=target_segment,
                constructs=[construct],
                generators=[generator],
                constraints=[],
                config=config,
                # No conditioning_fn either
            )

    def test_protein_hunter_requires_inverse_folding_generator(self):
        """Test that protein-hunter pipeline requires inverse_folding generator."""
        from proto_language.generator import (
            ESM2Generator,
            ESM2GeneratorConfig,
        )

        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])
        generator = ESM2Generator(ESM2GeneratorConfig())

        config = CyclingOptimizerConfig(num_steps=2, num_results=2, pipeline="protein-hunter")

        with pytest.raises(ValueError, match=r"requires a generator with input_type=.*STRUCTURE"):
            CyclingOptimizer(
                target_segment=target_segment,
                constructs=[construct],
                generators=[generator],
                constraints=[],
                config=config,
            )

    def test_protein_hunter_accepts_proteinmpnn(self):
        """Test that protein-hunter pipeline accepts ProteinMPNN generator."""
        target_segment = Segment(sequence="A" * 100, sequence_type="protein")
        construct = Construct([target_segment])
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(temperature=0.1))
        generator.assign(target_segment)

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
            pipeline="protein-hunter",
        )

        # Should not raise
        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[],
            config=config,
        )
        assert optimizer.pipeline == "protein-hunter"

    def test_protein_hunter_accepts_ligandmpnn(self):
        """Test that protein-hunter pipeline accepts LigandMPNN generator."""
        target_segment = Segment(sequence="A" * 100, sequence_type="protein")
        construct = Construct([target_segment])
        generator = LigandMPNNGenerator(LigandMPNNGeneratorConfig(temperature=0.1))
        generator.assign(target_segment)

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_results=2,
            pipeline="protein-hunter",
        )

        # Should not raise
        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[],
            config=config,
        )
        assert optimizer.pipeline == "protein-hunter"

    def test_protein_hunter_passes_cycle_seed_for_stochastic_tool(self, monkeypatch):
        """conditioning_fn must provide seeded runs to a `stochastic=True` @tool.

        Per-item seed derivation is the tool's responsibility — the proto-tools
        framework does not unroll multi-item dispatches. The cycling code is
        responsible only for sending one deterministic seed per cycle when the
        optimizer is seeded.
        """
        from proto_language.optimizer import cycling_optimizer as co

        captured: list[dict[str, Any]] = []

        class _FakeStructure:
            pass

        class _FakeOutput:
            def __init__(self, n: int):
                self.structures = [_FakeStructure() for _ in range(n)]

        def _fake_predict_structures(complexes, toolkit, tool_config):
            captured.append(dict(tool_config))
            return _FakeOutput(len(complexes))

        import proto_tools

        monkeypatch.setattr(proto_tools, "predict_structures", _fake_predict_structures)

        cfg = CyclingOptimizerConfig(num_steps=1, num_results=3, pipeline="protein-hunter", seed=42)
        fn = co._create_protein_hunter_conditioning_fn(cfg)

        sequences = [Sequence(sequence="A" * 10, sequence_type="protein") for _ in range(3)]
        fn(sequences)

        # One batched predict_structures call per cycle (lets ToolPool fan out across GPUs).
        assert len(captured) == 1
        tool_config = captured[0]
        assert tool_config["seed"] is not None
        assert tool_config["use_msa"] is False
        assert "seed_per_item" not in tool_config

    def test_protein_hunter_conditioning_fn_is_seed_resettable(self, monkeypatch):
        """Re-running a seeded cycling optimizer must replay the same conditioning seeds.

        Base ``Optimizer._reset_seed_state`` reseeds the optimizer/generators/constraints
        but doesn't know about pipeline-built conditioning_fns. The cycling-optimizer
        override calls into the closure's ``_reset_seed_state`` hook so the second run's
        per-cycle Boltz2 (etc.) seed matches the first.
        """
        from proto_language.optimizer import cycling_optimizer as co

        captured_seeds: list[int | None] = []

        class _FakeStructure:
            pass

        class _FakeOutput:
            def __init__(self, n: int):
                self.structures = [_FakeStructure() for _ in range(n)]

        def _fake_predict_structures(complexes, toolkit, tool_config):
            captured_seeds.append(tool_config.get("seed"))
            return _FakeOutput(len(complexes))

        import proto_tools

        monkeypatch.setattr(proto_tools, "predict_structures", _fake_predict_structures)

        cfg = CyclingOptimizerConfig(num_steps=1, num_results=2, pipeline="protein-hunter", seed=42)
        fn = co._create_protein_hunter_conditioning_fn(cfg)
        sequences = [Sequence(sequence="A" * 10, sequence_type="protein") for _ in range(2)]

        fn(sequences)
        first_seed = captured_seeds[-1]

        # Without a reset, the second call advances the RNG — different seed.
        fn(sequences)
        continued_seed = captured_seeds[-1]
        assert continued_seed != first_seed

        # After resetting, the stream replays — third call matches the first.
        fn._reset_seed_state()
        fn(sequences)
        replayed_seed = captured_seeds[-1]
        assert replayed_seed == first_seed


class TestCyclingProposalTracking:
    """Test proposal_results tracking in Cycling history."""

    def test_proposal_tracking(self):
        """History has proposal_results, all accepted when no filter rejects."""
        components = _setup_cycling_components(
            num_steps=3, num_results=2, include_constraint=True, constraint_passes=True
        )
        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=components["constraints"],
            config=components["config"],
            conditioning_fn=components["conditioning_fn"],
        )
        optimizer.track_proposals = True

        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].proposal_sequences:
                c.sequence = "MKTAYIAKQRQISFVKSHFS"

        components["generator"].sample = mock_sample

        optimizer.run()

        for entry in optimizer.history:
            if "proposal_results" not in entry:
                continue
            for cand in entry["proposal_results"]:
                assert isinstance(cand["accepted"], bool)
                if cand["accepted"]:
                    assert cand["rejected_by"] is None
                else:
                    assert cand["rejected_by"] is not None

        # At least one step should have proposal_results
        assert any("proposal_results" in e for e in optimizer.history)


class TestCyclingTrackingInterval:
    """Test tracking_interval in CyclingOptimizer."""

    def test_tracking_interval(self):
        """tracking_interval=3 reduces history snapshots."""
        components = _setup_cycling_components(num_steps=10, num_results=2)

        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].proposal_sequences:
                c.sequence = "MKTAYIAKQRQISFVKSHFS"

        components["generator"].sample = mock_sample

        # Override config to set tracking_interval=3
        config = CyclingOptimizerConfig(
            num_steps=10,
            num_results=2,
            tracking_interval=3,
        )

        optimizer = CyclingOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=config,
            conditioning_fn=components["conditioning_fn"],
        )

        optimizer.run()

        saved_steps = {entry["time_step"] for entry in optimizer.history}
        # 0 = initial state (always saved), then every tracking_interval=3 + final.
        assert saved_steps == {0, 3, 6, 9, 10}
