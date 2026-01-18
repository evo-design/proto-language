"""Tests for CyclingOptimizer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel, ValidationError

from proto_language.language.core import Constraint, Construct, Segment, Sequence
from proto_language.language.generator import (
    LigandMPNNGenerator,
    LigandMPNNGeneratorConfig,
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)
from proto_language.language.optimizer import (
    CyclingOptimizer,
    CyclingOptimizerConfig,
)
from proto_language.tools.inverse_folding.schemas import InverseFoldingStructureInput
from proto_language.tools.structures import ProteinStructure

# =============================================================================
# Helpers
# =============================================================================


class EmptyConfig(BaseModel):
    pass


def make_mock_structure() -> ProteinStructure:
    """Create a minimal mock ProteinStructure."""
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
END
"""
    return ProteinStructure(structure_filepath_or_content=pdb_content)


def make_mock_conditioning_fn(num_candidates: int):
    """Create a mock conditioning function that returns structures."""
    structures = [make_mock_structure() for _ in range(num_candidates)]

    def conditioning_fn(sequences: List[Sequence]) -> List[ProteinStructure]:
        return structures

    return conditioning_fn, structures


def _setup_cycling_components(
    num_steps: int = 2,
    num_candidates: int = 2,
    include_constraint: bool = False,
    constraint_passes: bool = True,
):
    """Helper to set up CyclingOptimizer with mocked components."""
    target_segment = Segment(sequence="ACDEFGHIKLMNPQRSTVWY", sequence_type="protein")
    construct = Construct([target_segment])

    mock_structure = make_mock_structure()
    generator = ProteinMPNNGenerator(
        ProteinMPNNGeneratorConfig(
            structure_inputs=InverseFoldingStructureInput(structure=mock_structure)
        )
    )
    generator.assign(target_segment)

    constraints = []
    if include_constraint:

        def filter_func(seq, config=None):
            return 0.0 if constraint_passes else 1.0

        filter_func._constraint_batched = False
        filter_func._constraint_concatenate = True
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
        num_candidates=num_candidates,
        conditioning_param_name="structure_inputs",
    )

    conditioning_fn, _ = make_mock_conditioning_fn(num_candidates)

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
        config = CyclingOptimizerConfig(
            num_steps=5,
            num_candidates=4,
            conditioning_param_name="structure_inputs",
        )
        assert config.num_steps == 5
        assert config.num_candidates == 4
        assert config.conditioning_param_name == "structure_inputs"
        assert config.verbose is False

    def test_invalid_config_values(self):
        """Test that invalid config values are rejected."""
        with pytest.raises(ValidationError):
            CyclingOptimizerConfig(
                num_steps=0, num_candidates=1, conditioning_param_name="structure_inputs"
            )
        with pytest.raises(ValidationError):
            CyclingOptimizerConfig(
                num_steps=1, num_candidates=0, conditioning_param_name="structure_inputs"
            )

    def test_conditioning_field_required(self):
        """Test that conditioning_field is required."""
        with pytest.raises(ValidationError):
            CyclingOptimizerConfig(num_steps=1, num_candidates=1)


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

        with pytest.raises(
            ValueError, match="is not in any of the provided constructs"
        ):
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

        def scoring_func(seq, config=None):
            return 0.5

        scoring_func._constraint_batched = False
        scoring_func._constraint_concatenate = True
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


# =============================================================================
# Run Method Tests
# =============================================================================


class TestCyclingOptimizerRun:
    """Tests for the CyclingOptimizer run method."""

    def test_run_completes_and_tracks_history(self):
        """Test run completes, calls conditioning_fn, and tracks history."""
        num_steps, num_candidates = 3, 2
        components = _setup_cycling_components(
            num_steps=num_steps, num_candidates=num_candidates
        )

        # Track conditioning function calls
        call_count = [0]
        original_fn = components["conditioning_fn"]

        def tracked_conditioning_fn(sequences):
            call_count[0] += 1
            return original_fn(sequences)

        # Mock the generator.sample to update sequences
        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].candidate_sequences:
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
        assert len(optimizer.history) == num_steps + 1
        for entry in optimizer.history:
            assert "time_step" in entry and "constructs" in entry

    def test_filter_constraint_rollback(self):
        """Test that failing candidates are rolled back to previous sequences."""
        components = _setup_cycling_components(
            num_steps=1,
            num_candidates=2,
            include_constraint=True,
            constraint_passes=False,
        )

        def mock_sample(structure_inputs=None):
            for c in components["target_segment"].candidate_sequences:
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

        original_seqs = [
            copy.deepcopy(s.sequence)
            for s in components["target_segment"].candidate_sequences
        ]
        optimizer.run()

        # All should be rolled back since constraint fails all
        for i, candidate in enumerate(
            components["target_segment"].candidate_sequences
        ):
            assert candidate.sequence == original_seqs[i]

    def test_partial_filter_rejection(self):
        """Test that only failing candidates are rolled back."""
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])

        generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(
                    structure=make_mock_structure()
                )
            )
        )
        generator.assign(target_segment)

        def partial_filter(seq, config=None):
            return 1.0 if "FAIL" in seq.sequence else 0.0

        partial_filter._constraint_batched = False
        partial_filter._constraint_concatenate = True
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
            target_segment.candidate_sequences[0].sequence = pass_seq
            target_segment.candidate_sequences[1].sequence = fail_seq
            target_segment.candidate_sequences[2].sequence = fail_seq

        generator.sample = mock_sample

        config = CyclingOptimizerConfig(
            num_steps=1,
            num_candidates=3,
            conditioning_param_name="structure_inputs",
        )

        optimizer = CyclingOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=config,
            conditioning_fn=conditioning_fn,
        )

        original_seqs = [
            copy.deepcopy(s.sequence) for s in target_segment.candidate_sequences
        ]
        optimizer.run()

        assert target_segment.candidate_sequences[0].sequence == pass_seq
        assert target_segment.candidate_sequences[1].sequence == original_seqs[1]
        assert target_segment.candidate_sequences[2].sequence == original_seqs[2]


# =============================================================================
# GPU Integration Tests
# =============================================================================


TEST_PDB_FILE = Path(__file__).parent.parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.fixture(scope="module")
def pdb_structure():
    """Load test PDB structure."""
    return ProteinStructure(structure_filepath_or_content=TEST_PDB_FILE)


@pytest.mark.uses_gpu
class TestCyclingOptimizerGPU:
    """Integration tests with real models (require GPU)."""

    @pytest.mark.slow
    def test_full_cycle_with_ligandmpnn(self, pdb_structure):
        """Test complete optimization cycle with LigandMPNN."""
        from proto_language.tools.structure_prediction.schemas import (
            StructurePredictionComplex,
        )
        from proto_language.utils.helpers import predict_structures

        chain_seq = pdb_structure.get_chain_sequence("A")
        seq_length = len(chain_seq)

        # Inverse folding generators auto-initialize to "X" when no sequence provided
        target_segment = Segment(length=seq_length, sequence_type="protein")
        construct = Construct([target_segment])

        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(
                    structure=pdb_structure, chain_ids=["A"]
                ),
                temperature=0.1,
            )
        )
        generator.assign(target_segment)

        def structure_conditioning_fn(sequences):
            complexes = [
                StructurePredictionComplex(chains=[seq.sequence])
                for seq in sequences
            ]
            return predict_structures(complexes, "boltz", {}).structures

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_candidates=2,
            conditioning_param_name="structure_inputs",
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

        assert len(target_segment.selected_sequences) == 2
        for seq in target_segment.selected_sequences:
            assert len(seq.sequence) == seq_length
            assert seq.sequence != "X" * seq_length

    @pytest.mark.slow
    def test_with_filter_constraint(self, pdb_structure):
        """Test with filter constraint using real models."""
        from proto_language.tools.structure_prediction.schemas import (
            StructurePredictionComplex,
        )
        from proto_language.utils.helpers import predict_structures

        chain_seq = pdb_structure.get_chain_sequence("A")
        seq_length = len(chain_seq)

        # Inverse folding generators auto-initialize to "X" when no sequence provided
        target_segment = Segment(length=seq_length, sequence_type="protein")
        construct = Construct([target_segment])

        generator = LigandMPNNGenerator(
            LigandMPNNGeneratorConfig(
                structure_inputs=InverseFoldingStructureInput(
                    structure=pdb_structure, chain_ids=["A"]
                ),
                temperature=0.1,
            )
        )
        generator.assign(target_segment)

        def length_filter(seq, config=None):
            return 0.0 if len(seq.sequence) > 10 else 1.0

        length_filter._constraint_batched = False
        length_filter._constraint_concatenate = True
        length_filter._constraint_config_class = EmptyConfig
        length_filter._constraint_supported_sequence_types = ["protein"]

        constraint = Constraint(
            inputs=[target_segment],
            function=length_filter,
            function_config=EmptyConfig(),
            threshold=0.5,
        )

        def structure_conditioning_fn(sequences):
            complexes = [
                StructurePredictionComplex(chains=[seq.sequence])
                for seq in sequences
            ]
            return predict_structures(complexes, "boltz", {}).structures

        config = CyclingOptimizerConfig(
            num_steps=2,
            num_candidates=2,
            conditioning_param_name="structure_inputs",
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

        assert len(target_segment.selected_sequences) == 2
        for seq in target_segment.selected_sequences:
            assert len(seq.sequence) > 10


# =============================================================================
# State Restart Tests
# =============================================================================


class TestCyclingOptimizerRestart:
    """Tests for CyclingOptimizer state restart behavior."""

    def test_run_restarts_from_initial_state(self):
        """Test that calling run() twice restarts from initial state."""
        components = _setup_cycling_components(num_steps=2, num_candidates=2)

        # Mock the generator.sample to track calls and modify sequences
        call_count = [0]
        # Valid 20-char protein sequences for each call
        protein_seqs = ["MKTAYIAKQRQISFVKSHFS", "GPLAFVTNLTGLRSQNEEIR", 
                        "YWDEIKNPLGRAVTYDKWFP", "HCLQMNSGVEATRIDFWYKP"]
        def mock_sample(structure_inputs=None):
            call_count[0] += 1
            for c in components["target_segment"].candidate_sequences:
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
        first_run_seqs = [s.sequence for s in components["target_segment"].selected_sequences]

        # Second run should restart - call count continues but state is fresh
        optimizer.run()
        assert call_count[0] > first_run_calls
        second_run_seqs = [s.sequence for s in components["target_segment"].selected_sequences]
        
        # Verify sequences were modified from original (mock changes them to valid protein seqs)
        original_seq = "ACDEFGHIKLMNPQRSTVWY"
        assert all(seq != original_seq for seq in first_run_seqs)
        assert all(seq != original_seq for seq in second_run_seqs)
        # History should be from second run only (cleared on restart)
        assert len(optimizer.history) == 3  # step 0, 1, 2

    def test_initial_state_captured_correctly(self):
        """Test that initial state captures segment state with actual sequence content."""
        components = _setup_cycling_components(num_steps=1, num_candidates=2)

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
        original_selected = [copy.deepcopy(s) for s in components["target_segment"].selected_sequences]
        original_candidates = [copy.deepcopy(s) for s in components["target_segment"].candidate_sequences]

        optimizer.run()

        # Verify state was captured
        assert optimizer._initial_state is not None
        assert len(optimizer._initial_state['segments']) == 1
        
        # Verify captured state contains actual sequence content (using index 0)
        captured_selected = optimizer._initial_state['segments'][0]['selected']
        captured_candidates = optimizer._initial_state['segments'][0]['candidates']
        
        assert len(captured_selected) == len(original_selected)
        assert len(captured_candidates) == len(original_candidates)
        
        # Verify sequences match
        for orig, captured in zip(original_selected, captured_selected):
            assert orig.sequence == captured.sequence
            assert orig.sequence_type == captured.sequence_type
            
        for orig, captured in zip(original_candidates, captured_candidates):
            assert orig.sequence == captured.sequence
            assert orig.sequence_type == captured.sequence_type


# =============================================================================
# Pipeline Resolution Tests
# =============================================================================


class TestCyclingOptimizerPipelineResolution:
    """Tests for _resolve_conditioning_fn helper and pipeline validation."""

    def test_cannot_specify_both_pipeline_and_conditioning_fn(self):
        """Test that specifying both pipeline and conditioning_fn raises error."""
        target_segment = Segment(sequence="A" * 100, sequence_type="protein")
        construct = Construct([target_segment])
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(temperature=0.1))
        
        config = CyclingOptimizerConfig(
            num_steps=2,
            num_candidates=2,
            conditioning_param_name="structure_inputs",
            pipeline="protein-hunter",
        )
        
        with pytest.raises(ValueError, match="Cannot specify both"):
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
            num_candidates=2,
            conditioning_param_name="structure_inputs",
            # No pipeline specified
        )
        
        with pytest.raises(ValueError, match="Must specify either"):
            CyclingOptimizer(
                target_segment=target_segment,
                constructs=[construct],
                generators=[generator],
                constraints=[],
                config=config,
                # No conditioning_fn either
            )

    def test_unknown_pipeline_raises_error(self):
        """Test that unknown pipeline name raises error."""
        target_segment = Segment(sequence="A" * 100, sequence_type="protein")
        construct = Construct([target_segment])
        generator = ProteinMPNNGenerator(ProteinMPNNGeneratorConfig(temperature=0.1))
        
        # Manually set invalid pipeline (bypassing Literal validation)
        config = CyclingOptimizerConfig(
            num_steps=2,
            num_candidates=2,
            conditioning_param_name="structure_inputs",
        )
        object.__setattr__(config, 'pipeline', 'nonexistent-pipeline')
        
        with pytest.raises(ValueError, match="Unknown pipeline"):
            CyclingOptimizer(
                target_segment=target_segment,
                constructs=[construct],
                generators=[generator],
                constraints=[],
                config=config,
            )

    def test_protein_hunter_requires_inverse_folding_generator(self):
        """Test that protein-hunter pipeline requires inverse_folding generator."""
        from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig
        
        target_segment = Segment(sequence="A" * 20, sequence_type="protein")
        construct = Construct([target_segment])
        generator = ESM2Generator(ESM2GeneratorConfig(mask_positions=[[0]]))
        
        config = CyclingOptimizerConfig(
            num_steps=2,
            num_candidates=2,
            conditioning_param_name="prompts",
            pipeline="protein-hunter",
        )
        
        with pytest.raises(ValueError, match="requires inverse_folding generator"):
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
        
        config = CyclingOptimizerConfig(
            num_steps=2,
            num_candidates=2,
            conditioning_param_name="structure_inputs",
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
        
        config = CyclingOptimizerConfig(
            num_steps=2,
            num_candidates=2,
            conditioning_param_name="structure_inputs",
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