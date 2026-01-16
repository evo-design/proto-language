"""Tests for CyclicalOptimizer."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from proto_language.language.core import Constraint, Construct, Segment
from proto_language.language.generator import (
    LigandMPNNGenerator,
    LigandMPNNGeneratorConfig,
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    CyclicalOptimizer,
    CyclicalOptimizerConfig,
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


def make_mock_predict_structures_return(num_candidates: int):
    """Create a mock return value for predict_structures."""
    mock_result = MagicMock()
    mock_result.structures = [make_mock_structure() for _ in range(num_candidates)]
    return mock_result


def _setup_cyclical_components(
    num_cycles: int = 2,
    num_candidates: int = 2,
    include_constraint: bool = False,
    constraint_passes: bool = True,
    include_context_segment: bool = False,
):
    """Helper to set up CyclicalOptimizer with mocked components."""
    target_segment = Segment(sequence="ACDEFGHIKLMNPQRSTVWY", sequence_type="protein")
    segments = [target_segment]

    context_segment = None
    if include_context_segment:
        context_segment = Segment(
            sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            sequence_type="protein",
        )
        segments.append(context_segment)

    construct = Construct(segments)

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

    config = CyclicalOptimizerConfig(
        num_cycles=num_cycles,
        num_candidates=num_candidates,
        structure_tool="boltz",
        tool_config={},
    )

    return {
        "target_segment": target_segment,
        "context_segment": context_segment,
        "construct": construct,
        "generator": generator,
        "constraints": constraints,
        "config": config,
    }


# =============================================================================
# Config Tests
# =============================================================================


class TestCyclicalOptimizerConfig:
    """Tests for CyclicalOptimizerConfig validation."""

    def test_valid_config(self):
        """Test valid configuration and defaults."""
        config = CyclicalOptimizerConfig(
            num_cycles=5,
            num_candidates=4,
            structure_tool="boltz",
            tool_config={"use_msa_server": False},
        )
        assert config.num_cycles == 5
        assert config.num_candidates == 4
        assert config.structure_tool == "boltz"
        assert config.verbose is False

    def test_invalid_config_values(self):
        """Test that invalid config values are rejected."""
        with pytest.raises(ValidationError):
            CyclicalOptimizerConfig(num_cycles=0, num_candidates=1)
        with pytest.raises(ValidationError):
            CyclicalOptimizerConfig(num_cycles=1, num_candidates=0)
        with pytest.raises(ValidationError):
            CyclicalOptimizerConfig(
                num_cycles=1, num_candidates=1, structure_tool="invalid"
            )


# =============================================================================
# Validation Tests
# =============================================================================


class TestCyclicalOptimizerValidation:
    """Tests for CyclicalOptimizer initialization validation."""

    def test_requires_exactly_one_generator(self):
        """Test that exactly one inverse_folding generator is required."""
        components = _setup_cyclical_components()

        with pytest.raises(ValueError, match="requires one inverse_folding generator"):
            CyclicalOptimizer(
                target_segment=components["target_segment"],
                constructs=[components["construct"]],
                generators=[],
                constraints=[],
                config=components["config"],
            )

    def test_target_segment_must_be_in_constructs(self):
        """Test that target_segment must belong to a construct."""
        components = _setup_cyclical_components()
        orphan_segment = Segment(sequence="ACDEFGHIK", sequence_type="protein")

        with pytest.raises(
            ValueError, match="is not in any of the provided constructs"
        ):
            CyclicalOptimizer(
                target_segment=orphan_segment,
                constructs=[components["construct"]],
                generators=[components["generator"]],
                constraints=[],
                config=components["config"],
            )

    def test_generator_must_be_inverse_folding(self):
        """Test that only inverse_folding generators are accepted."""
        target = Segment(sequence="ACDEFGHIK", sequence_type="protein")
        construct = Construct([target])

        wrong_gen = UniformMutationGenerator(
            UniformMutationGeneratorConfig(num_mutations=1)
        )
        wrong_gen.assign(target)

        with pytest.raises(ValueError, match="requires an inverse_folding generator"):
            CyclicalOptimizer(
                target_segment=target,
                constructs=[construct],
                generators=[wrong_gen],
                constraints=[],
                config=CyclicalOptimizerConfig(num_cycles=1, num_candidates=1),
            )

    def test_constraints_must_be_filters(self):
        """Test that constraints must have threshold set (filter mode)."""
        components = _setup_cyclical_components()

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
            CyclicalOptimizer(
                target_segment=components["target_segment"],
                constructs=[components["construct"]],
                generators=[components["generator"]],
                constraints=[scoring_constraint],
                config=components["config"],
            )

    def test_esmfold_not_supported(self):
        """Test that ESMFold is rejected as structure_tool at config level."""
        with pytest.raises(ValidationError, match="ESMFold is not supported"):
            CyclicalOptimizerConfig(
                num_cycles=1,
                num_candidates=1,
                structure_tool="esmfold",
            )


# =============================================================================
# Run Method Tests
# =============================================================================


class TestCyclicalOptimizerRun:
    """Tests for the CyclicalOptimizer run method."""

    @patch("proto_language.language.optimizer.cyclical_optimizer.predict_structures")
    def test_run_completes_and_tracks_history(self, mock_predict):
        """Test run completes, calls predict_structures, and tracks history."""
        num_cycles, num_candidates = 3, 2
        components = _setup_cyclical_components(
            num_cycles=num_cycles, num_candidates=num_candidates
        )

        mock_predict.return_value = make_mock_predict_structures_return(num_candidates)
        components["generator"].sample = lambda structure_inputs=None: [
            setattr(c, "sequence", "MKTAYIAKQRQISFVKSHFS")
            for c in components["target_segment"].candidate_sequences
        ]

        optimizer = CyclicalOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=components["config"],
        )
        optimizer.run()

        assert mock_predict.call_count == num_cycles
        assert len(optimizer.history) == num_cycles + 1
        for entry in optimizer.history:
            assert "time_step" in entry and "constructs" in entry

    @patch("proto_language.language.optimizer.cyclical_optimizer.predict_structures")
    def test_run_with_context_segment(self, mock_predict):
        """Test run with multi-chain structural context."""
        components = _setup_cyclical_components(
            num_cycles=1, num_candidates=2, include_context_segment=True
        )

        mock_predict.return_value = make_mock_predict_structures_return(2)
        components["generator"].sample = lambda structure_inputs=None: [
            setattr(c, "sequence", "MKTAYIAKQRQISFVKSHFS")
            for c in components["target_segment"].candidate_sequences
        ]

        optimizer = CyclicalOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=[],
            config=components["config"],
        )
        optimizer.run()

        # Verify both chains passed to predict_structures
        complexes = mock_predict.call_args_list[0][0][0]
        assert len(complexes[0].chains) == 2

    @patch("proto_language.language.optimizer.cyclical_optimizer.predict_structures")
    def test_filter_constraint_rollback(self, mock_predict):
        """Test that failing candidates are rolled back to previous sequences."""
        components = _setup_cyclical_components(
            num_cycles=1,
            num_candidates=2,
            include_constraint=True,
            constraint_passes=False,
        )

        mock_predict.return_value = make_mock_predict_structures_return(2)
        components["generator"].sample = lambda structure_inputs=None: [
            setattr(c, "sequence", "NEWSEQENCEAAAAAAAAAAA")
            for c in components["target_segment"].candidate_sequences
        ]

        optimizer = CyclicalOptimizer(
            target_segment=components["target_segment"],
            constructs=[components["construct"]],
            generators=[components["generator"]],
            constraints=components["constraints"],
            config=components["config"],
        )

        original_seqs = [
            copy.deepcopy(s.sequence)
            for s in components["target_segment"].candidate_sequences
        ]
        optimizer.run()

        # All should be rolled back since constraint fails all
        for i, candidate in enumerate(components["target_segment"].candidate_sequences):
            assert candidate.sequence == original_seqs[i]

    @patch("proto_language.language.optimizer.cyclical_optimizer.predict_structures")
    def test_partial_filter_rejection(self, mock_predict):
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

        mock_predict.return_value = make_mock_predict_structures_return(3)

        pass_seq, fail_seq = "MKTAYIAKQRQISFVKSHFS", "FAILAYIAKQRQISFVKSHF"

        def mock_sample(structure_inputs=None):
            target_segment.candidate_sequences[0].sequence = pass_seq
            target_segment.candidate_sequences[1].sequence = fail_seq
            target_segment.candidate_sequences[2].sequence = fail_seq

        generator.sample = mock_sample

        optimizer = CyclicalOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=CyclicalOptimizerConfig(num_cycles=1, num_candidates=3),
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
class TestCyclicalOptimizerGPU:
    """Integration tests with real models (require GPU)."""

    @pytest.mark.slow
    def test_full_cycle_with_ligandmpnn(self, pdb_structure):
        """Test complete optimization cycle with LigandMPNN and ESMFold."""
        chain_seq = pdb_structure.get_chain_sequence("A")
        seq_length = len(chain_seq)

        target_segment = Segment(sequence="A" * seq_length, sequence_type="protein")
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

        optimizer = CyclicalOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[],
            config=CyclicalOptimizerConfig(
                num_cycles=2, num_candidates=2, structure_tool="boltz", verbose=True
            ),
        )
        optimizer.run()

        assert len(target_segment.selected_sequences) == 2
        for seq in target_segment.selected_sequences:
            assert len(seq.sequence) == seq_length
            assert seq.sequence != "A" * seq_length

    @pytest.mark.slow
    def test_with_filter_constraint(self, pdb_structure):
        """Test with filter constraint using real models."""
        chain_seq = pdb_structure.get_chain_sequence("A")
        seq_length = len(chain_seq)

        target_segment = Segment(sequence="A" * seq_length, sequence_type="protein")
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

        constraint = Constraint(
            inputs=[target_segment],
            function=length_filter,
            function_config=EmptyConfig(),
            threshold=0.5,
        )

        optimizer = CyclicalOptimizer(
            target_segment=target_segment,
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=CyclicalOptimizerConfig(
                num_cycles=2, num_candidates=2, structure_tool="boltz", verbose=True
            ),
        )
        optimizer.run()

        assert len(target_segment.selected_sequences) == 2
        for seq in target_segment.selected_sequences:
            assert len(seq.sequence) > 10
