"""
Tests for serialize_program - converting Program objects to Darwin GPL JSON.

Tests cover:
- Basic serialization of single optimizer programs
- Multi-optimizer program serialization
- Round-trip: JSON -> Program -> JSON
- Segment ID generation consistency
- Generator and constraint config extraction
- Edge cases (empty sequences, length-only segments, thresholds)
"""

import json

from api.core.parser import DarwinParser
from api.core.serializer import serialize_program
from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.core import Construct, Program, Segment
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
    TopKOptimizer,
    TopKOptimizerConfig,
)

class TestProgramSerializer:
    """Tests for serialize_program function."""

    def test_serialize_single_optimizer(self):
        """Test serialization of a simple single-optimizer program."""
        # Create a simple program
        segment = Segment(length=20, sequence_type="dna", label="test_segment")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=5)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 40, "max_gc": 60},
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])

        # Serialize
        result = serialize_program(program)

        # Verify structure
        assert "constructs" in result
        assert "optimization_stages" in result
        assert "verbose" in result

        # Verify construct
        assert len(result["constructs"]) == 1
        assert result["constructs"][0]["type"] == "DNA"
        assert len(result["constructs"][0]["segments"]) == 1

        # Verify segment
        seg = result["constructs"][0]["segments"][0]
        assert seg["id"] == "construct0-segment0"
        assert seg["label"] == "test_segment"
        # Segments created with length should serialize with length
        assert "length" in seg
        assert seg["length"] == 20

        # Verify optimization stage
        assert len(result["optimization_stages"]) == 1
        stage = result["optimization_stages"][0]
        assert stage["optimizer"]["method"] == "topk"
        assert stage["optimizer"]["config"]["num_samples"] == 10
        assert stage["optimizer"]["config"]["k"] == 3
        assert stage["optimizer"]["config"]["batch_size"] == 2

        # Verify generator
        assert len(stage["generators"]) == 1
        gen = stage["generators"][0]
        assert gen["key"] == "uniform-mutation"
        assert gen["target"] == "construct0-segment0"
        assert gen["config"]["num_mutations"] == 5

        # Verify constraint
        assert len(stage["constraints"]) == 1
        con = stage["constraints"][0]
        assert con["key"] == "gc-content"
        assert con["targets"] == ["construct0-segment0"]
        assert con["config"]["min_gc"] == 40
        assert con["config"]["max_gc"] == 60

    def test_serialize_multiple_optimizers(self):
        """Test serialization of a program with multiple sequential optimizers."""
        segment = Segment(length=20, sequence_type="dna", label="sequence1")
        construct = Construct([segment])

        # First optimizer: TopK
        gen1_config = UniformMutationGeneratorConfig(num_mutations=10)
        gen1 = UniformMutationGenerator(gen1_config)
        gen1.assign(segment)

        con1 = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 50, "max_gc": 100},
        )

        opt1_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        opt1 = TopKOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[con1],
            config=opt1_config,
        )

        # Second optimizer: MCMC
        gen2_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen2 = UniformMutationGenerator(gen2_config)
        gen2.assign(segment)

        con2 = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 80, "max_gc": 90},
        )

        opt2_config = MCMCOptimizerConfig(num_selected=1, mcmc_width=20, num_steps=10)
        opt2 = MCMCOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[con2],
            config=opt2_config,
        )

        program = Program(optimizers=[opt1, opt2])

        # Serialize
        result = serialize_program(program)

        # Verify two optimization stages
        assert len(result["optimization_stages"]) == 2

        # Verify first stage
        stage1 = result["optimization_stages"][0]
        assert stage1["optimizer"]["method"] == "topk"
        assert stage1["generators"][0]["config"]["num_mutations"] == 10
        assert stage1["constraints"][0]["config"]["min_gc"] == 50

        # Verify second stage
        stage2 = result["optimization_stages"][1]
        assert stage2["optimizer"]["method"] == "mcmc"
        assert stage2["generators"][0]["config"]["num_mutations"] == 1
        assert stage2["constraints"][0]["config"]["min_gc"] == 80

    def test_round_trip_serialization(self):
        """Test that JSON -> Program -> JSON produces equivalent output."""
        original_json = {
            "constructs": [
                {
                    "type": "DNA",
                    "segments": [{"id": "seg1", "label": "segment1", "length": 20}],
                }
            ],
            "optimization_stages": [
                {
                    "optimizer": {
                        "method": "topk",
                        "config": {"num_samples": 10, "k": 3, "batch_size": 2},
                    },
                    "generators": [
                        {
                            "key": "uniform-mutation",
                            "target": "seg1",
                            "config": {"num_mutations": 5},
                        }
                    ],
                    "constraints": [
                        {
                            "key": "gc-content",
                            "targets": ["seg1"],
                            "config": {"min_gc": 40, "max_gc": 60},
                        }
                    ],
                }
            ],
            "verbose": False,
        }

        # Parse to Program
        parser = DarwinParser(original_json)
        program = parser.parse()

        # Serialize back to JSON
        result_json = serialize_program(program)

        # Verify key fields match
        assert result_json["verbose"] == original_json["verbose"]
        assert len(result_json["constructs"]) == len(original_json["constructs"])
        assert len(result_json["optimization_stages"]) == len(
            original_json["optimization_stages"]
        )

        # Verify construct type
        assert result_json["constructs"][0]["type"] == "DNA"

        # Verify optimizer config
        orig_opt = original_json["optimization_stages"][0]["optimizer"]
        result_opt = result_json["optimization_stages"][0]["optimizer"]
        assert result_opt["method"] == orig_opt["method"]
        assert result_opt["config"]["num_samples"] == orig_opt["config"]["num_samples"]
        assert result_opt["config"]["k"] == orig_opt["config"]["k"]
        assert result_opt["config"]["batch_size"] == orig_opt["config"]["batch_size"]

        # Verify generator config
        orig_gen = original_json["optimization_stages"][0]["generators"][0]
        result_gen = result_json["optimization_stages"][0]["generators"][0]
        assert result_gen["key"] == orig_gen["key"]
        assert (
            result_gen["config"]["num_mutations"] == orig_gen["config"]["num_mutations"]
        )

        # Verify constraint config
        orig_con = original_json["optimization_stages"][0]["constraints"][0]
        result_con = result_json["optimization_stages"][0]["constraints"][0]
        assert result_con["key"] == orig_con["key"]
        assert result_con["config"]["min_gc"] == orig_con["config"]["min_gc"]
        assert result_con["config"]["max_gc"] == orig_con["config"]["max_gc"]

    def test_multiple_segments_with_ids(self):
        """Test segment ID generation with multiple constructs and segments."""
        seg1 = Segment(length=10, sequence_type="dna", label="promoter")
        seg2 = Segment(length=20, sequence_type="dna", label="coding")
        construct1 = Construct([seg1, seg2])

        seg3 = Segment(length=15, sequence_type="dna", label="terminator")
        construct2 = Construct([seg3])

        # Generators for each segment that needs optimization
        gen1_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen1 = UniformMutationGenerator(gen1_config)
        gen1.assign(seg1)

        gen2_config = UniformMutationGeneratorConfig(num_mutations=2)
        gen2 = UniformMutationGenerator(gen2_config)
        gen2.assign(seg2)

        gen3_config = UniformMutationGeneratorConfig(num_mutations=3)
        gen3 = UniformMutationGenerator(gen3_config)
        gen3.assign(seg3)

        constraint = ConstraintRegistry.create(
            key="gc-content", segments=[seg1], config_dict={"min_gc": 40, "max_gc": 60}
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct1, construct2],
            generators=[gen1, gen2, gen3],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])
        result = serialize_program(program)

        # Verify segment IDs
        assert result["constructs"][0]["segments"][0]["id"] == "construct0-segment0"
        assert result["constructs"][0]["segments"][1]["id"] == "construct0-segment1"
        assert result["constructs"][1]["segments"][0]["id"] == "construct1-segment0"

        # Verify generator targets
        generators = result["optimization_stages"][0]["generators"]
        assert generators[0]["target"] == "construct0-segment0"
        assert generators[1]["target"] == "construct0-segment1"
        assert generators[2]["target"] == "construct1-segment0"

    def test_segment_with_sequence(self):
        """Test serialization of segment with sequence instead of length."""
        segment = Segment(
            sequence="ATGCATGCATGC", sequence_type="dna", label="with_seq"
        )
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 40, "max_gc": 60},
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])
        result = serialize_program(program)

        seg = result["constructs"][0]["segments"][0]
        assert "sequence" in seg
        assert seg["sequence"] == "ATGCATGCATGC"
        assert "length" not in seg

    def test_segment_with_sequence_serialization(self):
        """Test that segments with sequences are correctly serialized."""
        context_segment = Segment(sequence="ATGC", sequence_type="dna", label="context")
        var_segment = Segment(length=20, sequence_type="dna", label="variable")
        construct = Construct([context_segment, var_segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(var_segment)  # Only assign to variable segment

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[var_segment],
            config_dict={"min_gc": 40, "max_gc": 60},
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])
        result = serialize_program(program)

        segments = result["constructs"][0]["segments"]
        # Context segment has a sequence
        assert segments[0]["sequence"] == "ATGC"
        # Variable segment has length but no sequence
        assert "length" in segments[1]

    def test_multi_segment_constraint(self):
        """Test constraint targeting multiple segments."""
        seg1 = Segment(length=10, sequence_type="dna", label="seg1")
        seg2 = Segment(length=20, sequence_type="dna", label="seg2")
        construct = Construct([seg1, seg2])

        gen1_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen1 = UniformMutationGenerator(gen1_config)
        gen1.assign(seg1)

        gen2_config = UniformMutationGeneratorConfig(num_mutations=1)
        gen2 = UniformMutationGenerator(gen2_config)
        gen2.assign(seg2)

        # Constraint targeting both segments (concatenate=True by default)
        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[seg1, seg2],
            config_dict={"min_gc": 40, "max_gc": 60},
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[gen1, gen2],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])
        result = serialize_program(program)

        con = result["optimization_stages"][0]["constraints"][0]
        assert len(con["targets"]) == 2
        assert "construct0-segment0" in con["targets"]
        assert "construct0-segment1" in con["targets"]

    def test_constraint_with_custom_label(self):
        """Test that custom constraint labels are preserved."""
        segment = Segment(length=20, sequence_type="dna", label="test")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 40, "max_gc": 60},
            label="my_custom_gc_constraint",
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])
        result = serialize_program(program)

        con = result["optimization_stages"][0]["constraints"][0]
        assert con["label"] == "my_custom_gc_constraint"

    def test_serialize_program_function(self):
        """Test that serialize_program function works correctly."""
        segment = Segment(length=20, sequence_type="dna", label="test")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 40, "max_gc": 60},
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])

        # Call serialize_program function
        result = serialize_program(program)

        # Verify it returns valid JSON-compatible dict
        assert isinstance(result, dict)
        json_str = json.dumps(result)  # Should not raise
        assert len(json_str) > 0

    def test_round_trip_multiple_optimizers(self):
        """Test round-trip with multiple optimizers."""
        original_json = {
            "constructs": [
                {
                    "type": "DNA",
                    "segments": [{"id": "seg1", "label": "sequence1", "length": 20}],
                }
            ],
            "optimization_stages": [
                {
                    "optimizer": {
                        "method": "topk",
                        "config": {"num_samples": 10, "k": 3, "batch_size": 2},
                    },
                    "generators": [
                        {
                            "key": "uniform-mutation",
                            "target": "seg1",
                            "config": {"num_mutations": 10},
                        }
                    ],
                    "constraints": [
                        {
                            "key": "gc-content",
                            "targets": ["seg1"],
                            "config": {"min_gc": 50, "max_gc": 100},
                        }
                    ],
                },
                {
                    "optimizer": {
                        "method": "mcmc",
                        "config": {
                            "num_selected": 1,
                            "mcmc_width": 20,
                            "num_steps": 10,
                        },
                    },
                    "generators": [
                        {
                            "key": "uniform-mutation",
                            "target": "seg1",
                            "config": {"num_mutations": 1},
                        }
                    ],
                    "constraints": [
                        {
                            "key": "gc-content",
                            "targets": ["seg1"],
                            "config": {"min_gc": 80, "max_gc": 90},
                        }
                    ],
                },
            ],
            "verbose": False,
        }

        # Parse to Program
        parser = DarwinParser(original_json)
        program = parser.parse()

        # Serialize back to JSON
        result_json = serialize_program(program)

        # Verify two optimization stages
        assert len(result_json["optimization_stages"]) == 2

        # Verify first optimizer
        stage1 = result_json["optimization_stages"][0]
        assert stage1["optimizer"]["method"] == "topk"
        assert stage1["optimizer"]["config"]["num_samples"] == 10
        assert stage1["generators"][0]["config"]["num_mutations"] == 10
        assert stage1["constraints"][0]["config"]["min_gc"] == 50

        # Verify second optimizer
        stage2 = result_json["optimization_stages"][1]
        assert stage2["optimizer"]["method"] == "mcmc"
        assert stage2["optimizer"]["config"]["num_selected"] == 1
        assert stage2["optimizer"]["config"]["mcmc_width"] == 20
        assert stage2["generators"][0]["config"]["num_mutations"] == 1
        assert stage2["constraints"][0]["config"]["min_gc"] == 80

    def test_verbose_flag_preserved(self):
        """Test that verbose flag is correctly serialized."""
        segment = Segment(length=20, sequence_type="dna", label="test")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=1)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment],
            config_dict={"min_gc": 40, "max_gc": 60},
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )

        # Test with verbose=True
        program = Program(optimizers=[optimizer], verbose=True)
        result = serialize_program(program)
        assert result["verbose"] == True

        # Test with verbose=False (need to create new objects since Program modifies optimizer.verbose)
        segment2 = Segment(length=20, sequence_type="dna", label="test2")
        construct2 = Construct([segment2])

        gen_config2 = UniformMutationGeneratorConfig(num_mutations=1)
        generator2 = UniformMutationGenerator(gen_config2)
        generator2.assign(segment2)

        constraint2 = ConstraintRegistry.create(
            key="gc-content",
            segments=[segment2],
            config_dict={"min_gc": 40, "max_gc": 60},
        )

        optimizer2 = TopKOptimizer(
            constructs=[construct2],
            generators=[generator2],
            constraints=[constraint2],
            config=opt_config,
        )

        program2 = Program(optimizers=[optimizer2], verbose=False)
        result2 = serialize_program(program2)
        assert result2["verbose"] == False

    def test_protein_sequence_type(self):
        """Test serialization with protein sequence type."""
        segment = Segment(length=50, sequence_type="protein", label="protein_seq")
        construct = Construct([segment])

        gen_config = UniformMutationGeneratorConfig(num_mutations=2)
        generator = UniformMutationGenerator(gen_config)
        generator.assign(segment)

        constraint = ConstraintRegistry.create(
            key="sequence-length",
            segments=[segment],
            config_dict={"min_length": 40, "max_length": 60},
        )

        opt_config = TopKOptimizerConfig(num_samples=10, k=3, batch_size=2)
        optimizer = TopKOptimizer(
            constructs=[construct],
            generators=[generator],
            constraints=[constraint],
            config=opt_config,
        )

        program = Program(optimizers=[optimizer])
        result = serialize_program(program)

        assert result["constructs"][0]["type"] == "PROTEIN"


class TestCyclingOptimizerSerialization:
    """Tests for CyclingOptimizer serialization round-trip."""

    def test_cycling_optimizer_round_trip(self):
        """Test that CyclingOptimizer with pipeline serializes and parses correctly."""
        original_json = {
            "constructs": [{"type": "protein", "segments": [{"id": "prot0", "sequence": "ACDEFGHIKLMNPQRSTVWY" * 5}]}],
            "optimization_stages": [{
                "optimizer": {
                    "method": "cycling",
                    "target_segment": "prot0",
                    "config": {
                        "num_steps": 3,
                        "num_candidates": 2,
                        "conditioning_param_name": "structure_inputs",
                        "pipeline": "protein-hunter",
                        "protein_hunter": {"structure_tool": "chai"}
                    }
                },
                "generators": [{"key": "proteinmpnn", "target": "prot0", "config": {"temperature": 0.1}}],
                "constraints": []
            }]
        }

        # Parse
        program = DarwinParser(original_json).parse()

        # Serialize
        result = serialize_program(program)

        # Verify optimizer config
        opt = result["optimization_stages"][0]["optimizer"]
        assert opt["method"] == "cycling"
        # Serializer generates construct0-segment0 ID
        assert opt["target_segment"] == "construct0-segment0"
        assert opt["config"]["pipeline"] == "protein-hunter"
        assert opt["config"]["protein_hunter"]["structure_tool"] == "chai"

        # Re-parse and verify
        program2 = DarwinParser(result).parse()
        assert program2.optimizers[0].pipeline == "protein-hunter"
        assert program2.optimizers[0].protein_hunter.structure_tool == "chai"

    def test_cycling_optimizer_serializes_target_segment(self):
        """Test that target_segment is serialized correctly."""
        json_data = {
            "constructs": [{"type": "protein", "segments": [
                {"id": "seg1", "sequence": "A" * 50},
                {"id": "seg2", "sequence": "C" * 50}
            ]}],
            "optimization_stages": [{
                "optimizer": {
                    "method": "cycling",
                    "target_segment": "seg2",  # Targeting second segment
                    "config": {"num_steps": 2, "num_candidates": 2, "conditioning_param_name": "structure_inputs", "pipeline": "protein-hunter"}
                },
                "generators": [{"key": "proteinmpnn", "target": "seg2", "config": {"temperature": 0.1}}],
                "constraints": []
            }]
        }

        program = DarwinParser(json_data).parse()
        result = serialize_program(program)

        # Serializer generates IDs like construct0-segment1 (second segment)
        assert result["optimization_stages"][0]["optimizer"]["target_segment"] == "construct0-segment1"


class TestBeamSearchOptimizerSerialization:
    """Tests for BeamSearchOptimizer serialization round-trip."""

    def test_beam_search_optimizer_round_trip(self):
        """Test that BeamSearchOptimizer with target_segment serializes and parses correctly."""
        original_json = {
            "constructs": [{"type": "dna", "segments": [{"id": "seg0", "sequence": "ATGC" * 25}]}],
            "optimization_stages": [{
                "optimizer": {
                    "method": "beam-search",
                    "target_segment": "seg0",
                    "config": {
                        "prompt": "ATGC",
                        "beam_length": 10,
                        "beam_width": 4,
                        "candidates_per_beam": 8,
                    }
                },
                "generators": [{"key": "evo2", "target": "seg0", "config": {"prompts": ["ATGC"]}}],
                "constraints": [{"key": "gc-content", "targets": ["seg0"], "config": {"min_gc": 40, "max_gc": 60}}]
            }]
        }

        # Parse
        program = DarwinParser(original_json).parse()

        # Serialize
        result = serialize_program(program)

        # Verify optimizer config
        opt = result["optimization_stages"][0]["optimizer"]
        assert opt["method"] == "beam-search"
        assert opt["target_segment"] == "construct0-segment0"
        assert opt["config"]["beam_width"] == 4
        assert opt["config"]["beam_length"] == 10

        # Re-parse and verify
        program2 = DarwinParser(result).parse()
        assert program2.optimizers[0].__class__.__name__ == "BeamSearchOptimizer"
        assert program2.optimizers[0].target_segment is not None
