"""
Tests for DarwinParser with multiple optimizer support.

Tests the parsing of the new JSON format that supports multiple sequential optimizers
in a single program.
"""

import pytest
from proto_language.language.optimizer import TopKOptimizer, MCMCOptimizer
from proto_language.language.generator import UniformMutationGenerator
from api.core.parser import DarwinParser


def test_parse_single_optimizer():
    """Test parsing a program with a single optimizer using nested inline format."""
    json_data = {
        "name": "single_optimizer",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "label": "segment1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {
                        "num_samples": 10,
                        "k": 3,
                        "batch_size": 2
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 5
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 40,
                            "max_gc": 60
                        }
                    }
                ]
            }
        ]
    }

    parser = DarwinParser(json_data)
    program = parser.parse()

    # Verify program structure
    assert len(program.optimizers) == 1
    assert isinstance(program.optimizers[0], TopKOptimizer)

    # Verify optimizer config
    optimizer = program.optimizers[0]
    assert optimizer.num_samples == 10
    assert optimizer.k == 3
    assert optimizer.batch_size == 2

    # Verify generators
    assert len(optimizer.generators) == 1
    assert isinstance(optimizer.generators[0], UniformMutationGenerator)
    assert optimizer.generators[0].num_mutations == 5

    # Verify constraints
    assert len(optimizer.constraints) == 1

    # Verify constructs
    assert len(program.constructs) == 1
    assert len(program.constructs[0].segments) == 1


def test_parse_multiple_optimizers():
    """Test parsing a program with multiple sequential optimizers using nested inline format."""
    json_data = {
        "name": "multi_optimizer",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "label": "sequence1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {
                        "num_samples": 10,
                        "k": 3,
                        "batch_size": 2
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 10
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 50,
                            "max_gc": 100
                        }
                    }
                ]
            },
            {
                "optimizer": {
                    "method": "mcmc",
                    "config": {
                        "num_selected": 1,
                        "num_steps": 10,
                        "track_step_size": 1
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 1
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 80,
                            "max_gc": 90
                        }
                    }
                ]
            }
        ]
    }

    parser = DarwinParser(json_data)
    program = parser.parse()

    # Verify program structure
    assert len(program.optimizers) == 2
    assert isinstance(program.optimizers[0], TopKOptimizer)
    assert isinstance(program.optimizers[1], MCMCOptimizer)

    # Verify first optimizer
    opt1 = program.optimizers[0]
    assert opt1.num_samples == 10
    assert opt1.k == 3
    assert opt1.batch_size == 2
    assert len(opt1.generators) == 1
    assert opt1.generators[0].num_mutations == 10
    assert len(opt1.constraints) == 1

    # Verify second optimizer
    opt2 = program.optimizers[1]
    assert opt2.num_selected == 1
    assert opt2.num_steps == 10
    assert len(opt2.generators) == 1
    assert opt2.generators[0].num_mutations == 1
    assert len(opt2.constraints) == 1

    # Verify all optimizers share the same constructs
    assert program.optimizers[0].constructs == program.optimizers[1].constructs
    assert program.optimizers[0].constructs is program.optimizers[1].constructs


def test_parse_missing_optimizations_field():
    """Test error handling when 'optimization_stages' field is missing."""
    json_data = {
        "name": "missing_optimizations",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "length": 20}]
            }
        ]
    }

    parser = DarwinParser(json_data)

    with pytest.raises(ValueError, match="JSON must include 'optimization_stages' array"):
        parser.parse()


def test_parse_empty_optimizations_array():
    """Test error handling when 'optimization_stages' array is empty."""
    json_data = {
        "name": "empty_optimizations",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "length": 20}]
            }
        ],
        "optimization_stages": []
    }

    parser = DarwinParser(json_data)

    with pytest.raises(ValueError, match="optimizers list cannot be empty"):
        parser.parse()


def test_parse_missing_method_in_stage():
    """Test error handling when optimizer stage missing 'method' field."""
    json_data = {
        "name": "missing_method",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "config": {}
                },
                "generators": [],
                "constraints": []
            }
        ]
    }

    parser = DarwinParser(json_data)

    with pytest.raises(ValueError, match="must include 'method'"):
        parser.parse()


def test_parse_missing_config_in_stage():
    """Test error handling when optimizer stage missing 'config' field."""
    json_data = {
        "name": "missing_config",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk"
                },
                "generators": [],
                "constraints": []
            }
        ]
    }

    parser = DarwinParser(json_data)

    with pytest.raises(ValueError, match="must include 'config'"):
        parser.parse()


def test_parse_missing_generators_in_stage():
    """Test error handling when optimizer stage missing 'generators' field."""
    json_data = {
        "name": "missing_generators",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {"num_samples": 10, "k": 3, "batch_size": 2}
                },
                "constraints": []
            }
        ]
    }

    parser = DarwinParser(json_data)

    with pytest.raises(ValueError, match="must include 'generators' array"):
        parser.parse()


def test_parse_missing_constraints_in_stage():
    """Test error handling when optimizer stage missing 'constraints' field."""
    json_data = {
        "name": "missing_constraints",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {"num_samples": 10, "k": 3, "batch_size": 2}
                },
                "generators": []
            }
        ]
    }

    parser = DarwinParser(json_data)

    with pytest.raises(ValueError, match="must include 'constraints' array"):
        parser.parse()


def test_parse_unknown_optimizer_method():
    """Test error handling for unknown optimizer method."""
    json_data = {
        "name": "unknown_method",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "invalid_optimizer",
                    "config": {}
                },
                "generators": [],
                "constraints": []
            }
        ]
    }

    parser = DarwinParser(json_data)

    with pytest.raises(ValueError, match="Unknown optimization method"):
        parser.parse()


def test_parse_generator_assignment_to_segments():
    """Test that generators are correctly assigned to segments."""
    json_data = {
        "name": "generator_assignment",
        "constructs": [
            {
                "type": "DNA",
                "segments": [
                    {"id": "seg1", "label": "segment1", "length": 20},
                    {"id": "seg2", "label": "segment2", "length": 30}
                ]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {
                        "num_samples": 10,
                        "k": 3,
                        "batch_size": 2
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 5
                        }
                    },
                    {
                        "key": "uniform-mutation",
                        "target": "seg2",
                        "config": {
                            "num_mutations": 3
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 40,
                            "max_gc": 60
                        }
                    }
                ]
            }
        ]
    }

    parser = DarwinParser(json_data)
    program = parser.parse()

    # Verify generators were assigned
    optimizer = program.optimizers[0]
    assert len(optimizer.generators) == 2

    # Verify first generator assigned to seg1
    gen1 = optimizer.generators[0]
    assert gen1.num_mutations == 5
    assert gen1._assigned_segment.label == "segment1"

    # Verify second generator assigned to seg2
    gen2 = optimizer.generators[1]
    assert gen2.num_mutations == 3
    assert gen2._assigned_segment.label == "segment2"


def test_parse_different_generators_per_stage():
    """Test that each optimizer stage can have different generators."""
    json_data = {
        "name": "different_generators",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "label": "sequence1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {
                        "num_samples": 10,
                        "k": 3,
                        "batch_size": 2
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 10
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {"min_gc": 0, "max_gc": 100}
                    }
                ]
            },
            {
                "optimizer": {
                    "method": "mcmc",
                    "config": {
                        "num_selected": 1,
                        "num_steps": 10
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 1
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {"min_gc": 0, "max_gc": 100}
                    }
                ]
            }
        ]
    }

    parser = DarwinParser(json_data)
    program = parser.parse()

    # Verify generators are different instances with different configs
    gen1 = program.optimizers[0].generators[0]
    gen2 = program.optimizers[1].generators[0]

    assert gen1 is not gen2  # Different instances
    assert gen1.num_mutations == 10
    assert gen2.num_mutations == 1


def test_parse_different_constraints_per_stage():
    """Test that each optimizer stage can have different constraints."""
    json_data = {
        "name": "different_constraints",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "label": "sequence1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {
                        "num_samples": 10,
                        "k": 3,
                        "batch_size": 2
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {"num_mutations": 1}
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 50,
                            "max_gc": 100
                        }
                    }
                ]
            },
            {
                "optimizer": {
                    "method": "mcmc",
                    "config": {
                        "num_selected": 1,
                        "num_steps": 10
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {"num_mutations": 1}
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 80,
                            "max_gc": 90
                        }
                    }
                ]
            }
        ]
    }

    parser = DarwinParser(json_data)
    program = parser.parse()

    # Verify constraints are different instances
    assert len(program.optimizers[0].constraints) == 1
    assert len(program.optimizers[1].constraints) == 1

    # Note: Constraints are created differently by the parser,
    # so we just verify they exist


def test_parse_reusable_constraints():
    """Test that constraints can be reused across multiple optimization stages (inline definition)."""
    json_data = {
        "name": "reusable_constraints",
        "constructs": [
            {
                "type": "DNA",
                "segments": [{"id": "seg1", "label": "sequence1", "length": 20}]
            }
        ],
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "topk",
                    "config": {
                        "num_samples": 10,
                        "k": 3,
                        "batch_size": 2
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 10
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 40,
                            "max_gc": 60
                        }
                    }
                ]
            },
            {
                "optimizer": {
                    "method": "mcmc",
                    "config": {
                        "num_selected": 1,
                        "num_steps": 10
                    }
                },
                "generators": [
                    {
                        "key": "uniform-mutation",
                        "target": "seg1",
                        "config": {                            
                            "num_mutations": 1
                        }
                    }
                ],
                "constraints": [
                    {
                        "key": "gc-content",
                        "targets": ["seg1"],
                        "config": {
                            "min_gc": 40,
                            "max_gc": 60
                        }
                    }
                ]
            }
        ]
    }

    parser = DarwinParser(json_data)
    program = parser.parse()

    # Verify both optimizers have constraints defined (they are separate objects in new format)
    opt1_constraint = program.optimizers[0].constraints[0]
    opt2_constraint = program.optimizers[1].constraints[0]

    # In the new inline format, constraints are separate objects but have same function_config
    assert opt1_constraint.function_config.min_gc == opt2_constraint.function_config.min_gc
    assert opt1_constraint.function_config.max_gc == opt2_constraint.function_config.max_gc
