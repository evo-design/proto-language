# Test Templates

Complete test templates for each component type. Load this file on demand when writing new tests.

## Constraint Test Template

```python
import pytest
from proto_language.language.core import Constraint, Segment
from proto_language.language.constraint import my_constraint
from proto_language.language.constraint.{category}.{name}_constraint import MyConstraintConfig


class TestMyConstraint:
    @pytest.mark.parametrize(
        "sequence, param, expected_score",
        [
            ("GCGCGAATTA", 50, 0.0),   # Perfect score
            ("AAAAAAAAAA", 50, 1.0),    # Worst score
            ("GCATATAT", 50, 0.5),      # Partial score
            ("", 50, 1.0),             # Empty edge case
        ],
    )
    def test_scoring(self, sequence, param, expected_score):
        segment = Segment(sequence=sequence, sequence_type="dna")
        config = MyConstraintConfig(param=param)
        constraint = Constraint(
            inputs=[segment],
            function=my_constraint,
            function_config=config,
        )
        scores = constraint.evaluate()
        assert len(scores) == 1
        assert abs(scores[0] - expected_score) < 1e-9

    def test_wrong_sequence_type(self):
        """Protein sequences should raise TypeError at Constraint construction."""
        segment = Segment(sequence="MVLSPADKTNVK", sequence_type="protein")
        config = MyConstraintConfig(param=50)
        with pytest.raises(TypeError, match="does not support sequence type 'protein'"):
            Constraint(
                inputs=[segment],
                function=my_constraint,
                function_config=config,
            )

    def test_invalid_config(self):
        """Invalid config values should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MyConstraintConfig(param=-999)

    def test_metadata_propagation(self):
        """Verify metadata is stored on sequences after evaluation."""
        segment = Segment(sequence="GCGCGAATTA", sequence_type="dna")
        config = MyConstraintConfig(param=50)
        constraint = Constraint(
            inputs=[segment],
            function=my_constraint,
            function_config=config,
        )
        constraint.evaluate()

        # Check metadata on candidate sequences
        metadata = segment.candidate_sequences[0]._metadata
        constraints_meta = metadata["constraints"]
        assert "my_constraint" in constraints_meta
        assert "data" in constraints_meta["my_constraint"]
        # Check specific metadata fields
        assert "my_metric" in constraints_meta["my_constraint"]["data"]

    def test_rna_sequences(self):
        """Verify constraint works with RNA sequences (if supported)."""
        segment = Segment(sequence="GCGCGAUUUA", sequence_type="rna")
        config = MyConstraintConfig(param=50)
        constraint = Constraint(
            inputs=[segment],
            function=my_constraint,
            function_config=config,
        )
        scores = constraint.evaluate()
        assert 0.0 <= scores[0] <= 1.0
```

## Generator Test Template

```python
from __future__ import annotations

import copy
import pytest
from proto_language.language.core import Segment
from proto_language.language.generator import MyGenerator, MyGeneratorConfig


class TestMyGenerator:
    def test_initialization(self):
        """Config values stored correctly on instance."""
        config = MyGeneratorConfig(model_name="model_a", temperature=0.8)
        gen = MyGenerator(config)
        assert gen.model_name == "model_a"
        assert gen.temperature == 0.8

    def test_assign(self):
        """Segment assigned correctly, custom validation runs."""
        config = MyGeneratorConfig(model_name="model_a")
        gen = MyGenerator(config)
        segment = Segment(sequence="A" * 50, sequence_type="protein")
        gen.assign(segment)
        assert gen._assigned_segment is segment

    def test_sample_mutates_sequence(self):
        """sample() modifies candidate sequences in-place."""
        config = MyGeneratorConfig(model_name="model_a")
        gen = MyGenerator(config)
        segment = Segment(sequence="A" * 50, sequence_type="protein")
        gen.assign(segment)

        segment.candidate_sequences = [copy.deepcopy(segment.original_sequence)]
        initial = segment.candidate_sequences[0].sequence
        gen.sample()
        mutated = segment.candidate_sequences[0].sequence

        assert len(mutated) == 50
        assert mutated != initial  # Something changed

    def test_sample_batch(self):
        """sample() handles multiple candidates independently."""
        config = MyGeneratorConfig(model_name="model_a")
        gen = MyGenerator(config)
        segment = Segment(sequence="A" * 30, sequence_type="protein")
        gen.assign(segment)

        segment.candidate_sequences = [
            copy.deepcopy(segment.original_sequence) for _ in range(5)
        ]
        gen.sample()

        sequences = [s.sequence for s in segment.candidate_sequences]
        assert all(len(s) == 30 for s in sequences)

    def test_config_validation(self):
        """Invalid config raises ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MyGeneratorConfig(model_name="nonexistent")

    def test_sequence_type_validation(self):
        """Unsupported sequence type raises ValueError on assign."""
        config = MyGeneratorConfig(model_name="model_a")
        gen = MyGenerator(config)
        segment = Segment(sequence="ATCG", sequence_type="dna")
        # If generator only supports protein:
        with pytest.raises(ValueError, match="does not support sequence type"):
            gen.assign(segment)


class TestMyGeneratorValidation:
    """Sequence type compatibility tests."""

    def test_accepts_supported_type(self):
        config = MyGeneratorConfig(model_name="model_a")
        gen = MyGenerator(config)
        segment = Segment(length=50, sequence_type="protein")
        gen.assign(segment)
        assert gen._assigned_segment is segment
```

## Optimizer Test Template

```python
from __future__ import annotations

import copy
from typing import Tuple

import pytest
from pydantic import BaseModel

from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.core import Constraint, Construct, Segment
from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig
from proto_language.language.optimizer import MyOptimizer, MyOptimizerConfig


def _setup_components(
    seq_length: int = 10,
    num_results: int = 5,
    num_steps: int = 10,
    gc_range: Tuple[float, float] = (40.0, 60.0),
):
    """Helper to create optimizer with standard test components."""
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
    gen.assign(segment)

    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=gc_range[0], max_gc=gc_range[1]),
    )

    config = MyOptimizerConfig(num_results=num_results, num_steps=num_steps)
    opt = MyOptimizer(
        constructs=[construct],
        generators=[gen],
        constraints=[constraint],
        config=config,
    )
    return opt, gen, constraint, segment


class TestMyOptimizer:
    def test_initialization(self):
        """Optimizer initializes with correct config values."""
        opt, _, _, _ = _setup_components()
        assert opt.num_results == 5
        assert len(opt.generators) == 1
        assert len(opt.constraints) == 1

    def test_config_validation(self):
        """Invalid config raises ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MyOptimizerConfig(num_results=-1, num_steps=10)

    def test_run_completes(self):
        """run() completes without error."""
        opt, _, _, _ = _setup_components(num_steps=5)
        opt.run()
        assert len(opt.history) > 0

    def test_scores_improve(self):
        """Scores should generally improve over optimization."""
        opt, _, _, _ = _setup_components(num_steps=50)
        opt.run()

        initial_score = opt.history[0]["energy_scores"][0]
        final_score = opt.history[-1]["energy_scores"][0]
        # Final should be <= initial (lower = better)
        assert final_score <= initial_score

    def test_history_tracking(self):
        """Snapshots saved at correct intervals."""
        opt, _, _, _ = _setup_components(num_steps=20)
        opt.run()
        tracked_steps = [h["time_step"] for h in opt.history]
        assert 0 in tracked_steps
        assert 20 in tracked_steps

    def test_unassigned_generator_raises(self):
        """Unassigned generator should raise RuntimeError."""
        segment = Segment(sequence="A" * 10, sequence_type="dna")
        gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
        # NOT calling gen.assign(segment)

        def dummy(input_sequences, config=None):
            return [0.0 for _ in input_sequences]
        dummy._constraint_config_class = type("E", (BaseModel,), {})
        dummy._constraint_supported_sequence_types = ["dna"]

        constraint = Constraint(
            inputs=[segment], function=dummy, function_config=dummy._constraint_config_class(),
        )
        with pytest.raises(RuntimeError, match="has no segment assigned"):
            MyOptimizer(
                constructs=[Construct([segment])],
                generators=[gen],
                constraints=[constraint],
                config=MyOptimizerConfig(num_results=1, num_steps=1),
            )

    def test_filter_constraints(self):
        """Filter constraints (with threshold) reject bad candidates."""
        segment = Segment(sequence="A" * 20, sequence_type="dna")
        gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
        gen.assign(segment)

        # Filter: only accept sequences with GC in [40, 60]
        filter_constraint = Constraint(
            inputs=[segment],
            function=gc_content_constraint,
            function_config=GCContentConfig(min_gc=40, max_gc=60),
            threshold=0.1,  # Makes it a filter
        )

        config = MyOptimizerConfig(num_results=5, num_steps=10)
        opt = MyOptimizer(
            constructs=[Construct([segment])],
            generators=[gen],
            constraints=[filter_constraint],
            config=config,
        )
        opt.run()  # Should complete without error
```
