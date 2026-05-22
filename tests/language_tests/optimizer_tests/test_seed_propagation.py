"""Tests for hierarchical seed propagation: Program -> Optimizer -> Generator/Constraint."""

import pytest
from proto_tools.transforms.masking import MaskingStrategy

from proto_language import AlphaFold2MultimerStructureConfig, StructureBasedConstraintConfig
from proto_language.constraint import gc_content_constraint
from proto_language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.core import Constraint, ConstraintOutput, Construct, Program, Segment
from proto_language.core.optimizer import derive_seeds
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.utils.alphafold2_multimer import next_af2_multimer_seed
from proto_language.utils.base import BaseConfig, ConfigField

# Full-sequence mutation leaves nucleotide sampling as the only random choice.
_SEQ = "A" * 20
_MASKING = MaskingStrategy(num_mutations=20)
_GC = GCContentConfig(min_gc=0.0, max_gc=100.0)


class RuntimeNestedSeedConfig(BaseConfig):
    """Nested seed-bearing test config.

    Attributes:
        seed (int | None): Random seed for the nested config.
    """

    seed: int | None = ConfigField(default=None, ge=0, title="Seed", description="Random seed for the nested config.")


class RuntimeSeedConfig(BaseConfig):
    """Seed-bearing test config.

    Attributes:
        seed (int | None): Top-level random seed.
        seeds (list[int]): Multiple seed values used by the fixture.
        nested (RuntimeNestedSeedConfig): Nested seed-bearing config.
        raw (dict[str, object]): Raw seed payload used by the fixture.
    """

    seed: int | None = ConfigField(default=None, ge=0, title="Seed", description="Top-level random seed.")
    seeds: list[int] = ConfigField(
        default_factory=lambda: [999],
        title="Seeds",
        description="Multiple seed values used by the fixture.",
    )
    nested: RuntimeNestedSeedConfig = ConfigField(
        default_factory=RuntimeNestedSeedConfig,
        title="Nested",
        description="Nested seed-bearing config.",
    )
    raw: dict[str, object] = ConfigField(
        default_factory=lambda: {"seed": None, "seeds": [999]},
        title="Raw",
        description="Raw seed payload used by the fixture.",
    )


def _make_mcmc(seed=None, num_steps=5, num_results=2, constraint=None):
    """Create a minimal MCMC optimizer."""
    segment = Segment(sequence=_SEQ, sequence_type="dna")
    gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=_MASKING))
    gen.assign(segment)
    construct = Construct([segment])
    constraint = constraint or Constraint(inputs=[segment], function=gc_content_constraint, function_config=_GC)
    config = MCMCOptimizerConfig(num_results=num_results, num_steps=num_steps, seed=seed)
    optimizer = MCMCOptimizer(constructs=[construct], generators=[gen], constraints=[constraint], config=config)
    return optimizer, segment


def _run_program(program_seed, opt_seed=None, num_steps=5):
    """Run one optimizer and return result sequence strings."""
    optimizer, segment = _make_mcmc(seed=opt_seed, num_steps=num_steps)
    Program(optimizers=[optimizer], num_results=2, seed=program_seed).run()
    return [s.sequence for s in segment.result_sequences]


def _make_two_stage_program(seed=42):
    """Create a two-stage MCMC program."""
    segment = Segment(sequence=_SEQ, sequence_type="dna")
    construct = Construct([segment])
    optimizers = []
    for _ in range(2):
        gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=_MASKING))
        gen.assign(segment)
        constraint = Constraint(inputs=[segment], function=gc_content_constraint, function_config=_GC)
        opt = MCMCOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_results=2, num_steps=5),
        )
        optimizers.append(opt)
    program = Program(optimizers=optimizers, num_results=2, seed=seed)
    return program, segment, optimizers


class TestSeedPropagation:
    """Hierarchical seed propagation: Program -> Optimizer -> Generator/Constraint."""

    def test_program_seed_controls_run(self):
        seeded = _run_program(42, num_steps=10)

        assert seeded == _run_program(42, num_steps=10)
        assert seeded != _run_program(99, num_steps=10)

    def test_optimizer_seed_controls_run(self):
        def run(seed):
            opt, seg = _make_mcmc(seed=seed)
            opt.run()
            return [s.sequence for s in seg.result_sequences]

        seeded = run(42)
        assert seeded == run(42)
        assert seeded != run(99)

    def test_program_overrides_optimizer_seed(self):
        assert _run_program(42, opt_seed=100) == _run_program(42, opt_seed=200)

    def test_optimizer_seed_is_backed_by_config(self):
        opt, _ = _make_mcmc(seed=42)
        assert opt.seed == 42
        assert opt.config.seed == 42

        opt.seed = 99
        assert opt.seed == 99
        assert opt.config.seed == 99

        opt.config.seed = 123
        assert opt.seed == 123
        with pytest.raises(ValueError, match="non-negative"):
            opt.seed = -1

    def test_program_seed_overrides_optimizer_config_seed(self):
        opt, _ = _make_mcmc(seed=100)
        Program(optimizers=[opt], num_results=2, seed=42)

        assert opt.seed == derive_seeds(42, 1)[0]
        assert opt.config.seed == opt.seed

    def test_multi_optimizer_seed_streams(self):
        program, segment, optimizers = _make_two_stage_program(seed=42)
        assert optimizers[0].seed != optimizers[1].seed
        assert optimizers[0].config.seed == optimizers[0].seed
        assert optimizers[1].config.seed == optimizers[1].seed

        program.run()
        results = [s.sequence for s in segment.result_sequences]
        program2, segment2, _ = _make_two_stage_program(seed=42)
        program2.run()
        assert results == [s.sequence for s in segment2.result_sequences]

    def test_rerun_determinism(self):
        opt, seg = _make_mcmc()
        program = Program(optimizers=[opt], num_results=2, seed=42)
        program.run()
        results1 = [s.sequence for s in seg.result_sequences]
        program.run()
        results2 = [s.sequence for s in seg.result_sequences]
        assert results1 == results2

    def test_generator_call_seed_stream(self):
        gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=_MASKING))
        assert gen._next_seed() is None

        gen._set_program_seed(42)
        seeds = [gen._next_seed() for _ in range(5)]
        assert len(set(seeds)) == 5

        gen._set_program_seed(42)
        assert [gen._next_seed() for _ in range(5)] == seeds
        gen._set_program_seed(None)
        assert gen._next_seed() is None

    def test_unseeded_run_clears_generator_seed_stream(self):
        opt, _ = _make_mcmc(seed=42)
        opt.run()
        assert opt.generators[0]._next_seed() is not None

        opt.seed = None
        opt.run()
        assert opt.generators[0]._next_seed() is None

    def test_constraint_config_seeded_from_optimizer_seed(self):
        seen = []

        def record_seed(input_sequences, config):
            seen.append((config.seed, list(config.seeds), config.nested.seed, dict(config.raw)))
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        config = RuntimeSeedConfig()
        segment = Segment(sequence=_SEQ, sequence_type="dna")
        constraint = Constraint(inputs=[segment], function=record_seed, function_config=config)
        gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=_MASKING))
        gen.assign(segment)
        opt = MCMCOptimizer(
            constructs=[Construct([segment])],
            generators=[gen],
            constraints=[constraint],
            config=MCMCOptimizerConfig(num_results=2, num_steps=1, seed=42),
        )

        opt.run()

        expected_constraint_seed = derive_seeds(42, 2)[1]
        assert seen
        assert seen[0] == (
            expected_constraint_seed,
            [expected_constraint_seed],
            expected_constraint_seed,
            {"seed": expected_constraint_seed, "seeds": [expected_constraint_seed]},
        )

    def test_constraint_seed_state_resets_for_reruns(self):
        def score(input_sequences, config):
            return [ConstraintOutput(score=0.0) for _ in input_sequences]

        af2_config = AlphaFold2MultimerStructureConfig(target_pdb="ATOM", seed=99)
        config = StructureBasedConstraintConfig(
            structure_tool="alphafold2_multimer",
            alphafold2_multimer_config=af2_config,
        )
        segment = Segment(sequence="ACDE", sequence_type="protein")
        constraint = Constraint(inputs=[segment], function=score, function_config=config)

        constraint._set_program_seed(123)
        first = next_af2_multimer_seed(af2_config)
        second = next_af2_multimer_seed(af2_config)

        constraint._set_program_seed(123)

        assert af2_config._evaluation_seed_offset == 0
        assert af2_config.seed == 123
        assert next_af2_multimer_seed(af2_config) == first
        assert second != first

    def test_negative_program_seed_rejected(self):
        opt, _ = _make_mcmc()
        with pytest.raises(ValueError, match="non-negative"):
            Program(optimizers=[opt], num_results=2, seed=-1)
