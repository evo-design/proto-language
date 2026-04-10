"""Tests for hierarchical seed propagation: Program -> Optimizer -> Generator."""

from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.core import Constraint, Construct, Program, Segment
from proto_language.language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig

# num_mutations=20 (full sequence) avoids non-deterministic masking position selection
# so only nucleotide sampling (which IS seeded) determines the output.
_SEQ = "A" * 20
_MASKING = MaskingStrategy(num_mutations=20)
_GC = GCContentConfig(min_gc=0.0, max_gc=100.0)


def _make_mcmc(seed=None, gen_seed=None, num_steps=5, num_results=2):
    """Create a minimal MCMC optimizer + segment for seed testing."""
    segment = Segment(sequence=_SEQ, sequence_type="dna")
    gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=_MASKING, seed=gen_seed))
    gen.assign(segment)
    construct = Construct([segment])
    constraint = Constraint(inputs=[segment], function=gc_content_constraint, function_config=_GC)
    config = MCMCOptimizerConfig(num_results=num_results, num_steps=num_steps, seed=seed)
    optimizer = MCMCOptimizer(constructs=[construct], generators=[gen], constraints=[constraint], config=config)
    return optimizer, segment


def _run_program(program_seed, opt_seed=None, num_steps=5):
    """Run a single-optimizer Program, return result sequence strings."""
    optimizer, segment = _make_mcmc(seed=opt_seed, num_steps=num_steps)
    Program(optimizers=[optimizer], num_results=2, seed=program_seed).run()
    return [s.sequence for s in segment.result_sequences]


def _make_two_stage_program(seed=42):
    """Create a Program with two sequential MCMC optimizers sharing one segment."""
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
    """Hierarchical seed propagation: Program -> Optimizer -> Generator."""

    def test_program_seed_determinism(self):
        assert _run_program(42) == _run_program(42)

    def test_program_seed_divergence(self):
        assert _run_program(42, num_steps=10) != _run_program(99, num_steps=10)

    def test_optimizer_seed_determinism(self):
        results = []
        for _ in range(2):
            opt, seg = _make_mcmc(seed=42)
            opt.run()
            results.append([s.sequence for s in seg.result_sequences])
        assert results[0] == results[1]

    def test_optimizer_seed_divergence(self):
        results = []
        for seed in [42, 99]:
            opt, seg = _make_mcmc(seed=seed)
            opt.run()
            results.append([s.sequence for s in seg.result_sequences])
        assert results[0] != results[1]

    def test_program_overrides_optimizer_seed(self):
        assert _run_program(42, opt_seed=100) == _run_program(42, opt_seed=200)

    def test_multi_optimizer_determinism(self):
        results = []
        for _ in range(2):
            program, segment, _ = _make_two_stage_program(seed=42)
            program.run()
            results.append([s.sequence for s in segment.result_sequences])
        assert results[0] == results[1]

    def test_optimizers_get_different_seeds(self):
        _, _, optimizers = _make_two_stage_program(seed=42)
        assert optimizers[0].seed != optimizers[1].seed

    def test_rerun_determinism(self):
        opt, seg = _make_mcmc()
        program = Program(optimizers=[opt], num_results=2, seed=42)
        program.run()
        results1 = [s.sequence for s in seg.result_sequences]
        program.run()
        results2 = [s.sequence for s in seg.result_sequences]
        assert results1 == results2

    def test_next_seed_advances(self):
        """Consecutive _next_seed() calls return different values."""
        gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=_MASKING))
        gen._set_program_seed(42)
        seeds = [gen._next_seed() for _ in range(5)]
        assert len(set(seeds)) == 5

    def test_config_seed_fallback(self):
        """Generator with config seed (no program seed) is deterministic.

        Deterministic here because: single generator (no random selection) and
        trivial constraint (all proposals accepted, no MH randomness needed).
        """
        results = []
        for _ in range(2):
            opt, seg = _make_mcmc(gen_seed=42)
            opt.run()
            results.append([s.sequence for s in seg.result_sequences])
        assert results[0] == results[1]

    def test_unseeded_next_seed_returns_none(self):
        """_next_seed() returns None when no seed is configured."""
        gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=_MASKING))
        assert gen._next_seed() is None

    def test_negative_program_seed_rejected(self):
        """Program rejects negative seeds."""
        import pytest

        opt, _ = _make_mcmc()
        with pytest.raises(ValueError, match="non-negative"):
            Program(optimizers=[opt], num_results=2, seed=-1)
