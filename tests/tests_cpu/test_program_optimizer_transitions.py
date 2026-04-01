"""Tests for all 16 optimizer transition permutations.

Transition Matrix:
    From / To        | TopK | MCMC | BeamSearch | CyclingOptimizer
    -----------------|------|------|------------|------------------
    TopK             |  1   |  2   |     3      |        4
    MCMC             |  5   |  6   |     7      |        8
    BeamSearch       |  9   |  10  |    11      |       12
    CyclingOptimizer | 13   |  14  |    15      |       16

Note: BeamSearch ignores previous state by design (always starts from prompt).
"""

from __future__ import annotations

import random
from unittest.mock import Mock

from proto_tools.tools.masked_models.masking import MaskingStrategy

from proto_language.language.constraint import gc_content_constraint
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.language.optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
    CyclingOptimizer,
    CyclingOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
    TopKOptimizer,
    TopKOptimizerConfig,
)

# =============================================================================
# Mock Generators
# =============================================================================


class MockAutoregressiveGenerator(Generator):
    """Mock autoregressive generator for BeamSearch testing without GPU."""

    def __init__(self, use_kv_caching: bool = True):
        super().__init__()
        self.use_kv_caching = use_kv_caching
        self.kv_caches: list[dict] = []

    def assign(self, assigned_segment: Segment) -> None:
        self._assigned_segment = assigned_segment

    def sample(
        self,
        prompts: list[str] | None = None,
        prepend_prompt: bool | None = None,
        num_tokens: int | None = None,
        old_kv_cache: dict | None = None,
    ) -> None:
        if num_tokens is None:
            num_tokens = 20
        if prompts is None:
            prompts = [""]
        sequences = []
        for prompt in prompts:
            new_seq = "".join(random.choice("ATCG") for _ in range(num_tokens))  # noqa: S311 -- non-cryptographic, test mock
            sequences.append(prompt + new_seq if prepend_prompt else new_seq)
        self._assigned_segment.proposal_sequences = [Sequence(sequence=seq, sequence_type="dna") for seq in sequences]
        if self.use_kv_caching:
            mock_mha = Mock()
            mock_mha.key_value_memory_dict = {0: Mock(shape=(1, 2, 3))}
            mock_mha.seqlen_offset = 10
            self.kv_caches = [{"mha": mock_mha, "hcl": Mock()} for _ in range(len(prompts))]
        else:
            self.kv_caches = []

    def replicate_cache(self, cache: dict, n_replicates: int) -> dict:
        return cache


class MockCyclingGenerator(Generator):
    """Mock generator for CyclingOptimizer testing."""

    def __init__(self):
        super().__init__()

    def assign(self, assigned_segment: Segment) -> None:
        self._assigned_segment = assigned_segment

    def sample(self, structure_inputs=None) -> None:
        # Mutate each proposal sequence slightly
        for seq in self._assigned_segment.proposal_sequences:
            chars = list(seq.sequence)
            if chars:
                idx = random.randint(0, len(chars) - 1)  # noqa: S311 -- non-cryptographic, test mock
                chars[idx] = random.choice("ACGT")  # noqa: S311 -- non-cryptographic, test mock
                seq.sequence = "".join(chars)


# =============================================================================
# Helper Functions
# =============================================================================


def create_topk_optimizer(construct, segment, num_samples=20, num_results=3, num_mutations=3):
    """Create a TopK optimizer."""
    gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=num_mutations))
    )
    gen.assign(segment)
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 0, "max_gc": 100},
    )
    return TopKOptimizer(
        constructs=[construct],
        generators=[gen],
        constraints=[constraint],
        config=TopKOptimizerConfig(num_samples=num_samples, num_results=num_results),
    )


def create_mcmc_optimizer(construct, segment, num_results=3, num_steps=10, num_mutations=2):
    """Create an MCMC optimizer."""
    gen = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=num_mutations))
    )
    gen.assign(segment)
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 0, "max_gc": 100},
    )
    return MCMCOptimizer(
        constructs=[construct],
        generators=[gen],
        constraints=[constraint],
        config=MCMCOptimizerConfig(num_results=num_results, num_steps=num_steps),
    )


def create_beamsearch_optimizer(construct, segment, num_results=3, beam_length=10, prompt="ATCG"):
    """Create a BeamSearch optimizer with mock generator."""
    generator = MockAutoregressiveGenerator(use_kv_caching=True)
    generator._assigned_segment = segment
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 0, "max_gc": 100},
    )
    return BeamSearchOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=[constraint],
        config=BeamSearchOptimizerConfig(
            prompt=prompt,
            beam_length=beam_length,
            num_results=num_results,
            proposals_per_result=3,
            use_kv_caching=True,
        ),
        target_segment=segment,
    )


def create_cycling_optimizer(construct, segment, num_results=3, num_steps=5):
    """Create a CyclingOptimizer with mock generator."""
    generator = MockCyclingGenerator()
    generator.assign(segment)

    def mock_conditioning_fn(sequences):
        return [{"mock": True} for _ in sequences]

    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config={"min_gc": 0, "max_gc": 100},
        threshold=1.0,  # Filter constraint for CyclingOptimizer
    )
    return CyclingOptimizer(
        target_segment=segment,
        constructs=[construct],
        generators=[generator],
        constraints=[constraint],
        config=CyclingOptimizerConfig(
            num_steps=num_steps,
            num_results=num_results,
            conditioning_param_name="structure_inputs",
        ),
        conditioning_fn=mock_conditioning_fn,
    )


# =============================================================================
# Test Classes for All 16 Permutations
# =============================================================================


class TestTopKTransitions:
    """Tests 1-4: Transitions FROM TopK."""

    def test_1_topk_to_topk(self):
        """TopK -> TopK: Second optimizer uses sorted results from first."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_topk_optimizer(construct, segment, num_samples=15, num_results=3)
        opt2 = create_topk_optimizer(construct, segment, num_samples=10, num_results=2)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        opt1_seqs = [s.sequence for s in segment.result_sequences]
        assert len(opt1_seqs) == 3
        assert opt1.energy_scores == sorted(opt1.energy_scores), "Should be sorted"

        program.run_stage(1)
        assert len(segment.result_sequences) == 2
        assert opt2.energy_scores == sorted(opt2.energy_scores)

    def test_2_topk_to_mcmc(self):
        """TopK -> MCMC: MCMC inherits sorted sequences from TopK."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_topk_optimizer(construct, segment, num_samples=15, num_results=3)
        opt2 = create_mcmc_optimizer(construct, segment, num_results=2, num_steps=5)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        assert opt1.energy_scores == sorted(opt1.energy_scores)

        program.run_stage(1)
        assert len(segment.result_sequences) == 2

    def test_3_topk_to_beamsearch(self):
        """TopK -> BeamSearch: BeamSearch IGNORES TopK results, starts from prompt."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_topk_optimizer(construct, segment, num_samples=10, num_results=3)
        opt2 = create_beamsearch_optimizer(construct, segment, num_results=2, beam_length=10, prompt="GGGG")

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)

        program.run_stage(1)
        beam_seqs = [s.sequence for s in segment.result_sequences]

        # BeamSearch starts fresh from prompt, doesn't use TopK results
        assert len(beam_seqs) == 2
        # All beam sequences should start with the prompt
        assert all(s.startswith("GGGG") for s in beam_seqs), "BeamSearch should start from prompt"

    def test_4_topk_to_cycling(self):
        """TopK -> CyclingOptimizer: Cycling inherits sorted sequences."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_topk_optimizer(construct, segment, num_samples=10, num_results=3)
        opt2 = create_cycling_optimizer(construct, segment, num_results=2, num_steps=3)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        topk_seqs = [s.sequence for s in segment.result_sequences]
        assert len(topk_seqs) == 3

        program.run_stage(1)
        # CyclingOptimizer should have run with 2 proposals
        assert len(segment.result_sequences) == 2


class TestMCMCTransitions:
    """Tests 5-8: Transitions FROM MCMC."""

    def test_5_mcmc_to_topk(self):
        """MCMC -> TopK: TopK uses MCMC's results."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_mcmc_optimizer(construct, segment, num_results=3, num_steps=10)
        opt2 = create_topk_optimizer(construct, segment, num_samples=10, num_results=2)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)

        program.run_stage(1)
        assert len(segment.result_sequences) == 2
        assert opt2.energy_scores == sorted(opt2.energy_scores)

    def test_6_mcmc_to_mcmc(self):
        """MCMC -> MCMC: Second MCMC inherits and continues optimization."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_mcmc_optimizer(construct, segment, num_results=3, num_steps=10)
        opt2 = create_mcmc_optimizer(construct, segment, num_results=2, num_steps=5)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        mcmc1_seqs = [s.sequence for s in segment.result_sequences]
        assert len(mcmc1_seqs) == 3

        program.run_stage(1)
        mcmc2_seqs = [s.sequence for s in segment.result_sequences]
        assert len(mcmc2_seqs) == 2

    def test_7_mcmc_to_beamsearch(self):
        """MCMC -> BeamSearch: BeamSearch IGNORES MCMC results."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_mcmc_optimizer(construct, segment, num_results=3, num_steps=10)
        opt2 = create_beamsearch_optimizer(construct, segment, num_results=2, beam_length=10, prompt="CCCC")

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        program.run_stage(1)

        beam_seqs = [s.sequence for s in segment.result_sequences]
        assert all(s.startswith("CCCC") for s in beam_seqs), "BeamSearch ignores previous state"

    def test_8_mcmc_to_cycling(self):
        """MCMC -> CyclingOptimizer: Cycling inherits MCMC's results."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_mcmc_optimizer(construct, segment, num_results=3, num_steps=10)
        opt2 = create_cycling_optimizer(construct, segment, num_results=2, num_steps=3)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        mcmc_seqs = [s.sequence for s in segment.result_sequences]
        assert len(mcmc_seqs) == 3

        program.run_stage(1)
        assert len(segment.result_sequences) == 2


class TestBeamSearchTransitions:
    """Tests 9-12: Transitions FROM BeamSearch."""

    def test_9_beamsearch_to_topk(self):
        """BeamSearch -> TopK: TopK uses BeamSearch's results."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_beamsearch_optimizer(construct, segment, num_results=3, beam_length=10, prompt="AAAA")
        opt2 = create_topk_optimizer(construct, segment, num_samples=10, num_results=2)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        beam_seqs = [s.sequence for s in segment.result_sequences]
        assert len(beam_seqs) == 3
        assert all(s.startswith("AAAA") for s in beam_seqs)

        program.run_stage(1)
        assert len(segment.result_sequences) == 2

    def test_10_beamsearch_to_mcmc(self):
        """BeamSearch -> MCMC: MCMC refines BeamSearch results."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_beamsearch_optimizer(construct, segment, num_results=3, beam_length=10, prompt="TTTT")
        opt2 = create_mcmc_optimizer(construct, segment, num_results=2, num_steps=5)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        beam_seqs = [s.sequence for s in segment.result_sequences]
        assert all(s.startswith("TTTT") for s in beam_seqs)

        program.run_stage(1)
        assert len(segment.result_sequences) == 2

    def test_11_beamsearch_to_beamsearch(self):
        """BeamSearch -> BeamSearch: Second ignores first, starts from its own prompt."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_beamsearch_optimizer(construct, segment, num_results=2, beam_length=10, prompt="AAAA")
        opt2 = create_beamsearch_optimizer(construct, segment, num_results=2, beam_length=10, prompt="TTTT")

        program = Program(optimizers=[opt1, opt2], num_results=2)

        program.run_stage(0)
        beam1_seqs = [s.sequence for s in segment.result_sequences]
        assert all(s.startswith("AAAA") for s in beam1_seqs)

        program.run_stage(1)
        beam2_seqs = [s.sequence for s in segment.result_sequences]
        # Second BeamSearch uses its own prompt, ignores first
        assert all(s.startswith("TTTT") for s in beam2_seqs)

    def test_12_beamsearch_to_cycling(self):
        """BeamSearch -> CyclingOptimizer: Cycling uses BeamSearch results."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_beamsearch_optimizer(construct, segment, num_results=3, beam_length=10, prompt="GGGG")
        opt2 = create_cycling_optimizer(construct, segment, num_results=2, num_steps=3)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        beam_seqs = [s.sequence for s in segment.result_sequences]
        assert all(s.startswith("GGGG") for s in beam_seqs)

        program.run_stage(1)
        assert len(segment.result_sequences) == 2


class TestCyclingOptimizerTransitions:
    """Tests 13-16: Transitions FROM CyclingOptimizer."""

    def test_13_cycling_to_topk(self):
        """CyclingOptimizer -> TopK: TopK uses Cycling's results."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_cycling_optimizer(construct, segment, num_results=3, num_steps=5)
        opt2 = create_topk_optimizer(construct, segment, num_samples=10, num_results=2)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        cycling_seqs = [s.sequence for s in segment.result_sequences]
        assert len(cycling_seqs) == 3

        program.run_stage(1)
        assert len(segment.result_sequences) == 2

    def test_14_cycling_to_mcmc(self):
        """CyclingOptimizer -> MCMC: MCMC refines Cycling's results."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_cycling_optimizer(construct, segment, num_results=3, num_steps=5)
        opt2 = create_mcmc_optimizer(construct, segment, num_results=2, num_steps=5)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        program.run_stage(1)

        assert len(segment.result_sequences) == 2

    def test_15_cycling_to_beamsearch(self):
        """CyclingOptimizer -> BeamSearch: BeamSearch IGNORES Cycling's results."""
        segment = Segment(length=50, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_cycling_optimizer(construct, segment, num_results=3, num_steps=3)
        opt2 = create_beamsearch_optimizer(construct, segment, num_results=2, beam_length=10, prompt="ACGT")

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        program.run_stage(1)

        beam_seqs = [s.sequence for s in segment.result_sequences]
        assert all(s.startswith("ACGT") for s in beam_seqs), "BeamSearch ignores previous state"

    def test_16_cycling_to_cycling(self):
        """CyclingOptimizer -> CyclingOptimizer: Second continues from first."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_cycling_optimizer(construct, segment, num_results=3, num_steps=3)
        opt2 = create_cycling_optimizer(construct, segment, num_results=2, num_steps=3)

        program = Program(optimizers=[opt1, opt2], num_results=3)

        program.run_stage(0)
        cycling1_seqs = [s.sequence for s in segment.result_sequences]
        assert len(cycling1_seqs) == 3

        program.run_stage(1)
        cycling2_seqs = [s.sequence for s in segment.result_sequences]
        assert len(cycling2_seqs) == 2


# =============================================================================
# Content Verification Tests
# =============================================================================


class TestSortingContent:
    """Verify optimizer sorting behavior after Program.run_stage."""

    def test_mcmc_results_not_necessarily_sorted(self):
        """MCMC results are not sorted by energy (no Program-level sort)."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_mcmc_optimizer(construct, segment, num_results=5, num_steps=20)
        program = Program(optimizers=[opt1], num_results=5)
        program.run_stage(0)

        # MCMC doesn't sort, just verify we have valid energy scores
        assert len(opt1.energy_scores) == 5
        assert all(isinstance(e, float) for e in opt1.energy_scores)

    def test_topk_energies_ascending(self):
        """TopK energy scores are always in ascending order (sorted internally)."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_topk_optimizer(construct, segment, num_samples=30, num_results=5)
        program = Program(optimizers=[opt1], num_results=5)
        program.run_stage(0)

        assert opt1.energy_scores == sorted(opt1.energy_scores)


class TestCyclingContent:
    """Verify cycling produces expected sequence patterns."""

    def test_cycling_preserves_source_sequences(self):
        """Cycling should repeat source sequences in order."""
        segment = Segment(length=30, sequence_type="dna")
        construct = Construct([segment])

        opt1 = create_topk_optimizer(construct, segment, num_samples=10, num_results=2)
        opt2 = create_mcmc_optimizer(construct, segment, num_results=5, num_steps=1)

        program = Program(optimizers=[opt1, opt2], num_results=2)

        program.run_stage(0)
        source_seqs = [s.sequence for s in segment.result_sequences]
        assert len(source_seqs) == 2

        # Manually call _initialize_sequence_pools to check pattern before MCMC runs
        opt2._initialize_sequence_pools()
        initialized = [s.sequence for s in segment.result_sequences]

        # Pattern: [0, 1, 0, 1, 0]
        assert initialized[0] == source_seqs[0]
        assert initialized[1] == source_seqs[1]
        assert initialized[2] == source_seqs[0]
        assert initialized[3] == source_seqs[1]
        assert initialized[4] == source_seqs[0]
