"""Tests for the SemigreedyMutationGenerator."""

from collections import Counter

import numpy as np
import pytest

from proto_language.language.core import Segment
from proto_language.language.generator import (
    GeneratorRegistry,
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
)
from tests.helpers.mock_structure import MockStructure

VOCAB_SIZE = 20


def _mutation_position_counts(
    config: SemigreedyMutationGeneratorConfig,
    seq: str,
    logits: np.ndarray,
    n_trials: int = 100,
    structure: MockStructure | None = None,
) -> Counter[int]:
    """Run n_trials mutations and return per-position mutation counts."""
    counts: Counter[int] = Counter()
    for seed in range(n_trials):
        segment = Segment(sequence=seq, sequence_type="protein")
        segment.proposal_sequences[0].logits = logits.copy()
        if structure is not None:
            segment.proposal_sequences[0].structure = structure
        gen = SemigreedyMutationGenerator(config)
        gen._set_program_seed(seed)
        gen.assign(segment)
        gen.sample()
        for i, (a, b) in enumerate(zip(seq, segment.proposal_sequences[0].sequence, strict=True)):
            if a != b:
                counts[i] += 1
                break
    return counts


class TestSemigreedyMutationGenerator:
    def test_exclude_current(self):
        """exclude_current=True forces a different AA; False allows re-sampling current."""
        logits = np.full((5, VOCAB_SIZE), -100.0)
        logits[:, 0] = 100.0  # overwhelming A probability

        # True: must change
        segment = Segment(sequence="AAAAA", sequence_type="protein")
        segment.proposal_sequences[0].logits = logits.copy()
        gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig(exclude_current=True))
        gen._set_program_seed(7)
        gen.assign(segment)
        gen.sample()
        assert segment.proposal_sequences[0].sequence != "AAAAA"

        # False: can re-sample same AA
        segment = Segment(sequence="AAAAA", sequence_type="protein")
        segment.proposal_sequences[0].logits = logits.copy()
        gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig(exclude_current=False))
        gen._set_program_seed(42)
        gen.assign(segment)
        gen.sample()
        assert segment.proposal_sequences[0].sequence == "AAAAA"

    def test_entropy_weighting(self):
        """Entropy weighting preferentially selects high-entropy positions."""
        logits = np.full((5, VOCAB_SIZE), -100.0)
        for i in [0, 1, 3, 4]:
            logits[i, i % VOCAB_SIZE] = 100.0
        logits[2, :] = 0.0  # position 2 uniform -> max entropy

        config = SemigreedyMutationGeneratorConfig(position_weighting="entropy", exclude_current=True)
        counts = _mutation_position_counts(config, "ACDEF", logits)
        assert counts[2] > 80

    def test_plddt_weighting(self):
        """PLDDT weighting preferentially selects low-confidence positions."""
        logits = np.random.default_rng(0).standard_normal((5, VOCAB_SIZE))
        structure = MockStructure(metrics={"per_residue_plddt": [0.1, 0.95, 0.95, 0.95, 0.95]})

        config = SemigreedyMutationGeneratorConfig(position_weighting="plddt", exclude_current=True)
        counts = _mutation_position_counts(config, "ACDEF", logits, structure=structure)
        assert counts[0] > 60

    def test_registry_and_type_rejection(self):
        """Discoverable via registry; rejects ligand and non-protein segments."""
        gen = GeneratorRegistry.create("semigreedy-mutation", {})
        assert isinstance(gen, SemigreedyMutationGenerator)

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            gen.assign(Segment(sequence="CCC", sequence_type="ligand"))
        with pytest.raises(ValueError, match="does not support sequence type"):
            gen.assign(Segment(sequence="ACGT", sequence_type="dna"))

    @pytest.mark.parametrize(
        ("logits", "error_match"),
        [
            pytest.param(None, "has no logits", id="no-logits"),
            pytest.param(np.zeros((3, 4)), "does not match expected", id="wrong-shape"),
            pytest.param(
                np.where(np.arange(60).reshape(3, VOCAB_SIZE) == 25, np.inf, 0.0),
                "finite",
                id="non-finite",
            ),
        ],
    )
    def test_logits_validation(self, logits, error_match):
        """Sampling fails fast on missing or invalid logits."""
        segment = Segment(sequence="ACD", sequence_type="protein")
        if logits is not None:
            segment.proposal_sequences[0].logits = logits
        gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig())
        gen._set_program_seed(42)
        gen.assign(segment)
        with pytest.raises((ValueError, RuntimeError), match=error_match):
            gen.sample()

    @pytest.mark.parametrize(
        ("structure", "error_match"),
        [
            pytest.param(None, "requires a Structure", id="no-structure"),
            pytest.param(MockStructure(), "per_residue_plddt", id="no-metric"),
            pytest.param(
                MockStructure(metrics={"per_residue_plddt": [0.5, 0.5]}),
                "does not match sequence length",
                id="wrong-length",
            ),
        ],
    )
    def test_plddt_validation(self, structure, error_match):
        """PLDDT weighting validates structure requirements."""
        segment = Segment(sequence="ACD", sequence_type="protein")
        segment.proposal_sequences[0].logits = np.zeros((3, VOCAB_SIZE))
        if structure is not None:
            segment.proposal_sequences[0].structure = structure
        gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig(position_weighting="plddt"))
        gen._set_program_seed(42)
        gen.assign(segment)
        with pytest.raises(ValueError, match=error_match):
            gen.sample()

    def test_logit_bias_shifts_sampling(self):
        """Additive logit_bias shifts AA sampling away from proposal.logits preference."""
        segment = Segment(sequence="AAAAA", sequence_type="protein")
        # proposal.logits: uniform (zeros)
        segment.proposal_sequences[0].logits = np.zeros((5, VOCAB_SIZE))

        # logit_bias: strongly favor L (col 9) — additive, so logits+bias peaks at L
        bias = np.zeros((5, VOCAB_SIZE)).tolist()
        for row in bias:
            row[9] = 100.0

        gen = SemigreedyMutationGenerator(
            SemigreedyMutationGeneratorConfig(exclude_current=True, logit_bias=bias),
        )
        gen._set_program_seed(42)
        gen.assign(segment)
        gen.sample()

        result = segment.proposal_sequences[0].sequence
        mutated_aa = [c for c in result if c != "A"]
        assert len(mutated_aa) == 1
        assert mutated_aa[0] == "L"

    def test_logit_bias_validation(self):
        """logit_bias with wrong shape or non-finite values raises at config validation."""
        with pytest.raises(Exception, match="logit_bias must have shape"):
            SemigreedyMutationGeneratorConfig(logit_bias=[[1.0, 2.0]])

        bias_with_inf = np.zeros((3, VOCAB_SIZE)).tolist()
        bias_with_inf[1][5] = float("inf")
        with pytest.raises(Exception, match="finite"):
            SemigreedyMutationGeneratorConfig(logit_bias=bias_with_inf)

        # Length mismatch: bias has 3 rows but segment has 5 positions — caught at assign() time
        segment = Segment(sequence="ACDEF", sequence_type="protein")
        segment.proposal_sequences[0].logits = np.zeros((5, VOCAB_SIZE))
        gen = SemigreedyMutationGenerator(
            SemigreedyMutationGeneratorConfig(logit_bias=np.zeros((3, VOCAB_SIZE)).tolist())
        )
        with pytest.raises(ValueError, match="rows but sequence length"):
            gen.assign(segment)

    def test_frozen_positions_excluded_from_selection(self):
        """Frozen positions never mutate; unfrozen positions absorb every mutation."""
        config = SemigreedyMutationGeneratorConfig(frozen_positions=[0, 2])
        counts = _mutation_position_counts(config, "ACDEF", np.zeros((5, VOCAB_SIZE)))
        assert counts[0] == 0
        assert counts[2] == 0
        assert sum(counts.values()) == 100

    @pytest.mark.parametrize("weighting", ["entropy", "plddt"])
    def test_frozen_positions_override_weighting(self, weighting):
        """Freezing the top-ranked position under a non-uniform strategy redirects mutations to the next-ranked one."""
        logits = np.full((5, VOCAB_SIZE), -100.0)
        for i in [0, 1, 3]:
            logits[i, i % VOCAB_SIZE] = 100.0  # peaked → ~0 entropy
        logits[2, :] = 0.0  # max entropy (and matching low-pLDDT slot below), frozen
        logits[4, :] = 0.0  # max entropy (and matching low-pLDDT slot below), free
        # pLDDT mirrors the entropy ranking: positions 2 and 4 have the lowest confidence.
        structure = MockStructure(metrics={"per_residue_plddt": [0.95, 0.95, 0.1, 0.95, 0.1]})

        config = SemigreedyMutationGeneratorConfig(position_weighting=weighting, frozen_positions=[2])
        counts = _mutation_position_counts(config, "ACDEF", logits, structure=structure)
        assert counts[2] == 0
        assert counts[4] > 70

    def test_frozen_positions_compose_with_exclude_current(self):
        """Frozen position stays fixed while exclude_current still forces a mutation elsewhere."""
        config = SemigreedyMutationGeneratorConfig(frozen_positions=[0], exclude_current=True)
        counts = _mutation_position_counts(config, "AAAAA", np.zeros((5, VOCAB_SIZE)), n_trials=30)
        assert counts[0] == 0
        assert sum(counts.values()) == 30

    @pytest.mark.parametrize(
        ("seq", "frozen", "match"),
        [
            pytest.param("ACDEF", [7], "sequence length is 5", id="out-of-bounds"),
            pytest.param("AC", [0, 1], "All positions are frozen", id="all-frozen"),
        ],
    )
    def test_frozen_positions_assign_errors(self, seq, frozen, match):
        """assign() raises ValueError on each invalid runtime state."""
        segment = Segment(sequence=seq, sequence_type="protein")
        segment.proposal_sequences[0].logits = np.zeros((len(seq), VOCAB_SIZE))
        gen = SemigreedyMutationGenerator(SemigreedyMutationGeneratorConfig(frozen_positions=frozen))
        with pytest.raises(ValueError, match=match):
            gen.assign(segment)

    @pytest.mark.parametrize(
        ("frozen", "match"),
        [
            pytest.param([], "must not be empty", id="empty"),
            pytest.param([-1], "non-negative", id="negative-index"),
        ],
    )
    def test_frozen_positions_config_validation(self, frozen, match):
        """Invalid frozen_positions configurations raise at construction."""
        with pytest.raises(Exception, match=match):
            SemigreedyMutationGeneratorConfig(frozen_positions=frozen)
