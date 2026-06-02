"""tests/language_tests/generator_tests/test_base_generator.py."""

import logging
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from proto_language import GeneratorRegistry, GeneratorSpec
from proto_language.core import (
    Generator,
    GeneratorInputType,
    Segment,
    Sequence,
)


class ConcreteMutationGenerator(Generator):
    """Concrete mutation generator used as the default test generator."""

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self) -> None:
        super().__init__()

    def _sample(self) -> None:
        pass


class ConcreteAutoregressiveGenerator(Generator):
    """Concrete autoregressive generator used for prompt-driven tests."""

    input_type = GeneratorInputType.PROMPT

    def __init__(self) -> None:
        super().__init__()

    def _sample(self, *args: object, **kwargs: object) -> None:
        pass


class ConcreteInverseFoldingGenerator(Generator):
    """Concrete inverse folding generator used for structure-driven tests."""

    input_type = GeneratorInputType.STRUCTURE

    def __init__(self) -> None:
        super().__init__()

    def _sample(self, *args: object, **kwargs: object) -> None:
        pass


class ConcreteLogitsGenerator(Generator):
    """Concrete logits-input generator used for logits-driven tests."""

    input_type = GeneratorInputType.LOGITS

    def __init__(self) -> None:
        super().__init__()

    def _sample(self, *args: object, **kwargs: object) -> None:
        pass


def _mock_spec(category: str = "mutation", supported_types: list[str] | None = None) -> MagicMock:
    """Create a mock GeneratorSpec with the given category."""
    mock = MagicMock(spec=GeneratorSpec)
    mock.supported_sequence_types = supported_types or []
    mock.category = category
    return mock


def _patch_registry(mock_spec: MagicMock):
    """Return a pair of patch objects for GeneratorRegistry.get and get_key."""
    return (
        patch.object(GeneratorRegistry, "get", return_value=mock_spec),
        patch.object(GeneratorRegistry, "get_key", return_value="concrete-generator"),
    )


class TestGeneratorBase:
    """Tests for the base Generator class functionality."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Tests that Generator cannot be directly instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Generator()

    def test_assign_sets_segment_and_allows_all_types_when_empty(self):
        """Tests that assign sets segment and allows any type when supported_sequence_types is empty."""
        gen = ConcreteMutationGenerator()
        assert gen._assigned_segments is None

        p_get, p_key = _patch_registry(_mock_spec())
        with p_get, p_key:
            for seq, seq_type in [("ATCG", "dna"), ("ACGU", "rna"), ("MKKL", "protein")]:
                segment = Segment(sequence=seq, sequence_type=seq_type)
                gen.assign(segment)
                assert gen._assigned_segments == (segment,)
                assert gen.segment is segment

    def test_assign_rejects_ligand_and_incompatible_type(self):
        """Tests that assign rejects ligand segments and incompatible sequence types."""
        gen = ConcreteMutationGenerator()

        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            gen.assign(Segment(sequence="CCC", sequence_type="ligand"))

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            with pytest.raises(ValueError, match="does not support sequence type"):
                gen.assign(Segment(sequence="ATCG", sequence_type="dna"))

    def test_assign_multiple_segments(self):
        """Tests that assign can record multiple target segments."""
        gen = ConcreteMutationGenerator()
        segments = [
            Segment(sequence="MKKL", sequence_type="protein"),
            Segment(sequence="MAAA", sequence_type="protein"),
        ]

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segments)

        assert gen._assigned_segments == tuple(segments)
        assert gen.segment is segments[0]
        assert gen.segments == tuple(segments)

    def test_assign_hydrates_empty_tied_segment_pools(self, caplog):
        """Length-only tied segments start from a populated tied sibling."""
        gen = ConcreteMutationGenerator()
        primary = Segment(sequence="MKKL", sequence_type="protein", label="primary")
        tied = Segment(length=4, sequence_type="protein", label="tied")

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key, caplog.at_level(logging.WARNING, logger="proto_language.core.generator"):
            gen.assign([primary, tied])

        assert tied.result_sequences[0].sequence == "MKKL"
        assert tied.proposal_sequences[0].sequence == "MKKL"
        assert tied.result_sequences[0] is not primary.result_sequences[0]
        assert tied.proposal_sequences[0] is not primary.proposal_sequences[0]
        assert "Hydrated empty tied segment 'tied' result_sequences, proposal_sequences" in caplog.text

    def test_assign_hydrates_empty_primary_from_tied_sibling(self):
        """The sampling primary is populated if a later tied segment has the seed."""
        gen = ConcreteMutationGenerator()
        primary = Segment(length=4, sequence_type="protein", label="primary")
        tied = Segment(sequence="MKKL", sequence_type="protein", label="tied")

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign([primary, tied])

        assert primary.result_sequences[0].sequence == "MKKL"
        assert primary.proposal_sequences[0].sequence == "MKKL"

    def test_assign_preserves_nonempty_tied_segment_pools(self):
        """Assignment does not overwrite distinct non-empty tied starting pools."""
        gen = ConcreteMutationGenerator()
        primary = Segment(sequence="MKKL", sequence_type="protein", label="primary")
        tied = Segment(sequence="MAAA", sequence_type="protein", label="tied")

        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign([primary, tied])

        assert tied.result_sequences[0].sequence == "MAAA"
        assert tied.proposal_sequences[0].sequence == "MAAA"

    def test_assign_multiple_segments_requires_tie_compatible_segments(self):
        """Tests that tied segments must describe the same value space."""
        gen = ConcreteMutationGenerator()

        p_get, p_key = _patch_registry(_mock_spec())
        with p_get, p_key:
            with pytest.raises(ValueError, match="different sequence types"):
                gen.assign(
                    [
                        Segment(sequence="ATCG", sequence_type="dna"),
                        Segment(sequence="ACGU", sequence_type="rna"),
                    ]
                )
            with pytest.raises(ValueError, match="different lengths"):
                gen.assign(
                    [
                        Segment(sequence="MKKL", sequence_type="protein"),
                        Segment(sequence="MAAAA", sequence_type="protein"),
                    ]
                )

    def test_assign_rejects_duplicate_segment_instances(self):
        """Reusing one Segment object as multiple tied entries is rejected (defeats tying)."""
        gen = ConcreteMutationGenerator()
        seg = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key, pytest.raises(ValueError, match="duplicate Segment instances"):
            gen.assign([seg, seg])

    def test_assign_rejects_empty_iterable(self):
        """An empty iterable must raise rather than leave the generator in a half-assigned state."""
        gen = ConcreteMutationGenerator()
        p_get, p_key = _patch_registry(_mock_spec())
        with p_get, p_key, pytest.raises(ValueError, match="at least one segment"):
            gen.assign([])
        assert not gen.is_assigned

    def test_assign_rejects_mismatched_valid_chars(self):
        """Tied segments with custom but differing ``valid_chars`` are rejected.

        Without this guard, a primary's mirror would silently overwrite a tied
        segment with characters outside that segment's allowed alphabet.
        """
        gen = ConcreteMutationGenerator()
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key, pytest.raises(ValueError, match="different valid character sets"):
            gen.assign(
                [
                    Segment(sequence="MKKL", sequence_type="protein", valid_chars={"M", "K", "L"}),
                    Segment(sequence="MAAA", sequence_type="protein", valid_chars={"M", "A"}),
                ]
            )

    def test_sample_mirrors_proposals_to_tied_segments(self):
        """``sample()`` deep-copies primary proposals onto every tied segment."""
        from proto_language.core import Sequence

        gen = ConcreteMutationGenerator()
        segments = [
            Segment(sequence="MKKL", sequence_type="protein"),
            Segment(sequence="MAAA", sequence_type="protein"),
            Segment(sequence="MGGG", sequence_type="protein"),
        ]
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segments)

        # _sample is a no-op; plant a primary proposal so we observe the mirror.
        segments[0].proposal_sequences = [Sequence(sequence="WWWW", sequence_type="protein")]
        segments[1].proposal_sequences = [Sequence(sequence="ZZZZ", sequence_type="protein")]
        segments[2].proposal_sequences = [Sequence(sequence="QQQQ", sequence_type="protein")]

        gen.sample()

        assert segments[1].proposal_sequences[0].sequence == "WWWW"
        assert segments[2].proposal_sequences[0].sequence == "WWWW"
        # Deep copies, not aliases.
        assert segments[1].proposal_sequences[0] is not segments[0].proposal_sequences[0]
        assert segments[2].proposal_sequences[0] is not segments[0].proposal_sequences[0]

    def test_sample_is_noop_mirror_for_single_segment(self):
        """Single-segment ``sample()`` skips the deepcopy path."""
        from proto_language.core import Sequence

        gen = ConcreteMutationGenerator()
        segment = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segment)

        original = Sequence(sequence="WWWW", sequence_type="protein")
        segment.proposal_sequences = [original]
        gen.sample()
        # Same instance — no copying happened.
        assert segment.proposal_sequences[0] is original

    def test_mutation_generator_raises_on_empty_starting_sequence(self):
        """Mutation generators must raise when ``segment.proposal_sequences[].sequence`` is empty."""
        gen = ConcreteMutationGenerator()
        segment = Segment(length=20, sequence_type="dna", label="binder")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["dna"]))
        with p_get, p_key:
            gen.assign(segment)
        # Length-only segment with no starting sequence — proposals exist but have empty sequences.
        segment.proposal_sequences = [Sequence(sequence="", sequence_type="dna")]

        with pytest.raises(RuntimeError, match="requires a starting sequence"):
            gen._validate_generator()

    def test_sample_clears_stale_logits_for_non_logits_generators(self):
        """Non-LOGITS generators must clear ``proposal.logits`` since the new sequence makes them stale."""
        import numpy as np

        from proto_language.core import Sequence

        gen = ConcreteMutationGenerator()  # input_type == STARTING_SEQUENCE
        segment = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segment)
        proposal = Sequence(sequence="MAAA", sequence_type="protein")
        proposal.logits = np.zeros((4, 20))
        segment.proposal_sequences = [proposal]

        gen.sample()

        assert segment.proposal_sequences[0].logits is None

    def test_sample_preserves_logits_for_logits_generators(self):
        """LOGITS-input generators are the producers of logits — ``sample()`` must not clear them."""
        import numpy as np

        from proto_language.core import Sequence

        gen = ConcreteLogitsGenerator()  # input_type == LOGITS
        segment = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segment)
        proposal = Sequence(sequence="MAAA", sequence_type="protein")
        kept_logits = np.zeros((4, 20))
        proposal.logits = kept_logits
        segment.proposal_sequences = [proposal]

        gen.sample()

        assert segment.proposal_sequences[0].logits is kept_logits

    def test_sample_clears_stale_structure_for_non_structure_generators(self):
        """Non-STRUCTURE generators must clear ``proposal.structure`` since the new sequence makes it stale."""
        from proto_language.core import Sequence
        from tests.helpers.mock_structure import MockStructure

        gen = ConcreteMutationGenerator()  # input_type == STARTING_SEQUENCE
        segment = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segment)
        proposal = Sequence(sequence="MAAA", sequence_type="protein")
        proposal.structure = MockStructure()
        segment.proposal_sequences = [proposal]

        gen.sample()

        assert segment.proposal_sequences[0].structure is None

    def test_sample_preserves_structure_for_structure_generators(self):
        """STRUCTURE-input generators tag proposals with the structure they were designed for — keep it."""
        from proto_language.core import Sequence
        from tests.helpers.mock_structure import MockStructure

        gen = ConcreteInverseFoldingGenerator()  # input_type == STRUCTURE
        segment = Segment(sequence="MKKL", sequence_type="protein")
        p_get, p_key = _patch_registry(_mock_spec(supported_types=["protein"]))
        with p_get, p_key:
            gen.assign(segment)
        proposal = Sequence(sequence="MAAA", sequence_type="protein")
        kept_structure = MockStructure()
        proposal.structure = kept_structure
        segment.proposal_sequences = [proposal]

        gen.sample()

        assert segment.proposal_sequences[0].structure is kept_structure

    def test_validate_generator_empty_proposal_pool_raises(self):
        """Tests that _validate_generator raises on empty proposal_sequences."""
        from proto_tools.transforms.masking import MaskingStrategy

        from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig

        gen = RandomNucleotideGenerator(
            RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
        )
        segment = Segment(sequence="ATCG", sequence_type="dna")
        gen.assign(segment)
        segment.proposal_sequences = []

        with pytest.raises(RuntimeError, match="empty proposal_sequences pool"):
            gen._validate_generator()


class TestGeneratorRegistry:
    """Tests for GeneratorRegistry functionality."""

    def test_unknown_key_raises(self):
        """Tests that get_key and create raise for unregistered generators."""
        with pytest.raises(ValueError, match="is not registered"):
            GeneratorRegistry.get_key(ConcreteMutationGenerator())
        with pytest.raises(ValueError, match="Unknown generator"):
            GeneratorRegistry.create("nonexistent-generator", {})

    def test_list_all_returns_valid_specs(self):
        """Tests that all specs have required fields."""
        all_specs = GeneratorRegistry.list_all()
        assert len(all_specs) > 0

        for spec in all_specs:
            assert isinstance(spec, GeneratorSpec)
            for attr in (
                "key",
                "label",
                "description",
                "category",
                "input_type",
                "allows_empty_starting_sequence",
                "uses_gpu",
                "supported_sequence_types",
            ):
                assert hasattr(spec, attr)

    def test_random_generator_specs_allow_empty_starting_sequence(self):
        """Only random generators advertise length-only starting-sequence initialization."""
        assert GeneratorRegistry.get("random-nucleotide").allows_empty_starting_sequence
        assert GeneratorRegistry.get("random-protein").allows_empty_starting_sequence
        assert not GeneratorRegistry.get("semigreedy-mutation").allows_empty_starting_sequence

    def test_register_rejects_missing_input_type(self):
        """Decorator must reject classes that omit ``input_type``."""
        from pydantic import BaseModel

        from proto_language import generator

        class _Cfg(BaseModel):
            pass

        with pytest.raises(TypeError, match="must declare an ``input_type`` classvar"):

            @generator(key="missing-input-type", label="x", config=_Cfg, description="x")
            class _NoInputType(Generator):
                def __init__(self):
                    super().__init__()

                def _sample(self):
                    pass


class TestShortSequenceWarning:
    """Tests for the warning emitted from sample() when autoregressive output is shorter than target."""

    GENERATOR_LOGGER = "proto_language.core.generator"

    _CONCRETE_BY_CATEGORY: ClassVar[dict[str, type[Generator]]] = {
        "autoregressive": ConcreteAutoregressiveGenerator,
        "mutation": ConcreteMutationGenerator,
        "inverse_folding": ConcreteInverseFoldingGenerator,
    }

    def _assigned(self, category: str = "autoregressive", num_proposals: int = 3) -> tuple[Generator, Segment]:
        gen = self._CONCRETE_BY_CATEGORY[category]()
        segment = Segment(length=50, sequence_type="protein", label="binder")
        p_get, p_key = _patch_registry(_mock_spec(category=category))
        with p_get, p_key:
            gen.assign(segment)
        segment.proposal_sequences = [Sequence(sequence="", sequence_type="protein") for _ in range(num_proposals)]
        return gen, segment

    def _run(self, gen, target_segment, lengths: list[int], caplog) -> list[logging.LogRecord]:
        def fake_sample(*args, **kwargs) -> None:
            for proposal, length in zip(target_segment.proposal_sequences, lengths, strict=True):
                proposal.sequence = "M" * length

        gen._sample = fake_sample  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger=self.GENERATOR_LOGGER):
            gen.sample()
        return [r for r in caplog.records if "candidates shorter than target_length" in r.getMessage()]

    def test_warns_with_all_candidate_lengths(self, caplog):
        gen, segment = self._assigned()
        records = self._run(gen, segment, [50, 30, 40], caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        msg = records[0].getMessage()
        for fragment in (
            "ConcreteAutoregressiveGenerator",
            "'binder'",
            "target_length=50 aa",  # protein segment → aa unit
            "candidate #0: 50 aa",  # full-length candidate is shown too
            "candidate #1: 30 aa",
            "candidate #2: 40 aa",
            "end-of-sequence",
        ):
            assert fragment in msg, f"missing {fragment!r} in {msg!r}"

    @pytest.mark.parametrize(
        "category, lengths",
        [
            ("autoregressive", [50, 50, 50]),  # gate passes but no proposals are short
            ("mutation", [30, 30, 30]),  # input_type gated out
            ("inverse_folding", [30, 30, 30]),  # input_type gated out
        ],
    )
    def test_no_warning_when_gate_or_length_check_short_circuits(self, caplog, category, lengths):
        gen, segment = self._assigned(category=category)
        assert not self._run(gen, segment, lengths, caplog)

    def test_warning_fires_once_and_mirrors_to_tied_segments(self, caplog):
        gen = ConcreteAutoregressiveGenerator()
        primary = Segment(length=50, sequence_type="protein", label="primary")
        tied = Segment(length=50, sequence_type="protein", label="tied")
        p_get, p_key = _patch_registry(_mock_spec(category="autoregressive", supported_types=["protein"]))
        with p_get, p_key:
            gen.assign([primary, tied])
        primary.proposal_sequences = [Sequence(sequence="", sequence_type="protein")]
        assert len(self._run(gen, primary, [30], caplog)) == 1
        assert tied.proposal_sequences[0].sequence == "M" * 30
