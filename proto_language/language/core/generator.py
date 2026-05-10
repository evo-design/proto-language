"""Provides the abstract interface for sequence generation algorithms."""

import copy
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from proto_language.language.core.segment import Segment

logger = logging.getLogger(__name__)


class Generator(ABC):
    """Generator base class that modifies proposal_sequences of assigned segments during optimization.

    Mutation generators can operate on length-only segments by first seeding
    random starting sequences in ``_validate_generator``. That fallback provides
    initial proposal sequences for the mutation step.

    A generator may be assigned one segment or a tuple of "tied" segments that
    share the same generated value (e.g. protomers of a symmetric homo-oligomer).
    Subclasses implement ``_sample()`` writing to ``self.segment.proposal_sequences``;
    the public ``sample()`` orchestrator calls ``_sample()`` then mirrors those
    proposals to any tied segments via deepcopy.

    Subclasses must implement ``__init__()`` and ``_sample()``. Override
    ``assign()`` only to add extra validation (call ``super().assign()`` first).

    Attributes:
        batch_size (int): Number of sequences to generate per batch.
    """

    batch_size: int = 1  # GPU generators override to higher values

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the generator with configuration parameters."""
        self._assigned_segments: tuple[Segment, ...] | None = None
        self.__spec: "GeneratorSpec | None" = None  # type: ignore[name-defined]  # noqa: F821, UP037 -- circular import; lazy-loaded via property
        self._rng: random.Random | None = None

    # Required lazy loading for mock generators to function in tests.
    @property
    def _spec(self) -> "GeneratorSpec":  # type: ignore[name-defined]  # noqa: F821 -- circular import; resolved at runtime
        """Lazy-load the generator spec from the registry."""
        if self.__spec is None:
            from proto_language.language.generator.generator_registry import (
                GeneratorRegistry,
            )

            self.__spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self))
        return self.__spec

    @property
    def is_assigned(self) -> bool:
        """Whether the generator has been assigned at least one segment."""
        return self._assigned_segments is not None

    @property
    def segment(self) -> Segment:
        """The primary assigned segment (``segments[0]``). Raises if not assigned."""
        return self.segments[0]

    @property
    def segments(self) -> tuple[Segment, ...]:
        """All assigned segments (primary plus any tied segments). Raises if not assigned."""
        if self._assigned_segments is None:
            raise RuntimeError(f"{self.__class__.__name__} has no assigned segment. Call assign() first.")
        return self._assigned_segments

    def assign(self, segments: Segment | Iterable[Segment]) -> None:
        """Assign one or more tied Segments to the generator.

        Tied segments share generated values and must agree on ``sequence_type``,
        ``sequence_length``, and ``valid_chars``. Subclasses overriding this must
        call ``super().assign(segments)`` first.

        Args:
            segments (Segment | Iterable[Segment]): A single segment or an
                iterable of tied segments to receive generated sequences.

        Raises:
            ValueError: If the iterable is empty, contains duplicate Segment
                instances, any segment is a ligand, any sequence type is
                unsupported, or tied segments disagree on type, length, or
                valid characters.
        """
        normalized = (segments,) if isinstance(segments, Segment) else tuple(segments)
        if not normalized:
            raise ValueError(f"Generator {self.__class__.__name__} must be assigned at least one segment.")
        if len({id(s) for s in normalized}) != len(normalized):
            raise ValueError(
                f"Generator {self.__class__.__name__} cannot tie duplicate Segment instances; "
                f"each tied segment must be a distinct object."
            )

        for segment in normalized:
            if segment.is_ligand:
                raise ValueError(
                    f"Cannot assign generator to ligand segment '{segment.label}'. Ligand segments cannot be mutated."
                )

        supported_types = self._spec.supported_sequence_types
        for segment in normalized:
            if supported_types and segment.sequence_type not in supported_types:
                supported_types_str = ", ".join(supported_types)
                raise ValueError(
                    f"Generator {self.__class__.__name__} does not support sequence type '{segment.sequence_type}'. Supported types: [{supported_types_str}]"
                )

        primary = normalized[0]
        for segment in normalized[1:]:
            if segment.sequence_type != primary.sequence_type:
                raise ValueError(
                    f"Generator {self.__class__.__name__} cannot tie segments with different sequence types: "
                    f"{primary.sequence_type!r} and {segment.sequence_type!r}."
                )
            if segment.sequence_length != primary.sequence_length:
                raise ValueError(
                    f"Generator {self.__class__.__name__} cannot tie segments with different lengths: "
                    f"{primary.sequence_length} and {segment.sequence_length}."
                )
            if segment.valid_chars != primary.valid_chars:
                raise ValueError(
                    f"Generator {self.__class__.__name__} cannot tie segments with different valid character sets."
                )

        self._assigned_segments = normalized
        logger.debug(
            "Generator.assign: %s -> segments=%s",
            self.__class__.__name__,
            [segment.label for segment in normalized],
        )

    def sample(self, *args: Any, **kwargs: Any) -> None:
        """Run ``_sample()``, warn on short autoregressive output, then mirror primary proposals to tied segments."""
        self._sample(*args, **kwargs)
        primary = self.segments[0]

        if self._spec.category == "autoregressive":
            target = primary.sequence_length
            lengths = [len(p.sequence) for p in primary.proposal_sequences]
            if any(length < target for length in lengths):
                unit = {"protein": " aa", "dna": " bp", "rna": " nt"}.get(primary.sequence_type, "")
                candidates = ", ".join(f"candidate #{i}: {length}{unit}" for i, length in enumerate(lengths))
                logger.warning(
                    "%s: some candidates shorter than target_length=%d%s for segment %r. All candidates: %s. "
                    "The model emitted an end-of-sequence token before reaching the target length.",
                    self.__class__.__name__,
                    target,
                    unit,
                    primary.label or "unlabeled",
                    candidates,
                )

        if len(self.segments) > 1:
            for segment in self.segments[1:]:
                segment.proposal_sequences = [copy.deepcopy(sequence) for sequence in primary.proposal_sequences]

    @abstractmethod
    def _sample(self, *args: Any, **kwargs: Any) -> None:
        """Subclass hook: write proposals to ``self.segment.proposal_sequences`` in-place.

        Subclasses define their own typed signature (e.g. autoregressive generators
        take ``prompts``, ``num_tokens``, …); ``sample()`` forwards args here.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the _sample() method.")

    def _set_program_seed(self, seed: int | None) -> None:
        """Set or clear the program-derived seed stream."""
        self._rng = None if seed is None else random.Random(seed)  # noqa: S311 -- non-cryptographic

    def _next_seed(self) -> int | None:
        """Return an advancing per-call seed, or None if unseeded."""
        if self._rng is None:
            return None
        return self._rng.randint(0, 2**31 - 1)

    def _validate_generator(self) -> None:
        """Validate the primary segment and lazy-init its proposals if needed.

        Only the primary segment is validated/seeded; ``sample()`` mirrors its
        proposals onto any tied segments afterward, so tied segments don't
        need their own initialization here.
        """
        segment = self.segment  # raises RuntimeError if not assigned

        if not segment.proposal_sequences:
            raise RuntimeError(f"Segment '{segment.label or 'unlabeled'}' has an empty proposal_sequences pool.")

        # Warn if the segment already has populated sequences that will be overwritten (autoregressive only)
        if self._spec.category == "autoregressive" and segment.proposals_populated:
            logger.warning(
                "Segment %r input sequence will be overwritten by autoregressive generator %s",
                segment.label or "unlabeled",
                self.__class__.__name__,
            )

        # Lazy-init random starting sequences for mutation generators if no input template was provided.
        if self._spec.category == "mutation" and not segment.proposals_populated:
            logger.warning(
                "Mutation generator %s has no input proposals; seeding %d random starting sequences",
                self.__class__.__name__,
                len(segment.proposal_sequences),
            )
            assert segment.valid_chars is not None  # noqa: S101 -- mypy type narrowing
            valid_chars = list(segment.valid_chars - set(" "))
            rng = self._rng or random.Random()  # noqa: S311 -- non-cryptographic
            for sequence in segment.proposal_sequences:
                random_sequence = "".join(rng.choice(valid_chars) for _ in range(segment.sequence_length))
                sequence.sequence = random_sequence

        # Lazy-init unknown (X) sequences for inverse folding generators if no input sequence was provided.
        if self._spec.category == "inverse_folding" and not segment.proposals_populated:
            unknown_sequence = "X" * segment.sequence_length
            for sequence in segment.proposal_sequences:
                sequence.sequence = unknown_sequence

        logger.debug(
            "Generator validated: %s, category=%s, segments=%d",
            self.__class__.__name__,
            self._spec.category,
            len(self.segments),
        )
