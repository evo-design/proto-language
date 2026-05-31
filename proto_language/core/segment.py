"""Segment: one design region of a Construct, with a dual pool of candidate Sequences.

A Segment is the unit an optimizer works on. It carries two pools of
:class:`~proto_language.core.sequence.Sequence` objects: ``proposal_sequences``, the work space the
optimizer's generator fills with candidate mutations/offspring each iteration, and
``result_sequences``, the surviving best that constraints have selected and that the user reads out.
Create one length-only (``Segment(length=20)``) to design from scratch, or sequence-seeded
(``Segment(sequence="TATA")``) to optimize from a starting point; the type is fixed at construction via
``sequence_type`` ("dna", "rna", "protein", or "ligand"). A generator is bound to the Segment with the
generator's ``assign`` method, after which the optimizer proposes onto, scores, and selects within it.

Examples:
    Seed a region or design one from scratch, then bind a generator to it:
    >>> from proto_language.core import Segment
    >>> from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
    >>> region = Segment(length=20, sequence_type="dna", label="variable")
    >>> region.sequence_length  # 20
    >>> region.num_proposals  # 1 (proposal pool seeded with one empty Sequence)
    >>> gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    >>> gen.assign(region)  # generator now proposes onto this Segment's proposal pool

    After a run, the surviving best live in the result pool (proposal_sequences is scratch space):
    >>> promoter = Segment(sequence="TATA", sequence_type="dna", label="promoter")
    >>> promoter.result_sequences[0].sequence  # "TATA"
    >>> promoter.has_original_sequence  # True
"""

import copy
import logging
from collections.abc import Iterator
from typing import Any

from proto_language.core.sequence import Sequence, SequenceType

logger = logging.getLogger(__name__)


class Segment:
    """Building block for biological constructs with two sequence pools: proposal (work space) and result (results space):.

    - proposal_sequences: Working space for optimizer proposals (mutations, offspring, rollouts)
    - result_sequences: Results space containing current best sequences (user-facing)

    Examples:
        Creating a Segment with a sequence:
        >>> promoter = Segment(sequence="TATA", sequence_type="dna", label="promoter")
        >>> promoter.label  # "promoter"
        >>> promoter.sequence_length  # 4 (inferred from sequence)
        >>> promoter.result_sequences  # [Sequence("TATA")]

        Creating a Segment with just a length:
        >>> variable_region = Segment(length=100, sequence_type="dna", label="variable")
        >>> variable_region.sequence_length  # 100
        >>> variable_region.result_sequences  # [Sequence("")]
    """

    def __init__(
        self,
        sequence: str | None = None,
        length: int | None = None,
        sequence_type: SequenceType = "dna",
        valid_chars: set[str] | frozenset[str] | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Segment with dual sequence pools.

        Args:
            sequence (str | None): Optional biological sequence string. If provided, length is inferred.
            length (int | None): Optional desired length for sequences. Required if sequence not provided.
            sequence_type (SequenceType): Type of biological sequence ("dna", "rna", "protein", or "ligand"). Defaults to "dna".
            valid_chars (set[str] | frozenset[str] | None): Optional custom set of valid characters for sequence validation.
            label (str | None): Optional label for this segment (e.g., "promoter", "coding_region").
            metadata (dict[str, Any] | None): Additional data associated with this sequence.

        Raises:
            ValueError: If both sequence and length are provided, if neither is provided,
                or if a ligand segment is created with only a length (ligands require a sequence).
        """
        # Exactly one of sequence or length must be provided
        if sequence is None and length is None:
            raise ValueError("Must provide either 'sequence' or 'length'")
        if sequence is not None and length is not None:
            raise ValueError("Cannot provide both 'sequence' and 'length' - choose one")

        # Ligand segments must be initialized with a sequence (SMILES string), not just a length
        if sequence_type == "ligand" and sequence is None:
            raise ValueError("Ligand segments must be initialized with a sequence (SMILES string), not just a length")

        # Length must be positive
        if length is not None and length <= 0:
            raise ValueError(f"Segment length must be positive, got {length}")

        # If sequence is provided - set sequence_length and initial_sequence
        if sequence is not None:
            initial_sequence = sequence
            self.sequence_length = len(sequence)

        # If length is provided - set sequence_length accordingly and initial_sequence to empty
        else:
            initial_sequence = ""
            assert length is not None  # noqa: S101 -- mypy type narrowing
            self.sequence_length = length

        # Original sequence is read-only after construction
        self._original_sequence: Sequence = Sequence(
            sequence=initial_sequence,
            sequence_type=sequence_type,
            metadata=metadata,
            valid_chars=valid_chars,
        )
        # Dual pools: proposals (work space) and result (results space)
        # These are deep copies so modifications don't affect original_sequence
        self.proposal_sequences: list[Sequence] = [copy.deepcopy(self._original_sequence)]
        self.result_sequences: list[Sequence] = [copy.deepcopy(self._original_sequence)]

        self.label: str | None = label
        self.construct_label: str | None = None  # Set by Program for metadata tracking
        logger.debug(f"Created Segment: label={label}, type={sequence_type}, length={self.sequence_length}")

    @property
    def sequence_type(self) -> SequenceType:
        """Sequence type derived from original sequence (read-only)."""
        return self._original_sequence.sequence_type

    @property
    def valid_chars(self) -> set[str] | frozenset[str] | None:
        """Valid characters derived from original sequence (read-only)."""
        return self._original_sequence.valid_chars

    @property
    def num_results(self) -> int:
        """Number of sequences in result pool (solution space)."""
        return len(self.result_sequences)

    @property
    def num_proposals(self) -> int:
        """Number of sequences in proposal pool (proposal space)."""
        return len(self.proposal_sequences)

    @property
    def original_sequence(self) -> Sequence:
        """Original sequence (read-only). Preserves user intent for serialization."""
        return self._original_sequence

    @property
    def has_original_sequence(self) -> bool:
        """Whether segment was created with a sequence (vs just a length)."""
        return bool(self._original_sequence.sequence)

    @property
    def populated_sequences(self) -> bool:
        """Whether segment has sequences from original input or previous optimization.

        Only checks original sequence (original user input) and result sequences (previous optimization results).
        Proposal sequences are not considered because they the staging area for optimizations.
        """
        return bool(self._original_sequence.sequence or (self.result_sequences and self.result_sequences[0].sequence))

    @property
    def proposals_populated(self) -> bool:
        """Whether all proposal sequences have actual sequences (not empty)."""
        return all(bool(seq.sequence) for seq in self.proposal_sequences)

    @property
    def is_ligand(self) -> bool:
        """Whether this segment is a ligand (ligands cannot be mutated by generators)."""
        return self.sequence_type == "ligand"

    def ordered_vocab(self) -> list[str]:
        """Canonical alphabet for the segment's type, intersected with ``valid_chars``.

        Canonical order is preserved; any custom ``valid_chars`` outside the canonical
        alphabet are appended alphabetically. Raises ``ValueError`` for ligands.
        """
        if self.is_ligand:
            raise ValueError(f"Segment '{self.label or 'unlabeled'}' is a ligand; no fixed vocab.")
        return self._original_sequence.ordered_vocab()

    def __iter__(self) -> Iterator[Sequence]:
        """Iterate over result sequences (user-facing results)."""
        return iter(self.result_sequences)

    def __getitem__(self, index: int) -> Sequence:
        """Index into result sequences (user-facing results)."""
        return self.result_sequences[index]

    def to_dict(self, *, include_logits: bool = False, include_structure: bool = False) -> dict[str, Any]:
        """Serialize Segment to a JSON-safe dictionary."""
        return {
            "original_sequence": self.original_sequence.to_dict(
                include_logits=include_logits, include_structure=include_structure
            ),
            "sequence_length": self.sequence_length,
            "proposal_sequences": [
                seq.to_dict(include_logits=include_logits, include_structure=include_structure)
                for seq in self.proposal_sequences
            ],
            "result_sequences": [
                seq.to_dict(include_logits=include_logits, include_structure=include_structure)
                for seq in self.result_sequences
            ],
            "sequence_type": self.sequence_type,
            "valid_chars": list(self.valid_chars) if self.valid_chars else None,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        """Deserialize Segment from dictionary."""
        # Reconstruct original sequence
        original_seq = Sequence.from_dict(data["original_sequence"])

        # Use input sequence if available, otherwise use length
        segment = cls(
            sequence=original_seq.sequence or None,
            length=data["sequence_length"] if not original_seq.sequence else None,
            sequence_type=data["sequence_type"],
            valid_chars=set(data["valid_chars"]) if data.get("valid_chars") else None,
            label=data.get("label"),
            metadata=original_seq._metadata or None,
        )

        # Restore sequence pools
        segment.proposal_sequences = [Sequence.from_dict(seq_data) for seq_data in data["proposal_sequences"]]
        segment.result_sequences = [Sequence.from_dict(seq_data) for seq_data in data["result_sequences"]]

        return segment
