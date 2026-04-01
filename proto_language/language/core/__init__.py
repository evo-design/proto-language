"""Core language types: Constraint, Segment, Sequence, Construct, Program, Generator, Optimizer."""

# ruff: noqa: I001
# Import order matters: .sequence and .segment must come before .construct
# because construct.py does `from . import Segment, Sequence`.
from proto_language.base_config import BaseConfig
from proto_language.base_registry import BaseRegistry, BaseSpec

from proto_language.language.core.sequence import (
    DNA_NUCLEOTIDES,
    PROTEIN_AMINO_ACIDS,
    RNA_NUCLEOTIDES,
    Sequence,
    SequenceType,
    create_concatenated_sequence,
    detect_sequence_type,
    return_invalid_dna_chars,
    return_invalid_nucleotide_chars,
    return_invalid_protein_chars,
    return_invalid_rna_chars,
    validate_smiles,
)
from proto_language.language.core.segment import Segment
from proto_language.language.core.constraint import Constraint, ConstraintFunction
from proto_language.language.core.construct import Construct
from proto_language.language.core.generator import Generator
from proto_language.language.core.optimizer import Optimizer
from proto_language.language.core.program import Program

__all__ = [
    "BaseConfig",
    "Sequence",
    "SequenceType",
    "DNA_NUCLEOTIDES",
    "RNA_NUCLEOTIDES",
    "PROTEIN_AMINO_ACIDS",
    "return_invalid_dna_chars",
    "return_invalid_rna_chars",
    "return_invalid_nucleotide_chars",
    "return_invalid_protein_chars",
    "validate_smiles",
    "detect_sequence_type",
    "create_concatenated_sequence",
    "Segment",
    "Construct",
    "Constraint",
    "ConstraintFunction",
    "Generator",
    "Optimizer",
    "Program",
    "BaseRegistry",
    "BaseSpec",
]
