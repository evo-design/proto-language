"""tests/language_tests/test_sequence.py."""

import copy
import logging

import numpy as np
import pytest

from proto_language.language.core import Sequence
from proto_language.language.core.sequence import validate_smiles
from tests.helpers.mock_structure import MockStructure


class TestSequence:
    """Tests for the base Sequence class."""

    @pytest.mark.parametrize(
        "seq_type, valid_seq, invalid_char",
        [
            ("dna", "ATCG", "U"),
            ("rna", "AUCG", "T"),
            ("protein", "ACDEFGHIKLMNPQRSTVWY", "B"),
            # Note: ligands don't necessarily have invalid chars, requires more
            # specific validity tests.
        ],
    )
    def test_sequence_validation(self, seq_type, valid_seq, invalid_char, caplog):
        """Tests character validation for each sequence type."""
        # Test valid sequence
        seq = Sequence(valid_seq, seq_type)
        assert seq.sequence == valid_seq

        # Test invalid character on creation
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            Sequence(valid_seq + invalid_char, seq_type)
        assert "Invalid characters" in caplog.text
        caplog.clear()

        # Test invalid character on setter
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            seq.sequence = valid_seq + invalid_char
        assert "Invalid characters" in caplog.text

    def test_custom_validation(self, caplog):
        """Tests sequence validation with a custom character set."""
        custom_chars = {"0", "1"}
        seq = Sequence("0101", valid_chars=custom_chars)
        assert seq.sequence == "0101"
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            seq.sequence = "01012"
        assert "Invalid characters" in caplog.text

    def test_metadata(self):
        """Tests automatic and custom metadata handling."""
        seq = Sequence("ATCG", "dna", metadata={"id": "test1"})
        assert seq._metadata["id"] == "test1"
        # Identity fields are in the computed metadata property, not _metadata
        assert seq.metadata["sequence"] == "ATCG"
        assert seq.metadata["sequence_length"] == 4

        # Test metadata update on sequence change
        seq.sequence = "GATTACA"
        assert seq._metadata["id"] == "test1"  # Custom metadata preserved
        assert seq.metadata["sequence"] == "GATTACA"
        assert seq.metadata["sequence_length"] == 7

    def test_metadata_identity_fields_cannot_be_shadowed(self, caplog):
        """Identity fields in .metadata always reflect the actual sequence."""
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            seq = Sequence(
                "ATCG",
                "dna",
                metadata={"sequence": "WRONG", "sequence_length": 999},
            )
        assert "reserved keys" in caplog.text
        # Identity fields must win over user metadata
        assert seq.metadata["sequence"] == "ATCG"
        assert seq.metadata["sequence_length"] == 4
        # User data is still accessible in _metadata
        assert seq._metadata["sequence"] == "WRONG"

    def test_metadata_reserved_key_warning(self, caplog):
        """Warn when user-provided metadata contains reserved keys."""
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            Sequence("ATCG", "dna", metadata={"constraints": {}})
        assert "reserved keys" in caplog.text


class TestLigandSequence:
    """Tests for ligand (SMILES) sequences."""

    @pytest.mark.parametrize(
        "smiles, description",
        [
            ("C", "methane"),
            ("CC", "ethane"),
            ("CCO", "ethanol"),
            ("C(=O)O", "formic acid"),
            ("c1ccccc1", "benzene (aromatic)"),
            ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
            ("[Na+].[Cl-]", "sodium chloride"),
        ],
    )
    def test_valid_smiles(self, smiles, description):
        """Tests that valid SMILES strings are accepted."""
        seq = Sequence(smiles, "ligand")
        assert seq.sequence == smiles
        assert seq.sequence_type == "ligand"
        assert seq.valid_chars is None

    @pytest.mark.parametrize(
        "invalid_smiles",
        [
            "C(C(C",  # Unbalanced parentheses
            "C(=O",  # Unclosed parenthesis
            "XYZ",  # Invalid atoms
        ],
    )
    def test_invalid_smiles(self, invalid_smiles, caplog):
        """Tests that invalid SMILES strings trigger a warning."""
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            Sequence(invalid_smiles, "ligand")
        assert "could not parse SMILES" in caplog.text

    def test_smiles_setter(self, caplog):
        """Tests sequence setter with SMILES."""
        seq = Sequence("C", "ligand")
        seq.sequence = "CCO"
        assert seq.sequence == "CCO"
        assert seq.metadata["sequence"] == "CCO"

        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            seq.sequence = "invalid(("
        assert "could not parse SMILES" in caplog.text


class TestValidateSmiles:
    """Tests for the validate_smiles helper function."""

    def test_valid_smiles_returns_true(self):
        assert validate_smiles("CCO", verbose=False) is True

    def test_invalid_smiles_returns_false(self):
        assert validate_smiles("C(C(C", verbose=False) is False

    def test_invalid_smiles_warns_when_verbose(self, caplog):
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            validate_smiles("C(C(C", verbose=True)
        assert "could not parse SMILES" in caplog.text

    def test_valid_smiles_no_warning_when_verbose(self, caplog):
        with caplog.at_level(logging.WARNING, logger="proto_language.language.core.sequence"):
            validate_smiles("CCO", verbose=True)
        assert "could not parse SMILES" not in caplog.text


class TestSequenceDeepCopy:
    """Tests for optimized __deepcopy__ behavior."""

    def test_deepcopy_string_independence(self):
        """Verify that modifying a deepcopy's sequence doesn't affect the original."""
        seq1 = Sequence("ATCG", "dna")
        seq2 = copy.deepcopy(seq1)

        # Modify the copy
        seq2.sequence = "GGGG"

        # Original should be unaffected (strings are immutable, setter replaces reference)
        assert seq1.sequence == "ATCG"
        assert seq2.sequence == "GGGG"

    def test_deepcopy_metadata_independence(self):
        """Verify that modifying a deepcopy's metadata doesn't affect the original."""
        seq1 = Sequence("ATCG", "dna", metadata={"scores": [1, 2, 3]})
        seq2 = copy.deepcopy(seq1)

        # Modify nested mutable object in copy's metadata
        seq2._metadata["scores"].append(4)
        seq2._metadata["new_key"] = "new_value"

        # Original should be unaffected
        assert seq1._metadata["scores"] == [1, 2, 3]
        assert "new_key" not in seq1._metadata

    def test_deepcopy_shares_immutable_refs(self):
        """Verify that deepcopy shares immutable data for efficiency."""
        seq1 = Sequence("ATCG", "dna")
        seq2 = copy.deepcopy(seq1)

        # Immutable data should be shared (same object)
        assert seq1.valid_chars is seq2.valid_chars
        assert seq1._sequence_type is seq2._sequence_type


class TestConcatenatedSequence:
    """Tests for create_concatenated_sequence."""

    def test_concatenated_sequence_empty_input_raises(self):
        """Concatenating an empty iterable should fail with a clear error."""
        from proto_language.language.core.sequence import create_concatenated_sequence

        with pytest.raises(ValueError, match="empty sequence list"):
            create_concatenated_sequence([])

    def test_concatenated_sequence_metadata_independence(self):
        """Verify that mutating source metadata doesn't corrupt joined sequence (B6)."""
        from proto_language.language.core.sequence import create_concatenated_sequence

        seq1 = Sequence("ATCG", "dna", metadata={"nested": {"score": 0.5}})
        seq2 = Sequence("GGGG", "dna", metadata={"nested": {"score": 0.8}})

        joined = create_concatenated_sequence([seq1, seq2], ["seg1", "seg2"])

        # Mutate source metadata after concatenation
        seq1._metadata["nested"]["score"] = 999

        # Joined sequence's metadata should be independent
        assert joined._metadata["segments"]["seg1"]["nested"]["score"] == 0.5


class TestSequenceSerialization:
    """Tests for to_dict/from_dict serialization."""

    def test_to_dict_metadata_independence(self):
        """Verify that to_dict returns independent metadata (deep copied)."""
        seq = Sequence("ATCG", "dna", metadata={"scores": [1, 2, 3], "nested": {"a": 1}})
        serialized = seq.to_dict()

        # Modify the serialized metadata
        serialized["metadata"]["scores"].append(4)
        serialized["metadata"]["nested"]["a"] = 999

        # Original should be unaffected
        assert seq._metadata["scores"] == [1, 2, 3]
        assert seq._metadata["nested"]["a"] == 1

    def test_roundtrip_preserves_data(self):
        """Verify that to_dict/from_dict roundtrip preserves sequence data."""
        original = Sequence("ATCG", "dna", metadata={"custom": "value"})
        serialized = original.to_dict()
        restored = Sequence.from_dict(serialized)

        assert restored.sequence == original.sequence
        assert restored.sequence_type == original.sequence_type
        assert restored._metadata["custom"] == "value"


class TestSequenceLogits:
    """Tests for the optional logits field on Sequence."""

    def test_logits_default_none(self):
        """Logits default to None for discrete-only sequences."""
        seq = Sequence("EVQLV", "protein")
        assert seq.logits is None

    def test_logits_set_at_construction(self):
        """Logits can be provided at construction time."""
        logits = np.random.randn(5, 20)
        seq = Sequence("EVQLV", "protein", logits=logits)
        assert seq.logits is not None
        np.testing.assert_array_equal(seq.logits, logits)

    def test_logits_setter(self):
        """Logits can be set and cleared after construction."""
        seq = Sequence("EVQLV", "protein")
        logits = np.ones((5, 20))
        seq.logits = logits
        np.testing.assert_array_equal(seq.logits, logits)

        seq.logits = None
        assert seq.logits is None

    def test_logits_validates_2d(self):
        """1D logits raise ValueError at construction and via setter."""
        with pytest.raises(ValueError, match="logits must be 2D"):
            Sequence("EVQLV", "protein", logits=np.zeros(10))

        seq = Sequence("EVQLV", "protein")
        with pytest.raises(ValueError, match="logits must be 2D"):
            seq.logits = np.zeros(10)

    def test_logits_deepcopy(self):
        """Deepcopy creates an independent copy of logits."""
        logits = np.ones((5, 20))
        seq = Sequence("EVQLV", "protein", logits=logits)
        seq_copy = copy.deepcopy(seq)

        # Values match
        np.testing.assert_array_equal(seq_copy.logits, logits)

        # Independent: modifying copy doesn't affect original
        seq_copy.logits[0, 0] = 999.0
        assert seq.logits[0, 0] == 1.0

    def test_logits_deepcopy_none(self):
        """Deepcopy with logits=None preserves None."""
        seq = Sequence("EVQLV", "protein")
        seq_copy = copy.deepcopy(seq)
        assert seq_copy.logits is None

    def test_logits_serialization_omitted_by_default(self):
        """to_dict omits logits by default even when present."""
        logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        seq = Sequence("AT", "dna", logits=logits)
        assert "logits" not in seq.to_dict()

    def test_logits_serialization_roundtrip_opt_in(self):
        """to_dict(include_logits=True) preserves logits through round-trip."""
        logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        seq = Sequence("AT", "dna", logits=logits)

        serialized = seq.to_dict(include_logits=True)
        assert serialized["logits"] == logits.tolist()

        restored = Sequence.from_dict(serialized)
        np.testing.assert_array_almost_equal(restored.logits, logits)

    def test_logits_serialization_none_with_flag(self):
        """to_dict omits logits when None even with include_logits=True."""
        seq = Sequence("AT", "dna")
        assert "logits" not in seq.to_dict(include_logits=True)


class TestSequenceStructure:
    """Tests for the optional structure field on Sequence."""

    def test_structure_get_set(self):
        """Structure can be set via constructor, assignment, and cleared."""
        # Default
        seq = Sequence("EVQLV", "protein")
        assert seq.structure is None

        # Constructor
        struct = MockStructure()
        seq = Sequence("EVQLV", "protein", structure=struct)
        assert seq.structure is struct

        # Assignment and clear
        seq.structure = None
        assert seq.structure is None

    def test_structure_deepcopy_shares_reference(self):
        """Deepcopy shares the Structure reference (treated as immutable by convention)."""
        struct = MockStructure()
        seq = Sequence("EVQLV", "protein", structure=struct)
        seq_copy = copy.deepcopy(seq)
        assert seq_copy.structure is struct

        # None case
        seq2 = Sequence("EVQLV", "protein")
        assert copy.deepcopy(seq2).structure is None

    def test_structure_serialization_omitted_by_default(self):
        """to_dict omits structure by default even when present."""
        struct = MockStructure(metrics={"avg_plddt": 85.0})
        seq = Sequence("EVQLV", "protein", structure=struct)
        assert "structure" not in seq.to_dict()

    def test_structure_serialization_roundtrip_opt_in(self):
        """to_dict(include_structure=True) preserves structure through round-trip."""
        struct = MockStructure(metrics={"avg_plddt": 85.0})
        seq = Sequence("EVQLV", "protein", structure=struct)

        serialized = seq.to_dict(include_structure=True)
        restored = Sequence.from_dict(serialized)
        assert restored.structure is not None
        assert restored.structure.metrics["avg_plddt"] == 85.0

    def test_structure_serialization_none_with_flag(self):
        """to_dict omits structure when None even with include_structure=True."""
        seq = Sequence("EVQLV", "protein")
        assert "structure" not in seq.to_dict(include_structure=True)
