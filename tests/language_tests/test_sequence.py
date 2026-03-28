"""tests/language_tests/test_sequence.py"""
import copy
import warnings

import pytest

from proto_language.language.core import Sequence
from proto_language.language.core.sequence import validate_smiles


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
    def test_sequence_validation(self, seq_type, valid_seq, invalid_char):
        """Tests character validation for each sequence type."""
        # Test valid sequence
        seq = Sequence(valid_seq, seq_type)
        assert seq.sequence == valid_seq

        # Test invalid character on creation
        with pytest.warns(UserWarning):
            Sequence(valid_seq + invalid_char, seq_type)

        # Test invalid character on setter
        with pytest.warns(UserWarning):
            seq.sequence = valid_seq + invalid_char

    def test_custom_validation(self):
        """Tests sequence validation with a custom character set."""
        custom_chars = {"0", "1"}
        seq = Sequence("0101", valid_chars=custom_chars)
        assert seq.sequence == "0101"
        with pytest.warns(UserWarning):
            seq.sequence = "01012"

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

    def test_metadata_identity_fields_cannot_be_shadowed(self):
        """Identity fields in .metadata always reflect the actual sequence."""
        with pytest.warns(UserWarning, match="reserved keys"):
            seq = Sequence(
                "ATCG", "dna",
                metadata={"sequence": "WRONG", "sequence_length": 999},
            )
        # Identity fields must win over user metadata
        assert seq.metadata["sequence"] == "ATCG"
        assert seq.metadata["sequence_length"] == 4
        # User data is still accessible in _metadata
        assert seq._metadata["sequence"] == "WRONG"

    def test_metadata_reserved_key_warning(self):
        """Warn when user-provided metadata contains reserved keys."""
        with pytest.warns(UserWarning, match="reserved keys"):
            Sequence("ATCG", "dna", metadata={"constraints": {}})


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
            "C(C(C",        # Unbalanced parentheses
            "C(=O",         # Unclosed parenthesis
            "XYZ",          # Invalid atoms
        ],
    )
    def test_invalid_smiles(self, invalid_smiles):
        """Tests that invalid SMILES strings trigger a warning."""
        with pytest.warns(UserWarning, match="RDKit could not parse SMILES"):
            Sequence(invalid_smiles, "ligand")

    def test_smiles_setter(self):
        """Tests sequence setter with SMILES."""
        seq = Sequence("C", "ligand")
        seq.sequence = "CCO"
        assert seq.sequence == "CCO"
        assert seq.metadata["sequence"] == "CCO"

        with pytest.warns(UserWarning, match="RDKit could not parse SMILES"):
            seq.sequence = "invalid(("


class TestValidateSmiles:
    """Tests for the validate_smiles helper function."""

    def test_valid_smiles_returns_true(self):
        assert validate_smiles("CCO", verbose=False) is True

    def test_invalid_smiles_returns_false(self):
        assert validate_smiles("C(C(C", verbose=False) is False

    def test_invalid_smiles_warns_when_verbose(self):
        with pytest.warns(UserWarning, match="RDKit could not parse SMILES"):
            validate_smiles("C(C(C", verbose=True)

    def test_valid_smiles_no_warning_when_verbose(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_smiles("CCO", verbose=True)  # Should not raise


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
