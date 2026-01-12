import pytest
import warnings

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
        assert seq._metadata["sequence"] == "ATCG"
        assert seq._metadata["sequence_length"] == 4

        # Test metadata update on sequence change
        seq.sequence = "GATTACA"
        assert seq._metadata["id"] == "test1"  # Custom metadata preserved
        assert seq._metadata["sequence"] == "GATTACA"
        assert seq._metadata["sequence_length"] == 7


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
        assert seq._valid_chars is None

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
        assert seq._metadata["sequence"] == "CCO"

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
