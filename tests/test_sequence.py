import pytest

import sys
sys.path.append('.')
from language.sequence import ProgramDNASequence, ProgramRNASequence, ProgramProteinSequence
from language.base import ProgramGenerator

# Create a dummy generator for testing purposes.
class DummyGenerator(ProgramGenerator):
    def register(self, outputs=None):
        pass
    def sample(self):
        pass

dummy_gen = DummyGenerator()

########################
## DNA Sequence Tests ##
########################

def test_dna_sequence_creation_valid():
    """Tests successful creation of a ProgramDNASequence with valid characters."""
    valid_dna = "ATGCGATCGTAGCTAGCTAG"
    # Use a MagicMock or a simple dummy object if DummyGenerator is too complex
    seq = ProgramDNASequence(generator=dummy_gen, generator_output_idx=0, sequence=valid_dna)
    assert seq.sequence == valid_dna
    assert len(seq) == len(valid_dna)
    assert str(seq) == valid_dna

def test_dna_sequence_creation_invalid_chars():
    """Tests that ValueError is raised for ProgramDNASequence with invalid characters."""
    invalid_dna = "ATGCGATCGUAGCTAGCTAG" # Contains 'U'
    with pytest.raises(ValueError) as excinfo:
        ProgramDNASequence(generator=dummy_gen, generator_output_idx=0, sequence=invalid_dna)
    assert "Invalid characters found: " in str(excinfo.value)

def test_dna_sequence_setter_invalid():
    """Tests that ValueError is raised when setting invalid DNA sequence."""
    seq = ProgramDNASequence(generator=dummy_gen, generator_output_idx=0, sequence="AAAA")
    with pytest.raises(ValueError) as excinfo:
        seq.sequence = "AAAUAAA" # Invalid 'U'
    assert "Invalid characters found: " in str(excinfo.value)

########################
## RNA Sequence Tests ##
########################

def test_rna_sequence_creation_valid():
    """Tests successful creation of a ProgramRNASequence with valid characters."""
    valid_rna = "AUGCGAUCGUAGCUAGCUAG"
    seq = ProgramRNASequence(generator=dummy_gen, generator_output_idx=0, sequence=valid_rna)
    assert seq.sequence == valid_rna
    assert len(seq) == len(valid_rna)
    assert str(seq) == valid_rna

def test_rna_sequence_creation_invalid_chars():
    """Tests that ValueError is raised for ProgramRNASequence with invalid characters."""
    invalid_rna = "AUGCGAUCGTAGCTAGCTAT" # Contains 'T'
    with pytest.raises(ValueError) as excinfo:
        ProgramRNASequence(generator=dummy_gen, generator_output_idx=0, sequence=invalid_rna)
    assert "Invalid characters found: " in str(excinfo.value)

def test_rna_sequence_setter_invalid():
    """Tests that ValueError is raised when setting invalid RNA sequence."""
    seq = ProgramRNASequence(generator=dummy_gen, generator_output_idx=0, sequence="AAAA")
    with pytest.raises(ValueError) as excinfo:
        seq.sequence = "AAAUATAAA" # Invalid 'T'
    assert "Invalid characters found: " in str(excinfo.value)

############################
## Protein Sequence Tests ##
############################

def test_protein_sequence_creation_valid():
    """Tests successful creation of a ProgramProteinSequence with valid characters."""
    valid_protein = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFASFGNLSSPTAILGNPMVRAHGKKVLTSFGDAVKNLDNIKNTFSQLSELHCDKLHVDPENFRLLGNVLVCVLARNFGKEFTPQMQAAYQKVVAGVANALAHKYH"
    seq = ProgramProteinSequence(generator=dummy_gen, generator_output_idx=0, sequence=valid_protein)
    assert seq.sequence == valid_protein
    assert len(seq) == len(valid_protein)
    assert str(seq) == valid_protein

def test_protein_sequence_creation_valid_with_stop_gap():
    """Tests successful creation of a ProgramProteinSequence with stop (*) and gap (-)."""
    valid_protein_special = "ACDEFGHIKLMNPQRSTVWY*-"
    seq = ProgramProteinSequence(generator=dummy_gen, generator_output_idx=0, sequence=valid_protein_special)
    assert seq.sequence == valid_protein_special

def test_protein_sequence_creation_invalid_chars():
    """Tests that ValueError is raised for ProgramProteinSequence with invalid characters."""
    invalid_protein = "MVHLTPEXEKX" # Contains 'X'
    with pytest.raises(ValueError) as excinfo:
        ProgramProteinSequence(generator=dummy_gen, generator_output_idx=0, sequence=invalid_protein)
    assert "Invalid characters found: " in str(excinfo.value)

def test_protein_sequence_setter_invalid():
    """Tests that ValueError is raised when setting invalid protein sequence."""
    seq = ProgramProteinSequence(generator=dummy_gen, generator_output_idx=0, sequence="AAAA")
    with pytest.raises(ValueError) as excinfo:
        seq.sequence = "AAAARXAAA" # Invalid 'X'
    assert "Invalid characters found: " in str(excinfo.value)
