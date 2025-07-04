"""
test_structure_prediction.py

Test various structure prediction models
"""

import sys

sys.path.append(".")

import glob
import pytest
from Bio import SeqIO
from language.tools import (
    predict_structure_esmfold,
    predict_structure_chai1,
    predict_structure_boltz2,
    predict_structure_esm3,
)

# Read in all example sequence files
SEQUENCE_FILES = glob.glob(
    "language/tools/structure_prediction/example_sequences/*.fasta"
)

STRUCTURE_PREDICTORS = {
    "esmfold": predict_structure_esmfold,
    "chai": predict_structure_chai1,
    "boltz": predict_structure_boltz2,
    "esm3": predict_structure_esm3,
}


@pytest.mark.parametrize("sequence_file", SEQUENCE_FILES)
@pytest.mark.parametrize("predictor", STRUCTURE_PREDICTORS.keys())
def test_folding(sequence_file, predictor):

    # Read in the sequences in the fasta files as a list of strings
    sequences, entity_types = parse_fasta_examples(sequence_file)

    # Predict the structure
    predictor_func = STRUCTURE_PREDICTORS[predictor]

    # Model specific skip conditions
    if "esm" in predictor and any(
        entity_type != "protein" for entity_type in entity_types
    ):
        pytest.skip("ESM models only support protein sequences")

    if predictor == "esm3" and len(sequences) > 1:
        pytest.skip("ESM3 only supports a single sequence for now")

    # Run the prediction
    if "esm" in predictor:
        output = predictor_func(sequences=sequences, device="cuda:0")
    else:
        output = predictor_func(sequences=sequences, entity_types=entity_types)

    # Check that the output is not None
    assert output is not None


def parse_fasta_examples(sequence_file):
    sequences = [str(record.seq) for record in SeqIO.parse(sequence_file, "fasta")]
    entity_types = [
        record.description.split("|")[1]
        for record in SeqIO.parse(sequence_file, "fasta")
    ]
    return sequences, entity_types
