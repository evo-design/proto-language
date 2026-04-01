"""GPU integration checks for AlphaGenome SSU splice-site indexing."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from proto_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)

from proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage import (
    _extract_splice_site_usage_track_payload,
    _extract_track_matrix,
    _extract_track_metadata_records,
    _integrate_cassette_into_context,
    _select_track_columns,
)

# Lazy-imported inside tests because program_intron_design depends on `tap`
# which is not installed in CI. All tests in this file are @skip_ci anyway.
_intron_design = None


def _get_intron_design():
    global _intron_design  # noqa: PLW0603
    if _intron_design is None:
        from examples.scripts import program_intron_design

        _intron_design = program_intron_design
    return _intron_design


def _read_context_sequence(path: str) -> str:
    sequence = Path(path).read_text().strip().upper()
    if not sequence:
        raise ValueError(f"Context file is empty: {path}")
    return sequence


def _site_mean(matrix: np.ndarray, pos: int) -> float:
    return float(np.mean(matrix[pos, :]))


@pytest.mark.uses_gpu
@pytest.mark.skip_ci
@pytest.mark.parametrize("initialization", ["hbb1", "hbb2", "hbb2c"])
def test_ag_ssu_hbb_indexing_matches_donor_acceptor_eval_positions(initialization: str):
    """SSU signal should peak at donor_start-1 and acceptor_end+1 for HBB introns."""
    intron_args = SimpleNamespace(
        initialization=initialization,
        intron_length=301,
    )
    intron_design = _get_intron_design()
    intron_sequence = intron_design.get_initial_intron(intron_args)

    splice_args = SimpleNamespace(
        plasmid_context_path="examples/data/intron_plasmid_context.txt",
        gene_sequence_path="examples/data/mscarlet.txt",
        gene_insertion_pos=159 * 3,
    )
    (
        left_context,
        right_context,
        target_seq,
        _gene_start_pos,
        _gene_end_pos,
        donor_start_pos,
        acceptor_end_pos,
    ) = intron_design.process_splice_transformer_input(intron_sequence, splice_args)

    genomic_context = _read_context_sequence("examples/data/alphagenome_context_aavs1.txt")
    cassette = left_context + target_seq + right_context
    integrated_sequence, insert_start = _integrate_cassette_into_context(
        genomic_context=genomic_context,
        cassette_sequence=cassette,
    )
    cassette_offset = insert_start + len(left_context)

    donor_eval_pos = donor_start_pos - 1
    acceptor_eval_pos = acceptor_end_pos + 1
    donor_start_global = cassette_offset + donor_start_pos
    acceptor_end_global = cassette_offset + acceptor_end_pos
    donor_eval_global = cassette_offset + donor_eval_pos
    acceptor_eval_global = cassette_offset + acceptor_eval_pos

    output = run_alphagenome_predict_sequences(
        AlphaGenomePredictSequencesInput(sequences=[integrated_sequence]),
        AlphaGenomePredictSequencesConfig(
            requested_outputs=["SPLICE_SITE_USAGE"],
            ontology_terms=["EFO:0002067"],  # K562
            device="cuda",
            timeout=3600,
        ),
    ).results[0]

    payload = _extract_splice_site_usage_track_payload(output.result)
    matrix = _extract_track_matrix(payload)
    metadata_records = _extract_track_metadata_records(payload)
    plus_matrix, _selected = _select_track_columns(
        matrix=matrix,
        metadata_records=metadata_records,
        strand="positive",
    )

    donor_candidates = [
        donor_eval_global - 1,
        donor_eval_global,
        donor_eval_global + 1,
        donor_start_global,
    ]
    donor_candidates = [p for p in donor_candidates if 0 <= p < plus_matrix.shape[0]]
    donor_values = {p: _site_mean(plus_matrix, p) for p in donor_candidates}

    acceptor_candidates = [
        acceptor_eval_global - 1,
        acceptor_eval_global,
        acceptor_eval_global + 1,
        acceptor_end_global,
    ]
    acceptor_candidates = [p for p in acceptor_candidates if 0 <= p < plus_matrix.shape[0]]
    acceptor_values = {p: _site_mean(plus_matrix, p) for p in acceptor_candidates}

    donor_eval_value = donor_values[donor_eval_global]
    donor_non_eval_best = max(v for p, v in donor_values.items() if p != donor_eval_global)
    assert donor_eval_value >= donor_non_eval_best

    acceptor_eval_value = acceptor_values[acceptor_eval_global]
    acceptor_non_eval_best = max(v for p, v in acceptor_values.items() if p != acceptor_eval_global)
    assert acceptor_eval_value >= acceptor_non_eval_best

    # Explicitly check the canonical donor/acceptor base positions are not better
    # than the evaluation positions used by SpliceTransformer conventions.
    assert donor_eval_value >= donor_values[donor_start_global]
    assert acceptor_eval_value >= acceptor_values[acceptor_end_global]
