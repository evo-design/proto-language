"""Tests for nearest-neighbor protein gap Gini constraint."""

from unittest.mock import patch

from proto_tools import MafftOutput, Mmseqs2Hit, Mmseqs2SearchProteinsOutput, Mmseqs2SequenceSearchResult

from proto_language.language.constraint import ConstraintRegistry, protein_nearest_neighbor_gap_gini_constraint
from proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint import (
    ProteinNearestNeighborGapGiniConfig,
)
from proto_language.language.core import Sequence


def _reference_fasta(tmp_path) -> str:
    """Create a protein reference FASTA fixture."""
    fasta = tmp_path / "refs.fasta"
    fasta.write_text(">hit1\nMKTAYIAK\n")
    return str(fasta)


def test_registry_integration():
    """Test that the nearest-neighbor gap Gini constraint is registered."""
    spec = ConstraintRegistry.get("protein-nearest-neighbor-gap-gini")
    assert spec.label == "Protein Nearest-Neighbor Gap Gini"
    assert "dna" in spec.supported_sequence_types
    assert "protein" in spec.supported_sequence_types


def test_passes_when_pairwise_gap_gini_is_below_threshold(tmp_path):
    """Test that low gap concentration against the nearest neighbor passes."""
    sequence = Sequence("ATGAAATAA", "dna")
    fasta = _reference_fasta(tmp_path)
    config = ProteinNearestNeighborGapGiniConfig(
        mmseqs_db=fasta,
        reference_fasta=fasta,
        max_gap_gini=0.5,
        trim_alignment=False,
    )
    mmseqs_output = Mmseqs2SearchProteinsOutput(
        success=True,
        metadata={},
        results=[
            Mmseqs2SequenceSearchResult(
                query_id="seq_0",
                query_sequence="MKT",
                hits=[Mmseqs2Hit(target_id="hit1", pident=70.0, evalue=1e-6)],
            )
        ],
    )
    mafft_output = MafftOutput(success=True, metadata={}, msa=["MKTAYIAK", "MKTAYIAK"])

    with (
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.resolve_protein_complex_chains"
        ) as mock_resolve,
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.run_mmseqs2_search_proteins"
        ) as mock_mmseqs,
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.run_mafft_align"
        ) as mock_mafft,
    ):
        mock_resolve.return_value = [(["MKTAYIAK"], {"orfipy_orf_count": 1})]
        mock_mmseqs.return_value = mmseqs_output
        mock_mafft.return_value = mafft_output
        result = protein_nearest_neighbor_gap_gini_constraint([(sequence,)], config)[0]

    assert result.score == 0.0
    assert result.metadata["gap_gini"] == 0.0
    assert result.metadata["top_hit_target_id"] == "hit1"
    assert result.metadata["nearest_hit_seq"] == "MKTAYIAK"


def test_alignment_failure_returns_worst_score(tmp_path, caplog):
    """Test that MAFFT failures fail closed instead of silently passing."""
    sequence = Sequence("ATGAAATAA", "dna")
    fasta = _reference_fasta(tmp_path)
    config = ProteinNearestNeighborGapGiniConfig(
        mmseqs_db=fasta,
        reference_fasta=fasta,
        max_gap_gini=0.5,
        trim_alignment=False,
    )
    mmseqs_output = Mmseqs2SearchProteinsOutput(
        success=True,
        metadata={},
        results=[
            Mmseqs2SequenceSearchResult(
                query_id="seq_0",
                query_sequence="MKT",
                hits=[Mmseqs2Hit(target_id="hit1", pident=70.0, evalue=1e-6)],
            )
        ],
    )
    with (
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.resolve_protein_complex_chains"
        ) as mock_resolve,
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.run_mmseqs2_search_proteins"
        ) as mock_mmseqs,
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.run_mafft_align"
        ) as mock_mafft,
        caplog.at_level("WARNING"),
    ):
        mock_resolve.return_value = [(["MKTAYIAK"], {"orfipy_orf_count": 1})]
        mock_mmseqs.return_value = mmseqs_output
        mock_mafft.side_effect = RuntimeError("alignment failed")
        result = protein_nearest_neighbor_gap_gini_constraint([(sequence,)], config)[0]

    assert result.score == 1.0
    assert result.metadata["gap_gini"] == 1.0
    assert "Pairwise MAFFT alignment failed" in caplog.text


def test_no_hit_passes_when_configured(tmp_path):
    """Test that proposals with no nearest neighbor can pass."""
    sequence = Sequence("MKT", "protein")
    fasta = _reference_fasta(tmp_path)
    config = ProteinNearestNeighborGapGiniConfig(mmseqs_db=fasta, reference_fasta=fasta, pass_no_hits=True)
    mmseqs_output = Mmseqs2SearchProteinsOutput(
        success=True,
        metadata={},
        results=[Mmseqs2SequenceSearchResult(query_id="seq_0", query_sequence="MKT", hits=[])],
    )

    with (
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.resolve_protein_complex_chains"
        ) as mock_resolve,
        patch(
            "proto_language.language.constraint.protein_quality.protein_nearest_neighbor_gap_gini_constraint.run_mmseqs2_search_proteins"
        ) as mock_mmseqs,
    ):
        mock_resolve.return_value = [(["MKT"], {})]
        mock_mmseqs.return_value = mmseqs_output
        result = protein_nearest_neighbor_gap_gini_constraint([(sequence,)], config)[0]

    assert result.score == 0.0
    assert result.metadata["has_mmseqs_hit"] is False
