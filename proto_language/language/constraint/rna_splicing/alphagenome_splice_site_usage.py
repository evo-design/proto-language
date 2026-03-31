"""proto_language/language/constraint/rna_splicing/alphagenome_splice_site_usage.py

Accepts three segments (left_flank, intron_core, right_flank), concatenates
them into a target sequence, integrates the target into a genomic context
via cassette insertion, and scores splice-site usage with AlphaGenome.
Metadata is propagated back to all three input segments."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from proto_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence

logger = logging.getLogger(__name__)


def _normalize_output_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _safe_numeric_array(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0 or arr.size == 0:
        return None
    if not np.isfinite(arr).any():
        return None
    return arr


def _extract_splice_site_usage_track_payload(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    predictions = result_payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("AlphaGenome result payload missing 'predictions' dictionary.")

    requested_key = _normalize_output_key("SPLICE_SITE_USAGE")
    for key, value in predictions.items():
        if _normalize_output_key(str(key)) != requested_key:
            continue
        if not isinstance(value, dict):
            raise ValueError("AlphaGenome SPLICE_SITE_USAGE payload is not a dictionary.")
        return value

    raise ValueError("AlphaGenome prediction payload missing SPLICE_SITE_USAGE output.")


def _extract_track_metadata_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = payload.get("metadata")
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return [row for row in metadata if isinstance(row, dict)]
    if isinstance(metadata, dict):
        records = metadata.get("records")
        if isinstance(records, list):
            return [row for row in records if isinstance(row, dict)]
    return []


def _extract_track_matrix(payload: Dict[str, Any]) -> np.ndarray:
    arr = _safe_numeric_array(payload.get("values"))
    if arr is None:
        raise ValueError("Unable to extract SPLICE_SITE_USAGE values from payload.")

    matrix = arr
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return matrix


def _strand_to_symbol(strand: str) -> str:
    mapping = {
        "positive": "+",
        "negative": "-",
        "all": ".",
    }
    return mapping[strand]


def _select_track_columns(
    matrix: np.ndarray,
    metadata_records: List[Dict[str, Any]],
    strand: str,
) -> Tuple[np.ndarray, List[int]]:
    if strand == "all":
        return matrix, list(range(matrix.shape[1]))

    strand_symbol = _strand_to_symbol(strand)
    if not metadata_records:
        raise ValueError(
            "SPLICE_SITE_USAGE metadata is missing; cannot apply strand-specific track selection."
        )

    selected_indices: List[int] = []
    for idx, row in enumerate(metadata_records):
        if idx >= matrix.shape[1]:
            break
        if str(row.get("strand", "")).strip() == strand_symbol:
            selected_indices.append(idx)

    if not selected_indices:
        raise ValueError(
            f"No SPLICE_SITE_USAGE tracks matched strand='{strand_symbol}' in metadata."
        )

    return matrix[:, selected_indices], selected_indices


def _integrate_cassette_into_context(
    genomic_context: str,
    cassette_sequence: str,
) -> Tuple[str, int]:
    """Center-replace genomic sequence span with cassette, preserving total length."""
    if len(cassette_sequence) > len(genomic_context):
        raise ValueError(
            f"Cassette length {len(cassette_sequence)} exceeds context length {len(genomic_context)}."
        )

    insert_start = (len(genomic_context) - len(cassette_sequence)) // 2
    insert_end = insert_start + len(cassette_sequence)
    integrated = genomic_context[:insert_start] + cassette_sequence + genomic_context[insert_end:]
    if len(integrated) != len(genomic_context):
        raise RuntimeError("Integrated sequence length mismatch.")
    return integrated, insert_start


class AlphaGenomeSpliceSiteUsageConfig(BaseConfig):
    """Configuration for AlphaGenome splice-site-usage scoring.

    Takes three segments (left_flank, intron_core, right_flank), concatenates
    them into a target sequence, wraps with cassette contexts, and integrates
    into a genomic context for AlphaGenome prediction. Splice positions are
    specified relative to the concatenated target sequence.

    Attributes:
        genomic_context (str): Genomic context sequence for cassette integration.
        cassette_left_context (str): Left flanking context for the cassette.
        cassette_right_context (str): Right flanking context for the cassette.
        ontology_terms (list[str]): AlphaGenome ontology term(s) to score.
        splice_pos (list[int]): 0-indexed position(s) in the concatenated target to evaluate.
        direction (Literal['max', 'min']): Optimization direction ('max' or 'min').
        strand (Literal['positive', 'negative', 'all']): Track strand subset to aggregate over.
        model_version (str): AlphaGenome model version.
        organism (Literal['human', 'mouse']): Organism for AlphaGenome prediction.
        prediction_timeout (int): Timeout (seconds) for each prediction call.
        device (str): PyTorch device string for model inference (e.g. 'cpu', 'cuda').
    """

    # Cassette and genomic context fields
    genomic_context: str = ConfigField(
        title="Genomic Context",
        description="Genomic context sequence for cassette integration (e.g., AAVS1 safe harbor locus).",
    )
    cassette_left_context: str = ConfigField(
        title="Cassette Left Context",
        description="Left flanking context for the cassette (plasmid/gene sequence 5' of the target).",
    )
    cassette_right_context: str = ConfigField(
        title="Cassette Right Context",
        description="Right flanking context for the cassette (plasmid/gene sequence 3' of the target).",
    )

    # Scoring fields
    ontology_terms: List[str] = ConfigField(
        title="Ontology Terms",
        description="AlphaGenome ontology term(s) to score.",
    )
    splice_pos: List[int] = ConfigField(
        title="Splice Position(s)",
        description=(
            "0-indexed position(s) in the concatenated target to evaluate."
        ),
    )
    direction: Literal["max", "min"] = ConfigField(
        title="Optimization Direction",
        default="max",
        description="'max' returns 1-mean(SSU); 'min' returns mean(SSU).",
    )
    strand: Literal["positive", "negative", "all"] = ConfigField(
        title="Track Strand",
        default="positive",
        description="Track strand subset to aggregate over.",
    )
    model_version: str = ConfigField(
        title="Model Version",
        default="all_folds",
        description="AlphaGenome model version.",
        advanced=True,
    )
    organism: Literal["human", "mouse"] = ConfigField(
        title="Organism",
        default="human",
        description="Organism for AlphaGenome prediction.",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device for AlphaGenome prediction.",
        hidden=True,
    )
    prediction_timeout: int = ConfigField(
        title="Prediction Timeout",
        default=3600,
        ge=1,
        description="Timeout (seconds) for each AlphaGenome prediction call.",
        advanced=True,
    )

    @field_validator("ontology_terms", mode="before")
    @classmethod
    def _normalize_terms(cls, terms: List[str] | str) -> List[str]:
        if isinstance(terms, str):
            terms = [terms]
        normalized = [t.strip() for t in terms if t and t.strip()]
        if not normalized:
            raise ValueError("ontology_terms cannot be empty.")
        return normalized

    @field_validator("splice_pos", mode="before")
    @classmethod
    def _normalize_splice_pos(cls, positions: List[int] | int) -> List[int]:
        if isinstance(positions, int):
            positions = [positions]
        if not positions:
            raise ValueError("splice_pos cannot be empty.")
        return [int(pos) for pos in positions]


@constraint(
    key="alphagenome-splice-site-usage",
    label="AlphaGenome splice site usage score",
    config=AlphaGenomeSpliceSiteUsageConfig,
    description=(
        "Score splice-site usage with AlphaGenome. "
        "Takes three segments (left_flank, intron_core, right_flank), "
        "integrates into genomic context via cassette insertion, and "
        "scores SSU at specified positions."
    ),
    uses_gpu=True,
    tools_called=["alphagenome-predict-sequences"],
    category="rna splicing",
    supported_sequence_types=["dna"],
    num_input_sequences_per_tuple=3,
)
def alphagenome_splice_site_usage(
    input_sequences: List[Tuple[Sequence, ...]],
    config: AlphaGenomeSpliceSiteUsageConfig,
) -> List[float]:
    """Score AlphaGenome SSU at selected positions in a three-part target.

    Each input tuple contains three DNA sequences (left_flank, intron_core,
    right_flank) which are concatenated into a target, wrapped with cassette
    contexts, and integrated into a genomic context for AlphaGenome prediction.
    Metadata is propagated back to all three input segments.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of 3-tuples (left_flank, intron_core, right_flank).
        config (AlphaGenomeSpliceSiteUsageConfig): Configuration with genomic/cassette contexts and scoring params.

    Returns:
        list[float]: List of float scores in [0.0, 1.0]. Interpretation depends on direction.
    """
    if not input_sequences:
        return []

    # 1. Concatenate 3-part tuples into target sequences.
    target_seqs = []
    for left_flank, intron_core, right_flank in input_sequences:
        target_seqs.append(
            left_flank.sequence + intron_core.sequence + right_flank.sequence
        )

    # 2. Validate target lengths are consistent (batch requirement).
    target_lengths = {len(t) for t in target_seqs}
    if len(target_lengths) != 1:
        raise ValueError(
            "AlphaGenome SSU scoring requires equal-length target sequences in a batch."
        )
    target_length = target_lengths.pop()

    # 3. Validate splice_pos against target length.
    invalid_positions = [
        pos for pos in config.splice_pos if pos < 0 or pos >= target_length
    ]
    if invalid_positions:
        raise ValueError(
            f"splice_pos values {invalid_positions} are out of bounds "
            f"for target length {target_length}."
        )

    # 4. Build integrated sequences via cassette insertion.
    integrated_seqs = []
    insert_start_ref = None
    for target_seq in target_seqs:
        cassette = (
            config.cassette_left_context + target_seq + config.cassette_right_context
        )
        integrated, insert_start = _integrate_cassette_into_context(
            genomic_context=config.genomic_context,
            cassette_sequence=cassette,
        )
        if insert_start_ref is None:
            insert_start_ref = insert_start
        elif insert_start != insert_start_ref:
            raise RuntimeError("Cassette insertion start drifted across batch.")
        integrated_seqs.append(integrated)

    assert insert_start_ref is not None
    cassette_offset = insert_start_ref + len(config.cassette_left_context)
    absolute_splice_pos = [cassette_offset + pos for pos in config.splice_pos]

    # 5. Batched AlphaGenome prediction.
    prediction_config = AlphaGenomePredictSequencesConfig(
        model_version=config.model_version,
        requested_outputs=["SPLICE_SITE_USAGE"],
        ontology_terms=config.ontology_terms,
        organism=config.organism,
        device=config.device,
        timeout=config.prediction_timeout,
    )

    batch_output = run_alphagenome_predict_sequences(
        AlphaGenomePredictSequencesInput(sequences=integrated_seqs),
        prediction_config,
    )
    outputs = batch_output.results

    # 6. Extract scores and propagate metadata.
    scores: List[float] = []
    for (left_flank, intron_core, right_flank), output in zip(
        input_sequences, outputs, strict=True
    ):
        integrated_length = len(config.genomic_context)

        payload = _extract_splice_site_usage_track_payload(output.result)
        matrix = _extract_track_matrix(payload)
        if matrix.shape[0] != integrated_length:
            raise ValueError(
                "SPLICE_SITE_USAGE row count does not match integrated sequence length: "
                f"{matrix.shape[0]} != {integrated_length}."
            )

        metadata_records = _extract_track_metadata_records(payload)
        selected_matrix, selected_indices = _select_track_columns(
            matrix=matrix,
            metadata_records=metadata_records,
            strand=config.strand,
        )

        raw_usage = float(selected_matrix[absolute_splice_pos, :].mean())
        raw_usage = float(np.clip(raw_usage, 0.0, 1.0))
        score = float(1.0 - raw_usage) if config.direction == "max" else raw_usage

        selected_track_names = []
        selected_track_strands = []
        for idx in selected_indices:
            if idx < len(metadata_records):
                selected_track_names.append(str(metadata_records[idx].get("name", "")))
                selected_track_strands.append(str(metadata_records[idx].get("strand", "")))

        metadata = {
            "ontology_terms": config.ontology_terms,
            "splice_pos": list(config.splice_pos),
            "direction": config.direction,
            "strand": config.strand,
            "selected_track_count": int(selected_matrix.shape[1]),
            "selected_track_names": selected_track_names,
            "selected_track_strands": selected_track_strands,
            "alphagenome_splice_site_usage_raw": raw_usage,
            "alphagenome_splice_site_usage_score": score,
        }
        for seq in (left_flank, intron_core, right_flank):
            seq._metadata.update(metadata)

        scores.append(score)

    return scores
