"""Score splice-junction alignment with AlphaGenome."""

import logging
from typing import Any, Literal

import numpy as np
from proto_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.rna_splicing.alphagenome_splice_site_usage import (
    _extract_track_matrix,
    _extract_track_metadata_records,
    _integrate_cassette_into_context,
    _normalize_output_key,
    _select_track_columns,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)


def _extract_splice_junction_payload(result_payload: dict[str, Any]) -> dict[str, Any]:
    predictions = result_payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("AlphaGenome result payload missing 'predictions' dictionary.")

    requested_key = _normalize_output_key("SPLICE_JUNCTIONS")
    for key, value in predictions.items():
        if _normalize_output_key(str(key)) != requested_key:
            continue
        if not isinstance(value, dict):
            raise ValueError("AlphaGenome SPLICE_JUNCTIONS payload is not a dictionary.")
        return value

    raise ValueError("AlphaGenome prediction payload missing SPLICE_JUNCTIONS output.")


def _coerce_numeric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _best_junction_row_index(
    metadata_records: list[dict[str, Any]],
    donor_abs: int,
    acceptor_abs: int,
) -> int | None:
    """Return the metadata row closest to the requested donor/acceptor pair.

    AlphaGenome can emit splice-junction predictions as a compact junction table
    rather than a base-pair matrix. In that case the safest way to recover the
    requested junction is to search metadata for the pair of numeric fields that
    best matches the requested donor and acceptor positions.
    """
    donor_tokens = ("donor", "donor_pos", "donor_position", "donor_index", "five_prime", "5p")
    acceptor_tokens = ("acceptor", "acceptor_pos", "acceptor_position", "acceptor_index", "three_prime", "3p")
    paired_fallback_keys = (("start", "end"), ("left", "right"), ("upstream", "downstream"))

    best_row_index: int | None = None
    best_distance: float | None = None

    for row_index, record in enumerate(metadata_records):
        numeric_fields = {
            str(key).lower(): numeric
            for key, value in record.items()
            if (numeric := _coerce_numeric(value)) is not None
        }
        if not numeric_fields:
            continue

        candidate_pairs: list[tuple[float, float]] = []

        donor_candidates = [
            value for key, value in numeric_fields.items() if any(token in key for token in donor_tokens)
        ]
        acceptor_candidates = [
            value for key, value in numeric_fields.items() if any(token in key for token in acceptor_tokens)
        ]
        if donor_candidates and acceptor_candidates:
            candidate_pairs.extend(
                (donor_value, acceptor_value)
                for donor_value in donor_candidates
                for acceptor_value in acceptor_candidates
            )

        if not candidate_pairs:
            for donor_field, acceptor_field in paired_fallback_keys:
                donor_value = numeric_fields.get(donor_field)
                acceptor_value = numeric_fields.get(acceptor_field)
                if donor_value is not None and acceptor_value is not None:
                    candidate_pairs.append((donor_value, acceptor_value))

        if not candidate_pairs:
            numeric_values = list(numeric_fields.values())
            if len(numeric_values) >= 2:
                candidate_pairs.extend(
                    (numeric_values[first_idx], numeric_values[second_idx])
                    for first_idx in range(len(numeric_values))
                    for second_idx in range(first_idx + 1, len(numeric_values))
                )

        for donor_value, acceptor_value in candidate_pairs:
            distance = abs(donor_value - donor_abs) + abs(acceptor_value - acceptor_abs)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_row_index = row_index
                if distance == 0.0:
                    return best_row_index

    return best_row_index


class AlphaGenomeSpliceJunctionConfig(BaseConfig):
    """Configuration for AlphaGenome splice-junction scoring.

    Attributes:
        genomic_context (str): Genomic context sequence for cassette integration.
        cassette_left_context (str): Left flanking cassette context.
        cassette_right_context (str): Right flanking cassette context.
        ontology_terms (list[str]): AlphaGenome ontology terms to score.
        donor_pos (int): Donor position relative to the concatenated target.
        acceptor_pos (int): Acceptor position relative to the concatenated target.
        direction (Literal['max', 'min']): Whether high or low junction support is preferred.
        strand (Literal['positive', 'negative', 'all']): Strand subset to aggregate.
        model_version (str): AlphaGenome model version.
        organism (Literal['human', 'mouse']): Organism for AlphaGenome prediction.
        device (str): Device for AlphaGenome prediction.
        prediction_timeout (int): Timeout in seconds.
    """

    genomic_context: str = ConfigField(
        title="Genomic Context",
        description="Genomic context sequence for cassette integration.",
    )
    cassette_left_context: str = ConfigField(
        title="Cassette Left Context",
        description="Left flanking cassette context.",
    )
    cassette_right_context: str = ConfigField(
        title="Cassette Right Context",
        description="Right flanking cassette context.",
    )
    ontology_terms: list[str] = ConfigField(
        title="Ontology Terms",
        description="AlphaGenome ontology term(s) to score.",
    )
    donor_pos: int = ConfigField(
        title="Donor Position",
        ge=0,
        description="0-based donor position in left_flank + intron_core + right_flank.",
    )
    acceptor_pos: int = ConfigField(
        title="Acceptor Position",
        ge=0,
        description="0-based acceptor position in left_flank + intron_core + right_flank.",
    )
    direction: Literal["max", "min"] = ConfigField(
        default="max",
        title="Direction",
        description="'max' rewards junction support; 'min' penalizes it.",
    )
    strand: Literal["positive", "negative", "all"] = ConfigField(
        default="positive",
        title="Track Strand",
        description="Track strand subset to aggregate.",
    )
    model_version: str = ConfigField(
        default="all_folds",
        title="Model Version",
        description="AlphaGenome model version.",
    )
    organism: Literal["human", "mouse"] = ConfigField(
        default="human",
        title="Organism",
        description="Organism for AlphaGenome prediction.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for AlphaGenome prediction.",
    )
    prediction_timeout: int = ConfigField(
        default=3600,
        ge=1,
        title="Prediction Timeout",
        description="Timeout in seconds for AlphaGenome prediction.",
    )

    @field_validator("ontology_terms", mode="before")
    @classmethod
    def _normalize_terms(cls, terms: list[str] | str) -> list[str]:
        if isinstance(terms, str):
            terms = [terms]
        normalized = [term.strip() for term in terms if term and term.strip()]
        if not normalized:
            raise ValueError("ontology_terms cannot be empty.")
        return normalized


def _junction_signal(
    matrix: np.ndarray,
    donor: int,
    acceptor: int,
    metadata_records: list[dict[str, Any]] | None = None,
) -> float:
    if matrix.ndim != 2:
        raise ValueError(f"SPLICE_JUNCTIONS values must be 2D after extraction, got shape {matrix.shape}.")
    if matrix.shape[0] == matrix.shape[1] and donor < matrix.shape[0] and acceptor < matrix.shape[1]:
        return float(matrix[donor, acceptor])
    if metadata_records:
        row_index = _best_junction_row_index(metadata_records, donor, acceptor)
        if row_index is not None and row_index < matrix.shape[0]:
            return float(matrix[row_index, :].mean())
    if donor < matrix.shape[0] and acceptor < matrix.shape[0]:
        return float(matrix[[donor, acceptor], :].mean())
    if matrix.shape[0] > 0:
        return float(matrix[0, :].mean())
    raise ValueError(
        f"junction positions donor={donor}, acceptor={acceptor} are out of bounds for matrix shape {matrix.shape}."
    )


@constraint(
    key="alphagenome-splice-junction",
    label="AlphaGenome splice junction score",
    config=AlphaGenomeSpliceJunctionConfig,
    description="Score whether AlphaGenome splice junction predictions align to requested donor/acceptor positions.",
    uses_gpu=True,
    tools_called=["alphagenome-predict-sequences"],
    category="rna_splicing",
    supported_sequence_types=["dna"],
    input_labels=["Left Flank", "Intron Core", "Right Flank"],
)
def alphagenome_splice_junction_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: AlphaGenomeSpliceJunctionConfig,
) -> list[ConstraintOutput]:
    """Score splice-junction support at configured donor/acceptor positions."""
    if not input_sequences:
        return []

    target_seqs = [left.sequence + intron.sequence + right.sequence for left, intron, right in input_sequences]
    target_lengths = {len(sequence) for sequence in target_seqs}
    if len(target_lengths) != 1:
        raise ValueError("AlphaGenome splice junction requires equal-length target sequences in a batch.")
    target_length = target_lengths.pop()
    if config.donor_pos >= target_length or config.acceptor_pos >= target_length:
        raise ValueError(
            f"donor_pos/acceptor_pos must be within target length {target_length}; "
            f"got donor={config.donor_pos}, acceptor={config.acceptor_pos}."
        )

    integrated_seqs = []
    insert_start_ref = None
    for idx, target_seq in enumerate(target_seqs):
        cassette = config.cassette_left_context + target_seq + config.cassette_right_context
        integrated, insert_start = _integrate_cassette_into_context(config.genomic_context, cassette)
        if insert_start_ref is None:
            insert_start_ref = insert_start
        elif insert_start != insert_start_ref:
            raise RuntimeError(
                f"AlphaGenome splice junction cassette insertion drifted at sequence {idx}: "
                f"{insert_start} != {insert_start_ref}."
            )
        integrated_seqs.append(integrated)

    if insert_start_ref is None:
        raise RuntimeError("AlphaGenome splice junction insert_start_ref unset after integration.")
    cassette_offset = insert_start_ref + len(config.cassette_left_context)
    donor_abs = cassette_offset + config.donor_pos
    acceptor_abs = cassette_offset + config.acceptor_pos

    prediction_config = AlphaGenomePredictSequencesConfig(
        model_version=config.model_version,
        requested_outputs=["SPLICE_JUNCTIONS"],
        ontology_terms=config.ontology_terms,
        organism=config.organism,
        device=config.device,
        timeout=config.prediction_timeout,
    )
    batch_output = run_alphagenome_predict_sequences(
        AlphaGenomePredictSequencesInput(sequences=integrated_seqs),
        prediction_config,
    )

    results: list[ConstraintOutput] = []
    for output in batch_output.results:
        payload = _extract_splice_junction_payload(output.result)
        matrix = _extract_track_matrix(payload)
        metadata_records = _extract_track_metadata_records(payload)
        selected_junction_row_index = (
            _best_junction_row_index(metadata_records, donor_abs, acceptor_abs) if metadata_records else None
        )
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            selected_matrix = matrix
            selected_indices = list(range(matrix.shape[1]))
        elif matrix.shape[0] == len(config.genomic_context):
            selected_matrix, selected_indices = _select_track_columns(matrix, metadata_records, config.strand)
        else:
            selected_matrix = matrix
            selected_indices = list(range(matrix.shape[1])) if matrix.ndim == 2 else []

        raw_junction = float(
            np.clip(
                _junction_signal(selected_matrix, donor_abs, acceptor_abs, metadata_records=metadata_records),
                0.0,
                1.0,
            )
        )
        score = float(1.0 - raw_junction) if config.direction == "max" else raw_junction
        selected_track_names = [
            str(metadata_records[idx].get("name", "")) for idx in selected_indices if idx < len(metadata_records)
        ]
        selected_junction_name = (
            str(metadata_records[selected_junction_row_index].get("name", ""))
            if selected_junction_row_index is not None and selected_junction_row_index < len(metadata_records)
            else ""
        )
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "ontology_terms": config.ontology_terms,
                    "donor_pos": config.donor_pos,
                    "acceptor_pos": config.acceptor_pos,
                    "donor_abs": donor_abs,
                    "acceptor_abs": acceptor_abs,
                    "direction": config.direction,
                    "strand": config.strand,
                    "selected_track_names": selected_track_names,
                    "selected_junction_row_index": selected_junction_row_index,
                    "selected_junction_name": selected_junction_name,
                    "alphagenome_splice_junction_raw": raw_junction,
                    "alphagenome_splice_junction_score": score,
                },
            )
        )

    return results
