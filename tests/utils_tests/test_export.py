"""Tests for proto_language.utils.export module."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from proto_language.utils.export import (
    _flatten_generator_columns,
    _serialize_value,
    build_proposal_results,
    flatten_constraints,
    flatten_constructs,
    flatten_optimization,
    flatten_sequences,
    to_csv,
    to_fasta,
    to_json,
    to_tsv,
    write_export,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_results():
    """Sample results with 2 results, 1 construct, 2 segments.

    - promoter: gc_content_constraint + length_constraint
    - cds: gc_content_constraint only
    """
    return {
        "results": [
            {
                "result_idx": 0,
                "energy_score": 0.5,
                "constructs": [
                    {
                        "label": "construct_0",
                        "type": "dna",
                        "segments": [
                            {
                                "label": "promoter",
                                "sequence": "ATCGATCG",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.1,
                                        "weight": 1.0,
                                        "weighted_score": 0.1,
                                        "data": {"gc_content": 50.0},
                                    },
                                    "length_constraint": {
                                        "score": 0.0,
                                        "weight": 1.0,
                                        "weighted_score": 0.0,
                                        "data": {"length": 8},
                                    },
                                },
                                "generators": {
                                    "proteinmpnn": {"perplexity": 1.8, "sequence_recovery": 0.7},
                                    "evo1": {"score": -2.5},
                                },
                                "metadata": {"source": "synthetic"},
                            },
                            {
                                "label": "cds",
                                "sequence": "GCTAGCTA",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.05,
                                        "weight": 1.0,
                                        "weighted_score": 0.05,
                                        "data": {"gc_content": 52.0},
                                    },
                                },
                                "generators": {},
                                "metadata": {},
                            },
                        ],
                    },
                ],
            },
            {
                "result_idx": 1,
                "energy_score": 0.3,
                "constructs": [
                    {
                        "label": "construct_0",
                        "type": "dna",
                        "segments": [
                            {
                                "label": "promoter",
                                "sequence": "TTAATTAA",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.2,
                                        "weight": 1.0,
                                        "weighted_score": 0.2,
                                        "data": {"gc_content": 25.0},
                                    },
                                    "length_constraint": {
                                        "score": 0.0,
                                        "weight": 1.0,
                                        "weighted_score": 0.0,
                                        "data": {"length": 8},
                                    },
                                },
                                "metadata": {"source": "synthetic"},
                            },
                            {
                                "label": "cds",
                                "sequence": "CCGGCCGG",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.01,
                                        "weight": 1.0,
                                        "weighted_score": 0.01,
                                        "data": {"gc_content": 75.0},
                                    },
                                },
                                "metadata": {},
                            },
                        ],
                    },
                ],
            },
        ],
        "best_result_idx": 1,
    }


@pytest.fixture
def sample_history():
    """Sample optimizer history using standardized results format."""
    return [
        {
            "time_step": 0,
            "optimizer": {"type": "test", "iteration": 0},
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.8,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "AAAA",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.5,
                                            "weight": 1.0,
                                            "weighted_score": 0.5,
                                            "data": {"gc_content": 0.0},
                                        },
                                    },
                                    "generators": {"evo1": {"score": -3.1}},
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
                {
                    "result_idx": 1,
                    "energy_score": 0.9,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "TTTT",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.5,
                                            "weight": 1.0,
                                            "weighted_score": 0.5,
                                            "data": {"gc_content": 0.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        },
        {
            "time_step": 10,
            "optimizer": {"type": "test", "iteration": 10},
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "ATCG",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.2,
                                            "weight": 1.0,
                                            "weighted_score": 0.2,
                                            "data": {"gc_content": 50.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
                {
                    "result_idx": 1,
                    "energy_score": 0.6,
                    "constructs": [
                        {
                            "label": "construct_0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "promoter",
                                    "sequence": "GCTA",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.3,
                                            "weight": 1.0,
                                            "weighted_score": 0.3,
                                            "data": {"gc_content": 50.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        },
    ]


# =============================================================================
# Test _flatten_generator_columns
# =============================================================================


class TestFlattenGeneratorColumns:
    """Direct unit tests for the generator-column flattener helper."""

    def test_single_generator_flat(self):
        """One generator with several fields produces generator.{key}.{field} columns."""
        flat = _flatten_generator_columns({"proteinmpnn": {"perplexity": 1.8, "sequence_recovery": 0.7}})
        assert flat == {
            "generator.proteinmpnn.perplexity": 1.8,
            "generator.proteinmpnn.sequence_recovery": 0.7,
        }

    def test_multiple_generators_isolated(self):
        """Multiple generator namespaces don't collide."""
        flat = _flatten_generator_columns({"evo1": {"score": -2.5}, "proteinmpnn": {"perplexity": 1.8}})
        assert flat == {
            "generator.evo1.score": -2.5,
            "generator.proteinmpnn.perplexity": 1.8,
        }

    def test_with_segment_prefix(self):
        """Caller-supplied prefix nests under segment label (used by flatten_constructs)."""
        flat = _flatten_generator_columns({"proteinmpnn": {"perplexity": 1.8}}, prefix="promoter.")
        assert flat == {"promoter.generator.proteinmpnn.perplexity": 1.8}

    def test_empty(self):
        """Empty input emits no columns."""
        assert _flatten_generator_columns({}) == {}

    def test_serializes_nested_values(self):
        """Nested dict values are JSON-stringified via _serialize_value."""
        flat = _flatten_generator_columns({"mygen": {"nested_field": {"a": 1.0, "b": 2.0}}})
        assert "generator.mygen.nested_field" in flat
        # _serialize_value JSON-stringifies dicts so flat tables can write them as scalars
        assert isinstance(flat["generator.mygen.nested_field"], str)


# =============================================================================
# Test flatten_sequences
# =============================================================================


class TestFlattenSequences:
    """Tests for flatten_sequences: one row per (result_idx, construct, segment)."""

    def test_row_count(self, sample_results):
        """2 results x 1 construct x 2 segments = 4 rows."""
        rows = flatten_sequences(sample_results)
        assert len(rows) == 4

    def test_fixed_columns(self, sample_results):
        """Every row has result_idx, energy_score, construct, segment, sequence."""
        rows = flatten_sequences(sample_results)
        for row in rows:
            assert "result_idx" in row
            assert "energy_score" in row
            assert "construct" in row
            assert "segment" in row
            assert "sequence" in row

    def test_constraint_columns_present(self, sample_results):
        """Constraint fields use {label}.{field} namespacing."""
        rows = flatten_sequences(sample_results)
        promoter_row = next(r for r in rows if r["segment"] == "promoter" and r["result_idx"] == 0)

        # All constraint fields present
        assert promoter_row["gc_content_constraint.score"] == 0.1
        assert promoter_row["gc_content_constraint.weight"] == 1.0
        assert promoter_row["gc_content_constraint.weighted_score"] == 0.1
        assert promoter_row["gc_content_constraint.gc_content"] == 50.0
        assert promoter_row["length_constraint.score"] == 0.0
        assert promoter_row["length_constraint.length"] == 8

    def test_metadata_prefix(self, sample_results):
        """User metadata uses metadata.{key} prefix."""
        rows = flatten_sequences(sample_results)
        promoter_row = next(r for r in rows if r["segment"] == "promoter" and r["result_idx"] == 0)
        assert promoter_row["metadata.source"] == "synthetic"

    def test_generator_columns_present(self, sample_results):
        """Generator metadata uses generator.{registry_key}.{field} namespacing."""
        rows = flatten_sequences(sample_results)
        promoter_row = next(r for r in rows if r["segment"] == "promoter" and r["result_idx"] == 0)
        assert promoter_row["generator.proteinmpnn.perplexity"] == 1.8
        assert promoter_row["generator.proteinmpnn.sequence_recovery"] == 0.7
        assert promoter_row["generator.evo1.score"] == -2.5

    def test_no_generator_columns_when_empty(self, sample_results):
        """Segments with no generator metadata get no generator columns."""
        rows = flatten_sequences(sample_results)
        cds_row = next(r for r in rows if r["segment"] == "cds" and r["result_idx"] == 0)
        assert not any(k.startswith("generator.") for k in cds_row)

    def test_correct_values(self, sample_results):
        """Spot-check specific values."""
        rows = flatten_sequences(sample_results)
        cds_result1 = next(r for r in rows if r["segment"] == "cds" and r["result_idx"] == 1)
        assert cds_result1["sequence"] == "CCGGCCGG"
        assert cds_result1["energy_score"] == 0.3
        assert cds_result1["gc_content_constraint.gc_content"] == 75.0

    def test_empty_results(self):
        """Handles empty results."""
        assert flatten_sequences({"results": []}) == []


# =============================================================================
# Test flatten_constraints
# =============================================================================


class TestFlattenConstraints:
    """Tests for flatten_constraints: one row per (result, construct, segment, constraint)."""

    def test_row_count(self, sample_results):
        """result0: promoter(2) + cds(1) = 3; result1: same = 3; total = 6."""
        rows = flatten_constraints(sample_results)
        assert len(rows) == 6

    def test_fixed_columns(self, sample_results):
        """Every row has constraint identifier columns + standard metrics."""
        rows = flatten_constraints(sample_results)
        for row in rows:
            assert "result_idx" in row
            assert "construct" in row
            assert "segment" in row
            assert "constraint" in row
            assert "score" in row
            assert "weight" in row
            assert "weighted_score" in row

    def test_custom_data_unprefixed(self, sample_results):
        """Custom data fields are un-prefixed (one constraint per row)."""
        rows = flatten_constraints(sample_results)
        gc_row = next(
            r
            for r in rows
            if r["constraint"] == "gc_content_constraint" and r["result_idx"] == 0 and r["segment"] == "promoter"
        )
        assert gc_row["gc_content"] == 50.0
        assert gc_row["score"] == 0.1

    def test_multi_segment_info(self):
        """Multi-segment constraints include input_segments and position_in_inputs."""
        results = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "protein",
                            "segments": [
                                {
                                    "label": "protein_a",
                                    "sequence": "MVLS",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": [
                                                "c0.protein_a",
                                                "c0.protein_b",
                                            ],
                                            "position_in_inputs": 0,
                                            "data": {"binding_energy": -5.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        }
        rows = flatten_constraints(results)
        assert len(rows) == 1
        assert json.loads(rows[0]["input_segments"]) == ["c0.protein_a", "c0.protein_b"]
        assert rows[0]["position_in_inputs"] == 0
        assert rows[0]["binding_energy"] == -5.0

    def test_empty_results(self):
        """Handles empty results."""
        assert flatten_constraints({"results": []}) == []


# =============================================================================
# Test flatten_constructs
# =============================================================================


class TestFlattenConstructs:
    """Tests for flatten_constructs: one row per (result_idx, construct)."""

    def test_row_count(self, sample_results):
        """2 results x 1 construct = 2 rows."""
        rows = flatten_constructs(sample_results)
        assert len(rows) == 2

    def test_full_sequence(self, sample_results):
        """full_sequence is concatenation of all segment sequences."""
        rows = flatten_constructs(sample_results)
        assert rows[0]["full_sequence"] == "ATCGATCGGCTAGCTA"
        assert rows[1]["full_sequence"] == "TTAATTAACCGGCCGG"

    def test_per_segment_columns(self, sample_results):
        """Per-segment data uses {segment}.{field} prefix."""
        rows = flatten_constructs(sample_results)
        row = rows[0]
        assert row["promoter.sequence"] == "ATCGATCG"
        assert row["cds.sequence"] == "GCTAGCTA"
        assert row["promoter.gc_content_constraint.score"] == 0.1
        assert row["promoter.gc_content_constraint.gc_content"] == 50.0
        assert row["cds.gc_content_constraint.gc_content"] == 52.0

    def test_per_segment_metadata(self, sample_results):
        """Per-segment metadata uses {segment}.metadata.{key} prefix."""
        rows = flatten_constructs(sample_results)
        assert rows[0]["promoter.metadata.source"] == "synthetic"

    def test_per_segment_generator_columns(self, sample_results):
        """Per-segment generator metadata uses {segment}.generator.{registry_key}.{field} prefix."""
        rows = flatten_constructs(sample_results)
        assert rows[0]["promoter.generator.proteinmpnn.perplexity"] == 1.8
        assert rows[0]["promoter.generator.proteinmpnn.sequence_recovery"] == 0.7
        assert rows[0]["promoter.generator.evo1.score"] == -2.5

    def test_empty_results(self):
        """Handles empty results."""
        assert flatten_constructs({"results": []}) == []


# =============================================================================
# Test flatten_optimization
# =============================================================================


class TestFlattenOptimization:
    """Tests for flatten_optimization: one row per (timepoint, result_idx)."""

    def test_row_count(self, sample_history):
        """2 timepoints x 2 results = 4 rows."""
        rows = flatten_optimization(sample_history)
        assert len(rows) == 4

    def test_fixed_columns(self, sample_history):
        """Every row has timepoint, result_idx, energy_score."""
        rows = flatten_optimization(sample_history)
        for row in rows:
            assert "timepoint" in row
            assert "result_idx" in row
            assert "energy_score" in row

    def test_per_segment_sequences(self, sample_history):
        """Single-construct: per-segment data uses {segment}.sequence columns."""
        rows = flatten_optimization(sample_history)
        # First timepoint, result 0
        t0_b0 = next(r for r in rows if r["timepoint"] == 0 and r["result_idx"] == 0)
        assert t0_b0["promoter.sequence"] == "AAAA"
        assert t0_b0["energy_score"] == 0.8

    def test_per_segment_constraint_scores(self, sample_history):
        """Single-construct: constraint scores use {segment}.{constraint}.{field} prefix."""
        rows = flatten_optimization(sample_history)
        t10_b0 = next(r for r in rows if r["timepoint"] == 10 and r["result_idx"] == 0)
        assert t10_b0["promoter.gc_constraint.score"] == 0.2
        assert t10_b0["promoter.gc_constraint.gc_content"] == 50.0

    def test_per_segment_generator_columns(self, sample_history):
        """Single-construct: generator metadata uses {segment}.generator.{key}.{field} prefix."""
        rows = flatten_optimization(sample_history)
        t0_b0 = next(r for r in rows if r["timepoint"] == 0 and r["result_idx"] == 0)
        assert t0_b0["promoter.generator.evo1.score"] == -3.1

    def test_result_energy_scores(self, sample_history):
        """Each result member has its own energy score."""
        rows = flatten_optimization(sample_history)
        t0_b1 = next(r for r in rows if r["timepoint"] == 0 and r["result_idx"] == 1)
        assert t0_b1["energy_score"] == 0.9

    def test_optimizer_metadata_columns(self):
        """Snapshot-level optimizer metadata is available on optimization rows."""
        history = [
            {
                "time_step": 3,
                "optimizer": {"type": "beam-search", "beam_width": 2, "config": {"score_by": "last"}},
                "results": [
                    {
                        "result_idx": 0,
                        "energy_score": 0.5,
                        "constructs": [{"label": "construct", "type": "dna", "segments": []}],
                    }
                ],
                "proposal_results": [
                    {
                        "proposal_idx": 0,
                        "accepted": True,
                        "rejected_by": None,
                        "energy_score": 0.5,
                        "constructs": [{"label": "construct", "type": "dna", "segments": []}],
                    }
                ],
            }
        ]

        rows = flatten_optimization(history, include_proposals=True)

        assert rows[0]["optimizer.type"] == "beam-search"
        assert rows[0]["optimizer.beam_width"] == 2
        assert rows[0]["optimizer.config"] == '{"score_by": "last"}'
        assert rows[1]["pool"] == "proposal"
        assert rows[1]["optimizer.type"] == "beam-search"

    def test_empty_history(self):
        """Handles empty history."""
        assert flatten_optimization([]) == []

    def test_multi_construct_prefixed(self):
        """Multi-construct results prefix columns with construct label."""
        history = [
            {
                "time_step": 0,
                "optimizer": {"type": "test"},
                "results": [
                    {
                        "result_idx": 0,
                        "energy_score": 0.5,
                        "constructs": [
                            {
                                "label": "dna_construct",
                                "type": "dna",
                                "segments": [
                                    {
                                        "label": "seg",
                                        "sequence": "AAAA",
                                        "constraints": {},
                                        "metadata": {},
                                    },
                                ],
                            },
                            {
                                "label": "rna_construct",
                                "type": "rna",
                                "segments": [
                                    {
                                        "label": "seg",
                                        "sequence": "UUUU",
                                        "constraints": {},
                                        "metadata": {},
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ]
        rows = flatten_optimization(history)
        assert len(rows) == 1
        row = rows[0]
        # Both constructs have a segment named "seg"; prefix disambiguates
        assert row["dna_construct.seg.sequence"] == "AAAA"
        assert row["rna_construct.seg.sequence"] == "UUUU"
        assert row["dna_construct.sequence_type"] == "dna"
        assert row["rna_construct.sequence_type"] == "rna"
        # No unprefixed keys
        assert "seg.sequence" not in row
        assert "sequence_type" not in row


# =============================================================================
# Test Format Writers
# =============================================================================


class TestFormatWriters:
    """Tests for format writer functions."""

    @pytest.fixture
    def sample_rows(self):
        """Simple rows for testing format writers."""
        return [
            {"name": "A", "value": 1, "score": 0.5},
            {"name": "B", "value": 2, "score": 0.8},
        ]

    def test_to_csv_returns_string(self, sample_rows):
        """Test to_csv returns valid CSV string."""
        result = to_csv(sample_rows)

        assert isinstance(result, str)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # Header + 2 rows
        assert "name,value,score" in lines[0]

    def test_to_csv_writes_to_path(self, sample_rows):
        """Test to_csv writes to file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            to_csv(sample_rows, path)

            assert path.exists()
            content = path.read_text()
            assert "name,value,score" in content

    def test_to_tsv_uses_tabs(self, sample_rows):
        """Test to_tsv uses tab delimiters."""
        result = to_tsv(sample_rows)

        lines = result.strip().split("\n")
        assert "\t" in lines[0]
        assert "," not in lines[0]

    def test_to_json_returns_valid_json(self, sample_rows):
        """Test to_json returns valid JSON."""
        result = to_json(sample_rows)

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "A"

    def test_to_json_writes_to_path(self, sample_rows):
        """Test to_json writes to file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            to_json(sample_rows, path)

            assert path.exists()
            content = json.loads(path.read_text())
            assert len(content) == 2

    def test_empty_rows(self):
        """Test format writers handle empty rows."""
        assert to_csv([]) == ""
        assert to_tsv([]) == ""
        assert to_json([]) == "[]"


class TestWriteExport:
    """Tests for write_export function."""

    @pytest.fixture
    def sample_rows(self):
        return [{"a": 1, "b": 2}]

    def test_csv_format(self, sample_rows):
        """Test write_export with csv format."""
        result = write_export(sample_rows, "csv")
        assert "a,b" in result

    def test_tsv_format(self, sample_rows):
        """Test write_export with tsv format."""
        result = write_export(sample_rows, "tsv")
        assert "a\tb" in result

    def test_json_format(self, sample_rows):
        """Test write_export with json format."""
        result = write_export(sample_rows, "json")
        parsed = json.loads(result)
        assert parsed[0]["a"] == 1

    def test_xlsx_requires_path(self, sample_rows):
        """Test xlsx format raises error without path."""
        with pytest.raises(ValueError, match="xlsx format requires"):
            write_export(sample_rows, "xlsx")

    def test_invalid_format(self, sample_rows):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            write_export(sample_rows, "invalid")


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_missing_constraints(self):
        """Segments with no constraints produce rows with only fixed columns."""
        results = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.0,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "s0",
                                    "sequence": "ATCG",
                                    "constraints": {},
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        }

        seq_rows = flatten_sequences(results)
        assert len(seq_rows) == 1
        assert seq_rows[0]["sequence"] == "ATCG"

        constraint_rows = flatten_constraints(results)
        assert len(constraint_rows) == 0  # No constraints = no rows

    def test_heterogeneous_constraints(self):
        """Segments with different constraints produce union of columns in CSV."""
        results = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "s0",
                                    "sequence": "ATCG",
                                    "constraints": {
                                        "gc_constraint": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "data": {"gc_content": 50.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                                {
                                    "label": "s1",
                                    "sequence": "GCTA",
                                    "constraints": {
                                        "length_constraint": {
                                            "score": 0.0,
                                            "weight": 1.0,
                                            "weighted_score": 0.0,
                                            "data": {"length": 4},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        }

        rows = flatten_sequences(results)
        csv_output = to_csv(rows)

        # CSV should have columns from both constraints
        assert "gc_constraint.gc_content" in csv_output
        assert "length_constraint.length" in csv_output

    def test_multi_segment_constraint_in_sequences(self):
        """Multi-segment constraints include input_segments in sequences table."""
        results = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "protein",
                            "segments": [
                                {
                                    "label": "protein_a",
                                    "sequence": "MVLS",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": [
                                                "c0.protein_a",
                                                "c0.protein_b",
                                            ],
                                            "position_in_inputs": 0,
                                            "data": {"binding_energy": -5.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                                {
                                    "label": "protein_b",
                                    "sequence": "KAAW",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": [
                                                "c0.protein_a",
                                                "c0.protein_b",
                                            ],
                                            "position_in_inputs": 1,
                                            "data": {"interface_contacts": 12},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        }

        rows = flatten_sequences(results)
        assert len(rows) == 2

        csv_output = to_csv(rows)
        assert "interaction.score" in csv_output
        assert "interaction.input_segments" in csv_output

    def test_constructs_multi_segment_constraint(self):
        """Constructs table with multi-segment constraint uses full prefix."""
        results = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "protein",
                            "segments": [
                                {
                                    "label": "protein_a",
                                    "sequence": "MVLS",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "input_segments": [
                                                "c0.protein_a",
                                                "c0.protein_b",
                                            ],
                                            "position_in_inputs": 0,
                                            "data": {"binding_energy": -5.0},
                                        },
                                    },
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        }

        rows = flatten_constructs(results)
        assert len(rows) == 1
        assert rows[0]["protein_a.interaction.score"] == 0.1
        assert rows[0]["protein_a.interaction.binding_energy"] == -5.0

    def test_csv_round_trip(self, sample_results):
        """Flatten → CSV → parse produces correct data."""
        rows = flatten_sequences(sample_results)
        csv_str = to_csv(rows)

        # Parse CSV back
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(csv_str))
        parsed_rows = list(reader)
        assert len(parsed_rows) == 4
        assert parsed_rows[0]["sequence"] == "ATCGATCG"


# =============================================================================
# Test Filtering
# =============================================================================


class TestFiltering:
    """Tests for segment/constraint filtering on flatten functions."""

    def test_sequences_filter_segment(self, sample_results):
        """Only include rows for the specified segment."""
        rows = flatten_sequences(sample_results, segments={"promoter"})
        assert len(rows) == 2  # 2 results x 1 segment
        assert all(r["segment"] == "promoter" for r in rows)

    def test_sequences_filter_segment_multiple(self, sample_results):
        """Filter by multiple segments returns rows for all specified."""
        rows = flatten_sequences(sample_results, segments={"promoter", "cds"})
        assert len(rows) == 4  # All segments included

    def test_constraints_filter_segment(self, sample_results):
        """Only include constraint rows for the specified segment."""
        rows = flatten_constraints(sample_results, segments={"cds"})
        # cds has 1 constraint x 2 results = 2 rows
        assert len(rows) == 2
        assert all(r["segment"] == "cds" for r in rows)

    def test_constraints_filter_constraint(self, sample_results):
        """Only include rows for the specified constraint label."""
        rows = flatten_constraints(sample_results, constraints={"length_constraint"})
        # Only promoter has length_constraint, 2 results = 2 rows
        assert len(rows) == 2
        assert all(r["constraint"] == "length_constraint" for r in rows)

    def test_constraints_filter_both(self, sample_results):
        """Filter by both segment and constraint simultaneously."""
        rows = flatten_constraints(
            sample_results,
            segments={"promoter"},
            constraints={"gc_content_constraint"},
        )
        # promoter's gc_content_constraint x 2 results = 2 rows
        assert len(rows) == 2
        assert all(r["segment"] == "promoter" for r in rows)
        assert all(r["constraint"] == "gc_content_constraint" for r in rows)

    def test_constructs_filter_segment(self, sample_results):
        """Only include per-segment columns for the specified segment."""
        rows = flatten_constructs(sample_results, segments={"promoter"})
        assert len(rows) == 2  # Row count unchanged (one row per construct)
        row = rows[0]
        assert "promoter.sequence" in row
        assert "cds.sequence" not in row
        # full_sequence still includes all segments
        assert row["full_sequence"] == "ATCGATCGGCTAGCTA"

    def test_optimization_filter_segment(self, sample_history):
        """Only include per-segment columns for the specified segment."""
        rows = flatten_optimization(sample_history, segments={"promoter"})
        assert len(rows) == 4  # Row count unchanged
        assert "promoter.sequence" in rows[0]
        # No other segment columns should appear
        non_fixed_keys = {
            k
            for r in rows
            for k in r
            if not k.startswith(("timepoint", "result_idx", "energy_score", "sequence_type", "stage", "optimizer."))
        }
        assert all(k.startswith("promoter.") for k in non_fixed_keys)

    def test_filter_nonexistent_segment(self, sample_results):
        """Filtering by a segment that doesn't exist returns no segment rows."""
        rows = flatten_sequences(sample_results, segments={"nonexistent"})
        assert len(rows) == 0

    def test_filter_nonexistent_constraint(self, sample_results):
        """Filtering by a constraint that doesn't exist returns no rows."""
        rows = flatten_constraints(sample_results, constraints={"nonexistent"})
        assert len(rows) == 0

    def test_sequences_filter_result_idx(self, sample_results):
        """Only include rows for specified result indices."""
        rows = flatten_sequences(sample_results, result_indices={0})
        assert len(rows) == 2  # 1 result x 2 segments
        assert all(r["result_idx"] == 0 for r in rows)

    def test_constraints_filter_result_idx(self, sample_results):
        """Only include constraint rows for specified result indices."""
        rows = flatten_constraints(sample_results, result_indices={1})
        # result 1: promoter(2 constraints) + cds(1) = 3 rows
        assert len(rows) == 3
        assert all(r["result_idx"] == 1 for r in rows)

    def test_constructs_filter_result_idx(self, sample_results):
        """Only include construct rows for specified result indices."""
        rows = flatten_constructs(sample_results, result_indices={0})
        assert len(rows) == 1
        assert rows[0]["result_idx"] == 0

    def test_optimization_filter_result_idx(self, sample_history):
        """Only include optimization rows for specified result indices."""
        rows = flatten_optimization(sample_history, result_indices={0})
        assert len(rows) == 2  # 2 timepoints x 1 result
        assert all(r["result_idx"] == 0 for r in rows)

    def test_combined_segment_and_result_filter(self, sample_results):
        """Filter by both segment and result index."""
        rows = flatten_sequences(
            sample_results,
            segments={"promoter"},
            result_indices={1},
        )
        assert len(rows) == 1
        assert rows[0]["segment"] == "promoter"
        assert rows[0]["result_idx"] == 1
        assert rows[0]["sequence"] == "TTAATTAA"


# =============================================================================
# Bug 2: Complex value serialization
# =============================================================================


class TestSerializeValue:
    """Tests for _serialize_value helper."""

    def test_dict_to_json(self):
        """Regular dicts serialize to JSON string."""
        d = {"a": 1, "b": [2, 3]}
        result = _serialize_value(d)
        assert result == json.dumps(d)
        assert json.loads(result) == d

    def test_list_to_json(self):
        """Lists serialize to JSON string."""
        lst = [1.0, 2.0, 3.0]
        result = _serialize_value(lst)
        assert result == json.dumps(lst)
        assert json.loads(result) == lst

    def test_tuple_to_json(self):
        """Tuples serialize to JSON string (as arrays)."""
        t = (1, 2, 3)
        result = _serialize_value(t)
        assert json.loads(result) == [1, 2, 3]

    def test_scalar_passthrough(self):
        """Scalars pass through unchanged."""
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value("hello") == "hello"
        assert _serialize_value(None) is None


class TestComplexValueSerialization:
    """Tests that complex values in metadata/constraints serialize properly."""

    def test_complex_metadata_serialized(self):
        """Complex metadata values serialize to JSON strings."""
        results = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.5,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "dna",
                            "segments": [
                                {
                                    "label": "seg",
                                    "sequence": "ATCG",
                                    "constraints": {},
                                    "metadata": {
                                        "scores": [1, 2, 3],
                                        "nested": {"a": {"b": 1}},
                                        "simple": "text",
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        }

        # flatten_sequences
        rows = flatten_sequences(results)
        assert json.loads(rows[0]["metadata.scores"]) == [1, 2, 3]
        assert json.loads(rows[0]["metadata.nested"]) == {"a": {"b": 1}}
        assert rows[0]["metadata.simple"] == "text"  # Scalar unchanged

        # flatten_constructs
        rows = flatten_constructs(results)
        assert json.loads(rows[0]["seg.metadata.scores"]) == [1, 2, 3]


# =============================================================================
# Bug 3: Stage column in optimization history
# =============================================================================


class TestStageColumn:
    """Tests for stage annotation in flatten_optimization."""

    def test_stage_column_present(self):
        """History entries with 'stage' produce a stage column."""
        history = [
            {
                "time_step": 0,
                "stage": 0,
                "optimizer": {"type": "test", "stage": 0},
                "results": [
                    {
                        "result_idx": 0,
                        "energy_score": 0.8,
                        "constructs": [
                            {
                                "label": "c0",
                                "type": "dna",
                                "segments": [
                                    {
                                        "label": "seg",
                                        "sequence": "AAAA",
                                        "constraints": {},
                                        "metadata": {},
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "time_step": 0,
                "stage": 1,
                "optimizer": {"type": "test", "stage": 1},
                "results": [
                    {
                        "result_idx": 0,
                        "energy_score": 0.5,
                        "constructs": [
                            {
                                "label": "c0",
                                "type": "dna",
                                "segments": [
                                    {
                                        "label": "seg",
                                        "sequence": "GCGC",
                                        "constraints": {},
                                        "metadata": {},
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

        rows = flatten_optimization(history)
        assert len(rows) == 2
        assert rows[0]["stage"] == 0
        assert rows[1]["stage"] == 1
        # Timepoints now distinguishable by stage
        assert rows[0]["timepoint"] == 0
        assert rows[1]["timepoint"] == 0

    def test_no_stage_column_when_absent(self, sample_history):
        """History without 'stage' key doesn't produce a stage column."""
        rows = flatten_optimization(sample_history)
        assert "stage" not in rows[0]


# =============================================================================
# Improvement 1: sequence_type column
# =============================================================================


class TestSequenceTypeColumn:
    """Tests for sequence_type column in all flatten functions."""

    def test_sequences_has_sequence_type(self, sample_results):
        rows = flatten_sequences(sample_results)
        assert all(r["sequence_type"] == "dna" for r in rows)

    def test_constraints_has_sequence_type(self, sample_results):
        rows = flatten_constraints(sample_results)
        assert all(r["sequence_type"] == "dna" for r in rows)

    def test_constructs_has_sequence_type(self, sample_results):
        rows = flatten_constructs(sample_results)
        assert all(r["sequence_type"] == "dna" for r in rows)

    def test_optimization_has_sequence_type(self, sample_history):
        rows = flatten_optimization(sample_history)
        assert all(r["sequence_type"] == "dna" for r in rows)

    def test_protein_sequence_type(self):
        """sequence_type reflects actual construct type."""
        results = {
            "results": [
                {
                    "result_idx": 0,
                    "energy_score": 0.1,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "protein",
                            "segments": [
                                {
                                    "label": "seg",
                                    "sequence": "MVLS",
                                    "constraints": {},
                                    "metadata": {},
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_result_idx": 0,
        }
        rows = flatten_sequences(results)
        assert rows[0]["sequence_type"] == "protein"


# =============================================================================
# Improvement 3: FASTA export
# =============================================================================


class TestFastaExport:
    """Tests for to_fasta export."""

    def test_basic_fasta_output(self, sample_results):
        """FASTA output has correct header/sequence pairs."""
        result = to_fasta(sample_results)
        lines = result.strip().split("\n")
        # 2 results x 2 segments = 4 entries, each with header + sequence = 8 lines
        assert len(lines) == 8
        assert lines[0].startswith(">")
        assert lines[1] == "ATCGATCG"

    def test_fasta_default_header(self, sample_results):
        """Default header format: {construct}_{segment}_result{result_idx}."""
        result = to_fasta(sample_results)
        assert ">construct_0_promoter_result0" in result
        assert ">construct_0_cds_result1" in result

    def test_fasta_custom_header(self, sample_results):
        """Custom header format works."""
        result = to_fasta(
            sample_results,
            header_format="result{result_idx}|{segment}",
        )
        assert ">result0|promoter" in result

    def test_fasta_segment_filter(self, sample_results):
        """Segment filter limits output."""
        result = to_fasta(sample_results, segments={"promoter"})
        lines = [line for line in result.strip().split("\n") if line.startswith(">")]
        assert len(lines) == 2
        assert all("promoter" in line for line in lines)

    def test_fasta_result_filter(self, sample_results):
        """Result index filter limits output."""
        result = to_fasta(sample_results, result_indices={0})
        lines = [line for line in result.strip().split("\n") if line.startswith(">")]
        assert len(lines) == 2
        assert all("result0" in line for line in lines)

    def test_fasta_empty(self):
        """Empty results produce empty string."""
        result = to_fasta({"results": []})
        assert result == ""

    def test_fasta_writes_to_path(self, sample_results):
        """to_fasta writes to file path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.fasta"
            to_fasta(sample_results, output=path)
            assert path.exists()
            content = path.read_text()
            assert ">construct_0_promoter_result0" in content


# =============================================================================
# Improvement 4: Segment boundaries in construct export
# =============================================================================


class TestSegmentBoundaries:
    """Tests for segment boundary columns in flatten_constructs."""

    def test_boundaries_present(self, sample_results):
        """Segment start/end columns are present."""
        rows = flatten_constructs(sample_results)
        row = rows[0]
        assert "promoter.start" in row
        assert "promoter.end" in row
        assert "cds.start" in row
        assert "cds.end" in row

    def test_boundaries_correct(self, sample_results):
        """Boundaries match segment positions in full_sequence."""
        rows = flatten_constructs(sample_results)
        row = rows[0]
        full = row["full_sequence"]

        # promoter is first: [0:8]
        assert row["promoter.start"] == 0
        assert row["promoter.end"] == 8
        assert full[row["promoter.start"] : row["promoter.end"]] == row["promoter.sequence"]

        # cds follows: [8:16]
        assert row["cds.start"] == 8
        assert row["cds.end"] == 16
        assert full[row["cds.start"] : row["cds.end"]] == row["cds.sequence"]

    def test_boundaries_with_segment_filter(self, sample_results):
        """Boundaries are still correct when filtering segments."""
        rows = flatten_constructs(sample_results, segments={"cds"})
        row = rows[0]
        # cds should still be at offset 8 (promoter contributes 8 chars)
        assert row["cds.start"] == 8
        assert row["cds.end"] == 16


def test_build_results_includes_generator_metadata():
    """build_results emits a 'generators' key per segment, parallel to 'constraints'."""
    from unittest.mock import MagicMock

    from proto_language.language.core import Construct, Segment, Sequence
    from proto_language.utils.export import build_results, flatten_sequences

    seq = Sequence("ACGT", sequence_type="dna")
    seq._generator_metadata = {"proteinmpnn": {"perplexity": 1.8}, "evo1": {"score": -2.5}}

    segment = MagicMock(spec=Segment)
    segment.label = "promoter"
    segment.result_sequences = [seq]
    construct = MagicMock(spec=Construct)
    construct.label = "c0"
    construct.sequence_type = "dna"
    construct.segments = [segment]

    results = build_results([construct], [0.42])

    # Structured Results dict carries the namespaced store
    seg = results["results"][0]["constructs"][0]["segments"][0]
    assert seg["generators"] == {"proteinmpnn": {"perplexity": 1.8}, "evo1": {"score": -2.5}}

    # And it surfaces through to flat tables as generator.{key}.{field} columns
    rows = flatten_sequences(results)
    assert rows[0]["generator.proteinmpnn.perplexity"] == 1.8
    assert rows[0]["generator.evo1.score"] == -2.5


def test_build_results_filters_inf_nan_in_metadata():
    """Non-finite floats (Python and numpy) in any of the three metadata dicts are converted to None."""
    from unittest.mock import MagicMock

    import numpy as np

    from proto_language.language.core import Construct, Segment, Sequence
    from proto_language.utils.export import build_results

    seq = Sequence("ACGT", sequence_type="dna")
    seq._constraints_metadata = {"alphagenome-track": {"data": {"minimize_clipped_signal": math.nan}}}
    seq._generator_metadata = {"evo1": {"score": math.inf}}
    seq._metadata = {"user_note": -math.inf, "np32": np.float32("nan")}

    segment = MagicMock(spec=Segment)
    segment.label = "s"
    segment.result_sequences = [seq]
    construct = MagicMock(spec=Construct)
    construct.label = "c"
    construct.sequence_type = "dna"
    construct.segments = [segment]

    seg = build_results([construct], [0.42])["results"][0]["constructs"][0]["segments"][0]
    assert seg["constraints"]["alphagenome-track"]["data"]["minimize_clipped_signal"] is None
    assert seg["generators"]["evo1"]["score"] is None
    assert seg["metadata"]["user_note"] is None
    assert seg["metadata"]["np32"] is None


def test_build_results_makes_pydantic_generator_metrics_json_safe():
    """Generator metadata may contain proto-tools Metrics objects and must remain JSON serializable."""
    import json
    from types import SimpleNamespace

    from pydantic import BaseModel, ConfigDict

    from proto_language.utils.export import build_proposal_results, build_results

    class Metric(BaseModel):
        model_config = ConfigDict(extra="allow")
        logits: list[list[float]] | None = None
        vocab: list[str] | None = None

    score = Metric(log_likelihood=-100.0, avg_log_likelihood=-1.0, perplexity=2.72)
    seq = SimpleNamespace(
        sequence="ACGT",
        _constraints_metadata={},
        _generator_metadata={"evo1": {"score": score}},
        _metadata={},
    )
    segment = SimpleNamespace(label="s", result_sequences=[seq], proposal_sequences=[seq])
    construct = SimpleNamespace(label="c", sequence_type="dna", segments=[segment])

    results = build_results([construct], [0.42])
    proposals = build_proposal_results([construct], ["accepted"], [0.42])

    score_payload = results["results"][0]["constructs"][0]["segments"][0]["generators"]["evo1"]["score"]
    assert score_payload == {
        "logits": None,
        "vocab": None,
        "log_likelihood": -100.0,
        "avg_log_likelihood": -1.0,
        "perplexity": 2.72,
    }
    json.dumps({"stage_result": results, "proposal_results": proposals})


def test_build_results_makes_metadata_dict_keys_json_safe():
    """Non-string metadata keys should not leave exported results unable to encode as JSON."""
    import json
    from types import SimpleNamespace

    import numpy as np

    from proto_language.utils.export import build_results

    seq = SimpleNamespace(
        sequence="ACGT",
        _constraints_metadata={np.int64(7): {("tuple", np.int64(2)): "constraint"}},
        _generator_metadata={"evo1": {("score", np.int64(1)): 0.5}},
        _metadata={np.int64(3): "three", ("window", np.int64(4)): {"value": np.float32(1.5)}},
    )
    segment = SimpleNamespace(label="s", result_sequences=[seq])
    construct = SimpleNamespace(label="c", sequence_type="dna", segments=[segment])

    results = build_results([construct], [0.42])

    segment_payload = results["results"][0]["constructs"][0]["segments"][0]
    assert segment_payload["constraints"]["7"]['["tuple",2]'] == "constraint"
    assert segment_payload["generators"]["evo1"]['["score",1]'] == 0.5
    assert segment_payload["metadata"]["3"] == "three"
    assert segment_payload["metadata"]['["window",4]']["value"] == 1.5
    json.dumps(results)


# =============================================================================
# Test build_proposal_results
# =============================================================================


class TestBuildProposalResults:
    """Tests for build_proposal_results function."""

    def _make_constructs(self, proposal_sequences):
        """Helper to create mock constructs with proposal_sequences."""
        from unittest.mock import MagicMock

        from proto_language.language.core import Construct, Segment, Sequence

        segment = MagicMock(spec=Segment)
        segment.label = "promoter"
        segment.proposal_sequences = [Sequence(seq, sequence_type="dna") for seq in proposal_sequences]
        construct = MagicMock(spec=Construct)
        construct.label = "construct_0"
        construct.sequence_type = "dna"
        construct.segments = [segment]
        return [construct]

    def test_mixed_accepted_rejected(self):
        """Proposals with mixed pass/fail status are correctly annotated."""
        constructs = self._make_constructs(["ATCG", "GCTA", "TTAA"])
        outcomes = ["accepted", "GC Filter", "accepted"]
        energies = [0.5, float("inf"), 0.8]

        results = build_proposal_results(constructs, outcomes, energies)

        assert len(results) == 3
        assert results[0]["proposal_idx"] == 0
        assert results[0]["accepted"] is True
        assert results[0]["rejected_by"] is None
        assert results[0]["energy_score"] == 0.5
        assert results[1]["proposal_idx"] == 1
        assert results[1]["accepted"] is False
        assert results[1]["rejected_by"] == "GC Filter"
        assert results[1]["energy_score"] is None  # inf → None
        assert results[2]["accepted"] is True
        assert results[2]["energy_score"] == 0.8

    def test_all_accepted(self):
        """All proposals accepted produces correct output."""
        constructs = self._make_constructs(["ATCG", "GCTA"])
        outcomes = ["accepted", "accepted"]

        results = build_proposal_results(constructs, outcomes)

        assert len(results) == 2
        assert all(r["accepted"] for r in results)
        assert all(r["rejected_by"] is None for r in results)

    def test_all_rejected(self):
        """All proposals rejected produces correct output."""
        constructs = self._make_constructs(["ATCG", "GCTA"])
        outcomes = ["Filter A", "Filter B"]

        results = build_proposal_results(constructs, outcomes)

        assert len(results) == 2
        assert not any(r["accepted"] for r in results)
        assert results[0]["rejected_by"] == "Filter A"
        assert results[1]["rejected_by"] == "Filter B"

    def test_construct_structure(self):
        """Each proposal has correct construct/segment structure."""
        constructs = self._make_constructs(["ATCG"])
        results = build_proposal_results(constructs, ["accepted"])

        assert len(results) == 1
        cand = results[0]
        assert len(cand["constructs"]) == 1
        assert cand["constructs"][0]["label"] == "construct_0"
        assert cand["constructs"][0]["type"] == "dna"
        assert len(cand["constructs"][0]["segments"]) == 1
        assert cand["constructs"][0]["segments"][0]["sequence"] == "ATCG"

    def test_constraint_metadata_on_rejected_proposal(self):
        """Constraint metadata written to rejected proposals is exported."""
        from unittest.mock import MagicMock

        from proto_language.language.core import Sequence

        seq = Sequence("ATCG", sequence_type="dna")
        seq._constraints_metadata = {
            "GC Filter": {
                "score": 0.8,
                "weight": 1.0,
                "weighted_score": 0.8,
                "data": {"gc_content": 80.0},
            }
        }

        segment = MagicMock()
        segment.label = "seg"
        segment.proposal_sequences = [seq]
        construct = MagicMock()
        construct.label = "c0"
        construct.sequence_type = "dna"
        construct.segments = [segment]

        results = build_proposal_results([construct], ["GC Filter"])

        assert results[0]["accepted"] is False
        assert "GC Filter" in results[0]["constructs"][0]["segments"][0]["constraints"]

    def test_empty_constructs(self):
        """Empty constructs list returns empty results."""
        assert build_proposal_results([], ["accepted"]) == []

    def test_empty_segments(self):
        """Constructs with no segments returns empty results."""
        from unittest.mock import MagicMock

        construct = MagicMock()
        construct.segments = []
        assert build_proposal_results([construct], ["accepted"]) == []

    def test_outcomes_shorter_than_proposals_raises(self):
        """Mismatched outcomes length raises ValueError."""
        constructs = self._make_constructs(["ATCG", "GCTA", "TTAA"])
        with pytest.raises(ValueError, match="lengths must match"):
            build_proposal_results(constructs, ["accepted"])

    def test_energy_scores_shorter_than_proposals_raises(self):
        """Mismatched energy_scores length raises ValueError."""
        constructs = self._make_constructs(["ATCG", "GCTA", "TTAA"])
        outcomes = ["accepted", "accepted", "accepted"]
        with pytest.raises(ValueError, match=r"energy_scores.*lengths must match"):
            build_proposal_results(constructs, outcomes, energy_scores=[0.5])


# =============================================================================
# Test flatten_optimization with include_proposals
# =============================================================================


class TestFlattenOptimizationProposals:
    """Tests for flatten_optimization with include_proposals=True."""

    @pytest.fixture
    def history_with_proposals(self):
        """History with proposal_results alongside results."""
        return [
            {
                "time_step": 0,
                "optimizer": {"type": "test"},
                "results": [
                    {
                        "result_idx": 0,
                        "energy_score": 0.5,
                        "constructs": [
                            {
                                "label": "c0",
                                "type": "dna",
                                "segments": [
                                    {
                                        "label": "seg",
                                        "sequence": "AAAA",
                                        "constraints": {},
                                        "metadata": {},
                                    }
                                ],
                            }
                        ],
                    },
                ],
                "proposal_results": [
                    {
                        "proposal_idx": 0,
                        "accepted": True,
                        "rejected_by": None,
                        "energy_score": 0.5,
                        "constructs": [
                            {
                                "label": "c0",
                                "type": "dna",
                                "segments": [
                                    {
                                        "label": "seg",
                                        "sequence": "AAAA",
                                        "constraints": {},
                                        "metadata": {},
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "proposal_idx": 1,
                        "accepted": False,
                        "rejected_by": "GC Filter",
                        "energy_score": None,
                        "constructs": [
                            {
                                "label": "c0",
                                "type": "dna",
                                "segments": [
                                    {
                                        "label": "seg",
                                        "sequence": "GGGG",
                                        "constraints": {
                                            "GC Filter": {
                                                "score": 1.0,
                                                "weight": 1.0,
                                                "weighted_score": 1.0,
                                                "data": {"gc_content": 100.0},
                                            },
                                        },
                                        "metadata": {},
                                    }
                                ],
                            }
                        ],
                    },
                ],
                "best_result_idx": 0,
            },
        ]

    def test_include_proposals_false_unchanged(self, history_with_proposals):
        """include_proposals=False produces identical output to default (no new columns)."""
        rows_default = flatten_optimization(history_with_proposals)
        rows_false = flatten_optimization(history_with_proposals, include_proposals=False)
        assert rows_default == rows_false
        assert all("pool" not in r for r in rows_default)

    def test_include_proposals_adds_pool_column(self, history_with_proposals):
        """include_proposals=True adds pool column to result rows."""
        rows = flatten_optimization(history_with_proposals, include_proposals=True)
        result_rows = [r for r in rows if r.get("pool") == "result"]
        proposals = [r for r in rows if r.get("pool") == "proposal"]
        assert len(result_rows) == 1
        assert len(proposals) == 2

    def test_proposal_rows_have_tracking_columns(self, history_with_proposals):
        """Proposal rows have proposal_idx, accepted, rejected_by, energy_score columns."""
        rows = flatten_optimization(history_with_proposals, include_proposals=True)
        proposals = [r for r in rows if r.get("pool") == "proposal"]

        accepted_cand = next(c for c in proposals if c["proposal_idx"] == 0)
        assert accepted_cand["accepted"] is True
        assert accepted_cand["rejected_by"] is None
        assert accepted_cand["energy_score"] == 0.5

        rejected_cand = next(c for c in proposals if c["proposal_idx"] == 1)
        assert rejected_cand["accepted"] is False
        assert rejected_cand["rejected_by"] == "GC Filter"
        assert rejected_cand["energy_score"] is None

    def test_proposal_constraint_data_exported(self, history_with_proposals):
        """Constraint data on proposal rows is properly flattened."""
        rows = flatten_optimization(history_with_proposals, include_proposals=True)
        rejected_cand = next(r for r in rows if r.get("pool") == "proposal" and r.get("proposal_idx") == 1)
        assert rejected_cand["seg.GC Filter.score"] == 1.0
        assert rejected_cand["seg.GC Filter.gc_content"] == 100.0

    def test_include_proposals_without_proposal_results(self, sample_history):
        """History without proposal_results key works with include_proposals=True."""
        rows = flatten_optimization(sample_history, include_proposals=True)
        result_rows = [r for r in rows if r.get("pool") == "result"]
        proposals = [r for r in rows if r.get("pool") == "proposal"]
        # All rows are result, no proposals
        assert len(result_rows) == 4
        assert len(proposals) == 0
