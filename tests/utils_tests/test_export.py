"""Tests for proto_language.utils.export module."""

import json
import tempfile
from pathlib import Path

import pytest

from proto_language.utils.export import (
    flatten_segment_metadata,
    flatten_construct_metadata,
    flatten_program_metadata,
    flatten_batch_over_time,
    to_csv,
    to_tsv,
    to_json,
    write_export,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_batch_results():
    """Sample batch_results structure with nested 'data' key for custom metrics."""
    return {
        "batch_results": [
            {
                "batch_idx": 0,
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
                                        "multi_input": False,
                                        "data": {"gc_content": 50.0},
                                    },
                                    "length_constraint": {
                                        "score": 0.0,
                                        "weight": 1.0,
                                        "weighted_score": 0.0,
                                        "multi_input": False,
                                        "data": {"length": 8},
                                    },
                                },
                            },
                            {
                                "label": "cds",
                                "sequence": "GCTAGCTA",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.05,
                                        "weight": 1.0,
                                        "weighted_score": 0.05,
                                        "multi_input": False,
                                        "data": {"gc_content": 52.0},
                                    },
                                },
                            },
                        ],
                    },
                ],
            },
            {
                "batch_idx": 1,
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
                                        "multi_input": False,
                                        "data": {"gc_content": 25.0},
                                    },
                                    "length_constraint": {
                                        "score": 0.0,
                                        "weight": 1.0,
                                        "weighted_score": 0.0,
                                        "multi_input": False,
                                        "data": {"length": 8},
                                    },
                                },
                            },
                            {
                                "label": "cds",
                                "sequence": "CCGGCCGG",
                                "constraints": {
                                    "gc_content_constraint": {
                                        "score": 0.01,
                                        "weight": 1.0,
                                        "weighted_score": 0.01,
                                        "multi_input": False,
                                        "data": {"gc_content": 75.0},
                                    },
                                },
                            },
                        ],
                    },
                ],
            },
        ],
        "best_batch_idx": 1,
    }


@pytest.fixture
def sample_history():
    """Sample optimizer history structure with nested 'data' key."""
    return [
        {
            "time_step": 0,
            "energy_scores": [0.8, 0.9],
            "constructs": [
                {
                    "label": "construct_0",
                    "segments": [
                        {
                            "label": "promoter",
                            "selected_sequences": [
                                {
                                    "sequence": "AAAA",
                                    "metadata": {
                                        "constraints": {
                                            "gc_constraint": {
                                                "score": 0.5,
                                                "weight": 1.0,
                                                "weighted_score": 0.5,
                                                "multi_input": False,
                                                "data": {"gc_content": 0.0},
                                            },
                                        },
                                    },
                                },
                                {
                                    "sequence": "TTTT",
                                    "metadata": {
                                        "constraints": {
                                            "gc_constraint": {
                                                "score": 0.5,
                                                "weight": 1.0,
                                                "weighted_score": 0.5,
                                                "multi_input": False,
                                                "data": {"gc_content": 0.0},
                                            },
                                        },
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        },
        {
            "time_step": 10,
            "energy_scores": [0.5, 0.6],
            "constructs": [
                {
                    "label": "construct_0",
                    "segments": [
                        {
                            "label": "promoter",
                            "selected_sequences": [
                                {
                                    "sequence": "ATCG",
                                    "metadata": {
                                        "constraints": {
                                            "gc_constraint": {
                                                "score": 0.2,
                                                "weight": 1.0,
                                                "weighted_score": 0.2,
                                                "multi_input": False,
                                                "data": {"gc_content": 50.0},
                                            },
                                        },
                                    },
                                },
                                {
                                    "sequence": "GCTA",
                                    "metadata": {
                                        "constraints": {
                                            "gc_constraint": {
                                                "score": 0.3,
                                                "weight": 1.0,
                                                "weighted_score": 0.3,
                                                "multi_input": False,
                                                "data": {"gc_content": 50.0},
                                            },
                                        },
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ]


# =============================================================================
# Test flatten_segment_metadata
# =============================================================================

class TestFlattenSegmentMetadata:
    """Tests for flatten_segment_metadata function."""

    def test_wide_style(self, sample_batch_results):
        """Test wide style returns single row with constraint.metric columns."""
        rows = flatten_segment_metadata(
            sample_batch_results, "construct_0", "promoter", batch_idx=0, style="wide"
        )
        
        assert len(rows) == 1
        row = rows[0]
        assert row["sequence"] == "ATCGATCG"
        assert row["gc_content_constraint.gc_content"] == 50.0
        assert row["gc_content_constraint.score"] == 0.1
        assert row["length_constraint.length"] == 8

    def test_long_style(self, sample_batch_results):
        """Test long style returns one row per constraint."""
        rows = flatten_segment_metadata(
            sample_batch_results, "construct_0", "promoter", batch_idx=0, style="long"
        )
        
        assert len(rows) == 2  # Two constraints
        labels = {row["constraint_label"] for row in rows}
        assert labels == {"gc_content_constraint", "length_constraint"}

    def test_different_batch_idx(self, sample_batch_results):
        """Test accessing different batch indices."""
        rows = flatten_segment_metadata(
            sample_batch_results, "construct_0", "promoter", batch_idx=1, style="wide"
        )
        
        assert len(rows) == 1
        assert rows[0]["sequence"] == "TTAATTAA"
        assert rows[0]["gc_content_constraint.gc_content"] == 25.0

    def test_segment_not_found(self, sample_batch_results):
        """Test returns empty list when segment not found."""
        rows = flatten_segment_metadata(
            sample_batch_results, "construct_0", "nonexistent", batch_idx=0, style="wide"
        )
        assert rows == []

    def test_construct_not_found(self, sample_batch_results):
        """Test returns empty list when construct not found."""
        rows = flatten_segment_metadata(
            sample_batch_results, "nonexistent", "promoter", batch_idx=0, style="wide"
        )
        assert rows == []


# =============================================================================
# Test flatten_construct_metadata
# =============================================================================

class TestFlattenConstructMetadata:
    """Tests for flatten_construct_metadata function."""

    def test_wide_style(self, sample_batch_results):
        """Test wide style returns one row per segment."""
        rows = flatten_construct_metadata(
            sample_batch_results, "construct_0", batch_idx=0, style="wide"
        )
        
        assert len(rows) == 2  # Two segments
        labels = [row["segment_label"] for row in rows]
        assert labels == ["promoter", "cds"]

    def test_long_style(self, sample_batch_results):
        """Test long style returns one row per segment × constraint."""
        rows = flatten_construct_metadata(
            sample_batch_results, "construct_0", batch_idx=0, style="long"
        )
        
        # promoter has 2 constraints, cds has 1 = 3 total
        assert len(rows) == 3

    def test_construct_not_found(self, sample_batch_results):
        """Test returns empty list when construct not found."""
        rows = flatten_construct_metadata(
            sample_batch_results, "nonexistent", batch_idx=0, style="wide"
        )
        assert rows == []


# =============================================================================
# Test flatten_program_metadata
# =============================================================================

class TestFlattenProgramMetadata:
    """Tests for flatten_program_metadata function."""

    def test_wide_style(self, sample_batch_results):
        """Test wide style returns one row per batch."""
        rows = flatten_program_metadata(sample_batch_results, style="wide")
        
        assert len(rows) == 2  # Two batches
        assert rows[0]["batch_idx"] == 0
        assert rows[1]["batch_idx"] == 1
        assert "construct_0.promoter.sequence" in rows[0]

    def test_long_style(self, sample_batch_results):
        """Test long style returns one row per batch × segment."""
        rows = flatten_program_metadata(sample_batch_results, style="long")
        
        # 2 batches × 2 segments = 4 rows
        assert len(rows) == 4
        assert all("construct_label" in row for row in rows)
        assert all("segment_label" in row for row in rows)

    def test_empty_results(self):
        """Test handles empty batch_results."""
        rows = flatten_program_metadata({"batch_results": []}, style="wide")
        assert rows == []


# =============================================================================
# Test flatten_batch_over_time
# =============================================================================

class TestFlattenBatchOverTime:
    """Tests for flatten_batch_over_time function."""

    def test_wide_style(self, sample_history):
        """Test wide style returns one row per timepoint."""
        rows = flatten_batch_over_time(sample_history, batch_idx=0, style="wide")
        
        assert len(rows) == 2  # Two timepoints
        assert rows[0]["timepoint"] == 0
        assert rows[1]["timepoint"] == 10

    def test_long_style(self, sample_history):
        """Test long style returns one row per timepoint × segment."""
        rows = flatten_batch_over_time(sample_history, batch_idx=0, style="long")
        
        assert len(rows) == 2  # 2 timepoints × 1 segment
        assert all("construct_label" in row for row in rows)

    def test_different_batch_idx(self, sample_history):
        """Test accessing different batch indices."""
        rows = flatten_batch_over_time(sample_history, batch_idx=1, style="wide")
        
        assert len(rows) == 2
        assert rows[0]["energy_score"] == 0.9  # Second batch's energy
        assert rows[1]["energy_score"] == 0.6

    def test_empty_history(self):
        """Test handles empty history."""
        rows = flatten_batch_over_time([], batch_idx=0, style="wide")
        assert rows == []


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
        """Test handling segments with no constraints."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
                    "energy_score": 0.0,
                    "constructs": [
                        {
                            "label": "c0",
                            "type": "dna",
                            "segments": [
                                {"label": "s0", "sequence": "ATCG", "constraints": {}},
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }
        
        rows = flatten_segment_metadata(batch_results, "c0", "s0", 0, "wide")
        assert len(rows) == 1
        assert rows[0]["sequence"] == "ATCG"

    def test_heterogeneous_constraints(self):
        """Test handling segments with different constraints (union of columns)."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
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
                                            "multi_input": False,
                                            "data": {"gc_content": 50.0},
                                        },
                                    },
                                },
                                {
                                    "label": "s1",
                                    "sequence": "GCTA",
                                    "constraints": {
                                        "length_constraint": {
                                            "score": 0.0,
                                            "weight": 1.0,
                                            "weighted_score": 0.0,
                                            "multi_input": False,
                                            "data": {"length": 4},
                                        },
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }
        
        rows = flatten_construct_metadata(batch_results, "c0", 0, "wide")
        csv_output = to_csv(rows)
        
        # CSV should have columns from both constraints
        assert "gc_constraint.gc_content" in csv_output
        assert "length_constraint.length" in csv_output

    def test_multi_input_constraint(self):
        """Test handling multi-input constraints."""
        batch_results = {
            "batch_results": [
                {
                    "batch_idx": 0,
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
                                            "multi_input": True,
                                            "input_segments": ["c0.protein_a", "c0.protein_b"],
                                            "position_in_inputs": 0,
                                            "data": {"binding_energy": -5.0},
                                        },
                                    },
                                },
                                {
                                    "label": "protein_b",
                                    "sequence": "KAAW",
                                    "constraints": {
                                        "interaction": {
                                            "score": 0.1,
                                            "weight": 1.0,
                                            "weighted_score": 0.1,
                                            "multi_input": True,
                                            "input_segments": ["c0.protein_a", "c0.protein_b"],
                                            "position_in_inputs": 1,
                                            "data": {"interface_contacts": 12},
                                        },
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
            "best_batch_idx": 0,
        }
        
        rows = flatten_construct_metadata(batch_results, "c0", 0, "wide")
        assert len(rows) == 2
        
        # Both segments should have the interaction constraint columns
        csv_output = to_csv(rows)
        assert "interaction.score" in csv_output
        assert "interaction.multi_input" in csv_output
