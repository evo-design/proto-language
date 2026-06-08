"""Tests for ``proto_language.cli``.

In-process via ``cli.main(argv)`` to avoid subprocess overhead per call.
"""

from __future__ import annotations

import json

import pytest

from proto_language.cli import main


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """Invoke ``main`` with ``argv`` and capture (exit_code, stdout, stderr)."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ----------------------------------------------------------------------------
# list
# ----------------------------------------------------------------------------


def test_list_text(capsys: pytest.CaptureFixture[str]) -> None:
    """Per-kind listing in text mode emits one line per spec."""
    code, out, _ = _run(capsys, "constraint", "list")
    assert code == 0
    assert out.strip()


def test_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Per-kind listing with ``--json`` returns a parseable list of spec dicts."""
    code, out, _ = _run(capsys, "constraint", "list", "--json")
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert payload


def test_top_level_list_groups_all_kinds_in_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``proto-language list --json`` returns an object keyed by kind."""
    code, out, _ = _run(capsys, "list", "--json")
    assert code == 0
    payload = json.loads(out)
    assert set(payload) == {"constraint", "generator", "optimizer"}


def test_constraint_list_mode_filter(capsys: pytest.CaptureFixture[str]) -> None:
    """``constraint list --mode discrete --json`` only returns discrete-mode constraints."""
    code, out, _ = _run(capsys, "constraint", "list", "--mode", "discrete", "--json")
    assert code == 0
    payload = json.loads(out)
    assert all(item["mode"] == "discrete" for item in payload)


def test_generator_list_input_type_filter(capsys: pytest.CaptureFixture[str]) -> None:
    """``generator list --input-type prompt --json`` only returns prompt-input generators."""
    code, out, _ = _run(capsys, "generator", "list", "--input-type", "prompt", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload, "no autoregressive generators registered"
    assert all(item["input_type"] == "prompt" for item in payload)


def test_constraint_list_category_filter(capsys: pytest.CaptureFixture[str]) -> None:
    """``constraint list --category <real-category> --json`` narrows to that category."""
    cats_code, cats_out, _ = _run(capsys, "constraint", "categories", "--json")
    assert cats_code == 0
    cats = json.loads(cats_out)
    assert cats, "no categories registered"
    target = cats[0]
    code, out, _ = _run(capsys, "constraint", "list", "--category", target, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload, f"--category {target!r} returned no constraints"
    assert all(item["category"] == target for item in payload)


def test_category_filter_accepts_space_form(capsys: pytest.CaptureFixture[str]) -> None:
    """``--category "rna splicing"`` resolves to the normalized ``rna_splicing``."""
    code, out_under, _ = _run(capsys, "constraint", "list", "--category", "rna_splicing", "--json")
    assert code == 0
    under = json.loads(out_under)
    assert under, "expected rna_splicing constraints"
    code, out_space, _ = _run(capsys, "constraint", "list", "--category", "rna splicing", "--json")
    assert code == 0
    space = json.loads(out_space)
    assert {s["key"] for s in under} == {s["key"] for s in space}


def test_category_filter_is_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    """``--category`` normalizes case + spaces, so a mixed-case/spaced form still matches."""
    code, out_canonical, _ = _run(capsys, "constraint", "list", "--category", "protein_structure", "--json")
    assert code == 0
    canonical = json.loads(out_canonical)
    assert canonical, "expected protein_structure constraints"
    code, out_mixed, _ = _run(capsys, "constraint", "list", "--category", "Protein Structure", "--json")
    assert code == 0
    mixed = json.loads(out_mixed)
    assert {s["key"] for s in canonical} == {s["key"] for s in mixed}


def test_category_filter_empty_hints_did_you_mean(capsys: pytest.CaptureFixture[str]) -> None:
    """A substring of a real category triggers a 'did you mean' hint in text mode."""
    code, out, _ = _run(capsys, "constraint", "list", "--category", "structure")
    assert code == 0
    assert "did you mean" in out


# ----------------------------------------------------------------------------
# docs
# ----------------------------------------------------------------------------


def test_docs_json_roundtrip(capsys: pytest.CaptureFixture[str]) -> None:
    """``docs --json`` returns a payload that round-trips through ComponentDoc."""
    from proto_language.utils.docs_api import ComponentDoc

    code, out, _ = _run(capsys, "constraint", "docs", "gc-content", "--json")
    assert code == 0
    doc = ComponentDoc.model_validate(json.loads(out))
    assert doc.kind == "constraint"
    assert doc.key == "gc-content"
    assert doc.config.fields
    assert doc.spec_metadata


def test_docs_unknown_returns_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown identifier exits 2 with an error on stderr."""
    code, _, err = _run(capsys, "constraint", "docs", "not-a-real-constraint")
    assert code == 2
    assert "Unknown constraint" in err


def test_docs_default_includes_tool_summaries(capsys: pytest.CaptureFixture[str]) -> None:
    """``docs <constraint>`` appends a per-tool summary section by default for tool-backed constraints."""
    code, out, _ = _run(capsys, "constraint", "docs", "mpnn-perplexity")
    assert code == 0
    assert "### Tools (1)" in out
    assert "proteinmpnn-gradient" in out
    assert "[inverse_folding]" in out


def test_docs_default_groups_shared_category(capsys: pytest.CaptureFixture[str]) -> None:
    """When every tool shares a category the header surfaces it once."""
    code, out, _ = _run(capsys, "constraint", "docs", "structure-plddt")
    assert code == 0
    assert "### Tools (8, all `structure_prediction`)" in out
    # Per-tool category brackets are omitted when the shared header already names it.
    assert "[structure_prediction]" not in out


def test_docs_no_tools_suppresses_section(capsys: pytest.CaptureFixture[str]) -> None:
    """``--no-tools`` drops the per-tool summary section entirely."""
    code, out, _ = _run(capsys, "constraint", "docs", "mpnn-perplexity", "--no-tools")
    assert code == 0
    assert "### Tools" not in out


def test_docs_with_tool_inlines_one_full_section(capsys: pytest.CaptureFixture[str]) -> None:
    """``--with-tool=<key>`` inlines the README only for the named tool."""
    code, out, _ = _run(capsys, "constraint", "docs", "structure-plddt", "--with-tool=alphafold2-prediction")
    assert code == 0
    assert out.count("## Tool:") == 1
    assert "## Tool: AlphaFold2 Structure Prediction (`alphafold2-prediction`)" in out


def test_docs_with_tools_full_inlines_all(capsys: pytest.CaptureFixture[str]) -> None:
    """``--with-tools-full`` inlines a README block per tool key."""
    code, out, _ = _run(capsys, "constraint", "docs", "structure-plddt", "--with-tools-full")
    assert code == 0
    assert out.count("## Tool:") == 8


def test_docs_json_envelope_carries_tool_summaries(capsys: pytest.CaptureFixture[str]) -> None:
    """JSON mode adds ``tool_summaries`` by default and ``tool_docs`` only for full-doc keys."""
    code, out, _ = _run(capsys, "constraint", "docs", "structure-plddt", "--json", "--with-tool=alphafold2-prediction")
    assert code == 0
    payload = json.loads(out)
    assert len(payload["tool_summaries"]) == 8
    assert {s["category"] for s in payload["tool_summaries"]} == {"structure_prediction"}
    assert [t["key"] for t in payload["tool_docs"]] == ["alphafold2-prediction"]


# ----------------------------------------------------------------------------
# config + schema
# ----------------------------------------------------------------------------


def test_compatible_constraint_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``constraint compatible <key> --json`` round-trips through CompatibilityReport."""
    from proto_language.utils.docs_api import CompatibilityReport

    code, out, _ = _run(capsys, "constraint", "compatible", "gc-content", "--json")
    assert code == 0
    report = CompatibilityReport.model_validate(json.loads(out))
    assert report.kind == "constraint"
    assert report.key == "gc-content"
    assert report.compatible_optimizers


def test_schema_emits_valid_json_schema(capsys: pytest.CaptureFixture[str]) -> None:
    """``schema`` always emits JSON Schema (no --json toggle needed)."""
    code, out, _ = _run(capsys, "optimizer", "schema", "mcmc")
    assert code == 0
    schema = json.loads(out)
    assert "properties" in schema


# ----------------------------------------------------------------------------
# categories
# ----------------------------------------------------------------------------


def test_categories_text(capsys: pytest.CaptureFixture[str]) -> None:
    """``categories`` prints one category per line for kinds that have any."""
    code, out, _ = _run(capsys, "constraint", "categories")
    assert code == 0
    assert out.strip(), "constraint registry should have at least one category"


# ----------------------------------------------------------------------------
# types
# ----------------------------------------------------------------------------


def test_types_lists_core_names(capsys: pytest.CaptureFixture[str]) -> None:
    """``types`` without a name lists the four core types."""
    code, out, _ = _run(capsys, "types")
    assert code == 0
    assert "Sequence" in out
    assert "Program" in out


def test_types_docs_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``types <Name> --json`` returns a CoreTypeDoc payload."""
    code, out, _ = _run(capsys, "types", "Sequence", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["name"] == "Sequence"
    assert payload["params"]
