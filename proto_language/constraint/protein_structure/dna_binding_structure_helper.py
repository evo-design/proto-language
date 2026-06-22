"""Shared protein-DNA structure resolution for DNA-binding constraints.

DNA-binding constraints (motif specificity, motif/base contact, off-target ipTM,
Rosetta metrics) all need a predicted protein-operator complex as a PDB file to
hand to downstream scorers (NA-MPNN, DeepPBS, PyRosetta). This module resolves
that PDB once per candidate tuple: it reuses a Structure already attached to a
candidate sequence (e.g. predicted by an upstream structure constraint in the
same stage) or a cached PDB path, and otherwise predicts the complex with the
configured structure tool. Resolved PDBs are written to a stable temp cache and
the path is recorded on each candidate sequence's metadata for reuse.

Examples:
    >>> # resolve_structure_paths predicts/reuses a PDB per candidate tuple.
    >>> # paths = resolve_structure_paths(candidates, "alphafold3", af3_config)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence as TypingSequence
from pathlib import Path

from proto_tools import Complex, predict_structures
from pydantic import BaseModel

from proto_language.core import Sequence

# In-process, content-addressed structure cache keyed on sequence + tool + config
# (see ``_cache_key``), so entries are safe to reuse across runs within a process.
# Unbounded by design: lifetime is the process and keys are per candidate tuple.
_STRUCTURE_CACHE: dict[str, str] = {}
_STRUCTURE_CACHE_DIR = Path(tempfile.gettempdir()) / "proto_language_dna_binding_structures"
_STRUCTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_pdb_content(text: str) -> bool:
    """Return True if any line begins a PDB atom record (``ATOM ``/``HETATM``)."""
    return any(line.startswith(("ATOM ", "HETATM")) for line in text.splitlines())


def _serialize_tool_config(tool_config: object) -> str:
    """Convert a tool configuration into a stable JSON string for cache keys."""
    if isinstance(tool_config, BaseModel):
        payload: object = tool_config.model_dump(mode="json")
    elif hasattr(tool_config, "model_dump"):
        payload = tool_config.model_dump(mode="json")
    elif isinstance(tool_config, dict):
        payload = tool_config
    else:
        payload = str(tool_config)
    return json.dumps(payload, sort_keys=True)


def _cache_key(candidate: tuple[Sequence, ...], structure_tool: str, tool_config: object) -> str:
    """Build a deterministic cache key for one candidate tuple."""
    seq_payload = [{"sequence": seq.sequence, "sequence_type": seq.sequence_type} for seq in candidate]
    full_payload = {
        "structure_tool": structure_tool,
        "tool_config": _serialize_tool_config(tool_config),
        "candidate": seq_payload,
    }
    return hashlib.sha256(json.dumps(full_payload, sort_keys=True).encode("utf-8")).hexdigest()


def _write_pdb_to_cache(cache_key: str, pdb_content: str) -> str:
    """Write PDB content to a stable cache path and return that path."""
    pdb_path = _STRUCTURE_CACHE_DIR / f"{cache_key}.pdb"
    pdb_path.write_text(pdb_content, encoding="utf-8")
    return str(pdb_path)


def _existing_pdb_from_candidate(candidate: tuple[Sequence, ...], structure_tool: str, config_hash: str) -> str | None:
    """Reuse a structure already attached to a candidate sequence, if any.

    Reuse is tool-aware: a stored ``pdb_path`` is reused only when the sequence's
    ``structure_tool`` and ``structure_config_hash`` provenance match the requested
    ones, so a PDB folded by one DNA-binding constraint's tool is never silently
    reused by a constraint requesting a different tool/config. The provenance-free
    ``seq.structure`` / ``pdb_output`` branches are reused only when the sequence
    carries no conflicting tool provenance; otherwise the complex is re-predicted.
    Returns a path to a PDB file, or ``None`` if none is reusable.

    Args:
        candidate (tuple[Sequence, ...]): Sequences forming one complex candidate.
        structure_tool (str): Requested structure-prediction toolkit key.
        config_hash (str): Serialized tool-config hash for the requested prediction.

    Returns:
        str | None: A PDB file path to reuse, or ``None`` if none matches.
    """
    for seq in candidate:
        stored_tool = seq._metadata.get("structure_tool")
        stored_hash = seq._metadata.get("structure_config_hash")
        provenance_matches = stored_tool == structure_tool and stored_hash == config_hash
        provenance_absent = stored_tool is None and stored_hash is None

        cached_path = seq._metadata.get("pdb_path")
        if provenance_matches and isinstance(cached_path, str) and Path(cached_path).exists():
            return str(Path(cached_path).resolve())

        # Provenance-free sources: reuse only when nothing contradicts the request.
        if not (provenance_matches or provenance_absent):
            continue

        structure = getattr(seq, "structure", None)
        pdb_content = getattr(structure, "structure_pdb", None)
        if isinstance(pdb_content, str) and _is_pdb_content(pdb_content):
            digest = hashlib.sha256(pdb_content.encode("utf-8")).hexdigest()
            return _write_pdb_to_cache(digest, pdb_content)

        cached_output = seq._metadata.get("pdb_output")
        if isinstance(cached_output, str):
            if Path(cached_output).exists():
                return str(Path(cached_output).resolve())
            if _is_pdb_content(cached_output):
                digest = hashlib.sha256(cached_output.encode("utf-8")).hexdigest()
                return _write_pdb_to_cache(digest, cached_output)
    return None


def _annotate_candidate(
    candidate: tuple[Sequence, ...],
    pdb_path: str,
    structure_tool: str,
    config_hash: str,
    identifier: str,
) -> None:
    """Record the resolved structure path and its tool/config provenance.

    The ``structure_tool`` and ``structure_config_hash`` provenance lets
    ``_existing_pdb_from_candidate`` reuse the path only for matching requests.
    """
    for seq in candidate:
        seq._metadata["pdb_path"] = pdb_path
        seq._metadata["structure_tool"] = structure_tool
        seq._metadata["structure_config_hash"] = config_hash
        seq._metadata["structure_identifier"] = identifier


def resolve_structure_paths(
    candidates: TypingSequence[tuple[Sequence, ...]],
    structure_tool: str,
    tool_config: object,
) -> list[str]:
    """Resolve a protein-DNA complex PDB path for each candidate tuple.

    For each candidate, reuses a cached path, an attached Structure, or a raw PDB
    string when available; otherwise predicts the complex with ``structure_tool``.
    All predictions for missing candidates run in a single ``predict_structures``
    batch. Resolved paths are cached in-process and recorded on candidate metadata.

    Args:
        candidates (TypingSequence[tuple[Sequence, ...]]): Per-candidate sequence
            tuples (protein chain(s) + DNA chain(s)) to fold into a complex.
        structure_tool (str): Structure-prediction toolkit key (e.g. ``"alphafold3"``,
            ``"boltz2"``).
        tool_config (object): Tool configuration forwarded to ``predict_structures``.

    Returns:
        list[str]: One PDB file path per candidate, parallel to ``candidates``.

    Raises:
        RuntimeError: If structure prediction returns a mismatched count.
    """
    pdb_paths: list[str] = [""] * len(candidates)
    missing_indices: list[int] = []
    missing_complexes: list[Complex] = []
    missing_keys: list[str] = []
    config_hash = _serialize_tool_config(tool_config)

    for idx, candidate in enumerate(candidates):
        key = _cache_key(candidate, structure_tool, tool_config)
        cached_path = _STRUCTURE_CACHE.get(key)
        if cached_path and Path(cached_path).exists():
            pdb_paths[idx] = cached_path
            _annotate_candidate(candidate, cached_path, structure_tool, config_hash, Path(cached_path).stem)
            continue

        reuse_path = _existing_pdb_from_candidate(candidate, structure_tool, config_hash)
        if reuse_path is not None:
            _STRUCTURE_CACHE[key] = reuse_path
            pdb_paths[idx] = reuse_path
            _annotate_candidate(candidate, reuse_path, structure_tool, config_hash, Path(reuse_path).stem)
            continue

        chains = [{"sequence": seq.sequence, "entity_type": seq.sequence_type} for seq in candidate]
        missing_indices.append(idx)
        missing_keys.append(key)
        missing_complexes.append(Complex(chains=chains))

    if missing_complexes:
        output = predict_structures(missing_complexes, structure_tool, tool_config)
        if len(output.structures) != len(missing_complexes):
            raise RuntimeError("Structure prediction returned a mismatched number of structures.")
        for local_idx, structure in enumerate(output.structures):
            global_idx = missing_indices[local_idx]
            key = missing_keys[local_idx]
            pdb_path = _write_pdb_to_cache(key, structure.structure_pdb)
            _STRUCTURE_CACHE[key] = pdb_path
            pdb_paths[global_idx] = pdb_path
            _annotate_candidate(candidates[global_idx], pdb_path, structure_tool, config_hash, key)

    return pdb_paths
