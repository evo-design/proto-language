"""Profile-HMM matching constraint for protein sequences and translated DNA."""

import logging
from pathlib import Path
from typing import Any, Literal

from proto_tools import PyHmmsearchConfig, PyHmmsearchInput, run_pyhmmer_hmmsearch
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.orf_selection import resolve_protein_complex_chains

logger = logging.getLogger(__name__)


class ProteinProfileHMMConfig(BaseConfig):
    """Configuration for profile-HMM matching.

    Attributes:
        hmm_path (str): Path to an HMM file for ``hmmsearch``.
        required_profiles (list[str]): Optional profile names/accessions or
            description substrings that must be found among domain-level hits.
            When empty, any sequence-level HMM hit passes.
        match_all_required_profiles (bool): If True, require every configured
            profile. If False, any configured profile is sufficient.
        profile_match_field (Literal["query_name", "query_accession", "query_description"]):
            HMM hit field used to match ``required_profiles``.
        hmmsearch_config (PyHmmsearchConfig): PyHMMER hmmsearch configuration.
            Override ``evalue_threshold`` and ``domain_evalue_threshold`` here
            to tune match stringency. Defaults to ``1e-3`` for both (more
            stringent than PyHMMER's default of ``10.0``).
    """

    hmm_path: str = ConfigField(
        title="HMM Path",
        description="Path to a profile-HMM file for hmmsearch.",
    )
    required_profiles: list[str] = ConfigField(
        default_factory=list,
        title="Required Profiles",
        description="Profile names/accessions/descriptions that must be detected. Empty means any HMM hit passes.",
    )
    match_all_required_profiles: bool = ConfigField(
        default=True,
        title="Match All Required Profiles",
        description="Require all listed profiles instead of any listed profile.",
    )
    profile_match_field: Literal["query_name", "query_accession", "query_description"] = ConfigField(
        default="query_name",
        title="Profile Match Field",
        description="HMM hit field used to match required profile strings.",
    )
    hmmsearch_config: PyHmmsearchConfig = ConfigField(
        default_factory=lambda: PyHmmsearchConfig(evalue_threshold=1e-3, domain_evalue_threshold=1e-3),
        title="PyHMMER Config",
        description="PyHMMER hmmsearch configuration. Tune evalue_threshold / domain_evalue_threshold here.",
    )

    @field_validator("required_profiles")
    @classmethod
    def _strip_required_profiles(cls, value: list[str]) -> list[str]:
        """Normalize required profile strings."""
        return [profile.strip() for profile in value if profile.strip()]


@constraint(
    key="protein-profile-hmm",
    label="Protein Profile HMM",
    config=ProteinProfileHMMConfig,
    description="Search proteins, or longest ORFs from DNA, against a profile-HMM file.",
    tools_called=["pyhmmer-hmmsearch", "orfipy-prediction"],
    category="protein quality",
    supported_sequence_types=["dna", "protein"],
)
def protein_profile_hmm_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinProfileHMMConfig
) -> list[ConstraintOutput]:
    """Require profile-HMM support for proteins or translated DNA sequences.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal sequence
            tuples. Each tuple must contain one DNA or protein sequence. DNA is
            translated using the longest canonical ORF rule.
        config (ProteinProfileHMMConfig): HMM path, matching mode, and PyHMMER
            thresholds.

    Returns:
        list[ConstraintOutput]: One output per proposal. A score of 0.0 passes
            and 1.0 fails. Metadata contains selected ORF details for DNA,
            sequence/domain HMM hits, and matched required profiles.

    Raises:
        ValueError: If the HMM path does not exist.
        RuntimeError: If PyHMMER reports failure.
    """
    if not Path(config.hmm_path).exists():
        raise ValueError(f"HMM file not found: {config.hmm_path}")

    resolved_sequences = resolve_protein_complex_chains(input_sequences)
    proteins: list[str] = []
    valid_indices: list[int] = []
    metadata_by_idx: list[dict[str, Any]] = []
    for idx, (chain_sequences, metadata) in enumerate(resolved_sequences):
        metadata_by_idx.append(metadata)
        if chain_sequences is None:
            continue
        if len(chain_sequences) != 1:
            raise ValueError(
                f"protein_profile_hmm_constraint expects single-chain proposals; "
                f"got {len(chain_sequences)} chains at proposal index {idx}."
            )
        proteins.append(chain_sequences[0])
        valid_indices.append(idx)

    if not proteins:
        return [ConstraintOutput(score=MAX_ENERGY, metadata=metadata) for metadata in metadata_by_idx]

    hmm_result = run_pyhmmer_hmmsearch(
        PyHmmsearchInput(sequences=proteins, hmm=config.hmm_path), config.hmmsearch_config
    )

    sequence_hits_by_query: dict[int, list[Any]] = {idx: [] for idx in range(len(proteins))}
    for hit in hmm_result.sequence_hits:
        query_idx = _parse_sequence_index(hit.target_name)
        if query_idx is not None and query_idx in sequence_hits_by_query:
            sequence_hits_by_query[query_idx].append(hit)

    domain_hits_by_query: dict[int, list[Any]] = {idx: [] for idx in range(len(proteins))}
    for hit in hmm_result.domain_hits:
        query_idx = _parse_sequence_index(hit.target_name)
        if query_idx is not None and query_idx in domain_hits_by_query:
            domain_hits_by_query[query_idx].append(hit)

    outputs = [ConstraintOutput(score=MAX_ENERGY, metadata=metadata) for metadata in metadata_by_idx]
    for protein_idx, original_idx in enumerate(valid_indices):
        sequence_hits = sequence_hits_by_query.get(protein_idx, [])
        domain_hits = domain_hits_by_query.get(protein_idx, [])
        profiles_found = _matched_required_profiles(domain_hits, config)
        passes = (
            bool(sequence_hits) if not config.required_profiles else _passes_required_profiles(profiles_found, config)
        )

        metadata = {
            **metadata_by_idx[original_idx],
            "profile_hmm_sequence_hits": _hit_dicts(sequence_hits),
            "profile_hmm_domain_hits": _hit_dicts(domain_hits),
            "required_profiles": config.required_profiles,
            "profiles_found": profiles_found,
            "has_profile_hmm_hit": bool(sequence_hits),
            "resolved_protein_sequence": proteins[protein_idx],
        }
        outputs[original_idx] = ConstraintOutput(score=MIN_ENERGY if passes else MAX_ENERGY, metadata=metadata)

    n_pass = sum(1 for result in outputs if result.score == MIN_ENERGY)
    logger.info("protein_profile_hmm_constraint: %d/%d passed profile-HMM matching", n_pass, len(outputs))
    return outputs


def _parse_sequence_index(target_name: str) -> int | None:
    """Parse sequence index from PyHMMER target names."""
    try:
        if target_name.startswith("seq_"):
            target_name = target_name.removeprefix("seq_")
        return int(target_name)
    except ValueError:
        return None


def _hit_dicts(hits: list[Any]) -> list[dict[str, Any]] | None:
    """Return per-hit ``model_dump()`` dicts, or ``None`` if there are no hits."""
    return [hit.model_dump() for hit in hits] or None


def _matched_required_profiles(domain_hits: list[Any], config: ProteinProfileHMMConfig) -> list[str]:
    """Return required profile strings matched by domain hits."""
    found: list[str] = []
    for required in config.required_profiles:
        required_lower = required.lower()
        for hit in domain_hits:
            haystack = str(getattr(hit, config.profile_match_field)).lower()
            if required_lower in haystack:
                found.append(required)
                break
    return found


def _passes_required_profiles(profiles_found: list[str], config: ProteinProfileHMMConfig) -> bool:
    """Evaluate required-profile pass/fail logic."""
    if config.match_all_required_profiles:
        return set(profiles_found) >= set(config.required_profiles)
    return bool(profiles_found)
