"""Reusable structure preparation helpers for structure-scoring constraints."""

from __future__ import annotations

from typing import Any, Literal

from proto_tools import (
    FAMPNNPackConfig,
    FAMPNNPackInput,
    FAMPNNStructureInput,
    Structure,
    run_fampnn_pack,
)
from proto_tools.entities.structures import ResidueSelection
from proto_tools.entities.structures.utils import _serialize_gemmi

from proto_language.core import Sequence
from proto_language.utils.base import BaseConfig, ConfigField

AA_THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "GLY",
}
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


class StructurePreparationConfig(BaseConfig):
    """Generic structure preparation for sequence-dependent structure scorers."""

    mode: Literal["proposal_structure", "configured_structure", "fampnn_pack_from_scaffold"] = ConfigField(
        default="proposal_structure",
        title="Preparation Mode",
        description="How to obtain a structure for each proposal before scoring.",
    )
    configured_structure: Structure | None = ConfigField(
        default=None,
        title="Configured Structure",
        description="Static structure used when mode is configured_structure.",
    )
    scaffold_structure: Structure | None = ConfigField(
        default=None,
        title="Scaffold Structure",
        description="Backbone structure used when threading proposal sequences before FAMPNN packing.",
    )
    chain_ids: list[str] | None = ConfigField(
        default=None,
        title="Chain IDs",
        description="Scaffold chain IDs aligned to the input sequence tuple for sequence threading.",
    )
    fixed_positions: ResidueSelection | None = ConfigField(
        default=None,
        title="Fixed Positions",
        description="Optional fixed positions forwarded to FAMPNN packing.",
    )
    fixed_sidechain_positions: ResidueSelection | None = ConfigField(
        default=None,
        title="Fixed Sidechain Positions",
        description="Optional fixed sidechain positions forwarded to FAMPNN packing.",
    )
    fampnn_pack_config: FAMPNNPackConfig = ConfigField(
        default_factory=FAMPNNPackConfig,
        title="FAMPNN Pack Config",
        description="FAMPNN sidechain packing configuration for threaded scaffold structures.",
    )


def prepare_structures_for_proposals(
    input_sequences: list[tuple[Sequence, ...]],
    config: StructurePreparationConfig,
) -> list[Structure]:
    """Return one prepared structure per proposal tuple."""
    if config.mode == "proposal_structure":
        return [_proposal_structure(seq_tuple) for seq_tuple in input_sequences]

    if config.mode == "configured_structure":
        if config.configured_structure is None:
            raise ValueError("structure_preparation.configured_structure is required for configured_structure mode.")
        return [config.configured_structure for _ in input_sequences]

    if config.mode == "fampnn_pack_from_scaffold":
        if config.scaffold_structure is None:
            raise ValueError("structure_preparation.scaffold_structure is required for fampnn_pack_from_scaffold mode.")
        threaded_inputs = [
            FAMPNNStructureInput(
                structure=thread_sequences_onto_structure(
                    config.scaffold_structure,
                    _resolve_chain_sequence_map(seq_tuple, config),
                ),
                fixed_positions=config.fixed_positions,
                fixed_sidechain_positions=config.fixed_sidechain_positions,
            )
            for seq_tuple in input_sequences
        ]
        packed = run_fampnn_pack(
            inputs=FAMPNNPackInput(inputs=threaded_inputs),
            config=config.fampnn_pack_config,
        )
        return [packed_structures[0] for packed_structures in packed.packed_structures]

    raise ValueError(f"Unknown structure preparation mode: {config.mode!r}")


def _proposal_structure(seq_tuple: tuple[Sequence, ...]) -> Structure:
    for seq in seq_tuple:
        if seq.structure is not None:
            return seq.structure
    raise ValueError("No proposal sequence has an attached structure.")


def _resolve_chain_sequence_map(
    seq_tuple: tuple[Sequence, ...],
    config: StructurePreparationConfig,
) -> dict[str, str]:
    if config.scaffold_structure is None:
        raise ValueError("scaffold_structure is required.")
    chain_ids = config.chain_ids
    if chain_ids is None:
        available = config.scaffold_structure.get_chain_ids()
        if len(available) != len(seq_tuple):
            raise ValueError(
                "structure_preparation.chain_ids is required when the scaffold chain count does not match "
                f"the number of input sequences ({len(available)} chains vs {len(seq_tuple)} sequences)."
            )
        chain_ids = available
    if len(chain_ids) != len(seq_tuple):
        raise ValueError(
            f"structure_preparation.chain_ids has {len(chain_ids)} entries, expected {len(seq_tuple)}."
        )
    return {chain_id: seq.sequence for chain_id, seq in zip(chain_ids, seq_tuple, strict=True)}


def thread_sequences_onto_structure(scaffold: Structure, chain_sequences: dict[str, str]) -> Structure:
    """Return a PDB structure with ``chain_sequences`` threaded onto scaffold backbones."""
    struct = scaffold.gemmi_struct.clone()
    seen = set()
    for model in struct:
        for chain in model:
            sequence = chain_sequences.get(chain.name)
            if sequence is None:
                continue
            residues = [residue for residue in chain if _is_protein_residue(residue)]
            if len(sequence) != len(residues):
                raise ValueError(
                    f"Cannot thread sequence of length {len(sequence)} onto chain {chain.name!r} "
                    f"with {len(residues)} protein residues."
                )
            seen.add(chain.name)
            for residue, aa in zip(residues, sequence, strict=True):
                residue.name = AA_THREE_LETTER.get(aa.upper(), "GLY")
                _strip_sidechain_atoms(residue)

    missing = set(chain_sequences) - seen
    if missing:
        raise ValueError(f"Scaffold missing chain(s) requested for threading: {sorted(missing)}")

    return Structure(
        structure=_serialize_gemmi(struct, "pdb", source_format=scaffold.structure_format or "pdb"),
        structure_format="pdb",
        source="sequence-threaded-scaffold",
    )


def _is_protein_residue(residue: Any) -> bool:
    return any(atom.name.strip() == "CA" for atom in residue)


def _strip_sidechain_atoms(residue: Any) -> None:
    for idx in range(len(residue) - 1, -1, -1):
        atom = residue[idx]
        if atom.name.strip() not in BACKBONE_ATOMS:
            del residue[idx]
