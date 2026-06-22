"""DNA motif contact-count constraint for protein-DNA complexes.

Scores whether a predicted protein-DNA complex makes enough heavy-atom contacts
between the protein and a chosen DNA motif. The complex is predicted (or reused)
via the shared structure resolver, and this module parses the resulting PDB
geometry directly: it counts unique protein-residue / motif-DNA-position contact
pairs within a distance cutoff and turns the per-requirement deficits into a
``[0, 1]`` score where ``0`` is best (all minimum-contact requirements met) and
``1`` is worst. No NA-MPNN / DeepPBS call is involved; only PDB parsing.

Constraints:
- dna-motif-contact-count: Require motif-local protein-DNA contacts in a complex.

Examples:
    Require at least three motif-local contacts in an AF3 protein-DNA complex:

    >>> from proto_language.core import Segment
    >>> protein = Segment(length=100, sequence_type="protein")
    >>> operator = Segment(length=20, sequence_type="dna")
    >>> motif_contact = Constraint(
    ...     inputs=[protein, operator],
    ...     function=dna_motif_contact_count_constraint,
    ...     function_config={
    ...         "structure_tool": "alphafold3",
    ...         "dna_indices": [8, 9, 10, 11],
    ...         "min_contacts": 3,
    ...     },
    ... )
"""

import logging
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.dna_binding_structure_helper import (
    resolve_structure_paths,
)
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import ConfigField

logger = logging.getLogger(__name__)

AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "MSE",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
DNA3 = {
    "DA",
    "DC",
    "DG",
    "DT",
    "DI",
    "A",
    "C",
    "G",
    "T",
    "U",
    "DU",
}
DNA_BACKBONE_ATOMS = {
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O1P",
    "O2P",
    "O3P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
    "O2'",
    "O5*",
    "C5*",
    "C4*",
    "O4*",
    "C3*",
    "O3*",
    "C2*",
    "C1*",
    "O2*",
}

# A single XYZ heavy-atom coordinate.
Coord = np.ndarray[Any, np.dtype[np.float64]]


class DNAMotifContactCountConfig(StructureBasedConstraintConfig):
    """Configuration for the dna-motif-contact-count constraint.

    Predicts (or reuses) a protein-DNA complex and counts heavy-atom contacts
    between protein residues and a selected DNA motif. The score is the largest
    of three normalized deficits (contact pairs, unique protein residues, unique
    DNA positions) clamped to ``[0, 1]``, where ``0`` means every minimum
    requirement is met and ``1`` means none are.

    Attributes:
        dna_indices (list[int]): 0-based motif positions along the selected DNA chain
            (in residue order of appearance). Must be non-empty, non-negative, and unique.
        dna_chain_label (int): Index of the DNA chain in order of appearance in the PDB
            (0-based). Defaults to ``0`` (first DNA chain).
        min_contacts (int): Minimum number of motif-local protein-DNA residue-pair contacts
            required for a perfect score.
        min_unique_protein_residues (int): Minimum number of unique protein residues that
            must contact the motif.
        min_unique_dna_positions (int): Minimum number of motif DNA positions that must have
            at least one protein contact.
        contact_distance_angstrom (float): Heavy-atom distance cutoff (Angstroms) used to
            decide whether a protein residue and a DNA motif position are in contact.
        dna_atom_scope (Literal["base", "any", "backbone"]): Which DNA atoms are eligible for
            motif contacts: base atoms only, any atom, or backbone atoms only.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Structure-prediction tool.
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    dna_indices: list[int] = ConfigField(
        title="DNA Indices",
        description="0-based motif indices on the selected DNA chain.",
    )
    dna_chain_label: int = ConfigField(
        title="DNA Chain Label",
        default=0,
        ge=0,
        description="Index of DNA chain in order-of-appearance in the PDB (0-based).",
    )
    min_contacts: int = ConfigField(
        title="Min Contacts",
        default=1,
        ge=0,
        description="Minimum motif-local protein-DNA residue-pair contacts required.",
    )
    min_unique_protein_residues: int = ConfigField(
        title="Min Unique Prot Residues",
        default=1,
        ge=0,
        description="Minimum number of unique contacting protein residues.",
    )
    min_unique_dna_positions: int = ConfigField(
        title="Min Unique DNA Positions",
        default=1,
        ge=0,
        description="Minimum number of motif DNA positions with at least one contact.",
    )
    contact_distance_angstrom: float = ConfigField(
        title="Contact Distance (A)",
        default=4.0,
        gt=0.0,
        description="Heavy-atom distance cutoff for contact detection.",
    )
    dna_atom_scope: Literal["base", "any", "backbone"] = ConfigField(
        title="DNA Atom Scope",
        default="base",
        description="Which DNA atoms are considered for motif contacts.",
    )
    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )

    @field_validator("dna_indices")
    @classmethod
    def validate_dna_indices(cls, value: list[int]) -> list[int]:
        """Reject empty, negative, or duplicate motif indices."""
        if not value:
            raise ValueError("dna_indices cannot be empty")
        if any(idx < 0 for idx in value):
            raise ValueError("dna_indices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("dna_indices must be unique")
        return value


def _is_hydrogen(atom_name: str) -> bool:
    """Return True if an atom name denotes a hydrogen."""
    stripped = atom_name.strip().upper()
    return stripped.startswith("H") or (stripped[:1].isdigit() and stripped[1:2] == "H")


def _parse_pdb_atoms(
    pdb_path: str,
) -> tuple[
    list[str],
    dict[str, list[tuple[tuple[str, str, str], str]]],
    dict[tuple[str, str, str, str], list[tuple[str, Coord]]],
]:
    """Parse heavy-atom coordinates from a PDB file.

    Returns the chain order, per-chain residue lists (residue key + residue
    name), and per-residue atom lists (atom name + coordinate array). Hydrogens
    and alternate locations other than the primary one are skipped.
    """
    residues_by_chain: dict[str, list[tuple[tuple[str, str, str], str]]] = defaultdict(list)
    residue_atoms: dict[tuple[str, str, str, str], list[tuple[str, Coord]]] = defaultdict(list)
    chain_order: list[str] = []
    seen_residue = set()

    with open(pdb_path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip().upper()
            if _is_hydrogen(atom_name):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            res_uid = (chain, resseq, icode, resname)
            if chain not in residues_by_chain:
                chain_order.append(chain)
            residue_key = (chain, resseq, icode)
            if residue_key not in seen_residue:
                residues_by_chain[chain].append((residue_key, resname))
                seen_residue.add(residue_key)
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            residue_atoms[res_uid].append((atom_name, np.array([x, y, z], dtype=np.float64)))

    return chain_order, residues_by_chain, residue_atoms


def _dna_atom_allowed(atom_name: str, scope: str) -> bool:
    """Return True if a DNA atom is in the configured contact scope."""
    name = atom_name.strip().upper()
    if scope == "any":
        return True
    if scope == "backbone":
        return name in DNA_BACKBONE_ATOMS
    # Base scope: everything that is not a backbone atom.
    return name not in DNA_BACKBONE_ATOMS


def _pair_has_contact(
    protein_atoms: list[Coord],
    dna_atoms: list[Coord],
    cutoff: float,
) -> bool:
    """Return True if any protein/DNA heavy-atom pair is within the cutoff."""
    if not protein_atoms or not dna_atoms:
        return False
    cutoff2 = cutoff * cutoff
    dna_arr = np.asarray(dna_atoms)
    for p in protein_atoms:
        diff = dna_arr - p
        d2 = np.einsum("ij,ij->i", diff, diff)
        if np.any(d2 <= cutoff2):
            return True
    return False


def _score_pdb_motif_contacts(pdb_path: str, config: DNAMotifContactCountConfig) -> tuple[float, dict[str, object]]:
    """Score one predicted complex PDB and return ``(score, metadata)``.

    Counts unique protein-residue / motif-DNA-position contact pairs within the
    configured cutoff, converts the three minimum-contact deficits into a
    ``[0, 1]`` score (``0`` best, ``1`` worst), and returns metadata describing
    the contacts and configuration used.
    """
    chain_order, residues_by_chain, residue_atoms = _parse_pdb_atoms(pdb_path)

    dna_chains = []
    protein_res_uids = []
    for chain in chain_order:
        residues = residues_by_chain.get(chain, [])
        if not residues:
            continue
        dna_count = sum(1 for _, resn in residues if resn in DNA3)
        aa_count = sum(1 for _, resn in residues if resn in AA3)
        if dna_count > 0 and dna_count >= aa_count:
            dna_chains.append(chain)
        elif aa_count > 0:
            for residue_key, resn in residues:
                if resn in AA3:
                    chain_id, resseq, icode = residue_key
                    protein_res_uids.append((chain_id, resseq, icode, resn))

    if not dna_chains:
        raise ValueError(f"No DNA chains found in structure: {pdb_path}")
    if config.dna_chain_label >= len(dna_chains):
        raise ValueError(
            f"dna_chain_label={config.dna_chain_label} out of range for {len(dna_chains)} DNA chains in {pdb_path}"
        )

    selected_dna_chain = dna_chains[config.dna_chain_label]
    dna_residues = [
        (residue_key, resn) for (residue_key, resn) in residues_by_chain[selected_dna_chain] if resn in DNA3
    ]
    if not dna_residues:
        raise ValueError(f"Selected DNA chain has no DNA residues: {pdb_path}")
    if max(config.dna_indices) >= len(dna_residues):
        raise ValueError("dna_indices reference positions outside selected DNA chain length")

    motif_res_uids = []
    motif_pos_to_uid = {}
    for motif_pos in config.dna_indices:
        residue_key, resn = dna_residues[motif_pos]
        chain_id, resseq, icode = residue_key
        uid = (chain_id, resseq, icode, resn)
        motif_res_uids.append(uid)
        motif_pos_to_uid[motif_pos] = uid

    contacting_pairs = set()
    contacting_protein_residues = set()
    contacting_dna_positions = set()

    motif_atom_cache = {}
    for motif_pos, uid in motif_pos_to_uid.items():
        atoms = [
            coords
            for atom_name, coords in residue_atoms.get(uid, [])
            if _dna_atom_allowed(atom_name, config.dna_atom_scope)
        ]
        motif_atom_cache[motif_pos] = atoms

    protein_atom_cache = {}
    for p_uid in protein_res_uids:
        protein_atom_cache[p_uid] = [coords for _, coords in residue_atoms.get(p_uid, [])]

    for p_uid, p_atoms in protein_atom_cache.items():
        if not p_atoms:
            continue
        for motif_pos, d_atoms in motif_atom_cache.items():
            if _pair_has_contact(
                protein_atoms=p_atoms,
                dna_atoms=d_atoms,
                cutoff=float(config.contact_distance_angstrom),
            ):
                contacting_pairs.add((p_uid, motif_pos))
                contacting_protein_residues.add(p_uid)
                contacting_dna_positions.add(motif_pos)

    contact_count = len(contacting_pairs)
    protein_contact_count = len(contacting_protein_residues)
    dna_pos_contact_count = len(contacting_dna_positions)

    d_contacts = max(0, config.min_contacts - contact_count) / max(1, config.min_contacts)
    d_prot = max(
        0,
        config.min_unique_protein_residues - protein_contact_count,
    ) / max(1, config.min_unique_protein_residues)
    d_dna = max(
        0,
        config.min_unique_dna_positions - dna_pos_contact_count,
    ) / max(1, config.min_unique_dna_positions)
    score = float(np.clip(max(d_contacts, d_prot, d_dna), MIN_ENERGY, MAX_ENERGY))

    metadata: dict[str, object] = {
        "motif_contact_count": int(contact_count),
        "motif_contacting_protein_residue_count": int(protein_contact_count),
        "motif_contacting_dna_position_count": int(dna_pos_contact_count),
        "motif_contact_min_contacts": int(config.min_contacts),
        "motif_contact_min_unique_protein_residues": int(config.min_unique_protein_residues),
        "motif_contact_min_unique_dna_positions": int(config.min_unique_dna_positions),
        "motif_contact_distance_angstrom": float(config.contact_distance_angstrom),
        "motif_contact_dna_atom_scope": config.dna_atom_scope,
        "motif_contact_dna_chain_label": int(config.dna_chain_label),
        "motif_contact_dna_chain_id": selected_dna_chain,
        "motif_contact_dna_indices": list(config.dna_indices),
        "motif_contact_pdb_path": pdb_path,
    }
    return score, metadata


@constraint(
    key="dna-motif-contact-count",
    label="DNA Motif Contact Count",
    config=DNAMotifContactCountConfig,
    description="Require motif-local protein-DNA contacts in predicted complexes",
    uses_gpu=True,
    tools_called=["alphafold3-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def dna_motif_contact_count_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: DNAMotifContactCountConfig,
) -> list[ConstraintOutput]:
    """Score motif-local protein-DNA contact deficits in predicted complexes.

    Predicts (or reuses) one protein-DNA complex per proposal, parses the PDB
    geometry, and counts heavy-atom contacts between protein residues and the
    selected DNA motif. The score is the largest of three normalized deficits
    (contact pairs, unique protein residues, unique DNA positions), so ``0`` is
    best (all minimum requirements met) and ``1`` is worst.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of
            protein and DNA sequences forming the complex.
        config (DNAMotifContactCountConfig): Tool, motif, and contact-threshold
            parameters.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0, 1]`` (lower is
            better) and metadata describing the motif contacts (counts, the
            selected DNA chain, motif indices, cutoff, and resolved PDB path).
            A proposal whose structure shape prevents scoring (no DNA chains, no
            DNA residues, out-of-range ``dna_chain_label`` or ``dna_indices``)
            soft-fails to ``MAX_ENERGY`` with ``motif_contact_error`` metadata.
    """
    if not input_sequences:
        return []

    pdb_paths = resolve_structure_paths(
        input_sequences,
        structure_tool=config.structure_tool,
        tool_config=config.tool_config,
    )

    results: list[ConstraintOutput] = []
    for pdb_path in pdb_paths:
        try:
            score, metadata = _score_pdb_motif_contacts(pdb_path, config)
        except ValueError as exc:
            logger.warning("dna-motif-contact: scoring failed for %s: %s", pdb_path, exc)
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"motif_contact_error": str(exc), "motif_contact_pdb_path": pdb_path},
                )
            )
            continue
        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results
