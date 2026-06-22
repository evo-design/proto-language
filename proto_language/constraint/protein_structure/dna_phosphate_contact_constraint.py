"""DNA phosphate contact constraint.

Lightweight geometry-based scoring of protein-DNA backbone phosphate contacts.
Measures protein-DNA phosphate H-bond contacts directly from heavy-atom
distances in a predicted protein-operator complex PDB, WITHOUT requiring
Rosetta relaxation. Scores the protein-DNA phosphate-contact H-bond term separately
from the base-contact H-bond term. Counts protein polar side-chain atoms within a
distance cutoff of DNA backbone phosphate oxygens. The integer contact count drives
the score; a residue-weighted sum (ARG/GLN/TYR favored, LYS ignored at weight 0)
following the ``dbp_design`` phosphate H-bond weights is reported as auxiliary
metadata.

Examples:
    >>> from proto_language.core import Segment
    >>> protein = Segment(length=80, sequence_type="protein")
    >>> operator = Segment(sequence="ACGTACGTACGT", sequence_type="dna")
    >>> phosphate_contact = Constraint(
    ...     inputs=[protein, operator],
    ...     function=dna_phosphate_contact_constraint,
    ...     function_config={"structure_tool": "alphafold3", "desired_phosphate_contacts": 2},
    ... )
"""

import logging
from collections import defaultdict
from typing import Any, Literal

import numpy as np

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.dna_binding_structure_helper import (
    resolve_structure_paths,
)
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY
from proto_language.utils.base import ConfigField

logger = logging.getLogger(__name__)

# ── Atom classification ───────────────────────────────────────────────────

_AA_NAMES = {
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
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

_DNA_NAMES = {"DA", "DC", "DG", "DT", "A", "C", "G", "T"}

# Protein sidechain polar atoms capable of H-bonding (reused from the
# base-contact-quality constraint).
_POLAR_SC_ATOMS: dict[str, set[str]] = {
    "ARG": {"NH1", "NH2", "NE"},
    "LYS": {"NZ"},
    "ASN": {"OD1", "ND2"},
    "GLN": {"OE1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "HIS": {"ND1", "NE2"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "TYR": {"OH"},
    "TRP": {"NE1"},
}

# Per-residue phosphate-contact weights, ported from dbp_design
# ``_PHOSPHATE_SCORE_WEIGHTS`` (dbp_design_metrics_constraint.py): ARG/GLN/TYR
# favored, LYS ignored (weight 0), ASN/SER/THR/HIS lightly rewarded.
_PHOSPHATE_RESIDUE_WEIGHTS: dict[str, float] = {
    "ARG": 7.0,
    "LYS": 0.0,
    "GLN": 10.0,
    "TYR": 4.0,
    "ASN": 1.0,
    "SER": 1.0,
    "THR": 1.0,
    "HIS": 1.0,
}

# A single XYZ heavy-atom coordinate.
Coord = np.ndarray[Any, np.dtype[np.float64]]


# ── Config ────────────────────────────────────────────────────────────────


class DNAPhosphateContactConfig(StructureBasedConstraintConfig):
    """Config for geometry-based DNA phosphate contact scoring.

    Scores protein-DNA complexes based on protein polar side-chain H-bond
    contacts to DNA backbone phosphate oxygens, measured directly from
    heavy-atom distances in a predicted complex PDB without requiring Rosetta
    relaxation. Mirrors the AlphaFold3 phosphate-contact H-bond term (separate
    from the base-contact H-bond term). Inherits the structure-prediction tool
    selection and per-tool configs from ``StructureBasedConstraintConfig``.

    Attributes:
        contact_cutoff (float): Heavy-atom distance cutoff (Ångströms) for phosphate contacts.
        desired_phosphate_contacts (int): Target number of unique protein-DNA phosphate
            H-bond contacts. Score is 0 when this integer-count target is met, rising
            toward 1 as it falls short.
        phosphate_atoms (list[str]): DNA backbone phosphate atom names treated as eligible
            phosphate H-bond acceptors (default the two non-bridging oxygens OP1/OP2 plus
            their legacy O1P/O2P aliases).
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Structure-prediction tool; must be DNA-capable (default alphafold3).
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    contact_cutoff: float = ConfigField(
        title="Contact Cutoff (A)",
        default=3.5,
        gt=0.0,
        description="Heavy-atom distance cutoff for phosphate contacts.",
    )
    desired_phosphate_contacts: int = ConfigField(
        title="Desired Phosphate Contacts",
        default=2,
        ge=0,
        description="Target number of unique phosphate H-bond contacts; 0 score when met.",
    )
    phosphate_atoms: list[str] = ConfigField(
        title="Phosphate Atom Names",
        default=["OP1", "OP2", "O1P", "O2P"],
        description="DNA backbone phosphate atom names treated as eligible H-bond acceptors (OP1/OP2 + legacy O1P/O2P).",
    )
    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )


# ── PDB analysis ─────────────────────────────────────────────────────────


def _analyze_phosphate_contacts(
    pdb_path: str,
    cutoff: float,
    phosphate_atoms: set[str],
) -> dict[str, Any]:
    """Analyze protein-DNA phosphate contacts from a PDB file.

    Parses heavy-atom coordinates from a protein-DNA complex PDB and counts
    protein polar sidechain atoms within ``cutoff`` Ångströms of DNA backbone
    phosphate oxygens. A residue-weighted sum is also computed as auxiliary
    metadata; the integer contact count drives scoring.

    Args:
        pdb_path (str): Path to the protein-DNA complex PDB file.
        cutoff (float): Heavy-atom distance cutoff in Ångströms.
        phosphate_atoms (set[str]): DNA backbone atom names treated as phosphate
            H-bond acceptors.

    Returns:
        dict[str, Any]: Contact summary with ``n_phosphate_contacts`` (unique
            protein-residue-to-phosphate-residue contact pairs that drive the
            score), ``weighted_phosphate_score`` (residue-weighted sum, auxiliary),
            ``n_unique_residues``, and ``contacting_types``.
    """
    # Residues are identified by (chain, resseq) so equal residue numbers on
    # different chains (e.g. the two strands of a dsDNA operator) never collapse.
    prot_polar: list[tuple[str, str, tuple[str, int], Coord]] = []  # (atom, resname, res_id, xyz)
    dna_phos: list[tuple[str, str, tuple[str, int], Coord]] = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip().upper()
            resname = line[17:20].strip().upper()
            try:
                res_id = (line[21], int(line[22:26].strip()))
                xyz = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError:
                continue

            if resname in _AA_NAMES and atom_name in _POLAR_SC_ATOMS.get(resname, set()):
                prot_polar.append((atom_name, resname, res_id, xyz))
            elif resname in _DNA_NAMES and atom_name in phosphate_atoms:
                dna_phos.append((atom_name, resname, res_id, xyz))

    if not prot_polar or not dna_phos:
        return {
            "n_phosphate_contacts": 0,
            "weighted_phosphate_score": 0.0,
            "n_unique_residues": 0,
        }

    cutoff_sq = cutoff**2

    # (prot_resname, prot_res_id) → set of contacted DNA phosphate res_ids.
    contact_residues: dict[tuple[str, tuple[str, int]], set[tuple[str, int]]] = defaultdict(set)

    for _pa_name, p_resname, p_res_id, p_xyz in prot_polar:
        for _da_name, _d_resname, d_res_id, d_xyz in dna_phos:
            if np.sum((p_xyz - d_xyz) ** 2) <= cutoff_sq:
                contact_residues[(p_resname, p_res_id)].add(d_res_id)

    # Count of unique protein-residue ↔ DNA-phosphate-residue contact pairs.
    n_contacts = sum(len(v) for v in contact_residues.values())

    contacting_types: dict[str, int] = defaultdict(int)
    weighted_score = 0.0
    for (resname, _), dna_res in contact_residues.items():
        contacting_types[resname] += 1
        weighted_score += _PHOSPHATE_RESIDUE_WEIGHTS.get(resname, 0.0) * len(dna_res)

    return {
        "n_phosphate_contacts": n_contacts,
        "weighted_phosphate_score": weighted_score,
        "n_unique_residues": len(contact_residues),
        "contacting_types": dict(contacting_types),
    }


# ── Constraint ────────────────────────────────────────────────────────────


@constraint(
    key="dna-phosphate-contact",
    label="DNA Phosphate Contact",
    config=DNAPhosphateContactConfig,
    description=(
        "Score protein-DNA backbone phosphate H-bond contacts from PDB "
        "geometry. Mirrors the AlphaFold3 phosphate-contact H-bond term, "
        "rewarding protein polar sidechain contacts to DNA phosphate oxygens "
        "(optionally weighted by residue type). Does not require Rosetta "
        "relaxation."
    ),
    uses_gpu=True,
    tools_called=["alphafold3-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def dna_phosphate_contact_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: DNAPhosphateContactConfig,
) -> list[ConstraintOutput]:
    """Score protein-DNA phosphate contacts from a predicted complex PDB.

    Resolves (reuses or predicts) a protein-operator complex PDB per candidate
    tuple, then counts protein polar sidechain H-bond contacts to DNA backbone
    phosphate oxygens directly from heavy-atom geometry. The score is 0 when the
    integer phosphate-contact count meets the target and rises toward 1 as it
    falls short, mirroring the base-contact-quality penalty mapping.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-candidate tuples of
            input sequences (protein chain(s) + DNA chain(s)) to fold into a complex.
        config (DNAPhosphateContactConfig): Constraint configuration controlling
            the structure tool, contact cutoff, eligible phosphate atoms, and target.

    Returns:
        list[ConstraintOutput]: Per-candidate score in ``[0, 1]`` (lower is
            better) with phosphate-contact metadata (``n_phosphate_contacts``,
            ``weighted_phosphate_score``, ``n_unique_residues``,
            ``contacting_types``, and ``pdb_path``). Candidates whose structure
            could not be resolved receive ``MAX_ENERGY``.
    """
    if not input_sequences:
        return []

    candidates = list(input_sequences)
    pdb_paths = resolve_structure_paths(
        candidates,
        structure_tool=config.structure_tool,
        tool_config=config.tool_config,
    )

    phosphate_atoms = set(config.phosphate_atoms)

    results: list[ConstraintOutput] = []

    for pdb_path in pdb_paths:
        if not pdb_path:
            logger.warning("dna-phosphate-contact: structure could not be resolved; scoring MAX_ENERGY.")
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"phosphate_contact_error": "structure_unresolved"},
                )
            )
            continue

        result = _analyze_phosphate_contacts(
            pdb_path,
            config.contact_cutoff,
            phosphate_atoms,
        )

        # Penalty maps to 0 when the integer contact count meets the target,
        # rising toward 1 as it falls short.
        target = float(max(1, config.desired_phosphate_contacts))
        score = float(
            np.clip(
                (target - result["n_phosphate_contacts"]) / target,
                0.0,
                1.0,
            )
        )

        metadata = {
            "n_phosphate_contacts": result["n_phosphate_contacts"],
            "weighted_phosphate_score": result["weighted_phosphate_score"],
            "n_unique_residues": result["n_unique_residues"],
            "contacting_types": result.get("contacting_types", {}),
            "pdb_path": pdb_path,
        }
        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results
