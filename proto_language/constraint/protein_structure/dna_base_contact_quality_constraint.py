"""DNA base contact quality constraint.

Lightweight geometry-based scoring of protein-DNA base contacts. Measures
base-specific H-bond quality directly from heavy-atom distances in a predicted
protein-operator complex PDB, WITHOUT requiring Rosetta relaxation. Scores
bidentate contacts, contacting-residue diversity, and base H-bond contact count,
rewarding designs that read bases with GLN/ASN/SER/THR/HIS/TYR/TRP over ARG-only
contacts.

Examples:
    >>> from proto_language.core import Segment
    >>> protein = Segment(length=80, sequence_type="protein")
    >>> operator = Segment(sequence="ACGTACGTACGT", sequence_type="dna")
    >>> base_contact = Constraint(
    ...     inputs=[protein, operator],
    ...     function=dna_base_contact_quality_constraint,
    ...     function_config={"structure_tool": "alphafold3", "desired_bidentate": 2},
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

# Protein sidechain polar atoms capable of H-bonding to bases
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

# DNA base atoms that participate in H-bonds (major/minor groove)
_BASE_HBOND_ATOMS: dict[str, set[str]] = {
    "DG": {"O6", "N7", "N2"},
    "DA": {"N6", "N7", "N1"},
    "DC": {"N4", "O2", "N3"},
    "DT": {"O4", "O2", "N3"},
    "G": {"O6", "N7", "N2"},
    "A": {"N6", "N7", "N1"},
    "C": {"N4", "O2", "N3"},
    "T": {"O4", "O2", "N3"},
}

_DNA_BACKBONE_ATOMS = {
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
}

# Residues that provide base-specific readout (vs nonspecific charge)
_SPECIFIC_RESIDUES = {"GLN", "ASN", "SER", "THR", "HIS", "TYR", "TRP"}

# A single XYZ heavy-atom coordinate.
Coord = np.ndarray[Any, np.dtype[np.float64]]


# ── Config ────────────────────────────────────────────────────────────────


class DNABaseContactQualityConfig(StructureBasedConstraintConfig):
    """Config for geometry-based DNA base contact quality scoring.

    Scores protein-DNA complexes based on the quality of base-specific contacts,
    measured directly from heavy-atom distances in a predicted complex PDB
    without requiring Rosetta relaxation. Inherits the structure-prediction tool
    selection and per-tool configs from ``StructureBasedConstraintConfig``.

    Attributes:
        contact_cutoff (float): Heavy-atom distance cutoff (Ångströms) for base contacts.
        desired_bidentate (int): Target number of bidentate contacts (one residue reading
            2+ H-bond atoms on the same base). Score component is 0 when achieved.
        desired_base_contacts (int): Target number of polar sidechain-to-base contacts.
        desired_unique_residues (int): Target number of unique protein residues contacting bases.
        diversity_bonus_weight (float): Weight of the diversity term, rewarding
            GLN/ASN/SER/THR/HIS/TYR/TRP base readout over ARG-only contacts.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): DNA-capable structure-prediction tool (default alphafold3).
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )
    contact_cutoff: float = ConfigField(
        title="Contact Cutoff (A)",
        default=3.5,
        gt=0.0,
        description="Heavy-atom distance cutoff for base contacts.",
    )
    desired_bidentate: int = ConfigField(
        title="Desired Bidentate Contacts",
        default=2,
        ge=0,
        description=(
            "Target bidentate-contact count (one residue H-bonding 2+ atoms on a base); 0 component when met."
        ),
    )
    desired_base_contacts: int = ConfigField(
        title="Desired Base Contacts",
        default=8,
        ge=0,
        description="Target number of polar sidechain-to-base contacts.",
    )
    desired_unique_residues: int = ConfigField(
        title="Desired Unique Residues",
        default=4,
        ge=0,
        description="Target number of unique protein residues contacting bases.",
    )
    diversity_bonus_weight: float = ConfigField(
        title="Diversity Bonus Weight",
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of the diversity term rewarding base-specific readout over nonspecific ARG/LYS charge.",
    )


# ── PDB analysis ─────────────────────────────────────────────────────────


def _analyze_base_contacts(
    pdb_path: str,
    cutoff: float,
) -> dict[str, Any]:
    """Analyze base-specific contacts from a PDB file.

    Parses heavy-atom coordinates from a protein-DNA complex PDB and measures
    polar sidechain-to-base contacts, bidentate readout, and contacting-residue
    diversity within ``cutoff`` Ångströms.

    Args:
        pdb_path (str): Path to the protein-DNA complex PDB file.
        cutoff (float): Heavy-atom distance cutoff in Ångströms.

    Returns:
        dict[str, Any]: Contact summary with ``n_base_contacts``, ``n_bidentate``,
            ``n_unique_residues``, ``n_specific_residues``, ``pct_arg``,
            ``diversity_score``, ``parse_failed`` (True only when no protein polar
            atoms AND no DNA base atoms parsed, i.e. an empty/unparseable PDB), and
            (when contacts exist) ``contacting_types``.
    """
    # Residues are identified by (chain, resseq) so equal residue numbers on
    # different chains (e.g. the two strands of a dsDNA operator, or homodimer
    # protein chains) never collapse into one bucket.
    prot_polar: list[tuple[str, str, tuple[str, int], Coord]] = []  # (atom, resname, res_id, xyz)
    dna_hbond: list[tuple[str, str, tuple[str, int], Coord]] = []
    dna_base_all: list[tuple[str, str, tuple[str, int], Coord]] = []

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
            elif resname in _DNA_NAMES:
                if atom_name in _BASE_HBOND_ATOMS.get(resname, set()):
                    dna_hbond.append((atom_name, resname, res_id, xyz))
                if atom_name not in _DNA_BACKBONE_ATOMS and not atom_name.startswith("H"):
                    dna_base_all.append((atom_name, resname, res_id, xyz))

    if not prot_polar or not dna_base_all:
        return {
            "n_base_contacts": 0,
            "n_bidentate": 0,
            "n_unique_residues": 0,
            "n_specific_residues": 0,
            "pct_arg": 100.0,
            "diversity_score": 0.0,
            # No atoms on EITHER side -> the PDB is empty/unparseable, not a real
            # zero-contact structure.
            "parse_failed": not prot_polar and not dna_base_all,
        }

    cutoff_sq = cutoff**2

    # Count all polar sidechain ↔ base atom contacts
    # (resname, prot_res_id) → set of (dna_res_id, dna_atom)
    contact_residues: dict[tuple[str, tuple[str, int]], set[tuple[tuple[str, int], str]]] = defaultdict(set)
    total_contacts = 0

    for _pa_name, p_resname, p_res_id, p_xyz in prot_polar:
        for da_name, _d_resname, d_res_id, d_xyz in dna_base_all:
            if np.sum((p_xyz - d_xyz) ** 2) <= cutoff_sq:
                contact_residues[(p_resname, p_res_id)].add((d_res_id, da_name))
                total_contacts += 1

    # Bidentate: one protein residue contacting 2+ H-bond atoms on same DNA base
    bidentate_map: dict[tuple[str, tuple[str, int]], dict[tuple[str, int], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for _pa_name, p_resname, p_res_id, p_xyz in prot_polar:
        for da_name, _d_resname, d_res_id, d_xyz in dna_hbond:
            if np.sum((p_xyz - d_xyz) ** 2) <= cutoff_sq:
                bidentate_map[(p_resname, p_res_id)][d_res_id].add(da_name)

    n_bidentate = sum(
        1 for prot_res in bidentate_map.values() for dna_atoms in prot_res.values() if len(dna_atoms) >= 2
    )

    # Residue type diversity
    contacting_types: dict[str, int] = defaultdict(int)
    for (resname, _), dna_contacts in contact_residues.items():
        if dna_contacts:
            contacting_types[resname] += 1

    n_unique = len(contact_residues)
    n_specific = sum(v for k, v in contacting_types.items() if k in _SPECIFIC_RESIDUES)
    n_arg = contacting_types.get("ARG", 0)
    pct_arg = 100 * n_arg / max(1, n_unique)

    # Diversity score: fraction of contacting residues that provide base-specific
    # readout (_SPECIFIC_RESIDUES) rather than nonspecific ARG/LYS charge.
    diversity_score = n_specific / max(1, n_unique)

    return {
        "n_base_contacts": total_contacts,
        "n_bidentate": n_bidentate,
        "n_unique_residues": n_unique,
        "n_specific_residues": n_specific,
        "pct_arg": pct_arg,
        "diversity_score": diversity_score,
        "parse_failed": False,
        "contacting_types": dict(contacting_types),
    }


# ── Constraint ────────────────────────────────────────────────────────────


@constraint(
    key="dna-base-contact-quality",
    label="DNA Base Contact Quality",
    config=DNABaseContactQualityConfig,
    description=(
        "Score protein-DNA base contact quality from PDB geometry. "
        "Rewards bidentate contacts, diverse readout residues (GLN/ASN/SER/THR/HIS/TYR/TRP "
        "over ARG-only), and sufficient base-specific H-bond contacts. "
        "Does not require Rosetta relaxation."
    ),
    uses_gpu=True,
    tools_called=["alphafold3-prediction", "boltz2-prediction"],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def dna_base_contact_quality_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: DNABaseContactQualityConfig,
) -> list[ConstraintOutput]:
    """Score protein-DNA base contact quality from a predicted complex PDB.

    Resolves (reuses or predicts) a protein-operator complex PDB per candidate
    tuple, then scores base-specific contact quality directly from heavy-atom
    geometry. The score is a weighted combination of bidentate deficit (0.35),
    base-contact deficit (0.30), unique-residue deficit (0.15), and a diversity
    deficit weighted by ``config.diversity_bonus_weight`` (default 0.3), all in
    ``[0, 1]`` where 0 is best.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-candidate tuples of
            input sequences (protein chain(s) + DNA chain(s)) to fold into a complex.
        config (DNABaseContactQualityConfig): Constraint configuration controlling
            the structure tool and contact-quality targets.

    Returns:
        list[ConstraintOutput]: Per-candidate score in ``[0, 1]`` (lower is
            better) with contact-quality metadata (``n_base_contacts``,
            ``n_bidentate``, ``n_unique_residues``, ``n_specific_residues``,
            ``pct_arg``, ``diversity_score``, per-component deficits, and
            ``pdb_path``). Candidates whose structure could not be resolved, or
            whose PDB is empty/unparseable, receive ``MAX_ENERGY``.
    """
    if not input_sequences:
        return []

    candidates = list(input_sequences)
    pdb_paths = resolve_structure_paths(
        candidates,
        structure_tool=config.structure_tool,
        tool_config=config.tool_config,
    )

    results: list[ConstraintOutput] = []

    for pdb_path in pdb_paths:
        if not pdb_path:
            results.append(ConstraintOutput(score=MAX_ENERGY))
            continue

        result = _analyze_base_contacts(pdb_path, config.contact_cutoff)

        if result["parse_failed"]:
            logger.warning("dna-base-contact-quality: no atoms parsed from PDB %s", pdb_path)
            results.append(
                ConstraintOutput(
                    score=MAX_ENERGY,
                    metadata={"dna_base_contact_quality_error": "empty or unparseable PDB", "pdb_path": pdb_path},
                )
            )
            continue

        # Score components (each 0 = good, 1 = bad)
        bidentate_deficit = float(
            np.clip(
                (config.desired_bidentate - result["n_bidentate"]) / max(1, config.desired_bidentate),
                0.0,
                1.0,
            )
        )

        contacts_deficit = float(
            np.clip(
                (config.desired_base_contacts - result["n_base_contacts"]) / max(1, config.desired_base_contacts),
                0.0,
                1.0,
            )
        )

        residues_deficit = float(
            np.clip(
                (config.desired_unique_residues - result["n_unique_residues"]) / max(1, config.desired_unique_residues),
                0.0,
                1.0,
            )
        )

        # Diversity deficit: higher when contacts lean on nonspecific ARG/LYS charge
        diversity_deficit = 1.0 - result["diversity_score"]

        # Weighted combination (diversity_bonus_weight is the sole diversity weight).
        # Weights deliberately need not sum to 1: with the default 0.3 they total
        # 1.10, so a maximally-bad design saturates at 1.0 after the clip.
        score = float(
            np.clip(
                0.35 * bidentate_deficit
                + 0.30 * contacts_deficit
                + 0.15 * residues_deficit
                + config.diversity_bonus_weight * diversity_deficit,
                0.0,
                1.0,
            )
        )

        metadata = {
            "n_base_contacts": result["n_base_contacts"],
            "n_bidentate": result["n_bidentate"],
            "n_unique_residues": result["n_unique_residues"],
            "n_specific_residues": result["n_specific_residues"],
            "pct_arg": result["pct_arg"],
            "diversity_score": result["diversity_score"],
            "bidentate_deficit": bidentate_deficit,
            "contacts_deficit": contacts_deficit,
            "residues_deficit": residues_deficit,
            "diversity_deficit": diversity_deficit,
            "pdb_path": pdb_path,
        }
        results.append(ConstraintOutput(score=score, metadata=metadata))

    return results
