"""Template-guided superposition: build Stage-2 start models from Stage-1 survivors.

The ProtoRepressor pipeline (``protorepressor.py``) prioritizes repressor-operator
candidate pairs in Stage 1 (cofolding each candidate against its operator). Before
Stage 2's LigandMPNN sequence design, each surviving candidate must be positioned on
its operator with design-ready geometry. This module performs that template-guided
superposition in-pipeline, so no per-operator start-model PDBs are shipped -- only the
20 generic natural HTH-DNA crystal templates.

For one (candidate, operator, template):

    1. Build idealized B-form operator DNA (operator + flanking context) with X3DNA
       ``fiber`` (``build_bform_operator_dna``).
    2. Superpose the operator's half-site onto the template DNA, keeping the template's
       native (clash-free) protein-DNA pose (``_superpose_operator_onto_template``).
    3. Superpose the candidate's HTH motif (positioning + recognition helix, Calpha) onto
       the template protein's, placing the candidate's HTH in the major groove
       (``detect_recognition_helices`` + Kabsch). The start model is monomeric (the
       homodimer is expressed downstream by the cofold constraints).
    4. Emit the assembled candidate-protein + operator-DNA complex as a PDB string
       (``build_start_model``), which LigandMPNN consumes directly (no temp file).

The 20 docking templates are natural HTH-DNA crystal structures (PDB 1QPI, 1R8D, 2KEI,
2OR1, 2VZ4, 2XRO, 2ZHG, 3BDN, 3ZQL, 4EGY, 4EGZ, 4L62, 4PXI, 4WLS, 5D8C, 5YEJ, 6JGW,
7TEA, 7TEC, 8SVD).

The geometry primitives (Kabsch fit, B-form generation, recognition-helix detection)
are pure and locally testable; run ``python protorepressor_templating.py`` for a
self-test on 2OR1. Producing a *biologically valid* interface still depends on the
downstream cofold/Rosetta gates -- this step only supplies a registered starting pose.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

logger = logging.getLogger(__name__)

# DNA sugar-phosphate backbone atom names; every other atom in a DNA residue is a base atom.
_DNA_SUGAR_PHOSPHATE = frozenset({"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'"})
# Canonical DNA residue names (deoxyribonucleotides).
_DNA_RESIDUES = frozenset({"DA", "DC", "DG", "DT", "DU", "DI", "DN"})
# Atom used as the per-nucleotide registration handle for DNA backbone superposition.
_DNA_REGISTER_ATOM = "C1'"
# Heavy-atom contact cutoff (Angstrom) for protein-residue / DNA-base contacts.
_CONTACT_CUTOFF = 4.5
# A palindromic operator is bound by one homodimer: keep at most this many protomers.
_MAX_PROTOMERS = 2
# Max Calpha-to-operator-base distance (Angstrom) for a protomer to count as bound to the
# operator after posing -- filters extra asymmetric-unit complexes in multi-copy crystals.
_PROTOMER_ON_OPERATOR_CUTOFF = 16.0
# X3DNA install root for the ``fiber`` B-form builder, from the standard ``X3DNA`` environment
# variable (empty if unset). Passed through to the x3dna-fiber tool, which otherwise resolves
# X3DNA itself (``$X3DNA`` / ``PROTO_X3DNA_WEIGHTS_DIR``); user-provisioned (CC-BY-NC-4.0).
DEFAULT_X3DNA_ROOT = os.environ.get("X3DNA", "")


# --------------------------------------------------------------------------------------
# Rigid-body geometry (Kabsch).
# --------------------------------------------------------------------------------------
def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform mapping ``mobile`` points onto ``target`` (Kabsch/SVD).

    Args:
        mobile (np.ndarray): ``(N, 3)`` source coordinates to be moved.
        target (np.ndarray): ``(N, 3)`` reference coordinates, row-aligned to ``mobile``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Rotation ``R`` (``3x3``) and translation ``t``
            (``3,``) such that ``mobile @ R.T + t`` is the least-squares fit onto
            ``target`` (a proper rotation; reflections are corrected).
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"kabsch expects matching (N, 3) arrays, got {mobile.shape} and {target.shape}.")
    if mobile.shape[0] < 3:
        raise ValueError(f"kabsch needs >= 3 points to define a rigid transform, got {mobile.shape[0]}.")
    mob_center = mobile.mean(axis=0)
    tgt_center = target.mean(axis=0)
    mob = mobile - mob_center
    tgt = target - tgt_center
    covariance = mob.T @ tgt
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag(np.array([1.0, 1.0, d]))
    rotation = vt.T @ correction @ u.T
    translation = tgt_center - rotation @ mob_center
    return rotation, translation


def rmsd_after(mobile: np.ndarray, target: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> float:
    """RMSD between ``target`` and ``mobile`` after applying ``(rotation, translation)``."""
    moved = mobile @ rotation.T + translation
    return float(np.sqrt(np.mean(np.sum((moved - target) ** 2, axis=1))))


def apply_transform(structure: gemmi.Structure, rotation: np.ndarray, translation: np.ndarray) -> None:
    """Apply ``rotation`` then ``translation`` to every atom of ``structure`` in place."""
    mat = gemmi.Mat33(rotation.tolist())
    vec = gemmi.Vec3(*(float(x) for x in translation))
    transform = gemmi.Transform(mat, vec)
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.pos = gemmi.Position(transform.apply(atom.pos))


# --------------------------------------------------------------------------------------
# Chain classification and atom extraction on gemmi structures.
# --------------------------------------------------------------------------------------
def _is_dna_residue(residue: gemmi.Residue) -> bool:
    """Whether a gemmi residue is a (deoxy)nucleotide by residue name."""
    return residue.name.strip() in _DNA_RESIDUES


def _is_amino_acid(residue: gemmi.Residue) -> bool:
    """Whether a gemmi residue is a standard/modified amino acid (via gemmi's residue table)."""
    return gemmi.find_tabulated_residue(residue.name).is_amino_acid()


def _unique_chains(model: gemmi.Model) -> list[gemmi.Chain]:
    """First occurrence of each chain name (gemmi keeps duplicate-named asymmetric-unit copies)."""
    seen: set[str] = set()
    chains: list[gemmi.Chain] = []
    for chain in model:
        if chain.name not in seen:
            seen.add(chain.name)
            chains.append(chain)
    return chains


def _classify_chains(model: gemmi.Model) -> tuple[list[gemmi.Chain], list[gemmi.Chain]]:
    """Split a model's unique chains into (protein_chains, dna_chains) by residue content."""
    protein: list[gemmi.Chain] = []
    dna: list[gemmi.Chain] = []
    for chain in _unique_chains(model):
        residues = [r for r in chain if _is_amino_acid(r) or _is_dna_residue(r)]
        if not residues:
            continue
        n_dna = sum(1 for r in chain if _is_dna_residue(r))
        (dna if n_dna > len(residues) / 2 else protein).append(chain)
    return protein, dna


def _ca_coords(chain: gemmi.Chain) -> tuple[np.ndarray, list[int], list[gemmi.Residue]]:
    """Return (Calpha coordinates ``(N, 3)``, residue seqids, residues) for amino acids with a CA.

    All three lists share one index space (amino-acid residues that have a CA atom), so callers
    can slice coords/seqids and the residues with the same segment indices.
    """
    coords: list[list[float]] = []
    seqids: list[int] = []
    residues: list[gemmi.Residue] = []
    for residue in chain:
        if not _is_amino_acid(residue):
            continue
        atom = residue.find_atom("CA", "*")
        if atom is not None:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            seqids.append(residue.seqid.num)
            residues.append(residue)
    return np.asarray(coords, dtype=float), seqids, residues


def _dna_base_atom_coords(chains: list[gemmi.Chain]) -> np.ndarray:
    """Heavy base-atom coordinates ``(M, 3)`` across DNA chains (excludes sugar-phosphate)."""
    coords: list[list[float]] = []
    for chain in chains:
        for residue in chain:
            if not _is_dna_residue(residue):
                continue
            for atom in residue:
                if atom.name not in _DNA_SUGAR_PHOSPHATE and not atom.is_hydrogen():
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.asarray(coords, dtype=float)


def _register_atom_coords(chain: gemmi.Chain) -> tuple[np.ndarray, list[int]]:
    """Return (C1' coordinates ``(N, 3)``, residue seqids) along a DNA chain in residue order."""
    coords: list[list[float]] = []
    seqids: list[int] = []
    for residue in chain:
        if not _is_dna_residue(residue):
            continue
        atom = residue.find_atom(_DNA_REGISTER_ATOM, "*")
        if atom is not None:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            seqids.append(residue.seqid.num)
    return np.asarray(coords, dtype=float), seqids


# --------------------------------------------------------------------------------------
# Recognition-helix detection (DSSP-style SSE intersected with DNA-base contacts).
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RecognitionHelix:
    """A protein chain's HTH recognition helix: the major-groove-inserting alpha helix.

    Attributes:
        chain_name (str): Protein chain the helix belongs to.
        start_seqid (int): First residue seqid of the helix (inclusive).
        end_seqid (int): Last residue seqid of the helix (inclusive).
        ca_coords (np.ndarray): ``(L, 3)`` Calpha coordinates of the helix residues.
        base_contacts (int): Number of helix heavy-atom / DNA-base contacts (selection score).
        motif_ca_coords (np.ndarray): ``(M, 3)`` Calpha coordinates of the full HTH motif
            (the positioning helix + turn + recognition helix); used for superposition because
            two helices at an angle pin the orientation that a single helix's Calpha cannot.
    """

    chain_name: str
    start_seqid: int
    end_seqid: int
    ca_coords: np.ndarray
    base_contacts: int
    motif_ca_coords: np.ndarray


def _helical_segments(chain_name: str, residues: list[gemmi.Residue]) -> list[tuple[int, int]]:
    """Contiguous alpha-helix residue-index ranges over ``residues``, via biotite SSE.

    Returns 0-based [start, end) index ranges into ``residues`` (the CA-bearing amino acids
    from ``_ca_coords``), so callers can slice coords/seqids with the same indices.
    """
    import biotite.structure as bs
    import biotite.structure.io.pdbx as pdbx

    # Round-trip just these residues through a minimal CIF so biotite can annotate SSE.
    single = gemmi.Structure()
    model = gemmi.Model("1")
    new_chain = gemmi.Chain(chain_name)
    for residue in residues:
        new_chain.add_residue(residue)
    model.add_chain(new_chain)
    single.add_model(model)

    # biotite reads from CIF text; gemmi writes mmCIF.
    cif_text = single.make_mmcif_document().as_string()
    handle = pdbx.CIFFile.read(_text_stream(cif_text))
    atoms = pdbx.get_structure(handle, model=1)
    peptide = atoms[bs.filter_amino_acids(atoms)]
    sse = bs.annotate_sse(peptide)  # one label ('a'/'b'/'c') per residue, in order
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, label in enumerate(sse):
        if label == "a" and start is None:
            start = idx
        elif label != "a" and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(sse)))
    return segments


def _text_stream(text: str):  # noqa: ANN202 - tiny io helper
    """Wrap a string as a file-like object for biotite's CIF reader."""
    import io

    return io.StringIO(text)


def _best_helix_for_chain(chain: gemmi.Chain, base_coords: np.ndarray) -> RecognitionHelix | None:
    """The chain's alpha helix with the most DNA-base contacts, or None if none contact DNA."""
    ca_coords, seqids, residues = _ca_coords(chain)
    if len(seqids) < 3:
        return None
    # segments index into the same CA-bearing residue list as ca_coords/seqids.
    segments = [(s, e) for (s, e) in _helical_segments(chain.name, residues) if e - s >= 4]  # ignore 3_10 stubs
    best: RecognitionHelix | None = None
    for index, (seg_start, seg_end) in enumerate(segments):
        contacts = _count_base_contacts(residues[seg_start:seg_end], base_coords)
        if contacts == 0:
            continue
        if best is None or contacts > best.base_contacts:
            # HTH motif = the immediately preceding helix (positioning helix) + this recognition
            # helix; falls back to the recognition helix alone if it is the first helix.
            motif_start = segments[index - 1][0] if index > 0 else seg_start
            best = RecognitionHelix(
                chain_name=chain.name,
                start_seqid=seqids[seg_start],
                end_seqid=seqids[seg_end - 1],
                ca_coords=ca_coords[seg_start:seg_end],
                base_contacts=contacts,
                motif_ca_coords=ca_coords[motif_start:seg_end],
            )
    return best


def detect_recognition_helices(structure: gemmi.Structure) -> list[RecognitionHelix]:
    """Find one HTH recognition helix per DNA-contacting protein chain (a protomer each).

    Args:
        structure (gemmi.Structure): A protein-DNA complex (crystal template or Stage-1
            cofold) with at least one protein chain and one DNA chain.

    Returns:
        list[RecognitionHelix]: One recognition helix (most DNA-base contacts) per protein
            chain that contacts DNA, in chain order. For a homodimer this is two helices.

    Raises:
        ValueError: If there is no DNA chain, no protein chain, or no DNA-contacting helix.
    """
    model = structure[0]
    protein_chains, dna_chains = _classify_chains(model)
    if not dna_chains:
        raise ValueError("detect_recognition_helices: structure has no DNA chain.")
    if not protein_chains:
        raise ValueError("detect_recognition_helices: structure has no protein chain.")
    base_coords = _dna_base_atom_coords(dna_chains)
    if base_coords.size == 0:
        raise ValueError("detect_recognition_helices: DNA chains have no base atoms.")
    helices = [h for h in (_best_helix_for_chain(c, base_coords) for c in protein_chains) if h is not None]
    if not helices:
        raise ValueError("detect_recognition_helices: no DNA-contacting alpha helix found.")
    return helices


def detect_recognition_helix(structure: gemmi.Structure, chain_name: str | None = None) -> RecognitionHelix:
    """The single best recognition helix (most DNA-base contacts), optionally for one chain."""
    helices = detect_recognition_helices(structure)
    if chain_name is not None:
        helices = [h for h in helices if h.chain_name == chain_name]
        if not helices:
            raise ValueError(f"detect_recognition_helix: protein chain {chain_name!r} not found / not contacting.")
    return max(helices, key=lambda h: h.base_contacts)


def _count_base_contacts(residues: list[gemmi.Residue], base_coords: np.ndarray) -> int:
    """Number of residues with any heavy atom within the cutoff of a DNA base atom."""
    contacts = 0
    for residue in residues:
        residue_atoms = np.asarray(
            [[a.pos.x, a.pos.y, a.pos.z] for a in residue if not a.is_hydrogen()],
            dtype=float,
        )
        if residue_atoms.size == 0:
            continue
        dists = np.linalg.norm(residue_atoms[:, None, :] - base_coords[None, :, :], axis=2)
        if np.any(dists <= _CONTACT_CUTOFF):
            contacts += 1
    return contacts


# --------------------------------------------------------------------------------------
# Idealized B-form operator DNA via X3DNA fiber.
# --------------------------------------------------------------------------------------
def build_bform_operator_dna(scaffold: str, x3dna_root: str = DEFAULT_X3DNA_ROOT) -> gemmi.Structure:
    """Build an idealized B-form dsDNA duplex for the operator scaffold via the x3dna-fiber tool.

    Dispatches the proto-tools ``x3dna-fiber`` tool (which wraps X3DNA ``fiber``) so the
    pipeline runs through proto_tools rather than invoking the binary directly. X3DNA is
    user-provisioned (CC-BY-NC-4.0); the tool resolves it from ``x3dna_root`` / ``$X3DNA`` /
    the tool cache.

    Args:
        scaffold (str): Operator DNA sequence (operator motif + flanking context), 5'->3'.
        x3dna_root (str): X3DNA v2.4 install root passed to the tool's ``x3dna_dir`` config.

    Returns:
        gemmi.Structure: Two-chain B-form duplex (sense + antisense) in the X3DNA frame.

    Raises:
        ValueError: If the scaffold is not non-empty ACGT DNA.
    """
    from proto_tools.tools import X3DNAFiberConfig, X3DNAFiberInput, run_x3dna_fiber

    seq = scaffold.strip().upper()
    if not seq or any(base not in "ACGT" for base in seq):
        raise ValueError(f"build_bform_operator_dna: scaffold must be non-empty ACGT DNA, got {scaffold!r}.")
    output = run_x3dna_fiber(
        X3DNAFiberInput(sequences=[seq]),
        X3DNAFiberConfig(form="B-DNA", x3dna_dir=x3dna_root or None),
    )
    return gemmi.read_pdb_string(output.structures[0].structure_pdb)


# --------------------------------------------------------------------------------------
# Template DNA -> operator DNA registration and protein placement.
# --------------------------------------------------------------------------------------
def _longest_dna_chain(chains: list[gemmi.Chain]) -> gemmi.Chain:
    """The DNA chain with the most nucleotides (the primary strand to register on)."""
    return max(chains, key=lambda c: sum(1 for r in c if _is_dna_residue(r)))


def _register_windows(n_template: int, n_operator: int) -> tuple[slice, slice]:
    """Center-aligned matched index windows over template and operator register atoms.

    B-form backbone geometry is sequence-independent, so the register that matters is the
    *index* alignment: center the (shorter) template strand on the operator strand so the
    template's bound site lands on the operator dyad. Returns equal-length slices.
    """
    length = min(n_template, n_operator)
    t_start = (n_template - length) // 2
    o_start = (n_operator - length) // 2
    return slice(t_start, t_start + length), slice(o_start, o_start + length)


def _superpose_operator_onto_template(
    operator_dna: gemmi.Structure, template_structure: gemmi.Structure, half_site_center: int
) -> float:
    """Move ``operator_dna`` into the template's native frame so its half-site overlays the template DNA.

    Keeps the template's clash-free crystal protein-DNA register intact: rather than moving the
    template, the operator is brought onto the template DNA. Only a half-site-length window of the
    operator (centered on ``half_site_center``, a sense-strand residue index) is fit -- the template
    typically footprints one half-site, and a monomeric start model binds one half-site. Both strands
    are matched, trying strand pairings/reversals so the helical phase (which groove face) is correct;
    the lowest-RMSD rigid fit wins. Mutates ``operator_dna``; returns the fit RMSD.
    """
    _, op_strands = _classify_chains(operator_dna[0])
    _, tmpl_strands = _classify_chains(template_structure[0])
    op_strands = [c for c in op_strands if sum(1 for r in c if _is_dna_residue(r)) >= 4]
    tmpl_strands = [c for c in tmpl_strands if sum(1 for r in c if _is_dna_residue(r)) >= 4]
    if not op_strands or not tmpl_strands:
        raise ValueError("_superpose_operator_onto_template: operator/template DNA strands missing.")

    op_coords = [_register_atom_coords(c)[0] for c in op_strands[:2]]
    tmpl_coords = [_register_atom_coords(c)[0] for c in tmpl_strands[:2]]
    # Window length = template footprint (its shortest strand), capped by the operator length.
    window = min(min(len(t) for t in tmpl_coords), min(len(o) for o in op_coords))
    op_len = len(op_coords[0])

    def _window(strand: np.ndarray, center: int) -> np.ndarray:
        start = max(0, min(center - window // 2, len(strand) - window))
        return strand[start : start + window]

    # Sense strand window at the half-site; antisense mirrors it (antiparallel pairing).
    op_windows = [_window(op_coords[0], half_site_center)]
    if len(op_coords) > 1:
        op_windows.append(_window(op_coords[1], op_len - 1 - half_site_center))
    tmpl_windows = [_window(t, len(t) // 2) for t in tmpl_coords]

    n_pair = min(len(op_windows), len(tmpl_windows))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for op_sel in itertools.permutations(range(len(op_windows)), n_pair):
        for tm_sel in itertools.permutations(range(len(tmpl_windows)), n_pair):
            for reversals in itertools.product([False, True], repeat=n_pair):
                mobile_parts, target_parts = [], []
                for oi, ti, rev in zip(op_sel, tm_sel, reversals, strict=True):
                    o = op_windows[oi][::-1] if rev else op_windows[oi]
                    mobile_parts.append(o)
                    target_parts.append(tmpl_windows[ti])
                mobile = np.concatenate(mobile_parts)
                target = np.concatenate(target_parts)
                if len(mobile) < 3:
                    continue
                rotation, translation = kabsch(mobile, target)
                rmsd = rmsd_after(mobile, target, rotation, translation)
                if best is None or rmsd < best[0]:
                    best = (rmsd, rotation, translation)
    if best is None:
        raise ValueError("_superpose_operator_onto_template: could not fit operator to template DNA.")
    rmsd, rotation, translation = best
    apply_transform(operator_dna, rotation, translation)
    return rmsd


_PROTEIN_BACKBONE = frozenset({"N", "CA", "C", "O"})


def _protein_dna_clash_count(pdb_string: str, cutoff: float = 2.5) -> int:
    """Count protein-BACKBONE / DNA heavy-atom clashes (< ``cutoff`` A) in an assembled complex.

    Only backbone atoms count: LigandMPNN redesigns side chains, so a side-chain overlap is
    resolvable, but a backbone atom buried in the DNA cannot be -- it forces glycine. So this is
    the clash that actually drives start-model quality.
    """
    structure = gemmi.read_pdb_string(pdb_string)
    protein, dna = [], []
    for chain in structure[0]:
        for residue in chain:
            if _is_dna_residue(residue):
                dna.extend([a.pos.x, a.pos.y, a.pos.z] for a in residue if not a.is_hydrogen())
            elif _is_amino_acid(residue):
                protein.extend([a.pos.x, a.pos.y, a.pos.z] for a in residue if a.name in _PROTEIN_BACKBONE)
    if not protein or not dna:
        return 0
    p = np.asarray(protein)
    d = np.asarray(dna)
    return int((np.linalg.norm(p[:, None, :] - d[None, :, :], axis=2) < cutoff).sum())


# --------------------------------------------------------------------------------------
# Public entry point: build a start model.
# --------------------------------------------------------------------------------------
def _chain_heavy_coords(chain: gemmi.Chain, residue_filter) -> np.ndarray:  # noqa: ANN001
    """All heavy-atom coordinates ``(N, 3)`` of residues passing ``residue_filter`` in a chain."""
    coords = [
        [a.pos.x, a.pos.y, a.pos.z]
        for residue in chain
        if residue_filter(residue)
        for a in residue
        if not a.is_hydrogen()
    ]
    return np.asarray(coords, dtype=float)


def extract_bound_complex(structure: gemmi.Structure) -> gemmi.Structure:
    """Reduce a (possibly multi-copy) crystal to one operator-bound complex.

    Multi-copy crystals contain several protein-DNA complexes plus waters/ligands. For a
    docking template only one bound unit is needed. This keeps the most protein-contacted
    DNA duplex and the (<=2) protein protomers bound to it, dropping everything else, so the
    shipped template is a compact single complex. Returns a new ``gemmi.Structure``.
    """
    model = structure[0]
    protein_chains, dna_chains = _classify_chains(model)
    if not protein_chains or not dna_chains:
        raise ValueError("extract_bound_complex: need both protein and DNA chains.")

    # Heavy-atom coordinates per chain.
    prot_coords = {c.name: _chain_heavy_coords(c, _is_amino_acid) for c in protein_chains}
    dna_coords = {c.name: _chain_heavy_coords(c, _is_dna_residue) for c in dna_chains}

    def min_dist(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return float("inf")
        return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())

    # Seed on the DNA chain with the most protein heavy-atom contacts.
    def protein_contacts(dna_name: str) -> int:
        return sum(int(min_dist(dna_coords[dna_name], prot_coords[p.name]) <= _CONTACT_CUTOFF) for p in protein_chains)

    seed_dna = max(dna_chains, key=lambda c: protein_contacts(c.name))
    # Keep DNA strands close to the seed (its complementary strand pairs within a few A of C1').
    kept_dna = [
        c
        for c in dna_chains
        if c.name == seed_dna.name or min_dist(dna_coords[seed_dna.name], dna_coords[c.name]) <= 12.0
    ]
    kept_dna_coords = np.concatenate([dna_coords[c.name] for c in kept_dna], axis=0)
    # Keep up to 2 protein protomers nearest the kept DNA (the bound homodimer).
    prot_by_distance = sorted(protein_chains, key=lambda c: min_dist(prot_coords[c.name], kept_dna_coords))
    kept_prot = [c for c in prot_by_distance if min_dist(prot_coords[c.name], kept_dna_coords) <= 8.0][
        :_MAX_PROTOMERS
    ] or prot_by_distance[:1]

    out = gemmi.Structure()
    out_model = gemmi.Model("1")
    for chain in kept_prot + kept_dna:
        new_chain = gemmi.Chain(chain.name)
        for residue in chain:
            if _is_amino_acid(residue) or _is_dna_residue(residue):
                new_chain.add_residue(residue)
        out_model.add_chain(new_chain)
    out.add_model(out_model)
    out.setup_entities()
    return out


def _extract_protomer(structure: gemmi.Structure, chain_name: str) -> gemmi.Structure:
    """Copy one protein chain into its own single-chain Structure (for independent transform)."""
    source: gemmi.Chain | None = next((c for c in _unique_chains(structure[0]) if c.name == chain_name), None)
    if source is None:
        raise ValueError(f"_extract_protomer: protein chain {chain_name!r} not present.")
    out = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_name)
    for residue in source:
        if _is_amino_acid(residue):
            chain.add_residue(residue)
    model.add_chain(chain)
    out.add_model(model)
    return out


def _assemble_pdb(protomers: list[gemmi.Chain], operator_dna: gemmi.Structure) -> str:
    """Assemble placed candidate protomer chain(s) + operator DNA duplex into one PDB string.

    Protein protomers are emitted first as chains ``A``, ``B``, ... and the operator DNA
    strands follow, so LigandMPNN sees the protein chain(s) holding the DNA as fixed context.
    """
    out = gemmi.Structure()
    model = gemmi.Model("1")
    names = iter("ABCDEFGHIJKL")
    for protomer in protomers:
        protein_out = gemmi.Chain(next(names))
        for residue in protomer:
            if _is_amino_acid(residue):
                protein_out.add_residue(residue)
        model.add_chain(protein_out)

    _, dna_chains = _classify_chains(operator_dna[0])
    for chain in dna_chains:
        dna_out = gemmi.Chain(next(names))
        for residue in chain:
            if _is_dna_residue(residue):
                dna_out.add_residue(residue)
        model.add_chain(dna_out)
    out.add_model(model)
    out.setup_entities()
    return out.make_pdb_string()


def build_start_model(
    candidate_structure: gemmi.Structure,
    scaffold: str,
    template_structure: gemmi.Structure,
    x3dna_root: str = DEFAULT_X3DNA_ROOT,
    operator_dna: gemmi.Structure | None = None,
    half_site_center: int | None = None,
) -> tuple[str, dict[str, float]]:
    """Build a template-guided Stage-2 start model (PDB string) for one candidate.

    The template's crystal protein-DNA complex is the clash-free reference frame and is NOT moved.
    The idealized operator DNA is brought onto the template DNA (its half-site overlaying the
    template's footprint), and the candidate protomer is placed onto the template protein by
    recognition-helix superposition -- so the candidate's HTH contacts the operator in the
    template's native binding geometry instead of being jammed into the duplex.

    Args:
        candidate_structure (gemmi.Structure): Stage-1 cofold of the candidate repressor
            with its operator (supplies the candidate's recognition-helix pose + sequence).
        scaffold (str): Operator scaffold sequence used to build idealized B-form DNA.
        template_structure (gemmi.Structure): A natural HTH-DNA crystal template (the pose donor);
            read-only -- it stays in its native frame.
        x3dna_root (str): X3DNA v2.4 install root for ``fiber``.
        operator_dna (gemmi.Structure | None): Prebuilt idealized B-form operator DNA to reuse
            across calls; cloned internally before being moved, so the shared copy is untouched.
        half_site_center (int | None): Operator sense-strand residue index of the half-site center
            to overlay on the template DNA; defaults to the scaffold midpoint.

    Returns:
        tuple[str, dict[str, float]]: The assembled monomeric complex as a PDB string, and a
            metrics dict with ``dna_fit_rmsd`` (operator->template half-site fit RMSD),
            ``helix_fit_rmsd`` (candidate->template recognition-helix Calpha RMSD),
            ``num_protomers`` (always 1.0 -- monomeric for LigandMPNN), and
            ``protein_dna_clashes`` (heavy-atom protein/DNA overlaps < 2.5 A in the start model).

    Raises:
        ValueError: If either structure lacks the protein/DNA chains or recognition helix
            needed for superposition (helices too short to fit, etc.).
    """
    if operator_dna is None:
        operator_dna = build_bform_operator_dna(scaffold, x3dna_root=x3dna_root)
    # Clone (operator is moved into the template frame; never mutate the shared prebuilt copy).
    operator_dna = gemmi.read_pdb_string(operator_dna.make_pdb_string())
    if half_site_center is None:
        half_site_center = len(scaffold) // 2

    # 1) Bring the operator's half-site onto the template DNA (template stays in its native frame).
    dna_fit_rmsd = _superpose_operator_onto_template(operator_dna, template_structure, half_site_center)

    # 2) Recognition helices. Pick the best (most DNA-base contacts) on each side. It is the
    #    CANDIDATE's own recognition helix (and protein) that gets placed; the template (in its
    #    native frame) only donates the binding pose. The start model is MONOMERIC -- the pipeline's
    #    protein is a single Segment and the homodimer is expressed downstream by the cofold
    #    constraints taking it twice (a dimer would make LigandMPNN emit a "chainA/chainB" sequence).
    template_helix = max(
        _select_operator_protomers(detect_recognition_helices(template_structure), template_structure),
        key=lambda h: h.base_contacts,
    )
    candidate_helix = max(detect_recognition_helices(candidate_structure), key=lambda h: h.base_contacts)

    # 3) Superpose the candidate protomer's HTH MOTIF (positioning + recognition helix) onto the
    #    template protomer's, so the candidate's HTH contacts the operator where the template protein
    #    contacted its DNA. Using both helices (vs one) pins the orientation and keeps the body out
    #    of the duplex; the HTH fold is conserved, so this aligns well across different proteins.
    n = min(len(candidate_helix.motif_ca_coords), len(template_helix.motif_ca_coords))
    if n < 3:
        raise ValueError(
            f"build_start_model: HTH motifs too short to superpose "
            f"(candidate {len(candidate_helix.motif_ca_coords)}, template {len(template_helix.motif_ca_coords)})."
        )
    cand_window = _center_window(candidate_helix.motif_ca_coords, n)
    tmpl_window = _center_window(template_helix.motif_ca_coords, n)
    rotation, translation = kabsch(cand_window, tmpl_window)
    helix_fit_rmsd = rmsd_after(cand_window, tmpl_window, rotation, translation)
    protomer = _extract_protomer(candidate_structure, candidate_helix.chain_name)
    apply_transform(protomer, rotation, translation)

    # 4) Assemble the single placed candidate protomer + operator DNA (both in the template frame).
    pdb = _assemble_pdb([protomer[0][0]], operator_dna)
    metrics = {
        "dna_fit_rmsd": dna_fit_rmsd,
        "helix_fit_rmsd": helix_fit_rmsd,
        "num_protomers": 1.0,
        "protein_dna_clashes": float(_protein_dna_clash_count(pdb)),
    }
    return pdb, metrics


def _center_window(coords: np.ndarray, n: int) -> np.ndarray:
    """Center-aligned length-``n`` window of an ordered coordinate array."""
    start = (len(coords) - n) // 2
    return coords[start : start + n]


def _select_operator_protomers(
    helices: list[RecognitionHelix], operator_dna: gemmi.Structure
) -> list[RecognitionHelix]:
    """Reduce a multi-copy template to the one operator-bound unit; keep mono/dimers as-is.

    A template that is already a bound monomer or dimer (``<= _MAX_PROTOMERS`` protomers, e.g.
    the shipped ``extract_bound_complex`` templates) is returned unchanged -- trimming is only
    needed when a crystal carries extra asymmetric-unit copies. In that case keep the
    ``_MAX_PROTOMERS`` protomers whose recognition helix sits closest to the posed operator DNA
    (one bound dimer); the rest belong to other complexes far from the operator frame.
    """
    if len(helices) <= _MAX_PROTOMERS:
        return helices
    _, dna_chains = _classify_chains(operator_dna[0])
    base_coords = _dna_base_atom_coords(dna_chains)
    scored: list[tuple[float, RecognitionHelix]] = []
    for helix in helices:
        dists = np.linalg.norm(helix.ca_coords[:, None, :] - base_coords[None, :, :], axis=2)
        scored.append((float(dists.min()), helix))
    by_distance = sorted(scored, key=lambda pair: pair[0])
    on_operator = [helix for distance, helix in by_distance if distance <= _PROTOMER_ON_OPERATOR_CUTOFF]
    chosen = (on_operator or [by_distance[0][1]])[:_MAX_PROTOMERS]
    # Return in original chain order so protomer pairing with the candidate stays consistent.
    chosen_set = {id(helix) for helix in chosen}
    return [helix for helix in helices if id(helix) in chosen_set]


# --------------------------------------------------------------------------------------
# Self-test (no GPU / heavy tools): exercises the geometry on 2OR1.
# --------------------------------------------------------------------------------------
def _selftest() -> None:
    """Validate the geometry primitives on the 2OR1 homodimer (lambda 434 repressor-operator)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Kabsch round-trip: recover a known rotation+translation exactly.
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(12, 3))
    theta = 0.7
    rot_true = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    trans_true = np.array([3.0, -1.0, 2.0])
    moved = pts @ rot_true.T + trans_true
    rot, trans = kabsch(pts, moved)
    assert rmsd_after(pts, moved, rot, trans) < 1e-9, "Kabsch failed to recover known transform"
    logger.info("Kabsch round-trip RMSD ~ 0: OK")

    template_path = Path(__file__).resolve().parent.parent / "data" / "hth_dna_templates" / "2OR1.pdb"
    if not template_path.exists():
        logger.warning("2OR1 template not found at %s; skipping structural self-test.", template_path)
        return
    x3dna_root = os.environ.get("X3DNA")
    if not x3dna_root or not (Path(x3dna_root) / "bin" / "fiber").exists():
        logger.warning("X3DNA not found (set $X3DNA to a v2.4 install); skipping structural self-test.")
        return

    template = gemmi.read_structure(str(template_path))
    helix = detect_recognition_helix(template)
    logger.info(
        "2OR1 recognition helix: chain %s residues %d-%d (%d base contacts)",
        helix.chain_name,
        helix.start_seqid,
        helix.end_seqid,
        helix.base_contacts,
    )

    # Round-trip: use the template's own protein as a stand-in "candidate" -> helix RMSD ~ 0.
    scaffold = "AAAATGCACTGCACTTT"  # synpromoter77 operator motif (17 bp)
    candidate = gemmi.read_structure(str(template_path))
    pdb, metrics = build_start_model(candidate, scaffold, template)
    logger.info("Start-model metrics (template-as-candidate): %s", metrics)
    n_atoms = pdb.count("\nATOM")
    logger.info("Assembled start model: %d ATOM lines", n_atoms)
    assert metrics["helix_fit_rmsd"] < 1e-6, "self-superposition should be ~0 RMSD"
    assert n_atoms > 0, "assembled start model has no atoms"
    logger.info("Self-test passed.")


if __name__ == "__main__":
    _selftest()
