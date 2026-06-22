"""ProtoRepressor: a three-stage sequence-specific DNA-binding repressor design program.

Designs sequence-specific helix-turn-helix (HTH) repressors against the inverted-repeat
operator sites of the ProtoPromoters (designed by ``promoter_repressor.py``). A repressor
is designed against ONE target operator at a time (with a fixed off-target panel) and the
program runs as three optimizer stages that share the protein/DNA construct by identity:

    Stage 1 - source + reject
        Candidates come from two pools, selected by ``--source``. Pool A (``evo2``/``both``)
        is generated in-pipeline by Evo 2: prompted with the ProtoPromoter scaffold followed
        by the upstream context of an E. coli HTH gene CDS (shipped lacI context,
        ``examples/data/ecoli_hth_upstream.fasta``), sampling continues into candidate coding
        regions (temperature 1.0, top-k 4, 2500 nt per prompt); ORFs are called with Prodigal
        ``-meta`` (bacterial table), translated, and filtered to 80-200 aa. Pool B (``natural``,
        the default) is natural HTH proteins from the seed pool (``--seed-pool``; falls back to
        poly-A seeds when absent). ``both`` combines the pools. The chosen candidates seed the
        protein Segment, which is explored with point mutations. Candidates are scored by an
        HTH-domain term (PyHMMER/Pfam, weight 2.5) and a protein-quality term (length 80-200,
        low-complexity <=0.3, repetitiveness <=0.3, diversity >=0.3, weight 1.5), then co-folded
        with the operator under the structure predictor(s) in ``--stage1-structure-tools``
        (default AF3 + Boltz-2) and kept on overall + protein-DNA + protein-protein ipTM >= 0.5
        and pLDDT >= 70 hard gates per model.

    Stage 2 - template-guided start models + LigandMPNN design + Rosetta cascade
        Each Stage-1 survivor is positioned on its operator by *template-guided superposition*:
        idealized B-form operator DNA (operator + 10 bp flanks, X3DNA ``fiber``) is built, each
        of the natural HTH-DNA crystal templates (``--templates-dir``; up to
        ``--stage2-max-templates``) is registered onto it by DNA backbone superposition, and the
        candidate's recognition helix is superposed onto the template's to drop it into the major
        groove (see ``protorepressor_templating.build_start_model``). This is repeated over all
        passing survivor-template combinations, so no per-operator start models are shipped -- only
        the 20 generic crystals. The resulting start models are LigandMPNN's structure inputs (one
        design per start model), holding the operator DNA as fixed context, over three rounds at
        temperatures 0.1/0.2/0.1. Designs must satisfy a motif-contact gate (>= 2 protein-DNA base
        contacts at 4.0 A), the Rosetta DNA-binding metrics (dbp-design-metrics: interface ddG,
        CMS, shape complementarity, buried-unsats, base/bidentate H-bonds, with the ddG/CMS ML
        prefilter), and AF3/Boltz cofold gates (overall + protein-DNA ipTM >= 0.6, protein-protein
        ipTM >= 0.7, pLDDT >= 70).

    Stage 3 - interface-focused MCMC on operator specificity
        LigandMPNN is the mutation-proposal generator (interface residues within 8 A of the DNA
        variable; residues beyond are held fixed). Each proposal is scored by the composite cost
        S that the framework minimizes (every term scaled to [0, 1], 0 = best):

            S = 4*ipTM + 3*ipSAE + 1*pLDDT + 2*(NA-MPNN margin) + 2*(DeepPBS margin)
                + 2*(consensus z-score) + 2*(AF3 off-target margin) + 2*(motif-contact)
                + 1.5*(base HB) + 0.5*(phosphate HB) + 1*(Ca RMSD to the start model)

        against a fixed off-target panel (up to three same-length palindromic operators from the
        operator set plus three Hamming-distance-3 scrambles of the cognate operator). A stricter
        base-contact hard gate (>= 2 unique contacting residues and >= 2 unique DNA base
        positions at 4.0 A) is also applied.

Operator source: ``--operators-csv`` (ID, palindrome_sequence, scaffold_sequence) or
``--promoter-designs`` to take operators straight from a ``promoter_repressor.py`` run (its
export dir or a FASTA of designed promoters), where the operator-site detector locates each
dyad plus ``--operator-flank`` bp of context on each side. Other data inputs: ``--seed-pool``
(natural HTH seeds), ``--templates-dir`` (the 20 natural HTH-DNA crystal docking templates for
Stage-2 superposition), ``--pfam-hmm`` (HTH-domain term). The homodimer is expressed by folding
two protein chains; ``af3-chain-pair`` duplicates
the protein internally (single input), while the other structure constraints take the protein
twice (the framework's homo-oligomer pattern; the "duplicate Segment" log is advisory).

Heavy/external tools (Evo 2, Prodigal, PyHMMER, AF3/Boltz-2, LigandMPNN, NA-MPNN, DeepPBS,
PyRosetta) mean this script is illustrative and not executed in CI; use ``--dry-run`` to build
a stage's program (and validate its constraints) without running the heavy GPU/scoring tools.
``--dry-run`` still builds the Stage-2/3 template-guided start models (X3DNA ``fiber`` +
superposition, CPU-only) because the inverse-folding generator requires structure inputs to
validate. Run counts default to small illustrative values; scale them up via the stage knobs below.

Example:
    PYTHONPATH=$PWD/proto-tools:$PWD python examples/scripts/protorepressor.py \
        --operator synpromoter77 --stage all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import gemmi
import protorepressor_templating as templating
from proto_tools.transforms.masking import MaskingStrategy

from proto_language.constraint import (
    OperatorSiteConfig,
    af3_chain_pair_prot_dna_iptm_constraint,
    af3_offtarget_iptm_specificity_constraint,
    consensus_operator_specificity_constraint,
    dbp_design_metrics_constraint,
    deeppbs_motif_specificity_constraint,
    dna_base_contact_quality_constraint,
    dna_motif_contact_count_constraint,
    dna_phosphate_contact_constraint,
    na_mpnn_motif_specificity_constraint,
    operator_site_constraint,
    overall_protein_quality_constraint,
    protein_dna_ipsae_constraint,
    protein_domain_constraint,
    structure_iptm_constraint,
    structure_plddt_constraint,
    structure_rmsd_constraint,
)
from proto_language.core import Constraint, Construct, Optimizer, Program, Segment, Sequence
from proto_language.generator import RandomProteinGenerator, RandomProteinGeneratorConfig
from proto_language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)

if TYPE_CHECKING:
    from proto_language.generator import LigandMPNNGenerator

logger = logging.getLogger(__name__)

# Default data ships in examples/data/ so the example runs standalone from the proto-language
# repo for any shipped operator: four operators, a small pool of natural HTH seeds (obligate-
# homodimer MarR/MerR/HxlR families), and the 20 natural HTH-DNA crystal docking templates.
# Per-operator Stage-2 start models are NOT shipped: they are built in-pipeline by superposing
# each Stage-1 survivor onto a template over idealized B-form operator DNA (see
# protorepressor_templating.build_start_model). Larger operator/seed/template sets are selectable
# via --operators-csv / --seed-pool / --templates-dir.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OPERATORS_CSV = _DATA_DIR / "repressor_operators.csv"
DEFAULT_SEED_POOL = _DATA_DIR / "repressor_seed_pool.csv"
DEFAULT_TEMPLATES_DIR = _DATA_DIR / "hth_dna_templates"
# Shipped Evo 2 Pool-A prompt context (E. coli lacI ~290 nt upstream + CDS start); see examples/data/.
DEFAULT_HTH_PROMPT_FASTA = _DATA_DIR / "ecoli_hth_upstream.fasta"

# ProtoPromoter operators designed against when --operator is "all".
TARGET_PROMOTERS = [
    "synpromoter7",
    "synpromoter44",
    "synpromoter77",
    "synpromoter79",
]

# Stage-1 composite weights (HTH-domain 2.5, protein-quality 1.5).
HTH_DOMAIN_WEIGHT = 2.5
PROTEIN_QUALITY_WEIGHT = 1.5
# Pfam/InterPro HTH-family keywords used by the DNA-binding-domain term.
HTH_KEYWORDS = [
    "HTH",
    "helix-turn-helix",
    "AraC",
    "Cro",
    "LacI",
    "TetR",
    "MarR",
    "LysR",
    "GntR",
    "LuxR",
    "MerR",
    "IclR",
    "AsnC",
    "Crp",
]
# Protein-quality sub-config: length 80-200, low-complexity <=0.3, repetitiveness <=0.3, diversity >=0.3.
PROTEIN_QUALITY_SUBCONFIG = {
    "enable_length": True,
    "length_min_length": 80,
    "length_max_length": 200,
    "enable_complexity": True,
    "complexity_max_low_complexity": 0.3,
    "enable_repetitiveness": True,
    "repetitiveness_max_repetitiveness": 0.3,
    "enable_diversity": True,
    "diversity_min_diversity": 0.3,
}
COFOLD_IPTM_MIN = 0.5
COFOLD_PP_IPTM_MIN = 0.5  # Stage-1 protein-protein (homodimer) ipTM hard gate.
COFOLD_PLDDT_MIN = 70.0

# Base-specific contact requirement (Stage-3 gate; also the Stage-3 motif-contact term):
# >=2 protein-DNA base contacts, >=2 unique contacting residues, >=2 unique DNA base positions, 4.0 A.
BASE_CONTACT_GATE = {
    "min_contacts": 2,
    "min_unique_protein_residues": 2,
    "min_unique_dna_positions": 2,
    "contact_distance_angstrom": 4.0,
    "dna_atom_scope": "base",
}
# Stage-2 motif-contact gate (less strict than Stage 3): >=2 base contacts at 4.0 A.
STAGE2_CONTACT_GATE = {
    "min_contacts": 2,
    "min_unique_protein_residues": 1,
    "min_unique_dna_positions": 1,
    "contact_distance_angstrom": 4.0,
    "dna_atom_scope": "base",
}
# Stage-2 PyRosetta rescore thresholds, pinned into dbp-design-metrics.
DBP_STAGE2_THRESHOLDS = {
    "min_contact_molecular_surface": 225.0,
    "min_shape_complementarity": 0.60,
    "min_packstat": 0.55,
    "min_base_score": 10.0,
    "min_bidentate_score": 1.0,
    "max_buried_unsats": 2.0,
}
# Stage-2 AF3 co-fold complex-confidence gates.
STAGE2_AF3_PROTDNA_IPTM_MIN = 0.6
STAGE2_AF3_OVERALL_IPTM_MIN = 0.6
STAGE2_AF3_PP_IPTM_MIN = 0.7  # Stage-2 protein-protein (homodimer) ipTM hard gate.
STAGE2_AF3_PLDDT_MIN = 70.0

# Stage-3 composite-objective S weights (framework-scaled [0,1], 0 = best).
S_WEIGHTS = {
    "iptm": 4.0,  # protein-DNA ipTM (on-target confidence)
    "ipsae": 3.0,  # protein-DNA ipSAE (protein-dna-ipsae)
    "plddt": 1.0,
    "na_mpnn": 2.0,  # NA-MPNN specificity margin (dyad sliding-logprob scoring)
    "deeppbs": 2.0,  # DeepPBS specificity margin (dyad sliding-logprob scoring)
    "consensus": 2.0,  # cross-predictor consensus discrimination (z_NA-MPNN + z_DeepPBS)
    "af3_offtarget": 2.0,  # AF3 on-minus-off-target protein-DNA ipTM specificity margin
    "motif_contact": 2.0,
    "hbond": 1.5,  # base-contact H-bond term (AF3 base HB, weight 1.5)
    "hbond_phosphate": 0.5,  # AF3 phosphate-contact H-bond term (dna-phosphate-contact)
    "ca_rmsd": 1.0,  # Ca RMSD vs template-guided start model
}

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    """Reverse complement over the DNA alphabet."""
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass(frozen=True)
class Operator:
    """One ProtoPromoter operator: the palindrome, its flanked scaffold, and motif indices."""

    name: str
    motif: str  # the inverted-repeat operator (palindrome)
    scaffold: str  # operator + ProtoPromoter flanking context
    dna_indices: tuple[int, ...]  # 0-based positions of the motif within the scaffold

    @property
    def half_site_len(self) -> int:
        """Half-site length (the operator is split into two halves for dyad scoring)."""
        return len(self.motif) // 2


def load_operators(path: Path) -> dict[str, Operator]:
    """Load operators from the ProtoPromoter CSV (ID, palindrome_sequence, scaffold_sequence)."""
    operators: dict[str, Operator] = {}
    if not path.exists():
        logger.warning("Operators CSV not found at %s; operator data unavailable.", path)
        return operators
    with path.open() as handle:
        for row in csv.DictReader(handle):
            name = row["ID"].strip()
            motif = row["palindrome_sequence"].strip().upper()
            scaffold = row["scaffold_sequence"].strip().upper()
            start = scaffold.find(motif)
            if start < 0:
                logger.warning("Operator %s: motif not found in scaffold; skipping.", name)
                continue
            operators[name] = Operator(
                name=name,
                motif=motif,
                scaffold=scaffold,
                dna_indices=tuple(range(start, start + len(motif))),
            )
    return operators


def _read_fasta(path: Path) -> list[str]:
    """Read all DNA records from a FASTA (or one-sequence-per-line) file, upper-cased."""
    seqs: list[str] = []
    block: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if block:
                seqs.append("".join(block).upper())
                block = []
        elif line.strip():
            block.append(line.strip())
    if block:
        seqs.append("".join(block).upper())
    return seqs


def _read_promoter_designs(path: Path) -> list[str]:
    """Read designed promoter DNA sequences produced by ``promoter_repressor.py``.

    Accepts either the ProtoPromoter program's export directory (``program.export(...)``,
    from which ``sequences.fasta`` is read) or a FASTA / one-sequence-per-line file of
    promoter sequences directly. These are the upstream input to operator extraction.
    """
    if path.is_dir():
        fasta = next((c for c in (path / "sequences.fasta", *sorted(path.glob("*.fasta"))) if c.exists()), None)
        if fasta is None:
            raise FileNotFoundError(f"No sequences.fasta (or *.fasta) in ProtoPromoter export dir {path}.")
        path = fasta
    elif not path.exists():
        raise FileNotFoundError(f"Promoter-designs input not found at {path}.")
    return _read_fasta(path)


def operators_from_promoters(
    sequences: list[str], flank: int = 10, config: OperatorSiteConfig | None = None
) -> dict[str, Operator]:
    """Derive repressor target operators from ProtoPromoter designs.

    Runs the same ``operator-site`` detector that ``promoter_repressor.py`` selects on
    (inverted-repeat dyad occluding the -35/-10/TSS), and for each promoter that carries an
    operator builds an :class:`Operator` whose ``motif`` is the located dyad and whose
    ``scaffold`` is that dyad plus ``flank`` bp of promoter context on each side. Promoters
    with no detected operator are skipped with a warning.

    Args:
        sequences (list[str]): Designed promoter DNA sequences (from ``_read_promoter_designs``).
        flank (int): bp of promoter context retained on each side of the operator. Default 10.
        config (OperatorSiteConfig | None): Operator-detector config; defaults to the
            ProtoPromoter defaults (>= 7 bp half-sites, gap <= 1, <= 1 mismatch).

    Returns:
        dict[str, Operator]: Operators keyed ``promoterdesign{i}`` (index into ``sequences``).
    """
    config = config or OperatorSiteConfig()
    outs = operator_site_constraint([(Sequence(sequence=s, sequence_type="dna"),) for s in sequences], config)
    operators: dict[str, Operator] = {}
    for i, (seq, out) in enumerate(zip(sequences, outs, strict=True)):
        best = (out.metadata.get("operator") or {}).get("best_operator")
        if not best:
            logger.warning("Promoter design %d: no inverted-repeat operator detected; skipping.", i)
            continue
        start, end = int(best["start"]), int(best["end"])
        scaffold = seq[max(0, start - flank) : min(len(seq), end + flank)]
        motif = seq[start:end]
        idx0 = scaffold.find(motif)
        name = f"promoterdesign{i}"
        operators[name] = Operator(
            name=name,
            motif=motif,
            scaffold=scaffold,
            dna_indices=tuple(range(idx0, idx0 + len(motif))),
        )
    return operators


def build_off_targets(
    target: Operator, panel: dict[str, Operator], num_panel: int = 3, num_scramble: int = 3
) -> list[str]:
    """Fixed off-target panel: same-length panel operators + Hamming-3 scrambles of the cognate.

    Picks up to ``num_panel`` same-length palindromic operators from the operator
    panel, then adds ``num_scramble`` variants that substitute the cognate operator at
    three positions (Hamming distance 3). All off-targets match the target motif length
    so the specificity constraints can index them consistently.
    """
    motif_len = len(target.motif)
    off_targets: list[str] = [
        op.motif for name, op in sorted(panel.items()) if name != target.name and len(op.motif) == motif_len
    ][:num_panel]

    # Deterministic Hamming-3 scrambles: substitute three evenly-spaced positions with the
    # next base in ACGT order (non-palindrome-preserving).
    bases = "ACGT"
    positions = [motif_len // 4, motif_len // 2, (3 * motif_len) // 4]
    for shift in range(1, num_scramble + 1):
        variant = list(target.motif)
        for raw_pos in positions:
            idx = min(raw_pos, motif_len - 1)
            variant[idx] = bases[(bases.index(variant[idx]) + shift) % 4]
        off_targets.append("".join(variant))
    return off_targets


def load_seed_pool(path: Path, promoter: str, limit: int) -> list[tuple[str, str | None]]:
    """Load natural HTH seed (sequence, structure_path) pairs, preferring the promoter's seeds.

    Returns up to ``limit`` (sequence, structure_path) pairs, prioritizing rows whose
    ``promoter_id`` matches ``promoter`` (template-matched seeds) and falling back to the
    rest of the pool. ``structure_path`` is ``None`` when the row has no cofolded PDB.
    """
    if not path.exists():
        logger.warning("Seed pool not found at %s; falling back to poly-A protein seeds.", path)
        return []
    matched: list[tuple[str, str | None]] = []
    others: list[tuple[str, str | None]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            seq = (row.get("sequence") or "").strip().upper()
            if not seq or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
                continue
            struct = (row.get("structure_path") or "").strip() or None
            pair = (seq, struct)
            (matched if row.get("promoter_id", "").strip() == promoter else others).append(pair)
    pool = matched + others
    return pool[:limit]


def _seed_segment(
    segment: Segment,
    seqs: list[str],
    sequence_type: Literal["dna", "rna", "protein", "ligand"],
    metadata: dict[str, str] | None = None,
) -> None:
    """Seed a segment's result/proposal pools from explicit sequences (aligned across segments)."""
    seeded = [Sequence(sequence=s, sequence_type=sequence_type, metadata=dict(metadata or {})) for s in seqs]
    segment.result_sequences = seeded
    segment.proposal_sequences = [
        Sequence.from_dict(s.to_dict(include_logits=True, include_structure=True)) for s in seeded
    ]


def _setup_segments(
    args: argparse.Namespace, operator: Operator, seeds: list[str]
) -> tuple[Segment, Segment, Segment, list[Construct]]:
    """Build the variable repressor + a DOUBLE-STRANDED operator (fwd + revcomp) as constructs.

    A Construct must be single-type, so protein and each DNA strand live in separate
    constructs. The operator is supplied as both strands (forward scaffold + reverse
    complement) so cofolding yields a real B-form dsDNA duplex — required for DNA-helix
    detection (DeepPBS/DSSR) and base-pair contact geometry. All pools are seeded to the
    same size (``num_results``) so constraint input tuples stay aligned; only the protein varies.

    Each natural HTH is seeded at its OWN length — the pool may hold mixed-length scaffolds. A
    ``Segment``'s length is only nominal (for from-scratch design); the mutation generator
    preserves per-sequence length and the cofold/LigandMPNN steps run per item, so no padding or
    truncation to a fixed length is needed (padding would graft meaningless poly-A tails onto
    short naturals). ``--protein-length`` is only the from-scratch fallback when no seeds exist.
    """
    pool = seeds or ["A" * args.protein_length]
    fitted = [pool[i % len(pool)] for i in range(args.num_results)]
    rc = _reverse_complement(operator.scaffold)

    # Nominal length only (from-scratch fallback); seeded sequences keep their native lengths.
    nominal_length = len(fitted[0]) if fitted else args.protein_length
    protein = Segment(length=nominal_length, sequence_type="protein", label=f"{operator.name} repressor")
    dna = Segment(sequence=operator.scaffold, sequence_type="dna", label=f"{operator.name} operator (+)")
    dna_rc = Segment(sequence=rc, sequence_type="dna", label=f"{operator.name} operator (-)")
    _seed_segment(protein, fitted, "protein", {"seed_source": "natural_hth"})
    _seed_segment(dna, [operator.scaffold] * args.num_results, "dna")
    _seed_segment(dna_rc, [rc] * args.num_results, "dna")
    constructs = [
        Construct([protein], label="repressor"),
        Construct([dna], label="operator_dna_fwd"),
        Construct([dna_rc], label="operator_dna_rev"),
    ]
    return protein, dna, dna_rc, constructs


def resolve_operators(args: argparse.Namespace, operators: dict[str, Operator]) -> list[str]:
    """Which ProtoPromoter operators to design against (one, or all shipped)."""
    if args.operator == "all":
        return [p for p in TARGET_PROMOTERS if p in operators] or sorted(operators)
    if args.operator not in operators:
        raise ValueError(f"--operator {args.operator!r} not in operators CSV; choose from {sorted(operators)}.")
    return [args.operator]


def _tool_cfg(tool: str, args: argparse.Namespace) -> dict[str, object]:
    """Structure-tool config block for ``tool`` (e.g. ``alphafold3``/``boltz2``)."""
    return {"structure_tool": tool, f"{tool}_config": {"device": args.device}}


def _af3_cfg(args: argparse.Namespace) -> dict[str, object]:
    """Structure-tool config block for Stage-2/3 DNA-binding constraints (single tool).

    Defaults to AlphaFold3 (which surfaces the per-chain-pair ipTM matrix the chain-pair
    constraints read); ``--structure-tool boltz2`` folds with Boltz-2 instead (e.g. where AF3
    weights are gated), in which case the chain-pair ipTM term falls back to overall ipTM.
    (Stage 1 co-folds with the tools in ``--stage1-structure-tools``; see ``_stage1_cofold_gates``.)
    """
    return _tool_cfg(args.structure_tool, args)


def _stage1_cofold_gates(
    tool: str, args: argparse.Namespace, protein: Segment, dna: Segment, dna_rc: Segment
) -> list[Constraint]:
    """Stage-1 homodimer + dsDNA cofold HARD gates for ONE structure tool.

    Builds the overall, protein-DNA, and protein-protein ipTM (each >= 0.5) and protein pLDDT
    (>= 70) gates for one model. ``build_stage1`` calls this once per tool in
    ``--stage1-structure-tools`` so a candidate must clear the gates under every model.
    Labels are suffixed with the tool so the per-model gates stay distinct in the export.
    """
    cfg = _tool_cfg(tool, args)
    suffix = f"_{tool}"
    return [
        # Overall ipTM >= 0.5.
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=structure_iptm_constraint,
            function_config=cfg,
            threshold=1.0 - COFOLD_IPTM_MIN,
            label=f"cofold_overall_iptm_gate{suffix}",
        ),
        # Protein-DNA chain-pair ipTM >= 0.5 (score is 0 once prot-DNA ipTM >= desired_iptm).
        Constraint(
            inputs=[protein, dna, dna_rc],
            function=af3_chain_pair_prot_dna_iptm_constraint,
            function_config={**cfg, "num_protein_copies": 2, "desired_iptm": COFOLD_IPTM_MIN},
            threshold=0.01,
            label=f"cofold_protdna_iptm_gate{suffix}",
        ),
        # Protein-protein (homodimer) chain-pair ipTM >= 0.5 (falls back to overall ipTM on Boltz-2).
        Constraint(
            inputs=[protein, dna, dna_rc],
            function=af3_chain_pair_prot_dna_iptm_constraint,
            function_config={
                **cfg,
                "num_protein_copies": 2,
                "pair_type": "protein-protein",
                "desired_iptm": COFOLD_PP_IPTM_MIN,
            },
            threshold=0.01,
            label=f"cofold_protprot_iptm_gate{suffix}",
        ),
        # Protein pLDDT >= 70 (structure-plddt returns 1 - plddt_norm).
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=structure_plddt_constraint,
            function_config=cfg,
            threshold=1.0 - COFOLD_PLDDT_MIN / 100.0,
            label=f"cofold_plddt_gate{suffix}",
        ),
    ]


def _load_hth_upstream(path: Path) -> str:
    """Load the (first) HTH-gene-upstream DNA prompt context from a shipped FASTA.

    The shipped ``examples/data/ecoli_hth_upstream.fasta`` contains the ~290 nt immediately
    upstream of (and including the start of) a real E. coli K-12 HTH repressor CDS (lacI),
    so Evo 2 is primed to extend the ProtoPromoter scaffold straight into a coding region.
    Returns the concatenated nucleotide sequence (upper-case) of the first FASTA record.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"HTH-upstream prompt FASTA not found at {path}; required for --source evo2/both "
            f"(ship examples/data/ecoli_hth_upstream.fasta or pass --hth-prompt-fasta)."
        )
    lines: list[str] = []
    started = False
    for raw in path.read_text().splitlines():
        if raw.startswith(">"):
            if started:  # stop after the first record
                break
            started = True
            continue
        lines.append(raw.strip())
    seq = "".join(lines).upper()
    if not seq:
        raise ValueError(f"HTH-upstream prompt FASTA at {path} has no sequence in its first record.")
    return seq


def _evo2_generate_candidates(args: argparse.Namespace, operator: Operator, n_target: int) -> list[str]:
    """Generate de novo TF protein candidates with Evo 2 (Pool A), in-pipeline.

    Pool A: Evo 2 is prompted with the ProtoPromoter scaffold followed by the upstream context
    of an E. coli HTH gene CDS (the shipped lacI context), then autoregressively samples into
    candidate coding regions (temperature 1.0, top-k=4, 2500 nt per prompt). Generated DNA is
    ORF-called and translated with Prodigal (-meta, bacterial table, min-gene 240 nt), and the
    translated proteins are filtered to 80-200 aa.

    Args:
        args (argparse.Namespace): Parsed CLI options (device, evo2_num_prompts, hth_prompt_fasta).
        operator (Operator): Target operator; its ``scaffold`` is the prefix of every prompt.
        n_target (int): Desired number of protein candidates to return (up to this many).

    Returns:
        list[str]: Up to ``n_target`` translated HTH-candidate protein sequences (80-200 aa).
    """
    from proto_tools import (
        Evo2SampleConfig,
        Evo2SampleInput,
        ProdigalConfig,
        ProdigalInput,
        run_evo2_sample,
        run_prodigal_prediction,
    )

    hth_upstream = _load_hth_upstream(args.hth_prompt_fasta)
    prompt = operator.scaffold + hth_upstream
    prompts = [prompt] * max(1, args.evo2_num_prompts)
    logger.info(
        "Stage-1 Pool A: Evo 2 generating from %d prompt(s) (scaffold[%d nt] + HTH-upstream[%d nt]).",
        len(prompts),
        len(operator.scaffold),
        len(hth_upstream),
    )

    sampled = run_evo2_sample(
        Evo2SampleInput(prompts=prompts),
        Evo2SampleConfig(
            temperature=1.0,
            top_k=4,
            max_new_tokens=2500,
            prepend_prompt=False,
            model_checkpoint="evo2_7b",
            device=args.device,
        ),
    )
    generated_dna: list[str] = list(sampled.sequences)
    logger.info("Evo 2 produced %d generated DNA sequence(s); ORF-calling with Prodigal (-meta).", len(generated_dna))

    predicted = run_prodigal_prediction(
        ProdigalInput(input_sequences=generated_dna),
        ProdigalConfig(meta_mode=True, translation_table="bacterial", min_gene=240),
    )
    candidates: list[str] = []
    for per_sequence in predicted.predicted_orfs:
        for orf in per_sequence:
            protein = orf.amino_acid_sequence.strip().rstrip("*").upper()
            if 80 <= len(protein) <= 200 and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in protein):
                candidates.append(protein)
    logger.info(
        "Prodigal called %d ORF(s); %d translated protein(s) pass the 80-200 aa filter.",
        predicted.num_orfs,
        len(candidates),
    )
    return candidates[:n_target]


def _stage1_seeds(args: argparse.Namespace, operator: Operator, natural_seeds: list[str]) -> list[str]:
    """Assemble Stage-1 protein seeds per ``--source`` (natural Pool B, Evo 2 Pool A, or both).

    ``natural`` (default) keeps the existing natural-HTH seed-pool path (Pool B); ``evo2`` runs
    live Evo 2 + Prodigal generation (Pool A); ``both`` combines Pool A and Pool B. The combined
    or generated proteins seed the protein Segment exactly as the natural path does.
    """
    if args.source == "natural":
        return natural_seeds
    generated = _evo2_generate_candidates(args, operator, args.num_results)
    if args.source == "evo2":
        return generated
    return generated + natural_seeds


def build_stage1(args: argparse.Namespace, operator: Operator, seeds: list[str]) -> tuple[Program, Segment, Segment]:
    """Stage 1: source HTH candidates and reject on HTH-domain + quality + operator cofold ipTM.

    Per ``--source``: ``natural`` seeds from the natural HTH pool (Pool B; ``seeds`` already
    loaded by ``main``); ``evo2`` runs live Evo 2 + Prodigal generation (Pool A); ``both`` combines
    them. Evo 2 generation is skipped under ``--dry-run`` (the program still builds; the natural
    seeds, or the poly-A fallback when the pool is missing, seed the Segment for the dry build).
    """
    if args.source != "natural" and not args.dry_run:
        seeds = _stage1_seeds(args, operator, seeds)
    elif args.source != "natural":
        logger.info(
            "--dry-run with --source %s: skipping live Evo 2 generation; "
            "building Stage 1 on the natural/fallback seeds.",
            args.source,
        )
    protein, dna, dna_rc, constructs = _setup_segments(args, operator, seeds)

    # Mutate around the natural HTH seeds to explore the pool while staying HTH-like.
    generator = RandomProteinGenerator(RandomProteinGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3)))
    generator.assign(protein)

    constraints = [
        Constraint(
            inputs=[protein],
            function=overall_protein_quality_constraint,
            function_config={"protein_quality_config": PROTEIN_QUALITY_SUBCONFIG},
            weight=PROTEIN_QUALITY_WEIGHT,
            label="protein_quality",
        ),
    ]
    # Cofold hard gates on the homodimer + dsDNA operator, run under each structure tool
    # (each ipTM >= 0.5 and pLDDT >= 70 per model).
    logger.info(
        "Stage 1 co-folds with %s (each model gated on ipTM >= %.1f, pLDDT >= %.0f).",
        ", ".join(args.stage1_structure_tools),
        COFOLD_IPTM_MIN,
        COFOLD_PLDDT_MIN,
    )
    for tool in args.stage1_structure_tools:
        constraints.extend(_stage1_cofold_gates(tool, args, protein, dna, dna_rc))
    # DNA-binding-domain term (PyHMMER vs Pfam HTH/AraC/Cro/LacI...). Requires a Pfam HMM file.
    if args.pfam_hmm:
        constraints.insert(
            0,
            Constraint(
                inputs=[protein],
                function=protein_domain_constraint,
                function_config={"hmm_db": args.pfam_hmm, "keywords": HTH_KEYWORDS},
                weight=HTH_DOMAIN_WEIGHT,
                label="hth_domain",
            ),
        )
    else:
        logger.warning("No --pfam-hmm provided; omitting the HTH-domain term (supply --pfam-hmm to enable it).")
    optimizer = RejectionSamplingOptimizer(
        constructs=constructs,
        generators=[generator],
        constraints=constraints,
        config=RejectionSamplingOptimizerConfig(
            num_samples=args.stage1_samples,
            num_results=args.num_results,
            energy_threshold=args.stage1_threshold,
        ),
    )
    program = Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed)
    return program, protein, dna


def _stage2_constraints(
    args: argparse.Namespace, operator: Operator, protein: Segment, dna: Segment, dna_rc: Segment
) -> list[Constraint]:
    """Fresh Stage-2 constraint set (constraints can't be reused across optimizer stages).

    Motif-contact gate + dbp-design-metrics (pinned Rosetta thresholds) + the AF3 cofold gates
    (overall ipTM >= 0.6, protein-DNA ipTM >= 0.6, protein-protein ipTM >= 0.7, pLDDT >= 70).
    """
    return [
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=dna_motif_contact_count_constraint,
            function_config={**_af3_cfg(args), **STAGE2_CONTACT_GATE, "dna_indices": list(operator.dna_indices)},
            threshold=0.5,
            label="motif_contact_gate",
        ),
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=dbp_design_metrics_constraint,
            function_config={**_af3_cfg(args), **DBP_STAGE2_THRESHOLDS},
            weight=1.0,
            label="dbp_design_metrics",
        ),
        # AF3/Boltz cofold complex-confidence gates: overall ipTM >= 0.6, protein-DNA ipTM >= 0.6, pLDDT >= 70.
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=structure_iptm_constraint,
            function_config=_af3_cfg(args),
            threshold=1.0 - STAGE2_AF3_OVERALL_IPTM_MIN,
            label="overall_iptm_gate",
        ),
        Constraint(
            inputs=[protein, dna, dna_rc],
            function=af3_chain_pair_prot_dna_iptm_constraint,
            function_config={**_af3_cfg(args), "num_protein_copies": 2, "desired_iptm": STAGE2_AF3_PROTDNA_IPTM_MIN},
            threshold=0.01,
            label="protdna_iptm_gate",
        ),
        # Protein-protein (homodimer) chain-pair ipTM >= 0.7 (falls back to overall ipTM on Boltz-2).
        Constraint(
            inputs=[protein, dna, dna_rc],
            function=af3_chain_pair_prot_dna_iptm_constraint,
            function_config={
                **_af3_cfg(args),
                "num_protein_copies": 2,
                "pair_type": "protein-protein",
                "desired_iptm": STAGE2_AF3_PP_IPTM_MIN,
            },
            threshold=0.01,
            label="protprot_iptm_gate",
        ),
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=structure_plddt_constraint,
            function_config=_af3_cfg(args),
            threshold=1.0 - STAGE2_AF3_PLDDT_MIN / 100.0,
            label="plddt_gate",
        ),
    ]


def build_stage2(
    args: argparse.Namespace, operator: Operator, champions: list[Sequence] | list[str]
) -> tuple[Program, Segment, Segment]:
    """Stage 2: per-start-model LigandMPNN design over a three-round temperature schedule.

    Stage-1 survivors (``champions``, carrying their cofold structures) are positioned on the
    operator by template-guided superposition against the crystal docking templates -- "all
    passing operator-template combinations" (``build_operator_start_models``). Each start model
    gets its own three-round LigandMPNN design escalating temperature (0.1 / 0.2 / 0.1), so no
    per-operator start models are shipped. Each round uses a SINGLE start model (LigandMPNN
    broadcasts it to ``num_samples`` designs): single-structure mode is required because the
    optimizer handoff sets ``num_proposals`` from the upstream champion count, which a
    multi-structure batch (one design per structure) could not match. All rounds share the
    constructs by identity (champions flow round->round) and get a fresh generator + constraint
    set (neither is reusable across optimizer stages). Start models are built even under
    ``--dry-run`` (the inverse-folding generator requires structure inputs to validate); with no
    Stage-1 survivors the template protein stands in as the candidate.
    """
    protein, dna, dna_rc, constructs = _setup_segments(args, operator, [_seq_str(c) for c in champions])
    start_models = build_operator_start_models(args, operator, _champion_structures(champions))
    rounds = [
        (args.stage2_round1_temp, args.stage2_round1_samples),
        (args.stage2_round2_temp, args.stage2_round2_samples),
        (args.stage2_round3_temp, args.stage2_round3_samples),
    ]
    optimizers: list[Optimizer] = []
    for start_model in start_models:
        for temperature, num_samples in rounds:
            generator = _ligandmpnn_generator(args, protein, temperature=temperature, structures=[start_model])
            optimizers.append(
                RejectionSamplingOptimizer(
                    constructs=constructs,
                    generators=[generator],
                    constraints=_stage2_constraints(args, operator, protein, dna, dna_rc),
                    config=RejectionSamplingOptimizerConfig(
                        num_samples=num_samples, num_results=min(args.num_results, num_samples)
                    ),
                )
            )
    logger.info(
        "Stage 2 for %s: %d start model(s) x 3 rounds = %d optimizer stage(s).",
        operator.name,
        len(start_models),
        len(optimizers),
    )
    program = Program(optimizers=optimizers, num_results=args.num_results, seed=args.seed)
    return program, protein, dna


def build_stage3(
    args: argparse.Namespace, operator: Operator, off_targets: list[str], champions: list[Sequence] | list[str]
) -> tuple[Program, Segment, Segment]:
    """Stage 3: interface MCMC minimizing the composite specificity objective S.

    A fresh MONOMERIC template-guided start model is built for the interface-focused LigandMPNN
    proposals and the Ca-RMSD reference. We rebuild it via ``build_operator_start_models`` (which
    places a single protomer) rather than reusing a Stage-2 champion's attached structure: that
    attached structure is the homodimeric cofold complex (overwritten during Stage-2 scoring), and
    feeding a 2-chain structure to LigandMPNN would yield a "chainA/chainB" sequence the
    single-protein Segment cannot represent. With no prior Stage 2 (or under ``--dry-run``) the
    template protein stands in as the candidate.
    """
    protein, dna, dna_rc, constructs = _setup_segments(args, operator, [_seq_str(c) for c in champions])

    start_models = build_operator_start_models(args, operator, _champion_structures(champions))
    representative: object | None = start_models[0] if start_models else None

    # LigandMPNN is the mutation-proposal generator: interface-focused (residues >8 A of DNA fixed),
    # generating proposals from the single representative start model.
    generator = _ligandmpnn_generator(
        args,
        protein,
        temperature=args.stage3_temperature,
        structures=[representative] if representative is not None else None,
        interface_focus=True,
    )

    spec_cfg = {
        **_af3_cfg(args),
        "target_motif": operator.motif,
        "off_target_motifs": off_targets,
        "dna_indices": list(operator.dna_indices),
    }
    constraints = [
        Constraint(
            inputs=[protein, dna, dna_rc],
            function=af3_chain_pair_prot_dna_iptm_constraint,
            function_config={**_af3_cfg(args), "num_protein_copies": 2, "desired_iptm": 0.7},
            weight=S_WEIGHTS["iptm"],
            label="iptm",
        ),
        Constraint(
            inputs=[protein, dna, dna_rc],
            function=protein_dna_ipsae_constraint,
            function_config={**_af3_cfg(args), "num_protein_copies": 2, "desired_ipsae": 0.5},
            weight=S_WEIGHTS["ipsae"],
            label="ipsae",
        ),
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=structure_plddt_constraint,
            function_config=_af3_cfg(args),
            weight=S_WEIGHTS["plddt"],
            label="plddt",
        ),
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=na_mpnn_motif_specificity_constraint,
            function_config=spec_cfg,
            weight=S_WEIGHTS["na_mpnn"],
            label="na_mpnn_margin",
        ),
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=deeppbs_motif_specificity_constraint,
            function_config=spec_cfg,
            weight=S_WEIGHTS["deeppbs"],
            label="deeppbs_margin",
        ),
        # Cross-predictor consensus discrimination (z_NA-MPNN + z_DeepPBS), combining the two readout models.
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=consensus_operator_specificity_constraint,
            function_config=spec_cfg,
            weight=S_WEIGHTS["consensus"],
            label="consensus_specificity",
        ),
        # AF3 on-minus-off-target protein-DNA ipTM specificity margin.
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=af3_offtarget_iptm_specificity_constraint,
            function_config={**spec_cfg, "target_dna_sequence": operator.scaffold},
            weight=S_WEIGHTS["af3_offtarget"],
            label="af3_offtarget_margin",
        ),
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=dna_motif_contact_count_constraint,
            function_config={**_af3_cfg(args), **BASE_CONTACT_GATE, "dna_indices": list(operator.dna_indices)},
            weight=S_WEIGHTS["motif_contact"],
            label="motif_contact",
        ),
        # Stricter base-specific contact HARD gate (>=2 unique residues + >=2 unique DNA base positions @4.0A).
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=dna_motif_contact_count_constraint,
            function_config={**_af3_cfg(args), **BASE_CONTACT_GATE, "dna_indices": list(operator.dna_indices)},
            threshold=0.5,
            label="base_contact_gate",
        ),
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=dna_base_contact_quality_constraint,
            function_config={**_af3_cfg(args), "desired_bidentate": 2, "desired_base_contacts": 4},
            weight=S_WEIGHTS["hbond"],
            label="hbond",
        ),
        # AF3 phosphate-contact H-bond term (scored separately from the base-contact HB term).
        Constraint(
            inputs=[protein, protein, dna, dna_rc],
            function=dna_phosphate_contact_constraint,
            function_config={**_af3_cfg(args), "desired_phosphate_contacts": 2},
            weight=S_WEIGHTS["hbond_phosphate"],
            label="hbond_phosphate",
        ),
    ]
    if representative is not None:
        constraints.append(
            Constraint(
                inputs=[protein],
                function=structure_rmsd_constraint,
                function_config={**_af3_cfg(args), "target_structure": representative},
                weight=S_WEIGHTS["ca_rmsd"],
                label="ca_rmsd",
            )
        )
    else:
        logger.warning("No start model for %s; omitting the Ca-RMSD term.", operator.name)

    optimizer = MCMCOptimizer(
        constructs=constructs,
        generators=[generator],
        constraints=constraints,
        config=MCMCOptimizerConfig(
            num_results=args.num_results,
            num_steps=args.stage3_steps,
            proposals_per_result=args.stage3_proposals,
            max_temperature=1.0,
            min_temperature=1e-3,
            temperature_schedule="exponential",
        ),
    )
    program = Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed)
    return program, protein, dna


_DNA_ALPHABET = frozenset("ACGTUN")


def _is_dna_chain(sequence: str) -> bool:
    """Heuristic: a chain whose one-letter sequence is entirely DNA/RNA bases is nucleic acid."""
    letters = set(sequence.upper())
    return bool(letters) and letters <= _DNA_ALPHABET


def _interface_fixed_positions(structure_or_path: object, cutoff: float = 8.0) -> dict[str, list[int]]:
    """Protein residues BEYOND ``cutoff`` A of any DNA chain, per chain, in the Structure's numbering.

    Derives fixed positions from a :class:`proto_tools.Structure` (an in-memory object or a path
    it is loaded from) rather than a raw PDB re-parse, so the positions match how
    ``InverseFoldingStructureInput`` / ``ResidueSelection`` validate (which use
    ``Structure.get_chain_positions`` -- the structure's own 1-indexed residue numbering, with
    PDB gaps/HETATM dropped). For each protein chain P and the set of DNA chains, the interface
    residues within ``cutoff`` A of the DNA are found via ``Structure.interface_contact_residues``
    (also in that numbering), and fixed_positions are the structure's positions NOT in the
    interface. This validates against the Structure and applies (instead of degrading to full
    redesign as the raw-PDB renumbering did).
    """
    from proto_tools import Structure

    structure = (
        structure_or_path if isinstance(structure_or_path, Structure) else Structure.from_file(structure_or_path)
    )
    chain_ids = structure.get_chain_ids()
    dna_chains = [c for c in chain_ids if _is_dna_chain(structure.get_chain_sequence(c))]
    protein_chains = [c for c in chain_ids if not _is_dna_chain(structure.get_chain_sequence(c))]
    if not dna_chains or not protein_chains:
        return {}

    fixed: dict[str, list[int]] = {}
    for chain in protein_chains:
        positions = structure.get_chain_positions(chain)  # structure's own 1-indexed numbering
        interface = structure.interface_contact_residues(binder_chain=chain, target_chains=dna_chains, cutoff=cutoff)
        beyond = [pos for pos in positions if pos not in interface]
        if beyond:
            fixed[chain] = beyond
    return fixed


def _ligandmpnn_generator(
    args: argparse.Namespace,
    protein: Segment,
    temperature: float,
    structures: list[object] | None,
    interface_focus: bool = False,
) -> LigandMPNNGenerator:
    """Build a LigandMPNN generator that redesigns the protein onto template-guided start models.

    ``structures`` are in-memory start-model :class:`proto_tools.Structure` objects (built by
    ``build_operator_start_models`` from Stage-1 survivors + crystal templates); LigandMPNN holds
    the operator DNA as fixed ligand context and redesigns the protein. With multiple structures
    LigandMPNN designs one sequence per start model; with one it generates ``num_proposals`` from
    it. When ``interface_focus`` is set (Stage 3), residues beyond 8 A of the DNA are held fixed
    per structure so only the interface is varied. ``structures`` is None under ``--dry-run`` (the
    generator builds for program validation but is never sampled).
    """
    from proto_tools import InverseFoldingStructureInput

    from proto_language.generator import LigandMPNNGenerator, LigandMPNNGeneratorConfig

    structure_inputs: list[InverseFoldingStructureInput] | None = None
    if structures:
        structure_inputs = []
        total_fixed = 0
        for structure in structures:
            fixed: dict[str, list[int]] | None = None
            if interface_focus:
                # Hold residues >8 A from DNA fixed so only the interface is redesigned; derived
                # per start model from its own chain numbering. The try/except is a safety net:
                # if a model's numbering fails to validate, degrade to full redesign with a warning.
                try:
                    fixed = _interface_fixed_positions(structure, cutoff=8.0) or None
                except ValueError as exc:
                    logger.warning(
                        "Interface-focus fixed_positions did not align (%s); full redesign.",
                        str(exc).splitlines()[-1][:120],
                    )
                    fixed = None
            total_fixed += sum(len(v) for v in (fixed or {}).values())
            structure_inputs.append(InverseFoldingStructureInput(structure=structure, fixed_positions=fixed))
        if interface_focus and structure_inputs:
            logger.info(
                "Interface-focused design: %d start model(s) with %d non-interface residue(s) fixed "
                "(>8 A from DNA); interface residues remain variable.",
                len(structure_inputs),
                total_fixed,
            )
    generator = LigandMPNNGenerator(
        LigandMPNNGeneratorConfig(
            structure_inputs=structure_inputs, temperature=temperature, batch_size=1, device=args.device
        )
    )
    generator.assign(protein)
    return generator


def _seq_str(item: Sequence | str) -> str:
    """Sequence string from either a champion ``Sequence`` (prior stage) or a raw seed string."""
    return item.sequence if isinstance(item, Sequence) else item


def _champion_structures(champions: list[Sequence] | list[str]) -> list[object]:
    """Cofold/start-model structures carried by champion ``Sequence`` objects (empty for seeds)."""
    return [c.structure for c in champions if isinstance(c, Sequence) and c.structure is not None]


# A start model is "well-superposed" only if its HTH-motif Calpha RMSD is within this tolerance;
# below it, a low backbone clash means the HTH is genuinely seated in the operator's major groove
# (a low clash with a poor fit just means the protein was placed away from the DNA).
GOOD_MOTIF_FIT_RMSD = 2.0


def _rank_start_model(scored_item: tuple[float, float, object]) -> tuple[bool, float, float]:
    """Sort key for start models: well-fit HTH first, then fewest backbone clashes, then tightest fit."""
    clashes, motif_rmsd, _structure = scored_item
    return (motif_rmsd > GOOD_MOTIF_FIT_RMSD, clashes, motif_rmsd)


def build_operator_start_models(
    args: argparse.Namespace, operator: Operator, champion_structures: list[object]
) -> list[object]:
    """Build Stage-2 start models by template-guided superposition of Stage-1 survivors.

    For each candidate cofold structure (a Stage-1 survivor) and each crystal docking template,
    superpose the candidate's recognition helix onto the template's over idealized B-form operator
    DNA (``protorepressor_templating.build_start_model``) -- "all passing operator-template
    combinations". A candidate with no detectable recognition helix (e.g. a low-quality design
    that did not fold into an HTH) is skipped; if EVERY candidate fails, falls back to
    template-as-candidate (the template's own protein) so the stage still gets a start model. When
    no candidate structures are supplied at all (e.g. running Stage 2 directly without a prior
    Stage 1), template-as-candidate is used directly. Returns in-memory ``Structure`` objects.
    """
    from proto_tools import Structure

    templates_dir = Path(args.templates_dir) if args.templates_dir else DEFAULT_TEMPLATES_DIR
    # Consider ALL crystal templates and select by fit (lowest protein-DNA clash), not by filename:
    # a fold-incompatible template jams the candidate body into the DNA (e.g. a 72-aa HTH on a
    # 195-aa scaffold), which LigandMPNN can only satisfy with glycine. Picking the best-fitting
    # template per candidate keeps the HTH cleanly contacting the operator.
    template_paths = sorted(templates_dir.glob("*.pdb"))
    if not template_paths:
        raise FileNotFoundError(f"No HTH-DNA crystal templates (*.pdb) found in {templates_dir}.")

    operator_dna = templating.build_bform_operator_dna(operator.scaffold)
    half_site_center = operator.dna_indices[len(operator.dna_indices) // 4]
    max_keep = max(1, args.stage2_max_templates)

    def _build_for(candidates: list[object | None]) -> list[tuple[float, float, object]]:
        scored: list[tuple[float, float, object]] = []
        for candidate in candidates:
            candidate_pdb = candidate.structure_pdb if isinstance(candidate, Structure) else None
            per_candidate: list[tuple[float, float, object]] = []
            for template_path in template_paths:
                template = gemmi.read_structure(str(template_path))
                candidate_gemmi = (
                    gemmi.read_structure(str(template_path))
                    if candidate_pdb is None
                    else gemmi.read_pdb_string(candidate_pdb)
                )
                try:
                    pdb, metrics = templating.build_start_model(
                        candidate_gemmi, operator.scaffold, template,
                        operator_dna=operator_dna, half_site_center=half_site_center,
                    )
                except (ValueError, RuntimeError) as exc:
                    logger.warning("Start model failed (template %s, %s): %s", template_path.stem, operator.name, exc)
                    continue
                structure = Structure(structure=pdb, structure_format="pdb")
                for metric_name, metric_value in metrics.items():
                    structure.add_metric(metric_name, metric_value)
                per_candidate.append((metrics["protein_dna_clashes"], metrics["helix_fit_rmsd"], structure))
            # Keep the best templates: a real HTH superposition (motif RMSD within tolerance) FIRST,
            # then fewest backbone clashes. A low clash with a poor motif fit means the HTH isn't
            # actually in the groove, so fit must gate the selection.
            per_candidate.sort(key=_rank_start_model)
            scored.extend(per_candidate[:max_keep])
        return scored

    candidates: list[object | None] = list(champion_structures) or [None]
    scored = _build_for(candidates)
    if not scored and champion_structures:
        # Every candidate-derived attempt failed (e.g. a low-quality design with no detectable
        # recognition helix); fall back to template-as-candidate so the stage still has a start model.
        logger.warning(
            "All candidate-derived start models failed for %s; falling back to template-as-candidate.", operator.name
        )
        scored = _build_for([None])
    if not scored:
        raise RuntimeError(f"No Stage-2 start models could be built for {operator.name}.")
    # Best-fitting, lowest-clash first so Stage 3's representative is the cleanest start model.
    scored.sort(key=_rank_start_model)
    start_models = [structure for _, _, structure in scored]
    logger.info(
        "Stage 2 %s: %d start model(s) from %d candidate(s); best protein-DNA clash=%d.",
        operator.name,
        len(start_models),
        len(candidates),
        int(scored[0][0]),
    )
    return start_models


STAGE_BUILDERS = ("stage1", "stage2", "stage3")


def _run_stage(
    name: str,
    args: argparse.Namespace,
    operator: Operator,
    panel: dict[str, Operator],
    seeds: list[str],
    champions: dict[str, list[Sequence]],
) -> list[Sequence] | None:
    """Build one stage (optionally run it) and return its champion repressor ``Sequence`` objects.

    Champions are returned as full ``Sequence`` objects (not bare strings) so each carries its
    attached structure (Stage-1 cofold, Stage-2 start model) into the next stage's superposition.
    """
    logger.info("=== %s :: %s ===", name, operator.name)
    if name == "stage1":
        program, protein, _ = build_stage1(args, operator, seeds)
    elif name == "stage2":
        program, protein, _ = build_stage2(args, operator, champions.get("stage1", seeds))
    else:
        off_targets = build_off_targets(operator, panel)
        program, protein, _ = build_stage3(args, operator, off_targets, champions.get("stage2", seeds))

    n_constraints = sum(len(opt.constraints) for opt in program.optimizers)
    logger.info("Built %s: %d optimizer stage(s), %d constraint(s).", name, len(program.optimizers), n_constraints)
    if args.dry_run:
        logger.info("--dry-run: skipping execution of %s.", name)
        return None

    program.run()
    out = args.output_dir / operator.name / name
    out.mkdir(parents=True, exist_ok=True)
    program.export(out, format="json")
    designs = list(protein.result_sequences)
    logger.info("Exported %d %s designs -> %s", len(designs), name, out)
    return designs


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the ProtoRepressor example."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--operator",
        default="synpromoter77",
        help="ProtoPromoter operator to target, or 'all' for every shipped operator.",
    )
    p.add_argument("--stage", choices=[*STAGE_BUILDERS, "all"], default="all", help="Which stage to build/run.")
    p.add_argument("--dry-run", action="store_true", help="Build the program(s) without running.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--structure-tool",
        default="alphafold3",
        choices=["alphafold3", "boltz2", "protenix", "chai1"],
        help="Structure predictor for Stage 2/3 cofolding (AF3 default; boltz2 where AF3 weights are gated).",
    )
    p.add_argument(
        "--stage1-structure-tools",
        nargs="+",
        default=["alphafold3", "boltz2"],
        choices=["alphafold3", "boltz2", "protenix", "chai1"],
        help="Structure predictors for the Stage-1 cofold gates; a candidate must clear the "
        "ipTM/pLDDT gates under every listed model.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-results", type=int, default=14, help="Designs retained per stage.")
    p.add_argument("--protein-length", type=int, default=111,
                   help="From-scratch fallback length when no seeds exist; natural seeds keep their own length.")
    p.add_argument("--output-dir", type=Path, default=Path("protorepressor_outputs"))
    # Data inputs.
    p.add_argument("--operators-csv", type=Path, default=None, help="ProtoPromoter operators CSV.")
    p.add_argument(
        "--promoter-designs",
        type=Path,
        default=None,
        help="ProtoPromoter run to take operators from: promoter_repressor.py export dir (or a FASTA of "
        "designed promoters). Operators are extracted by the operator-site detector; overrides --operators-csv.",
    )
    p.add_argument(
        "--operator-flank",
        type=int,
        default=10,
        help="bp of promoter context kept on each side of an extracted operator (default 10 bp each side).",
    )
    p.add_argument("--seed-pool", type=Path, default=None, help="Natural HTH seed pool CSV.")
    p.add_argument(
        "--templates-dir",
        type=Path,
        default=None,
        help="Directory of natural HTH-DNA crystal docking templates (PDB) for Stage-2 superposition.",
    )
    p.add_argument("--pfam-hmm", default=None, help="Pfam-A HMM file (HTH/AraC/Cro/LacI) for the domain term.")
    p.add_argument("--seed-limit", type=int, default=64, help="Natural HTH seeds to load per operator.")
    # Stage-1 candidate source.
    p.add_argument(
        "--source",
        default="natural",
        choices=["natural", "evo2", "both"],
        help="Stage-1 candidate source: natural HTH seed pool (Pool B), live Evo 2 generation (Pool A), or both.",
    )
    p.add_argument(
        "--hth-prompt-fasta",
        type=Path,
        default=DEFAULT_HTH_PROMPT_FASTA,
        help="FASTA with ~290 nt upstream + CDS start of an E. coli HTH gene (Evo 2 Pool A prompt context).",
    )
    p.add_argument(
        "--evo2-num-prompts",
        type=int,
        default=8,
        help="Number of Evo 2 prompts/sequences to generate for Stage-1 Pool A (each 2500 nt).",
    )
    # Stage knobs.
    p.add_argument("--stage1-samples", type=int, default=2000)
    p.add_argument("--stage1-threshold", type=float, default=0.3, help="Stage-1 rejection energy cutoff.")
    # Stage-2 three-round production schedule (1/4/50 seq-per-struct, temperature escalation).
    p.add_argument("--stage2-round1-samples", type=int, default=64, help="Stage-2 round 1 samples (1 seq/struct).")
    p.add_argument("--stage2-round2-samples", type=int, default=256, help="Stage-2 round 2 samples (4 seq/struct).")
    p.add_argument("--stage2-round3-samples", type=int, default=1280, help="Stage-2 round 3 samples (50 seq/struct).")
    p.add_argument(
        "--stage2-max-templates",
        type=int,
        default=8,
        help="Stage-2 crystal docking templates used per survivor for template-guided superposition.",
    )
    p.add_argument("--stage2-round1-temp", type=float, default=0.1, help="Stage-2 round 1 LigandMPNN temperature.")
    p.add_argument("--stage2-round2-temp", type=float, default=0.2, help="Stage-2 round 2 LigandMPNN temperature.")
    p.add_argument("--stage2-round3-temp", type=float, default=0.1, help="Stage-2 round 3 LigandMPNN temperature.")
    p.add_argument("--stage3-steps", type=int, default=500)
    p.add_argument("--stage3-proposals", type=int, default=10, help="LigandMPNN proposals per MCMC step.")
    p.add_argument("--stage3-temperature", type=float, default=0.1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    """Build (and optionally run) the requested ProtoRepressor stage(s) per operator."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.promoter_designs is not None:
        designs = _read_promoter_designs(args.promoter_designs)
        operators = operators_from_promoters(designs, flank=args.operator_flank)
        logger.info(
            "Loaded %d operator(s) from %d promoter design(s) in %s.",
            len(operators),
            len(designs),
            args.promoter_designs,
        )
        # Promoter-derived operators are named promoterdesign{i}; design against all unless --operator overrides.
        if args.operator == "synpromoter77":  # the CSV default sentinel, meaningless for promoter designs
            args.operator = "all"
    else:
        operators = load_operators(args.operators_csv or DEFAULT_OPERATORS_CSV)
    if not operators:
        raise SystemExit("No operators loaded; supply --operators-csv or --promoter-designs.")

    stages = list(STAGE_BUILDERS) if args.stage == "all" else [args.stage]
    for name in resolve_operators(args, operators):
        operator = operators[name]
        seeds = [seq for seq, _ in load_seed_pool(args.seed_pool or DEFAULT_SEED_POOL, name, args.seed_limit)]
        logger.info(
            "Operator %s: motif=%s (%d bp), %d natural HTH seeds.",
            name,
            operator.motif,
            len(operator.motif),
            len(seeds),
        )
        champions: dict[str, list[Sequence]] = {}
        for stage in stages:
            designs = _run_stage(stage, args, operator, operators, seeds, champions)
            if designs:
                champions[stage] = designs


if __name__ == "__main__":
    main()
