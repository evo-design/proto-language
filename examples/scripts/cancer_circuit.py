"""NSCLC-gated HSV-TK delivery circuit: a five-stage Proto design program.

This example reproduces the structure of the NSCLC (non-small-cell lung cancer)
cancer-circuit program: a tumor-gated HSV-TK (herpes-simplex thymidine kinase)
suicide cassette whose every regulatory layer is biased toward A549 lung
adenocarcinoma and away from healthy lung. Cell-type selectivity is defined
throughout as A549 (AlphaGenome ontology ``EFO:0001086``) versus healthy lung
(``UBERON:0002048``).

The cassette, 5' -> 3', is:

    [variable 5' genomic locus] - enhancer(500) - promoter(100) -
    HSV-TK exon1(566) - synthetic intron(301) - HSV-TK exon2(569) -
    3'UTR off-switch(400) - [variable 3' genomic locus]

and is built in five separately-optimized stages, each conditioned on the
previous design step and on hypothetical lentiviral integration sites:

    1. EGFR miniprotein binder  (RFdiffusion3 + ProteinMPNN, rejection sampling,
       Boltz2/AlphaFold2 consensus interface scoring, AlphaFold3 filter)
    2. 500 bp enhancer          (Evo 2 generator, rejection sampling, AlphaGenome
       A549-vs-lung epigenomic marks)
    3. 100 bp promoter          (uniform-mutation MCMC, AlphaGenome promoter marks
       + Puffin initiation activity, scored in enhancer + host-locus context)
    4. 301 bp synthetic intron  (uniform-mutation MCMC, AlphaGenome splice-site
       usage + SpliceTransformer boundary, in full-cassette context)
    5. 400 bp 3'UTR off-switch  (uniform-mutation MCMC, miRNA off-switch + realism
       + AlphaGenome RNA-seq contrastive prior)

Regulatory stages are scored inside the gene bodies of broadly/highly-expressed
integration loci (GAPDH, ACTB, EEF1A1, FTL, plus the known lentiviral hot spot
HMGA2); flanking context is the gene-body midpoint +/- 8,192 bp (GRCh38). These
flanks ship in examples/data/integration_flanks.json and load by default; override
with ``--flanks-json``. If that file is missing the script falls back to short ``N``
padding and logs a warning, so the program still builds without reproducing context.

Several stages call GPU models (RFdiffusion3, MPNN, Boltz2/AlphaFold2/AlphaFold3,
Evo 2, AlphaGenome, Puffin, SpliceTransformer, miRanda). This script is
illustrative and is not executed in CI; use ``--dry-run`` to build a stage's
program (and validate its constraints) without running it.

Notes on fidelity:
    * The published runs split the design across many JSON configs and helper
      scripts; this single file distills the canonical recipe. Where the as-run
      configs diverge from the method description (e.g. MCMC step counts), the
      headline method values are used here and refinements are noted in comments.
    * Individual histone marks (H3K4me1, H3K27ac, H3K4me3) are scored separately
      by selecting tracks out of AlphaGenome's bundled CHIP_HISTONE output via
      ``track_name_keywords`` on the interval-track constraint.
    * The miRNA off-switch requires driver sites to be called CONCORDANTLY by
      both miRanda (thermodynamic) and TargetScan (canonical seed-match site
      typing) in-loop, and escapes tumor-high oncomiR sites under both callers.

Example:
    # Build the promoter stage and validate its constraints without running:
    PYTHONPATH=$PWD/proto-tools:$PWD python examples/scripts/cancer_circuit.py \
        --stage promoter --flanks-json integration_flanks.json --dry-run

    # Run the 3'UTR off-switch stage on GPU:
    PYTHONPATH=$PWD/proto-tools:$PWD python examples/scripts/cancer_circuit.py \
        --stage utr --flanks-json integration_flanks.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from proto_tools.transforms.masking import MaskingStrategy

from proto_language import (
    AlphaFold2BinderStructureConfig,
    StructureBasedConstraintConfig,
    structure_ipae_constraint,
    structure_iptm_constraint,
    structure_plddt_constraint,
)
from proto_language.constraint import (
    alphagenome_interval_track_constraint,
    alphagenome_splice_site_usage,
    dinucleotide_composition_constraint,
    gc_content_constraint,
    max_homopolymer_constraint,
    mirna_specificity_constraint,
    puffin_promoter_activity_constraint,
    splice_transformer_intron_boundary,
    structure_interface_contact_constraint,
    targetscan_site_constraint,
)
from proto_language.core import Constraint, Construct, Optimizer, Program, Segment, Sequence
from proto_language.generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)

logger = logging.getLogger(__name__)

# Data shipped with the example (see examples/data/): the real HSV-TK split CDS, natural
# enhancer 2048-bp upstream-context Evo 2 prompts, per-locus genomic flanks, and the measured
# natural-3'UTR dinucleotide profile.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_HSVTK_FASTA = _DATA_DIR / "hsv_tk_exons.fasta"
DEFAULT_ENHANCER_PROMPTS = _DATA_DIR / "natural_enhancer_prompts.fasta"
DEFAULT_FLANKS_JSON = _DATA_DIR / "integration_flanks.json"
DEFAULT_DINUC_JSON = _DATA_DIR / "natural_utr_dinucleotides.json"
DEFAULT_EGFR_PDB = _DATA_DIR / "egfr_ectodomain_25_645.pdb"  # AlphaFold P00533 ectodomain (residues 25-645)

# --------------------------------------------------------------------------------------
# Shared design constants (A549 NSCLC vs healthy lung)
# --------------------------------------------------------------------------------------

A549_TERMS = ["EFO:0001086"]  # A549 lung adenocarcinoma
LUNG_TERMS = ["UBERON:0002048"]  # healthy lung

# Highly/broadly-expressed integration loci scored as host context for the regulatory
# stages: GAPDH, ACTB, EEF1A1, FTL, plus the known lentiviral integration hot spot HMGA2.
REGULATORY_LOCI = ["gapdh", "actb", "eef1a1", "ftl", "hmga2"]

# EGFR ectodomain (UniProt P00533; AlphaFold model residues 25-645, chain A). The
# domain-III epitope hotspots (native EGFR numbering) bias RFdiffusion3 toward the
# receptor-dimerization face so the binder favors EGFR-high tumor cells.
EGFR_CHAIN = "A"
EGFR_HOTSPOTS = ["A384", "A408", "A409", "A443", "A465", "A467", "A468"]
AF3_IPTM_MIN = 0.7  # final AlphaFold3 interface-ipTM filter: reject designs below this

# HSV-TK split coding sequence: exon1 = first 566 bp, exon2 = final 569 bp. The synthetic
# intron sits between them; selective excision in NSCLC reconstitutes the ORF. The real
# sequence ships in examples/data/hsv_tk_exons.fasta and is loaded by default; override
# with --hsvtk-fasta.
HSV_TK_EXON1_LEN = 566
HSV_TK_EXON2_LEN = 569
INTRON_LEN = 301

# 3'UTR off-switch miRNA panel. Drivers are abundant in healthy lung and depleted in NSCLC
# (install their response elements -> silence in healthy tissue). Per-driver weight is the
# magnitude of the TCGA-LUAD tumor-vs-normal depletion (|log2 fold-change|, mean precursor RPM
# over 46 normal + 200 tumor samples; GDC open data), normalized so the strongest driver
# (miR-486-5p, -4.059) is 3.0 -- the same convention the published rounds used. miR-144-3p
# (precursor normal RPM 889.8, tumor 119.3, log2fc -2.888) was below the scan's abundance
# cutoff (normal RPM >= 1000) but is in fact a strong depletion driver; its weight uses the
# recomputed value. Mature-arm sequences are from miRBase. OncomiRs are tumor-high and ESCAPED.
_DRIVER_REF_LOG2FC = 4.059  # |log2fc| of the strongest driver (miR-486-5p), normalizes to 3.0
DRIVER_MIRNAS = {
    "hsa-miR-486-5p": ("UCCUGUACUGAGCUGCCCCGAG", 3.00),  # log2fc -4.059
    "hsa-miR-144-3p": ("UACAGUAUAGAUGAUGUACU", 2.13),  # log2fc -2.888
    "hsa-miR-197-3p": ("UUCACCACCUUCUCCACCCAGC", 2.09),  # log2fc -2.827
    "hsa-miR-451a": ("AAACCGUUACCAUUACUGAGUU", 1.82),  # log2fc -2.466
    "hsa-let-7d-5p": ("AGAGGUAGUAGGUUGCAUAGUU", 1.74),  # log2fc -2.352
}
ONCOMIR_MIRNAS = {
    "hsa-miR-21-5p": "UAGCUUAUCAGACUGAUGUUGA",
    "hsa-miR-210-3p": "CUGUGCGUGUGACAGCGGCUGA",
    "hsa-miR-31-5p": "AGGCAAGAUGCUGGCAUAGCU",
}

# Measured natural-3'UTR dinucleotide-frequency profile, computed from the natural-3'UTR
# templates used by the cancer-circuit rounds (7 UTRs, 2,093 dinucleotides). Shipped as
# examples/data/natural_utr_dinucleotides.json (loaded by default; override --dinuc-json).
NATURAL_UTR_DINUCLEOTIDES = {
    "AA": 0.0841, "AC": 0.0540, "AG": 0.0669, "AT": 0.0573,
    "CA": 0.0712, "CC": 0.0549, "CG": 0.0167, "CT": 0.0645,
    "GA": 0.0607, "GC": 0.0497, "GG": 0.0731, "GT": 0.0549,
    "TA": 0.0463, "TC": 0.0497, "TG": 0.0798, "TT": 0.1161,
}

_SPLICE_TRANSFORMER_CONTEXT_BP = 4000  # SpliceTransformer flank length each side
_ALPHAGENOME_CASSETTE_BP = 512  # cassette context length wrapped around the splice target
_ALPHAGENOME_FALLBACK_FLANK = "N" * 8192  # placeholder when --flanks-json is absent (mirrors +/- 8,192 bp)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def load_flanks(flanks_json: Path | None) -> dict[str, dict[str, str]]:
    """Load per-locus genomic flanks, or fall back to N-padding with a warning.

    Args:
        flanks_json (Path | None): Path to ``{locus: {"left":..., "right":...}}`` JSON
            (gene-body midpoint +/- 8,192 bp, GRCh38). If ``None`` or missing, short
            ``N`` flanks are returned so the program still builds.

    Returns:
        dict[str, dict[str, str]]: Mapping of locus -> ``{"left":..., "right":...}``.
    """
    if flanks_json is not None and flanks_json.exists():
        flanks = json.loads(flanks_json.read_text())
        return {k.lower(): {"left": v["left"], "right": v["right"]} for k, v in flanks.items()}
    logger.warning(
        "No --flanks-json provided (or file missing); using placeholder N flanks. "
        "Supply integration_flanks.json (gene-body midpoint +/- 8,192 bp, GRCh38) to "
        "reproduce host-locus context."
    )
    return {locus: {"left": _ALPHAGENOME_FALLBACK_FLANK, "right": _ALPHAGENOME_FALLBACK_FLANK} for locus in REGULATORY_LOCI}


def _read_fasta(path: Path | None) -> list[str]:
    """Read sequences from a FASTA or one-per-line file (empty list if missing)."""
    if path is None or not Path(path).exists():
        return []
    text = Path(path).read_text()
    if not text.lstrip().startswith(">"):
        return [ln.strip().upper() for ln in text.splitlines() if ln.strip()]
    seqs: list[str] = []
    block: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if block:
                seqs.append("".join(block).upper())
                block = []
        else:
            block.append(line.strip())
    if block:
        seqs.append("".join(block).upper())
    return seqs


def load_templates(path: Path | None, length: int, n: int, label: str) -> list[str]:
    """Load seed templates from a FASTA/one-per-line file, padded/cycled to ``n`` of ``length``.

    Args:
        path (Path | None): File of natural seed sequences (FASTA or plain lines). If
            ``None``/missing, returns ``n`` neutral poly-A seeds and warns.
        length (int): Target seed length (truncated/padded with A).
        n (int): Number of seeds to return.
        label (str): Human-readable name for logging.

    Returns:
        list[str]: Exactly ``n`` seed sequences of the requested length.
    """
    seqs = _read_fasta(path)
    if not seqs:
        logger.warning("No %s templates at %s; seeding from neutral poly-A.", label, path)
        seqs = ["A" * length]
    fitted = [(s + "A" * length)[:length] for s in seqs]
    return [fitted[i % len(fitted)] for i in range(n)]


def seed_segment(segment: Segment, seeds: list[str]) -> None:
    """Seed a segment's result/proposal pool from natural template sequences."""
    seeded = [Sequence(sequence=s, sequence_type="dna", metadata={"seed_source": "natural_template"}) for s in seeds]
    segment.result_sequences = seeded
    segment.proposal_sequences = [
        Sequence.from_dict(s.to_dict(include_logits=True, include_structure=True)) for s in seeded
    ]


def resolve_loci(args: argparse.Namespace) -> list[str]:
    """Host integration loci to score over (all REGULATORY_LOCI, or the --loci subset)."""
    if not getattr(args, "loci", None):
        return REGULATORY_LOCI
    chosen = [locus.strip().lower() for locus in args.loci.split(",") if locus.strip()]
    unknown = [locus for locus in chosen if locus not in REGULATORY_LOCI]
    if unknown:
        raise ValueError(f"--loci has unknown loci {unknown}; choose from {REGULATORY_LOCI}.")
    return chosen


def load_enhancer_seeds(args: argparse.Namespace) -> list[str]:
    """Top-N stage-2 enhancer champions used as upstream context for the promoter.

    Reads --enhancer-seeds (the top stage-2 enhancers); falls back to a single
    --enhancer-seed, then to the shipped natural-enhancer set as stand-ins. Capped at
    --num-contexts (the method's top-5) and trimmed to 500 bp.
    """
    seeds = _read_fasta(args.enhancer_seeds) if args.enhancer_seeds else []
    if not seeds and args.enhancer_seed:
        seeds = [args.enhancer_seed.upper()]
    if not seeds:
        logger.warning("No --enhancer-seeds; using shipped natural enhancers as stand-in champions.")
        seeds = _read_fasta(DEFAULT_ENHANCER_PROMPTS)
    seeds = [s[:500] for s in seeds[: args.num_contexts]]
    return seeds or [""]


def load_cassette_contexts(args: argparse.Namespace) -> list[str]:
    """Top-N promoter-enhancer cassette contexts (upstream of the intron) for stage 4.

    Reads --cassette-contexts directly, else pairs each top-N enhancer seed with a
    natural promoter template to form enhancer+promoter cassettes. Capped at --num-contexts.
    """
    explicit = _read_fasta(args.cassette_contexts) if args.cassette_contexts else []
    if explicit:
        return explicit[: args.num_contexts] or [""]
    enhancers = load_enhancer_seeds(args)
    promoters = load_templates(args.promoter_templates, 100, len(enhancers), "promoter")
    return [enh + prom for enh, prom in zip(enhancers, promoters, strict=True)][: args.num_contexts] or [""]


def _ag_track(
    segment: Segment,
    output: str,
    direction: str,
    weight: float,
    interval_len: int,
    left: str,
    right: str,
    device: str,
    label: str,
    track_keywords: list[str] | None = None,
) -> Constraint:
    """Build one A549-vs-lung AlphaGenome interval-track constraint over a host locus.

    ``track_keywords`` selects individual tracks within a bundled output -- e.g.
    ``requested_output='CHIP_HISTONE'`` with ``track_keywords=['H3K4me1']`` scores only
    that histone mark instead of the mean over all histone tracks.

    ``direction='maximize'`` maximizes the A549-vs-lung *margin* (contrastive); ``'minimize'``
    minimizes the absolute A549 signal to push a mark down (e.g. suppress a promoter-like
    signature in the enhancer). The interval-track constraint ignores ``direction`` whenever
    contrastive terms are set, so contrast is only applied for the maximize case.
    """
    return Constraint(
        inputs=[segment],
        function=alphagenome_interval_track_constraint,
        function_config={
            "intervals": [(0, interval_len)],
            "ontology_terms": A549_TERMS,
            "contrastive_ontology_terms": LUNG_TERMS if direction == "maximize" else None,
            "requested_output": output,
            "track_name_keywords": track_keywords,
            "direction": direction,
            "left_context": left,
            "right_context": right,
            "device": device,
        },
        weight=weight,
        label=label,
    )


# --------------------------------------------------------------------------------------
# Stage 1: EGFR miniprotein binder
# --------------------------------------------------------------------------------------


def build_binder_stage(args: argparse.Namespace) -> tuple[Program, Segment]:
    """De-novo EGFR binder via RFdiffusion3 + ProteinMPNN under consensus interface scoring.

    Backbones are generated by epitope-centered RFdiffusion3 and sequences by ProteinMPNN;
    a rejection-sampling optimizer keeps the lowest-energy designs. Each candidate is scored
    on a Boltz2 + AlphaFold2 interface consensus (ipTM, ipAE, interface-contact, pLDDT), then
    an AlphaFold3 interface-ipTM filter rejects designs below ``AF3_IPTM_MIN``.

    Args:
        args (argparse.Namespace): Parsed CLI options (target structure, device, seed, ...).

    Returns:
        tuple[Program, Segment]: The binder program and the designed binder segment.
    """
    from proto_tools import ProteinMPNNSampleConfig, RFdiffusion3Config
    from proto_tools.entities.structures import Structure

    from proto_language.generator import (
        RFdiffusionMPNNBinderGenerator,
        RFdiffusionMPNNBinderGeneratorConfig,
    )

    target_pdb = args.target_pdb or DEFAULT_EGFR_PDB
    if not Path(target_pdb).exists():
        raise FileNotFoundError(
            f"Stage 'binder' needs the EGFR ectodomain structure at {target_pdb} (UniProt P00533, "
            "AlphaFold residues 25-645, chain A); ships in examples/data/, or pass --target-pdb."
        )
    target_structure = Structure(structure=Path(target_pdb).read_text())
    target_sequence = target_structure.get_chain_sequence(EGFR_CHAIN, remove_non_standard=True)

    # One source (the target Structure) -> two artifacts: coordinates for the generator to
    # dock against, and the chain sequence for the fixed target segment the constraints fold.
    binder = Segment(length=args.binder_length, sequence_type="protein", label="EGFR binder")
    target = Segment(sequence=target_sequence, sequence_type="protein", label="EGFR target")
    construct = Construct([binder, target])

    generator = RFdiffusionMPNNBinderGenerator(
        RFdiffusionMPNNBinderGeneratorConfig(
            target_structure=target_structure,
            target_chains=[EGFR_CHAIN],
            hotspots=EGFR_HOTSPOTS,  # epitope-centered RFdiffusion3 origin (domain III)
            inverse_folding="proteinmpnn",
            rfdiffusion3_config=RFdiffusion3Config(device=args.device),
            proteinmpnn_config=ProteinMPNNSampleConfig(num_sequences_per_structure=1, device=args.device),
        )
    )
    generator.assign(binder)

    # AlphaFold2 (binder mode) exposes the interface metrics; one recycle as published. The
    # target template + chains let AF2 fold the target+binder complex for interface scoring.
    af2_cfg = StructureBasedConstraintConfig(
        structure_tool="alphafold2_binder",
        alphafold2_binder_config=AlphaFold2BinderStructureConfig(
            target_pdb=str(target_pdb),
            target_chains=[EGFR_CHAIN],
            binder_chain=None,  # de novo design: the binder is generated, not redesigned from a template chain
            num_recycles=1,
            intra_contact_num=2,
            intra_contact_cutoff=14.0,
            inter_contact_num=2,
            inter_contact_cutoff=20.0,
        ),
    )
    boltz2_cfg = StructureBasedConstraintConfig(structure_tool="boltz2")

    # Consensus interface objective. The published selection score is min(Boltz2 ipTM,
    # AF2 ipTM); the optimizer minimizes a weighted sum, so both ipTM terms enter at
    # weight 1.0 here (each pushes the weaker oracle up, approximating the min). ipTM and
    # pLDDT are scored on BOTH oracles (consensus); ipAE and interface-contact are AF2-only
    # (Boltz2 exposes no interface ipAE, and the interface-contact term is AF2-binder-only).
    consensus = [
        Constraint([binder, target], structure_iptm_constraint, boltz2_cfg, weight=1.0, label="boltz2_iptm"),
        Constraint([binder, target], structure_iptm_constraint, af2_cfg, weight=1.0, label="af2_iptm"),
        Constraint([binder, target], structure_plddt_constraint, boltz2_cfg, weight=0.1, label="boltz2_plddt"),
        Constraint([binder, target], structure_plddt_constraint, af2_cfg, weight=0.1, label="af2_plddt"),
        Constraint([binder, target], structure_ipae_constraint, af2_cfg, weight=1.0, label="af2_ipae"),
        Constraint(
            [binder, target], structure_interface_contact_constraint, af2_cfg, weight=0.25, label="interface_contact"
        ),
    ]

    # 48 candidates per round, keep top 8; repeat for --rounds rounds (default 15 -> 720
    # total). Chained rejection-sampling stages share the construct by identity, so each
    # round reseeds from the running survivors. The retained count is the published 8, but
    # capped at --candidates-per-round (so small candidate counts for quick tests are valid).
    retained = min(8, args.candidates_per_round)
    optimizers: list[Optimizer] = []
    for round_idx in range(args.rounds):
        round_gen = generator if round_idx == 0 else RFdiffusionMPNNBinderGenerator(generator.config)
        round_gen.assign(binder)
        round_constraints = (
            consensus
            if round_idx == 0
            else [Constraint(c.inputs, c.function, c.function_config, weight=c.weight, label=c.label) for c in consensus]
        )
        optimizers.append(
            RejectionSamplingOptimizer(
                constructs=[construct],
                generators=[round_gen],
                constraints=round_constraints,
                config=RejectionSamplingOptimizerConfig(num_samples=args.candidates_per_round, num_results=retained),
            )
        )

    # Final filtering stage: AlphaFold3 interface ipTM, reject ipTM < AF3_IPTM_MIN. The
    # ipTM constraint maps higher ipTM to lower energy, so the pass band is score <= 1 - min.
    # AF3 weights are access-gated and must be provisioned separately; --no-af3-filter omits
    # this stage for environments where AF3 is unavailable (the Boltz2/AF2 consensus remains).
    if not args.no_af3_filter:
        af3_cfg = StructureBasedConstraintConfig(structure_tool="alphafold3")
        af3_filter = Constraint(
            [binder, target],
            structure_iptm_constraint,
            af3_cfg,
            threshold=1.0 - AF3_IPTM_MIN,
            label="af3_iptm_filter",
        )
        # 'existing_results' re-scores the surviving binders (no generation), so no generator.
        optimizers.append(
            RejectionSamplingOptimizer(
                constructs=[construct],
                generators=[],
                constraints=[af3_filter],
                config=RejectionSamplingOptimizerConfig(
                    num_samples=retained, num_results=min(args.num_results, retained), proposal_source="existing_results"
                ),
            )
        )

    program = Program(optimizers=optimizers, num_results=args.num_results, seed=args.seed)
    return program, binder


# --------------------------------------------------------------------------------------
# Stage 2: 500 bp enhancer (Evo 2 + rejection sampling)
# --------------------------------------------------------------------------------------


def build_enhancer_stage(
    args: argparse.Namespace, flanks: dict[str, dict[str, str]], champions: dict[str, list[str]] | None = None
) -> tuple[Program, Segment]:
    """500 bp enhancer designed with Evo 2 under A549-vs-lung AlphaGenome epigenomic scoring.

    The enhancer is the first regulatory stage, so ``champions`` (upstream designs) is unused.

    Args:
        args (argparse.Namespace): Parsed CLI options.
        flanks (dict[str, dict[str, str]]): Per-locus genomic flanks for host context.
        champions (dict[str, list[str]] | None): Upstream-stage design champions for end-to-end
            auto-chaining (--stage all); falls back to CLI/defaults when None.

    Returns:
        tuple[Program, Segment]: The enhancer program and the designed enhancer segment.
    """
    del champions  # first regulatory stage: no upstream context
    from proto_language.generator import Evo2Generator, Evo2GeneratorConfig

    enhancer = Segment(length=500, sequence_type="dna", label="A549 enhancer")
    construct = Construct([enhancer])

    # Evo 2 prompts: upstream context derived from natural enhancer sequences. The set of
    # natural enhancers ships in examples/data/natural_enhancer_prompts.fasta; each is used
    # as a prompt so generation is seeded toward enhancer-like context. Falls back to the
    # HMGA2 genomic left-flank tail if the prompt file is absent.
    # Evo2 generates one candidate per call against a single prompt, so each run uses ONE
    # 2048-bp natural-enhancer upstream-context prompt, selected from the shipped set by --seed.
    # Varying --seed across runs draws different natural-enhancer contexts (this is how the
    # published "7,500 candidates total" aggregate over runs is produced).
    natural_enhancers = _read_fasta(args.enhancer_prompts or DEFAULT_ENHANCER_PROMPTS)
    if natural_enhancers:
        prompt = natural_enhancers[args.seed % len(natural_enhancers)][-args.enhancer_prompt_bp :]
    else:
        logger.warning("No enhancer prompts found; falling back to HMGA2 genomic left-flank tail.")
        prompt = flanks.get("hmga2", flanks[REGULATORY_LOCI[0]])["left"][-args.enhancer_prompt_bp :]
    prompts = [prompt]
    generator = Evo2Generator(
        Evo2GeneratorConfig(prompts=prompts, model_checkpoint="evo2_7b", device=args.device, stop_at_eos=False)
    )
    generator.assign(enhancer)

    # Objective: push toward active-enhancer histone marks (H3K4me1, H3K27ac) and open
    # chromatin (ATAC/DNASE), and AWAY from promoter-like marks (H3K4me3, CAGE); all A549 vs
    # lung, summed across host loci. Individual histone marks are pulled out of the bundled
    # CHIP_HISTONE output via track_name_keywords (see AlphaGenomeIntervalTrackConfig).
    enhancer_marks = [
        ("CHIP_HISTONE", ["H3K4me1"], "maximize", 4.0),
        ("CHIP_HISTONE", ["H3K27ac"], "maximize", 4.0),
        ("CHIP_HISTONE", ["H3K4me3"], "minimize", 1.0),  # suppress promoter-like signature
        ("ATAC", None, "maximize", 3.0),
        ("DNASE", None, "maximize", 1.0),
        ("CAGE", None, "minimize", 2.0),
    ]
    constraints: list[Constraint] = []
    for locus in resolve_loci(args):
        left, right = flanks[locus]["left"], flanks[locus]["right"]
        for output, keywords, direction, weight in enhancer_marks:
            tag = keywords[0] if keywords else output
            constraints.append(
                _ag_track(
                    enhancer, output, direction, weight, 500, left, right, args.device,
                    f"enh_{tag}_{locus}", track_keywords=keywords,
                )
            )

    optimizer = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=RejectionSamplingOptimizerConfig(num_samples=args.enhancer_samples, num_results=args.num_results),
    )
    program = Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed)
    return program, enhancer


# --------------------------------------------------------------------------------------
# Stage 3: 100 bp promoter (uniform-mutation MCMC)
# --------------------------------------------------------------------------------------


def build_promoter_stage(
    args: argparse.Namespace, flanks: dict[str, dict[str, str]], champions: dict[str, list[str]] | None = None
) -> tuple[Program, Segment]:
    """100 bp promoter optimized in enhancer + host-locus context with AlphaGenome + Puffin.

    The promoter is scored embedded immediately 3' of the upstream enhancer and 5' of the
    HSV-TK coding sequence, so each design is evaluated with its regulatory neighbors and
    host chromatin present.

    Args:
        args (argparse.Namespace): Parsed CLI options.
        flanks (dict[str, dict[str, str]]): Per-locus genomic flanks for host context.
        champions (dict[str, list[str]] | None): Upstream-stage design champions for end-to-end
            auto-chaining (--stage all); falls back to CLI/defaults when None.

    Returns:
        tuple[Program, Segment]: The promoter program and the designed promoter segment.
    """
    promoter = Segment(length=100, sequence_type="dna", label="A549 promoter")
    construct = Construct([promoter])
    seed_segment(promoter, load_templates(args.promoter_templates, 100, args.num_results, "promoter"))

    # Uniform-mutation generator: 3 substitutions per proposal, seeded from natural promoters.
    generator = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3)))
    generator.assign(promoter)

    # Top-5 stage-2 enhancer champions: the promoter is scored 3' of each, summed across
    # both the enhancer set and the host loci (method: "top 5 seeds from stage 2"). When
    # run end-to-end (--stage all), these are the actual stage-2 enhancer outputs (auto-
    # chained); otherwise they come from --enhancer-seeds / the shipped stand-ins.
    enhancer_seeds = ((champions or {}).get("enhancer") or load_enhancer_seeds(args))[: args.num_contexts]
    hsv_exon1 = _load_hsvtk(args)[0]

    # A single AlphaGenome term (total weight 6.0, matching Puffin) over the promoter interval.
    # The initiation/active marks CAGE, H3K4me3, H3K27ac, and DNase maximize the A549-vs-lung
    # margin (per-mark weighting prioritizes the initiation marks CAGE + H3K4me3); the enhancer
    # mark H3K4me1 is minimized to keep the design promoter-like rather than enhancer-like. The
    # per-mark weights below sum to 6.0; Puffin enters separately, also at weight 6.0. Individual
    # histone marks are pulled from CHIP_HISTONE via track_name_keywords.
    promoter_marks = [
        ("CAGE", None, "maximize", 2.0),
        ("CHIP_HISTONE", ["H3K4me3"], "maximize", 2.0),
        ("CHIP_HISTONE", ["H3K27ac"], "maximize", 1.0),
        ("DNASE", None, "maximize", 0.5),
        ("CHIP_HISTONE", ["H3K4me1"], "minimize", 0.5),
    ]
    constraints: list[Constraint] = []
    for ei, enhancer in enumerate(enhancer_seeds):
        for locus in resolve_loci(args):
            # Enhancer sits 5' of the promoter; HSV-TK exon1 sits 3' of it, in host chromatin.
            left = (flanks[locus]["left"] + enhancer)[-8192:] or flanks[locus]["left"]
            right = (hsv_exon1 + flanks[locus]["right"])[:8192]
            for output, keywords, direction, weight in promoter_marks:
                tag = keywords[0] if keywords else output
                constraints.append(
                    _ag_track(
                        promoter, output, direction, weight, 100, left, right, args.device,
                        f"prom_{tag}_e{ei}_{locus}", track_keywords=keywords,
                    )
                )
            # Puffin promoter-initiation activity as a weighted optimization term (weight 6.0,
            # no hard filter): the activity/sharpness thresholds are graded targets the
            # constraint scores toward, contributing to the objective rather than gating.
            constraints.append(
                Constraint(
                    inputs=[promoter],
                    function=puffin_promoter_activity_constraint,
                    function_config={
                        "left_context": left,
                        "right_context": right,
                        "activity_threshold": 0.08,
                        "sharpness_threshold": 0.15,
                    },
                    weight=6.0,
                    label=f"prom_puffin_e{ei}_{locus}",
                )
            )

    optimizer = MCMCOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=MCMCOptimizerConfig(
            num_results=args.num_results,
            num_steps=args.promoter_steps,
            max_temperature=1.0,
            min_temperature=1e-3,
            temperature_schedule="exponential",
            seed=args.seed,
        ),
    )
    program = Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed)
    return program, promoter


# --------------------------------------------------------------------------------------
# Stage 4: 301 bp synthetic intron (uniform-mutation MCMC, full-cassette context)
# --------------------------------------------------------------------------------------


def build_intron_stage(
    args: argparse.Namespace, flanks: dict[str, dict[str, str]], champions: dict[str, list[str]] | None = None
) -> tuple[Program, Segment]:
    """301 bp synthetic intron between HSV-TK exons, gated for NSCLC-selective excision.

    The intron is optimized in the full cassette (HSV-TK exon flanks + promoter/enhancer +
    genomic safe-harbor context). The canonical GT donor and AG acceptor dinucleotides are
    held fixed; AlphaGenome splice-site usage maximizes excision in A549 and minimizes it in
    healthy lung, with a low-weight SpliceTransformer boundary term preserving site geometry.

    Args:
        args (argparse.Namespace): Parsed CLI options.
        flanks (dict[str, dict[str, str]]): Per-locus genomic flanks for host context.
        champions (dict[str, list[str]] | None): Upstream-stage design champions for end-to-end
            auto-chaining (--stage all); falls back to CLI/defaults when None.

    Returns:
        tuple[Program, Segment]: The intron program and the designed intron segment.
    """
    exon1, exon2 = _load_hsvtk(args)

    # Three-segment construct: fixed exon1, variable intron, fixed exon2. Each segment is a
    # single initial sequence (the fixed flanks carry one proposal each, so the multi-segment
    # constraints stay aligned); the MCMC optimizer expands to num_results trajectories. The
    # intron is initialized from a natural/synthetic template with canonical GT...AG ends.
    intron_seed = load_templates(args.intron_templates, INTRON_LEN, 1, "intron")[0]
    intron_seed = "GT" + intron_seed[2:-2] + "AG"
    exon1_seg = Segment(sequence=exon1, sequence_type="dna", label="HSV-TK exon1")
    intron = Segment(sequence=intron_seed, sequence_type="dna", label="HSV-TK synthetic intron")
    exon2_seg = Segment(sequence=exon2, sequence_type="dna", label="HSV-TK exon2")
    construct = Construct([exon1_seg, intron, exon2_seg])

    # 3 substitutions per proposal, freezing the GT donor (positions 1-2) and AG acceptor
    # (positions L-1, L); 1-indexed within the variable intron.
    generator = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(
            masking_strategy=MaskingStrategy(num_mutations=3, fixed_positions=[1, 2, INTRON_LEN - 1, INTRON_LEN])
        )
    )
    generator.assign(intron)

    # Splice positions (0-indexed) following the published convention: donor at the last base
    # of exon1, acceptor at the intron's 3' boundary.
    donor_pos = len(exon1) - 1
    acceptor_pos = len(exon1) + INTRON_LEN
    # SpliceTransformer requires exactly 4000 bp of flanking context on each side of the
    # [exon1, intron, exon2] target; use the host-locus genomic flanks (padded with N to
    # guarantee the exact length).
    _loc0 = resolve_loci(args)[0]
    st_left = ("N" * _SPLICE_TRANSFORMER_CONTEXT_BP + flanks[_loc0]["left"])[-_SPLICE_TRANSFORMER_CONTEXT_BP:]
    st_right = (flanks[_loc0]["right"] + "N" * _SPLICE_TRANSFORMER_CONTEXT_BP)[:_SPLICE_TRANSFORMER_CONTEXT_BP]

    # Top-5 promoter-enhancer cassette contexts (method: "top five promoter-enhancer
    # combinations"). Each becomes the cassette context wrapped 5' of the splice target;
    # SSU is summed with equal weight across cassette contexts AND genomic integration loci.
    # When auto-chained (--stage all), pair the actual stage-2 enhancer and stage-3 promoter
    # champions; otherwise fall back to --cassette-contexts / paired stand-ins.
    ch = champions or {}
    if ch.get("enhancer") and ch.get("promoter"):
        cassette_contexts = [e + p for e, p in zip(ch["enhancer"], ch["promoter"], strict=False)][: args.num_contexts]
    else:
        cassette_contexts = load_cassette_contexts(args)

    constraints: list[Constraint] = []
    for ci, cassette in enumerate(cassette_contexts):
        cassette_left = cassette[-_ALPHAGENOME_CASSETTE_BP:]  # enhancer+promoter immediately 5' of exon1
        for locus in resolve_loci(args):
            genomic_context = flanks[locus]["left"] + flanks[locus]["right"]
            # Maximize splice-site usage in A549, minimize in lung (matched weight 8.0), at the
            # fixed donor/acceptor within a 3-nt peak-search radius on the positive strand.
            for direction, terms, weight, tag in (
                ("max", A549_TERMS, 8.0, "a549"),
                ("min", LUNG_TERMS, 8.0, "lung"),
            ):
                constraints.append(
                    Constraint(
                        inputs=[exon1_seg, intron, exon2_seg],
                        function=alphagenome_splice_site_usage,
                        function_config={
                            "genomic_context": genomic_context,
                            "cassette_left_context": cassette_left,
                            "cassette_right_context": "",
                            "ontology_terms": terms,
                            "splice_pos": [donor_pos, acceptor_pos],
                            "direction": direction,
                            "peak_search_radius": 3,
                            "strand": "positive",
                            "device": args.device,
                        },
                        weight=weight,
                        label=f"ssu_{tag}_c{ci}_{locus}",
                    )
                )
    # Low-weight SpliceTransformer boundary term: preserve correctly placed splice geometry.
    constraints.append(
        Constraint(
            inputs=[exon1_seg, intron, exon2_seg],
            function=splice_transformer_intron_boundary,
            function_config={
                "left_context": st_left,
                "right_context": st_right,
                "donor_pos": [donor_pos],
                "acceptor_pos": [acceptor_pos],
                "reduction": "mean",  # score = 1 - 0.5*(p_donor + p_acceptor)
            },
            weight=0.5,
            label="splice_transformer_boundary",
        )
    )

    optimizer = MCMCOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=MCMCOptimizerConfig(
            num_results=args.num_results,
            num_steps=args.intron_steps,
            max_temperature=1e-2,
            min_temperature=1e-3,
            temperature_schedule="exponential",
            seed=args.seed,
        ),
    )
    program = Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed)
    return program, intron


# --------------------------------------------------------------------------------------
# Stage 5: 400 bp 3'UTR off-switch (uniform-mutation MCMC)
# --------------------------------------------------------------------------------------


def build_utr_stage(
    args: argparse.Namespace, flanks: dict[str, dict[str, str]], champions: dict[str, list[str]] | None = None
) -> tuple[Program, Segment]:
    """400 bp 3'UTR off-switch installing healthy-lung miRNA sites while escaping oncomiRs.

    The objective installs microRNA response elements for drivers abundant in healthy lung
    and depleted in NSCLC (weighted by TCGA-LUAD lung-vs-A549 abundance) while escaping
    tumor-high oncomiR sites, enforces natural-3'UTR sequence realism (dinucleotide
    composition, GC, homopolymer), and adds a low-weight AlphaGenome A549-vs-lung RNA-seq
    contrastive prior across host loci.

    Args:
        args (argparse.Namespace): Parsed CLI options.
        flanks (dict[str, dict[str, str]]): Per-locus genomic flanks for host context.
        champions (dict[str, list[str]] | None): Upstream-stage design champions for end-to-end
            auto-chaining (--stage all); falls back to CLI/defaults when None.

    Returns:
        tuple[Program, Segment]: The off-switch program and the designed 3'UTR segment.
    """
    utr = Segment(length=400, sequence_type="dna", label="3' UTR off-switch")
    construct = Construct([utr], label="3' UTR off-switch")
    seed_segment(utr, load_templates(args.utr_templates, 400, args.num_results, "3'UTR"))

    generator = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=3)))
    generator.assign(utr)

    constraints: list[Constraint] = []

    # Driver sites must be called CONCORDANTLY by BOTH miRanda (thermodynamic duplex) AND
    # TargetScan (canonical seed-match site typing). Adding both terms per driver pushes the
    # optimizer toward sites both callers agree on. One constraint per driver (each saturates
    # on a single site) rewards site DIVERSITY across drivers rather than one strong site.
    # Per-driver weight is the TCGA-LUAD lung-vs-A549 ratio, normalized so the strongest
    # driver enters at 3.0; the weight is split evenly across the two callers.
    total_w = sum(w for _, w in DRIVER_MIRNAS.values())
    for mid, (query, ratio) in DRIVER_MIRNAS.items():
        driver_weight = 3.0 * ratio / total_w
        constraints.append(
            Constraint(
                inputs=[utr],
                function=mirna_specificity_constraint,
                function_config={
                    "mirna_queries": [query],
                    "mirna_ids": [mid],
                    "mirna_weights": [1.0],
                    "direction": "maximize",
                    "repression_threshold": 0.8,
                    "miranda_config": {"score_threshold": 140.0, "energy_threshold": -20.0},
                },
                weight=0.5 * driver_weight,
                label=f"driver_miranda_{mid}",
            )
        )
        constraints.append(
            Constraint(
                inputs=[utr],
                function=targetscan_site_constraint,
                function_config={
                    "mirna_queries": [query],
                    "mirna_ids": [mid],
                    "direction": "maximize",
                    "repression_threshold": 1.0,  # one good (>= 7mer) seed site suffices per driver
                    "include_6mer": False,
                },
                weight=0.5 * driver_weight,
                label=f"driver_targetscan_{mid}",
            )
        )
    # OncomiR escape: minimize any sites for tumor-high oncomiRs, by BOTH callers.
    oncomir_queries = list(ONCOMIR_MIRNAS.values())
    oncomir_ids = list(ONCOMIR_MIRNAS.keys())
    constraints.append(
        Constraint(
            inputs=[utr],
            function=mirna_specificity_constraint,
            function_config={
                "mirna_queries": oncomir_queries,
                "mirna_ids": oncomir_ids,
                "direction": "minimize",
                "repression_threshold": 2.0,
            },
            weight=1.0,
            label="oncomir_escape_miranda",
        )
    )
    constraints.append(
        Constraint(
            inputs=[utr],
            function=targetscan_site_constraint,
            function_config={
                "mirna_queries": oncomir_queries,
                "mirna_ids": oncomir_ids,
                "direction": "minimize",
                "repression_threshold": 2.0,
            },
            weight=1.0,
            label="oncomir_escape_targetscan",
        )
    )

    # Realism against measured natural-3'UTR statistics (shipped profile loaded by default).
    dinuc_path = args.dinuc_json or DEFAULT_DINUC_JSON
    dinuc_ref = json.loads(Path(dinuc_path).read_text()) if Path(dinuc_path).exists() else NATURAL_UTR_DINUCLEOTIDES
    constraints.append(
        Constraint(
            inputs=[utr],
            function=dinucleotide_composition_constraint,
            function_config={"reference_frequencies": dinuc_ref},
            weight=0.6,
            label="dinucleotide_realism",
        )
    )
    constraints.append(
        Constraint(inputs=[utr], function=gc_content_constraint, function_config={"min_gc": 35, "max_gc": 60}, weight=0.4, label="gc_realism")
    )
    constraints.append(
        Constraint(
            inputs=[utr], function=max_homopolymer_constraint, function_config={"max_length": 12}, weight=0.4, label="homopolymer_realism"
        )
    )

    # Low-weight (total 0.15) AlphaGenome RNA-seq A549-vs-lung contrastive prior, averaged
    # across genomic contexts AND the optimized upstream circuit elements: the 3'UTR is scored
    # with the full designed cassette (enhancer + promoter + HSV-TK exon1 + intron + exon2)
    # placed immediately 5' of it inside each host locus. The 0.15 weight is split evenly
    # across the contexts so the term averages to 0.15.
    # The optimized upstream circuit. When auto-chained (--stage all) these are the actual
    # enhancer/promoter/intron champions from the preceding stages; otherwise stand-ins.
    exon1, exon2 = _load_hsvtk(args)
    ch = champions or {}
    enh = (ch.get("enhancer") or load_enhancer_seeds(args))[0]
    prom = (ch.get("promoter") or load_templates(args.promoter_templates, 100, 1, "promoter"))[0]
    intr = (ch.get("intron") or [load_templates(args.intron_templates, INTRON_LEN, 1, "intron")[0]])[0]
    upstream_circuit = enh + prom + exon1 + intr + exon2
    rnaseq_loci = [loc for loc in ["hmga2", "gapdh", "eef1a1"] if loc in resolve_loci(args)] or resolve_loci(args)[:1]
    rnaseq_weight = 0.15 / len(rnaseq_loci)
    for locus in rnaseq_loci:
        f = flanks.get(locus, {"left": _ALPHAGENOME_FALLBACK_FLANK, "right": _ALPHAGENOME_FALLBACK_FLANK})
        left = f["left"] + upstream_circuit  # circuit sits immediately 5' of the UTR
        constraints.append(
            _ag_track(utr, "RNA_SEQ", "maximize", rnaseq_weight, 400, left, f["right"], args.device, f"utr_rnaseq_{locus}")
        )

    optimizer = MCMCOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=MCMCOptimizerConfig(
            num_results=args.num_results,
            num_steps=args.utr_steps,
            max_temperature=1e-2,
            min_temperature=1e-3,
            temperature_schedule="exponential",
            seed=args.seed,
        ),
    )
    program = Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed)
    return program, utr


# --------------------------------------------------------------------------------------
# HSV-TK loading + CLI
# --------------------------------------------------------------------------------------


def _load_hsvtk(args: argparse.Namespace) -> tuple[str, str]:
    """Return (exon1, exon2) HSV-TK coding fragments from a 2-record FASTA (exon1, exon2).

    Defaults to the real HSV-TK split CDS shipped in examples/data/hsv_tk_exons.fasta;
    override with --hsvtk-fasta. A single concatenated record is sliced by the published
    exon lengths.
    """
    path = args.hsvtk_fasta or DEFAULT_HSVTK_FASTA
    records = _read_fasta(Path(path))
    if len(records) >= 2:
        return records[0], records[1]
    if len(records) == 1:
        seq = records[0]
        return seq[:HSV_TK_EXON1_LEN], seq[HSV_TK_EXON1_LEN : HSV_TK_EXON1_LEN + HSV_TK_EXON2_LEN]
    logger.warning("HSV-TK FASTA %s not found; using placeholder poly-A exons.", path)
    return "A" * HSV_TK_EXON1_LEN, "A" * HSV_TK_EXON2_LEN


STAGE_BUILDERS = {
    "binder": "build_binder_stage",
    "enhancer": "build_enhancer_stage",
    "promoter": "build_promoter_stage",
    "intron": "build_intron_stage",
    "utr": "build_utr_stage",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the cancer-circuit example."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=[*STAGE_BUILDERS, "all"], default="all", help="Which design stage to build/run.")
    p.add_argument("--dry-run", action="store_true", help="Build the program (validate constraints) without running.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-results", type=int, default=10, help="Designs retained per stage.")
    p.add_argument("--output-dir", type=Path, default=Path("cancer_circuit_outputs"))
    # Context / data inputs.
    p.add_argument(
        "--flanks-json", type=Path, default=None,
        help="Per-locus genomic flanks JSON; defaults to examples/data/integration_flanks.json (5 loci).",
    )
    p.add_argument(
        "--hsvtk-fasta", type=Path, default=None,
        help="HSV-TK exon FASTA (exon1, exon2); defaults to examples/data/hsv_tk_exons.fasta.",
    )
    p.add_argument("--promoter-templates", type=Path, default=None, help="Natural promoter seed FASTA.")
    p.add_argument("--intron-templates", type=Path, default=None, help="Natural/synthetic intron seed FASTA (GT...AG).")
    p.add_argument("--utr-templates", type=Path, default=None, help="Natural 3'UTR seed FASTA.")
    p.add_argument("--dinuc-json", type=Path, default=None, help="Measured natural-3'UTR dinucleotide profile JSON.")
    p.add_argument(
        "--enhancer-prompts", type=Path, default=None,
        help="Natural-enhancer FASTA used as Evo2 prompts; defaults to examples/data/natural_enhancer_prompts.fasta.",
    )
    p.add_argument("--enhancer-prompt-bp", type=int, default=2048, help="Evo2 prompt length (bp) per natural enhancer.")
    p.add_argument("--enhancer-seed", default=None, help="Single champion enhancer to place 5' of the promoter.")
    p.add_argument(
        "--enhancer-seeds", type=Path, default=None,
        help="FASTA of top stage-2 enhancer champions scored as promoter upstream context (top-5 fan-out).",
    )
    p.add_argument(
        "--cassette-contexts", type=Path, default=None,
        help="FASTA of top promoter-enhancer cassette contexts for the intron stage (top-5 fan-out).",
    )
    p.add_argument(
        "--num-contexts", type=int, default=5,
        help="Top-N enhancer / cassette contexts to fan out over in the promoter and intron stages.",
    )
    p.add_argument(
        "--loci", default=None,
        help="Comma-separated subset of host loci to score (default: all gapdh,actb,eef1a1,ftl,hmga2).",
    )
    # Stage 1 (binder) knobs.
    p.add_argument(
        "--target-pdb", type=Path, default=None,
        help="EGFR ectodomain structure (chain A); defaults to examples/data/egfr_ectodomain_25_645.pdb.",
    )
    p.add_argument("--binder-length", type=int, default=80)
    p.add_argument("--rounds", type=int, default=15, help="Binder rejection-sampling rounds.")
    p.add_argument("--candidates-per-round", type=int, default=48)
    p.add_argument(
        "--no-af3-filter", action="store_true",
        help="Skip the final AlphaFold3 ipTM filter (use where AF3 weights are not provisioned).",
    )
    # Optimizer step/sample counts.
    p.add_argument("--enhancer-samples", type=int, default=500)
    p.add_argument("--promoter-steps", type=int, default=400)
    p.add_argument("--intron-steps", type=int, default=150)
    p.add_argument("--utr-steps", type=int, default=8000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _run_stage(
    name: str, args: argparse.Namespace, flanks: dict[str, dict[str, str]], champions: dict[str, list[str]]
) -> list[str] | None:
    """Build one stage and, unless --dry-run, run it and export results.

    ``champions`` carries the designed sequences from preceding stages so that, in an
    end-to-end ``--stage all`` run, each stage is scored in the context of the actual
    upstream designs. Returns this stage's champion sequences (or ``None`` on --dry-run).
    """
    logger.info("=== Stage: %s ===", name)
    if name == "binder":
        program, primary = build_binder_stage(args)  # standalone: EGFR binder feeds no downstream context
    else:
        builder = {"enhancer": build_enhancer_stage, "promoter": build_promoter_stage,
                   "intron": build_intron_stage, "utr": build_utr_stage}[name]
        program, primary = builder(args, flanks, champions)

    total_constraints = sum(len(opt.constraints) for opt in program.optimizers)
    logger.info("Built %s: %d optimizer stage(s), %d constraint(s).", name, len(program.optimizers), total_constraints)
    if args.dry_run:
        logger.info("--dry-run: skipping execution of stage '%s'.", name)
        return None

    program.run()
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    program.export(out, format="json")
    designs = [seq.sequence for seq in primary.result_sequences]
    for rank, seq in enumerate(designs):
        logger.info("%s result %d (%d nt/aa): %s...", name, rank, len(seq), seq[:60])
    logger.info("Exported %s results -> %s", name, out)
    return designs


def main() -> None:
    """Build (and optionally run) the requested cancer-circuit design stage(s).

    With ``--stage all`` the stages auto-chain: each design stage's champions are captured
    and fed into the context of the subsequent stages, reproducing a true end-to-end run.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    flanks = load_flanks(args.flanks_json or DEFAULT_FLANKS_JSON)
    stages = list(STAGE_BUILDERS) if args.stage == "all" else [args.stage]
    champions: dict[str, list[str]] = {}
    for name in stages:
        designs = _run_stage(name, args, flanks, champions)
        if designs:
            champions[name] = designs


if __name__ == "__main__":
    main()
