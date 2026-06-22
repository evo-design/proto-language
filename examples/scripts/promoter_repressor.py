"""ProtoPromoter: a two-stage repressor-gated sigma70 promoter design program.

Designs E. coli sigma70 promoters carrying an inverted-repeat repressor operator
positioned to occlude the core promoter, so the promoter can be switched off by a
cognate repressor.

The design runs as a single Proto ``Program`` with two optimizer stages operating
on the same 100 bp promoter segment (state flows between stages by construct
identity):

    Stage 1 - generate + reject
        Evo 2 (7B) autoregressively generates 100 nt of new promoter sequence
        conditioned on (but not retaining) a native RegulonDB sigma70 promoter
        prompt (80 nt) (temperature 0.9, top-k 4, top-p 1.0), and a
        rejection-sampling filter keeps the lowest-energy candidates under a
        composite energy = 5 x Promoter-Calculator(plus-strand dGtotal)
        + 1 x sigma70 -35/-10 box PWM + 1 x MEME/FIMO scan against E. coli TF
        motifs (penalizing cryptic regulatory sites).

    Stage 2 - MCMC refine + operator selection
        A Metropolis-Hastings sampler with single-nucleotide uniform mutations and
        a geometric (exponential) annealing schedule refines the survivors against
        the same composite energy, with an added operator-site term that rewards an
        inverted-repeat operator (two >= 7 bp half-sites, gap <= 1 bp, <= 1 mismatch
        between half-site 1 and the reverse complement of half-site 2) overlapping
        the -35 box, -10 box, or TSS by >= 3 bp. By default this operator term is a
        soft bias, so final designs are those with the lowest energy; pass
        --operator-filter to instead require a confirmed occluding operator.

Data inputs (override via CLI; the program still builds with documented fallbacks):
    --prompt-fasta   native sigma70 promoter prompts for Evo 2 (defaults to
                     examples/data/sigma70_promoters.fasta; falls back to a labeled
                     consensus scaffold prompt with a warning).
    --tf-motifs      MEME-format file of E. coli TF motif PWMs for the FIMO scan
                     (defaults to examples/data/ecoli_tf_motifs.meme; the FIMO term
                     is skipped with a warning if the file is absent).

Evo 2 and the Promoter Calculator / FIMO run heavy / external tools, so this script
is illustrative and is not executed in CI; use ``--dry-run`` to build the program
and validate its constraints without running it.

Example:
    # Build the program and validate constraints without running:
    PYTHONPATH=$PWD/proto-tools:$PWD python examples/scripts/promoter_repressor.py --dry-run

    # Run end-to-end on GPU:
    PYTHONPATH=$PWD/proto-tools:$PWD python examples/scripts/promoter_repressor.py \
        --prompt-fasta examples/data/sigma70_promoters.fasta \
        --tf-motifs examples/data/ecoli_tf_motifs.meme --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from proto_tools.transforms.masking import MaskingStrategy

from proto_language.constraint import (
    OperatorSiteConfig,
    operator_site_constraint,
    promoter_strength_constraint,
    seq_motif_constraint,
    sigma70_promoter_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)

logger = logging.getLogger(__name__)

# Default data ships in examples/data/:
#   sigma70_promoters.fasta - 1,997 native E. coli sigma70 promoters (RegulonDB pmSequence column).
#   ecoli_tf_motifs.meme     - 97 E. coli TF motif PWMs (MEME format) for the cryptic-site FIMO scan.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PROMPT_FASTA = _DATA_DIR / "sigma70_promoters.fasta"
DEFAULT_TF_MOTIFS = _DATA_DIR / "ecoli_tf_motifs.meme"

# Composite-energy weights (Promoter Calculator weighted 5x the sigma70 and FIMO terms).
PROMOTER_CALC_WEIGHT = 5.0
SIGMA70_WEIGHT = 1.0
MOTIF_WEIGHT = 1.0

# Documented fallback prompt when no native promoter FASTA is supplied. This is a labeled
# sigma70 consensus scaffold (-35 TTGACA, 17 bp spacer, -10 TATAAT), NOT a claimed native
# RegulonDB sequence; supply real promoters via --prompt-fasta for a realistic run.
CONSENSUS_SCAFFOLD_PROMPT = (
    "GCGCAACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTGACACTTTATGCTTCCGGCTCGTATAATG"
)


def _read_fasta(path: Path | None) -> list[str]:
    """Read sequences from a FASTA (or one-per-line) file; return [] if missing/empty."""
    if path is None or not Path(path).exists():
        return []
    text = Path(path).read_text()
    seqs: list[str] = []
    block: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if block:
                seqs.append("".join(block).upper())
                block = []
        elif line.strip():
            block.append(line.strip())
    if block:
        seqs.append("".join(block).upper())
    return seqs


def _select_prompt(args: argparse.Namespace) -> str:
    """Pick one native sigma70 promoter prompt (by --seed), truncated to --prompt-bp.

    Evo 2 conditions on a single equal-length prompt per run; varying --seed draws a
    different native promoter from the supplied set. Falls back to a labeled consensus
    scaffold (with a warning) when no prompt FASTA is available.
    """
    prompts = _read_fasta(args.prompt_fasta or DEFAULT_PROMPT_FASTA)
    if not prompts:
        logger.warning(
            "No sigma70 promoter prompts at %s; falling back to a labeled consensus scaffold. "
            "Supply real RegulonDB promoters via --prompt-fasta for a realistic run.",
            args.prompt_fasta or DEFAULT_PROMPT_FASTA,
        )
        prompts = [CONSENSUS_SCAFFOLD_PROMPT]
    chosen = prompts[int(args.seed) % len(prompts)]
    return chosen[: int(args.prompt_bp)]


def _energy_constraints(segment: Segment, args: argparse.Namespace, tag: str) -> list[Constraint]:
    """Fresh composite-energy constraints (5x Promoter-Calculator + sigma70 + FIMO).

    A new list is built per optimizer stage because constraints cannot be reused across
    stages. The FIMO term is included only when a TF-motif file is available.
    """
    constraints = [
        Constraint(
            inputs=[segment],
            function=promoter_strength_constraint,
            function_config={"scoring_type": "dG", "threads": args.threads},
            weight=PROMOTER_CALC_WEIGHT,
            label=f"{tag}_promoter_calc",
        ),
        Constraint(
            inputs=[segment],
            function=sigma70_promoter_constraint,
            function_config={},
            weight=SIGMA70_WEIGHT,
            label=f"{tag}_sigma70",
        ),
    ]

    motifs_path = args.tf_motifs or DEFAULT_TF_MOTIFS
    if Path(motifs_path).exists():
        constraints.append(
            Constraint(
                inputs=[segment],
                function=seq_motif_constraint,
                function_config={
                    "motifs_path": str(motifs_path),
                    "not_wanted": ["all"],  # penalize ANY cryptic TF site
                    "scale": MOTIF_WEIGHT,
                },
                weight=MOTIF_WEIGHT,
                label=f"{tag}_fimo",
            )
        )
    else:
        logger.warning(
            "No TF-motif file at %s; skipping the MEME/FIMO term. Supply --tf-motifs "
            "(E. coli TF motif PWMs) to include the cryptic-site penalty.",
            motifs_path,
        )
    return constraints


def _operator_constraint(segment: Segment, args: argparse.Namespace) -> Constraint:
    """Operator-site term for stage 2.

    By default the operator score is a weighted optimization term that pulls the sampler
    toward forming an occluding inverted-repeat operator. With --operator-filter it is a
    hard presence filter (threshold 0.5) that rejects any promoter lacking an operator.
    """
    config = OperatorSiteConfig(
        min_half_site=args.operator_half_site,
        max_gap=args.operator_gap,
        max_mismatch=args.operator_mismatch,
        min_overlap=args.operator_overlap,
    )
    if args.operator_filter:
        return Constraint(
            inputs=[segment],
            function=operator_site_constraint,
            function_config=config,
            threshold=0.5,
            label="operator_filter",
        )
    return Constraint(
        inputs=[segment],
        function=operator_site_constraint,
        function_config=config,
        weight=args.operator_weight,
        label="operator_site",
    )


def build_program(args: argparse.Namespace) -> tuple[Program, Segment]:
    """Build the two-stage ProtoPromoter design program over one 100 bp promoter segment.

    Args:
        args (argparse.Namespace): Parsed CLI options.

    Returns:
        tuple[Program, Segment]: The two-stage program and the designed promoter segment.
    """
    from proto_language.generator import Evo2Generator, Evo2GeneratorConfig

    promoter = Segment(length=args.promoter_length, sequence_type="dna", label="sigma70 promoter")
    construct = Construct([promoter])

    # --- Stage 1: Evo 2 generation + rejection-sampling energy filter ---
    prompt = _select_prompt(args)
    logger.info("Evo2 prompt (%d nt, seed %d): %s", len(prompt), args.seed, prompt)
    evo2_generator = Evo2Generator(
        Evo2GeneratorConfig(
            prompts=[prompt],
            model_checkpoint="evo2_7b",
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            stop_at_eos=False,
        )
    )
    evo2_generator.assign(promoter)
    stage1 = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[evo2_generator],
        constraints=_energy_constraints(promoter, args, "gen"),
        config=RejectionSamplingOptimizerConfig(
            num_samples=args.gen_samples,
            num_results=args.num_results,
            energy_threshold=args.energy_threshold,
        ),
    )

    # --- Stage 2: Metropolis-Hastings MCMC refinement + operator-site selection ---
    mut_generator = RandomNucleotideGenerator(
        RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1))
    )
    mut_generator.assign(promoter)
    stage2_constraints = _energy_constraints(promoter, args, "mcmc")
    stage2_constraints.append(_operator_constraint(promoter, args))
    stage2 = MCMCOptimizer(
        constructs=[construct],
        generators=[mut_generator],
        constraints=stage2_constraints,
        config=MCMCOptimizerConfig(
            num_results=args.num_results,
            num_steps=args.mcmc_steps,
            max_temperature=args.t_max,
            min_temperature=args.t_min,
            temperature_schedule="exponential",  # geometric annealing
        ),
    )

    program = Program(optimizers=[stage1, stage2], num_results=args.num_results, seed=args.seed)
    return program, promoter


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the ProtoPromoter example."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Build the program (validate constraints) without running.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-results", type=int, default=93, help="Designs retained (lowest energy + operator).")
    p.add_argument("--output-dir", type=Path, default=Path("promoter_repressor_outputs"))
    # Data inputs.
    p.add_argument("--prompt-fasta", type=Path, default=None, help="Native sigma70 promoter prompts (FASTA) for Evo2.")
    p.add_argument("--prompt-bp", type=int, default=80, help="Native-promoter prompt length (nt) fed to Evo2.")
    p.add_argument("--tf-motifs", type=Path, default=None, help="MEME-format E. coli TF motif PWMs for the FIMO scan.")
    p.add_argument("--threads", type=int, default=8, help="Promoter-Calculator CPU threads.")
    # Promoter geometry.
    p.add_argument("--promoter-length", type=int, default=100, help="Designed promoter length (nt).")
    # Stage 1: Evo2 + rejection sampling.
    p.add_argument("--temperature", type=float, default=0.9, help="Evo2 sampling temperature.")
    p.add_argument("--top-k", type=int, default=4, help="Evo2 top-k sampling.")
    p.add_argument("--top-p", type=float, default=1.0, help="Evo2 top-p (nucleus) sampling.")
    p.add_argument("--gen-samples", type=int, default=10000, help="Evo2 candidates drawn before rejection.")
    p.add_argument(
        "--energy-threshold", type=float, default=None,
        help="Optional rejection-sampling early-stop energy cutoff (keep candidates below it).",
    )
    # Stage 2: MCMC.
    p.add_argument("--mcmc-steps", type=int, default=2000, help="Metropolis-Hastings steps.")
    p.add_argument("--t-max", type=float, default=1.0, help="Initial (max) MCMC temperature.")
    p.add_argument("--t-min", type=float, default=1e-3, help="Final (min) MCMC temperature.")
    # Operator-site selection.
    p.add_argument("--operator-weight", type=float, default=2.0, help="Weight of the operator-site optimization term.")
    p.add_argument(
        "--operator-filter", action="store_true",
        help="Use the operator site as a hard presence filter (threshold 0.5) instead of a weighted term.",
    )
    p.add_argument("--operator-half-site", type=int, default=7, help="Minimum operator half-site length (bp).")
    p.add_argument("--operator-gap", type=int, default=1, help="Maximum gap between operator half-sites (bp).")
    p.add_argument("--operator-mismatch", type=int, default=1, help="Max half-site mismatches vs the dyad.")
    p.add_argument("--operator-overlap", type=int, default=3, help="Min operator overlap with -35/-10/TSS (bp).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    """Build (and optionally run) the two-stage ProtoPromoter design program."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    program, promoter = build_program(args)
    total_constraints = sum(len(opt.constraints) for opt in program.optimizers)
    logger.info(
        "Built ProtoPromoter program: %d optimizer stage(s), %d constraint(s).",
        len(program.optimizers),
        total_constraints,
    )
    if args.dry_run:
        logger.info("--dry-run: skipping execution.")
        return

    program.run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    program.export(args.output_dir, format="json")
    designs = [seq.sequence for seq in promoter.result_sequences]
    for rank, seq in enumerate(designs):
        logger.info("promoter result %d (%d nt): %s", rank, len(seq), seq)
    logger.info("Exported %d promoter designs -> %s", len(designs), args.output_dir)


if __name__ == "__main__":
    main()
