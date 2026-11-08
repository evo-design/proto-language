"""Ledidi edit design: a few edits that make a 200 bp sequence K562-specific under Malinois."""

import argparse
import random
import sys
from pathlib import Path

from proto_language.constraint import MalinoisActivityConfig, malinois_activity_constraint
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import PositionWeightGenerator, PositionWeightGeneratorConfig
from proto_language.optimizer import LedidiOptimizer, LedidiOptimizerConfig

CONSTRAINTS = [("K562", "max", "malinois_k562_max"), ("HepG2", "min", "malinois_hepg2_min"), ("SKNSH", "min", "malinois_sknsh_min")]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=None, help="Sequence to edit; a random 200 bp sequence when omitted.")
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--early-stopping-iter", type=int, default=100)
    parser.add_argument("--l", type=float, default=0.1, help="Weight on the number of edits.")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracking-interval", type=int, default=50)
    parser.add_argument("--export", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Edit the template and print each trajectory's best design."""
    args = parse_args()
    rng = random.Random(args.seed)  # noqa: S311 -- not cryptographic
    template = args.template or "".join(rng.choice("ACGT") for _ in range(200))
    segment = Segment(sequence=template, sequence_type="dna", label="enhancer_insert")
    construct = Construct([segment], label="malinois_k562_ledidi_edit")

    # Ledidi decodes its own samples; the generator only marks the segment being edited.
    generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
    generator.assign(segment)

    constraints = [
        Constraint(
            inputs=[segment],
            function=malinois_activity_constraint,
            function_config=MalinoisActivityConfig(
                cell_type=cell_type, direction=direction, seq_length=len(template), device=args.device
            ),
            label=label,
        )
        for cell_type, direction, label in CONSTRAINTS
    ]

    config = LedidiOptimizerConfig(
        n_repeats=args.n_repeats,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        early_stopping_iter=args.early_stopping_iter,
        l=args.l,
        tau=args.tau,
        lr=args.lr,
        tracking_interval=args.tracking_interval,
    )
    optimizer = LedidiOptimizer(
        target_segment=segment, constructs=[construct], generators=[generator], constraints=constraints, config=config
    )
    program = Program([optimizer], num_results=config.num_results, seed=args.seed)
    program.run()

    # Each trajectory returns its whole best batch; print the lowest-loss design of each.
    for repeat in range(args.n_repeats):
        block = segment.result_sequences[repeat * args.batch_size : (repeat + 1) * args.batch_size]
        best = min(block, key=lambda seq: seq.metadata["ledidi"]["total_loss"])
        meta = best.metadata["ledidi"]
        raw = {
            cell_type: best.metadata["constraints"][label]["data"]["malinois_raw_score"]
            for cell_type, _, label in CONSTRAINTS
        }
        edits = " ".join(f"{ref}{pos}{alt}" for pos, ref, alt in meta["edits"])
        sys.stdout.write(
            f"repeat {repeat}: {meta['num_edits']} edits, total loss {meta['total_loss']:.4f}, "
            f"K562 {raw['K562']:.3f} HepG2 {raw['HepG2']:.3f} SKNSH {raw['SKNSH']:.3f}\n  edits: {edits}\n"
        )

    if args.export is not None:
        program.export(args.export, format="json")


if __name__ == "__main__":
    main()
