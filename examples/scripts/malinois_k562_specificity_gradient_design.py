"""Fast SeqProp-style gradient design of K562-specific 200 bp regulatory DNA."""

import argparse
import sys
from pathlib import Path

from proto_language.language.constraint import MalinoisActivityConfig, malinois_activity_constraint
from proto_language.language.core import Constraint, Construct, Program, Segment
from proto_language.language.generator import PositionWeightGenerator, PositionWeightGeneratorConfig
from proto_language.language.optimizer import GradientOptimizer, GradientOptimizerConfig

TARGET_CELL_TYPE = "K562"
CONSTRAINT_LABELS = {
    "K562": "malinois_k562_max",
    "HepG2": "malinois_hepg2_min",
    "SKNSH": "malinois_sknsh_min",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--num-results", type=int, default=20)
    parser.add_argument("--seq-length", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--sigmoid-center", type=float, default=4.0)
    parser.add_argument("--sigmoid-scale", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--off-target-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracking-interval", type=int, default=10)
    parser.add_argument("--track-proposals", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--export", type=Path, default=None)
    return parser.parse_args()


def _malinois_constraint(
    *,
    segment: Segment,
    cell_type: str,
    direction: str,
    label: str,
    weight: float,
    args: argparse.Namespace,
) -> Constraint:
    """Create one public Malinois constraint using an explicit config object."""
    malinois_config = MalinoisActivityConfig(
        cell_type=cell_type,
        direction=direction,
        sigmoid_center=args.sigmoid_center,
        sigmoid_scale=args.sigmoid_scale,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    return Constraint(
        inputs=[segment],
        function=malinois_activity_constraint,
        function_config=malinois_config,
        label=label,
        weight=weight,
    )


def _metadata_float(sequence: object, label: str, key: str) -> float | None:
    """Read one numeric constraint metadata value from a result sequence."""
    metadata = getattr(sequence, "_constraints_metadata", {}).get(label, {}).get("data", {})
    value = metadata.get(key)
    return None if value is None else float(value)


def _format_float(value: float | None) -> str:
    """Format optional floats for terminal output."""
    return "nan" if value is None else f"{value:.6f}"


def main() -> None:
    """Run the example program and print optimized sequences."""
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.min_learning_rate <= 0.0:
        raise ValueError("--min-learning-rate must be positive")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate must be <= --learning-rate")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    # The all-A sequence is only a length/vocabulary placeholder; gumbel_logit_init below
    # creates the actual random A/C/G/T logit initialization used for optimization.
    segment = Segment(sequence="A" * args.seq_length, sequence_type="dna", label="enhancer_insert")
    construct = Construct([segment], label="malinois_k562_specific_design")

    generator = PositionWeightGenerator(PositionWeightGeneratorConfig(sampling_mode="argmax"))
    generator.assign(segment)

    k562_constraint = _malinois_constraint(
        segment=segment,
        cell_type=TARGET_CELL_TYPE,
        direction="max",
        label=CONSTRAINT_LABELS[TARGET_CELL_TYPE],
        weight=args.target_weight,
        args=args,
    )
    hepg2_constraint = _malinois_constraint(
        segment=segment,
        cell_type="HepG2",
        direction="min",
        label=CONSTRAINT_LABELS["HepG2"],
        weight=args.off_target_weight,
        args=args,
    )
    sknsh_constraint = _malinois_constraint(
        segment=segment,
        cell_type="SKNSH",
        direction="min",
        label=CONSTRAINT_LABELS["SKNSH"],
        weight=args.off_target_weight,
        args=args,
    )

    lr_min_scale = args.min_learning_rate / args.learning_rate
    optimizer = GradientOptimizer(
        target_segment=segment,
        constructs=[construct],
        generators=[generator],
        constraints=[k562_constraint, hepg2_constraint, sknsh_constraint],
        config=GradientOptimizerConfig(
            num_results=args.num_results,
            num_steps=args.num_steps,
            lr=args.learning_rate,
            soft_start=1.0,
            soft_end=1.0,
            hard_start=1.0,
            hard_end=1.0,
            temperature_start=1.0,
            temperature_end=lr_min_scale,
            softmax_schedule="constant",
            lr_schedule="cosine",
            scale_lr_by_temperature=True,
            ml_optimizer="adam",
            merger="weighted_sum",
            norm_alignment="none",
            normalize_gradients=False,
            gumbel_logit_init=True,
            gumbel_init_alpha=1.0,
            tracking_interval=args.tracking_interval,
            track_proposals=args.track_proposals,
            verbose=args.verbose,
            save_best=True,
        ),
    )

    program = Program(
        [optimizer],
        num_results=args.num_results,
        seed=args.seed,
        verbose=args.verbose,
    )
    program.run()

    sys.stdout.write(
        f"Top {len(program.energy_scores)} designs by Malinois K562 specificity objective "
        f"(K562 max, HepG2/SKNSH min)\n"
    )
    for rank, (sequence, energy) in enumerate(zip(segment.result_sequences, program.energy_scores, strict=True), 1):
        raw_scores = {
            cell_type: _metadata_float(sequence, label, "malinois_raw_score")
            for cell_type, label in CONSTRAINT_LABELS.items()
        }
        losses = {
            cell_type: _metadata_float(sequence, label, "malinois_activity_score")
            for cell_type, label in CONSTRAINT_LABELS.items()
        }
        sys.stdout.write(
            f"{rank:>2}. energy={energy:.6f} "
            f"K562_raw={_format_float(raw_scores['K562'])} HepG2_raw={_format_float(raw_scores['HepG2'])} "
            f"SKNSH_raw={_format_float(raw_scores['SKNSH'])} "
            f"K562_loss={_format_float(losses['K562'])} HepG2_loss={_format_float(losses['HepG2'])} "
            f"SKNSH_loss={_format_float(losses['SKNSH'])} sequence={sequence.sequence}\n"
        )

    if args.export is not None:
        program.export(args.export, format="json")


if __name__ == "__main__":
    main()
