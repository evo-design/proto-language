"""De novo binder design with the RFdiffusion3 + ProteinMPNN generator.

Demonstrates the idiomatic two-segment binder program:

- a length-only ``binder`` segment (the chain being designed), and
- a fixed ``target`` segment whose sequence is taken from the target structure.

The ``rfdiffusion-proteinmpnn-binder`` generator is assigned only to the ``binder``
segment and receives the target *structure* through its config (RFdiffusion3 needs the
coordinates to dock). The ``target`` segment exists so the scoring constraint can fold
the full target+binder complex — a structure-confidence constraint with
``inputs=[binder, target]``. A ``RejectionSamplingOptimizer`` generates a batch of binders
and keeps the lowest-energy ones.

The target therefore appears twice, deliberately: as coordinates in the generator config
(for docking) and as a fixed chain in the construct (for the constraint to fold). Both are
derived from the same ``Structure`` (one source → two artifacts), exactly as the
``examples/germinal`` and ``examples/bindcraft`` programs do.

This example calls RFdiffusion3, ProteinMPNN, and a structure predictor (Boltz2), so it
requires GPU access to actually run; it is illustrative and is not executed in CI.

Example:
    PYTHONPATH=$PWD/proto-tools:$PWD python examples/scripts/binder_design_rfdiffusion_proteinmpnn.py \
        --target-pdb examples/germinal/pdbs/pdl1.pdb \
        --target-chain A \
        --hotspots A37,A39,A41 \
        --binder-length 80 \
        --output-dir binder_design_outputs
"""

import argparse
import logging
from pathlib import Path

from proto_tools import ProteinMPNNSampleConfig, RFdiffusion3Config
from proto_tools.entities.structures import Structure

from proto_language import StructureBasedConstraintConfig, structure_iptm_constraint
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import (
    RFdiffusionProteinMPNNBinderGenerator,
    RFdiffusionProteinMPNNBinderGeneratorConfig,
)
from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig

logger = logging.getLogger(__name__)

DEFAULT_TARGET_PDB = Path(__file__).resolve().parents[1] / "germinal" / "pdbs" / "pdl1.pdb"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the binder-design example run."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-pdb", type=Path, default=DEFAULT_TARGET_PDB)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument(
        "--hotspots",
        default=None,
        help="Comma-separated target hotspots as '<chain><resnum>' (e.g. 'A37,A39').",
    )
    parser.add_argument("--binder-length", type=int, default=80)
    parser.add_argument("--num-samples", type=int, default=8, help="Total binders to generate and score.")
    parser.add_argument("--num-results", type=int, default=2, help="Top binders to keep (lowest energy).")
    parser.add_argument("--sequences-per-backbone", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("binder_design_outputs"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build and run a de-novo binder-design program against a fixed target."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # One source (the target Structure) → two artifacts: coordinates for the generator
    # to dock against, and the chain sequence for the fixed target segment.
    target_structure = Structure(structure=args.target_pdb.read_text())
    target_sequence = target_structure.get_chain_sequence(args.target_chain, remove_non_standard=True)
    hotspots = [h.strip() for h in args.hotspots.split(",") if h.strip()] if args.hotspots else None

    # Two segments: the binder is designed; the target is fixed (no generator).
    binder = Segment(length=args.binder_length, sequence_type="protein", label="binder")
    target = Segment(sequence=target_sequence, sequence_type="protein", label="target")
    construct = Construct([binder, target])

    # Generator: assigned only to the binder; target reaches it via config.
    generator = RFdiffusionProteinMPNNBinderGenerator(
        RFdiffusionProteinMPNNBinderGeneratorConfig(
            target_structure=target_structure,
            target_chains=[args.target_chain],
            hotspots=hotspots,  # also centers RFdiffusion3's generation origin on the epitope
            rfdiffusion3_config=RFdiffusion3Config(device=args.device),
            proteinmpnn_config=ProteinMPNNSampleConfig(
                num_sequences_per_structure=args.sequences_per_backbone,
                device=args.device,
            ),
        )
    )
    generator.assign(binder)

    # Constraint: fold the full target+binder complex and score interface confidence.
    # It targets BOTH segments, which is why the target must be a sibling segment.
    iptm_constraint = Constraint(
        inputs=[binder, target],
        function=structure_iptm_constraint,
        function_config=StructureBasedConstraintConfig(structure_tool="boltz2"),
        label="iptm",
        weight=1.0,
    )

    optimizer = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=[iptm_constraint],
        config=RejectionSamplingOptimizerConfig(
            num_samples=args.num_samples,
            num_results=args.num_results,
        ),
    )

    program = Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed)
    program.run()

    # Results: each kept binder + the stored target+binder complex.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta_lines: list[str] = []
    for rank, result in enumerate(binder.result_sequences):
        fasta_lines.append(f">binder_{rank}\n{result.sequence}")
        if result.structure is not None:
            result.structure.write_pdb(args.output_dir / f"binder_{rank}_complex.pdb")
        logger.info("Binder %d: %s", rank, result.sequence)
    (args.output_dir / "binders.fasta").write_text("\n".join(fasta_lines) + "\n")
    logger.info("Wrote %d binder(s) to %s", len(binder.result_sequences), args.output_dir)


if __name__ == "__main__":
    main()
