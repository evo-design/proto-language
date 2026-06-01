"""Minimal BindCraft-style hallucination-only binder design.

This is a re-implementation expressed in proto-language's program formulation,
not a byte-for-byte port of BindCraft: results are not 1-to-1 identical to
upstream.

This script intentionally does not implement the full BindCraft pipeline. It
only runs the pre-MPNN hallucination stages with AF2 multimer objectives:

1. logit gradient optimization
2. softmax gradient refinement
3. hard straight-through refinement
4. optional pLDDT-weighted semigreedy refinement

For readability, the logit phase is a single 0→1 soft ramp like BindCraft's
2stage/3stage modes, not the full pipeline's default logit_a/logit_b split.
The script also runs configured stages unconditionally; it does not apply the
full pipeline's pLDDT gates between logit, softmax, and hard stages.

Example:
    PYTHONPATH=$PWD/proto-tools:$PWD python examples/bindcraft/run_bindcraft_hallucination.py \
        --target-pdb examples/germinal/pdbs/pdl1.pdb \
        --target-chain A \
        --binder-length 65 \
        --output-dir bindcraft_hallucination_outputs
"""

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

from proto_tools.entities.structures import Structure

from proto_language import (
    AlphaFold2MultimerStructureConfig,
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
    StructureBasedConstraintConfig,
    structure_contact_constraint,
    structure_helix_constraint,
    structure_interface_contact_constraint,
    structure_ipae_constraint,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_radius_gyration_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import PositionWeightGenerator, PositionWeightGeneratorConfig
from proto_language.optimizer import (
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)
from proto_language.utils.ml_optimizers import AdamConfig

logger = logging.getLogger(__name__)

DEFAULT_TARGET_PDB = Path(__file__).resolve().parents[2] / "examples" / "germinal" / "pdbs" / "pdl1.pdb"

AF2_LOSS_FUNCTIONS = {
    "plddt": structure_plddt_constraint,
    "pae": structure_pae_constraint,
    "i_pae": structure_ipae_constraint,
    "con": structure_contact_constraint,
    "i_con": structure_interface_contact_constraint,
    "rg": structure_radius_gyration_constraint,
    "i_ptm": structure_iptm_constraint,
    "helix": structure_helix_constraint,
}
AF2_LOSS_WEIGHTS = {
    "plddt": 0.1,
    "pae": 0.4,
    "i_pae": 0.1,
    "con": 1.0,
    "i_con": 1.0,
    "rg": 0.3,
    "i_ptm": 0.05,
    "helix": -0.3,
}
GRADIENT_DEFAULTS: dict[str, object] = {
    "lr": 0.1,
    "merger": "weighted_sum",
    "normalize_gradients": True,
    "normalize_mode": "unit",
    "ml_optimizer": "adam",
    "adam_config": AdamConfig(),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the hallucination smoke/example run."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-pdb", type=Path, default=DEFAULT_TARGET_PDB)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--target-hotspot", default=None)
    parser.add_argument("--binder-length", type=int, default=65)
    parser.add_argument("--omit-aas", default="C")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-recycles", type=int, default=1)
    parser.add_argument("--logit-steps", type=int, default=75)
    parser.add_argument("--softmax-steps", type=int, default=45)
    parser.add_argument("--hard-steps", type=int, default=5)
    parser.add_argument("--semigreedy-steps", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("bindcraft_hallucination_outputs"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def make_af2_constraints(
    binder: Segment,
    target: Segment,
    structure_config: StructureBasedConstraintConfig,
) -> list[Constraint]:
    """Create the AF2 multimer objectives used by every hallucination stage."""
    constraints: list[Constraint] = []
    for loss_key, weight in AF2_LOSS_WEIGHTS.items():
        if weight == 0.0:
            continue
        constraints.append(
            Constraint(
                inputs=[binder, target],
                function=AF2_LOSS_FUNCTIONS[loss_key],
                function_config=copy.deepcopy(structure_config),
                label=f"af2_{loss_key}",
                weight=weight,
            )
        )
    return constraints


def jsonable(value: Any) -> Any:
    """Convert nested metadata into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


def main() -> None:
    """Build and run a minimal hallucination-only BindCraft-style program."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    target_pdb_text = args.target_pdb.read_text()
    target_structure = Structure(structure=target_pdb_text)
    target_sequence = target_structure.get_chain_sequence(args.target_chain, remove_non_standard=True)

    # Sequences
    binder = Segment(length=args.binder_length, sequence_type="protein", label="binder")
    target = Segment(sequence=target_sequence, sequence_type="protein", label="target")
    construct = Construct([binder, target])

    # Generators
    logit_generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
    logit_generator.assign(binder)
    softmax_generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
    softmax_generator.assign(binder)
    hard_generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
    hard_generator.assign(binder)
    semigreedy_generator = SemigreedyMutationGenerator(
        SemigreedyMutationGeneratorConfig(
            position_weighting="plddt",
            exclude_current=True,
            clear_logits=True,
        )
    )
    semigreedy_generator.assign(binder)

    # Constraints
    af2_config = AlphaFold2MultimerStructureConfig(
        target_pdb=target_pdb_text,
        target_chains=[args.target_chain],
        binder_chain=None,  # de novo: hallucinate the binder, no template
        target_hotspot=args.target_hotspot,
        omit_aas=[aa.strip().upper() for aa in args.omit_aas.split(",") if aa.strip()] or None,
        num_recycles=args.num_recycles,
        sample_models=True,
        use_multimer=True,
        rm_target_seq=False,
        rm_target_sc=False,
        rm_template_ic=False,
        seed=args.seed,
        intra_contact_num=2,
        intra_contact_cutoff=14.0,
        inter_contact_num=2,
        inter_contact_cutoff=20.0,
        backend="base",
    )
    structure_config = StructureBasedConstraintConfig(
        structure_tool="alphafold2_multimer",
        alphafold2_multimer_config=af2_config,
    )
    logit_constraints = make_af2_constraints(binder, target, structure_config)
    softmax_constraints = make_af2_constraints(binder, target, structure_config)
    hard_constraints = make_af2_constraints(binder, target, structure_config)
    semigreedy_constraints = make_af2_constraints(binder, target, structure_config)

    # Optimizers
    optimizers = []
    if args.logit_steps > 0:
        optimizers.append(
            GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[logit_generator],
                constraints=logit_constraints,
                config=GradientOptimizerConfig(
                    **copy.deepcopy(GRADIENT_DEFAULTS),
                    num_steps=args.logit_steps,
                    soft_start=0.0,
                    soft_end=1.0,
                    gumbel_logit_init=True,
                ),
            )
        )
    if args.softmax_steps > 0:
        optimizers.append(
            GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[softmax_generator],
                constraints=softmax_constraints,
                config=GradientOptimizerConfig(
                    **copy.deepcopy(GRADIENT_DEFAULTS),
                    num_steps=args.softmax_steps,
                    soft_start=1.0,
                    soft_end=1.0,
                    temperature_start=1.0,
                    temperature_end=0.01,
                    softmax_schedule="quadratic",
                    lr_schedule="quadratic",  # required for scale_lr_by_temperature to decay the LR
                    scale_lr_by_temperature=True,
                ),
            )
        )
    if args.hard_steps > 0:
        optimizers.append(
            GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[hard_generator],
                constraints=hard_constraints,
                config=GradientOptimizerConfig(
                    **copy.deepcopy(GRADIENT_DEFAULTS),
                    num_steps=args.hard_steps,
                    soft_start=1.0,
                    soft_end=1.0,
                    hard_start=1.0,
                    hard_end=1.0,
                    temperature_start=0.01,
                    temperature_end=0.01,
                    scale_lr_by_temperature=True,
                ),
            )
        )
    if args.semigreedy_steps > 0:
        optimizers.append(
            MCMCOptimizer(
                constructs=[construct],
                generators=[semigreedy_generator],
                constraints=semigreedy_constraints,
                config=MCMCOptimizerConfig(
                    num_steps=args.semigreedy_steps,
                    proposals_per_result=max(1, math.ceil(args.binder_length * 0.01)),
                    max_temperature=1e-6,
                    min_temperature=1e-7,
                ),
            )
        )
    if not optimizers:
        raise ValueError("At least one hallucination stage must have >0 steps.")

    # Program
    program = Program(optimizers=optimizers, num_results=1, seed=args.seed)
    program.run()

    # Results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binder_result = binder.result_sequences[0]
    target_result = target.result_sequences[0]
    (args.output_dir / "hallucinated_binder.fasta").write_text(
        f">binder\n{binder_result.sequence}\n>target_{args.target_chain}\n{target_sequence}\n"
    )
    if binder_result.structure is not None:
        binder_result.structure.write_pdb(args.output_dir / "hallucinated_binder.pdb")
    if binder_result.structure is not None and target_result.structure is not None:
        complex_structure = Structure.concat([binder_result.structure, target_result.structure])
        complex_structure.write_pdb(args.output_dir / "hallucinated_complex.pdb")

    summary = {
        "binder_sequence": binder_result.sequence,
        "target_chain": args.target_chain,
        "target_sequence": target_sequence,
        "energy_scores": program.energy_scores,
        "binder_constraints": binder_result.metadata.get("constraints", {}),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))

    logger.info("Designed binder: %s", binder_result.sequence)
    logger.info("Wrote hallucination outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
