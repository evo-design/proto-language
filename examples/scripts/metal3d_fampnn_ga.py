"""dEVA-style Metal3D enzyme design with LigandMPNN packing and a genetic algorithm.

This mirrors ``examples/jsons/metal3d_fampnn_ga.json`` as a regular Proto program.
It requires GPU-backed LigandMPNN and Metal3D services to run.
"""

import argparse
import logging
from pathlib import Path

from proto_tools import InverseFoldingStructureInput, LigandMPNNSampleConfig, Metal3DPredictionConfig

from proto_language.constraint.protein_structure.metal3d_probability_constraint import (
    Metal3DProbabilityConfig,
    metal3d_probability_constraint,
)
from proto_language.constraint.protein_structure.structure_preparation import StructurePreparationConfig
from proto_language.constraint.sequence_scoring.mpnn_sequence_probability_constraint import (
    MPNNSequenceProbabilityConfig,
    mpnn_sequence_probability_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import (
    LigandMPNNGenerator,
    LigandMPNNGeneratorConfig,
    MPNNMutationGenerator,
    MPNNMutationGeneratorConfig,
)
from proto_language.optimizer import GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig

logger = logging.getLogger(__name__)

SCAFFOLD_URL = "https://raw.githubusercontent.com/gelnesr/dEVA/main/inputs/2VVB.pdb"
SCAFFOLD_SEQUENCE = (
    "HHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTY"
    "RLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADF"
    "TNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
)
# Literal dEVA behavior: its 2VVB config names X1-X7, but this PDB starts at X3,
# so dEVA's sequence model leaves chain X fully designable.


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Metal3D GA example."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=2)
    parser.add_argument("--offspring-per-generation", type=int, default=2)
    parser.add_argument("--num-results", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("metal3d_ligandmpnn_ga_outputs"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_program(args: argparse.Namespace) -> tuple[Program, Segment]:
    """Build the dEVA-style Metal3D program."""
    # Sequences.
    enzyme = Segment(sequence=SCAFFOLD_SEQUENCE, sequence_type="protein", label="2VVB chain X")
    construct = Construct([enzyme])

    # Generators.
    initialization_generator = LigandMPNNGenerator(
        LigandMPNNGeneratorConfig(
            structure_inputs=InverseFoldingStructureInput(structure=SCAFFOLD_URL),
            temperature=0.5,
            excluded_amino_acids=["C"],
            ligand_mpnn_use_side_chain_context=True,
            ligand_mpnn_cutoff_for_score=20.0,
            batch_size=args.batch_size,
            device=args.device,
            verbose=args.verbose,
        )
    )
    initialization_generator.assign(enzyme)

    mutation_generator = MPNNMutationGenerator(
        MPNNMutationGeneratorConfig(
            model="ligandmpnn",
            structure_inputs=InverseFoldingStructureInput(
                structure=SCAFFOLD_URL,
                chains_to_redesign=["X"],
            ),
            output_chain_id="X",
            num_mutations=4,
            excluded_amino_acids=["C"],
            replacement_strategy="sample",
            replacement_temperature=1.0,
            ligand_mpnn_use_side_chain_context=True,
            ligand_mpnn_cutoff_for_score=20.0,
            device=args.device,
            verbose=args.verbose,
        )
    )
    mutation_generator.assign(enzyme)

    # Constraints.
    mpnn_probability_constraint = Constraint(
        inputs=[enzyme],
        function=mpnn_sequence_probability_constraint,
        function_config=MPNNSequenceProbabilityConfig(
            model="ligandmpnn",
            structure_inputs=InverseFoldingStructureInput(
                structure=SCAFFOLD_URL,
                chains_to_redesign=["X"],
            ),
            output_chain_id="X",
            score_mode="probability_loss",
            ligand_mpnn_use_side_chain_context=True,
            ligand_mpnn_cutoff_for_score=20.0,
            device=args.device,
            verbose=args.verbose,
        ),
        label="LigandMPNN sequence probability",
        weight=1.0,
    )

    metal3d_constraint = Constraint(
        inputs=[enzyme],
        function=metal3d_probability_constraint,
        function_config=Metal3DProbabilityConfig(
            min_probability=0.2,
            structure_preparation=StructurePreparationConfig(
                mode="ligandmpnn_pack_from_scaffold",
                scaffold_structure=SCAFFOLD_URL,
                chain_ids=["X"],
                ligandmpnn_pack_config=LigandMPNNSampleConfig(
                    temperature=0.5,
                    ligand_mpnn_use_side_chain_context=True,
                    ligand_mpnn_cutoff_for_score=20.0,
                    batch_size=1,
                    device=args.device,
                    verbose=args.verbose,
                ),
            ),
            metal3d_config=Metal3DPredictionConfig(
                model_checkpoint="metal3d-cat",
                cluster_distance_threshold=7.0,
                max_sites=8,
                device=args.device,
                verbose=args.verbose,
            ),
        ),
        label="Metal3D-Cat metal-site probability",
        weight=1.0,
    )

    # Optimizer.
    optimizer = GeneticAlgorithmOptimizer(
        constructs=[construct],
        generators=[initialization_generator, mutation_generator],
        constraints=[mpnn_probability_constraint, metal3d_constraint],
        config=GeneticAlgorithmOptimizerConfig(
            num_generations=args.num_generations,
            num_results=args.num_results,
            population_size=args.population_size,
            offspring_per_generation=args.offspring_per_generation,
            elite_fraction=0.25,
            crossover_rate=1.0,
            crossover_strategy="two_point",
            parent_selection="tournament",
            tournament_size=2,
            replacement="elitist",
            survivor_selection="nsga2",
            refine_offspring_with_generators=False,
            initialize_with_mutation_generators=False,
            tracking_interval=1,
            track_proposals=False,
            verbose=args.verbose,
            seed=args.seed,
        ),
    )

    # Program.
    return Program(optimizers=[optimizer], num_results=args.num_results, seed=args.seed), enzyme


def write_results(enzyme: Segment, output_dir: Path, energy_scores: list[float]) -> None:
    """Write designed sequences and any attached structures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_lines: list[str] = []
    for rank, sequence in enumerate(enzyme.result_sequences):
        fasta_lines.append(f">metal3d_design_{rank}|energy={energy_scores[rank]:.6g}\n{sequence.sequence}")
        if sequence.structure is not None:
            sequence.structure.write_pdb(output_dir / f"metal3d_design_{rank}.pdb")
        logger.info("Design %d: energy=%s sequence=%s", rank, energy_scores[rank], sequence.sequence)
    (output_dir / "designs.fasta").write_text("\n".join(fasta_lines) + "\n")


def main() -> None:
    """Run the Metal3D GA example."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    program, enzyme = build_program(args)
    program.run()
    write_results(enzyme, args.output_dir, program.energy_scores)


if __name__ == "__main__":
    main()
