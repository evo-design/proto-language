"""
Program symmetric proteins with configurable parameters.

Usage:
    python program_symmetric_proteins.py --monomer-length 100 --n-symmetric-units 3 --n-steps 10000 --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Tuple

from proto_language.language.constraint import (
    protein_globularity_constraint,
    protein_symmetry_ring_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.storage import get_file_content


def parse_args():
    parser = argparse.ArgumentParser(
        description="Design symmetric proteins using MCMC optimization"
    )
    parser.add_argument(
        "--monomer-length",
        type=int,
        default=100,
        help="Length of each monomer unit (default: 100)",
    )
    parser.add_argument(
        "--n-symmetric-units",
        type=int,
        default=3,
        help="Number of symmetric units in the assembly (default: 3)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10000,
        help="Number of MCMC optimization steps (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Base output directory (default: ./outputs)",
    )
    return parser.parse_args()


def setup_output_dir(base_output_dir: str, monomer_length: int, n_symmetric_units: int, n_steps: int) -> str:
    """Create timestamped output directory with parameter info."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(
        base_output_dir,
        f"sym{n_symmetric_units}_len{monomer_length}_steps{n_steps}/run_{run_timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class LogCapture:
    """Capture logging output to both console and file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, message: str):
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def run_optimization(
    monomer_length: int,
    n_symmetric_units: int,
    n_steps: int,
    output_dir: str,
) -> None:
    """Run the symmetric protein optimization."""

    # Setup output directory
    run_dir = setup_output_dir(output_dir, monomer_length, n_symmetric_units, n_steps)
    print(f"Output directory: {run_dir}")

    # Setup logging
    log_path = os.path.join(run_dir, "optimization.log")
    log_capture = LogCapture(log_path)
    sys.stdout = log_capture

    try:
        # Log parameters
        print(f"=" * 60)
        print(f"Symmetric Protein Design")
        print(f"=" * 60)
        print(f"Parameters:")
        print(f"  Monomer length: {monomer_length}")
        print(f"  N symmetric units: {n_symmetric_units}")
        print(f"  N steps: {n_steps}")
        print(f"  Output directory: {run_dir}")
        print(f"=" * 60)

        #######################
        ## Segments ##
        #######################

        protomer = Segment(
            length=monomer_length,
            sequence_type="protein",
        )

        ################
        ## Constructs ##
        ################

        protomer_construct = Construct([protomer])

        ################
        ## Generators ##
        ################

        uniform_gen_config = UniformMutationGeneratorConfig()
        uniform_gen = UniformMutationGenerator(uniform_gen_config)
        uniform_gen.assign(protomer)

        #################
        ## Constraints ##
        #################

        esmfold_plddt = Constraint(
            inputs=[protomer for _ in range(n_symmetric_units)],
            function=structure_plddt_constraint,
            function_config={"structure_tool": "esmfold"},
            weight=1,
        )

        esmfold_ptm = Constraint(
            inputs=[protomer for _ in range(n_symmetric_units)],
            function=structure_ptm_constraint,
            function_config={"structure_tool": "esmfold"},
            weight=1,
        )

        symmetry = Constraint(
            inputs=[protomer],
            function=protein_symmetry_ring_constraint,
            function_config={
                "n_replications": n_symmetric_units,
                "all_to_all_protomer_symmetry": True,
            },
            weight=1,
        )

        globularity = Constraint(
            inputs=[protomer],
            function=protein_globularity_constraint,
            function_config={"n_replications": n_symmetric_units},
            weight=5,
        )

        #################
        ## Custom Logging ##
        #################

        def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
            output_sequence: Sequence = outputs[0].selected_sequences[0]
            constraints = output_sequence._metadata["constraints"]

            # Get pLDDT from structure_plddt_constraint
            plddt = constraints["structure_plddt_constraint"]["data"]["avg_plddt"]

            # Get pTM from structure_ptm_constraint
            ptm = constraints["structure_ptm_constraint"]["data"]["ptm"]

            # Get esmfolded_sequence from symmetry constraint
            folded_sequence = constraints["protein_symmetry_ring_constraint"]["data"]["esmfolded_sequence"]

            print(
                f"Iteration {step} | \n"
                f"\tsequence (monomer): {output_sequence._sequence}, \n"
                f"\tsequence (duplicated): {folded_sequence}, \n"
                f"\tpLDDT: {plddt}, \n"
                f"\tpTM: {ptm}"
            )

        #############
        ## Program ##
        #############

        mcmc_optimizer_config = MCMCOptimizerConfig(
            num_steps=n_steps,
            max_temperature=1.,
            min_temperature=0.0001,
            verbose=True,
        )

        # Create optimizer
        optimizer = MCMCOptimizer(
            constructs=[protomer_construct],
            generators=[uniform_gen],
            constraints=[
                esmfold_plddt,
                esmfold_ptm,
                symmetry,
                globularity,
            ],
            config=mcmc_optimizer_config,
            custom_logging=custom_logging,
        )

        # Create program with optimizer
        program = Program(optimizers=[optimizer], num_results=1)

        program.run()

        #################
        ## Save Outputs ##
        #################

        # Get sequence from the protomer segment (where constraint metadata is stored)
        protomer_sequence: Sequence = protomer.selected_sequences[0]
        constraints = protomer_sequence._metadata["constraints"]

        # Save PDB (stored as file reference, need to retrieve content)
        pdb_ref = constraints["protein_symmetry_ring_constraint"]["data"]["pdb_output"]
        pdb_content = get_file_content(pdb_ref)
        pdb_path = os.path.join(run_dir, "design.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_content)
        print(f"Saved PDB to: {pdb_path}")

        # Extract final metrics
        final_plddt = constraints["structure_plddt_constraint"]["data"]["avg_plddt"]
        final_ptm = constraints["structure_ptm_constraint"]["data"]["ptm"]
        folded_sequence = constraints["protein_symmetry_ring_constraint"]["data"]["esmfolded_sequence"]

        # Save final sequence and metrics
        seq_path = os.path.join(run_dir, "final_sequence.txt")
        with open(seq_path, "w") as f:
            f.write(f"# Symmetric Protein Design Result\n")
            f.write(f"# Monomer length: {monomer_length}\n")
            f.write(f"# N symmetric units: {n_symmetric_units}\n")
            f.write(f"# N steps: {n_steps}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"\n")
            f.write(f"Monomer sequence:\n{protomer_sequence._sequence}\n")
            f.write(f"\nFolded sequence (duplicated):\n{folded_sequence}\n")
            f.write(f"\nFinal pLDDT: {final_plddt}\n")
            f.write(f"Final pTM: {final_ptm}\n")
        print(f"Saved sequence to: {seq_path}")

        print(f"\n{'=' * 60}")
        print(f"Optimization complete!")
        print(f"Results saved to: {run_dir}")
        print(f"{'=' * 60}")

    finally:
        # Restore stdout and close log
        sys.stdout = log_capture.stdout
        log_capture.close()
        print(f"Log saved to: {log_path}")


def main():
    args = parse_args()
    run_optimization(
        monomer_length=args.monomer_length,
        n_symmetric_units=args.n_symmetric_units,
        n_steps=args.n_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
