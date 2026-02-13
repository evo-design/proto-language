"""
Protein Hunter Example

Demonstrates the CyclingOptimizer for de novo protein design using the
Protein Hunter algorithm: iteratively cycling between structure prediction
and inverse folding to refine protein sequences.

This example designs a protein starting from an all-X (unknown) sequence,
using Boltz for structure prediction and ProteinMPNN for inverse folding.

Algorithm:
1. Predict 3D structure from current sequence (starts with all-X)
2. Use inverse folding (ProteinMPNN) to design sequences for predicted structure
3. Repeat for num_steps iterations

Usage:
    python protein_hunter.py --length 100 --cycles 5 --candidates 2 --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple

from proto_language.language.core import Construct, Program, Segment, Sequence
from proto_language.language.generator import (
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)
from proto_language.language.optimizer import CyclingOptimizer, CyclingOptimizerConfig
from proto_tools.tools.structure_prediction import (
    StructurePredictionComplex,
    predict_structures,
)


# =============================================================================
# Logging & Setup Helpers
# =============================================================================

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


def setup_output_dir(base_output_dir: str, length: int, cycles: int, tool: str) -> str:
    """Create timestamped output directory with parameter info."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        base_output_dir,
        f"protein_hunter_sweeps/{tool}_len{length}_cyc{cycles}/run_{run_timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protein Hunter: De novo design via hallucination cycles."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Length of the protein to design (default: 100)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of structure prediction -> inverse folding cycles (default: 5)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=2,
        help="Number of parallel candidate trajectories (default: 2)",
    )
    parser.add_argument(
        "--structure-tool",
        type=str,
        default="boltz2",
        choices=["boltz2", "chai1"],
        help="Structure prediction tool to use (default: boltz2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Base output directory (default: ./outputs)",
    )
    return parser.parse_args()


# =============================================================================
# Core Logic
# =============================================================================

def run_protein_hunter(
    design_length: int,
    num_cycles: int,
    num_candidates: int,
    structure_tool: str,
    output_dir: str,
) -> None:
    """Run the Protein Hunter pipeline."""

    # Setup output and logging.
    run_dir = setup_output_dir(output_dir, design_length, num_cycles, structure_tool)
    log_path = os.path.join(run_dir, "protein_hunter.log")
    log_capture = LogCapture(log_path)
    sys.stdout = log_capture

    try:
        print("=" * 60)
        print("Protein Hunter - De Novo Protein Design")
        print("=" * 60)
        print(f"  Design Length: {design_length}")
        print(f"  Num Cycles:    {num_cycles}")
        print(f"  Candidates:    {num_candidates}")
        print(f"  Structure Tool: {structure_tool}")
        print(f"  Output Dir:    {run_dir}")
        print("=" * 60)

        # Initialize with 'X' (unknown) residues for hallucination
        protein = Segment(
            sequence="X" * design_length,
            sequence_type="protein",
            label="designed_protein",
        )

        protein_construct = Construct([protein])

        # ProteinMPNN for inverse folding
        proteinmpnn_generator = ProteinMPNNGenerator(
            ProteinMPNNGeneratorConfig(
                temperature=0.1,
                excluded_amino_acids=["C"],
            )
        )

        # Define conditioning function.
        def structure_conditioning_fn(sequences: List[Sequence]) -> List:
            """
            Predict 3D structures and store PDBs in metadata for retrieval later.
            """
            complexes = [
                StructurePredictionComplex(chains=[seq.sequence])
                for seq in sequences
            ]

            structures = predict_structures(complexes, structure_tool, {}).structures

            # Store PDB in metadata.
            for seq, structure in zip(sequences, structures):
                seq._metadata["designed_structure_pdb"] = structure.structure_pdb

            return structures

        def step_logging(cycle: int, segments: Tuple[Segment, ...]) -> None:
            """Log progress for the best candidate after each cycle."""
            # Just grab the first candidate to show progress
            best_seq = segments[0].selected_sequences[0]
            print(f"Cycle {cycle}/{num_cycles}: {best_seq.sequence}... (len={len(best_seq)})")

        optimizer_config = CyclingOptimizerConfig(
            num_steps=num_cycles,
            num_candidates=num_candidates,
            conditioning_param_name="structure_inputs",
            verbose=True,
        )

        optimizer = CyclingOptimizer(
            target_segment=protein,
            constructs=[protein_construct],
            generators=[proteinmpnn_generator],
            constraints=[],
            config=optimizer_config,
            conditioning_fn=structure_conditioning_fn,
            custom_logging=step_logging,
        )

        program = Program(optimizers=[optimizer])
        program.run()

        print("\n" + "=" * 60)
        print("Saving Results")
        print("=" * 60)

        results_file = os.path.join(run_dir, "results_summary.txt")

        with open(results_file, "w") as f:
            f.write(f"# Protein Hunter Design Results\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")

            for i, seq in enumerate(protein.selected_sequences):
                candidate_id = i + 1

                # Save PDB if available.
                pdb_content = seq.metadata.get("designed_structure_pdb")
                if pdb_content:
                    pdb_filename = f"candidate_{candidate_id}.pdb"
                    pdb_path = os.path.join(run_dir, pdb_filename)
                    with open(pdb_path, "w") as pdb_f:
                        pdb_f.write(pdb_content)
                    print(f"Saved {pdb_filename}")
                else:
                    print(f"Warning: No structure found for Candidate {candidate_id}")

                info = f"\nCandidate {candidate_id}\n"
                info += f"Sequence: {seq.sequence}\n"
                info += f"Length:   {len(seq)}\n"
                info += "-" * 30 + "\n"

                print(info.strip())
                f.write(info)

        print(f"\nOptimization complete! All results saved to: {run_dir}")

    finally:
        # Restore stdout and close log.
        sys.stdout = log_capture.stdout
        log_capture.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    args = parse_args()
    run_protein_hunter(
        design_length=args.length,
        num_cycles=args.cycles,
        num_candidates=args.candidates,
        structure_tool=args.structure_tool,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
