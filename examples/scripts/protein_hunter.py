"""
Protein Hunter Example

Demonstrates the CyclicalOptimizer for de novo protein design using the
Protein Hunter algorithm: iteratively cycling between structure prediction
and inverse folding to refine protein sequences.

This example designs a protein starting from an all-X (unknown) sequence,
using Boltz for structure prediction and LigandMPNN for inverse folding.

Algorithm:
1. Predict 3D structure from current sequence (starts with all-X)
2. Use inverse folding (LigandMPNN) to design sequences for predicted structure
3. Repeat for num_cycles iterations

Usage:
    python protein_hunter.py
"""
from __future__ import annotations
from typing import Tuple

from proto_language.language.core import (
    Construct,
    Segment,
    Sequence,
    Program,
)
from proto_language.language.generator import (
    LigandMPNNGenerator,
    LigandMPNNGeneratorConfig,
)
from proto_language.language.optimizer import (
    CyclicalOptimizer,
    CyclicalOptimizerConfig,
)


# =============================================================================
# Configuration
# =============================================================================

NUM_CYCLES = 5           # Number of structure prediction -> inverse folding cycles
NUM_CANDIDATES = 2       # Number of parallel candidate trajectories
DESIGN_LENGTH = 100      # Length of the protein to design
STRUCTURE_TOOL = "boltz" # Structure prediction tool: "boltz", "chai", "esmfold", "alphafold3"

# Tool-specific configuration for structure prediction
TOOL_CONFIG = {
    "use_msa_server": False,  # Set True for better quality predictions (slower)
}


# =============================================================================
# Define the Protein Segment
# =============================================================================

# Define segment with just the length - CyclicalOptimizer automatically initializes
# sequences to 'X' (unknown residues) for the Protein Hunter hallucination trick.
protein = Segment(
    length=DESIGN_LENGTH,
    sequence_type="protein",
    label="designed_protein",
)


# =============================================================================
# Define the Construct
# =============================================================================

protein_construct = Construct([protein])


# =============================================================================
# Define the Generator
# =============================================================================

# LigandMPNN generator for inverse folding.
# No structure_inputs needed here - CyclicalOptimizer will provide predicted
# structures at runtime via sample(structure_inputs=...).
ligandmpnn_generator = LigandMPNNGenerator(
    LigandMPNNGeneratorConfig(
        temperature=0.1,  # Low temperature for more confident designs
        excluded_amino_acids=["C"],  # Exclude cysteine to avoid disulfide complications
    )
)


# =============================================================================
# Custom Logging
# =============================================================================

def custom_logging(cycle: int, segments: Tuple[Segment, ...]) -> None:
    """Log progress after each cycle."""
    output_sequence: Sequence = segments[0].selected_sequences[0]
    seq = output_sequence.sequence


    print(f"\n  Cycle {cycle}: {seq} (len={len(seq)})")


# =============================================================================
# Define the Optimizer
# =============================================================================

optimizer_config = CyclicalOptimizerConfig(
    num_cycles=NUM_CYCLES,
    num_candidates=NUM_CANDIDATES,
    structure_tool=STRUCTURE_TOOL,
    tool_config=TOOL_CONFIG,
    verbose=True,
)

optimizer = CyclicalOptimizer(
    target_segment=protein,
    constructs=[protein_construct],
    generators=[ligandmpnn_generator],  # Must be a list with exactly one generator
    constraints=[],  # No filtering constraints - pure Protein Hunter cycling
    config=optimizer_config,
    custom_logging=custom_logging,
)


# =============================================================================
# Run the Program
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Protein Hunter - De Novo Protein Design")
    print("=" * 60)
    print(f"  Design length: {DESIGN_LENGTH}")
    print(f"  Num cycles: {NUM_CYCLES}")
    print(f"  Num candidates: {NUM_CANDIDATES}")
    print(f"  Structure tool: {STRUCTURE_TOOL}")
    print("=" * 60)

    program = Program(optimizers=[optimizer])
    program.run()

    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    for i, seq in enumerate(protein.selected_sequences):
        print(f"\nCandidate {i + 1}:")
        print(f"  Sequence: {seq.sequence}")
        print(f"  Length: {len(seq.sequence)}")
