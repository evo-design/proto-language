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
    python protein_hunter.py
"""
from __future__ import annotations

from typing import List, Tuple

from proto_language.language.core import (
    Construct,
    Segment,
    Sequence,
    Program,
)
from proto_language.language.generator import (
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)
from proto_language.language.optimizer import (
    CyclingOptimizer,
    CyclingOptimizerConfig,
)
from proto_language.tools.structure_prediction.schemas import (
    StructurePredictionComplex,
)
from proto_language.utils.helpers import predict_structures


# =============================================================================
# Configuration
# =============================================================================

NUM_CYCLES = 5           # Number of structure prediction -> inverse folding cycles
NUM_CANDIDATES = 2       # Number of parallel candidate trajectories
DESIGN_LENGTH = 100      # Length of the protein to design
STRUCTURE_TOOL = "boltz" # Structure prediction tool: "boltz", "chai", "alphafold3"

# Tool-specific configuration for structure prediction
TOOL_CONFIG = {} # use all default values


# =============================================================================
# Define the Protein Segment
# =============================================================================

# Initialize with 'X' (unknown) residues - the 'hallucination trick' from Protein Hunter.
# Starting with unknown residues allows structure predictors to explore novel folds
# without being biased by an input sequence.
protein = Segment(
    sequence="X" * DESIGN_LENGTH,
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

# ProteinMPNN generator for inverse folding.
# No structure_inputs needed here - CyclingOptimizer will provide predicted
# structures at runtime via the conditioning function.
proteinmpnn_generator = ProteinMPNNGenerator(
    ProteinMPNNGeneratorConfig(
        temperature=0.1,  # Low temperature for more confident designs
        excluded_amino_acids=["C"],  # Exclude cysteine to avoid disulfide complications
    )
)


# =============================================================================
# Conditioning Function for Structure Prediction
# =============================================================================

def structure_conditioning_fn(sequences: List[Sequence]) -> List:
    """
    Predict 3D structures from current sequences.

    This is the conditioning function for the Protein Hunter algorithm.
    It takes current candidate sequences and predicts their structures,
    which are then used to condition the inverse folding generator.
    """
    complexes = [
        StructurePredictionComplex(chains=[seq.sequence])
        for seq in sequences
    ]
    return predict_structures(complexes, STRUCTURE_TOOL, TOOL_CONFIG).structures


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

optimizer_config = CyclingOptimizerConfig(
    num_steps=NUM_CYCLES,
    num_candidates=NUM_CANDIDATES,
    conditioning_param_name="structure_inputs",  # Pass structures to generator.sample()
    verbose=True,
)

optimizer = CyclingOptimizer(
    target_segment=protein,
    constructs=[protein_construct],
    generators=[proteinmpnn_generator],
    constraints=[],
    config=optimizer_config,
    conditioning_fn=structure_conditioning_fn,
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
