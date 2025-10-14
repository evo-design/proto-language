"""
Demo script for BeamSearchGenerator.

This script demonstrates how beam search works with multiple segments
using simplified generators that generate one character at a time.

Usage:
    python demo_beam_search.py                    # Use local MockAutoregressiveGenerator
    python demo_beam_search.py --use-cloud        # Use cloud Evo2Generator
    python demo_beam_search.py --help             # Show help
"""

import sys
import os
import argparse

from proto_language.language.core import Construct, Segment, Constraint, SequenceType
from proto_language.language.generator import BeamSearchGenerator, Evo2Generator

# Import shared utilities
from .beam_search_utils import (
    MockAutoregressiveGenerator,
    gc_content_constraint,
)

def create_generators(use_cloud: bool, prepend_prompt: bool = True, evo2_type: str = "evo2_7b", temperature: float = 0.8):
    """
    Create generators based on the use_cloud flag.
    
    Args:
        use_cloud: If True, use cloud Evo2Generator; if False, use MockAutoregressiveGenerator
        prepend_prompt: Whether to prepend the prompt to generated sequences
        evo2_type: Evo2 model type to use (only used when use_cloud=True)
        temperature: Sampling temperature (only used when use_cloud=True)
    
    Returns:
        List of generators
    """
    generators = []
    
    if use_cloud:
        print("Creating cloud Evo2Generators...")
        
        # Create Evo2Generators with different prompts
        evo2_gen1 = Evo2Generator(
            prompt_seqs=["A"],  # Initial prompt
            evo2_type=evo2_type,  # Model type
            sequence_length=1,  # Generate 1 token at a time
            temperature=temperature,
            top_k=4,
            top_p=1.0,
            batch_size=1,
            prepend_prompt=prepend_prompt,
            verbose=1
        )
        generators.append(evo2_gen1)
        
        evo2_gen2 = Evo2Generator(
            prompt_seqs=["T"],
            evo2_type=evo2_type,
            sequence_length=1,
            temperature=temperature,
            top_k=4,
            top_p=1.0,
            batch_size=1,
            prepend_prompt=prepend_prompt,
            verbose=1
        )
        generators.append(evo2_gen2)
        
        # Segment 3 gets 2 generators (no initial prompt)
        evo2_gen3 = Evo2Generator(
            prompt_seqs=[""],  # Empty prompt
            evo2_type=evo2_type,
            sequence_length=1,
            temperature=temperature,
            top_k=4,
            top_p=1.0,
            batch_size=1,
            prepend_prompt=prepend_prompt,
            verbose=1
        )
        generators.append(evo2_gen3)
        
        evo2_gen4 = Evo2Generator(
            prompt_seqs=[""],
            evo2_type=evo2_type,
            sequence_length=1,
            temperature=temperature,
            top_k=4,
            top_p=1.0,
            batch_size=1,
            prepend_prompt=prepend_prompt,
            verbose=1
        )
        generators.append(evo2_gen4)
        
    else:
        print("Creating local MockAutoregressiveGenerators...")
        
        # Create MockAutoregressiveGenerators with different random seeds for variety
        mock_gen1 = MockAutoregressiveGenerator(sequence_length=1, random_seed=69, prepend_prompt=prepend_prompt)
        generators.append(mock_gen1)
        
        mock_gen2 = MockAutoregressiveGenerator(sequence_length=1, random_seed=42, prepend_prompt=prepend_prompt)
        generators.append(mock_gen2)
        
        # Segment 3 gets 2 generators (no initial prompt)
        mock_gen3 = MockAutoregressiveGenerator(sequence_length=1, random_seed=123, prepend_prompt=prepend_prompt)
        generators.append(mock_gen3)
        
        mock_gen4 = MockAutoregressiveGenerator(sequence_length=1, random_seed=456, prepend_prompt=prepend_prompt)
        generators.append(mock_gen4)
    
    return generators

def main():
    """Run the beam search demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Demo script for BeamSearchGenerator with optional cloud Evo2Generator support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_beam_search.py                    # Use local MockAutoregressiveGenerator
  python demo_beam_search.py --use-cloud        # Use cloud Evo2Generator
  python demo_beam_search.py --beam-width 3     # Custom beam width
  python demo_beam_search.py --samples 5        # Generate 5 samples
        """
    )
    parser.add_argument(
        "--use-cloud",
        action="store_true",
        help="Use cloud Evo2Generator instead of local MockAutoregressiveGenerator"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=2,
        help="Beam width for beam search (default: 2)"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=3,
        help="Number of candidates to keep (default: 3)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of samples to generate (default: 2)"
    )
    parser.add_argument(
        "--evo2-type",
        type=str,
        default="evo2_7b",
        help="Evo2 model type to use (default: evo2_7b)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BEAM SEARCH DEMO - MULTI-SEGMENT")
    print("=" * 60)
    print()
    
    # Configuration
    beam_width = args.beam_width
    num_candidates = args.candidates
    num_samples = args.samples
    PREPEND_PROMPT = True
    
    generator_type = "cloud Evo2Generator" if args.use_cloud else "Local MockAutoregressiveGenerator"
    
    print(f"Configuration:")
    print(f"  Generator type: {generator_type}")
    print(f"  Initial prompts: 'A', 'T', '' (segment 3 has no initial prompt)")
    print(f"  Beam width: {beam_width}")
    print(f"  Num candidates: {num_candidates}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Constraint: G+C content optimization")
    print(f"  Generator sequence_length: 1 (generates 1 new token per segment)")
    print(f"  Segment 3 has 2 generators assigned")
    if args.use_cloud:
        print(f"  Evo2 model type: {args.evo2_type}")
        print(f"  Temperature: {args.temperature}")
    print()
    
    # Create multiple segments with different initial sequences
    segment1 = Segment(sequence="A", sequence_type=SequenceType.DNA)
    segment2 = Segment(sequence="T", sequence_type=SequenceType.DNA)
    segment3 = Segment(sequence="", sequence_type=SequenceType.DNA)  # No initial prompt
    
    # Create construct with multiple segments
    construct = Construct([segment1, segment2, segment3])
    
    print("Initial segments:")
    print(f"  Segment 1: {segment1.batch_sequences[0].sequence}")
    print(f"  Segment 2: {segment2.batch_sequences[0].sequence}")
    print(f"  Segment 3: {segment3.batch_sequences[0].sequence}")
    print()
    
    # Create generators based on the use_cloud flag
    generators = create_generators(args.use_cloud, PREPEND_PROMPT, args.evo2_type, args.temperature)
    
    # Assign generators to segments
    generators[0].assign(segment1)
    generators[1].assign(segment2)
    generators[2].assign(segment3)
    generators[3].assign(segment3)  # Assign to the same segment
    
    # Create constraint that applies to all segments
    constraint = Constraint(
        inputs=[segment1, segment2, segment3],
        scoring_function=gc_content_constraint,
        scoring_function_config={}
    )
    
    # Create beam search generator
    beam_gen = BeamSearchGenerator(
        constructs=[construct],
        generators=generators,  # Include all generators
        constraints=[constraint],
        beam_width=beam_width,
        num_candidates=num_candidates,
        verbose=True  # Enable verbose logging
    )
    
    print("Starting beam search...")
    print()
    
    # Run beam search steps
    for step in range(num_samples):
        print(f"SAMPLE {step + 1}")
        print("=" * 80)
        # Run beam search step
        beam_gen.sample()
        print()
    
if __name__ == "__main__":
    main() 