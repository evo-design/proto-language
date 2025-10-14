"""
Shared utilities for beam search demo scripts.

This module provides common classes and functions used by both single-segment
and multi-segment beam search demos to reduce code duplication.
"""

import sys
import os

from proto_language.language.core import Segment, Generator


class MockAutoregressiveGenerator(Generator):
    """Mock extension-based generator that predicts next base pair using state transition probabilities."""
    
    def __init__(self, sequence_length: int = 10, prepend_prompt: bool = True, random_seed: int = None, batch_size: int = 1):
        super().__init__(batch_size=batch_size)
        self.sequence_length = sequence_length
        self.prepend_prompt = prepend_prompt
        self.prompt_seqs = None
        self.random_seed = random_seed
        
        # Create a persistent random state for this generator
        import random
        if self.random_seed is not None:
            self.random_state = random.Random(self.random_seed)
        else:
            self.random_state = random.Random()
        
        # Define state transition probabilities
        self.transition_probs = {
            'A': {'T': 0.25, 'A': 0.25, 'C': 0.25, 'G': 0.25},
            'T': {'T': 0.25, 'A': 0.25, 'C': 0.25, 'G': 0.25},
            'G': {'G': 0.25, 'C': 0.25, 'A': 0.25, 'T': 0.25},
            'C': {'C': 0.25, 'G': 0.25, 'A': 0.25, 'T': 0.25}
        }
    
    def assign(self, segment: Segment) -> None:
        """Assign this generator to a segment."""
        if self._is_initialized:
            raise ValueError("Generator already assigned")
        
        self._generator_output = segment
        self._is_initialized = True
        segment._is_assigned = True
    
    def sample(self, prompt_seqs=None) -> None:
        """Generate sequences autoregressively based on transition probabilities."""
        if not self._is_initialized:
            raise ValueError("Generator not initialized")
        
        # Use provided prompts or current sequences
        if prompt_seqs is not None:
            self.prompt_seqs = prompt_seqs
            base_sequences = prompt_seqs
        else:
            base_sequences = [seq.sequence for seq in self._generator_output.batch_sequences]
        
        # Generate new sequences for each base sequence
        for i, base_sequence in enumerate(base_sequences):
            if i >= len(self._generator_output.batch_sequences):
                break
                
            # Always generate sequence_length NEW tokens (like Evo2Generator)
            generated_tokens = ""
            
            if base_sequence:
                # Use last character of base sequence to start generation
                last_char = base_sequence[-1]
            else:
                # If no base sequence, start with 'A'
                last_char = 'A'
            
            # Generate exactly sequence_length new tokens
            for _ in range(self.sequence_length):
                if last_char in self.transition_probs:
                    # Sample next character based on transition probabilities
                    probs = self.transition_probs[last_char]
                    next_char = self.random_state.choices(list(probs.keys()), weights=list(probs.values()))[0]
                else:
                    # Default to 'A' if last character is not in transition table
                    next_char = 'A'
                generated_tokens += next_char
                last_char = next_char
            
            # Handle prepend_prompt behavior (like Evo2Generator)
            if self.prepend_prompt and prompt_seqs is not None:
                # If prepend_prompt is True, return prompt + generated tokens
                final_sequence = base_sequence + generated_tokens
            else:
                # If prepend_prompt is False, return only the generated tokens
                final_sequence = generated_tokens
            
            self._generator_output.batch_sequences[i].sequence = final_sequence
    
    def get_generator_outputs(self):
        """Get the generated sequences."""
        if not self._is_initialized:
            return ()
        return (self._generator_output,)

def gc_content_constraint(sequence) -> float:
    """Constraint that scores lower energy for higher G+C content."""
    if hasattr(sequence, 'sequence'):
        sequence_str = sequence.sequence
    else:
        sequence_str = sequence
    
    # Count G and C occurrences
    g_count = sequence_str.count('G')
    c_count = sequence_str.count('C')
    total_length = len(sequence_str)
    
    if total_length == 0:
        return 100.0  # High energy for empty sequences
    
    # Calculate G+C ratio (percentage)
    gc_ratio = ((g_count + c_count) / total_length) * 100
    
    # Lower energy (better) for higher G+C ratio
    return max(0.0, 100.0 - gc_ratio)



