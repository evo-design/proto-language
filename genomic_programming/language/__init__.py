"""
High-level programming language framework for genomic sequence design.

This package provides the core abstractions for constraint-driven sequence optimization:
- base: Core classes (Sequence, Segment, Construct, Constraint, Generator)
- generator: Sequence generation algorithms (MCMC, Evo2, ESM2, etc.)
- constraint: Scoring functions for sequence evaluation
- program: User-friendly wrapper for optimization workflows
"""

from .base import (
    Sequence,
    Segment,
    Construct,
    Constraint,
    Generator,
    IterativeGenerator,
)

from .generator import (
    UniformMutationGenerator,
    Evo2Generator,
    ESM2Generator,
    ESM3Generator,
    SlowMutationGenerator,
    MCMCGenerator,
    BeamSearchGenerator,
    GeneratorRegistry,
)

from .base import Program

__all__ = [
    # Base classes
    "Sequence",
    "Segment",
    "Construct",
    "Constraint",
    "Generator",
    "IterativeGenerator",
    # Generators
    "UniformMutationGenerator",
    "Evo2Generator",
    "ESM2Generator",
    "ESM3Generator",
    "SlowMutationGenerator",
    "MCMCGenerator",
    "BeamSearchGenerator",
    "GeneratorRegistry",
    # Program
    "Program",
]

