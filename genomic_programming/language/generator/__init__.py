"""
Generator implementations for the proto-language.

This module provides concrete implementations of sequence generation algorithms:
- UniformMutationGenerator: Random point mutations
- SlowMutationGenerator: Slow mutations for testing
- Evo2Generator: Evo2 genome language model generation
- ESM2Generator: ESM-2 protein language model generation
- ESM3Generator: ESM-3 protein language model generation
- MCMCGenerator: Metropolis-Hastings MCMC optimization
- BeamSearchGenerator: Beam search optimization

Registry System:
- GeneratorRegistry: Central registry for generator discovery and execution
- All generators are registered with metadata for API/client integration
"""

# Registry and base infrastructure
from .registry import GeneratorRegistry

# Simple mutation generators
from .uniform_mutation import (
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from .slow_mutation import (
    SlowMutationGenerator,
    SlowMutationGeneratorConfig,
)

# Language model generators
from .evo2 import (
    Evo2Generator,
    Evo2GeneratorConfig,
)
from .esm2 import (
    ESM2Generator,
    ESM2GeneratorConfig,
)
from .esm3 import (
    ESM3Generator,
    ESM3GeneratorConfig,
)

# Optimization generators (not in registry - these are meta-generators)
from .mcmc import MCMCGenerator
from .beam_search import BeamSearchGenerator

__all__ = [
    # Registry
    "GeneratorRegistry",
    # Mutation generators
    "UniformMutationGenerator",
    "UniformMutationGeneratorConfig",
    "SlowMutationGenerator",
    "SlowMutationGeneratorConfig",
    # Language model generators
    "Evo2Generator",
    "Evo2GeneratorConfig",
    "ESM2Generator",
    "ESM2GeneratorConfig",
    "ESM3Generator",
    "ESM3GeneratorConfig",
    # Optimization generators
    "MCMCGenerator",
    "BeamSearchGenerator",
]
