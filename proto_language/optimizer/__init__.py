"""Optimizer registry and all registered optimization strategies."""

# Registry and base infrastructure
from proto_language.optimizer.beam_search_optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
    BeamState,
)
from proto_language.optimizer.cycling_optimizer import CyclingOptimizer, CyclingOptimizerConfig
from proto_language.optimizer.evolutionary_optimizer import EvolutionaryOptimizer, EvolutionaryOptimizerConfig

# Optimizers
from proto_language.optimizer.gradient_optimizer import (
    ConstraintWeightSchedule,
    GradientOptimizer,
    GradientOptimizerConfig,
)
from proto_language.optimizer.mcmc_optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.optimizer.optimizer_registry import OptimizerRegistry, OptimizerSpec, optimizer
from proto_language.optimizer.rejection_sampling_optimizer import (
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)
from proto_language.utils.base import BaseOptimizerConfig

__all__ = [
    # Registry and base
    "BaseOptimizerConfig",
    "OptimizerRegistry",
    "OptimizerSpec",
    "optimizer",
    # MCMC Optimizer
    "MCMCOptimizer",
    "MCMCOptimizerConfig",
    # Beam Search Optimizer (single-segment)
    "BeamSearchOptimizer",
    "BeamSearchOptimizerConfig",
    "BeamState",
    # Rejection Sampling Optimizer
    "RejectionSamplingOptimizer",
    "RejectionSamplingOptimizerConfig",
    # Cycling Optimizer
    "CyclingOptimizer",
    "CyclingOptimizerConfig",
    # Evolutionary Optimizer
    "EvolutionaryOptimizer",
    "EvolutionaryOptimizerConfig",
    # Gradient Optimizer
    "GradientOptimizer",
    "GradientOptimizerConfig",
    "ConstraintWeightSchedule",
]
