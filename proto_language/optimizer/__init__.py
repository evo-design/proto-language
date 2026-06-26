"""Optimizer registry and all registered optimization strategies."""

# Registry and base infrastructure
from proto_language.optimizer.beam_search_optimizer import (
    BeamSearchOptimizer,
    BeamSearchOptimizerConfig,
    BeamState,
)
from proto_language.optimizer.cycling_optimizer import CyclingOptimizer, CyclingOptimizerConfig

# Optimizers
from proto_language.optimizer.gradient_optimizer import (
    ConstraintWeightSchedule,
    GradientOptimizer,
    GradientOptimizerConfig,
)
from proto_language.optimizer.genetic_algorithm_optimizer import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
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
    # Genetic Algorithm Optimizer
    "GeneticAlgorithmOptimizer",
    "GeneticAlgorithmOptimizerConfig",
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
    # Gradient Optimizer
    "GradientOptimizer",
    "GradientOptimizerConfig",
    "ConstraintWeightSchedule",
]
