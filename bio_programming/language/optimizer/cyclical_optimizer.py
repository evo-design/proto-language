"""
Cyclical Optimizer that cycles between structure prediction and inverse folding.

Implements a Protein Hunter-style algorithm that iteratively refines protein sequences
by predicting their structures and then using inverse folding to generate new sequences
conditioned on those structures.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Literal, Optional, final

from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
)
from proto_language.language.generator.generator_registry import GeneratorRegistry
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry
from proto_language.tools.structure_prediction.schemas import (
    StructurePredictionComplex,
)
from proto_language.utils.helpers import predict_structures


class CyclicalOptimizerConfig(BaseConfig):
    """Configuration for CyclicalOptimizer.

    This optimizer cycles between structure prediction and inverse folding to
    iteratively refine protein sequences. It operates on a single target segment
    while optionally including other context segments for structural context
    (e.g., a target protein for binder design).

    Attributes:
        num_cycles (int): Number of structure prediction -> inverse folding cycles.
            Each cycle predicts structures for current sequences, then generates
            new sequences conditioned on those structures. Must be >= 1.

        num_candidates (int): Number of independent candidate trajectories to
            maintain. Each candidate gets its own predicted structure and
            generates its own sequence per cycle. Must be >= 1.

        structure_tool (str): Structure prediction tool to use. Options:
            ``"alphafold3"``, ``"boltz"``, ``"chai"``. Note: ESMFold is not
            supported as it cannot predict structures from unknown sequences.
            Default: ``"boltz"``.

        tool_config (dict): Tool-specific configuration options passed to the
            structure prediction tool. See individual tool documentation for
            available options. Default: ``{}``.

        initialize_unknown (bool): Whether to initialize the target segment sequences
            with 'X' (unknown) residues. This is the 'hallucination trick' from
            Protein Hunter - starting with unknown residues allows structure predictors
            to explore novel folds without being biased by the input sequence.
            Default: ``True``.

        verbose (bool): Whether to print progress information. Default: ``False``.

    Note:
        - Only works with ``inverse_folding`` generators (ProteinMPNN, LigandMPNN)
        - Constraints are optional but if provided must be filter constraints
          (must have ``threshold`` set)
    """

    num_cycles: int = ConfigField(
        ge=1,
        title="Number of Cycles",
        description="Number of structure prediction -> inverse folding cycles to run.",
    )
    num_candidates: int = ConfigField(
        ge=1,
        title="Number of Candidates",
        description="Number of independent candidate trajectories to maintain.",
    )
    structure_tool: Literal["alphafold3", "boltz", "chai"] = ConfigField(
        default="boltz",
        title="Structure Prediction Tool",
        description="Tool to use for structure prediction.",
    )
    tool_config: Dict[str, Any] = ConfigField(
        default_factory=dict,
        title="Tool Config",
        description="Tool-specific configuration options.",
    )
    initialize_unknown: bool = ConfigField(
        default=True,
        title="Initialize with 'X' Tokens",
        description="If True, initialize candidate sequences with 'X' (unknown) tokens.",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
        hidden=True,
    )

    @field_validator("structure_tool", mode="before")
    @classmethod
    def validate_structure_tool(cls, v: str) -> str:
        if v == "esmfold":
            raise ValueError(
                "ESMFold is not supported for CyclicalOptimizer. Use 'alphafold3', 'boltz', or 'chai' instead."
            )
        return v


@OptimizerRegistry.register(
    key="cyclical",
    label="Cyclical Optimizer",
    config=CyclicalOptimizerConfig,
    description="Iterative optimizer that cycles between structure prediction and inverse folding",
)
@final
class CyclicalOptimizer(Optimizer):
    """Cyclical optimizer for iterative protein sequence refinement.

    Implements a Protein Hunter-style algorithm that iteratively refines protein
    sequences through structure prediction and inverse folding cycles:

    1. Predict 3D structures for all candidate sequences (including context segments)
    2. Condition the inverse folding generator on the predicted structure
    3. Generate a new sequence for the target segment
    4. Optionally filter sequences using constraints (with rollback for rejected)
    5. Repeat for num_cycles

    The optimizer operates on a single target segment while including other
    context segments for structural context (e.g., binder design with a target).

    Attributes:
        target_segment (Segment): The segment to optimize with inverse folding.
        generator (Generator): Single inverse_folding generator.
        num_cycles (int): Number of prediction-generation cycles.
        num_candidates (int): Number of independent candidate trajectories.
        structure_tool (str): Structure prediction tool name.
        tool_config (dict): Tool-specific configuration.

    Example:
        >>> segment = Segment(sequence="X" * 100, sequence_type="protein")
        >>> construct = Construct([segment])
        >>> generator = ProteinMPNNGenerator(config)
        >>> config = CyclicalOptimizerConfig(
        ...     num_cycles=5,
        ...     num_candidates=4,
        ...     structure_tool="boltz",
        ...     tool_config={"use_msa_server": False},
        ... )
        >>> optimizer = CyclicalOptimizer(
        ...     target_segment=segment,
        ...     constructs=[construct],
        ...     generators=[generator],
        ...     constraints=[],
        ...     config=config,
        ... )
        >>> optimizer.run()

    Note:
        - Only supports ``inverse_folding`` generators (ProteinMPNN, LigandMPNN)
        - Constraints are optional; if provided, must be filter constraints
          (have ``threshold`` set) - sequences that fail are rolled back
        - For best performance, use the same structure prediction tool in both
          ``structure_tool`` and any structure-based constraints. Tool caching
          provides significant speedups when the same predictor is reused.
    """

    config_class = CyclicalOptimizerConfig

    def __init__(
        self,
        target_segment: Segment,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: CyclicalOptimizerConfig,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the Cyclical Optimizer.

        Args:
            target_segment: The specific Segment to optimize. Must belong to one
                of the constructs.
            constructs: List of Construct objects. The target_segment must belong
                to one of these. Other segments provide structural context and
                must have input sequences.
            generators: List containing exactly one inverse_folding Generator
                (ProteinMPNN or LigandMPNN).
            constraints: List of Constraint objects for filtering. Can be empty.
                If provided, all constraints must have ``threshold`` set (filter mode).
            config: Configuration object with algorithm parameters.
            custom_logging: Optional callback called after each cycle with
                signature ``(cycle: int, segments: tuple) -> None``.
            clear_tool_cache: Cache management setting. (int) byte threshold,
                (bool) clear all, or (List[str]) specific tool names.

        Raises:
            ValueError: If generators list doesn't contain exactly one generator,
                target_segment is not in constructs, non-target segments don't have
                input sequences, or constraints don't have thresholds set.
        """
        if len(generators) != 1:
            raise ValueError(
                f"CyclicalOptimizer requires one inverse_folding generator, but got {len(generators)}."
            )
        generator = generators[0]
        generator.assign(target_segment)

        # Store for validation before super().__init__
        self.target_segment: Segment = target_segment
        self.generator: Generator = generator

        super().__init__(
            constructs=constructs,
            generators=[generator],
            constraints=constraints,
            num_candidates=config.num_candidates,
            num_selected=config.num_candidates,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
        )

        # Store optimizer-specific parameters
        self.num_cycles: int = config.num_cycles
        self.num_candidates: int = config.num_candidates
        self.structure_tool: str = config.structure_tool
        self.tool_config: Dict[str, Any] = config.tool_config

        # Override num_steps for progress tracking
        self.num_steps = config.num_cycles

        # Initialize sequences with unknown residues (Protein Hunter hallucination trick)
        if config.initialize_unknown:
            self._initialize_unknown_sequences()

    def run(self) -> None:
        """Execute the cyclical optimization loop."""
        if self.verbose:
            print(
                f"CyclicalOptimizer: {self.num_cycles} cycles, {self.num_candidates} candidates"
            )
        self._save_progress_snapshot(time_step=0)

        for cycle in range(1, self.num_cycles + 1):
            # 1. Save state for potential rollback
            if self.constraints:
                previous_sequences = [
                    copy.deepcopy(self.target_segment.candidate_sequences[i])
                    for i in range(self.num_candidates)
                ]

            # 2. Predict structures for all candidates
            complexes = [
                StructurePredictionComplex(
                    chains=[
                        seg.candidate_sequences[i].sequence for seg in self.segments
                    ]
                )
                for i in range(self.num_candidates)
            ]
            structures = predict_structures(
                complexes, self.structure_tool, self.tool_config
            ).structures

            # 3. Generate sequences conditioned on predicted structures
            self.generator.sample(structure_inputs=structures)

            # 4. Evaluate filter constraints and rollback rejected
            num_passed = self.num_candidates
            if self.constraints:
                self.score_energy()
                num_passed = self._revert_rejected_candidates(previous_sequences)

            # 5. Sync and save
            self.target_segment.selected_sequences = [
                copy.deepcopy(seq) for seq in self.target_segment.candidate_sequences
            ]
            self._save_progress_snapshot(time_step=cycle)
            self._log_cycle_progress(cycle, num_passed)

    def _initialize_unknown_sequences(self) -> None:
        """Initialize target segment sequences with 'X' (unknown) residues.

        This is the 'hallucination trick' from Protein Hunter - starting with unknown
        residues allows structure predictors to explore novel folds without being
        biased by the input sequence.
        """
        unknown_seq = "X" * self.target_segment.sequence_length
        for seq in self.target_segment.candidate_sequences:
            seq.sequence = unknown_seq
        for seq in self.target_segment.selected_sequences:
            seq.sequence = unknown_seq

    def _validate_optimizer(self) -> None:
        """Validate optimizer configuration.

        Validates:
        1. Constructs are valid and non-empty
        2. target_segment belongs to one of the constructs
        3. Generator is valid (inverse_folding category)
        4. Constraints (if any) must be filter constraints (have threshold set)
        """
        # Validate constructs
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise TypeError(
                    f"Construct {i} has type {type(construct)}, expected Construct"
                )
            if not construct.segments:
                raise ValueError(f"Construct {i} has no segments")

        # Validate target_segment belongs to one of the constructs
        if self.target_segment not in self.segments:
            raise ValueError(
                f"target_segment '{self.target_segment.label or 'unlabeled'}' is not in any of the provided constructs"
            )

        # Validate generator is inverse_folding category
        if not isinstance(self.generator, Generator):
            raise TypeError(
                f"Generator has type {type(self.generator)}, expected Generator"
            )
        spec = GeneratorRegistry.get(GeneratorRegistry.get_key(self.generator))
        if spec.category != "inverse_folding":
            raise ValueError(
                f"CyclicalOptimizer requires an inverse_folding generator. "
                f"Got {self.generator.__class__.__name__} with category '{spec.category}'. "
            )

        # Validate constraints (optional, but if present must be filters)
        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise TypeError(
                    f"Constraint {i} has type {type(constraint)}, expected Constraint"
                )
            if not constraint.inputs:
                raise RuntimeError(f"Constraint {i} has no input segment(s) assigned")
            # This optimizer only supports filter constraints
            if constraint.threshold is None:
                raise ValueError(
                    f"CyclicalOptimizer only supports filter constraints. Constraint {i} ('{constraint.label}') has no threshold set."
                )

    def _revert_rejected_candidates(self, previous_sequences: List[Any]) -> int:
        """Roll back candidates that failed filter constraints. Returns num_passed."""
        num_rejected = 0
        for candidate_idx, score in enumerate(self.energy_scores):
            if math.isinf(score):
                num_rejected += 1
                self.target_segment.candidate_sequences[candidate_idx] = copy.deepcopy(
                    previous_sequences[candidate_idx]
                )
        return self.num_candidates - num_rejected

    def _log_cycle_progress(self, cycle: int, num_passed: int) -> None:
        """Log cycle progress."""
        if self.verbose:
            seq = self.target_segment.selected_sequences[0].sequence
            print(f"Cycle {cycle}/{self.num_cycles}")
            print(f"passed: {num_passed}/{self.num_candidates}")
            print(f"seq: {seq}")
        if self.custom_logging:
            self.custom_logging(cycle, self.segments)
