"""
Cycling Optimizer that cycles between a conditioning function and a generator.

A generalized optimizer that iteratively runs a user-defined conditioning function
and passes its output to a generator. Supports optional constraint filtering with
rollback for rejected candidates.
"""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Callable, List, Optional, final

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry


class CyclingOptimizerConfig(BaseConfig):
    """Configuration for CyclingOptimizer.

    This optimizer cycles between a user-defined conditioning function and a generator.
    On each cycle, the conditioning function receives the current candidate sequences,
    produces conditioning data, which is then passed to the generator's sample() method.

    Attributes:
        num_steps (int): Number of conditioning -> generation cycles to run.
            Each cycle calls the conditioning function, then the generator.
            Must be >= 1.

        num_candidates (int): Number of independent candidate trajectories to
            maintain. Each candidate is processed independently through the
            conditioning function and generator. Must be >= 1.

        conditioning_param_name (str): The keyword argument name to pass conditioning
            data to in the generator's ``sample()`` method. For example:
            - ``"structure_inputs"`` for inverse folding generators (ProteinMPNN, LigandMPNN)
            - ``"prompts"`` for autoregressive generators (Evo2)

        verbose (bool): Whether to print progress information. Default: ``False``.

    Note:
        - Works with any generator that accepts the specified conditioning_param_name
        - Constraints are optional but if provided must be filter constraints
          (must have ``threshold`` set)

    Example:
        >>> config = CyclingOptimizerConfig(
        ...     num_steps=5,
        ...     num_candidates=4,
        ...     conditioning_param_name="structure_inputs",
        ... )
    """

    num_steps: int = ConfigField(
        ge=1,
        title="Number of Steps",
        description="Number of conditioning -> generation cycles to run.",
    )
    num_candidates: int = ConfigField(
        ge=1,
        title="Number of Candidates",
        description="Number of independent candidate trajectories to maintain.",
    )
    conditioning_param_name: str = ConfigField(
        title="Conditioning Param Name",
        description="Generator sample() parameter name to pass conditioning data into.",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
        hidden=True,
    )

# TODO: Cycling optimizer conditioning_fn is not supported in client at all, since we can't serialize or define callables
# In the future, we can optionally include constraints scores into the conditioning_fn for more granular control over the optimization process.

@OptimizerRegistry.register(
    key="cycling",
    label="Cycling Optimizer",
    config=CyclingOptimizerConfig,
    description="Iterative optimizer that cycles between a conditioning function and generator",
)
@final
class CyclingOptimizer(Optimizer):
    """Cycling optimizer for iterative sequence refinement.

    A generalized optimizer that cycles between a user-defined conditioning function
    and a generator:

    1. Call conditioning function with current sequences
    2. Pass conditioning output to generator's sample() method
    3. Optionally filter sequences using constraints (with rollback for rejected)
    4. Repeat for num_steps

    This enables flexible optimization patterns such as:
    - Protein Hunter: Structure prediction -> inverse folding cycles
    - Evo2 with feedback: Constraint-guided prompt modification -> generation cycles

    Attributes:
        target_segment (Segment): The segment being optimized.
        generator (Generator): The generator to use for sequence generation.
        conditioning_fn (Callable): User-defined function that produces conditioning data.
        conditioning_param_name (str): Generator sample() parameter name for conditioning data.
        num_steps (int): Number of cycles to run.
        num_candidates (int): Number of independent candidate trajectories.

    Example:
        >>> def my_conditioning_fn(sequences):
        ...     # Process sequences and return conditioning data
        ...     return [process(seq) for seq in sequences]
        ...
        >>> optimizer = CyclingOptimizer(
        ...     target_segment=segment,
        ...     constructs=[construct],
        ...     generators=[generator],
        ...     constraints=[],
        ...     config=CyclingOptimizerConfig(
        ...         num_steps=5,
        ...         num_candidates=4,
        ...         conditioning_param_name="structure_inputs",
        ...     ),
        ...     conditioning_fn=my_conditioning_fn,
        ... )
        >>> optimizer.run()

    Note:
        - Constraints are optional; if provided, must be filter constraints
          (have ``threshold`` set) - sequences that fail are rolled back
    """

    config_class = CyclingOptimizerConfig

    def __init__(
        self,
        target_segment: Segment,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: CyclingOptimizerConfig,
        conditioning_fn: Callable[[List[Sequence]], List[Any]],
        custom_logging: Optional[Callable[[int, tuple], None]] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the Cycling Optimizer.

        Args:
            target_segment: The specific Segment to optimize. Must belong to one
                of the constructs.
            constructs: List of Construct objects. The target_segment must belong
                to one of these.
            generators: List containing exactly one Generator.
            constraints: List of Constraint objects for filtering. Can be empty.
                If provided, all constraints must have ``threshold`` set (filter mode).
            config: Configuration object with algorithm parameters.
            conditioning_fn: User-defined function that produces conditioning data.
                Signature: ``(sequences: List[Sequence]) -> List[Any]``
                Returns one conditioning item per candidate.
            custom_logging: Optional callback called after each cycle with
                signature ``(cycle: int, segments: tuple) -> None``.
            clear_tool_cache: Cache management setting. (int) byte threshold,
                (bool) clear all, or (List[str]) specific tool names.

        Raises:
            ValueError: If generators list doesn't contain exactly one generator,
                target_segment is not in constructs, or constraints don't have
                thresholds set.
        """
        if len(generators) != 1:
            raise ValueError(f"CyclingOptimizer requires exactly one generator, got {len(generators)}.")
        generator = generators[0]
        generator.assign(target_segment)

        # Store for validation before super().__init__
        self.target_segment: Segment = target_segment
        self.generator: Generator = generator
        self.conditioning_fn = conditioning_fn
        self.conditioning_param_name: str = config.conditioning_param_name

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
        self.num_steps: int = config.num_steps
        self.num_candidates: int = config.num_candidates

    def run(self) -> None:
        """Execute the cycling optimization loop."""
        if self.verbose:
            print(f"CyclingOptimizer: {self.num_steps} steps, {self.num_candidates} candidates")
        self._save_progress_snapshot(time_step=0)

        for step in range(1, self.num_steps + 1):
            # 1. Save state for potential rollback
            if self.constraints:
                previous_sequences = [
                    copy.deepcopy(self.target_segment.candidate_sequences[i])
                    for i in range(self.num_candidates)
                ]

            # 2. Call conditioning function with current sequences
            current_sequences = list(self.target_segment.candidate_sequences)
            conditioning_data = self.conditioning_fn(current_sequences)

            # 3. Generate sequences conditioned on the conditioning data
            self.generator.sample(**{self.conditioning_param_name: conditioning_data})

            # 4. Evaluate filter constraints and rollback rejected
            num_passed = self.num_candidates
            if self.constraints:
                self.score_energy()
                num_passed = self._revert_rejected_candidates(previous_sequences)

            # 5. Sync and save
            self.target_segment.selected_sequences = [copy.deepcopy(seq) for seq in self.target_segment.candidate_sequences]
            self._save_progress_snapshot(time_step=step)
            self._log_step_progress(step, num_passed)

    def _validate_optimizer(self) -> None:
        """Validate optimizer configuration."""
        # Validate constructs
        if not self.constructs:
            raise ValueError("Constructs list cannot be empty")
        for i, construct in enumerate(self.constructs):
            if not isinstance(construct, Construct):
                raise TypeError(f"Construct {i} has type {type(construct)}, expected Construct")
            if not construct.segments:
                raise ValueError(f"Construct {i} has no segments")

        # Validate target_segment belongs to one of the constructs
        if self.target_segment not in self.segments:
            raise ValueError(f"target_segment '{self.target_segment.label or 'unlabeled'}' is not in any of the provided constructs")

        # Validate generator
        if not isinstance(self.generator, Generator):
            raise TypeError(f"Generator has type {type(self.generator)}, expected Generator")

        # Validate conditioning_fn is callable
        if not callable(self.conditioning_fn):
            raise TypeError(f"conditioning_fn must be callable, got {type(self.conditioning_fn)}")

        # Validate conditioning_param_name is accepted by generator.sample()
        sample_sig = inspect.signature(self.generator.sample)
        valid_params = set(sample_sig.parameters.keys()) - {"self"}
        if self.conditioning_param_name not in valid_params:
            raise ValueError(
                f"Generator {self.generator.__class__.__name__}.sample() does not accept parameter '{self.conditioning_param_name}'. "
                f"Valid parameters: {sorted(valid_params)}"
            )

        # Validate constraints (optional, but if present must be filters)
        for i, constraint in enumerate(self.constraints):
            if not isinstance(constraint, Constraint):
                raise TypeError(f"Constraint {i} has type {type(constraint)}, expected Constraint")
            if not constraint.inputs:
                raise RuntimeError(f"Constraint {i} has no input segment(s) assigned")
            if constraint.threshold is None:
                raise ValueError(f"CyclingOptimizer only supports filter constraints. Constraint {i} ('{constraint.label}') has no threshold set.")

    def _revert_rejected_candidates(self, previous_sequences: List[Any]) -> int:
        """Roll back candidates that failed filter constraints. Returns num_passed."""
        num_rejected = 0
        for candidate_idx, score in enumerate(self.energy_scores):
            if math.isinf(score):
                num_rejected += 1
                self.target_segment.candidate_sequences[candidate_idx] = copy.deepcopy(previous_sequences[candidate_idx])
        return self.num_candidates - num_rejected

    def _log_step_progress(self, step: int, num_passed: int) -> None:
        """Log step progress."""
        if self.verbose:
            seq = self.target_segment.selected_sequences[0].sequence
            print(f"Step {step}/{self.num_steps}")
            print(f"passed: {num_passed}/{self.num_candidates}")
            print(f"seq: {seq}")
        if self.custom_logging:
            self.custom_logging(step, self.segments)
