"""
proto_language/language/optimizer/cycling_optimizer.py

A generalized optimizer that iteratively runs a user-defined conditioning function
and passes its output to a generator. Supports optional constraint filtering with
accept pattern for passing proposals.
"""

from __future__ import annotations

import copy
import inspect
import logging
import math
from typing import Any, Callable, Dict, List, Literal, Optional, final

from pydantic import model_validator

logger = logging.getLogger(__name__)

from proto_language.base_config import BaseConfig, BaseOptimizerConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.language.optimizer.optimizer_registry import optimizer

# =============================================================================
# Predefined Pipelines
# =============================================================================

class ProteinHunterPipelineConfig(BaseConfig):
    """Configuration for the protein-hunter pipeline.

    The protein-hunter pipeline implements iterative structure prediction -> inverse folding
    cycles for de novo protein design (hallucination).

    Attributes:
        structure_tool (Literal['boltz2', 'chai1', 'alphafold3']): Structure prediction tool to use. Options: "boltz2", "chai1", "alphafold3".
    """
    structure_tool: Literal["boltz2", "chai1", "alphafold3"] = ConfigField(
        default="boltz2",
        title="Structure Tool",
        description="Structure prediction tool: 'boltz2', 'chai1', or 'alphafold3'.",
    )

def _create_protein_hunter_conditioning_fn(config: CyclingOptimizerConfig) -> Callable:
    """
    Create protein hunter conditioning function (structure prediction -> inverse folding).

    The Protein Hunter algorithm predicts 3D structures from current sequences,
    then uses those structures to condition inverse folding for the next iteration.

    Args:
        config (CyclingOptimizerConfig): Constraint configuration controlling evaluation parameters.
    """
    from proto_tools import StructurePredictionComplex, predict_structures

    structure_tool = config.protein_hunter.structure_tool if config.protein_hunter else "boltz2"

    def conditioning_fn(sequences: List[Sequence]) -> List:
        complexes = [
            StructurePredictionComplex(chains=[seq.sequence])
            for seq in sequences
        ]
        return predict_structures(complexes, structure_tool, {}).structures

    return conditioning_fn


# Registry mapping pipeline names to factory functions and required generator categories
CYCLING_PIPELINES: Dict[str, Dict[str, Any]] = {
    "protein-hunter": {
        "factory": _create_protein_hunter_conditioning_fn,
        "required_generator_category": "inverse_folding",
    },
}


# =============================================================================
# Predefined Pipeline Helpers
# =============================================================================

def _resolve_conditioning_fn(
    config: CyclingOptimizerConfig,
    generator: Generator,
    conditioning_fn: Optional[Callable] = None,
) -> Callable:
    """
    Resolve the conditioning function from either direct parameter or pipeline config.

    Args:
        config (CyclingOptimizerConfig): Optimizer config containing optional pipeline specification
        generator (Generator): The generator to validate against pipeline requirements
        conditioning_fn (Callable | None): Optional directly-provided conditioning function

    Returns:
        Callable: The resolved conditioning function

    Raises:
        ValueError: If both or neither of conditioning_fn/pipeline are provided,
            or if generator doesn't match pipeline requirements
    """
    # Mutual exclusivity check
    if config.pipeline is not None and conditioning_fn is not None:
        raise ValueError(
            "Cannot specify both 'conditioning_fn' and 'pipeline'. "
            "Use 'pipeline' for API/JSON or 'conditioning_fn' for programmatic use."
        )

    # Must have one or the other
    if config.pipeline is None and conditioning_fn is None:
        raise ValueError(
            f"Must specify either 'conditioning_fn' or 'pipeline'. "
            f"Available pipelines: {list(CYCLING_PIPELINES.keys())}"
        )

    # If conditioning_fn provided directly, use it
    if conditioning_fn is not None:
        return conditioning_fn

    # Validate pipeline exists
    if config.pipeline not in CYCLING_PIPELINES:
        raise ValueError(
            f"Unknown pipeline '{config.pipeline}'. "
            f"Available: {list(CYCLING_PIPELINES.keys())}"
        )

    # Validate generator category matches pipeline requirements
    pipeline_spec = CYCLING_PIPELINES[config.pipeline]
    required_category = pipeline_spec.get("required_generator_category")
    if required_category:
        from proto_language.language.generator import GeneratorRegistry
        generator_key = GeneratorRegistry.get_key(generator)
        actual_category = GeneratorRegistry.get(generator_key).category
        if actual_category != required_category:
            raise ValueError(
                f"Pipeline '{config.pipeline}' requires {required_category} generator, "
                f"but '{generator_key}' is {actual_category}. "
                f"Use 'proteinmpnn' or 'ligandmpnn'."
            )

    return pipeline_spec["factory"](config)


# =============================================================================
# Config
# =============================================================================

class CyclingOptimizerConfig(BaseOptimizerConfig):
    """Configuration for CyclingOptimizer.

    This optimizer cycles between a conditioning function and a generator.
    On each cycle, the conditioning function receives the current proposal sequences,
    produces conditioning data, which is then passed to the generator's sample() method.

    The conditioning function can be provided either:
    1. Directly via the ``conditioning_fn`` parameter (programmatic use)
    2. Via the ``pipeline`` field using a predefined pipeline (API/JSON use)

    Attributes:
        num_steps (int): Number of conditioning -> generation cycles to run.
            Each cycle calls the conditioning function, then the generator.
            Must be >= 1.

        num_results (int | None): Number of independent proposal trajectories
            to maintain. Each proposal is processed independently through the
            conditioning function and generator. Overrides program-level ``num_results``
            if set.

        conditioning_param_name (str): The keyword argument name to pass conditioning
            data to in the generator's ``sample()`` method. For example:
            - ``"structure_inputs"`` for inverse folding generators (ProteinMPNN, LigandMPNN)
            - ``"prompts"`` for autoregressive generators (Evo2)

        pipeline (Literal['protein-hunter'] | None): Predefined conditioning pipeline.
            - ``"protein-hunter"``: Structure prediction -> inverse folding cycle.
              Requires an inverse_folding generator (ProteinMPNN or LigandMPNN).

        protein_hunter (ProteinHunterPipelineConfig | None): Configuration for protein-hunter pipeline.
            Only used when ``pipeline="protein-hunter"``.

        verbose (bool): Whether to print progress information. Default: ``False``.
        tracking_interval (int): Number of steps between progress snapshots.
        track_proposals (bool): Whether to record proposal sequences alongside accepted results.

    Note:
        - Pipeline-specific constraints:
          - ``protein-hunter`` requires an inverse_folding generator (ProteinMPNN, LigandMPNN)
        - Constraints are optional but if provided must be filter constraints
          (must have ``threshold`` set)

    Example:
        >>> config = CyclingOptimizerConfig(
        ...     num_steps=5,
        ...     num_results=4,
        ...     conditioning_param_name="structure_inputs",
        ...     pipeline="protein-hunter",
        ...     protein_hunter=ProteinHunterPipelineConfig(structure_tool="boltz2"),
        ... )
    """

    # Required parameters
    num_steps: int = ConfigField(
        ge=1,
        title="Number of Steps",
        description="Number of conditioning -> generation cycles to run.",
    )
    num_results: Optional[int] = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Candidate design trajectories for this optimizer. Overrides program-level count.",
        advanced=True,
    )
    conditioning_param_name: str = ConfigField(
        title="Conditioning Param Name",
        description="Generator sample() parameter name to pass conditioning data into.",
    )
    pipeline: Optional[Literal["protein-hunter"]] = ConfigField(
        default=None,
        title="Pipeline",
        description="Predefined conditioning pipeline. 'protein-hunter' uses structure prediction -> inverse folding.",
    )
    protein_hunter: Optional[ProteinHunterPipelineConfig] = ConfigField(
        default=None,
        title="Protein Hunter Config",
        description="Configuration for protein-hunter pipeline. Only used when pipeline='protein-hunter'.",
        depends_on={"field": "pipeline", "value": "protein-hunter"},
    )

    @model_validator(mode="after")
    def validate_pipeline_config(self):
        """Validate that pipeline-specific config is provided when pipeline is set."""
        if self.pipeline == "protein-hunter" and self.protein_hunter is None:
            # Auto-create default config if not provided
            self.protein_hunter = ProteinHunterPipelineConfig()
        return self

@optimizer(
    key="cycling",
    label="Cycling Optimizer",
    config=CyclingOptimizerConfig,
    description="Iterative optimizer that cycles between a conditioning function and generator",
    targets_single_segment=True,
)
@final
class CyclingOptimizer(Optimizer):
    """Cycling optimizer for iterative sequence refinement.

    A generalized optimizer that cycles between a user-defined conditioning function
    and a generator:

    1. Call conditioning function with current sequences (from result_sequences)
    2. Pass conditioning output to generator's sample() method (into proposal_sequences)
    3. Accept passing proposals into result_sequences (failed stay unchanged)
    4. Repeat for num_steps

    This enables flexible optimization patterns such as:
    - Protein Hunter: Structure prediction -> inverse folding cycles
    - Evo2 with feedback: Constraint-guided prompt modification -> generation cycles

    Attributes:
        target_segment: The segment being optimized.
        generator: The generator to use for sequence generation.
        conditioning_fn: User-defined function that produces conditioning data.
        conditioning_param_name: Generator sample() parameter name for conditioning data.
        num_steps: Number of cycles to run.
        num_results: Number of independent proposal trajectories.

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
        ...         num_results=4,
        ...         conditioning_param_name="structure_inputs",
        ...     ),
        ...     conditioning_fn=my_conditioning_fn,
        ... )
        >>> optimizer.run()

    Note:
        - Constraints are optional; if provided, must be filter constraints
          (have ``threshold`` set) - only passing proposals update result_sequences
    """

    config_class = CyclingOptimizerConfig
    _require_non_empty_constraints = False

    def __init__(
        self,
        target_segment: Segment,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: CyclingOptimizerConfig,
        conditioning_fn: Optional[Callable[[List[Sequence]], List[Any]]] = None,
        custom_logging: Optional[Callable[[int, tuple], None]] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the Cycling Optimizer.

        Args:
            target_segment (Segment): The specific Segment to optimize. Must belong to one
                of the constructs.
            constructs (list[Construct]): List of Construct objects. The target_segment must belong
                to one of these.
            generators (list[Generator]): List containing exactly one Generator.
            constraints (list[Constraint]): List of Constraint objects for filtering. Can be empty.
                If provided, all constraints must have ``threshold`` set (filter mode).
            config (CyclingOptimizerConfig): Configuration object with algorithm parameters.
            conditioning_fn (Callable[[list[Sequence]], list[Any]] | None): User-defined function that produces conditioning data.
                Signature: ``(sequences: List[Sequence]) -> List[Any]``
                Returns one conditioning item per proposal.
                Mutually exclusive with ``config.pipeline`` - use one or the other.
            custom_logging (Callable[[int, tuple], None] | None): Optional callback called at tracked steps (governed by ``tracking_interval``)
                with signature ``(step: int, segments: tuple) -> None``.
            clear_tool_cache (int | bool | list[str]): Cache management setting. (int) byte threshold,
                (bool) clear all, or (List[str]) specific tool names.

        Raises:
            ValueError: If generators list doesn't contain exactly one generator,
                target_segment is not in constructs, constraints don't have thresholds set,
                both conditioning_fn and pipeline are provided, or neither is provided,
                or num_results cannot be determined.
        """
        if len(generators) != 1:
            raise ValueError(f"CyclingOptimizer requires exactly one generator, got {len(generators)}.")
        generator = generators[0]
        generator.assign(target_segment)

        # Resolve conditioning_fn from pipeline or direct parameter
        conditioning_fn = _resolve_conditioning_fn(config, generator, conditioning_fn)

        self.config = config

        # Store for validation before super().__init__
        self.target_segment: Segment = target_segment
        self.generator: Generator = generator
        self.conditioning_fn = conditioning_fn
        self.conditioning_param_name: str = config.conditioning_param_name
        self.pipeline: Optional[str] = config.pipeline
        self.protein_hunter: Optional[ProteinHunterPipelineConfig] = config.protein_hunter

        super().__init__(
            constructs=constructs,
            generators=[generator],
            constraints=constraints,
            num_results=config.num_results,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
        )

        self.num_steps: int = config.num_steps

    def run(self) -> None:
        """Execute the cycling optimization loop."""
        self._prepare_run()

        if self.verbose:
            logger.info(f"CyclingOptimizer: {self.num_steps} steps, {self.num_proposals} proposals")

        # Track initial state only if we have meaningful scores (not all inf/nan)
        if any(math.isfinite(score) for score in self.energy_scores):
            self._save_progress_snapshot(time_step=0)

        for step in range(1, self.num_steps + 1):
            # 1. Condition from current best (result_sequences)
            current_sequences = list(self.target_segment.result_sequences)
            conditioning_data = self.conditioning_fn(current_sequences)

            # Validate conditioning_fn returned the correct number of items
            if len(conditioning_data) != self.num_proposals:
                raise ValueError(f"conditioning_fn returned {len(conditioning_data)} items, expected {self.num_proposals}. The conditioning function must return one conditioning item per proposal.")

            # 2. Generate proposals into proposal_sequences
            self.generator.sample(**{self.conditioning_param_name: conditioning_data})

            # 3. Evaluate and accept/reject
            if self.constraints:
                prev_energies = list(self.energy_scores)
                self.score_energy()
                for i in range(self.num_proposals):
                    # accept
                    if self._proposal_outcomes[i] == "accepted":
                        self.target_segment.result_sequences[i] = copy.deepcopy(self.target_segment.proposal_sequences[i])
                    else: # reject
                        self.energy_scores[i] = prev_energies[i]
            else:
                self.target_segment.result_sequences = [copy.deepcopy(seq) for seq in self.target_segment.proposal_sequences]
                self.energy_scores = [0] * self.num_proposals
                self._proposal_outcomes = ["accepted"] * self.num_proposals
                self._proposal_energy_scores = [0] * self.num_proposals

            if step % self.tracking_interval == 0 or step == self.num_steps:
                self._save_progress_snapshot(time_step=step)
                self._log_step_progress(step)

    def _validate_optimizer(self) -> None:
        """Validate cycling optimizer configuration.

        Extends base validation with cycling-specific checks:
        target_segment membership, callable conditioning_fn, valid
        conditioning_param_name, and filter-only constraints.
        """
        super()._validate_optimizer()
        self._validate_target_segment(self.target_segment)

        # Conditioning function checks
        if not callable(self.conditioning_fn):
            raise TypeError(f"conditioning_fn must be callable, got {type(self.conditioning_fn)}")

        sample_sig = inspect.signature(self.generator.sample)
        valid_params = set(sample_sig.parameters.keys()) - {"self"}
        if self.conditioning_param_name not in valid_params:
            raise ValueError(
                f"Generator {self.generator.__class__.__name__}.sample() does not accept parameter '{self.conditioning_param_name}'. "
                f"Valid parameters: {sorted(valid_params)}"
            )

        # All constraints must be filters
        for i, constraint in enumerate(self.constraints):
            if constraint.threshold is None:
                raise ValueError(
                    f"CyclingOptimizer only supports filter constraints. "
                    f"Constraint {i} ('{constraint.label}') has no threshold set."
                )

    def _log_step_progress(self, step: int) -> None:
        """Log step progress."""
        if self.verbose:
            num_accepted = self._proposal_outcomes.count("accepted")
            first_seq = self.target_segment.result_sequences[0].sequence
            logger.info(f"Step {step}/{self.num_steps}")
            logger.info(f"  Accepted: {num_accepted}/{self.num_proposals}")
            logger.info(f"  First seq: {first_seq}")
        if self.custom_logging:
            self.custom_logging(step, self.segments)
