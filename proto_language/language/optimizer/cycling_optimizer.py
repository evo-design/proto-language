"""A generalized optimizer that iteratively runs a user-defined conditioning function.

and passes its output to a generator. Supports optional constraint filtering with
accept pattern for passing proposals.
"""

import copy
import logging
import random
from collections.abc import Callable
from typing import Any, Literal, final

from proto_tools.utils.tool_io import MissingAssetError
from pydantic import model_validator

from proto_language.base_config import BaseConfig, BaseOptimizerConfig, ConfigField
from proto_language.language.core import (
    Constraint,
    Construct,
    Generator,
    GeneratorInputType,
    Optimizer,
    Segment,
    Sequence,
)
from proto_language.language.optimizer.optimizer_registry import optimizer

logger = logging.getLogger(__name__)

# =============================================================================
# Predefined Pipelines
# =============================================================================

# AlphaFold2 is intentionally excluded — deterministic in our codepath
# (use_msa=False, dropout=False) would break cycling diversity. See
# proto-tools/notes/seeding.md.
CyclingStructureTool = Literal["boltz2", "chai1", "alphafold3"]


class ProteinHunterPipelineConfig(BaseConfig):
    """Configuration for the protein-hunter pipeline.

    The protein-hunter pipeline implements iterative structure prediction -> inverse folding
    cycles for de novo protein design (hallucination).

    Attributes:
        structure_tool (CyclingStructureTool): Structure prediction tool to use. One of "boltz2", "chai1", "alphafold3".
    """

    structure_tool: CyclingStructureTool = ConfigField(
        default="boltz2",
        title="Structure Tool",
        description="Structure prediction tool: 'boltz2', 'chai1', or 'alphafold3'.",
    )


def _create_protein_hunter_conditioning_fn(config: "CyclingOptimizerConfig") -> Callable[..., Any]:
    """Create protein hunter conditioning function (structure prediction -> inverse folding).

    The Protein Hunter algorithm predicts 3D structures from current sequences,
    then uses those structures to condition inverse folding for the next iteration.

    Args:
        config (CyclingOptimizerConfig): Constraint configuration controlling evaluation parameters.
    """
    from proto_tools import StructurePredictionComplex, predict_structures

    structure_tool = config.protein_hunter.structure_tool if config.protein_hunter else "boltz2"

    def _make_rng() -> random.Random | None:
        if config.seed is None:
            return None
        return random.Random(config.seed)  # noqa: S311 -- non-cryptographic

    # Mutable RNG container so `_reset_seed_state` can swap it; advances one seed per cycle.
    state = {"rng": _make_rng()}

    def conditioning_fn(sequences: list[Sequence]) -> list[Any]:
        # Hallucinated sequences have no homologs, so skip ColabFold MSA search.
        tool_config: dict[str, Any] = {"use_msa": False}
        rng = state["rng"]
        if rng is not None:
            tool_config["seed"] = rng.randint(0, 2**31 - 1)
        complexes = [StructurePredictionComplex(chains=[seq.sequence]) for seq in sequences]
        structures = predict_structures(complexes, structure_tool, tool_config).structures
        for seq, structure in zip(sequences, structures, strict=True):
            seq.structure = structure
        return structures  # type: ignore[no-any-return]

    def _reset_seed_state() -> None:
        state["rng"] = _make_rng()

    conditioning_fn._reset_seed_state = _reset_seed_state  # type: ignore[attr-defined]
    return conditioning_fn


# Pipeline registry: factory + required generator input_type.
CYCLING_PIPELINES: dict[str, dict[str, Any]] = {
    "protein-hunter": {
        "factory": _create_protein_hunter_conditioning_fn,
        "required_input_type": GeneratorInputType.STRUCTURE,
    },
}


# =============================================================================
# Predefined Pipeline Helpers
# =============================================================================


def _build_pipeline_conditioning_fn(config: "CyclingOptimizerConfig", generator: Generator) -> Callable[..., Any]:
    """Build a conditioning_fn from a registered pipeline; validate the generator's input_type."""
    assert config.pipeline is not None  # noqa: S101 -- mypy type narrowing; caller checks
    spec = CYCLING_PIPELINES[config.pipeline]
    required_input_type: GeneratorInputType = spec["required_input_type"]
    if generator.input_type != required_input_type:
        raise ValueError(
            f"Pipeline '{config.pipeline}' requires a generator with input_type={required_input_type!r}, "
            f"got {generator.input_type!r}."
        )
    return spec["factory"](config)  # type: ignore[no-any-return]


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

        pipeline (Literal['protein-hunter'] | None): Predefined conditioning pipeline.
            - ``"protein-hunter"``: Structure prediction -> inverse folding cycle.
              Requires an inverse_folding generator.

        protein_hunter (ProteinHunterPipelineConfig | None): Configuration for protein-hunter pipeline.
            Only used when ``pipeline="protein-hunter"``.

        verbose (bool): Whether to print progress information. Default: ``False``.
        tracking_interval (int): Number of steps between progress snapshots.
        track_proposals (bool): Whether to record proposal sequences alongside accepted results.

    Note:
        - Pipeline-specific constraints:
          - ``protein-hunter`` requires an inverse_folding generator
        - Constraints are optional but if provided must be filter constraints
          (must have ``threshold`` set)

    Example:
        >>> config = CyclingOptimizerConfig(
        ...     num_steps=5,
        ...     num_results=4,
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
    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Candidate design trajectories for this optimizer. Overrides program-level count.",
        advanced=True,
    )
    pipeline: Literal["protein-hunter"] | None = ConfigField(
        default=None,
        title="Pipeline",
        description="Predefined conditioning pipeline. 'protein-hunter' uses structure prediction -> inverse folding.",
    )
    protein_hunter: ProteinHunterPipelineConfig | None = ConfigField(
        default=None,
        title="Protein Hunter Config",
        description="Configuration for protein-hunter pipeline. Only used when pipeline='protein-hunter'.",
        depends_on={"field": "pipeline", "value": "protein-hunter"},
    )

    @model_validator(mode="after")
    def validate_pipeline_config(self) -> "CyclingOptimizerConfig":
        """Auto-create pipeline-specific sub-config when omitted."""
        if self.pipeline == "protein-hunter" and self.protein_hunter is None:
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
        generator: The generator to use for sequence generation. Its ``_sample()`` must
            accept the conditioning data as its first non-self positional argument.
        conditioning_fn: User-defined function that produces conditioning data.
        num_steps: Number of cycles to run.
        num_results: Number of independent proposal trajectories.

    Example:
        >>> def my_conditioning_fn(sequences):
        ...     return [process(seq) for seq in sequences]
        >>> optimizer = CyclingOptimizer(
        ...     target_segment=segment,
        ...     constructs=[construct],
        ...     generators=[generator],
        ...     constraints=[],
        ...     config=CyclingOptimizerConfig(num_steps=5, num_results=4),
        ...     conditioning_fn=my_conditioning_fn,
        ... )
        >>> optimizer.run()

    Note:
        - Constraints are optional; if provided, must be filter constraints
          (have ``threshold`` set) - only passing proposals update result_sequences
    """

    config_class = CyclingOptimizerConfig
    config: CyclingOptimizerConfig
    _require_non_empty_constraints = False

    def __init__(
        self,
        target_segment: Segment,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: CyclingOptimizerConfig,
        conditioning_fn: Callable[[list[Sequence]], list[Any]] | None = None,
        custom_logging: Callable[[int, tuple[Segment, ...]], None] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
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
                Signature: ``(sequences: List[Sequence]) -> List[Any]``. Returns one conditioning item per proposal.
                Mutually exclusive with ``config.pipeline`` — use one or the other.
            custom_logging (Callable[[int, tuple[Segment, ...]], None] | None): Optional callback called at tracked steps (governed by ``tracking_interval``)
                with signature ``(step: int, segments: tuple[Segment, ...]) -> None``.
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

        self.config = config

        # Mutex: exactly one of pipeline / conditioning_fn must be set.
        if (config.pipeline is None) == (conditioning_fn is None):
            raise ValueError(
                f"Specify exactly one of 'conditioning_fn' or 'pipeline'. "
                f"Available pipelines: {list(CYCLING_PIPELINES.keys())}"
            )
        if conditioning_fn is None:
            conditioning_fn = _build_pipeline_conditioning_fn(config, generator)

        # Store for validation before super().__init__
        self.target_segment: Segment = target_segment
        self.generator: Generator = generator
        self.conditioning_fn = conditioning_fn
        self.pipeline: str | None = config.pipeline
        self.protein_hunter: ProteinHunterPipelineConfig | None = config.protein_hunter

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
            seed=config.seed,
        )

        self.num_steps: int = config.num_steps

    def _reset_seed_state(self) -> None:
        # Base resets optimizer/generator/constraint RNGs; also reset the conditioning_fn's RNG if it exposes a hook.
        super()._reset_seed_state()
        reset = getattr(self.conditioning_fn, "_reset_seed_state", None)
        if callable(reset):
            reset()

    def run(self) -> None:
        """Execute the cycling optimization loop."""
        self._prepare_run()

        logger.info(
            f"CyclingOptimizer: {self.num_steps} steps, {self.num_proposals} proposals, "
            f"pipeline={self.pipeline!r}, "
            f"{len(self.constraints)} constraints (filter only)"
        )

        self._save_progress_snapshot(
            time_step=0,
            optimizer_metadata={
                "type": "cycling",
                "num_steps": self.num_steps,
                "num_results": self.num_results,
                "num_proposals": self.num_proposals,
                "pipeline": self.pipeline,
                "proposal_count": len(self._proposal_outcomes),
                "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
            },
        )

        for step in range(1, self.num_steps + 1):
            # 1. Condition from current best (result_sequences)
            current_sequences = list(self.target_segment.result_sequences)
            conditioning_data = self.conditioning_fn(current_sequences)

            if len(conditioning_data) != self.num_proposals:
                raise ValueError(
                    f"conditioning_fn returned {len(conditioning_data)} items, expected {self.num_proposals}."
                )

            # 2. Generate proposals — ``conditioning_data`` binds to the generator's first ``_sample()`` kwarg.
            #    MissingAssetError carve-out preserves the proto-tools skip hook.
            try:
                self.generator.sample(conditioning_data)
            except MissingAssetError:
                raise
            except Exception as exc:
                raise RuntimeError(f"CyclingOptimizer step {step}/{self.num_steps} failed") from exc

            # 3. Evaluate and accept/reject
            if self.constraints:
                prev_energies = list(self.energy_scores)
                self.score_energy()
                for i in range(self.num_proposals):
                    # accept
                    if self._proposal_outcomes[i] == "accepted":
                        self.target_segment.result_sequences[i] = copy.deepcopy(
                            self.target_segment.proposal_sequences[i]
                        )
                    else:  # reject
                        self.energy_scores[i] = prev_energies[i]
            else:
                self.target_segment.result_sequences = [
                    copy.deepcopy(seq) for seq in self.target_segment.proposal_sequences
                ]
                self.energy_scores = [0] * self.num_proposals
                self._proposal_outcomes = ["accepted"] * self.num_proposals
                self._proposal_energy_scores = [0] * self.num_proposals

            if step % self.tracking_interval == 0 or step == self.num_steps:
                self._save_progress_snapshot(
                    time_step=step,
                    optimizer_metadata={
                        "type": "cycling",
                        "num_steps": self.num_steps,
                        "num_results": self.num_results,
                        "num_proposals": self.num_proposals,
                        "pipeline": self.pipeline,
                        "proposal_count": len(self._proposal_outcomes),
                        "accepted_proposal_count": self._proposal_outcomes.count("accepted"),
                    },
                )
                self._log_step_progress(step)

    def _validate_optimizer(self) -> None:
        """Validate target segment membership, callable conditioning_fn, and filter-only constraints."""
        super()._validate_optimizer()
        self._validate_target_segment(self.target_segment)

        if not callable(self.conditioning_fn):
            raise TypeError(f"conditioning_fn must be callable, got {type(self.conditioning_fn)}")

        for i, constraint in enumerate(self.constraints):
            if constraint.threshold is None:
                raise ValueError(
                    f"CyclingOptimizer only supports filter constraints. "
                    f"Constraint {i} ('{constraint.label}') has no threshold set."
                )

    def _log_step_progress(self, step: int) -> None:
        """Log step progress as a multi-line INFO block."""
        logger.info(f"Step {step}/{self.num_steps}")
        filter_summary = self._format_filter_summary()
        if filter_summary is not None:
            logger.info(f"  filters: {filter_summary}")
        for line in self._format_scoring_lines():
            logger.info(f"  {line}")
        logger.info(f"  energy:  {self._format_energy_summary()}")
        num_accepted = self._proposal_outcomes.count("accepted")
        logger.info(f"  accepted {num_accepted}/{self.num_proposals} proposals")
        logger.debug(f"  original_seq[0]: {self.target_segment.result_sequences[0].sequence}")
        if self.custom_logging:
            self.custom_logging(step, self.segments)
