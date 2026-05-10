# Optimizer Implementation Templates

Complete templates for config class and optimizer class. Load this file on demand when implementing a new optimizer.

## Optimizer ABC Full Contract

```python
class Optimizer(ABC):
    @abstractmethod
    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        num_results: int | None,
        tracking_interval: int,
        track_proposals: bool,
        verbose: bool,
        proposals_per_result: int = 1,
        num_proposals: int | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
        custom_logging: Callable | None = None,
        seed: int | None = None,
    ) -> None:
        # Stores all parameters as instance attributes
        # Calls _validate_optimizer()
        # Key attributes after init:
        #   self.segments (property) — all segments from all constructs
        #   self.energy_scores — populated by score_energy()
        #   self.history — populated by _save_progress_snapshot()

    @abstractmethod
    def run(self) -> None:
        # Executes the optimization loop
        # Modifies segments' result_sequences and proposal_sequences
```

## Key Base Class Methods

### `score_energy(operation="add", filter_penalty=float("inf"))`

Evaluates ALL constraints on current `proposal_sequences`:

```python
# In your run() method:
self.score_energy(operation="add")      # Additive scoring (default)
self.score_energy(operation="multiply") # Multiplicative scoring

# After calling, self.energy_scores is populated:
# list[float] of length num_proposals
```

### `_initialize_sequence_pools()`

Sets up `proposal_sequences` from `result_sequences` with cycling:

```python
# If num_proposals > num_results, cycles through results to fill
# If num_proposals < num_results, takes first N
# Preserves diversity by round-robin assignment
```

### `_save_progress_snapshot(time_step)`

Saves current state to `self.history`:

```python
self._save_progress_snapshot(step)
# Appends: {"time_step": step, "constructs": [...], "energy_scores": [...]}
```

### `_validate_optimizer()`

Comprehensive validation called in `__init__`:
- Non-empty constructs, generators, constraints lists
- All generators assigned to segments
- No duplicate constraint labels
- All constraint inputs reference segments in constructs
- Generator segments exist in constructs

### State Management

```python
self._prepare_run()              # Reset history, prepare for fresh run
self._capture_initial_state()    # Snapshot state before run (for multi-run)
self._restore_initial_state()    # Restore to captured state
```

## Single-Segment Optimizer Pattern

If your optimizer only works with one segment (like BeamSearch or Cycling):

```python
# In the @optimizer decorator, set targets_single_segment=True:
@optimizer(
    key="my-optimizer",
    label="My Optimizer",
    config=MyOptimizerConfig,
    description="...",
    targets_single_segment=True,
)

# In __init__, add target_segment as the first parameter:
def __init__(
    self,
    target_segment: Segment,      # First parameter for single-segment optimizers
    constructs: list[Construct],
    generators: list[Generator],
    constraints: list[Constraint],
    config: MyOptimizerConfig,
    ...
) -> None:
```

This enables the `target_segment` field for single-segment selection.

## Config Class Template

File: `proto_language/language/optimizer/{name}_optimizer.py`

```python
import copy
import logging
from typing import Callable, final

from pydantic import model_validator

from proto_language.base_config import BaseOptimizerConfig, ConfigField
from proto_language.language.core import Construct, Constraint, Generator, Optimizer
from proto_language.language.optimizer.optimizer_registry import optimizer

logger = logging.getLogger(__name__)


class MyOptimizerConfig(BaseOptimizerConfig):
    """Configuration for MyOptimizer.

    Detailed description of the optimization algorithm and its parameters.

    Note: tracking_interval, track_proposals, and verbose are inherited
    from BaseOptimizerConfig — do NOT redeclare them here.
    """

    num_steps: int = ConfigField(
        ge=1,
        title="Num Steps",
        description="Total optimization iterations",
    )

    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Num Results",
        description="Number of top sequences to maintain. Overrides program-level num_results if set.",
        advanced=True,
    )

    # Conditional field example — only visible when a related field has a specific value:
    # cooling_rate: float = ConfigField(
    #     default=0.95,
    #     title="Cooling Rate",
    #     description="Temperature decay per step (simulated annealing)",
    #     gt=0.0,
    #     lt=1.0,
    #     depends_on={"field": "use_annealing"},
    # )

    @model_validator(mode="after")
    def validate_config(self):
        return self
```

## Optimizer Class Template

```python
@optimizer(
    key="my-optimizer",                         # Unique, kebab-case
    label="My Optimizer",                       # Human-readable name
    config=MyOptimizerConfig,                   # Config class
    description="Optimizes sequences using ...",# UI description
)
@final
class MyOptimizer(Optimizer):
    """Optimize sequences using MyAlgorithm.

    Detailed description of the algorithm.
    """

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: MyOptimizerConfig,
        custom_logging: Callable | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        self.config = config

        super().__init__(
            constructs=constructs,
            generators=generators,
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

    def run(self) -> None:
        """Execute the optimization loop."""
        self._prepare_run()

        # Initialize sequence pools
        self._initialize_sequence_pools()

        # Score initial state
        self.score_energy()
        self._save_progress_snapshot(0)

        for step in range(1, self.num_steps + 1):
            # 1. Prepare proposals from results
            self._initialize_sequence_pools()

            # 2. Apply generators (mutate proposals)
            for gen in self.generators:
                gen.sample()

            # 3. Score proposals
            self.score_energy()

            # 4. Update result_sequences with best proposals
            # NOTE: Each optimizer implements its own selection logic.
            # See MCMCOptimizer for MH acceptance, RejectionSamplingOptimizer for sorted insertion.
            self._update_results(step)

            # 5. Track progress (gated by tracking_interval)
            if step % self.tracking_interval == 0 or step == self.num_steps:
                self._save_progress_snapshot(step)
                if self.verbose:
                    best_score = min(self.energy_scores)
                    logger.info(f"Step {step}/{self.num_steps}: best={best_score:.4f}")

    def _update_results(self, step: int) -> None:
        """Update result_sequences with top proposals.

        This is optimizer-specific. Common patterns:
        - Greedy: sort by energy, take top num_results (see RejectionSamplingOptimizer._insert_into_results)
        - MCMC: MH acceptance criterion (see MCMCOptimizer._select_topk_with_mcmc_acceptance)
        """
        scored = list(zip(self.energy_scores, range(len(self.energy_scores))))
        scored.sort(key=lambda x: x[0])
        for seg in self.segments:
            new_results = []
            for _, idx in scored[:self.num_results]:
                new_results.append(copy.deepcopy(seg.proposal_sequences[idx]))
            seg.result_sequences = new_results
```

## `_update_results` Patterns

### Greedy (Rejection Sampling-style)
```python
def _update_results(self, step):
    scored = list(zip(self.energy_scores, range(len(self.energy_scores))))
    scored.sort(key=lambda x: x[0])
    for seg in self.segments:
        new_results = []
        for _, idx in scored[:self.num_results]:
            new_results.append(copy.deepcopy(seg.proposal_sequences[idx]))
        seg.result_sequences = new_results
```

### MCMC (Metropolis-Hastings acceptance)
```python
def _update_results(self, step):
    # Compare each proposal to its corresponding result sequence
    for i, (new_score, old_score) in enumerate(
        zip(self.energy_scores, self._previous_scores)
    ):
        delta = new_score - old_score
        if delta <= 0 or random.random() < math.exp(-delta / self.temperature):
            # Accept: copy proposal into results
            for seg in self.segments:
                seg.result_sequences[i] = copy.deepcopy(seg.proposal_sequences[i])
```

### Setup Helper for Tests

Every optimizer test file should have a `_setup_components()` helper:

```python
from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.core import Constraint, Construct, Segment
from proto_language.language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig, MaskingStrategy

def _setup_components(
    seq_length=10,
    num_results=5,
    num_steps=10,
):
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    gen = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=1)))
    gen.assign(segment)
    construct = Construct([segment])
    constraint = Constraint(
        inputs=[segment],
        function=gc_content_constraint,
        function_config=GCContentConfig(min_gc=40, max_gc=60),
    )
    config = MyOptimizerConfig(num_results=num_results, num_steps=num_steps)
    opt = MyOptimizer(
        constructs=[construct],
        generators=[gen],
        constraints=[constraint],
        config=config,
    )
    return opt, gen, constraint, segment
```
