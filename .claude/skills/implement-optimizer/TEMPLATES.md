# Optimizer Implementation Templates

Complete templates for config class and optimizer class. Load this file on demand when implementing a new optimizer.

## Config Class Template

File: `proto_language/language/optimizer/{name}_optimizer.py`

```python
from __future__ import annotations

import copy
import logging
from typing import Callable, List, Optional, final

from pydantic import model_validator

from proto_language.base_config import BaseOptimizerConfig, ConfigField
from proto_language.language.core import Construct, Constraint, Generator, Optimizer
from proto_language.language.optimizer.optimizer_registry import optimizer

logger = logging.getLogger(__name__)


class MyOptimizerConfig(BaseOptimizerConfig):
    """Configuration for MyOptimizer.

    Detailed description of the optimization algorithm and its parameters.

    Note: tracking_interval, track_candidates, and verbose are inherited
    from BaseOptimizerConfig — do NOT redeclare them here.
    """

    num_steps: int = ConfigField(
        ge=1,
        title="Num Steps",
        description="Total optimization iterations",
    )

    num_results: Optional[int] = ConfigField(
        default=None,
        ge=1,
        title="Num Results",
        description="Number of top sequences to maintain. Overrides program-level num_results if set.",
        advanced=True,
    )

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
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: MyOptimizerConfig,
        custom_logging: Optional[Callable] = None,
        clear_tool_cache: int | bool | List[str] = 100 * 1024 * 1024,
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
            track_candidates=config.track_candidates,
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
            # 1. Prepare candidates from selected
            self._initialize_sequence_pools()

            # 2. Apply generators (mutate candidates)
            for gen in self.generators:
                gen.sample()

            # 3. Score candidates
            self.score_energy()

            # 4. Update selected_sequences with best candidates
            # NOTE: Each optimizer implements its own selection logic.
            # See MCMCOptimizer for MH acceptance, TopKOptimizer for sorted insertion.
            self._update_selected(step)

            # 5. Track progress (gated by tracking_interval)
            if step % self.tracking_interval == 0 or step == self.num_steps:
                self._save_progress_snapshot(step)
                if self.verbose:
                    best_score = min(self.energy_scores)
                    logger.info(f"Step {step}/{self.num_steps}: best={best_score:.4f}")

    def _update_selected(self, step: int) -> None:
        """Update selected_sequences with top candidates.

        This is optimizer-specific. Common patterns:
        - Greedy: sort by energy, take top num_results (see TopKOptimizer._insert_into_topk)
        - MCMC: MH acceptance criterion (see MCMCOptimizer._select_topk_with_mcmc_acceptance)
        """
        scored = list(zip(self.energy_scores, range(len(self.energy_scores))))
        scored.sort(key=lambda x: x[0])
        for seg in self.segments:
            new_selected = []
            for _, idx in scored[:self.num_results]:
                new_selected.append(copy.deepcopy(seg.candidate_sequences[idx]))
            seg.selected_sequences = new_selected
```

## `_update_selected` Patterns

### Greedy (TopK-style)
```python
def _update_selected(self, step):
    scored = list(zip(self.energy_scores, range(len(self.energy_scores))))
    scored.sort(key=lambda x: x[0])
    for seg in self.segments:
        new_selected = []
        for _, idx in scored[:self.num_results]:
            new_selected.append(copy.deepcopy(seg.candidate_sequences[idx]))
        seg.selected_sequences = new_selected
```

### MCMC (Metropolis-Hastings acceptance)
```python
def _update_selected(self, step):
    # Compare each candidate to its corresponding selected sequence
    for i, (new_score, old_score) in enumerate(
        zip(self.energy_scores, self._previous_scores)
    ):
        delta = new_score - old_score
        if delta <= 0 or random.random() < math.exp(-delta / self.temperature):
            # Accept: copy candidate into selected
            for seg in self.segments:
                seg.selected_sequences[i] = copy.deepcopy(seg.candidate_sequences[i])
```

### Setup Helper for Tests

Every optimizer test file should have a `_setup_components()` helper:

```python
from proto_language.language.constraint import gc_content_constraint
from proto_language.language.constraint.sequence_composition.gc_content_constraint import GCContentConfig
from proto_language.language.core import Constraint, Construct, Segment
from proto_language.language.generator import UniformMutationGenerator, UniformMutationGeneratorConfig

def _setup_components(
    seq_length=10,
    num_results=5,
    num_steps=10,
):
    segment = Segment(sequence="A" * seq_length, sequence_type="dna")
    gen = UniformMutationGenerator(UniformMutationGeneratorConfig(num_mutations=1))
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
