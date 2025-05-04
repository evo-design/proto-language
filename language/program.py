from typing import Any, Dict, List, Optional

from .base import ProgramEnergyBasedModel, ProgramSequence


class Program:
    """
    Small wrapper class that samples from an EBM-typed generator. Can keep track of state across
    multiple calls to `sample()`.

    TODO(@brianhie): Decide if this class is even needed.
    """
    def __init__(
        self,
        ebm: ProgramEnergyBasedModel,
        num_mcmc_steps: Optional[int] = 100_000,
        track_step_size: Optional[int] = 10,
        **kwargs: Any,
    ) -> None:
        self.ebm: ProgramEnergyBasedModel = ebm
        self.num_mcmc_steps: int = num_steps
        self.track_step_size: int = track_step_size
        self.config: Dict[str, Any] = kwargs

    def run(self) -> List[ProgramSequence]:
        """
        Run MCMC on an EBM generator while keeping track of state.
        """
        sequence_history = [sequence.sequence]
        energy_history = [program.score_energy()]
        steps_history = [0]

        print(f"Initial sequence: {sequence_history[0]}")
        print(f"Initial energy: {energy_history[0]:.4f}")

        for step in range(1, self.num_mcmc_steps + 1):
            # Run a single MCMC step.
            self.ebm.sample()

            # Track sequence and energy periodically.
            if step % self.track_step_size == 0:
                current_sequence = self.ebm.get_outputs()[0].sequence
                sequence_history.append(current_sequence)
                energy_history.append(self.ebm.score_energy())
                steps_history.append(step)

        # Get the final sequence.
        final_sequence = self.ebm.get_outputs()[0]

        print(f"Final sequence: {final_sequence}")
        print(f"Final energy: {self.ebm.score_energy():.4f}")



