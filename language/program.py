from typing import Any, Dict, List, Optional, Tuple

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
        track_step_size: Optional[int] = 10,
        **kwargs: Any,
    ) -> None:
        self.ebm: ProgramEnergyBasedModel = ebm
        self.track_step_size: int = track_step_size
        self.config: Dict[str, Any] = kwargs

    def run(self) -> Tuple[List[Tuple[str, ...]], List[float], List[int]]:
        """
        Run MCMC on an EBM generator while keeping track of state.
        """
        # Get initial state for printing
        initial_sequence = tuple(output.sequence for output in self.ebm.get_outputs())
        initial_energy = self.ebm.score_energy()

        print(f"Initial sequence: {initial_sequence}")
        print(f"Initial energy: {initial_energy:.4f}")

        # Run MCMC
        sequence_history, energy_history, steps_history = self.ebm.sample()

        # Get the final sequence.
        final_sequences = tuple(output.sequence for output in self.ebm.get_outputs())

        print(f"Final sequence: {final_sequences}")
        print(f"Final energy: {self.ebm.score_energy():.4f}")

        return sequence_history, energy_history, steps_history

