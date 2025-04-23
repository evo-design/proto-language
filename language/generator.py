from abc import ABC, abstractmethod
from language.sequence import ProgramSequence
from typing import Any, List, Dict


class ProgramGenerator(ABC):
    """
    Abstract base class for program generation algorithms (samplers).

    Defines the interface for initializing and sampling sequences.
    Subclasses implement specific generation strategies (e.g., MCMC, autoregressive decoding).

    Subclasses must implement both initialize() and sample().
    """
    def __init__(self, **hyperparameters: Any) -> None:
        """
        Initializes the generator with specific hyperparameters.

        Args:
            **hyperparameters (Any): Keyword arguments representing the
                                     configuration and hyperparameters for the
                                     specific generator implementation.
        """
        self.hyperparameters: Dict[str, Any] = hyperparameters
        self._is_initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initializes the internal state of the generator's sampler.

        This method should be distinguished from the constructor `__init__()` as the place
        to do compute-intensive initialization of sampling state. Simple hyperparameter state
        can be saved in the constructor.
        """
        self._is_initialized = True
        raise NotImplementedError("Subclasses must implement the initialize method.")

    @abstractmethod
    def sample(self) -> List[ProgramSequence]:
        """
        Generates and returns a list of ProgramSequence instances based on the
        generator's internal state and hyperparameters.

        Returns:
            List[ProgramSequence]: A list of newly generated sequences.

        Raises:
            RuntimeError: If called before initialize().
        """
        if not self._is_initialized:
            raise RuntimeError(f"Generator {self.__class__.__name__} has not been initialized. Call initialize() first.")
        raise NotImplementedError("Subclasses must implement the sample method.")


class MCMCGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        self.num_steps: int = hyperparameters.get("num_steps", 1000)
        self.num_samples: int = hyperparameters.get("num_samples", 100)


class Evo2Generator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.num_generations: int = hyperparameters.get("num_generations", 100)
        self.population_size: int = hyperparameters.get("population_size", 100)
        self.mutation_rate: float = hyperparameters.get("mutation_rate", 0.1)
        

class SemanticMiningGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.num_steps: int = hyperparameters.get("num_steps", 1000)
        self.num_samples: int = hyperparameters.get("num_samples", 100)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        
        
class BindCraftGenerator(ProgramGenerator):
    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.num_steps: int = hyperparameters.get("num_steps", 1000)
        self.num_samples: int = hyperparameters.get("num_samples", 100)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
