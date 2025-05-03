from copy import deepcopy
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd
from .base import ProgramSequence, ProgramGenerator, ProgramConstraint
from .sequence import ProgramDNASequence, ProgramRNASequence, ProgramProteinSequence
import random
import math

class Program:
    """
    Toy Metropolis-Hastings MCMC example using:
      - generator: list of ProgramGenerator instances (proposals)
      - constraints: list of ProgramConstraint instances (energies)
    """
    def __init__(self,
                 generators: List[ProgramGenerator],
                 constraints: List[ProgramConstraint],
                 **kwargs: Any,
    ) -> None:
        self.generators: List[ProgramGenerator] = generators
        self.constraints: List[ProgramConstraint] = constraints
        self.config: Dict[str, Any] = kwargs


    def score_energy(self) -> float:
        """
        Multiplicative energy
        """
        energy = 1.0
        for c in self.constraints:
            energy *= c.evaluate()
        return energy


    def sample(self,
               num_steps: int) -> List[ProgramSequence]:
        """
        Metropolis-Hastings MCMC algorithm
        """

        # register all generators and initialize their ProgramSequence outputs
        for g in self.generators:
            if not g._is_initialized:
                g.register()

        # calculate initial energy
        old_energy = self.score_energy()

        # MCMC optimization algorithm
        for t in range(num_steps):

            # 1. pick pick a generator
            generator = random.choice(self.generators)
            # track old sequences x(t)
            old_seqs = [s.sequence for s in generator.get_outputs()]

            # 2. sample x' from generator
            generator.sample()
            # evaluate new energy for x'
            new_energy = self.score_energy()

            # 3. compute acceptance probability g(x') / g(x(t))
            alpha = new_energy / (old_energy + 1e-12)
            alpha = min(1.0, alpha)

            # 4. accept/reject according to random number [0.0, 1.0)
            if random.random() > alpha:
                old_energy = new_energy
            else:
                for seq_obj, old in zip(generator.get_outputs(), old_seqs):
                    seq_obj.sequence = old
        # return the final sequences
        final_sequences = []
        for g in self.generators:
            final_sequences.extend(g.get_outputs())
        return final_sequences

