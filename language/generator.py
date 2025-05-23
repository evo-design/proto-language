from abc import ABC, abstractmethod
import numpy as np
import random
from typing import Any, List, Dict, Optional, Tuple
import copy
import sys

from .base import (
    ProgramGenerator,
    ProgramIterativeGenerator,
    ProgramSequence,
    ProgramConstraint,
)


class UniformMutationGenerator(ProgramGenerator):
    """
    A uniform proposal over DNA, RNA, or protein sequences.

    Initializes with a random sequence and samples a point mutation on each call to `sample()`.
    """

    def __init__(
        self,
        sequence_length: int,
        sequence_type: str = "dna",
    ) -> None:
        """
        Initializes the uniform proposal.

        Args:
            sequence_length (int): The length of the random sequence.
            sequence_type (str): The type of sequence ('dna', 'rna', or 'protein').

        Raises:
            ValueError: If the provided sequence type is not supported.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_type = sequence_type.lower()

        self.vocab: Optional[Set[str]]
        if self.sequence_type == "dna":
            self.vocab = "ACGT"
        elif self.sequence_type == "rna":
            self.vocab = "ACGU"
        elif self.sequence_type == "protein":
            self.vocab = "ACDEFGHIKLMNPQRSTVWY"
        else:
            raise ValueError(f"Sequence type {self.sequence_type} not supported.")

    def register(
        self,
        outputs: Optional[Tuple[ProgramSequence]] = None,
    ) -> Tuple[ProgramSequence]:
        """
        Initialize a random sequence.

        outputs (Optional[Tuple[ProgramSequence]]): Optional initialization of output
                                                   variables.
        Returns:
            Tuple[ProgramSequence]: Output sequence variables. These variables get updated
                                    in-place throughout generation.
        """
        self._is_initialized = True

        if outputs is None:
            random_sequence = "".join(
                random.choices(self.vocab, k=self.sequence_length)
            )
            self.outputs = (
                ProgramSequence(
                    sequence=random_sequence,
                    sequence_type=self.sequence_type,
                ),
            )
        else:
            if len(outputs) != 1:
                raise ValueError("Provided outputs must have one entry")
            if not isinstance(outputs[0], ProgramSequence):
                raise ValueError("Must provide a ProgramSequence")
            self.outputs = outputs

        return self.outputs

    def sample(self) -> None:
        """
        Introduces a mutation at a random position in the sequence.
        """
        if not self._is_initialized:
            self.register()

        mutated_index = random.randint(0, self.sequence_length - 1)
        current_sequence = self.outputs[0].sequence
        current_char = current_sequence[mutated_index]

        # Make sure the mutated character is different from the current one
        possible_mutations = [c for c in self.vocab if c != current_char]
        mutated_char = random.choice(possible_mutations)

        self.outputs[0].sequence = (
            current_sequence[:mutated_index]
            + mutated_char
            + current_sequence[mutated_index + 1 :]
        )


class Evo2Generator(ProgramGenerator):
    """
    Wraps Evo 2 generation for use in the programming language.
    """

    def __init__(
        self,
        prompt_seqs: List[str],
        evo2_type: str = 'evo2_7b',
        evo2_local_path: str = None,
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int = None,
        **kwargs,
    ) -> None:
        """
        Refer to https://github.com/arcinstitute/evo2 and https://github.com/Zymrael/vortex
        for documentation of sampling parameters for Evo 2.

        Notable details:
        - If all prompts are the same length, this can do batched generation.
        - Also supports cached generation for efficient sampling.
        """
        super().__init__(**kwargs)

        self.prompt_seqs = prompt_seqs
        self.evo2_type = evo2_type
        self.evo2_local_path = evo2_local_path
        self.n_tokens = n_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batched = batched
        self.cached_generation = cached_generation
        self.verbose = verbose
        self.force_prompt_threshold = force_prompt_threshold

    def register(self, *args: Any, **kwargs: Any) -> Tuple[ProgramSequence]:
        """
        Initialize empty sequences to be populated by `sample()`. The number of sequences
        registered equals the number of prompt sequences provided during initialization.

        Returns:
            Tuple[ProgramSequence]: A tuple with empty DNA sequences.
        """
        self._is_initialized = True

        from evo2 import Evo2  # Lazily import Evo 2.

        self.evo2_model = Evo2(
            model_name=self.evo2_type,
            local_path=self.evo2_local_path,
        )

        outputs: List[ProgramSequence] = []
        for _ in range(len(self.prompt_seqs)):
            outputs.append(ProgramSequence(sequence_type="dna"))
        self.outputs = tuple(outputs)

        return self.outputs

    def sample(self, *args: Any, **kwargs: Any) -> None:
        """
        Wrap the generation loop from Evo 2. Sequences generated by Evo 2 fill `self.outputs`.
        """
        if not self._is_initialized:
            self.register()

        from evo2 import Evo2  # Lazily import Evo 2.

        evo2_model = Evo2(
            model_name=self.evo2_type,
            local_path=self.evo2_local_path,
        )

        output = evo2_model.generate(
            prompt_seqs=self.prompt_seqs,
            n_tokens=self.n_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            batched=self.batched,
            cached_generation=self.cached_generation,
            verbose=self.verbose,
            force_prompt_threshold=self.force_prompt_threshold,
        )

        assert len(output.sequences) == len(
            self.outputs
        ), "Number of output sequences differs from the number of provided prompts."

        for idx, sequence in enumerate(output.sequences):
            self.outputs[idx].sequence = sequence


class BindCraftGenerator(ProgramGenerator):
    """
    TODO(@brianhie): Implement this.
    """

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)

    def register(self) -> Tuple[ProgramSequence]:
        self.outputs = ( ProgramSequence(sequence_type='protein'), )
        return self.outputs

    def sample(self) -> None:
        pass


class ESM2Generator(ProgramGenerator):
    """
    Uses ESM-2 to propose a protein sequence and mutations based on the ESM-2 logits.
    """

    def __init__(
        self,
        esm2_type: str = "esm2_t33_650M_UR50D",
        sequence_length: int = 100,
        temperature: float = 1.0,
        decoding_method: str = "entropy",
        top_k: int = 5,
    ):
        """
        Initializes the ESM-2 protein sequence proposal.

        Args:
            esm2_type (str): The ESM-2 model to use, refer to https://github.com/facebookresearch/esm
                             for a list of models.
            sequence_length (int): The length of sequence to generate.
            temperature (float): Sampling temperature.
            decoding_method (str): How to pick top-k positions to decode. Options are:
                - 'entropy': Pick positions with highest entropy after softmax.
                - 'max_logit': Pick positions based on the maximum logit value.
                - 'random': Pick positions uniformly at random.
            top_k (int): The number of positions to sample per iteration.
        """
        self.esm2_type = esm2_type
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.decoding_method = decoding_method
        self.top_k = top_k

        # Determine how to pick positions for sampling.
        if self.decoding_method == "entropy":

            def _decoding_func(logits: np.ndarray) -> np.ndarray:
                """
                Returns per-position entropy (so higher values mean higher entropy).
                """
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

                eps: float = 1e-12
                probabilities = np.clip(probabilities, eps, 1.0)

                return -np.sum(probabilities * np.log(probabilities), axis=-1)

            self._decoding_func = _decoding_func

        elif self.decoding_method == "max_logit":

            def _decoding_func(logits: np.ndarray) -> np.ndarray:
                """
                Returns the negative of the maximum logit value at each position.
                Values with high per-position max-logits are downweighted.
                """
                return -np.max(logits, axis=-1)

            self._decoding_func = _decoding_func

        else:

            def _decoding_func(logits: np.ndarray) -> np.ndarray:
                """
                Default to returning random per position scores.
                """
                return np.random.permutation(self.sequence_length)

            self._decoding_func = _decoding_func

    def _esm2_forward(self, sequence: str) -> np.ndarray:
        """
        Runs a forward pass on a sequence and returns the logits.
        """
        import torch

        _, _, batch_tokens = self.batch_converter([("protein1", sequence)])
        with torch.inference_mode():
            results = self.esm2_model(batch_tokens)
        logits = results["logits"].detach().cpu().numpy()

        return logits[0][1:-1]

    def _sample_logit(self, logits: np.ndarray, position: int) -> str:
        """
        Sample an amino acid at a given position into the logits.
        """
        if position < 0 or position >= logits.shape[0]:
            raise ValueError(
                f"Invalid position {position}, needs to be in [0, {logits.shape[0]})"
            )

        aa_idx = [
            self.alphabet.get_idx(tok)
            for tok in self.alphabet.standard_toks
            if tok not in "BJXZ"
        ]

        logits = np.array(logits[position][aa_idx], dtype=np.float64)
        logits = logits / max(self.temperature, 1e-8)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        index = np.random.choice(len(logits), p=probabilities)

        sampled_aa_idx = aa_idx[index]
        sampled_aa = self.alphabet.get_tok(sampled_aa_idx)

        return sampled_aa

    def register(self) -> Tuple[ProgramSequence]:
        """
        Randomly initialize a sequence using the logits from all mask tokens.
        """
        import torch

        self._is_initialized = True

        outputs: List[ProgramSequence] = [ProgramSequence(sequence_type="protein")]
        self.outputs = tuple(outputs)

        # Lazily import ESM-2.
        self.esm2_model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", self.esm2_type
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm2_model.eval()

        logits = self._esm2_forward("<mask>" * self.sequence_length)
        assert logits.shape[0] == self.sequence_length

        initial_sequence = "".join(
            [self._sample_logit(logits, pos) for pos in range(self.sequence_length)]
        )
        self.outputs[0].sequence = initial_sequence

        return self.outputs

    def sample(self) -> None:
        """
        Sample top-k positions and then sample a new amino acid at each of those positions.
        """
        from .utils import sample_k_weighted_no_replacement

        if not self._is_initialized:
            self.register()

        sequence = self.outputs[0].sequence

        logits = self._esm2_forward(sequence)

        position_scores = self._decoding_func(logits)  # Score positions.

        for idx in sample_k_weighted_no_replacement(position_scores, self.top_k):
            sequence = (
                sequence[:idx] + self._sample_logit(logits, idx) + sequence[idx + 1 :]
            )

        self.outputs[0].sequence = sequence


class ProgramMCMCGenerator(ProgramIterativeGenerator):
    """
    Metropolis-Hastings MCMC loop using generators to implement the proposal distribution and
    constraints to define the remainder of the energy function.
    """

    def __init__(
        self,
        generators: List[ProgramGenerator],
        constraints: List[ProgramConstraint],
        **hyperparameters: Any,
    ) -> None:
        """
        Initializes the MCMC generator. Wraps a list of generators and constraints.

        Args:
            generators (List[ProgramGenerator]): A list of generators. These must be registered.
            constraints (List[ProgramConstraint]): A list of constraints. The inputs to these
                                                   constraints must be the same objects produced by
                                                   generator registration.
            constraint_weights (Optional[List[float]]): Weights to use for each constraint. Must be the
                                                        same length as the `constraints` list.
            num_steps (Optional[int]): The number of steps to run on each call of `sample()`.
            temperature (Optional[float]): The Metropolis-Hastings sampling temperature.
            track_step_size (Optional[int]): The number of steps between tracking the sequence and energy.
            custom_logging (Optional[Callable[[int, Tuple[ProgramSequence]], None]]):
                A custom logging function to print out at each logged step.
                Takes as input the step number and the generator outputs.
            verbose (bool): If true, print out sampling information.
            **hyperparameters (Any): Keyword arguments representing the
                                     configuration and hyperparameters for the
                                     specific generator implementation.

        Raises:
            ValueError: Throws an error if there are mismatches between constraints/weights or if there
                        are problems in the configuration of generators and constraints.
        """
        super().__init__(**hyperparameters)
        self.generators = generators
        self.constraints = constraints
        self.constraint_weights: List[float] = hyperparameters.get(
            "constraint_weights",
            [1.0] * len(constraints),
        )
        self.num_steps: int = hyperparameters.get("num_steps", 1)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        self.track_step_size: int = hyperparameters.get("track_step_size", 10)
        self.custom_logging: Callable[[int, Tuple[ProgramSequence]], None]
        self.custom_logging = hyperparameters.get("custom_logging", None)
        self.verbose: bool = hyperparameters.get("verbose", True)

        if len(self.constraints) != len(self.constraint_weights):
            raise ValueError("Constraint weights must match number of constraints.")

        # Generators must already be registered, since their variables are hooked up to constraints.
        variable_ids = set()
        for generator in self.generators:
            if not generator._is_initialized:
                raise ValueError("Not all generators have been registered.")
            outputs = generator.get_outputs()
            for output in outputs:
                variable_ids.add(id(output))

        # All constraint inputs must be the same as generator outputs.
        for constraint in self.constraints:
            for input_ in constraint.inputs:
                if id(input_) not in variable_ids:
                    raise ValueError(
                        "Found a constraint not tied to a given generator."
                    )

    def register(self) -> Tuple[ProgramSequence]:
        """
        Because this generator wraps a list of pre-registered sub-generators, simply
        return an in-order tuple of these generators' outputs.

        Returns:
            Tuple[ProgramSequence]: Output sequence variables. These variables get updated
                                    in-place throughout generation.
        """
        self._is_initialized = True

        self.outputs = []
        for generator in self.generators:
            self.outputs += list(generator.get_outputs())
        self.outputs = tuple(self.outputs)

        return self.outputs

    def sample(self) -> List[Tuple[ProgramSequence]]:
        """
        Runs the MCMC sampling loop to update sequences in-place.

        Performs Metropolis-Hastings sampling by proposing changes from a randomly
        selected generator and accepting or rejecting based on the energy ratio.

        The temperature parameter controls the acceptance probability:
        - Higher temperature (>1.0): More likely to accept worse solutions (more exploration)
        - Lower temperature (<1.0): Less likely to accept worse solutions (more exploitation)
        - Temperature = 1.0: Standard Metropolis-Hastings behavior

        Returns:
            List[Tuple[ProgramSequence]]: A list of ProgramSequence tuple outputs at tracked steps with metadata stored in the ProgramSequence objects.
        """
        # Calculate and store initial energy score
        old_energy = self.score_energy_additive()
        for output in self.outputs:
            output._metadata["energy_score"] = old_energy
            output._metadata["num_step"] = 0

        # Initialize history tracking
        sequence_snapshot = tuple(copy.deepcopy(output) for output in self.outputs)
        sequence_history = [sequence_snapshot]

        # Execute one MCMC optimization step
        for step in range(1, self.num_steps + 1):
            step_temperature = (0.0001 / self.temperature) ** (step / self.num_steps)

            # 1. Pick a generator.
            generator = random.choice(self.generators)
            # Track old sequences x(t).
            old_seqs = [s.sequence for s in generator.get_outputs()]

            # 2. Sample x' from generator.
            generator.sample()
            # Evaluate new energy for x'.
            new_energy = self.score_energy_additive()

            # 3. Compute acceptance probability g(x') / g(x(t)) with temperature.
            #alpha = (new_energy / (old_energy + 1e-12)) ** (self.temperature)
            alpha = np.exp(-(new_energy - old_energy) / step_temperature)
            alpha = min(1.0, alpha)

            # 4. Accept/reject according to random number [0.0, 1.0).
            accept = random.random() < alpha

            if self.verbose and step % self.track_step_size == 0:
                print(
                    f"Iteration {step} | "
                    f"old energy: {old_energy}, "
                    f"new energy: {new_energy}, "
                    f"alpha: {alpha}, "
                    f"temperature: {step_temperature}, "
                    f"accept: {accept}"
                )
                if self.custom_logging:
                    self.custom_logging(step, self.get_outputs())
                sys.stdout.flush()

            if accept:
                old_energy = new_energy
            else:
                for seq_obj, old in zip(generator.get_outputs(), old_seqs):
                    seq_obj.sequence = old

            # Track sequence and energy periodically
            if step % self.track_step_size == 0:
                # Store the energy score
                for output in self.outputs:
                    output._metadata["energy_score"] = old_energy
                    output._metadata["num_step"] = step
                # Create deep copies of the sequence objects
                sequence_snapshot = tuple(
                    copy.deepcopy(output) for output in self.outputs
                )
                sequence_history.append(sequence_snapshot)

        # Return a dictionary with the tracked state information
        return sequence_history
