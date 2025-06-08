import numpy as np
import random
from typing import Any, List, Optional, Tuple, Callable
import copy
import sys

from .base import *


class UniformMutationGenerator(ProgramGenerator):
    """
    A sequence generator that proposes random point mutations.

    This generator initializes with a random sequence and samples single-nucleotide
    or amino acid mutations on each call to sample(). It's useful as a simple baseline
    for evolutionary algorithms and MCMC sampling.

    The generator maintains a uniform probability distribution over valid characters
    for the sequence type, excluding the current character at each position to
    ensure mutations always change the sequence.

    Examples:
        Creating a DNA mutation generator:
        >>> gen = UniformMutationGenerator(
        ...     sequence_length=100,
        ...     sequence_type=SequenceType.DNA,
        ...     batch_size=5
        ... )
        >>> batches = gen.register()
        >>> gen.sample()  # Mutates one position per sequence
        
        Using with MCMC:
        >>> mcmc = ProgramMCMCGenerator(
        ...     generators=[gen],
        ...     constraints=[gc_constraint],
        ...     sequence_order=((batches[0],),)
        ... )
    """

    def __init__(
        self,
        sequence_length: int,
        sequence_type: SequenceType = SequenceType.DNA,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the uniform mutation generator.

        Args:
            sequence_length: Length of sequences to generate and mutate.
            sequence_type: Type of biological sequence (DNA, RNA, or PROTEIN).
            batch_size: Number of sequence variants to maintain simultaneously.

        Raises:
            ValueError: If the sequence_type is not supported.
        """
        super().__init__(batch_size=batch_size)
        self.sequence_length = sequence_length
        self.sequence_type = sequence_type

        if self.sequence_type == SequenceType.DNA:
            self.vocab = "ACGT"
        elif self.sequence_type == SequenceType.RNA:
            self.vocab = "ACGU"
        elif self.sequence_type == SequenceType.PROTEIN:
            self.vocab = "ACDEFGHIKLMNPQRSTVWY"
        else:
            raise ValueError(f"Sequence type {self.sequence_type} not supported.")

    def register(
        self,
        outputs: Optional[Tuple[BatchedProgramSequence]] = None,
    ) -> Tuple[BatchedProgramSequence]:
        """
        Initialize sequence variables, either randomly or from provided sequences.

        If no sequences are provided, generates random sequences of the specified
        length using uniform sampling from the vocabulary.

        Args:
            outputs: Optional pre-initialized BatchedProgramSequence objects.
                    If None, random sequences will be generated.

        Returns:
            Tuple containing a single BatchedProgramSequence with sequence variants
            that will be modified in-place during sampling.

        Raises:
            ValueError: If outputs is provided but has incorrect structure.
        """
        self._is_initialized = True

        if outputs is None:
            random_sequence = "".join(
                random.choices(self.vocab, k=self.sequence_length)
            )
            sequence_batch = BatchedProgramSequence(
                ProgramSequence(
                    sequence=random_sequence,
                    sequence_type=self.sequence_type,
                ) for i in range(self.batch_size)
            )
            self._generator_outputs = (sequence_batch,)


        else:
            if len(outputs) != 1:
                raise ValueError("Provided outputs must have one entry")
            if not isinstance(outputs[0], BatchedProgramSequence):
                raise ValueError("Must provide a BatchedProgramSequence")
            self._generator_outputs = outputs

        return self._generator_outputs

    def sample(self) -> None:
        """
        Introduce a random point mutation in each sequence.

        For each sequence in the batch, selects a random position and replaces
        the character with a different random character from the vocabulary.
        Ensures that mutations always change the sequence (no silent mutations).
        """
        if not self._is_initialized:
            self.register()

        for i in range(self.batch_size):
            mutated_index = random.randint(0, self.sequence_length - 1)
            current_sequence = self._generator_outputs[0][i].sequence
            current_char = current_sequence[mutated_index]

            # Make sure the mutated character is different from the current one
            possible_mutations = [c for c in self.vocab if c != current_char]
            mutated_char = random.choice(possible_mutations)

            self._generator_outputs[0][i].sequence = (
                current_sequence[:mutated_index]
                + mutated_char
                + current_sequence[mutated_index + 1 :]
            )


class Evo2Generator(ProgramGenerator):
    """
    A sequence generator that uses the Evo2 genome language model for DNA sequence generation.
    
    This generator wraps the Evo2 model to provide autoregressive sequence generation
    from prompt sequences. It supports batched generation, temperature control, and
    model caching for efficient reuse across multiple generator instances.
    
    The generator can handle single prompts (replicated across batch) or multiple
    prompts (one per batch element), with automatic model instance sharing between
    generators that use the same model configuration.
    
    Examples:
        Basic DNA generation:
        >>> gen = Evo2Generator(
        ...     prompt_seqs=["+~GA"],
        ...     evo2_type="evo2_7b",
        ...     n_tokens=1000,
        ...     temperature=0.8,
        ...     batch_size=5
        ... )
        >>> batches = gen.register()
        >>> gen.sample()  # Generates sequences from prompts
        
        Custom model with local weights:
        >>> gen = Evo2Generator(
        ...     prompt_seqs=["+~GA", "+~GC"],
        ...     evo2_type="evo2_7b_phage",
        ...     evo2_local_path="/path/to/weights.pt",
        ...     batch_size=2
        ... )
    """
    
    # Class-level cache for sharing model instances
    _model_cache = {}

    def __init__(
        self,
        prompt_seqs: List[str],
        evo2_type: str = 'evo2_7b',
        evo2_local_path: Optional[str] = None,
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: Optional[int] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize the Evo2 generator with model configuration and sampling parameters.

        For detailed documentation of Evo2 sampling parameters, refer to:
        https://github.com/arcinstitute/evo2 and https://github.com/Zymrael/vortex

        Args:
            prompt_seqs: List of prompt sequences to start generation from.
                        Single prompt gets replicated batch_size times, or provide
                        one prompt per batch element.
            evo2_type: Name of the Evo2 model variant to use.
            evo2_local_path: Optional path to local model weights file.
            n_tokens: Number of tokens to generate after each prompt.
            temperature: Sampling temperature for nucleus sampling.
            top_k: Top-k parameter for sampling.
            top_p: Top-p (nucleus) parameter for sampling.
            batched: Whether to use batched generation for efficiency.
            cached_generation: Whether to cache model states for faster sampling.
            verbose: Verbosity level for generation logging.
            force_prompt_threshold: Optional threshold for forcing prompt continuation.
            batch_size: Number of sequences to generate simultaneously.
            **kwargs: Additional arguments passed to parent class.

        Note:
            Model instances are automatically shared between generators with the same
            evo2_type and evo2_local_path to save memory and initialization time.
        """
        super().__init__(batch_size=batch_size, **kwargs)

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(prompt_seqs) == 1:
            self.prompt_seqs = prompt_seqs * batch_size
        else:
            assert len(prompt_seqs) == batch_size, f"Multiple prompts ({len(prompt_seqs)}) must equal batch_size ({batch_size})"
            assert len(set(len(seq) for seq in prompt_seqs)) == 1, f"All prompts must have same length, got: {[len(seq) for seq in prompt_seqs]}"
            self.prompt_seqs = prompt_seqs

        self.batch_size = batch_size
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

    def _get_model_key(self) -> str:
        """
        Generate a unique key for model caching based on configuration.
        
        Returns:
            String key uniquely identifying this model configuration.
        """
        return f"{self.evo2_type}:{self.evo2_local_path}"

    def register(
        self, 
        outputs: Optional[Tuple[BatchedProgramSequence]] = None,
    ) -> Tuple[BatchedProgramSequence]:
        """
        Initialize empty DNA sequences and load the Evo2 model.

        Creates BatchedProgramSequence objects that will be populated by sample().
        Loads the Evo2 model from cache if available, otherwise initializes a new
        instance and caches it for future use.

        Args:
            outputs: Optional pre-initialized BatchedProgramSequence objects.
                    If None, empty sequences will be created.

        Returns:
            Tuple containing a single BatchedProgramSequence with empty DNA sequences
            that will be filled during sampling.

        Raises:
            ValueError: If outputs is provided but has incorrect structure.
        """
        self._is_initialized = True

        # Check if model is already cached
        model_key = self._get_model_key()
        if model_key not in self._model_cache:
            from evo2 import Evo2  # Lazily import Evo 2.

            print(f"Loading new Evo2 model with key: {model_key}")
            self._model_cache[model_key] = Evo2(
                model_name=self.evo2_type,
                local_path=self.evo2_local_path,
            )
        else:
            print(f"Using cached Evo2 model with key: {model_key}")

        # Use the cached model
        self.evo2_model = self._model_cache[model_key]

        if outputs is None:
            # Create one BatchedProgramSequence containing all prompt sequences
            sequence_batch = BatchedProgramSequence(
                ProgramSequence(sequence_type=SequenceType.DNA) 
                for _ in range(self.batch_size)
            )
            self._generator_outputs = (sequence_batch,)
        else:
            if len(outputs) != 1:
                raise ValueError("Provided outputs must have one entry")
            if not isinstance(outputs[0], BatchedProgramSequence):
                raise ValueError("Must provide a BatchedProgramSequence")
            self._generator_outputs = outputs

        return self._generator_outputs

    @classmethod
    def clear_model_cache(cls):
        """
        Clear the model cache to free GPU memory.
        
        Call this method to force reloading of models if you need to free memory
        or switch to different model configurations.
        """
        cls._model_cache.clear()

    @classmethod
    def get_cached_models(cls):
        """
        Get information about currently cached models.
        
        Returns:
            List of model keys currently stored in the cache.
        """
        return list(cls._model_cache.keys())

    def sample(self, *args: Any, **kwargs: Any) -> None:
        """
        Generate sequences using the Evo2 model and update _generator_outputs.

        Uses the Evo2 model to generate continuations from the prompt sequences,
        updating the sequences in the BatchedProgramSequence in-place.

        Args:
            *args: Unused positional arguments for compatibility.
            **kwargs: Unused keyword arguments for compatibility.

        Raises:
            RuntimeError: If called before register().
            AssertionError: If number of generated sequences doesn't match prompts.
        """
        if not self._is_initialized:
            self.register()

        output = self.evo2_model.generate(
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

        assert len(output.sequences) == len(self.prompt_seqs), \
            "Number of output sequences differs from the number of provided prompts."

        # Update sequences in the BatchedProgramSequence
        for idx, sequence in enumerate(output.sequences):
            self._generator_outputs[0][idx].sequence = sequence


class BindCraftGenerator(ProgramGenerator):
    """
    A placeholder generator for the BindCraft protein design method.
    
    This generator is currently a stub implementation and needs to be completed
    with the actual BindCraft integration. It will be used for protein sequence
    generation with binding specificity constraints.
    
    Note:
        This is a TODO item - the implementation needs to be completed.
    """

    def __init__(self, batch_size: int = 1, **hyperparameters: Any) -> None:
        """
        Initialize the BindCraft generator.

        Args:
            batch_size: Number of sequence variants to generate simultaneously.
            **hyperparameters: Configuration parameters for BindCraft (to be defined).
        """
        super().__init__(batch_size=batch_size, **hyperparameters)

    def register(self, outputs: Optional[Tuple[BatchedProgramSequence]] = None) -> Tuple[BatchedProgramSequence]:
        """
        Initialize empty protein sequences for BindCraft generation.

        Args:
            outputs: Optional pre-initialized BatchedProgramSequence objects.
                    If None, empty protein sequences will be created.

        Returns:
            Tuple containing a BatchedProgramSequence with protein sequences.
            
        Raises:
            ValueError: If outputs is provided but has incorrect structure.
        """
        self._is_initialized = True
        
        if outputs is None:
            sequence_batch = BatchedProgramSequence(
                ProgramSequence(sequence_type=SequenceType.PROTEIN)
                for _ in range(self.batch_size)
            )
            self._generator_outputs = (sequence_batch,)
        else:
            if len(outputs) != 1:
                raise ValueError("Provided outputs must have one entry")
            if not isinstance(outputs[0], BatchedProgramSequence):
                raise ValueError("Must provide a BatchedProgramSequence")
            self._generator_outputs = outputs
            
        return self._generator_outputs

    def sample(self) -> None:
        """
        Generate protein sequences using BindCraft.
        
        Note:
            Currently a stub - implementation needed.
        """
        pass


class ESM2Generator(ProgramGenerator):
    """
    A protein sequence generator using the ESM-2 language model.

    This generator uses the ESM-2 protein language model to propose sequences and
    mutations based on the model's logits. It supports various decoding strategies
    for selecting positions to mutate and uses temperature-controlled sampling
    for amino acid selection.

    The generator iteratively selects high-uncertainty positions and samples
    new amino acids, making it suitable for protein design applications where
    gradual refinement is desired.

    Examples:
        Basic protein generation:
        >>> gen = ESM2Generator(
        ...     esm2_type="esm2_t33_650M_UR50D",
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5,
        ...     batch_size=3
        ... )
        >>> batches = gen.register()  # Creates random initial sequences
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(
        self,
        esm2_type: str = "esm2_t33_650M_UR50D",
        sequence_length: int = 100,
        temperature: float = 1.0,
        decoding_method: str = "entropy",
        top_k: int = 5,
        batch_size: int = 1,
        **kwargs,
    ):
        """
        Initialize the ESM-2 generator with model and sampling configuration.

        Args:
            esm2_type: ESM-2 model variant to use. See Facebook ESM repository
                      for available models.
            sequence_length: Length of protein sequences to generate.
            temperature: Sampling temperature for amino acid selection.
            decoding_method: Strategy for selecting positions to sample:
                - 'entropy': Choose positions with highest prediction entropy
                - 'max_logit': Choose positions with highest maximum logits
                - 'random': Choose positions randomly
            top_k: Number of positions to sample per iteration.
            batch_size: Number of sequences to generate simultaneously.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(batch_size, **kwargs)
        self.esm2_type = esm2_type
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.decoding_method = decoding_method
        self.top_k = top_k
        self.batch_size = batch_size

        # Determine how to pick positions for sampling.
        if self.decoding_method == "entropy":

            def _decoding_func(logits: np.ndarray) -> np.ndarray:
                """
                Calculate per-position entropy for position selection.
                
                Args:
                    logits: Model logits of shape (seq_len, vocab_size).
                    
                Returns:
                    Per-position entropy values (higher = more uncertain).
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
                Calculate negative max logits for position selection.
                
                Args:
                    logits: Model logits of shape (seq_len, vocab_size).
                    
                Returns:
                    Negative max logit values (prioritizes uncertain positions).
                """
                return -np.max(logits, axis=-1)

            self._decoding_func = _decoding_func

        else:

            def _decoding_func(logits: np.ndarray) -> np.ndarray:
                """
                Generate random scores for position selection.
                
                Args:
                    logits: Model logits (unused for random selection).
                    
                Returns:
                    Random permutation of position indices.
                """
                return np.random.permutation(self.sequence_length)

            self._decoding_func = _decoding_func

    def _esm2_forward(self, sequence: str) -> np.ndarray:
        """
        Run a forward pass through ESM-2 and return logits.

        Args:
            sequence: Protein sequence to process.

        Returns:
            Logits array of shape (seq_len, vocab_size) for the sequence,
            excluding special start/end tokens.
        """
        import torch

        _, _, batch_tokens = self.batch_converter([("protein1", sequence)])
        with torch.inference_mode():
            results = self.esm2_model(batch_tokens)
        logits = results["logits"].detach().cpu().numpy()

        return logits[0][1:-1]

    def _sample_logit(self, logits: np.ndarray, position: int) -> str:
        """
        Sample an amino acid at a specific position using temperature-controlled sampling.

        Args:
            logits: Model logits for the entire sequence.
            position: Position index to sample at.

        Returns:
            Single-letter amino acid code for the sampled residue.

        Raises:
            ValueError: If position is out of bounds.
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

    def register(
        self,
        outputs: Optional[Tuple[BatchedProgramSequence]] = None,
    ) -> Tuple[BatchedProgramSequence]:
        """
        Initialize a random protein sequence using ESM-2 logits.

        Creates initial sequences by running ESM-2 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If sequences are provided, they will be used as starting points.

        Args:
            outputs: Optional pre-initialized BatchedProgramSequence objects.
                    If None, random sequences will be generated.

        Returns:
            Tuple containing a single BatchedProgramSequence with protein sequences
            that will be modified in-place during sampling.

        Raises:
            ValueError: If outputs is provided but has incorrect structure.
        """
        import torch

        self._is_initialized = True

        # Lazily import ESM-2.
        self.esm2_model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", self.esm2_type
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm2_model.eval()

        if outputs is None:
            # Generate initial sequences using mask tokens
            logits = self._esm2_forward("<mask>" * self.sequence_length)
            assert logits.shape[0] == self.sequence_length

            initial_sequence = "".join(
                [self._sample_logit(logits, pos) for pos in range(self.sequence_length)]
            )
            
            sequence_batch = BatchedProgramSequence(
                ProgramSequence(
                    sequence=initial_sequence,
                    sequence_type=SequenceType.PROTEIN,
                ) for _ in range(self.batch_size)
            )
            self._generator_outputs = (sequence_batch,)
        else:
            if len(outputs) != 1:
                raise ValueError("Provided outputs must have one entry")
            if not isinstance(outputs[0], BatchedProgramSequence):
                raise ValueError("Must provide a BatchedProgramSequence")
            self._generator_outputs = outputs

        return self._generator_outputs

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM-2 logits, 
        selects top-k positions based on the decoding method, and samples new amino acids 
        at those positions.
        """
        from .utils import sample_k_weighted_no_replacement

        if not self._is_initialized:
            self.register()

        for i in range(self.batch_size):
            sequence = self._generator_outputs[0][i].sequence

            logits = self._esm2_forward(sequence)

            position_scores = self._decoding_func(logits)  # Score positions.

            for idx in sample_k_weighted_no_replacement(position_scores, self.top_k):
                sequence = (
                    sequence[:idx] + self._sample_logit(logits, idx) + sequence[idx + 1 :]
                )

            self._generator_outputs[0][i].sequence = sequence


class ProgramMCMCGenerator(ProgramIterativeGenerator):
    """
    Metropolis-Hastings MCMC generator for constraint-driven sequence optimization.

    This generator implements a Metropolis-Hastings sampling algorithm that uses
    multiple sub-generators as proposal distributions and constraints to define
    the energy function. It's designed for iterative sequence refinement where
    proposals are accepted or rejected based on energy improvements.

    The generator supports temperature annealing, multiple constraints with weights,
    and flexible sequence concatenation for complex multi-part designs.

    Examples:
        Basic MCMC optimization:
        >>> mcmc = ProgramMCMCGenerator(
        ...     generators=[mutation_gen, crossover_gen],
        ...     constraints=[gc_constraint, length_constraint],
        ...     sequence_order=((batch1,), (batch2,)),
        ...     num_steps=1000,
        ...     temperature=1.0
        ... )
        >>> history = mcmc.sample()
        
        Multi-constraint optimization:
        >>> mcmc = ProgramMCMCGenerator(
        ...     generators=[evo2_gen],
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     constraint_weights=[1.0, 2.0],  # Weight homopolymer constraint more
        ...     temperature=0.5  # More greedy sampling
        ... )
    """

    def __init__(
        self,
        generators: List[ProgramGenerator],
        constraints: List[ProgramConstraint],
        sequence_order: Tuple[Tuple[BatchedProgramSequence]],
        **hyperparameters: Any,
    ) -> None:
        """
        Initialize the MCMC generator with sub-generators and constraints.

        Args:
            generators: List of registered generators that will propose sequence changes.
                       Each generator must already be registered with initialized sequences.
            constraints: List of constraint functions that define the energy landscape.
            sequence_order: Tuple defining how to concatenate sequences for user output.
                          Each inner tuple represents sequences to concatenate together.
            **hyperparameters: Additional configuration options:
                - constraint_weights (List[float]): Weights for each constraint.
                - num_steps (int): Number of MCMC steps per sample() call.
                - temperature (float): Metropolis-Hastings temperature.
                - track_step_size (int): Interval for progress tracking.
                - custom_logging (Callable): Custom logging function.
                - verbose (bool): Whether to print progress information.

        Raises:
            ValueError: If generators are not registered, constraints don't match
                       generator outputs, or configuration is invalid.
        """
        super().__init__(**hyperparameters)
        self.generators = generators
        self.constraints = constraints
        self.sequence_order = sequence_order
        self.constraint_weights: List[float] = hyperparameters.get(
            "constraint_weights",
            [1.0] * len(constraints),
        )
        self.num_steps: int = hyperparameters.get("num_steps", 1)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        self.temperature_min: float = hyperparameters.get("temperature_min", 0.0001)
        self.track_step_size: int = hyperparameters.get("track_step_size", 10)
        self.custom_logging: Callable[[int, BatchedProgramSequence], None]
        self.custom_logging = hyperparameters.get("custom_logging", None)
        self.verbose: bool = hyperparameters.get("verbose", True)
        self.current_step: int = 0

        # Validate all configuration using the parent class validation method
        self._validate_init()

    def register(self) -> Tuple[BatchedProgramSequence]:
        """
        Collect _generator_outputs from all registered sub-generators.

        Since this generator wraps pre-registered sub-generators, it simply
        returns their combined outputs rather than creating new sequences.

        Returns:
            Tuple of all BatchedProgramSequence objects from sub-generators,
            flattened into a single tuple for unified access.
        """
        self._is_initialized = True
        
        self._generator_outputs = tuple(seq for gen in self.generators for seq in gen.get_generator_outputs())

        return self._generator_outputs

    def sample(self) -> List[Tuple[ProgramSequence]]:
        """
        Execute Metropolis-Hastings MCMC sampling for sequence optimization.

        Runs the specified number of MCMC steps, where each step:
        1. Selects a random sub-generator
        2. Proposes sequence changes via that generator
        3. Evaluates energy change using constraints
        4. Accepts or rejects based on Metropolis-Hastings criterion
        5. Optionally logs progress and tracks state

        Returns:
            List of user_sequences snapshots taken at tracked intervals.
            Each snapshot contains sequences with energy and step metadata.

        Note:
            Temperature annealing is applied with the formula:
            T(step) = (T_min / T_max) ^ (step / num_steps)
            where T_min = self.temperature_min and T_max = self.temperature.
        """
        # Initialize MCMC states
        self.current_step = 0
        energies = self.score_energy()
        current_best_energy = np.min(energies)
        current_best_idx = np.argmin(energies)
        sequence_history = [self.user_sequences]

        # Execute MCMC optimization steps
        for step in range(1, self.num_steps + 1):
            self.current_step = step
            step_temperature = (self.temperature_min / self.temperature) ** (step / self.num_steps)

            # 1. Pick generator and store old sequences for potential revert
            generator = random.choice(self.generators)
            old_generator_outputs = copy.deepcopy(generator.get_generator_outputs())

            # 2. Sample new proposal and evaluate
            generator.sample()
            new_energies = self.score_energy()
            new_best_energy = np.min(new_energies)
            new_best_idx = np.argmin(new_energies)

            # 3. Compute acceptance probability and decide
            accept = self._compute_acceptance(current_best_energy, new_best_energy, step_temperature)

            # 4. Log progress
            if self.verbose and step % self.track_step_size == 0:
                alpha = min(1.0, np.exp(-(new_best_energy - current_best_energy) / step_temperature))
                self._log_step(step, current_best_energy, new_best_energy, alpha, accept, new_best_idx, step_temperature)

            # 5. Accept or reject the proposal
            current_best_energy, current_best_idx = self._accept_or_reject_proposal(
                accept, 
                generator, 
                old_generator_outputs, 
                current_best_energy, 
                current_best_idx, 
                new_best_energy, 
                new_best_idx
            )

            # 6. Track progress periodically
            if step % self.track_step_size == 0:
                sequence_history.append(self.user_sequences)

        return sequence_history

    def _compute_acceptance(self, current_best_energy: float, new_best_energy: float, temperature: float) -> bool:
        """
        Compute Metropolis-Hastings acceptance probability and make decision.
        
        Args:
            current_best_energy: Energy of current best sequence.
            new_best_energy: Energy of proposed sequence.
            temperature: Current temperature for acceptance calculation.
            
        Returns:
            Boolean indicating whether to accept the proposal.
        """
        alpha = np.exp(-(new_best_energy - current_best_energy) / temperature)
        alpha = min(1.0, alpha)
        return random.random() < alpha

    def _accept_or_reject_proposal(self, accept: bool, generator: ProgramGenerator, old_generator_outputs: Tuple[BatchedProgramSequence],
                                   current_best_energy: float, current_best_idx: int,
                                   new_best_energy: float, new_best_idx: int) -> Tuple[float, int]:
        """
        Execute accept/reject decision and update sequences accordingly.
        
        Args:
            accept: Whether to accept the proposal.
            generator: The generator that made the proposal.
            old_generator_outputs: Backup of sequences before proposal.
            current_best_energy: Current best energy value.
            current_best_idx: Index of current best sequence.
            new_best_energy: Proposed best energy value.
            new_best_idx: Index of proposed best sequence.
            
        Returns:
            Tuple of (best_energy, best_idx) after accept/reject decision.
        """
        if accept:
            # Accept: copy best sequences to all positions
            self._propagate_best_sequence(new_best_idx)
            return new_best_energy, new_best_idx
        else:
            # Reject: revert the sampled generator's sequences
            for i, sequence_batch in enumerate(generator.get_generator_outputs()):
                for j, program_seq in enumerate(sequence_batch):
                    program_seq.sequence = old_generator_outputs[i][j].sequence
            return current_best_energy, current_best_idx

    def _log_step(self, step: int, old_energy: float, new_energy: float, 
                  alpha: float, accept: bool, best_idx: int, temperature: float) -> None:
        """
        Log information about the current MCMC step.
        
        Args:
            step: Current step number.
            old_energy: Energy before proposal.
            new_energy: Energy after proposal.
            alpha: Acceptance probability.
            accept: Whether proposal was accepted.
            best_idx: Index of best sequence.
            temperature: Current temperature.
        """
        print(
            f"Iteration {step} | "
            f"old best energy: {old_energy:.4f}, "
            f"new best energy: {new_energy:.4f}, "
            f"alpha: {alpha:.4f}, "
            f"temperature: {temperature:.6f}, "
            f"accept: {accept}, "
            f"best_idx: {best_idx}"
        )
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()


class ProgramSequentialGenerator(ProgramIterativeGenerator):
    """
    Sequential generator for multi-stage sequence optimization.

    This generator implements a sequential sampling strategy where multiple
    generators are applied in order, with optional Metropolis-Hastings
    acceptance criteria. Unlike MCMC which randomly selects generators,
    this applies them in a fixed sequence each iteration.

    Useful for multi-stage design pipelines where different generators
    handle different aspects of the optimization (e.g., coarse structure
    followed by fine-tuning).

    Examples:
        Two-stage design pipeline:
        >>> sequential = ProgramSequentialGenerator(
        ...     generators=[coarse_gen, fine_gen],  # Applied in order
        ...     constraints=[structure_constraint, stability_constraint],
        ...     sequence_order=((batch1, batch2),),
        ...     num_steps=500
        ... )
        >>> history = sequential.sample()
        
        With temperature control:
        >>> sequential = ProgramSequentialGenerator(
        ...     generators=[gen1, gen2, gen3],
        ...     constraints=[constraint1, constraint2],
        ...     temperature=0.8,  # Accept/reject after all generators
        ...     track_step_size=50
        ... )
    """

    def __init__(
        self,
        generators: List[ProgramGenerator],
        constraints: List[ProgramConstraint],
        sequence_order: Tuple[Tuple[BatchedProgramSequence]],
        **hyperparameters: Any,
    ) -> None:
        """
        Initialize the sequential generator with ordered sub-generators.

        Args:
            generators: List of registered generators applied sequentially each step.
            constraints: List of constraint functions defining the energy landscape.
            sequence_order: Tuple defining sequence concatenation for user output.
            **hyperparameters: Additional configuration options:
                - constraint_weights (List[float]): Weights for each constraint.
                - num_steps (int): Number of sequential sampling steps.
                - temperature (float): Temperature for accept/reject decisions.
                - track_step_size (int): Interval for progress tracking.
                - custom_logging (Callable): Custom logging function.
                - verbose (bool): Whether to print progress information.

        Raises:
            ValueError: If configuration is invalid or generators not registered.
        """
        super().__init__(**hyperparameters)
        self.generators = generators
        self.constraints = constraints
        self.sequence_order = sequence_order
        self.constraint_weights: List[float] = hyperparameters.get(
            "constraint_weights",
            [1.0] * len(constraints),
        )
        self.num_steps: int = hyperparameters.get("num_steps", 1)
        self.temperature: float = hyperparameters.get("temperature", 1.0)
        self.track_step_size: int = hyperparameters.get("track_step_size", 10)
        self.custom_logging: Callable[[int, BatchedProgramSequence], None]
        self.custom_logging = hyperparameters.get("custom_logging", None)
        self.verbose: bool = hyperparameters.get("verbose", True)
        self.current_step: int = 0

        # Validate all configuration using the parent class validation method
        self._validate_init()

    def register(self) -> Tuple[BatchedProgramSequence]:
        """
        Collect _generator_outputs from all registered sub-generators.

        Returns:
            Tuple of all BatchedProgramSequence objects from sub-generators.
        """
        self._is_initialized = True
        
        self._generator_outputs = tuple(seq for gen in self.generators for seq in gen.get_generator_outputs())

        return self._generator_outputs

    def sample(self) -> List[Tuple[ProgramSequence]]:
        """
        Execute sequential sampling with optional accept/reject decisions.

        Each step applies all generators in sequence, then evaluates the
        combined changes and accepts or rejects based on energy improvement
        and temperature.

        Returns:
            List of user_sequences snapshots taken at tracked intervals.

        Note:
            All generators are applied before the accept/reject decision,
            unlike MCMC which evaluates each generator separately.
            Temperature annealing is applied with the formula:
            T(step) = (0.0001 / T_initial) ^ (step / num_steps)
            where T_initial is the initial temperature parameter.
        """
        # Initialize history tracking
        self.current_step = 0
        old_energies = self.score_energy()
        old_best_energy = np.min(old_energies)
        sequence_history = [self.user_sequences]

        # Execute sequential optimization steps
        for step in range(1, self.num_steps + 1):
            self.current_step = step
            step_temperature = (0.0001 / self.temperature) ** (step / self.num_steps)

            # Store old sequences for potential revert
            old_sequences_by_gen = []
            for generator in self.generators:
                gen_old_seqs = []
                for sequence_batch in generator.get_generator_outputs():
                    for program_seq in sequence_batch:
                        gen_old_seqs.append(program_seq.sequence)
                old_sequences_by_gen.append(gen_old_seqs)

            # Sample from each generator in sequence
            for i, generator in enumerate(self.generators):
                # Sample from generator
                generator.sample()
                
            # Evaluate new energy
            new_energies = self.score_energy()
            new_best_energy = np.min(new_energies)

            # Compute acceptance probability with temperature
            alpha = np.exp(-(new_best_energy - old_best_energy) / step_temperature)
            alpha = min(1.0, alpha)

            # Accept/reject according to random number [0.0, 1.0)
            accept = random.random() < alpha

            if self.verbose and step % self.track_step_size == 0:
                print(
                    f"Iteration {step} | "
                    f"old best energy: {old_best_energy:.4f}, "
                    f"new best energy: {new_best_energy:.4f}, "
                    f"alpha: {alpha:.4f}, "
                    f"temperature: {step_temperature:.6f}, "
                    f"accept: {accept}"
                )
                if self.custom_logging:
                    self.custom_logging(step, self.get_generator_outputs())
                sys.stdout.flush()

            if accept:
                # Accept: copy best sequences to all positions
                new_best_idx = np.argmin(new_energies)
                self._propagate_best_sequence(new_best_idx)
                old_best_energy = new_best_energy
            else:
                # Revert changes if rejected
                for i, generator in enumerate(self.generators):
                    seq_idx = 0
                    for sequence_batch in generator.get_generator_outputs():
                        for program_seq in sequence_batch:
                            program_seq.sequence = old_sequences_by_gen[i][seq_idx]
                            seq_idx += 1

            # Track sequence snapshots periodically
            if step % self.track_step_size == 0:
                sequence_history.append(self.user_sequences)

        return sequence_history
