import copy
import os
import random
import sys
import json
from typing import Any, List, Optional, Tuple, Callable

import numpy as np
import requests

from .base import *

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0


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
        prepend_prompt: bool = False,
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
            prepend_prompt: Whether to prepend the prompt to generated sequences. Default: False.
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
        self.prepend_prompt = prepend_prompt

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
        valid_chars: Optional[Set[str]] = None,
    ) -> Tuple[BatchedProgramSequence]:
        """
        Initialize empty DNA sequences and load the Evo2 model.

        Creates BatchedProgramSequence objects that will be populated by sample().
        Loads the Evo2 model from cache if available, otherwise initializes a new
        instance and caches it for future use.

        Args:
            outputs: Optional pre-initialized BatchedProgramSequence objects.
                    If None, empty sequences will be created.
            valid_chars: Optional custom set of valid characters for sequence validation.
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
                ProgramSequence(sequence_type=SequenceType.DNA, valid_chars=valid_chars) 
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

    def sample(self, prompt_seqs: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> None:
        """
        Generate sequences using the Evo2 model and update _generator_outputs.

        Uses the Evo2 model to generate continuations from the provided prompt sequences
        or the default prompt sequences, updating the sequences in the BatchedProgramSequence in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.
            *args: Unused positional arguments for compatibility.
            **kwargs: Unused keyword arguments for compatibility.

        Raises:
            RuntimeError: If called before register().
            AssertionError: If number of generated sequences doesn't match prompts.
        """
        if not self._is_initialized:
            self.register()

        # Use provided prompts or fall back to the default prompt
        prompts = prompt_seqs if prompt_seqs is not None else self.prompt_seqs

        output = self.evo2_model.generate(
            prompt_seqs=prompts,
            n_tokens=self.n_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            batched=self.batched,
            cached_generation=self.cached_generation,
            verbose=self.verbose,
            force_prompt_threshold=self.force_prompt_threshold,
        )

        # Update sequences in the BatchedProgramSequence
        valid_chars = self._generator_outputs[0].valid_chars
        for idx, sequence in enumerate(output.sequences):
            if self.prepend_prompt:
                prompt = prompts[idx].lstrip(''.join(c for c in prompts[idx] if c not in valid_chars))
                final_sequence = prompt + sequence
            else:
                # Use generated sequence without prompt
                final_sequence = sequence
            self._generator_outputs[0][idx].sequence = final_sequence


class NimEvo2Generator(ProgramGenerator):
    """
    A sequence generator that uses the Nvidia NIM Evo2 API for DNA sequence generation.
    
    Users must provide a NVIDIA API key for authentication or set it in the NV_API_KEY environment variable.
    
    Examples:
        >>> gen = NimEvo2Generator(
        ...     prompt_seqs=["+~GA"],
        ...     api_key="your_api_key",
        ...     n_tokens=1000,
        ...     temperature=0.8
        ... )
        >>> gen.register()
        >>> gen.sample()
    """

    def __init__(
        self,
        prompt_seqs: List[str],
        nim_api_url: str = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate",
        api_key: Optional[str] = None,
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        enable_sampled_probs: bool = False,
        verbose: int = 1,
        batch_size: int = 1,
        prepend_prompt: bool = False,
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        """
        Initialize the NIM Evo2 generator with API configuration and sampling parameters.

        Args:
            prompt_seqs: List of prompt sequences to start generation from.
                        Single prompt gets replicated batch_size times, or provide
                        one prompt per batch element.
            nim_api_url: Full URL for the Nvidia NIM API endpoint.
            api_key: API key for authentication. If None, will try to get from NV_API_KEY environment variable.
            n_tokens: Number of tokens to generate after each prompt.
            temperature: Sampling temperature for nucleus sampling.
            top_k: Top-k parameter for sampling.
            top_p: Top-p (nucleus) parameter for sampling.
            enable_sampled_probs: Whether to enable sampled probabilities in API response.
            verbose: Verbosity level for generation logging.
            batch_size: Number of sequences to generate simultaneously.
            prepend_prompt: Whether to prepend the prompt to generated sequences. Default: False.
            timeout: Request timeout in seconds. Default: 120.0.
            **kwargs: Additional arguments passed to parent class.

        Note:
            The API key can be provided directly or set in the NV_API_KEY environment variable.
        """
        super().__init__(batch_size=batch_size, **kwargs)

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(prompt_seqs) == 1:
            self.prompt_seqs = prompt_seqs * batch_size
        else:
            assert len(prompt_seqs) == batch_size, f"Multiple prompts ({len(prompt_seqs)}) must equal batch_size ({batch_size})"
            self.prompt_seqs = prompt_seqs

        self.batch_size = batch_size
        self.api_endpoint = nim_api_url
        self.api_key = api_key or self._get_api_key_from_env()
        self.n_tokens = n_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.enable_sampled_probs = enable_sampled_probs
        self.verbose = verbose
        self.prepend_prompt = prepend_prompt
        self.timeout = timeout

    def _get_api_key_from_env(self) -> Optional[str]:
        """
        Get API key from environment variables.
        
        Returns:
            API key from NV_API_KEY environment variable, or None if not set.
        """
        return os.getenv('NV_API_KEY') or input("Paste the Run Key: ")

    def register(
        self, 
        outputs: Optional[Tuple[BatchedProgramSequence]] = None,
        valid_chars: Optional[Set[str]] = None,
    ) -> Tuple[BatchedProgramSequence]:
        """
        Initialize empty DNA sequences for NIM API generation.

        Creates BatchedProgramSequence objects that will be populated by sample().
        No model loading is required since generation happens via API calls.

        Args:
            outputs: Optional pre-initialized BatchedProgramSequence objects.
                    If None, empty sequences will be created.
            valid_chars: Optional custom set of valid characters for sequence validation.
            
        Returns:
            Tuple containing a single BatchedProgramSequence with empty DNA sequences
            that will be filled during sampling.

        Raises:
            ValueError: If outputs is provided but has incorrect structure.
        """
        self._is_initialized = True

        if outputs is None:
            # Create one BatchedProgramSequence containing all prompt sequences
            sequence_batch = BatchedProgramSequence(
                ProgramSequence(sequence_type=SequenceType.DNA, valid_chars=valid_chars) 
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

    def _make_api_request(self, prompt_seq: str) -> str:
        """Make API request to generate a sequence."""
        payload = {
            "sequence": prompt_seq,
            "num_tokens": self.n_tokens,
            "top_k": self.top_k,
            "enable_sampled_probs": self.enable_sampled_probs,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.api_endpoint, 
                json=payload, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if self.verbose >= 1:
                print(f"API response: {result}")
                
            return result["sequence"]
            
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to parse API response: {e}")

    def sample(self, prompt_seqs: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> None:
        """
        Generate sequences using the NIM Evo2 API and update _generator_outputs.

        Uses the NIM API to generate continuations from the provided prompt sequences
        or the default prompt sequences, updating the sequences in the BatchedProgramSequence in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.
            *args: Unused positional arguments for compatibility.
            **kwargs: Unused keyword arguments for compatibility.

        Raises:
            RuntimeError: If called before register() or if API requests fail.
        """
        if not self._is_initialized:
            self.register()

        # Use provided prompts or fall back to the default prompt
        prompts = prompt_seqs if prompt_seqs is not None else self.prompt_seqs

        if self.verbose >= 1:
            print(f"Generating {len(prompts)} sequences via NIM API...")

        # Generate sequences for each prompt
        generated_sequences = []
        for i, prompt in enumerate(prompts):
            if self.verbose >= 1:
                print(f"Generating sequence {i+1}/{len(prompts)}")
            
            try:
                generated_seq = self._make_api_request(prompt)
                generated_sequences.append(generated_seq)
            except Exception as e:
                print(f"Failed to generate sequence {i+1}: {e}")
                # Use empty string as fallback
                generated_sequences.append("")

        # Update sequences in the BatchedProgramSequence
        valid_chars = self._generator_outputs[0].valid_chars
        for idx, sequence in enumerate(generated_sequences):
            if self.prepend_prompt:
                prompt = prompts[idx].lstrip(''.join(c for c in prompts[idx] if c not in (valid_chars or set())))
                final_sequence = prompt + sequence
            else:
                # Use generated sequence without prompt
                final_sequence = sequence
            self._generator_outputs[0][idx].sequence = final_sequence


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
    

    @property
    def user_sequences(self) -> Tuple[ProgramSequence]:
        """
        Get user-facing sequences with energy metadata and proper concatenation.
        
        This is the main API for accessing generation results. It transforms
        internal _generator_outputs into clean user outputs by:
        
        1. Concatenating sequences according to sequence_order groups
        2. Adding rich metadata (energy_score, time_step)
        3. Preserving all constraint-generated metadata
        4. Creating immutable snapshots of the current state
        
        Each returned ProgramSequence includes metadata:
        - energy_score: Current energy from score_energy() for this sequence
        - time_step: Current step number in the optimization process
        - All metadata from constraint evaluations (e.g., avg_plddt, ptm, pdb_output)
        
        Technical details:
        - Uses sequences with lowest energy from each BatchedProgramSequence
        - Creates new ProgramSequence objects on each access
        - Validates sequence type consistency within groups
        - Preserves all metadata from the original sequences
        - Elements at the same index across batches get concatenated together
        
        Returns:
            Tuple of ProgramSequence objects with concatenated sequences and metadata.
            One sequence per group defined in sequence_order.
            
        Raises:
            ValueError: If sequences within a group have inconsistent types.
            
        Examples:
            If sequence_order = ((batch1, batch2), (batch3,)), this returns:
            (
                ProgramSequence(batch1[best_idx] + batch2[best_idx], metadata={...}),
                ProgramSequence(batch3[best_idx], metadata={...})
            )
            """
        if not hasattr(self, 'sequence_order') or not self.sequence_order:
            return tuple()
        
        # Get energy scores and find the best sequence index
        energies = self.score_energy()
        best_idx = int(np.argmin(energies)) if energies else 0
        best_energy = energies[best_idx] if energies and best_idx < len(energies) else float('inf')
        time_step = getattr(self, 'current_step', 0)
        
        result = []
        for group in self.sequence_order:
            # Extract best sequences from each batch in the group
            sequences = []
            for batch in group:
                if batch and len(batch) > best_idx:
                    sequences.append(batch[best_idx])
                else:
                    sequences.append(None)
            
            # Build concatenated sequence string
            sequence_strings = [seq.sequence or "" for seq in sequences if seq]
            final_sequence = "".join(sequence_strings)
            
            # Validate sequence type consistency
            sequence_types = {seq.sequence_type for seq in sequences if seq and seq.sequence_type}
            if len(sequence_types) > 1:
                raise ValueError(f"Inconsistent sequence types in group: {sequence_types}")
            
            valid_chars = group[0].valid_chars if group else None
            
            # Create result sequence with metadata
            user_seq = ProgramSequence(
                sequence=final_sequence,
                sequence_type=sequence_types.pop() if sequence_types else None,
                valid_chars=valid_chars
            )
            
            # Merge metadata from all sequences in the group
            merged_metadata = {}
            for seq in sequences:
                if seq and seq._metadata:
                    merged_metadata.update(seq._metadata)
            
            # Add/update standard metadata
            merged_metadata.update({
                "energy_score": best_energy,
                "time_step": time_step
            })
            
            user_seq._metadata.update(merged_metadata)
            result.append(user_seq)
        
        return tuple(result)

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
                # Clamp exponent to prevent overflow (same as in _compute_acceptance)
                energy_diff = -(new_best_energy - current_best_energy) / step_temperature
                energy_diff = min(energy_diff, MAX_EXP_ARG)
                alpha = min(1.0, np.exp(energy_diff))
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
        # Clamp exponent to prevent overflow
        energy_diff = -(new_best_energy - current_best_energy) / temperature
        energy_diff = min(energy_diff, MAX_EXP_ARG)
        alpha = np.exp(energy_diff)
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
            # Reject: revert the sampled generator's sequences and metadata
            for i, sequence_batch in enumerate(generator.get_generator_outputs()):
                for j, program_seq in enumerate(sequence_batch):
                    program_seq.sequence = old_generator_outputs[i][j].sequence
                    program_seq._metadata = old_generator_outputs[i][j]._metadata.copy()
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
    Sequential generator for chaining autoregressive sequence generators.

    Applies multiple generators in sequence where each uses the previous generator's
    output as input prompts. After all generators run, accepts or rejects the
    combined changes based on energy improvement and temperature annealing.

    Requirements:
    - All generators must output exactly one BatchedProgramSequence
    - Generators after the first must accept prompt_seqs parameter in sample()
    - The sequence_order must contain exactly one group

    Examples:
        >>> sequential = ProgramSequentialGenerator(
        ...     generators=[gen1, gen2, gen3],  # Chain: gen1 -> gen2(gen1_out) -> gen3(gen2_out)
        ...     constraints=[constraint1, constraint2],
        ...     constraint_weights=[1.0, 2.0],  # Weight constraint2 more heavily
        ...     temperature=0.8,  # Accept/reject after all generators
        ...     track_step_size=50,
        ...     verbose=True
        ... )
        >>> final_sequences = sequential.sample()

    Notes:
        - Final sequences: initial_prompt + gen1_output + gen2_output + ...
        - Temperature annealing: T(step) = T_initial * (T_min / T_initial)^(step / num_steps)
        - Metadata includes energy_score, time_step, and initial_prompt
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
            generators: Registered generators applied sequentially (must output one BatchedProgramSequence each).
            constraints: Constraint functions defining the energy landscape.
            sequence_order: Sequence concatenation order (must contain exactly one group).
            **hyperparameters: Additional options:
                - constraint_weights (List[float]): Constraint weights. Default: [1.0, ...]
                - num_steps (int): Number of sampling steps. Default: 1
                - temperature (float): Initial temperature. Default: 1.0
                - temperature_min (float): Final temperature for annealing. Default: 0.0001
                - track_step_size (int): Progress tracking interval. Default: 10
                - custom_logging (Callable): Custom logging function. Default: None
                - verbose (bool): Print progress. Default: True
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

        # Validate batch size consistency across all generators
        self._validate_batch_sizes()

        # Validate all configuration using the parent class validation method
        self._validate_init()

    def _validate_batch_sizes(self) -> None:
        """Validate that all generators have consistent batch sizes for sequential chaining."""
        if not self.generators:
            return
            
        # Get batch sizes from all generators
        batch_sizes = [getattr(gen, 'batch_size', None) for gen in self.generators]
        
        # Check for missing batch_size attributes
        if None in batch_sizes:
            missing = [i for i, size in enumerate(batch_sizes) if size is None]
            raise ValueError(f"Generators {missing} are missing batch_size attribute")
        
        # Check that all batch sizes are the same
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"All generators must have the same batch_size. Found: {batch_sizes}")
    
    def register(self) -> Tuple[BatchedProgramSequence]:
        """
        Collect _generator_outputs from all registered sub-generators.

        Returns:
            Tuple of all BatchedProgramSequence objects from sub-generators.
        """
        self._is_initialized = True
        
        self._generator_outputs = tuple(seq for gen in self.generators for seq in gen.get_generator_outputs())

        return self._generator_outputs

    @property
    def user_sequences(self) -> Tuple[ProgramSequence]:
        """
        Get user-facing sequences with concatenated generator outputs and metadata.
        
        Returns sequences by concatenating all generator outputs in sequence order.
        Individual generators handle their own output formatting (e.g., prompt inclusion).
        
        Returns:
            Tuple containing one ProgramSequence with concatenated sequences and metadata
            including energy_score and time_step.
            
        Examples:
            >>> gen1 = Evo2Generator(prompt_seqs=["+~ATG"], prepend_prompt=True, batch_size=2)
            >>> gen2 = Evo2Generator(prompt_seqs=None, prepend_prompt=False, batch_size=2)
            >>> sequential = ProgramSequentialGenerator(
            ...     generators=[gen1, gen2],
            ...     constraints=[gc_constraint],
            ...     sequence_order=((gen1_batch, gen2_batch),)
            ... )
            >>> final_seqs = sequential.user_sequences
            >>> print(final_seqs[0].sequence)  # gen1_output (with prompt) + gen2_output (without prompt)
        """
        # Sequential generators must have exactly one sequence group
        assert hasattr(self, 'sequence_order') and self.sequence_order, "sequence_order must be defined and non-empty"
        assert len(self.sequence_order) == 1, f"Sequential generators must have exactly one sequence group, got {len(self.sequence_order)}"
        
        # Get energy scores and find the best sequence index
        energies = self.score_energy()
        best_idx = int(np.argmin(energies)) if energies else 0
        best_energy = energies[best_idx] if energies and best_idx < len(energies) else float('inf')
        time_step = getattr(self, 'current_step', 0)
        
        # Process the single sequence group
        group = self.sequence_order[0]
        
        # Extract best sequences from each batch in the group
        sequences = []
        for batch in group:
            if batch and len(batch) > best_idx:
                sequences.append(batch[best_idx])
            else:
                sequences.append(None)
        
        # Build concatenated sequence string
        sequence_strings = [seq.sequence or "" for seq in sequences if seq]
        final_sequence = "".join(sequence_strings)
        
        # Validate sequence type consistency
        sequence_types = {seq.sequence_type for seq in sequences if seq and seq.sequence_type}
        if len(sequence_types) > 1:
            raise ValueError(f"Inconsistent sequence types in group: {sequence_types}")
        
        valid_chars = group[0].valid_chars if group else None
        
        # Create result sequence with metadata
        user_seq = ProgramSequence(
            sequence=final_sequence,
            sequence_type=sequence_types.pop() if sequence_types else None,
            valid_chars=valid_chars
        )
        
        # Merge metadata from all sequences in the group
        merged_metadata = {}
        for seq in sequences:
            if seq and seq._metadata:
                merged_metadata.update(seq._metadata)
        
        # Add/update standard metadata
        merged_metadata.update({
            "energy_score": best_energy,
            "time_step": time_step
        })
        
        user_seq._metadata.update(merged_metadata)
        
        # Return tuple with exactly one element
        return (user_seq,)

    def sample(self) -> List[Tuple[ProgramSequence]]:
        """
        Execute sequential sampling with chained autoregressive generators.

        Each step: (1) applies all generators sequentially with chaining, 
        (2) evaluates energy change, (3) accepts/rejects based on Metropolis-Hastings 
        with temperature annealing.

        Returns:
            List of user_sequences snapshots taken at tracked intervals.
        """
        # Initialize history tracking
        self.current_step = 0
        old_energies = self.score_energy()
        old_best_energy = np.min(old_energies)
        sequence_history = [self.user_sequences]

        # Execute sequential optimization steps
        for step in range(1, self.num_steps + 1):
            self.current_step = step
            step_temperature = self.temperature * ((self.temperature_min / self.temperature) ** (step / self.num_steps))

            # Store old sequences for potential revert
            old_sequences_by_gen = self._backup_sequences()

            # Apply all generators sequentially with chaining
            self._sample_sequential_generators()
            
            # Evaluate new energy
            new_energies = self.score_energy()
            new_best_energy = np.min(new_energies)

            # Compute acceptance probability and decide
            accept = self._compute_acceptance(old_best_energy, new_best_energy, step_temperature)

            # Log progress
            if self.verbose and step % self.track_step_size == 0:
                # Clamp exponent to prevent overflow (same as in _compute_acceptance)
                energy_diff = -(new_best_energy - old_best_energy) / step_temperature
                energy_diff = min(energy_diff, MAX_EXP_ARG)
                alpha = min(1.0, np.exp(energy_diff))
                self._log_step(step, old_best_energy, new_best_energy, alpha, accept, step_temperature)

            # Accept or reject the proposal
            old_best_energy = self._accept_or_reject_proposal(
                accept, 
                old_sequences_by_gen, 
                old_best_energy, 
                new_best_energy, 
                new_energies
            )

            # Track sequence snapshots periodically
            if step % self.track_step_size == 0:
                sequence_history.append(self.user_sequences)

        return sequence_history

    def _backup_sequences(self) -> List[List[Any]]:
        """
        Create backup copies of all sequences from all generators.
        
        Returns:
            List of backed up sequences organized by generator.
        """
        old_sequences_by_gen = []
        for generator in self.generators:
            gen_old_seqs = []
            for sequence_batch in generator.get_generator_outputs():
                for program_seq in sequence_batch:
                    gen_old_seqs.append(copy.deepcopy(program_seq))
            old_sequences_by_gen.append(gen_old_seqs)
        return old_sequences_by_gen

    def _sample_sequential_generators(self) -> None:
        """
        Apply all generators sequentially, chaining outputs between them.
        
        Each generator uses the accumulated output from previous generators
        as prompts for its own generation.
        """
        #TODO: FIX THIS TEMPORARY SOLUTION
        first_gen = self.generators[0]
        
        # Start with original prompts to preserve prefix tokens
        running_prompts = first_gen.prompt_seqs.copy()
        
        # Sample from each generator in sequence, chaining outputs
        for i, generator in enumerate(self.generators):
            prompt_seqs = running_prompts if i > 0 else None
            generator.sample(prompt_seqs=prompt_seqs)
            
            # Accumulate this generator's output
            outputs = generator.get_generator_outputs()
            assert len(outputs) == 1, f"Generator {i} must output exactly one BatchedProgramSequence for chaining"
            batch = outputs[0]
            
            for batch_idx in range(len(batch)):
                if i == 0 and getattr(generator, 'prepend_prompt', False):
                    # First generator with prepend_prompt: output already includes prompt content,
                    # just add back the prefix tokens that were stripped
                    original_prompt = first_gen.prompt_seqs[batch_idx]
                    generated = batch[batch_idx].sequence or ""
                    valid_chars = batch.valid_chars or set()
                    prefix_tokens = ''.join(c for c in original_prompt if c not in valid_chars)
                    running_prompts[batch_idx] = prefix_tokens + generated
                else:
                    # Normal case: accumulate output to running prompts
                    running_prompts[batch_idx] += batch[batch_idx].sequence or ""

    def _compute_acceptance(self, old_best_energy: float, new_best_energy: float, temperature: float) -> bool:
        """
        Compute Metropolis-Hastings acceptance probability and make decision.
        
        Args:
            old_best_energy: Energy of current best sequence.
            new_best_energy: Energy of proposed sequence.
            temperature: Current temperature for acceptance calculation.
            
        Returns:
            Boolean indicating whether to accept the proposal.
        """
        # Clamp exponent to prevent overflow
        energy_diff = -(new_best_energy - old_best_energy) / temperature
        energy_diff = min(energy_diff, MAX_EXP_ARG)
        alpha = np.exp(energy_diff)
        alpha = min(1.0, alpha)
        return random.random() < alpha

    def _accept_or_reject_proposal(self, accept: bool, old_sequences_by_gen: List[List[Any]], 
                                   old_best_energy: float, new_best_energy: float, 
                                   new_energies: List[float]) -> float:
        """
        Execute accept/reject decision and update sequences accordingly.
        
        Args:
            accept: Whether to accept the proposal.
            old_sequences_by_gen: Backup of sequences before proposal.
            old_best_energy: Current best energy value.
            new_best_energy: Proposed best energy value.
            new_energies: All energy values for the new sequences.
            
        Returns:
            Updated best energy value after accept/reject decision.
        """
        if accept:
            # Accept: copy best sequences to all positions
            new_best_idx = np.argmin(new_energies)
            self._propagate_best_sequence(new_best_idx)
            return new_best_energy
        else:
            # Revert changes if rejected
            for i, generator in enumerate(self.generators):
                seq_idx = 0
                for sequence_batch in generator.get_generator_outputs():
                    for program_seq in sequence_batch:
                        program_seq.sequence = old_sequences_by_gen[i][seq_idx].sequence
                        program_seq._metadata = old_sequences_by_gen[i][seq_idx]._metadata.copy()
                        seq_idx += 1
            return old_best_energy

    def _log_step(self, step: int, old_energy: float, new_energy: float, 
                  alpha: float, accept: bool, temperature: float) -> None:
        """
        Log information about the current sequential generation step.
        
        Args:
            step: Current step number.
            old_energy: Energy before proposal.
            new_energy: Energy after proposal.
            alpha: Acceptance probability.
            accept: Whether proposal was accepted.
            temperature: Current temperature.
        """
        print(
            f"Iteration {step} | "
            f"old best energy: {old_energy:.4f}, "
            f"new best energy: {new_energy:.4f}, "
            f"alpha: {alpha:.4f}, "
            f"temperature: {temperature:.6f}, "
            f"accept: {accept}"
        )
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()
