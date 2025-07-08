import copy
import os
import random
import sys
import json
from typing import Any, List, Optional, Tuple, Callable, Iterable, final

import numpy as np
import requests

from .base import *

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0


@final
class UniformMutationGenerator(Generator):
    """
    A sequence generator that proposes random point mutations.

    This generator initializes with a random sequence and samples single-nucleotide
    or amino acid mutations on each call to sample().

    Examples:
        Creating a DNA mutation generator:
        >>> segment = ConstructSegment(sequence="ATCGG", sequence_type=SequenceType.DNA)
        >>> gen = UniformMutationGenerator(
        ...     batch_size=5,
        ...     sequence_length=5
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()  # Introduces random mutations
        >>> outputs = gen.get_generator_outputs()
        >>> len(outputs[0])  # 5 (batch size)
    """

    def __init__(
        self,
        batch_size: int = 1,
        sequence_length: int = 100,
    ) -> None:
        """
        Initialize the uniform mutation generator.

        Args:
            batch_size: Number of sequence variants to maintain simultaneously.
            sequence_length: Length of the sequence to generate.
        """
        super().__init__(batch_size=batch_size)
        self.sequence_length = sequence_length

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "UniformMutationGenerator must be assigned exactly one ConstructSegment"
            )

        # Initialize _generator_output (singular) and create batch
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True

        initial_sequence = self._generator_output.batch_sequences[0].sequence
        valid_chars = self._generator_output._valid_chars - set(" ")
        valid_chars_list = list(valid_chars)
        if initial_sequence == "":
            self._generator_output.batch_sequences[0].sequence = "".join(
                random.choice(valid_chars_list) for _ in range(self.sequence_length)
            )
        else:
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )
        self._generator_output.create_batch(self.batch_size)

        # No model initialization needed for this generator
        self._is_initialized = True

    def sample(self) -> None:
        """
        Introduce a random point mutation in each sequence.

        For each sequence in the batch, selects a random position and replaces
        the character with a different random character from the vocabulary.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Sample mutation for each output in the segment batch
        for sequence in self._generator_output.batch_sequences:
            mutated_index = random.randint(0, len(sequence.sequence) - 1)
            current_sequence = sequence.sequence
            current_char = current_sequence[mutated_index]

            # Make sure the mutated character is different from the current one
            possible_mutations = [
                c for c in self._generator_output._valid_chars if c != current_char
            ]
            mutated_char = random.choice(possible_mutations)
            sequence.sequence = (
                current_sequence[:mutated_index]
                + mutated_char
                + current_sequence[mutated_index + 1 :]
            )


@final
class Evo2Generator(Generator):
    """
    A sequence generator that uses the Evo2 genome language model for DNA sequence generation.

    This generator wraps the Evo2 model to provide autoregressive sequence generation
    from prompt sequences. The generator can handle single prompts (replicated across batch)
    or multiple prompts (one per batch element), with automatic model instance sharing
    between generators that use the same model configuration.

    Examples:
        Basic DNA generation:
        >>> segment = ConstructSegment(sequence="", sequence_type=SequenceType.DNA)
        >>> gen = Evo2Generator(
        ...     prompt_seqs=["+~GA"],
        ...     evo2_type="evo2_7b",
        ...     sequence_length=1000,
        ...     temperature=0.8,
        ...     batch_size=5
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates sequences from prompts

        Custom model with local weights:
        >>> gen = Evo2Generator(
        ...     prompt_seqs=["+~GA", "+~GC"],
        ...     evo2_type="evo2_7b_phage",
        ...     evo2_local_path="/path/to/weights.pt",
        ...     batch_size=2
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()  # Uses local model weights
    """

    # Class-level cache for sharing model instances
    _model_cache = {}

    def __init__(
        self,
        prompt_seqs: List[str],
        evo2_type: str = "evo2_7b",
        evo2_local_path: Optional[str] = None,
        sequence_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: Optional[int] = None,
        batch_size: int = 1,
        prepend_prompt: bool = False,
        **sampling_kwargs,
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
            sequence_length: Number of tokens to generate after each prompt.
            temperature: Sampling temperature for nucleus sampling.
            top_k: Top-k parameter for sampling.
            top_p: Top-p (nucleus) parameter for sampling.
            batched: Whether to use batched generation for efficiency.
            cached_generation: Whether to cache model states for faster sampling.
            verbose: Verbosity level for generation logging.
            force_prompt_threshold: Optional threshold for forcing prompt continuation.
            batch_size: Number of sequences to generate simultaneously.
            prepend_prompt: Whether to prepend the prompt to generated sequences.
            **sampling_kwargs: Additional arguments passed to Evo2 model sampling.

        Note:
            Model instances are automatically shared between generators with the same
            evo2_type, evo2_local_path, and sampling_kwargs to save memory and initialization time.
        """
        super().__init__(batch_size=batch_size)

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(prompt_seqs) == 1:
            self.prompt_seqs = prompt_seqs * batch_size
        else:
            assert (
                len(prompt_seqs) == batch_size
            ), f"Multiple prompts ({len(prompt_seqs)}) must equal batch_size ({batch_size})"
            assert (
                len(set(len(seq) for seq in prompt_seqs)) == 1
            ), f"All prompts must have same length, got: {[len(seq) for seq in prompt_seqs]}"
            self.prompt_seqs = prompt_seqs

        self.batch_size = batch_size
        self.evo2_type = evo2_type
        self.evo2_local_path = evo2_local_path
        self.n_tokens = sequence_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batched = batched
        self.cached_generation = cached_generation
        self.verbose = verbose
        self.force_prompt_threshold = force_prompt_threshold
        self.prepend_prompt = prepend_prompt
        self.sampling_kwargs = sampling_kwargs

    def _get_model_key(self) -> str:
        """
        Generate a unique key for model caching based on configuration.

        Returns:
            String key uniquely identifying this model configuration.
        """
        return f"{self.evo2_type}:{self.evo2_local_path}"

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.

        Warning:
            Any existing sequences in the assigned segment will be overwritten when sample()
            is called, as Evo2 performs autoregressive generation from prompt sequences.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "Evo2Generator must be assigned exactly one ConstructSegment"
            )

        # Warn user if existing sequences will be overwritten
        existing_sequences = [
            seq.sequence for seq in assigned_segments.batch_sequences if seq.sequence
        ]
        if existing_sequences:
            print(
                f"Warning: Evo2Generator will overwrite {len(existing_sequences)} existing sequence(s) "
                f"when sample() is called due to autoregressive generation."
            )

        # Initialize _generator_output (singular) and create batch
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)

        # Initialize Evo2 model
        model_key = self._get_model_key()
        if model_key not in self._model_cache:
            from evo2 import Evo2  # Lazily import Evo2

            print(f"Loading new Evo2 model with key: {model_key}")
            self._model_cache[model_key] = Evo2(
                model_name=self.evo2_type,
                local_path=self.evo2_local_path,
            )
        else:
            print(f"Using cached Evo2 model with key: {model_key}")

        self.evo2_model = self._model_cache[model_key]
        self._is_initialized = True

    # TODO: generalize the model caching system, maybe move to base class
    # @classmethod
    # def clear_model_cache(cls):
    #     """
    #     Clear the model cache to free GPU memory.

    #     Call this method to force reloading of models if you need to free memory
    #     or switch to different model configurations.
    #     """
    #     cls._model_cache.clear()

    # @classmethod
    # def get_cached_models(cls):
    #     """
    #     Get information about currently cached models.

    #     Returns:
    #         List of model keys currently stored in the cache.
    #     """
    #     return list(cls._model_cache.keys())

    def sample(self, prompt_seqs: Optional[List[str]] = None) -> None:
        """
        Generate sequences using the Evo2 model and update generator output.

        Uses the Evo2 model to generate continuations from the provided prompt sequences
        or the default prompt sequences, updating the sequences in the ConstructSegment in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.

        Raises:
            RuntimeError: If called before assign().
            AssertionError: If number of generated sequences doesn't match prompts.
        """
        self._validate_generator()

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
            **self.sampling_kwargs,
        )

        # Update sequences in the ConstructSegment
        for idx, sequence in enumerate(output.sequences):
            if self.prepend_prompt:
                sequence = prompts[idx] + sequence
            self._generator_output.batch_sequences[idx].sequence = sequence


@final
class NimEvo2Generator(Generator):
    """
    A sequence generator that uses the Nvidia NIM Evo2 API for DNA sequence generation.

    Users must provide a NVIDIA API key for authentication or set it in the NV_API_KEY environment variable.

    Examples:
        >>> segment = ConstructSegment(sequence="", sequence_type=SequenceType.DNA)
        >>> gen = NimEvo2Generator(
        ...     prompt_seqs=["+~GA"],
        ...     api_key="your_api_key",
        ...     n_tokens=1000,
        ...     temperature=0.8,
        ...     batch_size=5
        ... )
        >>> gen.assign(segment)
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
            prepend_prompt: Whether to prepend the prompt to generated sequences.
            timeout: Request timeout in seconds.

        Note:
            The API key can be provided directly or set in the NV_API_KEY environment variable.
        """
        super().__init__(batch_size=batch_size)

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(prompt_seqs) == 1:
            self.prompt_seqs = prompt_seqs * batch_size
        else:
            assert (
                len(prompt_seqs) == batch_size
            ), f"Multiple prompts ({len(prompt_seqs)}) must equal batch_size ({batch_size})"
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
        return os.getenv("NV_API_KEY") or input("Paste the Run Key: ")

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.

        Warning:
            Any existing sequences in the assigned segment will be overwritten when sample()
            is called, as NIM Evo2 performs autoregressive generation from prompt sequences.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "NimEvo2Generator must be assigned exactly one ConstructSegment"
            )

        # Warn user if existing sequences will be overwritten
        existing_sequences = [
            seq.sequence for seq in assigned_segments.batch_sequences if seq.sequence
        ]
        if existing_sequences:
            print(
                f"Warning: NimEvo2Generator will overwrite {len(existing_sequences)} existing sequence(s) "
                f"when sample() is called due to autoregressive generation."
            )

        # Initialize _generator_output (singular) and create batch
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)

        # No model initialization needed for this generator (uses API calls)
        self._is_initialized = True

    def _make_api_request(self, prompt_seq: str) -> str:
        """
        Make API request to generate a sequence.

        Args:
            prompt_seq: The prompt sequence to generate from.

        Returns:
            Generated sequence from the API.

        Raises:
            RuntimeError: If API request fails or response cannot be parsed.
        """
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
                self.api_endpoint, json=payload, headers=headers, timeout=self.timeout
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

    def sample(self, prompt_seqs: Optional[List[str]] = None) -> None:
        """
        Generate sequences using the NIM Evo2 API and update generator output.

        Uses the NIM API to generate continuations from the provided prompt sequences
        or the default prompt sequences, updating the sequences in the ConstructSegment in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.

        Raises:
            RuntimeError: If called before assign() or if API requests fail.
        """
        self._validate_generator()

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

        # Update sequences in the ConstructSegment
        for idx, sequence in enumerate(generated_sequences):
            if self.prepend_prompt:
                sequence = prompts[idx] + sequence
            self._generator_output.batch_sequences[idx].sequence = sequence


class BindCraftGenerator(Generator):
    """
    A placeholder generator for the BindCraft protein design method.

    This generator is currently a stub implementation and needs to be completed
    with the actual BindCraft integration. It will be used for protein sequence
    generation with binding specificity constraints.

    TODO: Consult John to implement this generator. This generator will have multiple outputs.
    """

    pass


@final
class ESM2Generator(Generator):
    """
    A protein sequence generator using the ESM-2 protein language model.

    This generator uses the ESM-2 protein language model to propose sequences and
    mutations based on the model's logits. It supports various decoding strategies
    for selecting positions to mutate and uses temperature-controlled sampling
    for amino acid selection.

    Examples:
        Basic protein generation:
        >>> segment = ConstructSegment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen = ESM2Generator(
        ...     esm2_type="esm2_t33_650M_UR50D",
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5,
        ...     batch_size=3
        ... )
        >>> gen.assign(segment)  # Creates random initial sequences from mask tokens
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
    ) -> None:
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
        """
        super().__init__(batch_size=batch_size)
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
                    Random scores for each position (uniform random values).
                """
                return np.random.random(logits.shape[0])

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

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Creates initial sequences by running ESM-2 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If the segment already contains sequences, they will be used as starting points.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.
            AssertionError: If provided sequence length doesn't match configured length.
        """
        import torch

        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "ESM2Generator must be assigned exactly one ConstructSegment"
            )

        # Initialize _generator_output (singular)
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True

        # Lazily import ESM-2 model
        self.esm2_model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", self.esm2_type
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm2_model.eval()

        # Randomly initialize initial sequence or validate provided sequence length
        initial_sequence = self._generator_output.batch_sequences[0].sequence
        if initial_sequence == "":
            # Generate initial sequences using mask tokens if none provided
            logits = self._esm2_forward(" ".join(["<mask>"] * self.sequence_length))
            assert logits.shape[0] == self.sequence_length

            initial_sequence = "".join(
                [self._sample_logit(logits, pos) for pos in range(self.sequence_length)]
            )

            # Set the generated sequence on the first batch element
            self._generator_output.batch_sequences[0].sequence = initial_sequence
        else:
            # Validate that provided sequence length matches expected length
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )

        self._generator_output.create_batch(self.batch_size)

        self._is_initialized = True

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM-2 logits,
        selects top-k positions based on the decoding method, and samples new amino acids
        at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        from .utils import sample_k_weighted_no_replacement

        self._validate_generator()

        for i in range(self.batch_size):
            sequence = self._generator_output.batch_sequences[i].sequence

            logits = self._esm2_forward(sequence)

            position_scores = self._decoding_func(logits)  # Score positions.

            for idx in sample_k_weighted_no_replacement(position_scores, self.top_k):
                sequence = (
                    sequence[:idx]
                    + self._sample_logit(logits, idx)
                    + sequence[idx + 1 :]
                )

            self._generator_output.batch_sequences[i].sequence = sequence


@final
class MCMCGenerator(IterativeGenerator):
    """
    Metropolis-Hastings MCMC generator for constraint-driven sequence optimization.

    This generator implements a Metropolis-Hastings sampling algorithm that uses
    multiple sub-generators as proposal distributions and constraints to define
    the energy function. It's designed for iterative sequence refinement where
    proposals are accepted or rejected based on energy improvements.

    The generator supports simulated annealing, multiple constraints with weights,
    and flexible sequence concatenation for complex multi-part designs.

    Examples:
        Basic MCMC optimization:
        >>> constructs = [Construct([segment1, segment2])]
        >>> mcmc = MCMCGenerator(
        ...     constructs=constructs,
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     constraint_weights=[1.0, 2.0],  # Weight homopolymer constraint more
        ...     num_steps=100,
        ...     temperature=0.5,  # More greedy sampling
        ...     temperature_min=0.001
        ... )
        >>> history = mcmc.sample()  # Returns optimization history
        >>> final_constructs = mcmc.user_sequences
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        num_steps: int = 1,
        temperature: float = 1.0,
        temperature_min: float = 0.0001,
        track_step_size: int = 10,
        custom_logging: Optional[Callable[[int, Sequence], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the MCMC generator with sub-generators and constraints.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects to generate sequences.
            constraints: List of Constraint objects to evaluate sequences.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            num_steps: Number of MCMC steps per sample() call.
            temperature: Maximum temperature for annealing.
            temperature_min: Minimum temperature for annealing.
            track_step_size: Interval for progress tracking.
            custom_logging: Custom logging function that takes (step, sequences) arguments.
            verbose: Whether to print progress information.

        Raises:
            ValueError: If any validation checks fail.
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
        )
        self.num_steps: int = num_steps
        self.temperature: float = temperature
        self.temperature_min: float = temperature_min
        self.track_step_size: int = track_step_size
        self.custom_logging: Optional[
            Callable[[int, Tuple[ConstructSegment, ...]], None]
        ] = custom_logging
        self.verbose: bool = verbose

        self._validate_generator()

    def _validate_generator(self) -> None:
        """
        Validate configuration for MCMCGenerator.

        Raises:
            ValueError: If temperature parameters are invalid.
        """
        super()._validate_generator()

        # Validate temperature parameters
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_min <= 0:
            raise ValueError(
                f"temperature_min must be positive, got {self.temperature_min}"
            )
        if self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be less than temperature ({self.temperature}) for annealing to work properly"
            )

    def sample(self) -> Optional[List[Tuple[Construct, ...]]]:
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
            Each snapshot contains Construct objects with energy and step metadata.

        Note:
            Temperature annealing is applied with the formula:
            T(step) = (self.temperature_min / self.temperature) ^ (step / num_steps)
        """
        # Initialize MCMC states
        energies = self.score_energy()
        current_best_energy = np.min(energies)
        current_best_idx = np.argmin(energies)
        sequence_history = [self.user_sequences]

        # Execute MCMC optimization steps
        for step in range(1, self.num_steps + 1):
            self.current_step = step
            cur_temp = (self.temperature_min / self.temperature) ** (
                step / self.num_steps
            )

            # Execute single MCMC step
            current_best_energy, current_best_idx = self._execute_mcmc_step(
                step, cur_temp, current_best_energy, current_best_idx
            )

            # Track progress periodically
            if step % self.track_step_size == 0:
                sequence_history.append(self.user_sequences)

        return sequence_history

    def _execute_mcmc_step(
        self,
        step: int,
        cur_temp: float,
        current_best_energy: float,
        current_best_idx: int,
    ) -> Tuple[float, int]:
        """
        Execute a single MCMC step including proposal, evaluation, and acceptance decision.

        Args:
            step: Current step number.
            cur_temp: Current temperature for this step.
            current_best_energy: Current best energy value.
            current_best_idx: Index of current best sequence.

        Returns:
            Tuple of (updated_best_energy, updated_best_idx).
        """
        # 1. Pick generator and store old sequences for potential revert
        generator = random.choice(self.generators)
        old_generator_outputs = copy.deepcopy(generator.get_generator_outputs())

        # 2. Sample new proposal and evaluate
        generator.sample()
        new_energies = self.score_energy()
        new_best_energy = np.min(new_energies)
        new_best_idx = np.argmin(new_energies)

        # 3. Accept or reject proposal according to Metropolis-Hastings algorithm
        original_best_energy = current_best_energy  # Save original for logging
        current_best_energy, current_best_idx, accept, alpha = (
            self._accept_or_reject_proposal(
                current_best_energy,
                current_best_idx,
                new_best_energy,
                new_best_idx,
                cur_temp,
                generator,
                old_generator_outputs,
            )
        )

        # 4. Log progress
        if self.verbose and step % self.track_step_size == 0:
            self._log_step(
                step,
                original_best_energy,
                new_best_energy,
                alpha,
                accept,
                current_best_idx,
                cur_temp,
            )

        return current_best_energy, current_best_idx

    def _accept_or_reject_proposal(
        self,
        current_best_energy: float,
        current_best_idx: int,
        new_best_energy: float,
        new_best_idx: int,
        cur_temp: float,
        generator: Generator,
        old_generator_outputs: Tuple[ConstructSegment, ...],
    ) -> Tuple[float, int, bool, float]:
        """
        Compute Metropolis-Hastings acceptance probability and execute the decision.

        Args:
            current_best_energy: Energy of current best sequence.
            current_best_idx: Index of current best sequence.
            new_best_energy: Energy of proposed sequence.
            new_best_idx: Index of proposed best sequence.
            cur_temp: Current temperature for acceptance calculation.
            generator: The generator that made the proposal.
            old_generator_outputs: Backup of sequences before proposal.

        Returns:
            Tuple of (final_best_energy, final_best_idx, accept, alpha).
        """
        # Compute acceptance probability
        energy_diff = -(new_best_energy - current_best_energy) / cur_temp
        energy_diff = min(energy_diff, MAX_EXP_ARG)  # Clamp to prevent overflow
        alpha = np.exp(energy_diff)
        alpha = min(1.0, alpha)
        accept = random.random() < alpha

        # Execute the decision
        if accept:
            # Accept: copy best sequences to all positions
            self._replicate_best_sequence(new_best_idx)
            return new_best_energy, new_best_idx, accept, alpha
        else:
            # Reject: revert the sampled generator's sequences and metadata
            for i, sequence_batch in enumerate(generator.get_generator_outputs()):
                for j, program_seq in enumerate(sequence_batch):
                    program_seq.sequence = (
                        old_generator_outputs[i].batch_sequences[j].sequence
                    )
                    program_seq._metadata = (
                        old_generator_outputs[i].batch_sequences[j]._metadata.copy()
                    )
            return current_best_energy, current_best_idx, accept, alpha

    def _log_step(
        self,
        step: int,
        old_energy: float,
        new_energy: float,
        alpha: float,
        accept: bool,
        best_idx: int,
        cur_temp: float,
    ) -> None:
        """
        Log information about the current MCMC step.

        Args:
            step: Current step number.
            old_energy: Energy before proposal.
            new_energy: Energy after proposal.
            alpha: Acceptance probability.
            accept: Whether proposal was accepted.
            best_idx: Index of best sequence.
            cur_temp: Current temperature.
        """
        print(
            f"Iteration {step} | "
            f"old best energy: {old_energy:.4f}, "
            f"new best energy: {new_energy:.4f}, "
            f"alpha: {alpha:.4f}, "
            f"temperature: {cur_temp:.6f}, "
            f"accept: {accept}, "
            f"best_idx: {best_idx}"
        )
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()


@final
class SequentialGenerator(IterativeGenerator):
    """
    Sequential generator for chaining autoregressive sequence generators.

    Applies multiple generators in sequence where each uses the previous generator's
    output as input prompts. After all generators run, accepts or rejects the
    combined changes based on energy improvement and temperature annealing.

    Requirements:
    - All generators must output exactly one ConstructSegment
    - Generators after the first must accept prompt_seqs parameter in sample()

    Examples:
        Basic sequential chaining:
        >>> constructs = [Construct([segment1, segment2])]
        >>> sequential = SequentialGenerator(
        ...     constructs=constructs,
        ...     generators=[gen1, gen2, gen3],  # Chain: gen1 -> gen2(gen1_out) -> gen3(gen2_out)
        ...     constraints=[constraint1, constraint2],
        ...     constraint_weights=[1.0, 2.0],  # Weight constraint2 more heavily
        ...     num_steps=50,
        ...     temperature=0.8,  # Accept/reject after all generators
        ...     temperature_min=0.001
        ... )
        >>> history = sequential.sample()  # Returns optimization history
        >>> final_sequences = sequential.user_sequences

    Notes:
        - Final sequences: initial_prompt + gen1_output + gen2_output + ...
        - Temperature annealing: T(step) = (T_min / T_max) ^ (step / num_steps)
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        num_steps: int = 1,
        temperature: float = 1.0,
        temperature_min: float = 0.0001,
        track_step_size: int = 10,
        custom_logging: Optional[Callable[[int, Sequence], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the sequential generator with ordered sub-generators.

        Args:
            constructs: List of Construct objects to be optimized.
            generators: List of Generator objects to be chained sequentially.
            constraints: List of Constraint objects to evaluate sequences.
            constraint_weights: List of weights for each constraint. If None, all weights are 1.0.
            num_steps: Number of optimization steps per sample() call.
            temperature: Maximum temperature for annealing.
            temperature_min: Minimum temperature for annealing.
            track_step_size: Progress tracking interval.
            custom_logging: Custom logging function that takes (step, sequences) arguments.
            verbose: Whether to print progress information.
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
        )
        self.num_steps: int = num_steps
        self.temperature: float = temperature
        self.temperature_min: float = temperature_min
        self.track_step_size: int = track_step_size
        self.custom_logging: Optional[
            Callable[[int, Tuple[ConstructSegment, ...]], None]
        ] = custom_logging
        self.verbose: bool = verbose

        self._validate_generator()

    def _validate_generator(self) -> None:
        """
        Validate configuration for SequentialGenerator.

        Raises:
            ValueError: If generators have different batch sizes or temperature parameters are invalid.
        """
        super()._validate_generator()

        # Check that all batch sizes are the same
        batch_sizes = [gen.batch_size for gen in self.generators]
        if len(set(batch_sizes)) > 1:
            raise ValueError(
                f"All generators must have the same batch_size. Found: {batch_sizes}"
            )

        # Validate temperature parameters
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_min <= 0:
            raise ValueError(
                f"temperature_min must be positive, got {self.temperature_min}"
            )
        if self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be less than temperature ({self.temperature}) for annealing to work properly"
            )

    def sample(self) -> List[Tuple[Construct, ...]]:
        """
        Execute sequential sampling with chained autoregressive generators.

        Each step: (1) applies all generators sequentially with chaining,
        (2) evaluates energy change, (3) accepts/rejects based on Metropolis-Hastings
        with temperature annealing.

        Returns:
            List of user_sequences snapshots taken at tracked intervals.
            Each snapshot contains Construct objects with energy and step metadata.
        """
        # Initialize sequential states
        old_energies = self.score_energy()
        current_best_energy = np.min(old_energies)
        sequence_history = [self.user_sequences]

        # Execute sequential optimization steps
        for step in range(1, self.num_steps + 1):
            self.current_step = step
            cur_temp = (self.temperature_min / self.temperature) ** (
                step / self.num_steps
            )

            # Execute single sequential step
            current_best_energy = self._execute_sequential_step(
                step, cur_temp, current_best_energy
            )

            # Track progress periodically
            if step % self.track_step_size == 0:
                sequence_history.append(self.user_sequences)
        return sequence_history

    def _execute_sequential_step(
        self, step: int, cur_temp: float, current_best_energy: float
    ) -> float:
        """
        Execute a single sequential step including chaining, evaluation, and acceptance decision.

        Args:
            step: Current step number.
            cur_temp: Current temperature for this step.
            current_best_energy: Current best energy value.

        Returns:
            Updated best energy value.
        """
        # 1. Store old sequences for potential revert
        old_sequences_by_gen = self._backup_sequences()

        # 2. Apply all generators sequentially with chaining
        self._sample_sequential_generators()

        # 3. Evaluate new energy
        new_energies = self.score_energy()
        new_best_energy = np.min(new_energies)

        # 4. Accept or reject proposal according to Metropolis-Hastings algorithm
        original_best_energy = current_best_energy  # Save original for logging
        current_best_energy, accept, alpha = self._accept_or_reject_proposal(
            current_best_energy,
            new_best_energy,
            cur_temp,
            old_sequences_by_gen,
            new_energies,
        )

        # 5. Log progress
        if self.verbose and step % self.track_step_size == 0:
            self._log_step(
                step, original_best_energy, new_best_energy, alpha, accept, cur_temp
            )

        return current_best_energy

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
        # TODO: FIX THIS TEMPORARY SOLUTION
        first_gen = self.generators[0]

        # Start with original prompts to preserve prefix tokens
        running_prompts = first_gen.prompt_seqs.copy()

        # Sample from each generator in sequence, chaining outputs
        for i, generator in enumerate(self.generators):
            prompt_seqs = running_prompts if i > 0 else None
            generator.sample(prompt_seqs=prompt_seqs)

            # Accumulate this generator's output
            outputs = generator.get_generator_outputs()
            assert (
                len(outputs) == 1
            ), f"Generator {i} must output exactly one ConstructSegment for chaining"
            batch = outputs[0]

            for batch_idx in range(len(batch)):
                if i == 0 and getattr(generator, "prepend_prompt", False):
                    # First generator with prepend_prompt: output already includes prompt content,
                    # just add back the prefix tokens that were stripped
                    original_prompt = first_gen.prompt_seqs[batch_idx]
                    generated = batch[batch_idx].sequence
                    valid_chars = batch._valid_chars or set()
                    prefix_tokens = "".join(
                        c for c in original_prompt if c not in valid_chars
                    )
                    running_prompts[batch_idx] = prefix_tokens + generated
                else:
                    # Normal case: accumulate output to running prompts
                    running_prompts[batch_idx] += batch[batch_idx].sequence

    def _accept_or_reject_proposal(
        self,
        current_best_energy: float,
        new_best_energy: float,
        cur_temp: float,
        old_sequences_by_gen: List[List[Any]],
        new_energies: List[float],
    ) -> Tuple[float, bool, float]:
        """
        Compute Metropolis-Hastings acceptance probability and make decision.

        Args:
            current_best_energy: Energy of current best sequence.
            new_best_energy: Energy of proposed sequence.
            cur_temp: Current temperature for acceptance calculation.
            old_sequences_by_gen: Backup of sequences before proposal.
            new_energies: All energy values for the new sequences.

        Returns:
            Tuple of (updated_best_energy, accept, alpha).
        """
        # Compute acceptance probability
        energy_diff = -(new_best_energy - current_best_energy) / cur_temp
        energy_diff = min(energy_diff, MAX_EXP_ARG)  # Clamp to prevent overflow
        alpha = np.exp(energy_diff)
        alpha = min(1.0, alpha)
        accept = random.random() < alpha

        # Execute the decision
        if accept:
            # Accept: copy best sequences to all positions
            new_best_idx = np.argmin(new_energies)
            self._replicate_best_sequence(new_best_idx)
            return new_best_energy, accept, alpha
        else:
            # Revert changes if rejected
            for i, generator in enumerate(self.generators):
                seq_idx = 0
                for sequence_batch in generator.get_generator_outputs():
                    for program_seq in sequence_batch:
                        program_seq.sequence = old_sequences_by_gen[i][seq_idx].sequence
                        program_seq._metadata = old_sequences_by_gen[i][
                            seq_idx
                        ]._metadata.copy()
                        seq_idx += 1
            return current_best_energy, accept, alpha

    def _log_step(
        self,
        step: int,
        old_energy: float,
        new_energy: float,
        alpha: float,
        accept: bool,
        cur_temp: float,
    ) -> None:
        """
        Log information about the current sequential generation step.

        Args:
            step: Current step number.
            old_energy: Energy before proposal.
            new_energy: Energy after proposal.
            alpha: Acceptance probability.
            accept: Whether proposal was accepted.
            cur_temp: Current temperature.
        """
        print(
            f"Iteration {step} | "
            f"old best energy: {old_energy:.4f}, "
            f"new best energy: {new_energy:.4f}, "
            f"alpha: {alpha:.4f}, "
            f"temperature: {cur_temp:.6f}, "
            f"accept: {accept}"
        )
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()
