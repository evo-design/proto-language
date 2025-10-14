"""
Evo2 Generator

Extracted from generator.py for better code organization.
"""

from typing import Any, List, Optional, Dict, final

from pydantic import Field, field_validator

from ..base import Generator, Segment
from proto_language.base_config import BaseConfig
from .registry import GeneratorRegistry


class Evo2GeneratorConfig(BaseConfig):
    """Configuration for Evo2Generator."""
    prompt_seqs: List[str] = Field(description="Prompt sequences for generation (1 or batch_size prompts)")
    evo2_type: str = Field(default="evo2_7b", description="Evo2 model variant to use")
    evo2_local_path: Optional[str] = Field(default=None, description="Optional path to local model weights")
    sequence_length: int = Field(default=500, ge=1, description="Number of tokens to generate after prompt")
    temperature: float = Field(default=1.0, gt=0.0, description="Sampling temperature")
    top_k: int = Field(default=4, ge=1, description="Top-k sampling parameter")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    batched: bool = Field(default=True, description="Whether to use batched generation")
    cached_generation: bool = Field(default=True, description="Whether to cache model states")
    verbose: int = Field(default=1, ge=0, description="Verbosity level for logging")
    force_prompt_threshold: Optional[int] = Field(default=None, description="Optional threshold for forcing prompt continuation")
    batch_size: int = Field(default=1, ge=1, description="Number of sequences to generate")
    prepend_prompt: bool = Field(default=False, description="Whether to prepend prompt to generated sequences")
    sampling_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional sampling arguments")
    
    @field_validator('prompt_seqs')
    @classmethod
    def validate_prompt_seqs(cls, v):
        if not v:
            raise ValueError("prompt_seqs must not be empty")
        return v


@GeneratorRegistry.register(
    key="evo2",
    config=Evo2GeneratorConfig,
    description="Evo2 genome language model for DNA sequence generation",
    category="language_model",
    requires_gpu=True,
    supports_batch=True,
    estimated_runtime="slow",
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
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> config = Evo2GeneratorConfig(
        ...     prompt_seqs=["+~GA"],
        ...     evo2_type="evo2_7b",
        ...     sequence_length=1000,
        ...     temperature=0.8,
        ...     batch_size=5
        ... )
        >>> gen = Evo2Generator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.DNA)
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates sequences from prompts

        Custom model with local weights:
        >>> config = Evo2GeneratorConfig(
        ...     prompt_seqs=["+~GA", "+~GC"],
        ...     evo2_type="evo2_7b_phage",
        ...     evo2_local_path="/path/to/weights.pt",
        ...     batch_size=2
        ... )
        >>> gen = Evo2Generator(config)
        >>> gen.assign(segment)
        >>> gen.sample()  # Uses local model weights
    """

    def __init__(self, config: Evo2GeneratorConfig) -> None:
        """
        Initialize the Evo2 generator with model configuration and sampling parameters.

        For detailed documentation of Evo2 sampling parameters, refer to:
        https://github.com/arcinstitute/evo2 and https://github.com/Zymrael/vortex

        Args:
            config: Configuration object containing all generator parameters.

        Note:
            Model instances are automatically shared between generators with the same
            evo2_type, evo2_local_path, and sampling_kwargs to save memory and initialization time.
        """
        super().__init__(batch_size=config.batch_size)
        self.config = config

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(config.prompt_seqs) == 1:
            self.prompt_seqs = config.prompt_seqs * config.batch_size
        else:
            if len(config.prompt_seqs) != config.batch_size:
                raise ValueError(
                    f"Multiple prompts ({len(config.prompt_seqs)}) must equal batch_size ({config.batch_size})"
                )
            if len(set(len(seq) for seq in config.prompt_seqs)) != 1:
                raise ValueError(
                    f"All prompts must have same length, got: {[len(seq) for seq in config.prompt_seqs]}"
                )
            self.prompt_seqs = config.prompt_seqs

        self.batch_size = config.batch_size
        self.evo2_type = config.evo2_type
        self.evo2_local_path = config.evo2_local_path
        self.n_tokens = config.sequence_length
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.batched = config.batched
        self.cached_generation = config.cached_generation
        self.verbose = config.verbose
        self.force_prompt_threshold = config.force_prompt_threshold
        self.prepend_prompt = config.prepend_prompt
        self.sampling_kwargs = config.sampling_kwargs

    def assign(
        self, assigned_segments: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        Args:
            assigned_segments: A single Segment to be assigned to this generator.

        Raises:
            ValueError: If assigned_segments is not a single Segment object.

        Warning:
            Any existing sequences in the assigned segment will be overwritten when sample()
            is called, as Evo2 performs autoregressive generation from prompt sequences.
        """
        # Validate that we received a single Segment, not a list or other type
        if not isinstance(assigned_segments, Segment):
            raise ValueError(
                f"Evo2Generator.assign() expects a single Segment object, "
                f"got {type(assigned_segments).__name__}. If you have multiple segments, "
                f"assign them to separate generator instances."
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
        self._is_initialized = True

    def sample(self, prompt_seqs: Optional[List[str]] = None) -> None:
        """
        Generate sequences using the Evo2 model and update generator output.

        Uses the Evo2 model to generate continuations from the provided prompt sequences
        or the default prompt sequences, updating the sequences in the Segment in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Use provided prompts or fall back to the default prompt
        prompts = prompt_seqs if prompt_seqs is not None else self.prompt_seqs

        # Choose execution mode based on configuration
        from ...utils import use_cloud_gpu
        
        if use_cloud_gpu():
            # Use cloud for cloud GPU execution
            print("Using cloud for Evo2 generation...")
            import cloud
            evo2_generate_cloud = cloud.Function.from_name('proto-language', 'evo2_generate_cloud')
            generated_sequences = evo2_generate_cloud.remote(
                prompt_seqs=prompts,
                evo2_type=self.evo2_type,
                evo2_local_path=self.evo2_local_path,
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
        else:
            # Use local GPU execution
            print("Using local GPU for Evo2 generation...")
            generated_sequences = self._evo2_generate_gpu(
                prompt_seqs=prompts,
                evo2_type=self.evo2_type,
                evo2_local_path=self.evo2_local_path,
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

        # Update sequences in the Segment
        for idx, sequence in enumerate(generated_sequences):
            if self.prepend_prompt:
                sequence = prompts[idx] + sequence
            self._generator_output.batch_sequences[idx].sequence = sequence

    def _evo2_generate_gpu(
        self,
        prompt_seqs: List[str],
        evo2_type: str,
        evo2_local_path: Optional[str],
        n_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        batched: bool,
        cached_generation: bool,
        verbose: int,
        force_prompt_threshold: Optional[int],
        **sampling_kwargs
    ) -> List[str]:
        """
        Local GPU function for Evo2 generation.
        
        Returns:
            List of generated sequences
        """
        from evo2 import Evo2

        # Load and generate
        print(f"Loading Evo2 model: {evo2_type}")
        evo2_model = Evo2(model_name=evo2_type, local_path=evo2_local_path)
        
        output = evo2_model.generate(
            prompt_seqs=prompt_seqs,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batched=batched,
            cached_generation=cached_generation,
            verbose=verbose,
            force_prompt_threshold=force_prompt_threshold,
            **sampling_kwargs,
        )
        return output.sequences

