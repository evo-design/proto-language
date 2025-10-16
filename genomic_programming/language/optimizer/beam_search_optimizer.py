"""
Beam Search Optimizer

Beam search optimizer that processes segments sequentially with context accumulation.
"""

from typing import Any, Dict, List, Optional, Tuple, final
import copy

import heapq
from pydantic import Field

from ..core import Optimizer, Construct, Segment, Constraint, Sequence, Generator, SequenceType
from proto_language.base_config import BaseConfig
from .optimizer_registry import OptimizerRegistry


class BeamSearchOptimizerConfig(BaseConfig):
    """Configuration for BeamSearchOptimizer algorithm parameters only.

    Runtime objects (constructs, generators, constraints) should be passed
    separately to the optimizer's __init__ method, not in the config.
    """
    beam_width: int = Field(
        default=5,
        ge=1,
        description="Number of candidates to maintain per sequence (K). This overrides "
                   "any batch_size parameters of individual generators."
    )
    num_candidates: int = Field(
        default=10,
        ge=1,
        description="Number of candidates to generate per beam candidate (N)"
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature for candidate generation"
    )
    verbose: bool = Field(
        default=True,
        description="Whether to print progress information"
    )


@OptimizerRegistry.register(
    key="beam-search",
    config=BeamSearchOptimizerConfig,
    description="Beam search optimizer that processes segments sequentially with context accumulation",
)
class BeamSearchOptimizer(Optimizer):
    # Class attribute: Config class for this optimizer
    config_class = BeamSearchOptimizerConfig
    """
    Beam search optimizer that processes segments sequentially with context accumulation.

    This generator implements a beam search where:
    1. Segments are processed one at a time, in order
    2. For each segment, the top K sequences accumulated from previous segments are used as prompts
    3. Generators are applied within each segment to generate num_candidates
    4. Constraints are evaluated on concatenated sequences after each segment
    5. Top K combinations are selected and used as prompts for the next segment

    **Key Features:**
    - Processes segments sequentially (not independently)
    - Accumulates context from previous segments as prompts
    - Applies constraints after each segment to guide optimization
    - Maintains beam search across segment boundaries
    - Freezes optimization of earlier segments as new segments are added

    **Important: Generator Batch Size Override**
    - Generator `batch_size` parameters are ignored during beam search
    - The `beam_width` parameter controls how many sequences are maintained (K)
    - The `num_candidates` parameter controls how many candidates are generated per beam (N)
    - Generators are applied to individual sequences, not batches

    Examples:
        Basic beam search:
        >>> constructs = [Construct([segment1, segment2, segment3])]
        >>> config = BeamSearchOptimizerConfig(
        ...     beam_width=5,
        ...     num_candidates=10,
        ...     temperature=1.0
        ... )
        >>> beam_search = BeamSearchOptimizer(
        ...     constructs=constructs,
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     config=config,
        ...     constraint_weights=[1.0, 2.0]
        ... )
        >>> beam_search.sample()
        >>> final_constructs = beam_search.constructs
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        config: BeamSearchOptimizerConfig,
        constraint_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize the Beam Search Optimizer.

        Args:
            constructs: List containing exactly one Construct object to optimize.
            generators: List of Generator objects for sequence modification.
            constraints: List of Constraint objects for evaluation.
            config: Configuration object containing algorithm parameters (beam_width, num_candidates, etc.).
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.

        Raises:
            ValueError: If no constructs are provided or if more than one construct is provided.
        """
        # Validate constructs parameter
        if len(constructs) == 0:
            raise ValueError("At least one construct must be provided")
        if len(constructs) > 1:
            raise ValueError(f"BeamSearchOptimizer only supports a single construct, but {len(constructs)} constructs were provided")

        # BeamSearchOptimizer uses beam_width as its batch_size
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
            batch_size=config.beam_width,
        )

        # Store config
        self.config = config

        self.beam_width = config.beam_width
        self.num_candidates = config.num_candidates
        self.temperature = config.temperature
        self.verbose = config.verbose

        # Cache expensive operations
        self._segment_generators_map: Dict[Segment, List[Generator]] = {}

        # Initialize beam candidates for each segment
        self._initialize_beam_candidates()

    def _initialize_beam_candidates(self) -> None:
        """Initialize beam candidates by calling create_batch(K) on each segment."""
        for construct in self.constructs:
            for segment in construct.segments:
                if len(segment.batch_sequences) != self.beam_width:
                    segment.create_batch(self.beam_width)

        # Populate caches for expensive operations
        self._populate_caches()

        # After adjusting segment batch sizes, recreate constraints with correct batch size
        self._recreate_constraints_with_correct_batch_size()

    def _recreate_constraints_with_correct_batch_size(self) -> None:
        """Recreate constraints with the correct batch size after segments have been adjusted."""
        # Note: Constraints now dynamically compute batch_size from their input segments,
        # so no manual adjustment is needed. This method is kept for backwards compatibility
        # but is now a no-op.
        pass

    def _generate_candidates_for_segment_with_prompts(self, segment: Segment, prompts: List[str]) -> List[Sequence]:
        """
        Generate candidates for a segment using accumulated prompts from previous segments.

        Args:
            segment: The segment to generate candidates for
            prompts: List of accumulated sequences from previous segments to use as prompts

        Returns:
            List of generated candidate sequences with metadata
        """
        candidates = []

        # Create a temporary segment for generation
        temp_segment = Segment(sequence_type=segment.sequence_type)
        temp_sequence = Sequence(sequence="", sequence_type=segment.sequence_type, metadata={})
        temp_segment.batch_sequences = [temp_sequence]

        # Get generators assigned to this segment
        segment_generators = self._get_segment_generators(segment)

        # Get the segment's current sequence to use as the base prompt
        # For now, always use the current sequence in the segment
        # This will be the extended sequence from the previous sample for subsequent segments
        segment_current_sequence = segment.batch_sequences[0].sequence if segment.batch_sequences else ""

        # Generate candidates for each prompt (each prompt represents a beam from previous segments)
        for prompt_idx, accumulated_prompt in enumerate(prompts):
            for candidate_idx in range(self.num_candidates):
                full_prompt = accumulated_prompt + segment_current_sequence

                # Initialize temp_sequence with the full prompt and metadata
                temp_sequence.sequence = full_prompt
                temp_sequence._metadata = {
                    "accumulated_prompt": accumulated_prompt,
                    "segment_current_sequence": segment_current_sequence,
                    "full_prompt": full_prompt,
                    "prompt_idx": prompt_idx,
                    "candidate_idx": candidate_idx,
                    "generation_steps": []
                }

                # Apply each generator assigned to this segment in sequence
                current_prompt = full_prompt  # Start with the full prompt
                final_extended_sequence = segment_current_sequence  # Start with current segment sequence

                for generator_idx, generator in enumerate(segment_generators):
                    accepts_prompts = self._is_extension_based_generator(generator)

                    # Update temp_sequence with current prompt for this generator
                    temp_sequence.sequence = current_prompt

                    # Apply the generator to get a new sequence
                    new_sequence = self._apply_generator(
                        generator, temp_segment, temp_sequence,
                        accepts_prompts
                    )

                    if new_sequence:
                        # Determine if this is an extension-based or mutation-based generator
                        is_extension_based = self._is_extension_based_generator(generator)

                        if is_extension_based:
                            # For extension-based generators, extract the extension and add it
                            if new_sequence.startswith(current_prompt):
                                segment_extension = new_sequence[len(current_prompt):]
                            else:
                                segment_extension = new_sequence
                            final_extended_sequence += segment_extension
                        else:
                            # For mutation-based generators, replace the segment sequence
                            # Extract only the segment part from the full sequence
                            if new_sequence.startswith(accumulated_prompt):
                                segment_part = new_sequence[len(accumulated_prompt):]
                            else:
                                # If the sequence doesn't start with the prompt, use the whole sequence
                                # but this should be the same length as the original segment
                                segment_part = new_sequence

                            # For mutation-based generators, the segment part should be the same length
                            # as the original segment sequence
                            if len(segment_part) == len(segment_current_sequence):
                                final_extended_sequence = segment_part
                            else:
                                # Fallback: use the segment part as is
                                final_extended_sequence = segment_part

                        # Record this generation step
                        self._record_generation_step(
                            temp_sequence._metadata, generator, generator_idx,
                            current_prompt, new_sequence, temp_segment
                        )

                        # For the next generator, update the prompt to include this generator's output
                        if generator_idx < len(segment_generators) - 1:
                            current_prompt = new_sequence

                final_candidate = self._create_final_candidate(
                    final_extended_sequence, segment.sequence_type, temp_sequence._metadata
                )
                candidates.append(final_candidate)

        return candidates

    def _apply_generator(self, generator: Generator, temp_segment: Segment, temp_sequence: Sequence,
                        accepts_prompts: bool) -> Optional[str]:
        """Apply a single generator and return the new sequence."""
        # Temporarily assign the generator to our temp segment
        original_generator_output = generator._generator_output
        generator._generator_output = temp_segment

        try:
            # Apply the generator
            if accepts_prompts:
                # Always pass the current sequence as prompt if the generator accepts prompts
                generator.sample(prompt_seqs=[temp_sequence.sequence])
            else:
                generator.sample()

            # Get the generated output
            outputs = generator.get_generator_outputs()
            if outputs and outputs[0] and outputs[0][0]:
                return outputs[0][0].sequence

            return None

        finally:
            # Restore the original generator assignment
            generator._generator_output = original_generator_output



    def _record_generation_step(self, metadata: Dict[str, Any], generator: Generator, generator_idx: int,
                              current_sequence: str, new_sequence: str, temp_segment: Segment) -> None:
        """Record metadata about a generation step."""
        step_metadata = {
            "generator_type": type(generator).__name__,
            "step_idx": generator_idx,
            "input_sequence": current_sequence,
            "output_sequence": new_sequence,
            "generator_metadata": temp_segment.batch_sequences[0]._metadata
        }
        metadata["generation_steps"].append(step_metadata)

    def _create_final_candidate(self, final_sequence: str, sequence_type: SequenceType,
                              metadata: Dict[str, Any]) -> Sequence:
        """Create the final candidate sequence with metadata."""
        return Sequence(
            sequence=final_sequence,
            sequence_type=sequence_type,
            metadata=metadata.copy()
        )

    def _evaluate_concatenated_combinations(self, combinations: List[Dict[int, Sequence]], construct: Construct, current_segment_idx: int) -> List[Tuple[Dict[int, Sequence], float]]:
        """
        Evaluates concatenated sequences from all segments up to current one.

        A "combination" in beam search represents a complete set of sequence choices across
        multiple segments. Each combination is a dictionary where:
        - Keys are segment indices (int)
        - Values are Sequence objects representing the chosen sequence for that segment

        For example, a combination might look like:
        {0: Sequence("ATGCTAGCTA"), 1: Sequence("GCTAGCTAGC"), 2: Sequence("TAGCTAGCTA")}
        This represents choosing "ATGCTAGCTA" for segment 0, "GCTAGCTAGC" for segment 1, etc.

        The method concatenates these sequences in order and evaluates them against constraints
        to compute an overall energy score (lower is better).

        Args:
            combinations: List of combinations to evaluate. Each combination is a Dict[int, Sequence]
                         mapping segment indices to their chosen sequences.
            construct: The construct containing all segments
            current_segment_idx: Index of the current segment being processed

        Returns:
            List of tuples (combination, energy_score) for all evaluated combinations,
            where energy_score is the total constraint violation score (lower is better)
        """
        evaluated_combinations = []

        for combination in combinations:
            if current_segment_idx == 0:
                # For first segment, evaluate just that segment's candidate
                concatenated_sequence = combination[current_segment_idx].sequence
            else:
                # For subsequent segments, create concatenated sequence for evaluation
                # Only include segments that have been processed so far (up to current_segment_idx)
                concatenated_sequence = ""
                for seg_idx in range(current_segment_idx + 1):
                    if seg_idx in combination:
                        concatenated_sequence += combination[seg_idx].sequence
                    else:
                        concatenated_sequence += ""  # Empty for missing segments

            # Evaluate constraints on the concatenated sequence
            total_energy = 0.0
            for constraint in self.constraints:
                # Create a temporary sequence object for evaluation
                temp_sequence = Sequence(
                    sequence=concatenated_sequence,
                    sequence_type=SequenceType.DNA,  # Assume DNA for concatenated evaluation
                    metadata={"concatenated": True}
                )

                energy = constraint.scoring_function(temp_sequence, **constraint.scoring_function_config)
                total_energy += energy

            # Add evaluation metadata to each candidate
            for seg_idx, candidate in combination.items():
                candidate._metadata["evaluation_energy"] = total_energy

            evaluated_combinations.append((combination, total_energy))

        if self.verbose:
            print(f"Evaluated {len(evaluated_combinations)} concatenated combinations")

        return evaluated_combinations



    def _select_top_combinations(self, evaluated_combinations: List[Tuple[Dict[int, Sequence], float]]) -> List[Tuple[Dict[int, Sequence], float]]:
        """
        Select the top-K combinations based on energy scores.

        This method implements the core beam search selection mechanism. It takes the
        evaluated combinations (each with their computed energy score) and selects
        the top K combinations to maintain in the beam. The beam width (K) determines
        how many promising sequence combinations are kept for the next iteration.

        Args:
            evaluated_combinations: List of (combination, energy) tuples where:
                - combination: Dict[int, Sequence] mapping segment indices to sequences
                - energy: float representing the total constraint violation score

        Returns:
            Top-K combinations sorted by energy (lower is better), maintaining the beam
            for the next segment's optimization
        """
        # Use heapq.nlargest for efficient top-K selection (lower energy is better, so we negate)
        # Note: heapq.nlargest returns items in descending order, so we negate energy scores
        # to get the lowest energy scores first
        top_combinations = heapq.nlargest(
            self.beam_width,
            evaluated_combinations,
            key=lambda x: -x[1]  # Negate energy so lower values come first
        )

        if self.verbose:
            if top_combinations:
                best_energy = top_combinations[0][1]
                print(f"Selected top {len(top_combinations)} combinations, best energy: {best_energy:.4f}")
            else:
                print("Warning: No valid combinations found after constraint evaluation")

        return top_combinations

    def _update_segments_with_combinations(self, construct: Construct, top_combinations: List[Tuple[Dict[int, Sequence], float]]) -> None:
        """
        Update all segments with the best combinations from beam search.

        This method distributes the top-K combinations across the beam structure.
        Each combination represents a complete set of sequence choices across all
        segments processed so far. The method updates each segment's batch_sequences
        to maintain the beam candidates for the next iteration.

        The beam structure ensures that:
        - Each segment maintains exactly beam_width sequences
        - Sequences are organized by their beam index (0 to beam_width-1)
        - Each beam represents a different promising path through the sequence space

        Args:
            construct: The construct to update with new beam candidates
            top_combinations: Top-K combinations to distribute, where each combination is:
                - Dict[int, Sequence]: mapping segment indices to their chosen sequences
                - float: the energy score for this combination (lower is better)
        """
        # Check if we have any combinations to work with
        if not top_combinations:
            raise ValueError("Cannot update segments: no valid combinations were found during beam search")

        segments = construct.segments

        # Initialize beam candidates for each segment
        for segment in segments:
            while len(segment.batch_sequences) < self.beam_width:
                new_sequence = Sequence(sequence="", sequence_type=segment.sequence_type, metadata={})
                segment.batch_sequences.append(new_sequence)

        # Distribute combinations across beam width
        for beam_idx, (combination, energy) in enumerate(top_combinations):
            if beam_idx >= self.beam_width:
                break

            for seg_idx, segment in enumerate(segments):
                if seg_idx in combination:
                    candidate = combination[seg_idx]
                    existing_seq = segment.batch_sequences[beam_idx]

                    # Update the sequence
                    existing_seq.sequence = candidate.sequence
                    existing_seq._metadata.clear()
                    existing_seq._metadata.update(candidate._metadata)
                    existing_seq._metadata["energy"] = energy
                    existing_seq._metadata["beam_idx"] = beam_idx

        # Trim excess sequences if we have more than beam_width
        for segment in segments:
            if len(segment.batch_sequences) > self.beam_width:
                segment.batch_sequences = segment.batch_sequences[:self.beam_width]

    def sample(self) -> List[Construct]:
        """
        Run beam search across all segments with context accumulation.

        This method implements the core beam search algorithm by processing segments
        maintaining beams of the most promising sequence combinations.

        **Beam Search Process:**
        1. For each segment, use accumulated sequences from previous segments as prompts
        2. Generate candidates for the current segment using these prompts
        3. Create combinations by pairing new candidates with sequences from previous segments
        4. Evaluate constraints on concatenated sequences to compute energy scores
        5. Select top-K combinations (lowest energy) to maintain in the beam
        6. Use these combinations as prompts for the next segment
        7. Continue until all segments are processed

        **Combination Management:**
        - Combinations are dictionaries mapping segment indices to Sequence objects
        - Each combination represents a complete path through the sequence space
        - The beam maintains K such combinations, where K = beam_width
        - Combinations are evaluated by concatenating sequences and applying constraints
        - Only the best K combinations (lowest energy scores) are kept for the next iteration

        Returns:
            List containing the construct with updated beam candidates, where each
            segment maintains exactly beam_width sequences representing the best
            combinations found during the search.
        """
        for construct in self.constructs:
            if self.verbose:
                print(f"Processing {len(construct.segments)} segments with beam search")

            # Initialize beam candidates for the first segment
            # Use existing beam candidates from previous sample if available, otherwise start with empty sequences
            if hasattr(self, '_beam_candidates') and self._beam_candidates:
                beam_candidates = self._beam_candidates.copy()
                if self.verbose:
                    print(f"Using existing beam candidates from previous sample")
            else:
                beam_candidates = [""] * self.beam_width
                if self.verbose:
                    print(f"Starting with empty beam candidates")

            # Process each segment sequentially
            for segment_idx, segment in enumerate(construct.segments):
                if self.verbose:
                    print(f"\n--- Processing Segment {segment_idx + 1}/{len(construct.segments)} ---")
                    print(f"Using {len(beam_candidates)} beam candidates as prompts")

                # Step 1: Generate candidates for current segment using beam prompts
                # For each beam candidate, extract only the sequences from segments that have been processed so far
                current_beam_prompts = []
                for beam_idx, beam_sequence in enumerate(beam_candidates):
                    # Extract only the sequences from segments processed so far in this sample
                    current_pos = 0
                    accumulated_prompt = ""

                    for seg_idx in range(segment_idx):
                        prev_segment = construct.segments[seg_idx]

                        segment_length = self._get_segment_length(prev_segment)

                        # Extract this segment's contribution from the beam sequence
                        if current_pos < len(beam_sequence):
                            segment_contribution = beam_sequence[current_pos:current_pos + segment_length]
                            current_pos += segment_length
                            accumulated_prompt += segment_contribution
                        else:
                            break

                    current_beam_prompts.append(accumulated_prompt)

                segment_candidates = self._generate_candidates_for_segment_with_prompts(segment, current_beam_prompts)

                if self.verbose:
                    print(f"Generated {len(segment_candidates)} candidates for segment {segment_idx + 1}")

                # Step 2: Create combinations with previous segments
                # A combination is a Dict[int, Sequence] mapping segment indices to their chosen sequences
                # For the first segment, just use the candidates directly
                if segment_idx == 0:
                    # Create combinations where each candidate is paired with empty previous segments
                    # Each combination represents a single candidate for segment 0
                    combinations = []
                    for candidate in segment_candidates:
                        combination = {segment_idx: candidate}
                        combinations.append(combination)
                else:
                    # Create combinations with all previous segments
                    # Each combination will contain sequences from segments 0 to segment_idx
                    previous_segments = construct.segments[:segment_idx]

                    # Create combinations by pairing current candidates with previous beam candidates
                    # IMPORTANT: Each beam must maintain continuity - candidates from different beams cannot be mixed
                    # This ensures that combinations represent coherent paths through the sequence space
                    combinations = []

                    # Group candidates by their beam (prompt_idx)
                    candidates_by_beam = {}
                    for candidate in segment_candidates:
                        prompt_idx = candidate._metadata.get("prompt_idx", 0)
                        if prompt_idx not in candidates_by_beam:
                            candidates_by_beam[prompt_idx] = []
                        candidates_by_beam[prompt_idx].append(candidate)

                    # For each beam, create combinations only with candidates from the same beam
                    for prompt_idx, beam_prompt in enumerate(beam_candidates):
                        if prompt_idx in candidates_by_beam:
                            for candidate in candidates_by_beam[prompt_idx]:
                                combination = {segment_idx: candidate}
                                # For previous segments, we need to extract their individual contributions
                                # The beam_prompt contains the full sequence, but we need to split it
                                # into individual segment contributions
                                current_pos = 0
                                for prev_seg_idx, prev_segment in enumerate(previous_segments):
                                    segment_length = self._get_segment_length(prev_segment)

                                    # Extract this segment's contribution from the beam sequence
                                    if current_pos < len(beam_prompt):
                                        segment_contribution = beam_prompt[current_pos:current_pos + segment_length]
                                        current_pos += segment_length
                                    else:
                                        segment_contribution = ""

                                    # Create a sequence object for this segment's contribution
                                    prev_sequence = Sequence(
                                        sequence=segment_contribution,
                                        sequence_type=prev_segment.sequence_type,
                                        metadata={"beam": True, "prompt_idx": prompt_idx}
                                    )
                                    combination[prev_seg_idx] = prev_sequence
                                combinations.append(combination)

                # Step 3: Evaluate combinations with constraints
                # For all segments, evaluate constraints on the concatenated sequences
                evaluated_combinations = self._evaluate_concatenated_combinations(combinations, construct, segment_idx)

                # Step 4: Select top-K combinations
                top_combinations = self._select_top_combinations(evaluated_combinations)

                # Check if no valid combinations were found
                if not top_combinations:
                    raise RuntimeError(f"No valid combinations found for segment {segment_idx + 1}. "
                                     f"All {len(evaluated_combinations)} candidate combinations violated constraints. "
                                     f"Consider relaxing constraints or adjusting generator parameters.")

                # Unified logging for all segments
                if self.verbose:
                    print(f"Segment {segment_idx + 1} Candidates:")
                    existing_sequence = segment.batch_sequences[0].sequence if segment.batch_sequences else ""
                    print(f"    Existing sequence: '{existing_sequence}'")

                    # Group candidates by beam (prompt_idx)
                    candidates_by_beam = {}
                    for candidate in segment_candidates:
                        prompt_idx = candidate._metadata.get("prompt_idx", 0)
                        if prompt_idx not in candidates_by_beam:
                            candidates_by_beam[prompt_idx] = []
                        candidates_by_beam[prompt_idx].append(candidate)

                    # Show candidates grouped by beam
                    for prompt_idx in sorted(candidates_by_beam.keys()):
                        beam_candidates_list = candidates_by_beam[prompt_idx]

                        # Get the prompt used for this beam
                        if segment_idx == 0:
                            # For first segment, just show the existing sequence
                            formatted_prompt = existing_sequence
                        else:
                            # For subsequent segments, show the accumulated sequence
                            if prompt_idx < len(current_beam_prompts):
                                beam_prompt = current_beam_prompts[prompt_idx]
                                # Use the same sequence as the generator prompt (no separators)
                                formatted_prompt = beam_prompt + existing_sequence
                            else:
                                formatted_prompt = "unknown"

                        print(f"    Candidates generated from '{formatted_prompt}' (Beam {prompt_idx}):")

                        for candidate_idx, candidate in enumerate(beam_candidates_list):
                            # Check if this candidate is in the top combinations and get its energy
                            is_selected = False
                            energy = candidate._metadata.get("evaluation_energy", 0.0)  # Get energy from candidate metadata

                            # Check if this candidate is selected (in top combinations)
                            for combination, combo_energy in top_combinations:
                                if segment_idx in combination and combination[segment_idx].sequence == candidate.sequence:
                                    is_selected = True
                                    break

                            status = "✓ SELECTED" if is_selected else "✗ REJECTED"
                            # Show the full segment sequence (not just the extension)
                            final_sequence = candidate._metadata.get("final_sequence", candidate.sequence)
                            print(f"        {status} '{final_sequence}' (Energy: {energy:.4f}, Candidate #{candidate_idx})")

                            # Show generator steps
                            generation_steps = candidate._metadata.get("generation_steps", [])
                            for step_idx, step in enumerate(generation_steps):
                                generator_num = step_idx + 1
                                input_seq = step.get("input_sequence", "")
                                output_seq = step.get("output_sequence", "")

                                # Calculate extension
                                if output_seq.startswith(input_seq):
                                    extension = output_seq[len(input_seq):]
                                else:
                                    extension = output_seq

                                # Show the actual sequences that the generators work with (no separators)
                                print(f"            Generator {generator_num}: prompt='{input_seq}' -> generated '{output_seq}' -> extension '{extension}'")
                    print()

                # Step 5: Update beam candidates for next segment
                beam_candidates = []
                for combination, energy in top_combinations:
                    if segment_idx == 0:
                        # For first segment, just use the candidate sequence
                        concatenated_sequence = combination[segment_idx].sequence
                    else:
                        # For subsequent segments, build the full accumulated sequence
                        # by concatenating all segment contributions in order
                        concatenated_sequence = ""
                        for seg_idx in range(segment_idx + 1):
                            if seg_idx in combination:
                                concatenated_sequence += combination[seg_idx].sequence

                    beam_candidates.append(concatenated_sequence)

                if self.verbose:
                    print(f"Updated accumulated sequences for next segment")
                    if top_combinations:
                        print(f"Best energy: {top_combinations[0][1]:.4f}")
                    else:
                        print("Best energy: No valid combinations found")
                    print(f"Top {len(beam_candidates)} beam sequences:")
                    for beam_idx, sequence in enumerate(beam_candidates):
                        # Show the full accumulated sequence (no separators for now)
                        print(f"  Beam {beam_idx}: '{sequence}'")

            # Final step: Update all segments with their individual extended sequences
            # Each segment should contain its own extended sequence, not the full accumulated sequence
            final_combinations = []
            for acc_idx, beam_sequence in enumerate(beam_candidates):
                combination = {}

                # For beam search, each segment gets its own extended sequence
                # We need to extract each segment's contribution from the accumulated sequence
                current_pos = 0
                for seg_idx, segment in enumerate(construct.segments):
                    segment_length = self._get_segment_length(segment)

                    # Extract this segment's contribution from the beam sequence
                    if current_pos < len(beam_sequence):
                        segment_sequence = beam_sequence[current_pos:current_pos + segment_length]
                        current_pos += segment_length
                    else:
                        segment_sequence = ""

                    seq_obj = Sequence(
                        sequence=segment_sequence,
                        sequence_type=segment.sequence_type,
                        metadata={"combination_idx": acc_idx, "segment_idx": seg_idx}
                    )
                    combination[seg_idx] = seq_obj

                final_combinations.append((combination, 0.0))  # Energy already evaluated

            # Update segments with final combinations
            self._update_segments_with_combinations(construct, final_combinations)

            # Store beam candidates for the next sample
            self._beam_candidates = beam_candidates.copy()

        # Add final state to history
        energy_scores = []
        for construct in self.constructs:
            # Calculate min energy across all batch sequences for this construct
            construct_energies = []
            for seq in construct.batch_sequences:
                if hasattr(seq, '_metadata') and 'energy' in seq._metadata:
                    construct_energies.append(seq._metadata['energy'])
            # Use min energy if available, otherwise 0
            min_energy = min(construct_energies) if construct_energies else 0.0
            energy_scores.append(min_energy)

        # Set energy_scores attribute
        if not energy_scores:
            energy_scores = [0.0] * max(1, len(self.constructs))
        self.energy_scores = energy_scores

        history_entry = {
            "time_step": 1,  # BeamSearch doesn't have steps, use 1 for final state
            "energy_scores": energy_scores,
            "constructs": copy.deepcopy(self.constructs)
        }
        self.history.append(history_entry)

        if self.verbose:
            self._log_progress()

        return self.constructs

    def _log_progress(self) -> None:
        """Log current beam search progress."""
        total_candidates = 0
        total_energy = 0.0

        for construct in self.constructs:
            for segment in construct.segments:
                total_candidates += len(segment.batch_sequences)
                for sequence in segment.batch_sequences:
                    energy = sequence._metadata.get("energy", 0.0)
                    total_energy += energy

        avg_energy = total_energy / total_candidates if total_candidates > 0 else 0.0
        print(f"BeamSearchOptimizer: {total_candidates} total candidates, avg energy: {avg_energy:.4f}")

    def _is_extension_based_generator(self, generator) -> bool:
        """
        Determine if a generator is extension-based or mutation-based.

        Args:
            generator: The generator to check

        Returns:
            True if the generator is extension-based, False if mutation-based
        """
        # Extension-based generators have prepend_prompt attribute
        # Mutation-based generators don't have this attribute
        return hasattr(generator, 'prepend_prompt') and generator.prepend_prompt

    def _populate_caches(self) -> None:
        """Populate caches for expensive operations to avoid repeated calculations."""
        # Cache segment generators mapping
        for segment in self.constructs[0].segments:
            self._segment_generators_map[segment] = [
                gen for gen in self.generators if gen._generator_output == segment
            ]

    def _get_segment_generators(self, segment: Segment) -> List[Generator]:
        """Get cached generators for a segment."""
        return self._segment_generators_map.get(segment, [])

    def _get_segment_length(self, segment: Segment) -> int:
        """Calculate the total length a segment contributes to the accumulated sequence."""
        initial_sequence = segment.batch_sequences[0].sequence if segment.batch_sequences else ""
        initial_length = len(initial_sequence)

        segment_generators = self._get_segment_generators(segment)

        # Calculate the total contribution from generators
        total_generated_length = 0
        for generator in segment_generators:
            if self._is_extension_based_generator(generator):
                if hasattr(generator, 'sequence_length'):
                    total_generated_length += generator.sequence_length
                elif hasattr(generator, 'n_tokens'):
                    total_generated_length += generator.n_tokens

        return initial_length + total_generated_length
