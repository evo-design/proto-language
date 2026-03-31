"""
proto_language/language/constraint/rna_secondary_structure/structure_similarity_constraint.py

Constraint functions for RNA secondary structure similarity comparison.

Uses ViennaRNA for secondary structure prediction and provides four independent
constraint functions for different aspects of structural comparison.
"""

from __future__ import annotations

from logging import getLogger
from typing import List, Set, Tuple

import numpy as np
from proto_tools import ViennaRNAConfig, ViennaRNAInput, run_viennarna

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence

logger = getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _fold_sequences(
    sequences: List[str],
    temperature: float = 37.0,
) -> List[Tuple[str, float]]:
    """
    Fold multiple sequences using ViennaRNA.

    Args:
        sequences (list[str]): List of RNA/DNA sequences
        temperature (float): Folding temperature in Celsius

    Returns:
        list[tuple[str, float]]: List of (structure, mfe) tuples
    """
    inputs = ViennaRNAInput(sequences=sequences)
    config = ViennaRNAConfig(temperature=temperature)
    output = run_viennarna(inputs, config)
    return [(r.structure, r.mfe) for r in output.results]


def _get_base_pairs(structure: str) -> Set[Tuple[int, int]]:
    """Extract base pairs as (i, j) tuples from dot-bracket notation."""
    pairs = set()
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                pairs.add((stack.pop(), i))
            else:
                logger.debug(f"Unmatched closing bracket at position {i} in structure")
    if stack:
        logger.debug(f"Unmatched opening brackets at positions {stack} in structure")
    return pairs


def _extract_structural_motifs(structure: str) -> List[str]:
    """Extract structural motifs (stems, hairpins, bulges, etc.)."""
    if not structure:
        return []

    motifs = []

    # 1. Stem-loop patterns
    i = 0
    while i < len(structure):
        if structure[i] == '(':
            stem_count = 1
            i += 1

            while i < len(structure) and structure[i] == '(':
                stem_count += 1
                i += 1

            loop_count = 0
            while i < len(structure) and structure[i] == '.':
                loop_count += 1
                i += 1

            close_count = 0
            while i < len(structure) and structure[i] == ')':
                close_count += 1
                i += 1

            if close_count > 0:
                if loop_count == 0:
                    motifs.append(f"STEM_{min(stem_count, close_count)}")
                else:
                    motifs.append(f"HAIRPIN_{min(stem_count, close_count)}:{loop_count}")
        else:
            i += 1

    # 2. Bulge patterns
    bulge_pattern = ""
    in_stem = False
    for char in structure:
        if char == '(':
            if not in_stem:
                bulge_pattern = '('
                in_stem = True
            else:
                bulge_pattern += char
        elif char == ')':
            bulge_pattern += char
            if bulge_pattern.count('(') == bulge_pattern.count(')'):
                dots = bulge_pattern.count('.')
                stems = bulge_pattern.count('(')
                if dots > 0 and stems > 1:
                    motifs.append(f"BULGE_{stems}:{dots}")
                bulge_pattern = ""
                in_stem = False
        elif char == '.' and in_stem:
            bulge_pattern += char

    # 3. Nesting depth
    max_depth = 0
    current_depth = 0
    for char in structure:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    if max_depth > 0:
        motifs.append(f"DEPTH_{max_depth}")

    # 4. Unpaired regions
    current_unpaired = 0
    for char in structure:
        if char == '.':
            current_unpaired += 1
        else:
            if current_unpaired >= 3:
                motifs.append(f"UNPAIRED_{min(current_unpaired, 10)}")
            current_unpaired = 0
    if current_unpaired >= 3:
        motifs.append(f"UNPAIRED_{min(current_unpaired, 10)}")

    return motifs


def _extract_structure_features(structure: str, mfe: float) -> np.ndarray:
    """Convert structure into a 10-dimensional feature vector."""
    if not structure:
        return np.zeros(10)

    length = len(structure)
    pairs = _get_base_pairs(structure)
    num_pairs = len(pairs)
    pairing_ratio = num_pairs / length if length > 0 else 0

    # Extract stems (contiguous runs of nested base pairs)
    # A stem is a maximal set of base pairs (i,j), (i+1,j-1), (i+2,j-2), ...
    stems = []
    if pairs:
        sorted_pairs = sorted(pairs)
        visited = set()
        for i, j in sorted_pairs:
            if (i, j) in visited:
                continue
            # Trace this stem
            stem_length = 0
            ci, cj = i, j
            while (ci, cj) in pairs and (ci, cj) not in visited:
                visited.add((ci, cj))
                stem_length += 1
                ci += 1
                cj -= 1
            if stem_length > 0:
                stems.append(stem_length)

    # Extract loop regions (runs of unpaired positions)
    loops = []
    current_loop = 0
    for char in structure:
        if char == '.':
            current_loop += 1
        else:
            if current_loop > 0:
                loops.append(current_loop)
                current_loop = 0
    if current_loop > 0:
        loops.append(current_loop)

    avg_stem_length = np.mean(stems) if stems else 0
    avg_loop_length = np.mean(loops) if loops else 0
    max_stem_length = max(stems) if stems else 0
    num_stems = len(stems)
    mfe_per_nt = mfe / length if length > 0 else 0

    # Count hairpin loops (unpaired regions enclosed by a base pair)
    # A hairpin occurs when (i, j) is a pair and positions i+1 to j-1 are all unpaired
    num_hairpins = 0
    for i, j in pairs:
        if j > i + 1:  # There's space for a loop
            loop_region = structure[i+1:j]
            if all(c == '.' for c in loop_region):
                num_hairpins += 1

    return np.array([
        length,
        num_pairs,
        pairing_ratio,
        avg_stem_length,
        avg_loop_length,
        max_stem_length,
        num_stems,
        mfe,
        mfe_per_nt,
        num_hairpins,
    ])


# =============================================================================
# Config Classes
# =============================================================================

class RNAStructureConstraintBaseConfig(BaseConfig):
    """Base configuration for RNA secondary structure constraints.

    Attributes:
        reference_sequence (str): Reference RNA/DNA sequence to compare against.
            T nucleotides will be automatically converted to U for RNA folding.
            Required parameter.

        temperature (float): Folding temperature in Celsius for thermodynamic
            calculations. Affects predicted structure stability. Default: 37.0
            (physiological temperature).
    """

    reference_sequence: str = ConfigField(
        title="Reference Sequence",
        description="Reference RNA/DNA sequence to compare against",
    )
    temperature: float = ConfigField(
        title="Temperature",
        default=37.0,
        ge=-273.15,
        description="Folding temperature in Celsius",
    )


class RNAPropertySimilarityConfig(RNAStructureConstraintBaseConfig):
    """Configuration for structural property similarity constraint.

    Inherits from ``RNAStructureConstraintBaseConfig``.

    Attributes:
        reference_sequence (str): Reference RNA/DNA sequence to compare against.
            Inherited from ``RNAStructureConstraintBaseConfig``. Required.

        temperature (float): Folding temperature in Celsius. Inherited from
            ``RNAStructureConstraintBaseConfig``. Default: 37.0.

        length_weight (float): Weight for length similarity in the combined score.
            The pairing ratio similarity receives weight (1 - length_weight).
            Range: 0.0-1.0. Default: 0.6.
    """

    length_weight: float = ConfigField(
        title="Length Weight",
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for length similarity",
    )


class RNAMotifSimilarityConfig(RNAStructureConstraintBaseConfig):
    """Configuration for structural motif similarity constraint.

    Compares extracted structural motifs (stems, hairpins, bulges, nesting depth,
    unpaired regions) using Jaccard similarity.

    Inherits from ``RNAStructureConstraintBaseConfig``.

    Attributes:
        reference_sequence (str): Reference RNA/DNA sequence to compare against.
            Inherited from ``RNAStructureConstraintBaseConfig``. Required.

        temperature (float): Folding temperature in Celsius. Inherited from
            ``RNAStructureConstraintBaseConfig``. Default: 37.0.
    """
    pass


class RNAFeatureSimilarityConfig(RNAStructureConstraintBaseConfig):
    """Configuration for feature vector similarity constraint.

    Compares 10-dimensional structural feature vectors using cosine similarity.
    Features include: length, number of pairs, pairing ratio, stem statistics
    (avg, max, count), loop statistics, MFE, MFE per nucleotide, and hairpin count.

    Inherits from ``RNAStructureConstraintBaseConfig``.

    Attributes:
        reference_sequence (str): Reference RNA/DNA sequence to compare against.
            Inherited from ``RNAStructureConstraintBaseConfig``. Required.

        temperature (float): Folding temperature in Celsius. Inherited from
            ``RNAStructureConstraintBaseConfig``. Default: 37.0.
    """
    pass


class RNABasePairSimilarityConfig(RNAStructureConstraintBaseConfig):
    """Configuration for base pair similarity constraint.

    Compares sets of base pairs (i, j) using Jaccard similarity. This is a strict
    positional comparison requiring identical base pair positions for a match.

    Inherits from ``RNAStructureConstraintBaseConfig``.

    Attributes:
        reference_sequence (str): Reference RNA/DNA sequence to compare against.
            Inherited from ``RNAStructureConstraintBaseConfig``. Required.

        temperature (float): Folding temperature in Celsius. Inherited from
            ``RNAStructureConstraintBaseConfig``. Default: 37.0.

        max_length_ratio_diff (float): Maximum allowed relative length difference
            between reference and proposal sequences. If exceeded, similarity
            returns 0.0 (and score returns 1.0, i.e., the worst value).
            Computed as |len1 - len2| / max(len1, len2).
            Range: 0.0-1.0. Default: 0.5.
    """

    max_length_ratio_diff: float = ConfigField(
        title="Max Length Ratio Difference",
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum allowed length ratio difference",
    )


# =============================================================================
# Constraint 1: Structural Property Similarity
# =============================================================================

@constraint(
    key="rna-property-similarity",
    label="RNA Structural Property Similarity",
    config=RNAPropertySimilarityConfig,
    description="Compare RNA structural properties (length, pairing ratio) against a reference.",
    uses_gpu=False,
    tools_called=["viennarna-prediction"],
    category="rna_secondary_structure",
    supported_sequence_types=["dna", "rna"],
    num_input_sequences_per_tuple=1,
)
def rna_property_similarity_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: RNAPropertySimilarityConfig,
) -> List[float]:
    """
    Compare basic structural properties (length, pairing ratio) between proposals
    and reference.

    Returns 1 - similarity (so 0 is perfect match, 1 is worst).

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (RNAPropertySimilarityConfig): Constraint configuration controlling evaluation parameters.
    """
    # Fold reference
    ref_results = _fold_sequences([config.reference_sequence], config.temperature)
    ref_structure, _ = ref_results[0]
    if not ref_structure:
        logger.warning("Reference folding failed, returning worst scores")
        return [1.0] * len(input_sequences)

    ref_len = len(ref_structure)
    ref_pairs = ref_structure.count('(')
    ref_ratio = ref_pairs / ref_len if ref_len > 0 else 0

    # Fold all proposals
    proposal_seqs = [seq.sequence for (seq,) in input_sequences]
    cand_results = _fold_sequences(proposal_seqs, config.temperature)

    scores = []
    for (cand_structure, _), (seq,) in zip(cand_results, input_sequences):
        if not cand_structure:
            scores.append(1.0)
            continue

        cand_len = len(cand_structure)
        cand_pairs = cand_structure.count('(')
        cand_ratio = cand_pairs / cand_len if cand_len > 0 else 0

        # Length similarity
        length_sim = 1.0 - abs(ref_len - cand_len) / max(ref_len, cand_len)

        # Pairing ratio similarity
        pairing_sim = 1.0 - abs(ref_ratio - cand_ratio)

        # Combined similarity
        similarity = config.length_weight * length_sim + (1 - config.length_weight) * pairing_sim

        # Store metadata
        seq._metadata.update({
            "rna_property_similarity": similarity,
            "length_similarity": length_sim,
            "pairing_ratio_similarity": pairing_sim,
            "structure": cand_structure,
        })

        # Return 1 - similarity (constraint convention: lower is better)
        scores.append(1.0 - similarity)

    return scores


# =============================================================================
# Constraint 2: Structural Motif Similarity
# =============================================================================

@constraint(
    key="rna-motif-similarity",
    label="RNA Structural Motif Similarity",
    config=RNAMotifSimilarityConfig,
    description="Compare RNA structural motifs (stems, hairpins, bulges) using Jaccard similarity.",
    uses_gpu=False,
    tools_called=["viennarna-prediction"],
    category="rna_secondary_structure",
    supported_sequence_types=["dna", "rna"],
    num_input_sequences_per_tuple=1,
)
def rna_motif_similarity_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: RNAMotifSimilarityConfig,
) -> List[float]:
    """
    Compare structural motifs (stems, hairpins, bulges) between proposals and
    reference using Jaccard similarity.

    Returns 1 - similarity (so 0 is perfect match, 1 is worst).

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (RNAMotifSimilarityConfig): Constraint configuration controlling evaluation parameters.
    """
    # Fold reference
    ref_results = _fold_sequences([config.reference_sequence], config.temperature)
    ref_structure, _ = ref_results[0]
    if not ref_structure:
        logger.warning("Reference folding failed, returning worst scores")
        return [1.0] * len(input_sequences)
    ref_motifs = set(_extract_structural_motifs(ref_structure))

    # Fold all proposals
    proposal_seqs = [seq.sequence for (seq,) in input_sequences]
    cand_results = _fold_sequences(proposal_seqs, config.temperature)

    scores = []
    for (cand_structure, _), (seq,) in zip(cand_results, input_sequences):
        if not cand_structure:
            scores.append(1.0)
            continue

        cand_motifs = set(_extract_structural_motifs(cand_structure))

        # Jaccard similarity
        if not ref_motifs and not cand_motifs:
            similarity = 1.0
        else:
            intersection = len(ref_motifs & cand_motifs)
            union = len(ref_motifs | cand_motifs)
            similarity = intersection / union if union > 0 else 0.0

        # Store metadata
        seq._metadata.update({
            "rna_motif_similarity": similarity,
            "ref_motifs": list(ref_motifs),
            "cand_motifs": list(cand_motifs),
            "shared_motifs": list(ref_motifs & cand_motifs),
            "structure": cand_structure,
        })

        scores.append(1.0 - similarity)

    return scores


# =============================================================================
# Constraint 3: Feature Vector Similarity
# =============================================================================

@constraint(
    key="rna-feature-similarity",
    label="RNA Feature Vector Similarity",
    config=RNAFeatureSimilarityConfig,
    description="Compare RNA structures using cosine similarity of 10-dim feature vectors.",
    uses_gpu=False,
    tools_called=["viennarna-prediction"],
    category="rna_secondary_structure",
    supported_sequence_types=["dna", "rna"],
    num_input_sequences_per_tuple=1,
)
def rna_feature_similarity_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: RNAFeatureSimilarityConfig,
) -> List[float]:
    """
    Construct 10-dim feature vectors and compare using cosine similarity.

    Returns 1 - similarity (so 0 is perfect match, 1 is worst).

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (RNAFeatureSimilarityConfig): Constraint configuration controlling evaluation parameters.
    """
    # Fold reference
    ref_results = _fold_sequences([config.reference_sequence], config.temperature)
    ref_structure, ref_mfe = ref_results[0]
    if not ref_structure:
        logger.warning("Reference folding failed, returning worst scores")
        return [1.0] * len(input_sequences)
    ref_features = _extract_structure_features(ref_structure, ref_mfe)
    ref_norm = np.linalg.norm(ref_features)

    # Fold all proposals
    proposal_seqs = [seq.sequence for (seq,) in input_sequences]
    cand_results = _fold_sequences(proposal_seqs, config.temperature)

    scores = []
    for (cand_structure, cand_mfe), (seq,) in zip(cand_results, input_sequences):
        if not cand_structure:
            scores.append(1.0)
            continue

        cand_features = _extract_structure_features(cand_structure, cand_mfe)
        cand_norm = np.linalg.norm(cand_features)

        # Cosine similarity
        if ref_norm < 1e-8 or cand_norm < 1e-8:
            similarity = 0.0
        else:
            similarity = float(np.dot(ref_features, cand_features) / (ref_norm * cand_norm))

        # Store metadata
        seq._metadata.update({
            "rna_feature_similarity": similarity,
            "ref_features": ref_features.tolist(),
            "cand_features": cand_features.tolist(),
            "structure": cand_structure,
            "mfe": cand_mfe,
        })

        scores.append(1.0 - similarity)

    return scores


# =============================================================================
# Constraint 4: Base Pair Similarity
# =============================================================================

@constraint(
    key="rna-basepair-similarity",
    label="RNA Base Pair Similarity",
    config=RNABasePairSimilarityConfig,
    description="Compare RNA base pair sets using Jaccard similarity.",
    uses_gpu=False,
    tools_called=["viennarna-prediction"],
    category="rna_secondary_structure",
    supported_sequence_types=["dna", "rna"],
    num_input_sequences_per_tuple=1,
)
def rna_basepair_similarity_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: RNABasePairSimilarityConfig,
) -> List[float]:
    """
    Compare base pair sets between proposals and reference using Jaccard similarity.

    Returns 1 - similarity (so 0 is perfect match, 1 is worst).

    Args:
        input_sequences (list[Tuple[Sequence, ...]]): Mapping of segment IDs to their current sequences.
        config (RNABasePairSimilarityConfig): Constraint configuration controlling evaluation parameters.
    """
    # Fold reference
    ref_results = _fold_sequences([config.reference_sequence], config.temperature)
    ref_structure, _ = ref_results[0]
    if not ref_structure:
        logger.warning("Reference folding failed, returning worst scores")
        return [1.0] * len(input_sequences)
    ref_pairs = _get_base_pairs(ref_structure)
    ref_len = len(ref_structure)

    # Fold all proposals
    proposal_seqs = [seq.sequence for (seq,) in input_sequences]
    cand_results = _fold_sequences(proposal_seqs, config.temperature)

    scores = []
    for (cand_structure, _), (seq,) in zip(cand_results, input_sequences):
        if not cand_structure:
            scores.append(1.0)
            continue

        cand_len = len(cand_structure)

        # Check length ratio
        len_diff = abs(ref_len - cand_len)
        max_len = max(ref_len, cand_len)
        if max_len > 0 and len_diff / max_len > config.max_length_ratio_diff:
            similarity = 0.0
        else:
            cand_pairs = _get_base_pairs(cand_structure)

            # Jaccard similarity
            if not ref_pairs and not cand_pairs:
                similarity = 1.0  # Both unstructured
            else:
                intersection = len(ref_pairs & cand_pairs)
                union = len(ref_pairs | cand_pairs)
                similarity = intersection / union if union > 0 else 0.0

        # Store metadata
        seq._metadata.update({
            "rna_basepair_similarity": similarity,
            "num_ref_pairs": len(ref_pairs),
            "num_cand_pairs": len(_get_base_pairs(cand_structure)),
            "structure": cand_structure,
        })

        scores.append(1.0 - similarity)

    return scores
