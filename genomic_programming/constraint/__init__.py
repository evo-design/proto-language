"""
Constraint functions for sequence optimization and validation.

This module provides constraint functions for evaluating and optimizing biological
sequences. Constraints assess sequence properties like length, composition, structure,
and functional characteristics
"""

# Import all constraints
from .sequence_composition import (
    sequence_length_constraint,
    gc_content_constraint,
    max_homopolymer_constraint,
    dinucleotide_frequency_constraint,
    tetranucleotide_usage_constraint,
)

from .protein_structure import (
    esmfold_plddt_constraint,
    esmfold_ptm_constraint,
    protein_symmetry_ring_constraint,
    protein_globularity_constraint,
    boltz_binding_strength_constraint,
)

from .protein_quality import (
    protein_length_constraint,
    protein_complexity_constraint,
    protein_repetitiveness_constraint,
    protein_diversity_constraint,
    balanced_aa_constraint,
    overall_protein_quality_constraint,
    protein_domain_constraint,
)

from .sequence_annotation import (
    orfipy_mmseqs_gene_hit_count_constraint,
    orfipy_mmseqs_gene_homology_constraint,
    sigma70_promoter_constraint,
    seq_motif_constraint,
    promoter_strength_constraint,
)

# Import shared constants
from .utils import (
    DNA_NUCLEOTIDES,
    RNA_NUCLEOTIDES,
    MIN_ENERGY,
    MAX_ENERGY,
    LOG_BASE,
    MIN_GC_CONTENT,
    MAX_GC_CONTENT,
)

__all__ = [
    # Sequence Composition
    "sequence_length_constraint",
    "gc_content_constraint",
    "max_homopolymer_constraint",
    "dinucleotide_frequency_constraint",
    "tetranucleotide_usage_constraint",
    # Protein Structure
    "esmfold_plddt_constraint",
    "esmfold_ptm_constraint",
    "protein_symmetry_ring_constraint",
    "protein_globularity_constraint",
    "boltz_binding_strength_constraint",
    # Protein Quality
    "protein_length_constraint",
    "protein_complexity_constraint",
    "protein_repetitiveness_constraint",
    "protein_diversity_constraint",
    "balanced_aa_constraint",
    "overall_protein_quality_constraint",
    "protein_domain_constraint",
    # Sequence Annotation
    "orfipy_mmseqs_gene_hit_count_constraint",
    "orfipy_mmseqs_gene_homology_constraint",
    "sigma70_promoter_constraint",
    "seq_motif_constraint",
    "promoter_strength_constraint",
    # Constants
    "DNA_NUCLEOTIDES",
    "RNA_NUCLEOTIDES",
    "MIN_ENERGY",
    "MAX_ENERGY",
    "LOG_BASE",
    "MIN_GC_CONTENT",
    "MAX_GC_CONTENT",
]
