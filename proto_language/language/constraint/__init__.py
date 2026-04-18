"""Constraint registry and all registered constraint functions."""

# Base infrastructure
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
    ConstraintSpec,
    InputSlot,
    constraint,
)

# Differentiable constraints
from proto_language.language.constraint.differentiable import (
    ablang_scfv_gradient_backward,
    ablang_vhh_gradient_backward,
    af2_binder_backward,
    af2_binder_forward,
)

# Protein quality constraints
from proto_language.language.constraint.protein_quality import (
    balanced_aa_constraint,
    overall_protein_quality_constraint,
    protein_complexity_constraint,
    protein_diversity_constraint,
    protein_domain_constraint,
    protein_length_constraint,
    protein_repetitiveness_constraint,
)

# Protein structure constraints
from proto_language.language.constraint.protein_structure import (
    boltz_binding_strength_constraint,
    gyration_radius_constraint,
    protein_globularity_constraint,
    protein_symmetry_ring_constraint,
    structure_ensemble_rmsd_constraint,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
    structure_rmsd_constraint,
    structure_tmscore_constraint,
)

# RNA secondary structure constraints
from proto_language.language.constraint.rna_secondary_structure import (
    rna_basepair_similarity_constraint,
    rna_feature_similarity_constraint,
    rna_motif_similarity_constraint,
    rna_property_similarity_constraint,
)

# RNA splicing constraints
from proto_language.language.constraint.rna_splicing import (
    alphagenome_splice_site_usage,
    splice_transformer_intron_boundary,
    splice_transformer_specificity,
)

# Sequence alignment constraints
from proto_language.language.constraint.sequence_alignment import gap_gini_constraint

# Sequence annotation constraints
from proto_language.language.constraint.sequence_annotation import (
    alphagenome_interval_track_constraint,
    mmseqs_similarity_constraint,
    promoter_strength_constraint,
    seq_motif_constraint,
    sigma70_promoter_constraint,
)

# Sequence composition constraints
from proto_language.language.constraint.sequence_composition import (
    gc_content_constraint,
    kmer_frequency_constraint,
    max_homopolymer_constraint,
    sequence_length_constraint,
    specific_kmer_constraint,
)

__all__ = [
    # Base infrastructure
    "ConstraintRegistry",
    "ConstraintSpec",
    "InputSlot",
    "constraint",
    # Differentiable
    "ablang_vhh_gradient_backward",
    "ablang_scfv_gradient_backward",
    "af2_binder_backward",
    "af2_binder_forward",
    # Sequence composition
    "sequence_length_constraint",
    "gc_content_constraint",
    "max_homopolymer_constraint",
    "kmer_frequency_constraint",
    "specific_kmer_constraint",
    # Protein structure
    "gyration_radius_constraint",
    "structure_rmsd_constraint",
    "structure_tmscore_constraint",
    "structure_ensemble_rmsd_constraint",
    "structure_plddt_constraint",
    "structure_ptm_constraint",
    "structure_iptm_constraint",
    "structure_pae_constraint",
    "protein_symmetry_ring_constraint",
    "protein_globularity_constraint",
    "boltz_binding_strength_constraint",
    # Protein quality
    "protein_length_constraint",
    "protein_complexity_constraint",
    "protein_repetitiveness_constraint",
    "protein_diversity_constraint",
    "balanced_aa_constraint",
    "overall_protein_quality_constraint",
    "protein_domain_constraint",
    # Sequence alignment
    "gap_gini_constraint",
    # Sequence annotation
    "mmseqs_similarity_constraint",
    "sigma70_promoter_constraint",
    "seq_motif_constraint",
    "promoter_strength_constraint",
    "alphagenome_interval_track_constraint",
    # RNA secondary structure
    "rna_property_similarity_constraint",
    "rna_motif_similarity_constraint",
    "rna_feature_similarity_constraint",
    "rna_basepair_similarity_constraint",
    # RNA splicing
    "alphagenome_splice_site_usage",
    "splice_transformer_intron_boundary",
    "splice_transformer_specificity",
]
