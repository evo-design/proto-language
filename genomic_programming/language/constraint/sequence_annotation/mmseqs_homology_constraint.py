"""
ORF prediction + MMseqs gene homology constraint for evaluating homology (percent identity).
Supports DNA (with ORF prediction) and Protein sequences (direct search).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from proto_language.language.core import Sequence, SequenceType, DNA_NUCLEOTIDES
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.tools.orf_prediction.orfipy import OrfipyConfig, OrfipyInput, run_orfipy_prediction
from proto_language.tools.orf_prediction.prodigal import ProdigalConfig, ProdigalInput, run_prodigal_prediction
from proto_language.tools.gene_annotation.mmseqs import (
    MmseqsSearchProteinsConfig,
    MmseqsSearchProteinsInput,
    mmseqs_search_proteins,
)
from proto_language.utils import MIN_ENERGY, MAX_ENERGY, calculate_percentage_range_deviation, resolve_paths


class MMseqsHomologyConfig(BaseConfig):
    """Configuration for ORF + MMseqs gene homology constraint."""
    
    min_homology: float = Field(
        ge=0.0,
        le=100.0,
        description="Minimum acceptable percent identity (0-100). Lower values are more permissive."
    )
    max_homology: float = Field(
        ge=0.0,
        le=100.0,
        description="Maximum acceptable percent identity (0-100). Higher values allow more similar hits."
    )
    mmseqs_db: str = Field(
        description="Path to MMseqs2 protein database for homology search"
    )
    orf_predictor: Literal["orfipy", "prodigal"] = Field(
        default="prodigal",
        description="ORF prediction tool (DNA only): 'orfipy' (general) or 'prodigal' (prokaryotic, faster). Ignored for protein sequences."
    )
    orfipy_config: Optional[OrfipyConfig] = Field(
        default=None,
        description="ORFipy configuration (DNA only, used if orf_predictor='orfipy')"
    )
    prodigal_config: Optional[ProdigalConfig] = Field(
        default=None,
        description="Prodigal configuration (DNA only, used if orf_predictor='prodigal')"
    )
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = Field(
        default=None,
        description="MMseqs configuration (threads, sensitivity, etc.)"
    )


@ConstraintRegistry.register(
    key="mmseqs-gene-homology",
    label="Gene/Protein Homology",
    config=MMseqsHomologyConfig,
    description="Evaluate homology (percent identity) using MMseqs. For DNA: predicts ORFs first. For proteins: searches directly.",
    vectorized=True,
    concatenate=True
)
def mmseqs_homology_constraint(
    sequences: List[Sequence],
    config: MMseqsHomologyConfig
) -> List[float]:
    """
    Evaluate homology (percent identity) of sequences or their predicted ORFs.
    
    For DNA: Runs ORF prediction, then MMseqs. For Protein: MMseqs directly.
    Uses batched processing - single MMseqs call for all sequences.

    Args:
        sequences: List of DNA or protein sequences.
        config: Configuration with min/max homology, predictor choice, configs.

    Returns:
        List of scores (0.0 = all within range, higher = violations).
    """
    if not sequences:
        return []
    
    sequence_type = sequences[0].sequence_type
    
    # Validate all same type
    if not all(seq.sequence_type == sequence_type for seq in sequences):
        raise ValueError("All sequences must be same type (all DNA or all PROTEIN)")
    
    scores = []
    all_proteins_data = [] 
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Get proteins (ORF prediction for DNA, direct for protein)
        if sequence_type == SequenceType.DNA:
            if config.orf_predictor == "prodigal":
                prodigal_config = config.prodigal_config or ProdigalConfig()
                sequences_clean = [
                    "".join(c for c in seq.sequence.upper() if c in DNA_NUCLEOTIDES)
                    for seq in sequences
                ]
                
                prodigal_input = ProdigalInput(input_sequences=sequences_clean)
                result = run_prodigal_prediction(inputs=prodigal_input, config=prodigal_config)
                
                for seq_idx, orfs_df in enumerate(result.results_per_sequence):
                    sequences[seq_idx]._metadata["prodigal_orfs"] = (
                        orfs_df.to_dict("records") if not orfs_df.empty else []
                    )
                    
                    if not orfs_df.empty:
                        for _, row in orfs_df.iterrows():
                            all_proteins_data.append((
                                seq_idx, row['id'], row['protein_sequence']
                            ))
            
            else:  # orfipy
                orfipy_config = config.orfipy_config or OrfipyConfig(output_dir="")
                input_fasta = temp_path / "all_dna.fasta"
                
                with open(input_fasta, "w") as f:
                    for seq_idx, seq in enumerate(sequences):
                        clean = "".join(c for c in seq.sequence.upper() if c in DNA_NUCLEOTIDES)
                        f.write(f">seq_{seq_idx}\n{clean}\n")
                
                orfipy_input = OrfipyInput(input_fasta=str(input_fasta))
                orfipy_run_config = orfipy_config.model_copy(
                    update={"output_dir": str(temp_path / "orfipy_out")}
                )
                result = run_orfipy_prediction(inputs=orfipy_input, config=orfipy_run_config)
                full_orfs = result.results_df if result.results_df is not None else pd.DataFrame()
                
                for seq_idx in range(len(sequences)):
                    if not full_orfs.empty:
                        seq_orfs = full_orfs[full_orfs['parent_id'] == f'seq_{seq_idx}'].copy()
                    else:
                        seq_orfs = pd.DataFrame()
                    
                    sequences[seq_idx]._metadata["orfipy_orfs"] = (
                        seq_orfs.to_dict("records") if not seq_orfs.empty else []
                    )
                    
                    if not seq_orfs.empty:
                        for _, row in seq_orfs.iterrows():
                            all_proteins_data.append((
                                seq_idx, row['orf_id'], row['amino_acid_sequence']
                            ))
        
        else:  # PROTEIN sequences - use directly
            for seq_idx, seq in enumerate(sequences):
                protein_id = f"protein_{seq_idx}"
                all_proteins_data.append((seq_idx, protein_id, seq.sequence))
                sequences[seq_idx]._metadata["direct_protein"] = {
                    "id": protein_id,
                    "sequence": seq.sequence,
                    "length": len(seq.sequence)
                }
        
        # Write all proteins to combined FASTA
        if not all_proteins_data:
            # No proteins found
            for seq in sequences:
                seq._metadata.update({
                    "mmseqs_results": [],
                    "unique_orfs_with_hits": 0,
                    "orfs_with_acceptable_homology": 0,
                    "total_orfs_with_hits": 0,
                    "homology_compliance_rate": 0.0,
                })
            return [MAX_ENERGY] * len(sequences)
        
        combined_fasta = temp_path / "all_proteins.faa"
        protein_to_seq = {}
        
        with open(combined_fasta, "w") as f:
            for seq_idx, prot_id, prot_seq in all_proteins_data:
                f.write(f">{prot_id}\n{prot_seq}\n")
                protein_to_seq[prot_id] = seq_idx
        
        # MMseqs call
        resolved_db = resolve_paths(config.mmseqs_db)
        mmseqs_config = config.mmseqs_config or MmseqsSearchProteinsConfig(results_dir="")
        
        mmseqs_input = MmseqsSearchProteinsInput(
            query_fasta=str(combined_fasta),
            mmseqs_db=resolved_db
        )
        mmseqs_run_config = mmseqs_config.model_copy(
            update={"results_dir": str(temp_path / "mmseqs_out")}
        )
        mmseqs_result = mmseqs_search_proteins(mmseqs_input, mmseqs_run_config)
        
        if not mmseqs_result.success:
            for seq in sequences:
                seq._metadata.update({
                    "mmseqs_error": True,
                    "mmseqs_error_messages": mmseqs_result.errors,
                    "mmseqs_results": [],
                    "unique_orfs_with_hits": 0,
                    "orfs_with_acceptable_homology": 0,
                    "total_orfs_with_hits": 0,
                    "homology_compliance_rate": 0.0,
                })
            return [MAX_ENERGY] * len(sequences)
        
        all_results = mmseqs_result.results_df if mmseqs_result.results_df is not None else pd.DataFrame()
    
    # Split results back to sequences and score
    for seq_idx, seq in enumerate(sequences):
        seq_prot_ids = [pid for pid, sid in protein_to_seq.items() if sid == seq_idx]
        
        if not all_results.empty and seq_prot_ids:
            seq_results = all_results[all_results['query'].isin(seq_prot_ids)].copy()
        else:
            seq_results = pd.DataFrame()
        
        num_hits = len(seq_results) if not seq_results.empty else 0
        
        seq._metadata.update({
            "mmseqs_results": seq_results.to_dict("records") if not seq_results.empty else [],
            "unique_orfs_with_hits": num_hits,
        })
        
        if seq_results.empty or "pident" not in seq_results.columns:
            seq._metadata.update({
                "orfs_with_acceptable_homology": 0,
                "total_orfs_with_hits": 0,
                "homology_compliance_rate": 0.0,
            })
            scores.append(MAX_ENERGY)
            continue
        
        # Score each hit
        acceptable = 0
        violations = []
        
        for _, row in seq_results.iterrows():
            pident = row["pident"]
            if config.min_homology <= pident <= config.max_homology:
                acceptable += 1
            else:
                violations.append(calculate_percentage_range_deviation(
                    pident, config.min_homology, config.max_homology
                ))
        
        seq._metadata.update({
            "orfs_with_acceptable_homology": acceptable,
            "total_orfs_with_hits": num_hits,
            "homology_compliance_rate": acceptable / num_hits,
        })
        
        scores.append(MIN_ENERGY if not violations else min(MAX_ENERGY, np.mean(violations)))
    
    return scores