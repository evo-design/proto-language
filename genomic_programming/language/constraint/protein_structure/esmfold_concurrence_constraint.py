"""
esmfold_concurrence_constraint.py

Contains implementation of structure similarity constraints.
- ESMFold RMSD constraint (structural similarity to a target)
"""

from __future__ import annotations

import numpy as np
import os
import tempfile
from typing import Optional, List, Dict, Any
from logging import getLogger

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
)
from proto_language.tools.structure_prediction import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
    StructurePredictionComplex,
)
from proto_language.tools.orf_prediction.prodigal import (
    run_prodigal_prediction,
    ProdigalInput,
    ProdigalConfig,
)
from proto_language.utils import MAX_ENERGY


logger = getLogger(__name__)


def compute_ce_aligned_rmsd(pdb_text1: str, pdb_text2: str) -> Dict[str, Any]:
    """
    Compute CE-aligned RMSD using PyMOL's cealign.

    Text strings are the full PDB file contents.
    """
    try:
        import pymol
        from pymol import cmd
    except ImportError as e:
        raise ImportError(
            "PyMOL is required for RMSD constraints but was not found. "
            "Please install the open-source version via Conda:\n\n"
            "  conda install -c conda-forge pymol-open-source\n\n"
            "Note: Standard 'pip install pymol' often requires a license or fails to build."
        ) from e

    # Initialize PyMOL in quiet mode without GUI.
    pymol.finish_launching(['pymol', '-qc'])
    cmd.reinitialize()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1:
        f1.write(pdb_text1)
        tmp1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
        f2.write(pdb_text2)
        tmp2 = f2.name

    try:
        cmd.load(tmp1, "ref")
        cmd.load(tmp2, "mobile")

        # cealign aligns 'mobile' to 'ref'.
        result = cmd.cealign("ref", "mobile")

        return {
            'rmsd': result['RMSD'],
            'aligned_length': result['alignment_length'],
            'alignment_score': result.get('raw_score', None)
        }
    finally:
        if os.path.exists(tmp1):
            os.unlink(tmp1)
        if os.path.exists(tmp2):
            os.unlink(tmp2)
        cmd.delete("all")


def _sigmoid_score(rmsd: float, inflection: float, slope: float) -> float:
    """
    Squeezes RMSD into a 0-1 score using a sigmoid function.

    Args:
        rmsd: The calculated RMSD value.
        inflection: The RMSD value where the score is 0.5.
        slope: The steepness of the curve.

    Returns:
        float: Score between 0.0 (good/low RMSD) and 1.0 (bad/high RMSD).
    """
    # 1 / (1 + e^(-k(x - x0)))
    # We want low RMSD -> 0 and high RMSD -> 1.
    # The standard sigmoid 1/(1+e^-x) goes 0->1 as x increases.
    # We use slope * (rmsd - inflection).
    return 1.0 / (1.0 + np.exp(-slope * (rmsd - inflection)))


class ESMFoldRMSDConfig(BaseConfig):
    """Configuration for ESMFold structural similarity constraints.

    This class defines configuration parameters for evaluating the structural
    similarity between a generated sequence and a target reference sequence using
    the ESMFold structure prediction model and PyMOL alignment.

    The constraint folds both the candidate sequence and the target sequence (if
    provided as a string) and calculates the Root Mean Square Deviation (RMSD)
    between their structures. The raw RMSD in Angstroms is then normalized to a
    0-1 score using a sigmoid function, where 0 represents a perfect match (low
    RMSD) and 1 represents a poor match (high RMSD). This is particularly useful
    for structure-based design tasks where preserving a specific fold or motif
    is required.

    Attributes:
        target_sequence (str): The amino acid sequence of the reference protein.
            The candidate sequence will be folded and structurally aligned against
            the structure predicted for this sequence.

        min_target_plddt (float): If the target sequence has an ESMFold pLDDT
            below this value, simply refuse to make a comparison and return
            the maximum penalty. Default: 0.6.

        inflection_point_angstroms (float): The RMSD value (in Angstroms) at
            which the penalty score is 0.5. RMSD values below this threshold
            yield low penalty scores (good), while values above yield high
            penalty scores (bad). Generally, an RMSD < 2.0A is considered a good
            structural match. Default: 2.0.

        sigmoid_slope (float): Controls the steepness of the penalty curve
            around the inflection point. A higher slope results in a sharper
            transition between "good" and "bad" scores. Default: 3.0.

        n_replications (int): Number of times to replicate the sequence for
            multimeric structure prediction. Must be a positive integer.
            Will replicate both the scored sequence and `target_sequence`.
            Default: 1.

        esmfold_config (Optional[ESMFoldConfig]): Optional advanced ESMFold
            configuration parameters. If None, uses default settings.
            Default: None.
    """
    # Required parameters

    target_sequence: str = ConfigField(
        title="Target Reference Sequence",
        description="The amino acid sequence to compare against.",
    )

    # Optional parameters

    min_target_plddt: float = ConfigField(
        title="Minimum pLDDT of Target",
        default=0.6,
        description="The minimum ESMFold pLDDT value that the target should have before using it for comparison.",
    )

    inflection_point_angstroms: float = ConfigField(
        title="RMSD Inflection Point",
        default=2.0,
        description="The RMSD value (in Angstroms) where the score will be 0.5. Good score -> 0, bad score -> 1.",
    )

    sigmoid_slope: float = ConfigField(
        title="Sigmoid Slope",
        default=3.0,
        description="Controls the steepness of the penalty curve around the inflection point.",
    )

    n_replications: int = ConfigField(
        title="Number of Replications",
        default=1,
        ge=1,
        description="Number of times to replicate the sequence for structure prediction.",
    )

    esmfold_config: Optional[ESMFoldConfig] = ConfigField(
        title="ESMFold Config",
        default=None,
        description="Optional ESMFold configuration.",
        advanced=True,
    )


@ConstraintRegistry.register(
    key="esmfold-rmsd",
    label="ESMFold RMSD to Target",
    config=ESMFoldRMSDConfig,
    description="Evaluate structural similarity (RMSD) to a target sequence using ESMFold",
    mode="score",
    batched=True,
    concatenate=True,
    gpu_required=True,
    tools_called=["esmfold", "prodigal", "pymol"],
    category="protein_structure",
)
def esmfold_rmsd_constraint(
    sequences: List[Sequence], config: ESMFoldRMSDConfig
) -> List[float]:
    """
    Predicts structure of input sequences and compares RMSD against a target sequence.
    Returns a score between 0 and 1, where 0 is a perfect/low RMSD match.
    """
    
    # 1. Fold the reference target sequence.

    target_complex = StructurePredictionComplex(
        chains=[config.target_sequence] * config.n_replications,
        entity_types=["protein"] * config.n_replications,
    )
    
    # Run ESMFold on the target.
    target_esm_output = run_esmfold(
        inputs=ESMFoldInput(complexes=[target_complex]), 
        config=config.esmfold_config or ESMFoldConfig()
    )
    # If the target is too low confidence, return max penalty (refuse to compare).
    if target_esm_output[0].avg_plddt < config.min_target_plddt:
        return [1.0] * len(sequences)
    target_pdb = target_esm_output[0].structure_pdb

    # 2. Define helper to process a single PDB string against the target.

    def _calculate_score_for_pdb(candidate_pdb: str) -> float:
        try:
            rmsd_data = compute_ce_aligned_rmsd(target_pdb, candidate_pdb)
            rmsd_val = rmsd_data['rmsd']
            return _sigmoid_score(
                rmsd_val, 
                config.inflection_point_angstroms, 
                config.sigmoid_slope
            )
        except Exception as e:
            # If alignment fails (e.g., structures too different), return max penalty.
            logger.warning(f"RMSD alignment failed: {e}")
            return 1.0

    scores = []

    if sequences[0].sequence_type == 'protein':
        # Protein path.
        complexes = [
            StructurePredictionComplex(
                chains=[seq.sequence] * config.n_replications,
                entity_types=["protein"] * config.n_replications,
            )
            for seq in sequences
        ]

        candidates_output = run_esmfold(
            inputs=ESMFoldInput(complexes=complexes),
            config=config.esmfold_config or ESMFoldConfig()
        )

        for seq, structure in zip(sequences, candidates_output):
            score = _calculate_score_for_pdb(structure.structure_pdb)

            seq._metadata.update({
                "rmsd_score": score,
                "pdb_output": structure.structure_pdb,
            })
            scores.append(score)

    else:
        # DNA path (via Prodigal).
        prodigal_result = run_prodigal_prediction(
            ProdigalInput(input_sequences=[seq.sequence for seq in sequences]),
            ProdigalConfig(),
        )

        for i, (dna_seq, proteins_df, num_genes) in enumerate(
            zip(sequences, prodigal_result.results_per_sequence, prodigal_result.total_num_genes_per_sequence)
        ):
            if num_genes == 0 or len(proteins_df) == 0:
                scores.append(MAX_ENERGY)
                continue

            # Fold all ORFs found in this DNA sequence.
            protein_seqs = proteins_df["protein_sequence"].tolist()
            complexes = [
                StructurePredictionComplex(
                    chains=[seq] * config.n_replications,
                    entity_types=["protein"] * config.n_replications,
                )
                for seq in protein_seqs
            ]

            esmfold_output = run_esmfold(
                inputs=ESMFoldInput(complexes=complexes),
                config=config.esmfold_config or ESMFoldConfig(),
            )

            # Find the best (lowest) score among all ORFs in this DNA.
            orf_scores = []
            for structure in esmfold_output:
                orf_scores.append(_calculate_score_for_pdb(structure.structure_pdb))

            best_score = min(orf_scores)

            dna_seq._metadata["esmfold_rmsd_best"] = best_score
            dna_seq._metadata["esmfold_all_rmsds"] = orf_scores

            scores.append(best_score)

    return scores
