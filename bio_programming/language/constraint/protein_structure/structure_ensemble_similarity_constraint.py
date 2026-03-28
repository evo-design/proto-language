"""
proto_language/language/constraint/protein_structure/structure_ensemble_similarity_constraint.py

Contains implementation of structure ensemble similarity constraints
for conformational ensemble sampling and PyMOL-based RMSD alignment.

This constraint generates a conformational ensemble for a protein sequence and
computes the similarity between ensemble members and an experimental target
structure using PyMOL's align command.
"""

from __future__ import annotations

import os
import tempfile
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from proto_tools import (
    BioEmuConfig,
    BioEmuInput,
    Structure,
    StructurePredictionComplex,
    run_bioemu,
)
from pydantic import field_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import Sequence
from proto_language.utils import MAX_ENERGY, sigmoid_score

logger = getLogger(__name__)


# ============================================================================
# PyMOL RMSD Computation
# ============================================================================

def _compute_pymol_aligned_rmsd(
    target_pdb_text: str,
    mobile_pdb_text: str,
    target_selection: str = "name CA",
    mobile_selection: str = "name CA",
) -> Dict[str, Any]:
    """
    Compute aligned RMSD using PyMOL's align command.

    PyMOL's align performs sequence alignment followed by structural superposition,
    making it robust to differences in residue numbering or sequence length.

    Args:
        target_pdb_text (str): PDB content of the target (reference) structure.
        mobile_pdb_text (str): PDB content of the mobile structure to align.
        target_selection (str): PyMOL selection string for target atoms (default: "name CA").
        mobile_selection (str): PyMOL selection string for mobile atoms (default: "name CA").

    Returns:
        dict[str, Any]: Dictionary containing:
            - rmsd: The RMSD after alignment (Angstroms).
            - aligned_atoms: Number of atoms used in the alignment.
            - alignment_cycles: Number of refinement cycles performed.
    """
    try:
        import pymol
        from pymol import cmd
    except ImportError as e:
        raise ImportError(
            "PyMOL is required for ensemble RMSD constraints but was not found. "
            "Please install the open-source version via Conda:\n\n"
            "  conda install -c conda-forge pymol-open-source\n"
        ) from e

    # Initialize PyMOL in quiet mode without GUI.
    pymol.finish_launching(['pymol', '-qc'])
    cmd.reinitialize()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1:
        f1.write(target_pdb_text)
        tmp_target = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
        f2.write(mobile_pdb_text)
        tmp_mobile = f2.name

    try:
        cmd.load(tmp_target, "target")
        cmd.load(tmp_mobile, "mobile")

        # align performs sequence alignment + structural superposition.
        # Returns: (RMSD, n_atoms_aligned, n_cycles, RMSD_pre, n_atoms_pre, score, n_res)
        result = cmd.align(
            f"mobile and {mobile_selection}",
            f"target and {target_selection}",
        )

        return {
            'rmsd': result[0],
            'aligned_atoms': result[1],
            'alignment_cycles': result[2],
        }
    except Exception as e:
        logger.warning(f"PyMOL alignment failed: {e}, returning very bad RMSD value")
        return {'rmsd': 999.0, 'aligned_atoms': 0, 'alignment_cycles': 0}
    finally:
        if os.path.exists(tmp_target):
            os.unlink(tmp_target)
        if os.path.exists(tmp_mobile):
            os.unlink(tmp_mobile)
        cmd.delete("all")
        cmd.reinitialize()


def _compute_ensemble_rmsds(
    target_pdb_text: str,
    ensemble_pdb_frames: List[str],
    target_selection: str = "name CA",
    mobile_selection: str = "name CA",
    verbose: bool = False,
) -> List[float]:
    """
    Compute RMSD between a target structure and all frames in an ensemble.

    Args:
        target_pdb_text (str): PDB content of the target (reference) structure.
        ensemble_pdb_frames (list[str]): List of PDB content strings, one per ensemble frame.
        target_selection (str): PyMOL selection for target atoms.
        mobile_selection (str): PyMOL selection for ensemble atoms.
        verbose (bool): Whether to log progress.

    Returns:
        list[float]: List of RMSD values (one per frame).
    """
    rmsds = []
    n_frames = len(ensemble_pdb_frames)

    for i, frame_pdb in enumerate(ensemble_pdb_frames):
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Computing RMSD for frame {i + 1}/{n_frames}")

        result = _compute_pymol_aligned_rmsd(
            target_pdb_text=target_pdb_text,
            mobile_pdb_text=frame_pdb,
            target_selection=target_selection,
            mobile_selection=mobile_selection,
        )
        rmsds.append(result['rmsd'])

    return rmsds


def _summarize_rmsds(
    rmsds: List[float],
    aggregation: Literal["min", "p10", "mean", "median"] = "min",
) -> Optional[float]:
    """
    Summarize a list of RMSD values into a single value. Returns None if the list
    is empty

    Args:
        rmsds (list[float]): List of RMSD values.
        aggregation (Literal['min', 'p10', 'mean', 'median']): How to summarize:
            - "min": Minimum RMSD (best match in ensemble).
            - "p10": 10th percentile RMSD.
            - "mean": Mean RMSD across ensemble.
            - "median": Median RMSD across ensemble.

    Returns:
        float | None: Summarized RMSD value or None if list is empty.
    """
    if not rmsds:
        return None

    arr = np.array(rmsds)

    if aggregation == "min":
        return float(np.min(arr))
    elif aggregation == "p10":
        return float(np.percentile(arr, 10))
    elif aggregation == "mean":
        return float(np.mean(arr))
    elif aggregation == "median":
        return float(np.median(arr))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


# ============================================================================
# Target Structure Preparation
# ============================================================================

def _prepare_target_structure(
    target_structure: Structure | str,
    residue_range: Optional[Tuple[int, int]] = None,
    chain_id: Optional[str] = None,
) -> str:
    """
    Resolve the target structure to a PDB string, optionally extracting a
    specific chain and residue range.

    Args:
        target_structure (Structure | str): A Structure object, file path (.pdb/.cif), or raw PDB/CIF content string.
        residue_range (tuple[int, int] | None): Optional (start, end) 1-indexed residue range to extract.
        chain_id (str | None): Optional chain ID to extract.

    Returns:
        str: PDB content string for the target structure.
    """
    if isinstance(target_structure, Structure):
        pdb_content = target_structure.structure_pdb
    else:
        pdb_content = Structure(target_structure).structure_pdb

    # Extract specific chain if requested.
    if chain_id is not None:
        pdb_content = _extract_chain_from_pdb(pdb_content, chain_id)

    # Extract residue range if requested.
    if residue_range is not None:
        pdb_content = _extract_residue_range_from_pdb(
            pdb_content, residue_range[0], residue_range[1]
        )

    return pdb_content


def _extract_chain_from_pdb(pdb_text: str, chain_id: str) -> str:
    """
    Extract a specific chain from PDB content.

    Args:
        pdb_text (str): Full PDB content.
        chain_id (str): Chain identifier to extract (e.g., 'A').

    Returns:
        str: PDB content with only the specified chain.
    """
    extracted_lines = []
    for line in pdb_text.splitlines():
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if len(line) >= 22 and line[21] == chain_id:
                extracted_lines.append(line)
        elif line.startswith('TER'):
            # Include TER records for the correct chain
            if len(line) >= 22 and line[21] == chain_id:
                extracted_lines.append(line)
        elif line.startswith('END'):
            extracted_lines.append(line)
            break
        elif not line.startswith(('ATOM', 'HETATM', 'TER')):
            # Keep header lines
            extracted_lines.append(line)

    return "\n".join(extracted_lines)


def _extract_residue_range_from_pdb(
    pdb_text: str,
    start_res: int,
    end_res: int,
) -> str:
    """
    Extract a residue range from PDB content.

    Args:
        pdb_text (str): Full PDB content.
        start_res (int): Starting residue number (1-indexed, inclusive).
        end_res (int): Ending residue number (1-indexed, inclusive).

    Returns:
        str: PDB content with only residues in the specified range.
    """
    extracted_lines = []
    for line in pdb_text.splitlines():
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                # Residue number is in columns 23-26 (indices 22:26)
                res_num = int(line[22:26].strip())
                if start_res <= res_num <= end_res:
                    extracted_lines.append(line)
            except ValueError:
                # Keep line if parsing fails
                extracted_lines.append(line)
        elif line.startswith('END'):
            extracted_lines.append(line)
        elif not line.startswith(('ATOM', 'HETATM', 'TER')):
            # Keep header lines
            extracted_lines.append(line)

    return "\n".join(extracted_lines)


# ============================================================================
# Configuration
# ============================================================================

class StructureEnsembleSimilarityConfig(BaseConfig):
    """Configuration for structure ensemble similarity constraints.

    This constraint generates a conformational ensemble for a proposal protein
    sequence and computes the RMSD between ensemble members and an experimental
    target structure using PyMOL's align command.

    Attributes:
        target_structure (Structure | str):
            The target structure. Accepts a Structure object, a file path to a
            PDB/CIF file (identified by .pdb/.cif/.mmcif extension), or raw
            PDB/CIF content as a string.

        target_chain_id (str | None):
            If specified, extract only this chain from the target structure.
            Useful when the target PDB contains multiple chains.

        target_residue_range (tuple[int, int] | None):
            If specified, extract only residues within this range (1-indexed,
            inclusive) from the target structure.

        proposal_residue_range (tuple[int, int] | None):
            If specified, use only this range (1-indexed) of the proposal
            sequence for ensemble sampling.

        bioemu_config (BioEmuConfig):
            Configuration parameters to pass directly to BioEmu. Important parameters
            are:
            - num_samples (int): Number of conformational samples to generate with
              BioEmu. Default: 500.
            - bioemu_model (Literal["bioemu-v1.0", "bioemu-v1.1"]): BioEmu model variant
              to use. Default: "bioemu-v1.1".
            - filter_samples (bool): Whether to filter out low-quality BioEmu samples.
              Default: True.
            - batch_size (int): BioEmu batch size parameter. Default: 10.

        rmsd_aggregation (Literal['min', 'p10', 'mean', 'median']):
            How to summarize RMSD values across the ensemble:
            - "min": Use minimum RMSD (best match). Default.
            - "p10": Use 10th percentile RMSD.
            - "mean": Use mean RMSD.
            - "median": Use median RMSD.

        inflection_point_angstroms (float):
            The RMSD value (in Angstroms) at which the sigmoid scoring function
            returns 0.5. Lower values are stricter. Default: 2.0.

        sigmoid_slope (float):
            Steepness of the sigmoid penalty curve. Higher values create a
            sharper transition. Default: 3.0.

        verbose (bool):
            Whether to print progress messages. Default: False.
    """

    # Target specification (required)
    target_structure: Structure | str = ConfigField(
        title="Target Structure",
        description=(
            "Target structure: a Structure object, file path (.pdb/.cif), "
            "or raw PDB/CIF content string."
        ),
    )

    # Target subsetting
    target_chain_id: Optional[str] = ConfigField(
        title="Target Chain ID",
        default=None,
        description="Chain ID to extract from the target structure (e.g., 'A').",
        advanced=True,
    )
    target_residue_range: Optional[Tuple[int, int]] = ConfigField(
        title="Target Residue Range",
        default=None,
        description="Residue range (start, end) to extract from target (1-indexed, inclusive).",
        advanced=True,
    )

    # Proposal subsetting
    proposal_residue_range: Optional[Tuple[int, int]] = ConfigField(
        title="Proposal Residue Range",
        default=None,
        description="Residue range (start, end) of the proposal sequence to use.",
        advanced=True,
    )

    # BioEmu tool configuration
    bioemu_config: BioEmuConfig = ConfigField(
        title="BioEmu Config",
        default_factory=BioEmuConfig,
        description="Dictionary of configuration parameters passed to the ensemble prediction tool.",
        advanced=True,
    )

    # RMSD configuration
    rmsd_aggregation: Literal["min", "p10", "mean", "median"] = ConfigField(
        title="RMSD Aggregation",
        default="min",
        description="How to summarize RMSD values across the ensemble.",
    )

    # Scoring configuration
    inflection_point_angstroms: float = ConfigField(
        title="RMSD Inflection Point",
        default=3.0,
        description="RMSD (Angstroms) where score is 0.5. < 3.0 is generally good.",
        gt=0.0,
        advanced=True,
    )
    sigmoid_slope: float = ConfigField(
        title="Sigmoid Slope",
        default=3.0,
        description="Steepness of the penalty curve.",
        gt=0.0,
        advanced=True,
    )

    # Runtime configuration
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print progress messages.",
        hidden=True,
    )

    @field_validator("target_residue_range", "proposal_residue_range", mode="after")
    @classmethod
    def validate_residue_range(
        cls, v: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Validate residue ranges."""
        if v is not None:
            start, end = v
            if start < 1:
                raise ValueError("Residue range start must be >= 1 (1-indexed).")
            if end < start:
                raise ValueError("Residue range end must be >= start.")
        return v


# ============================================================================
# Constraint Implementation
# ============================================================================

@constraint(
    key="structure-ensemble-rmsd",
    label="Structure Ensemble RMSD",
    config=StructureEnsembleSimilarityConfig,
    description=(
        "Generate conformational ensemble and compute RMSD against "
        "an experimental target structure using PyMOL alignment."
    ),
    uses_gpu=True,
    tools_called=["bioemu-sample", "pymol"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    num_input_sequences_per_tuple=1,
)
def structure_ensemble_rmsd_constraint(
    input_sequences: List[Tuple[Sequence, ...]],
    config: StructureEnsembleSimilarityConfig,
) -> List[float]:
    """
    Generate conformational ensembles and compute RMSD against an experimental
    target structure.

    This constraint:
    1. Prepares the target structure (extracting chain/residue range if specified).
    2. For each proposal sequence, generate a conformational ensemble.
    3. Computes PyMOL-aligned RMSD between each ensemble frame and the target.
    4. Summarizes the RMSDs using the specified aggregation method.
    5. Converts the summarized RMSD to a 0-1 score using a sigmoid function.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of single-sequence tuples. Each tuple contains one protein
            Sequence object to evaluate.
        config (StructureEnsembleSimilarityConfig): Configuration specifying target structure, ensemble prediction
                parameters, and scoring settings.

    Returns:
        list[float]: List of scores (0-1), where 0 is a perfect match and 1 is poor.
    """
    # Prepare target structure
    target_pdb = _prepare_target_structure(
        target_structure=config.target_structure,
        residue_range=config.target_residue_range,
        chain_id=config.target_chain_id,
    )

    if config.verbose:
        logger.info(f"Target structure prepared ({len(target_pdb)} characters)")

    scores = []

    for seq_idx, (seq,) in enumerate(input_sequences):
        if config.verbose:
            logger.info(f"Processing sequence {seq_idx + 1}/{len(input_sequences)}")

        try:
            # Extract proposal subsequence if range is specified.
            proposal_sequence = seq.sequence
            if config.proposal_residue_range is not None:
                start_res, end_res = config.proposal_residue_range
                # Convert from 1-indexed to 0-indexed for Python slicing.
                proposal_sequence = seq.sequence[start_res - 1 : end_res]
                if config.verbose:
                    logger.info(
                        f"Using residue range {start_res}-{end_res}: "
                        f"{len(proposal_sequence)} residues"
                    )

            # Configure and run ensemble prediction.

            bioemu_input = BioEmuInput(
                complexes=[
                    StructurePredictionComplex(
                        chains=[{"sequence": proposal_sequence, "entity_type": "protein"}]
                    )
                ]
            )

            # Use maximum verbosity.
            if config.verbose:
                config.bioemu_config.verbose = config.verbose

            if config.verbose:
                logger.info(
                    f"Running BioEmu: {config.bioemu_config.num_samples} samples for "
                    f"sequence of length {len(proposal_sequence)}"
                )

            result = run_bioemu(bioemu_input, config.bioemu_config)

            if not result.ensembles or len(result.ensembles[0].structures) == 0:
                logger.warning(f"BioEmu returned no structures for sequence {seq_idx}")
                scores.append(1.0)
                continue

            ensemble = result.ensembles[0]

            if config.verbose:
                logger.info(f"Generated {len(ensemble.structures)} conformations")

            ensemble_pdb_frames = [s.structure_pdb for s in ensemble.structures]

            # Compute RMSDs and distribution statistics.
            rmsds = _compute_ensemble_rmsds(
                target_pdb_text=target_pdb,
                ensemble_pdb_frames=ensemble_pdb_frames,
                verbose=config.verbose,
            )

            rmsd_summary = _summarize_rmsds(rmsds, config.rmsd_aggregation)

            if config.verbose:
                logger.info(
                    f"RMSD summary ({config.rmsd_aggregation}): {rmsd_summary:.2f} Å"
                )
                logger.info(
                    f"RMSD stats: min={np.min(rmsds):.2f}, "
                    f"mean={np.mean(rmsds):.2f}, max={np.max(rmsds):.2f}"
                )

            # Convert to score in [0, 1].
            score = sigmoid_score(
                rmsd_summary,
                config.inflection_point_angstroms,
                config.sigmoid_slope,
            )

            rmsd_arr = np.array(rmsds)
            seq._metadata.update({
                "ensemble_rmsd_summary": rmsd_summary,
                "ensemble_rmsd_aggregation": config.rmsd_aggregation,
                "ensemble_rmsd_all": rmsds,
                "ensemble_rmsd_min": float(np.min(rmsd_arr)),
                "ensemble_rmsd_mean": float(np.mean(rmsd_arr)),
                "ensemble_rmsd_median": float(np.median(rmsd_arr)),
                "ensemble_rmsd_p10": float(np.percentile(rmsd_arr, 10)),
                "ensemble_rmsd_std": float(np.std(rmsd_arr)),
                "ensemble_size": len(rmsds),
                "ensemble_score": score,
                "pct_within_2A": float(np.mean(rmsd_arr < 2.0) * 100),
                "pct_within_3A": float(np.mean(rmsd_arr < 3.0) * 100),
            })

            scores.append(score)

        except Exception as e:
            logger.error(f"Error processing sequence {seq_idx}: {e}")
            scores.append(MAX_ENERGY)

    return scores
