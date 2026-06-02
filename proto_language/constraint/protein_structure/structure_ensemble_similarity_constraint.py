"""Contains implementation of structure ensemble similarity constraints.

for conformational ensemble sampling and PyMOL-based RMSD alignment.

This constraint generates a conformational ensemble for a protein sequence and
computes the similarity between ensemble members and an experimental target
structure using PyMOL alignment.
"""

from logging import getLogger
from typing import Literal

import numpy as np
from proto_tools import (
    BioEmuConfig,
    BioEmuInput,
    Complex,
    PyMOLRMSDConfig,
    PyMOLRMSDInput,
    Structure,
    run_bioemu,
    run_pymol_rmsd_alignment,
)
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, sigmoid_score
from proto_language.utils.base import BaseConfig, ConfigField

logger = getLogger(__name__)


# ============================================================================
# PyMOL RMSD Computation
# ============================================================================


def _compute_pymol_aligned_rmsd(
    target_structure: Structure,
    mobile_structure: Structure,
    target_selection: str = "name CA",
    mobile_selection: str = "name CA",
    method: Literal["cealign", "align"] = "align",
) -> dict[str, float | int]:
    """Compute aligned RMSD using the hosted PyMOL alignment tool.

    Args:
        target_structure (Structure): Target/reference structure.
        mobile_structure (Structure): Mobile structure to align.
        target_selection (str): PyMOL selection string for target atoms (default: "name CA").
        mobile_selection (str): PyMOL selection string for mobile atoms (default: "name CA").
        method (Literal['cealign', 'align']): PyMOL alignment routine to use.

    Returns:
        dict[str, float | int]: Dictionary containing RMSD and method-specific
            alignment metrics.
    """
    output = run_pymol_rmsd_alignment(
        PyMOLRMSDInput(
            target_structure=target_structure,
            mobile_structure=mobile_structure,
        ),
        PyMOLRMSDConfig(
            method=method,
            target_selection=f"target and {target_selection}",
            mobile_selection=f"mobile and {mobile_selection}",
        ),
    )
    return dict(output.metrics.items())


def _compute_ensemble_rmsds(
    target_pdb_text: str,
    ensemble_pdb_frames: list[str],
    target_selection: str = "name CA",
    mobile_selection: str = "name CA",
    method: Literal["cealign", "align"] = "align",
    verbose: bool = False,
) -> list[float]:
    """Compute RMSD between a target structure and all frames in an ensemble.

    Args:
        target_pdb_text (str): PDB content of the target (reference) structure.
        ensemble_pdb_frames (list[str]): List of PDB content strings, one per ensemble frame.
        target_selection (str): PyMOL selection for target atoms.
        mobile_selection (str): PyMOL selection for ensemble atoms.
        method (Literal['cealign', 'align']): PyMOL alignment routine to use.
        verbose (bool): Whether to log progress.

    Returns:
        list[float]: List of RMSD values (one per frame).
    """
    rmsds = []
    n_frames = len(ensemble_pdb_frames)
    target_structure = Structure(structure=target_pdb_text)

    for i, frame_pdb in enumerate(ensemble_pdb_frames):
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Computing RMSD for frame {i + 1}/{n_frames}")

        result = _compute_pymol_aligned_rmsd(
            target_structure=target_structure,
            mobile_structure=Structure(structure=frame_pdb),
            target_selection=target_selection,
            mobile_selection=mobile_selection,
            method=method,
        )
        rmsds.append(result["rmsd"])

    return rmsds


def _summarize_rmsds(
    rmsds: list[float],
    aggregation: Literal["min", "p10", "mean", "median"] = "min",
) -> float | None:
    """Summarize a list of RMSD values into a single value. Returns None if the list.

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
    if aggregation == "p10":
        return float(np.percentile(arr, 10))
    if aggregation == "mean":
        return float(np.mean(arr))
    if aggregation == "median":
        return float(np.median(arr))
    raise ValueError(f"Unknown aggregation method: {aggregation}")


# ============================================================================
# Target Structure Preparation
# ============================================================================


def _prepare_target_structure(
    target_structure: Structure | str,
    residue_range: tuple[int, int] | None = None,
    chain_id: str | None = None,
) -> str:
    """Resolve the target structure to a PDB string, optionally extracting a.

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
        pdb_content = Structure(structure=target_structure).structure_pdb

    # Extract specific chain if requested.
    if chain_id is not None:
        pdb_content = _extract_chain_from_pdb(pdb_content, chain_id)

    # Extract residue range if requested.
    if residue_range is not None:
        pdb_content = _extract_residue_range_from_pdb(pdb_content, residue_range[0], residue_range[1])

    return pdb_content  # type: ignore[no-any-return]


def _extract_chain_from_pdb(pdb_text: str, chain_id: str) -> str:
    """Extract a specific chain from PDB content.

    Args:
        pdb_text (str): Full PDB content.
        chain_id (str): Chain identifier to extract (e.g., 'A').

    Returns:
        str: PDB content with only the specified chain.
    """
    extracted_lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if len(line) >= 22 and line[21] == chain_id:
                extracted_lines.append(line)
        elif line.startswith("TER"):
            # Include TER records for the correct chain
            if len(line) >= 22 and line[21] == chain_id:
                extracted_lines.append(line)
        elif line.startswith("END"):
            extracted_lines.append(line)
            break
        elif line.startswith("ANISOU"):
            continue
        elif not line.startswith(("ATOM", "HETATM", "TER")):
            # Keep header lines
            extracted_lines.append(line)

    return "\n".join(extracted_lines)


def _extract_residue_range_from_pdb(
    pdb_text: str,
    start_res: int,
    end_res: int,
) -> str:
    """Extract a residue range from PDB content.

    Args:
        pdb_text (str): Full PDB content.
        start_res (int): Starting residue number (1-indexed, inclusive).
        end_res (int): Ending residue number (1-indexed, inclusive).

    Returns:
        str: PDB content with only residues in the specified range.
    """
    extracted_lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                # Residue number is in columns 23-26 (indices 22:26)
                res_num = int(line[22:26].strip())
                if start_res <= res_num <= end_res:
                    extracted_lines.append(line)
            except ValueError:
                # Keep line if parsing fails
                extracted_lines.append(line)
        elif line.startswith("END"):
            extracted_lines.append(line)
        elif line.startswith("ANISOU"):
            continue
        elif not line.startswith(("ATOM", "HETATM", "TER")):
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
    target structure using PyMOL alignment.

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

        pymol_alignment_method (Literal["cealign", "align"]):
            PyMOL alignment routine to use for ensemble RMSD calculation.
            Default: "align".

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
        description=("Target structure: a Structure object, file path (.pdb/.cif), or raw PDB/CIF content string."),
    )

    # Target subsetting
    target_chain_id: str | None = ConfigField(
        title="Target Chain ID",
        default=None,
        description="Chain ID to extract from the target structure (e.g., 'A').",
    )
    target_residue_range: tuple[int, int] | None = ConfigField(
        title="Target Residue Range",
        default=None,
        description="Residue range (start, end) to extract from target (1-indexed, inclusive).",
    )

    # Proposal subsetting
    proposal_residue_range: tuple[int, int] | None = ConfigField(
        title="Proposal Residue Range",
        default=None,
        description="Residue range (start, end) of the proposal sequence to use.",
    )

    # BioEmu tool configuration
    bioemu_config: BioEmuConfig = ConfigField(
        title="BioEmu Config",
        default_factory=BioEmuConfig,
        description="Dictionary of configuration parameters passed to the ensemble prediction tool.",
    )

    # RMSD configuration
    rmsd_aggregation: Literal["min", "p10", "mean", "median"] = ConfigField(
        title="RMSD Aggregation",
        default="min",
        description="How to summarize ensemble RMSD values: min (best match), p10, mean, or median.",
    )
    pymol_alignment_method: Literal["cealign", "align"] = ConfigField(
        title="PyMOL Alignment Method",
        default="align",
        description="PyMOL alignment routine for ensemble RMSD calculation.",
    )

    # Scoring configuration
    inflection_point_angstroms: float = ConfigField(
        title="RMSD Inflection (Å)",
        default=3.0,
        description="RMSD in Ångströms where the sigmoid score equals 0.5; values below 3 Å are generally a good match.",
        gt=0.0,
    )
    sigmoid_slope: float = ConfigField(
        title="Sigmoid Slope",
        default=3.0,
        description="Steepness of the penalty curve.",
        gt=0.0,
    )

    # Runtime configuration
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print progress messages.",
    )

    @field_validator("target_residue_range", "proposal_residue_range", mode="after")
    @classmethod
    def validate_residue_range(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
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
    description="Generate a conformational ensemble and compute RMSD against a target structure via PyMOL.",
    uses_gpu=True,
    tools_called=["bioemu-sample", "pymol-rmsd-alignment"],
    category="protein_structure",
    supported_sequence_types=["protein"],
)
def structure_ensemble_rmsd_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: StructureEnsembleSimilarityConfig,
) -> list[ConstraintOutput]:
    """Generate conformational ensembles and compute RMSD against an experimental.

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
        list[ConstraintOutput]: Per-proposal score in ``[0, 1]`` (0 is a perfect match)
            with ensemble RMSD summary/distribution metadata.
    """
    # Prepare target structure
    target_pdb = _prepare_target_structure(
        target_structure=config.target_structure,
        residue_range=config.target_residue_range,
        chain_id=config.target_chain_id,
    )

    if config.verbose:
        logger.info(f"Target structure prepared ({len(target_pdb)} characters)")

    results: list[ConstraintOutput] = []

    n = len(input_sequences)
    for seq_idx, (seq,) in enumerate(input_sequences):
        if config.verbose:
            logger.info(f"Processing sequence {seq_idx + 1}/{n}")

        try:
            # Extract proposal subsequence if range is specified.
            proposal_sequence = seq.sequence
            if config.proposal_residue_range is not None:
                start_res, end_res = config.proposal_residue_range
                # Convert from 1-indexed to 0-indexed for Python slicing.
                proposal_sequence = seq.sequence[start_res - 1 : end_res]
                if config.verbose:
                    logger.info(f"Using residue range {start_res}-{end_res}: {len(proposal_sequence)} residues")

            bioemu_input = BioEmuInput(
                complexes=[Complex(chains=[{"sequence": proposal_sequence, "entity_type": "protein"}])]
            )

            if config.verbose:
                config.bioemu_config.verbose = config.verbose

            if config.verbose:
                logger.info(
                    f"Running BioEmu: {config.bioemu_config.num_samples} samples for "
                    f"sequence of length {len(proposal_sequence)}"
                )

            result = run_bioemu(bioemu_input, config.bioemu_config)

            if not result.ensembles or len(result.ensembles[0].structures) == 0:
                logger.warning("structure-ensemble-rmsd: BioEmu returned no structures for sequence %d", seq_idx)
                results.append(
                    ConstraintOutput(
                        score=MAX_ENERGY,
                        metadata={"ensemble_rmsd_error": f"BioEmu returned no structures for sequence {seq_idx}"},
                    )
                )
                continue

            ensemble = result.ensembles[0]

            if config.verbose:
                logger.info(f"Generated {len(ensemble.structures)} conformations")

            ensemble_pdb_frames = [s.structure_pdb for s in ensemble.structures]

            rmsds = _compute_ensemble_rmsds(
                target_pdb_text=target_pdb,
                ensemble_pdb_frames=ensemble_pdb_frames,
                method=config.pymol_alignment_method,
                verbose=config.verbose,
            )

            rmsd_summary = _summarize_rmsds(rmsds, config.rmsd_aggregation)

            if config.verbose:
                logger.info(f"RMSD summary ({config.rmsd_aggregation}): {rmsd_summary:.2f} Å")
                logger.info(f"RMSD stats: min={np.min(rmsds):.2f}, mean={np.mean(rmsds):.2f}, max={np.max(rmsds):.2f}")

            assert rmsd_summary is not None  # noqa: S101 -- mypy type narrowing
            score = sigmoid_score(
                rmsd_summary,
                config.inflection_point_angstroms,
                config.sigmoid_slope,
            )

            rmsd_arr = np.array(rmsds)
            results.append(
                ConstraintOutput(
                    score=score,
                    metadata={
                        "ensemble_rmsd_summary": rmsd_summary,
                        "ensemble_rmsd_aggregation": config.rmsd_aggregation,
                        "ensemble_rmsd_alignment_method": config.pymol_alignment_method,
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
                    },
                )
            )

        except Exception as e:
            # Per-proposal soft-fail (see CLAUDE.md error policy).
            logger.warning("structure-ensemble-rmsd: sequence %d/%d failed: %s", seq_idx, n, e)
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"ensemble_rmsd_error": str(e)}))

    return results
