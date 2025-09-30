"""
Protein structure prediction and analysis constraints.
"""

from __future__ import annotations
from io import StringIO
from typing import Optional, Union, List, Dict, Any
import numpy as np
from ..base import *
from ..tool_cache import ToolCache
from ..schemas import ESMFoldKwargs
from ..tools.structure_prediction.esmfold import predict_structure_esmfold
from ..tools.structure_prediction.boltz import predict_structure_boltz2


def _run_esmfold(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> None:
    """
    Execute ESMFold protein structure prediction on a sequence.

    Args:
        input_sequence: The protein sequence to fold.
        n_replications: Number of sequence replications for multimeric prediction (default: 1).
        esmfold_kwargs: ESMFold configuration arguments (optional, uses defaults if None).

    Raises:
        ValueError: If input_sequence is not SequenceType.PROTEIN.

    Note:
        Results are cached globally to avoid redundant predictions.
        Updates metadata with 'avg_plddt', 'ptm', 'pdb_output', and 'esmfolded_sequence'.
    """

    if input_sequence.sequence_type != SequenceType.PROTEIN:
        raise ValueError("Can only run ESMFold on a protein sequence.")

    if esmfold_kwargs is None:
        esmfold_kwargs = ESMFoldKwargs()

    esmfold_kwargs_dict = esmfold_kwargs.model_dump()

    # Check if prediction already cached
    cached_results = ToolCache.get_cached_results(
        input_sequence, "esmfold", n_replications=n_replications, **esmfold_kwargs_dict
    )
    if cached_results:
        input_sequence._metadata.update(cached_results)
        return

    # Run expensive computation
    esmfolded_sequence = ":".join([input_sequence.sequence] * n_replications)
    folding_output = predict_structure_esmfold(
        sequences=esmfolded_sequence, **esmfold_kwargs_dict
    )

    results = {
        **folding_output.metrics,
        "pdb_output": folding_output.structure_pdb_output,
        "esmfolded_sequence": esmfolded_sequence,
    }

    # Cache results and update metadata
    ToolCache.cache_results(
        input_sequence,
        "esmfold",
        results,
        n_replications=n_replications,
        **esmfold_kwargs_dict,
    )
    input_sequence._metadata.update(results)


def esmfold_plddt_constraint(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted LDDT (pLDDT) score.

    Args:
        input_sequence: The protein sequence to evaluate.
        n_replications: Number of sequence replications (default: 1).
        esmfold_kwargs: ESMFold configuration arguments.

    Returns:
        Constraint score where 0.0 indicates perfect structure confidence (pLDDT = 1.0)
        and higher values indicate lower structure confidence.

    Examples:
        Evaluating protein structure confidence:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> # Using defaults:
        >>> score = esmfold_plddt_constraint(seq, 1)
        >>> # With custom args:
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> score = esmfold_plddt_constraint(seq, 1, kwargs)
    """

    _run_esmfold(input_sequence, n_replications, esmfold_kwargs)
    return 1.0 - input_sequence._metadata["avg_plddt"]


def esmfold_ptm_constraint(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> float:
    """
    Evaluate protein structure quality using ESMFold's predicted TM-score (pTM).

    Args:
        input_sequence: The protein sequence to evaluate.
        n_replications: Number of sequence replications (default: 1).
        esmfold_kwargs: ESMFold configuration arguments.

    Returns:
        Constraint score where 0.0 indicates perfect structure quality (pTM = 1.0)
        and higher values indicate lower structure quality.

    Examples:
        Evaluating protein structure quality:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> score = esmfold_ptm_constraint(seq, 1, kwargs)
    """

    _run_esmfold(input_sequence, n_replications, esmfold_kwargs)
    return 1.0 - input_sequence._metadata["ptm"]


def protein_symmetry_ring_constraint(
    input_sequence: Sequence,
    n_replications: int = 1,
    all_to_all_protomer_symmetry: bool = False,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> float:
    """
    Constrain a protein to form a symmetric ring-like multimeric structure.

    Args:
        input_sequence: The protein sequence to evaluate.
        n_replications: Number of protomers in the ring (default: 1).
        all_to_all_protomer_symmetry: Use all pairwise distances vs adjacent (default: False).
        esmfold_kwargs: ESMFold configuration arguments.

    Returns:
        Constraint score based on standard deviation of inter-protomer distances.
        Lower values indicate more symmetric ring-like arrangements.

    Examples:
        Evaluating ring symmetry:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> score = protein_symmetry_ring_constraint(seq, 6, False, kwargs)  # Hexameric ring
    """
    from biotite.structure import get_chains
    from ..utils import (
        adjacent_distances,
        get_backbone_atoms,
        get_centroid,
        pairwise_distances,
        pdb_file_to_atomarray,
    )

    _run_esmfold(input_sequence, n_replications, esmfold_kwargs)

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))

    centroids = []
    for chain_id in get_chains(atom_array):
        chain_backbone = get_backbone_atoms(
            atom_array[atom_array.chain_id == chain_id]
        ).coord
        centroids.append(get_centroid(chain_backbone))

    assert len(centroids) == n_replications
    centroids = np.vstack(centroids)

    distance_func = (
        pairwise_distances if all_to_all_protomer_symmetry else adjacent_distances
    )
    return float(np.std(distance_func(centroids)))


def protein_globularity_constraint(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_kwargs: Optional[ESMFoldKwargs] = None,
) -> float:
    """
    Encourage compact, globular protein structures.

    Args:
        input_sequence: The protein sequence to evaluate.
        n_replications: Number of sequence replications (default: 1).
        esmfold_kwargs: ESMFold configuration arguments.

    Returns:
        Constraint score based on standard deviation of distances from backbone atoms to centroid.
        Lower values indicate more compact, globular structures.

    Examples:
        Evaluating protein globularity:

        >>> seq = Sequence("MVLSPADKTNVK", SequenceType.PROTEIN)
        >>> kwargs = ESMFoldKwargs(verbose=True)
        >>> score = protein_globularity_constraint(seq, 1, kwargs)
    """
    from ..utils import distances_to_centroid, get_backbone_atoms, pdb_file_to_atomarray

    _run_esmfold(input_sequence, n_replications, esmfold_kwargs)

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))
    backbone = get_backbone_atoms(atom_array).coord
    return float(np.std(distances_to_centroid(backbone)))


def boltz_binding_strength_constraint(
    complexes: Union[
        "Sequence",
        List["Sequence"],
        List[List["Sequence"]],
    ],
    config: Optional[Dict[str, Any]] = None,
    return_component: str = "total_penalty",
) -> Union[float, List[float]]:
    """
    Run Boltz2 to predict structure(s)/complex(es) and compute a binding-strength
    penalty in [0,1], where:
        0 = close to ideal (desired binding/structure)
        1 = poor (≥ tolerance away from targets)

    Works for monomers, pairs, or multi-chain complexes. Supports batch evaluation.

    Args:
      complexes:
        - Single Sequence (monomer), or
        - List/Tuple of Sequences (complex), or
        - List of such complexes.
        Examples:
          Sequence("protA")                        # monomer
          [Sequence("protA"), Sequence("protB")]  # protein–protein pair
          [Sequence("prot"), Sequence("rna")]      # protein–RNA complex
          [[Sequence("A"), Sequence("B")],
           [Sequence("C"), Sequence("D")]]         # multiple pairs

      config (dict):
        - desired_higher: dict target values for "higher is better" metrics
            default {
              "iptm": 0.90, "ligand_iptm": 0.80, "protein_iptm": 0.85,
              "complex_iplddt": 0.85, "complex_plddt": 0.80,
              "ptm": 0.70, "confidence_score": 0.85
            }
        - desired_lower: dict target values for "lower is better" metrics
            default { "complex_ipde": 2.0, "complex_pde": 2.0 }  # Å
        - tol_higher: dict tolerances (distance below target = penalty 1.0)
            default {
              "iptm": 0.05, "ligand_iptm": 0.10, "protein_iptm": 0.07,
              "complex_iplddt": 0.10, "complex_plddt": 0.15,
              "ptm": 0.15, "confidence_score": 0.10
            }
        - tol_lower: dict tolerances (distance above target = penalty 1.0)
            default { "complex_ipde": 2.0, "complex_pde": 3.0 }  # Å
        - weights: dict weights for combining penalties
            default depends on complex type:
              - monomer:
                { "ptm": 0.35, "complex_plddt": 0.45, "complex_pde": 0.20 }
              - protein–ligand:
                { "ligand_iptm": 0.50, "complex_iplddt": 0.25,
                  "complex_ipde": 0.15, "complex_plddt": 0.10 }
              - protein–protein / mixed:
                { "iptm": 0.45, "complex_iplddt": 0.30,
                  "complex_ipde": 0.15, "complex_plddt": 0.10 }
        - include_confidence_score: bool (default True, adds weight 0.10)
        - on_error: "penalize" or "raise" (default "penalize").
                    If penalize, returns 1.0 on failure.
        - batch_size: int (fold this many complexes at once)
        - predict_kwargs: dict of pass-through kwargs to predict_structure_boltz2
                          (e.g., msa_server_url, recycling_steps, diffusion_samples, etc.)

      return_component (str):
        - "total_penalty" (default): weighted combination in [0,1]
        - or any individual metric penalty name among those used:
            e.g., "iptm", "ligand_iptm", "protein_iptm",
                  "complex_iplddt", "complex_plddt",
                  "complex_ipde", "complex_pde",
                  "ptm", "confidence_score"

    Returns:
      float or list[float]: penalty score(s).
    """
    if config is None:
        config = {}

    # Normalize input → list of complexes (each complex = list of Sequences)
    def _normalize(x):
        # Single Sequence
        if isinstance(x, Sequence):
            return [[x]]

        # A single complex: list of Sequences
        if isinstance(x, list) and all(isinstance(s, Sequence) for s in x):
            return [x]

        # Multiple complexes: list of list-of-Sequences
        if isinstance(x, list) and all(
            isinstance(c, list) and all(isinstance(s, Sequence) for s in c) for c in x
        ):
            return x

        raise ValueError(
            "Unsupported input format. Expected Sequence, list[Sequence], or list[list[Sequence]]."
        )

    complexes = _normalize(complexes)
    is_single = len(complexes) == 1
    # Config
    batch_size = config.get("batch_size", None)
    predict_kwargs = dict(config.get("predict_kwargs", {}))

    # Default targets and tolerances
    desired_higher = {
        "iptm": 0.90,
        "ligand_iptm": 0.80,
        "protein_iptm": 0.85,
        "complex_iplddt": 0.85,
        "complex_plddt": 0.80,
        "ptm": 0.70,
        "confidence_score": 0.85,
    }
    desired_higher.update(config.get("desired_higher", {}))

    desired_lower = {"complex_ipde": 2.0, "complex_pde": 2.0}
    desired_lower.update(config.get("desired_lower", {}))

    tol_higher = {
        "iptm": 0.05,
        "ligand_iptm": 0.10,
        "protein_iptm": 0.07,
        "complex_iplddt": 0.10,
        "complex_plddt": 0.15,
        "ptm": 0.15,
        "confidence_score": 0.10,
    }
    tol_higher.update(config.get("tol_higher", {}))

    tol_lower = {"complex_ipde": 2.0, "complex_pde": 3.0}
    tol_lower.update(config.get("tol_lower", {}))

    def _clamp(x, a=0.0, b=1.0):
        return a if x < a else b if x > b else x

    def _penalty_hi(val, tgt, tol):
        return 1.0 if val is None else _clamp((tgt - val) / max(tol, 1e-9))

    def _penalty_lo(val, tgt, tol):
        return 1.0 if val is None else _clamp((val - tgt) / max(tol, 1e-9))

    def _map_entity_type(seq: Sequence) -> str:
        if seq.sequence_type == SequenceType.DNA:
            return "dna"
        elif seq.sequence_type == SequenceType.RNA:
            return "rna"
        elif seq.sequence_type == SequenceType.PROTEIN:
            return "protein"
        else:
            raise ValueError(f"Unsupported sequence_type: {seq.sequence_type}")

    # Prepare inputs for Boltz2
    inputs = []
    for complex in complexes:
        seqs = [s.sequence for s in complex]
        ets = [_map_entity_type(s) for s in complex]
        inputs.append({"sequences": seqs, "entity_types": ets, "seq_objs": complex})

    penalties = []

    # Batch processing
    def _process_batch(batch):
        try:
            if len(batch) == 1:
                out_list = predict_structure_boltz2(
                    sequences=batch[0]["sequences"],
                    entity_types=batch[0]["entity_types"],
                    **predict_kwargs,
                )
                out_list = [out_list]
            else:
                out_list = predict_structure_boltz2(
                    sequences=[b["sequences"] for b in batch],
                    entity_types=[b["entity_types"] for b in batch],
                    **predict_kwargs,
                )
                if not isinstance(out_list, list):
                    out_list = [out_list]
        except Exception:
            if str(config.get("on_error", "penalize")).lower() == "raise":
                raise
            out_list = [None for _ in batch]
        return out_list

    if batch_size and batch_size < len(inputs):
        outputs = []
        for i in range(0, len(inputs), batch_size):
            outputs.extend(_process_batch(inputs[i : i + batch_size]))
    else:
        outputs = _process_batch(inputs)

    # Scoring each complex
    for inp, out in zip(inputs, outputs):
        seq_objs = inp["seq_objs"]

        if out is None:
            penalty = 1.0
            for s in seq_objs:
                s._metadata.setdefault("boltz_binding", []).append(
                    {
                        "penalty": penalty,
                        "reason": "prediction_failed",
                        "raw_output": None,
                    }
                )
            penalties.append(penalty)
            continue

        m = dict(out.metrics or {})

        # Determine complex type
        n_chains = len(inp["sequences"])
        has_ligand = any(et.lower() == "ligand" for et in inp["entity_types"])
        is_monomer = n_chains == 1

        # Default weights by case
        if is_monomer:
            default = {"ptm": 0.35, "complex_plddt": 0.45, "complex_pde": 0.20}
        elif has_ligand:
            default = {
                "ligand_iptm": 0.50,
                "complex_iplddt": 0.25,
                "complex_ipde": 0.15,
                "complex_plddt": 0.10,
            }
        else:
            default = {
                "iptm": 0.45,
                "complex_iplddt": 0.30,
                "complex_ipde": 0.15,
                "complex_plddt": 0.10,
            }
        weights = dict(default)
        weights.update(config.get("weights", {}))
        if config.get("include_confidence_score", True):
            weights.setdefault("confidence_score", 0.10)

        penalties_dict = {}

        def _get(name):
            v = m.get(name, None)
            return float(v) if isinstance(v, (int, float)) else None

        # Case-specific penalties
        if is_monomer:
            penalties_dict["ptm_penalty"] = _penalty_hi(
                _get("ptm"), desired_higher["ptm"], tol_higher["ptm"]
            )
            penalties_dict["complex_plddt_penalty"] = _penalty_hi(
                _get("complex_plddt"),
                desired_higher["complex_plddt"],
                tol_higher["complex_plddt"],
            )
            if _get("complex_pde") is not None:
                penalties_dict["complex_pde_penalty"] = _penalty_lo(
                    _get("complex_pde"),
                    desired_lower["complex_pde"],
                    tol_lower["complex_pde"],
                )

        elif has_ligand:
            penalties_dict["ligand_iptm_penalty"] = _penalty_hi(
                _get("ligand_iptm"),
                desired_higher["ligand_iptm"],
                tol_higher["ligand_iptm"],
            )
            penalties_dict["complex_iplddt_penalty"] = _penalty_hi(
                _get("complex_iplddt"),
                desired_higher["complex_iplddt"],
                tol_higher["complex_iplddt"],
            )
            if _get("complex_ipde") is not None:
                penalties_dict["complex_ipde_penalty"] = _penalty_lo(
                    _get("complex_ipde"),
                    desired_lower["complex_ipde"],
                    tol_lower["complex_ipde"],
                )
            penalties_dict["complex_plddt_penalty"] = _penalty_hi(
                _get("complex_plddt"),
                desired_higher["complex_plddt"],
                tol_higher["complex_plddt"],
            )

        else:  # protein–protein or mixed
            prot_iptm = _get("protein_iptm")
            iptm = _get("iptm")
            chosen = "protein_iptm" if (prot_iptm and prot_iptm > 0) else "iptm"
            val = prot_iptm if chosen == "protein_iptm" else iptm
            if chosen == "iptm":
                penalties_dict["iptm_penalty"] = _penalty_hi(
                    val, desired_higher["iptm"], tol_higher["iptm"]
                )
            else:
                penalties_dict["protein_iptm_penalty"] = _penalty_hi(
                    val, desired_higher["protein_iptm"], tol_higher["protein_iptm"]
                )
            penalties_dict["complex_iplddt_penalty"] = _penalty_hi(
                _get("complex_iplddt"),
                desired_higher["complex_iplddt"],
                tol_higher["complex_iplddt"],
            )
            if _get("complex_ipde") is not None:
                penalties_dict["complex_ipde_penalty"] = _penalty_lo(
                    _get("complex_ipde"),
                    desired_lower["complex_ipde"],
                    tol_lower["complex_ipde"],
                )
            penalties_dict["complex_plddt_penalty"] = _penalty_hi(
                _get("complex_plddt"),
                desired_higher["complex_plddt"],
                tol_higher["complex_plddt"],
            )

        if "confidence_score" in weights:
            penalties_dict["confidence_score_penalty"] = _penalty_hi(
                _get("confidence_score"),
                desired_higher["confidence_score"],
                tol_higher["confidence_score"],
            )

        # If user requests a specific component
        if return_component != "total_penalty":
            key = return_component.strip()
            if not key.endswith("_penalty"):
                key = f"{key}_penalty"
            if key not in penalties_dict:
                raise ValueError(
                    f"Requested component '{return_component}' not available."
                )
            penalty = _clamp(float(penalties_dict[key]))
        else:
            # Weighted sum
            used_weights = {
                k: weights[k.replace("_penalty", "")]
                for k in penalties_dict
                if k.replace("_penalty", "") in weights
            }
            wsum = sum(used_weights.values()) or 1.0
            penalty = _clamp(
                sum((w / wsum) * penalties_dict[k] for k, w in used_weights.items())
            )

        # Store metadata for all Sequences in complex
        for s in seq_objs:
            s._metadata.setdefault("boltz_binding", []).append(
                {
                    "penalty": penalty,
                    "metrics": m,
                    "penalties": penalties_dict,
                    "raw_output": getattr(out, "__dict__", out),
                }
            )

        penalties.append(penalty)

    return penalties[0] if is_single else penalties
