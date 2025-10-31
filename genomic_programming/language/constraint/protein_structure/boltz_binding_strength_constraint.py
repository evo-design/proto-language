"""
Boltz binding strength constraint for protein-protein and protein-ligand interactions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import Field

from ...core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.models.structure_prediction.boltz import (
    run_boltz,
    BoltzInput,
    BoltzConfig,
)


# Default target values and tolerances for binding strength metrics
DEFAULT_DESIRED_HIGHER = {
    "iptm": 0.90,
    "ligand_iptm": 0.80,
    "protein_iptm": 0.85,
    "complex_iplddt": 0.85,
    "complex_plddt": 0.80,
    "ptm": 0.70,
    "confidence_score": 0.85,
}

DEFAULT_DESIRED_LOWER = {
    "complex_ipde": 2.0,  # Angstroms
    "complex_pde": 2.0,   # Angstroms
}

DEFAULT_TOL_HIGHER = {
    "iptm": 0.05,
    "ligand_iptm": 0.10,
    "protein_iptm": 0.07,
    "complex_iplddt": 0.10,
    "complex_plddt": 0.15,
    "ptm": 0.15,
    "confidence_score": 0.10,
}

DEFAULT_TOL_LOWER = {
    "complex_ipde": 2.0,  # Angstroms
    "complex_pde": 3.0,   # Angstroms
}


class BoltzBindingStrengthConfig(BaseConfig):
    """Configuration for Boltz binding strength constraint."""

    # Constraint-specific parameters
    desired_higher: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_DESIRED_HIGHER.copy(),
        description="Target values for 'higher is better' metrics. Provide partial dict to override specific metrics while keeping defaults for others."
    )
    desired_lower: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_DESIRED_LOWER.copy(),
        description="Target values for 'lower is better' metrics (in Angstroms). Provide partial dict to override specific metrics."
    )
    tol_higher: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_TOL_HIGHER.copy(),
        description="Tolerances for 'higher is better' metrics (distance below target = penalty 1.0). Provide partial dict to override."
    )
    tol_lower: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_TOL_LOWER.copy(),
        description="Tolerances for 'lower is better' metrics (distance above target = penalty 1.0, in Angstroms). Provide partial dict to override."
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for combining penalties. If None, defaults based on complex type (monomer/ligand/protein-protein)"
    )
    include_confidence_score: bool = Field(
        default=True,
        description="Whether to include confidence_score in penalty calculation (adds weight 0.10)"
    )
    on_error: str = Field(
        default="penalize",
        description="How to handle prediction errors: 'penalize' (return 1.0) or 'raise' (raise exception)"
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Number of complexes to fold at once (None = process all together)"
    )
    return_component: str = Field(
        default="total_penalty",
        description="Component to return: 'total_penalty' (weighted combination) or specific metric name like 'iptm', 'ligand_iptm', 'complex_iplddt', etc."
    )

    # Nested Boltz2 configuration
    boltz_config: Optional[BoltzConfig] = Field(
        default=None,
        description="Optional Boltz2 configuration (use_msa_server, msa_server_url, recycling_steps, sampling_steps, diffusion_samples, num_workers, devices, verbose). If None, uses defaults.",
    )

# TODO: Convert complexes to Sequences
@ConstraintRegistry.register(
    key="boltz-binding-strength",
    label="Boltz Binding Strength",
    config=BoltzBindingStrengthConfig,
    description="Evaluate protein-protein/protein-ligand binding using Boltz2 structure prediction",
    vectorized=False,
    concatenate=False,  # Boltz handles multi-chain complexes
    gpu_required=True
)
def boltz_binding_strength_constraint(complexes: Union[Sequence, List[Sequence], List[List[Sequence]]], config: BoltzBindingStrengthConfig) -> Union[float, List[float]]:
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

      config (BoltzBindingStrengthConfig):
        Configuration containing penalty calculation parameters and Boltz2 prediction parameters.
        See BoltzBindingStrengthConfig for full parameter descriptions

    Returns:
      float or list[float]: penalty score(s).
    """
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

    # Merge user overrides with defaults (allows partial specification)
    desired_higher = {**DEFAULT_DESIRED_HIGHER, **config.desired_higher}
    desired_lower = {**DEFAULT_DESIRED_LOWER, **config.desired_lower}
    tol_higher = {**DEFAULT_TOL_HIGHER, **config.tol_higher}
    tol_lower = {**DEFAULT_TOL_LOWER, **config.tol_lower}

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
            # Note: Boltz doesn't support batch processing in the new API yet
            # Process sequentially
            out_list = []
            for b in batch:
                boltz_input = BoltzInput(
                    sequences=b["sequences"],
                    entity_types=b["entity_types"],
                )
                boltz_cfg = (
                    config.boltz_config
                    if config.boltz_config is not None
                    else BoltzConfig()
                )
                out_list.append(run_boltz(boltz_input, boltz_cfg))
        except Exception:
            if config.on_error.lower() == "raise":
                raise
            out_list = [None for _ in batch]
        return out_list

    if config.batch_size and config.batch_size < len(inputs):
        outputs = []
        for i in range(0, len(inputs), config.batch_size):
            outputs.extend(_process_batch(inputs[i : i + config.batch_size]))
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
        if config.weights is not None:
            weights = dict(config.weights)
        else:
            if is_monomer:
                weights = {"ptm": 0.35, "complex_plddt": 0.45, "complex_pde": 0.20}
            elif has_ligand:
                weights = {
                    "ligand_iptm": 0.50,
                    "complex_iplddt": 0.25,
                    "complex_ipde": 0.15,
                    "complex_plddt": 0.10,
                }
            else:
                weights = {
                    "iptm": 0.45,
                    "complex_iplddt": 0.30,
                    "complex_ipde": 0.15,
                    "complex_plddt": 0.10,
                }
        if config.include_confidence_score:
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
        if config.return_component != "total_penalty":
            key = config.return_component.strip()
            if not key.endswith("_penalty"):
                key = f"{key}_penalty"
            if key not in penalties_dict:
                raise ValueError(
                    f"Requested component '{config.return_component}' not available."
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
