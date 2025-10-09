"""
Boltz binding strength constraint for protein-protein and protein-ligand interactions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from ...base import Sequence, SequenceType
from ...base.config import BaseConfig
from ..registry import ConstraintRegistry
from ....tools.models.structure_prediction.boltz import run_boltz, BoltzConfig


class BoltzBindingStrengthConfig(BaseConfig):
    """Configuration for Boltz binding strength constraint."""
    boltz_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Boltz2 configuration dictionary containing desired_higher, desired_lower, tol_higher, tol_lower, weights, include_confidence_score, on_error, batch_size, and predict_kwargs. See function docstring for detailed structure."
    )
    return_component: str = Field(
        default="total_penalty",
        description="Component to return: 'total_penalty' (weighted combination) or specific metric name like 'iptm', 'ligand_iptm', 'complex_iplddt', etc."
    )


@ConstraintRegistry.register(
    key="boltz-binding-strength",
    config=BoltzBindingStrengthConfig,
    description="Evaluate protein-protein/protein-ligand binding using Boltz2 structure prediction",
    vectorized=False,
    concatenate=False,  # Boltz handles multi-chain complexes
    gpu_required=True
)
def boltz_binding_strength_constraint(
    complexes: Union[Sequence, List[Sequence], List[List[Sequence]]],
    config: BoltzBindingStrengthConfig
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

      config (BoltzBindingStrengthConfig):
        Configuration containing boltz_config (dict) and return_component (str).
        
      boltz_config (dict):
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
        - predict_kwargs: dict of pass-through kwargs to BoltzConfig
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
    boltz_cfg = config.boltz_config or {}
    return_component = config.return_component
    batch_size = boltz_cfg.get("batch_size", None)
    predict_kwargs = dict(boltz_cfg.get("predict_kwargs", {}))

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
    desired_higher.update(boltz_cfg.get("desired_higher", {}))

    desired_lower = {"complex_ipde": 2.0, "complex_pde": 2.0}
    desired_lower.update(boltz_cfg.get("desired_lower", {}))

    tol_higher = {
        "iptm": 0.05,
        "ligand_iptm": 0.10,
        "protein_iptm": 0.07,
        "complex_iplddt": 0.10,
        "complex_plddt": 0.15,
        "ptm": 0.15,
        "confidence_score": 0.10,
    }
    tol_higher.update(boltz_cfg.get("tol_higher", {}))

    tol_lower = {"complex_ipde": 2.0, "complex_pde": 3.0}
    tol_lower.update(boltz_cfg.get("tol_lower", {}))

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
                config = BoltzConfig(
                    sequences=batch[0]["sequences"],
                    entity_types=batch[0]["entity_types"],
                    **predict_kwargs,
                )
                out_list = [run_boltz(config)]
            else:
                # Note: Boltz doesn't support batch processing in the new API yet
                # Process sequentially
                out_list = []
                for b in batch:
                    config = BoltzConfig(
                        sequences=b["sequences"],
                        entity_types=b["entity_types"],
                        **predict_kwargs,
                    )
                    out_list.append(run_boltz(config))
        except Exception:
            if str(boltz_cfg.get("on_error", "penalize")).lower() == "raise":
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
        weights.update(boltz_cfg.get("weights", {}))
        if boltz_cfg.get("include_confidence_score", True):
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
