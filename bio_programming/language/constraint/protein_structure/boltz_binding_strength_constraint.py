"""
Boltz binding strength constraint for protein-protein and protein-ligand interactions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Literal

from pydantic import field_validator

from proto_language.language.core import Sequence
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
)
from proto_language.tools.structure_prediction.boltz import (
    run_boltz,
    BoltzInput,
    BoltzConfig,
)
from proto_language.tools.structure_prediction.schemas import (
    StructurePredictionComplex,
)
from numpy import clip
from proto_language.tools.ligands import Ligands


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
    "complex_pde": 2.0,  # Angstroms
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
    "complex_pde": 3.0,  # Angstroms
}


class BoltzBindingStrengthConfig(BaseConfig):
    """Configuration for Boltz binding strength constraint.
    
    This class defines configuration parameters for evaluating protein-protein,
    and protein-nucleic acid binding using Boltz, a biomolecular structure prediction
    model. Boltz predicts complex structures and provides confidence metrics for binding
    quality, interface accuracy, and overall structure reliability. The constraint evaluates
    these metrics against target values to assess binding strength and quality.
    
    The constraint uses a penalty-based scoring system where each metric is evaluated
    against its target value and tolerance. Metrics are classified as "higher is better"
    (e.g., interface confidence scores) or "lower is better" (e.g., predicted distance
    errors). Penalties are combined using weighted averages, with default weights
    optimized for different complex types (monomers, protein-nucleic acid, protein-protein).
    
    Attributes:
        ligands (Optional[List[str]]): Ligands to use for binding strength evaluation.
            Can be a SMILES string (automatically converted to list), or a path to SMI or SDF files.
            To delimit multiple ligands, use '.'. Each individual molecular fragment will be treated
            as a separate chain in the input complex.
        desired_higher (Dict[str, float]): Target values for "higher is better" metrics.
            Metrics in this category should ideally be close to 1.0 (high confidence).
            Available metrics:
            - ``iptm``: Interface predicted TM-score (protein-protein interactions, 0-1)
            - ``ligand_iptm``: Ligand interface pTM-score (protein-ligand, 0-1)
            - ``protein_iptm``: Protein-specific interface pTM (multi-chain, 0-1)
            - ``complex_iplddt``: Interface predicted LDDT (0-1)
            - ``complex_plddt``: Overall complex pLDDT (0-1)
            - ``ptm``: Predicted TM-score for overall structure (0-1)
            - ``confidence_score``: Boltz aggregate confidence metric (0-1)
            Provide partial dict to override specific metrics while keeping defaults.
            Default: See DEFAULT_DESIRED_HIGHER.

        desired_lower (Dict[str, float]): Target values for "lower is better" metrics
            (in Ångströms). Metrics should ideally be low (tight interfaces). Available:
            - ``complex_ipde``: Interface predicted distance error (Å)
            - ``complex_pde``: Overall complex predicted distance error (Å)
            Lower values indicate tighter, more accurate predicted interfaces.
            Default: See DEFAULT_DESIRED_LOWER.

        tol_higher (Dict[str, float]): Tolerances for "higher is better" metrics.
            Defines acceptable deviation below target before penalty reaches 1.0.
            For example, if iptm target is 0.90 and tolerance is 0.05, then iptm=0.85
            receives penalty 1.0 (at tolerance limit). Smaller tolerances are stricter.
            Default: See DEFAULT_TOL_HIGHER.

        tol_lower (Dict[str, float]): Tolerances for "lower is better" metrics (in Å).
            Defines acceptable deviation above target before penalty reaches 1.0.
            For example, if complex_ipde target is 2.0 Å and tolerance is 2.0 Å,
            then complex_ipde=4.0 Å receives penalty 1.0. Default: See DEFAULT_TOL_LOWER.

        weights (Optional[Dict[str, float]]): Custom weights for combining metric
            penalties into total score. If None, uses automatic weights based on
            complex type:
            - **Monomer**: ptm=0.35, complex_plddt=0.45, complex_pde=0.20
            - **Protein-ligand**: ligand_iptm=0.50, complex_iplddt=0.25,
              complex_ipde=0.15, complex_plddt=0.10
            - **Protein-protein**: iptm=0.45, complex_iplddt=0.30, complex_ipde=0.15,
              complex_plddt=0.10
            Weights should sum to ~1.0 for interpretability. Default: None (auto).

        include_confidence_score (bool): Whether to include Boltz's aggregate
            confidence_score in penalty calculation. Adds weight 0.10 to the metric
            combination. Recommended for overall quality assessment. Default: True.

        on_error (Literal["penalize", "raise"]): How to handle Boltz prediction
            errors or failures. Options:
            - "penalize": Return penalty 1.0 (maximum) if prediction fails
            - "raise": Raise exception and halt execution
            Use "penalize" for robust pipelines, "raise" for debugging. Default: "penalize".

        batch_size (Optional[int]): Number of complexes to fold simultaneously.
            If None, processes all complexes together (fastest but highest memory).
            Use smaller batches (e.g., 2-5) to control GPU memory usage with large
            complexes or limited GPU memory. Default: None.

        return_component (Literal[...]): Which component to return as the constraint
            score. Options:
            - "total_penalty": Weighted combination of all metrics (default)
            - Specific metric names: "iptm", "ligand_iptm", "complex_iplddt", etc.
            Use specific metrics to focus on particular aspects like interface
            quality (iptm) or distance accuracy (complex_ipde). Default: "total_penalty".

        boltz_config (Optional[BoltzConfig]): Optional advanced Boltz configuration
            including MSA usage, recycling steps, sampling parameters, device
            settings, and verbosity. If None, uses default Boltz settings. The
            ``complexes`` field is set programmatically from input sequences.
            Default: None.
    
    Note:
        **Metric interpretation:**
        - **iptm/ligand_iptm/protein_iptm**: Interface confidence (0-1). Higher = better
          binding prediction. Values >0.8 indicate confident binding interfaces.
        - **complex_iplddt**: Interface per-residue confidence (0-1). Higher = more
          reliable interface residue predictions.
        - **complex_plddt**: Overall structure confidence (0-1). Similar to ESMFold pLDDT.
        - **ptm**: Overall structural accuracy (0-1). Similar to ESMFold pTM.
        - **complex_ipde/complex_pde**: Predicted distance errors in Ångströms. Lower =
          more accurate structure. Values <3 Å indicate high accuracy.
        - **confidence_score**: Boltz's aggregate confidence combining multiple factors.
    """
    ligands: Optional[List[str]] = ConfigField(
        default=None,
        title="Ligands",
        description="Ligand to use for binding strength evaluation.",
    )

    @field_validator('ligands', mode='before')
    @classmethod
    def convert_ligands_to_list(cls, v):
        """Convert single string to list of strings."""
        if v is None:
            return v
        if isinstance(v, str):
            return [v]
        return v

    desired_higher: Dict[str, float] = ConfigField(
        default=DEFAULT_DESIRED_HIGHER,
        title="Desired Higher Bound Metrics",
        description="Target values for 'higher is better' metrics.",  #  Provide partial dict to override specific metrics while keeping defaults for others.
    )
    desired_lower: Dict[str, float] = ConfigField(
        default=DEFAULT_DESIRED_LOWER,
        title="Desired Lower Bound Metrics",
        description="Target values for 'lower is better' metrics.",  # Provide partial dict to override specific metrics.
    )
    tol_higher: Dict[str, float] = ConfigField(
        default=DEFAULT_TOL_HIGHER,
        title="Tolerances Higher Bound Metrics",
        description="Tolerances for 'higher is better' metrics (distance below target = penalty 1.0).",  # Provide partial dict to override.
    )
    tol_lower: Dict[str, float] = ConfigField(
        default=DEFAULT_TOL_LOWER,
        title="Tolerances Lower Bound Metrics",
        description="Tolerances for 'lower is better' metrics (distance above target = penalty 1.0, in Angstroms).",  #  Provide partial dict to override.
    )
    weights: Optional[Dict[str, float]] = ConfigField(
        default=None,
        title="Penalty Weights",
        description="Weights for combining penalties",
    )
    include_confidence_score: bool = ConfigField(
        default=True,
        title="Include Confidence Score",
        description="Whether to include confidence_score in penalty calculation (adds weight 0.10)",
    )
    on_error: Literal["penalize", "raise"] = ConfigField(
        default="penalize",
        title="Behavior on Error",
        description="How to handle prediction errors: 'penalize' (return 1.0) or 'raise' (raise exception)",
    )
    batch_size: Optional[int] = ConfigField(
        default=None,
        title="Batch Size",
        description="Number of complexes to fold at once (None = process all together)",
    )
    return_component: Literal[
        "total_penalty",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
        "complex_iplddt",
        "complex_plddt",
        "complex_pde",
        "complex_ipde",
        "confidence_score",
        "ptm",
    ] = ConfigField(
        default="total_penalty",
        title="Return Component",
        description="Component to return: 'total_penalty' (weighted combination) or specific metric name",
    )

    # Nested Boltz2 configuration
    boltz_config: Optional[BoltzConfig] = ConfigField(
        default=None,
        title="Boltz Config",
        description="Optional Boltz2 configuration. If None, uses default configuration.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Merges user overrides with defaults after validation. If no user overrides
        are provided, uses defaults.
        """
        super().model_post_init(__context)
        self.desired_higher = {**DEFAULT_DESIRED_HIGHER, **self.desired_higher}
        self.desired_lower = {**DEFAULT_DESIRED_LOWER, **self.desired_lower}
        self.tol_higher = {**DEFAULT_TOL_HIGHER, **self.tol_higher}
        self.tol_lower = {**DEFAULT_TOL_LOWER, **self.tol_lower}


@ConstraintRegistry.register(
    key="boltz-binding-strength",
    label="Boltz Binding Strength",
    config=BoltzBindingStrengthConfig,
    description="Evaluate protein-protein/protein-ligand binding using Boltz2 structure prediction",
    batched=True,
    concatenate=False,  # Boltz handles multi-chain complexes
    gpu_required=True,
    tools_called=["boltz"],
    category="protein_structure",
    supported_sequence_types=["dna", "rna", "protein", "ligand"],
)
def boltz_binding_strength_constraint(
    complex_sequences: List[List[Sequence]], config: BoltzBindingStrengthConfig
) -> float | List[float]:
    """Evaluate binding strength and quality using Boltz structure prediction.
    
    This constraint function uses Boltz to predict complex structures and evaluate
    binding quality. Boltz predicts structures for protein-protein, protein-ligand,
    protein-DNA, and protein-RNA complexes, providing confidence metrics that assess
    interface quality, binding strength, and overall structural accuracy.
    
    The constraint evaluates multiple Boltz output metrics (iptm, iplddt, ipde,
    plddt, ptm, confidence_score) against configurable target values and tolerances.
    Each metric is scored as a penalty (0.0 = meets target, 1.0 = at tolerance limit),
    then penalties are combined using weighted averaging. Default weights are
    automatically selected based on complex type (monomer, protein-ligand, or
    protein-protein).
    
    Structure prediction is GPU-intensive and may take several minutes per complex
    depending on size and hardware.

    Args:
        complex_sequences (List[List[Sequence]]): List of complexes to evaluate,
            where each complex is a list of Sequence objects representing the
            chains/molecules. Examples:
            - [[protein_seq]]: Single monomer
            - [[protein_A, protein_B]]: Protein-protein complex
            - [[protein, dna_seq, protein]]: Multi-component complex
            Each Sequence must have appropriate sequence_type (PROTEIN, DNA, or RNA).
        config (BoltzBindingStrengthConfig): Configuration object containing target
            values, tolerances, weights, and Boltz parameters. Uses complex-type-specific
            defaults if not customized.

    Returns:
        List[float]: Constraint scores for each complex, ranging from 0.0 (perfect
            binding, all metrics meet targets) to 1.0 (poor binding, metrics at or
            beyond tolerance limits). The score is either:
            - Weighted combination of all metric penalties (return_component="total_penalty")
            - Specific metric penalty (return_component set to metric name)
            Lower scores indicate stronger, higher-quality predicted binding.

    Raises:
        ValueError: If return_component specifies a metric not available for the
            complex type, or if a metric appears in both desired_higher and desired_lower.
    
    Note:
        This function modifies the input sequences by adding metadata to each
        ``Sequence`` object's ``_metadata`` dictionary. Since complexes contain
        multiple chains, all sequences in a complex receive the same metadata
        under the key ``boltz_binding`` (a list of dictionaries, one per evaluation):
        
        - ``penalty``: Float overall constraint score (0.0-1.0)
        - ``metrics``: Dictionary of all raw Boltz metrics (iptm, iplddt, etc.)
        - ``penalties``: Dictionary of individual metric penalties before weighting
        
        Multiple evaluations on the same sequence (e.g., in different complexes)
        append to the list.
    
    Examples:
        Evaluating protein complex binding:
        
        >>> from proto_language.language.core import Sequence, SequenceType
        >>> protein_a = Sequence("MVLSPADKTNVKAAWGKV", "protein")
        >>> protein_b = Sequence("QFSKPQRTVLMKALNE", "protein")
        >>> config = BoltzBindingStrengthConfig()  # Use defaults
        >>> scores = boltz_binding_strength_constraint([[protein_a, protein_b]], config)
        >>> print(scores[0])  # e.g., 0.15 (good binding)
        >>> metadata = protein_a._metadata["boltz_binding"][0]
        >>> print(f"iptm: {metadata['metrics']['iptm']:.3f}")
        >>> print(f"complex_iplddt: {metadata['metrics']['complex_iplddt']:.3f}")
    """
    ligands = None
    if config.ligands is not None:
        ligands = Ligands(config.ligands)

    complexes = []
    for sequence_list in complex_sequences:
        # Pull chains and entity types from complex sequences
        chains = [s.sequence for s in sequence_list]
        entity_types = [s.sequence_type for s in sequence_list]

        # Add ligands if they are provided
        if ligands is not None:
            chains.extend([fragment.smiles for fragment in ligands.fragments])
            entity_types.extend(["ligand"] * len(ligands.fragments))

        # Add this complex
        complexes.append(
            StructurePredictionComplex(
                chains=chains,
                entity_types=entity_types,
            )
        )

    # Prepare inputs for Boltz2
    inputs = BoltzInput(
        complexes=complexes
    )

    # Run Boltz2
    outputs = run_boltz(inputs=inputs, config=config.boltz_config or BoltzConfig())

    # Scoring each complex
    penalties = []
    for seq_obj_tuple, comp, structure in zip(complex_sequences, inputs, outputs):

        # Determine complex type
        n_chains = comp.num_chains()
        has_ligand = "ligand" in comp.get_entity_type_set()

        # Default weights by case
        if config.weights is not None:
            weights = dict(config.weights)
        else:
            # Weights for monomer
            if n_chains == 1:
                weights = {"ptm": 0.35, "complex_plddt": 0.45, "complex_pde": 0.20}

            # Weights for complex that contains a ligand
            elif has_ligand:
                weights = {
                    "ligand_iptm": 0.50,
                    "complex_iplddt": 0.25,
                    "complex_ipde": 0.15,
                    "complex_plddt": 0.10,
                }
            # Weights for multi-chain complex (no ligand)
            else:
                weights = {
                    "iptm": 0.45,
                    "complex_iplddt": 0.30,
                    "complex_ipde": 0.15,
                    "complex_plddt": 0.10,
                }

        # Add confidence score weight if requested
        if config.include_confidence_score:
            weights.setdefault("confidence_score", 0.10)

        # Initialize penalties dictionary
        penalties_dict = {}

        # Case-specific penalties
        if n_chains == 1:
            # Monomer penalties
            penalties_dict["ptm_penalty"] = get_penalty_for_metric(
                metric_name="ptm", metric_value=structure.ptm, config=config
            )
            penalties_dict["complex_plddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_plddt",
                metric_value=structure.complex_plddt,
                config=config,
            )
            if structure.complex_pde is not None:
                penalties_dict["complex_pde_penalty"] = get_penalty_for_metric(
                    metric_name="complex_pde",
                    metric_value=structure.complex_pde,
                    config=config,
                )

        elif has_ligand:
            penalties_dict["ligand_iptm_penalty"] = get_penalty_for_metric(
                metric_name="ligand_iptm",
                metric_value=structure.ligand_iptm,
                config=config,
            )
            penalties_dict["complex_iplddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_iplddt",
                metric_value=structure.complex_iplddt,
                config=config,
            )
            if structure.complex_ipde is not None:
                penalties_dict["complex_ipde_penalty"] = get_penalty_for_metric(
                    metric_name="complex_ipde",
                    metric_value=structure.complex_ipde,
                    config=config,
                )
            penalties_dict["complex_plddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_plddt",
                metric_value=structure.complex_plddt,
                config=config,
            )

        else:
            prot_iptm = structure.protein_iptm
            iptm = structure.iptm
            chosen = "protein_iptm" if (prot_iptm and prot_iptm > 0) else "iptm"
            if chosen == "iptm":
                penalties_dict["iptm_penalty"] = get_penalty_for_metric(
                    metric_name="iptm",
                    metric_value=iptm,
                    config=config,
                )
            else:
                penalties_dict["protein_iptm_penalty"] = get_penalty_for_metric(
                    metric_name="protein_iptm",
                    metric_value=prot_iptm,
                    config=config,
                )
            penalties_dict["complex_iplddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_iplddt",
                metric_value=structure.complex_iplddt,
                config=config,
            )
            if structure.complex_ipde is not None:
                penalties_dict["complex_ipde_penalty"] = get_penalty_for_metric(
                    metric_name="complex_ipde",
                    metric_value=structure.complex_ipde,
                    config=config,
                )
            penalties_dict["complex_plddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_plddt",
                metric_value=structure.complex_plddt,
                config=config,
            )

        if "confidence_score" in weights:
            penalties_dict["confidence_score_penalty"] = get_penalty_for_metric(
                metric_name="confidence_score",
                metric_value=structure.confidence_score,
                config=config,
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
            penalty = clip(float(penalties_dict[key]), 0.0, 1.0)
        else:
            # Weighted sum
            used_weights = {
                k: weights[k.replace("_penalty", "")]
                for k in penalties_dict
                if k.replace("_penalty", "") in weights
            }
            wsum = sum(used_weights.values()) or 1.0
            penalty = clip(
                sum((w / wsum) * penalties_dict[k] for k, w in used_weights.items()),
                0.0,
                1.0,
            )

        # Store metadata for all Sequences in complex
        for seq_obj in seq_obj_tuple:
            seq_obj._metadata.setdefault("boltz_binding", []).append(
                {
                    "penalty": penalty,
                    "metrics": structure.metrics,
                    "penalties": penalties_dict,
                }
            )

        penalties.append(penalty)

    return penalties


def get_penalty_for_metric(
    metric_name: str, metric_value: float, config: BoltzBindingStrengthConfig
) -> float:
    """
    Retrieves the penalty for the given metric's value based on the default target
    and tolerance values.

    Args:
        metric_name: The name of the metric to retrieve the penalty for.
        metric_value: The value of the metric to retrieve the penalty for.

    Returns:
        The penalty for the given metric's value.
    """

    higher_is_better = None
    target = None
    tolerance = None
    if metric_name in config.desired_higher and metric_name in config.desired_lower:
        raise ValueError(
            f"Metric {metric_name} has both desired_higher and tol_higher values. Please provide only one."
        )
    elif metric_name in config.desired_higher:
        higher_is_better = True
        target = config.desired_higher[metric_name]
        tolerance = config.tol_higher[metric_name]
    elif metric_name in config.desired_lower:
        higher_is_better = False
        target = config.desired_lower[metric_name]
        tolerance = config.tol_lower[metric_name]
    else:
        raise ValueError(
            f"Metric {metric_name} not found in config.desired_higher or config.desired_lower"
        )

    deviation = (target - metric_value) if higher_is_better else (metric_value - target)
    normalized = deviation / max(tolerance, 1e-9)
    return max(0.0, min(1.0, normalized))
