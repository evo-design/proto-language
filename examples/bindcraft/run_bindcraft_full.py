r"""BindCraft de novo protein binder design pipeline.

Reimplementation of BindCraft (Pacesa et al., Nature 2025) on proto-language.
Per trajectory: AF2 hallucination → gates (CA clashes, pLDDT, contacts) →
PyRosetta relax → ProteinMPNN redesign → AF2 cross-validation + monomer RMSD →
PyRosetta interface analysis → ~16 metric filters.

Defaults validated against ``martinpacesa/BindCraft`` (commit 7cd4ace).

Trajectories will not be numerically identical to upstream: proto-tools pins
ColabDesign ``gamma`` (162 commits ahead of ``main``), which restructures the
loss path, fixes the helix-loss masked-diagonal normalization, changes
``aux["seq"]`` to a tensor, rescales ``_tmp["seq_logits"]`` by ``opt["alpha"]``,
and applies pair_mask to MSA row-attention bias.

Known intentional divergences (proto-tools API limits, not bugs): hotspot /
binder monomer RMSD uses CA only (upstream: all heavy atoms via PyRosetta
``RMSDMetric``); ``dropout=False`` not honored in the 5-step hard stage;
``ramp_recycles`` not available (upstream toggles it off on logit_b/softmax/hard).

Usage:
    python examples/bindcraft/run_bindcraft_full.py \
        --target-pdb target.pdb --target-chain A \
        --binder-length-min 50 --binder-length-max 120 \
        --max-trajectories 100 --max-passing 100 \
        --output-dir ./outputs
"""

import argparse
import copy
import itertools
import json
import logging
import math
import string
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.shared_data_models import InverseFoldingStructureInput
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2BinderConfig,
    AlphaFold2BinderInput,
    run_alphafold2_binder,
)
from proto_tools.tools.structure_prediction.dispatch import predict_structures
from proto_tools.tools.structure_scoring.dssp import (
    DSSPSecondaryStructureInput,
    DSSPStructureInput,
    run_dssp_secondary_structure,
)
from proto_tools.tools.structure_scoring.pyrosetta import (
    InterfaceStructureInput,
    PyRosettaEnergyConfig,
    PyRosettaEnergyInput,
    PyRosettaInterfaceAnalyzerConfig,
    PyRosettaInterfaceAnalyzerInput,
    PyRosettaRelaxConfig,
    PyRosettaRelaxInput,
    run_pyrosetta_energy,
    run_pyrosetta_interface_analyzer,
    run_pyrosetta_relax,
)
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import ScoringStructureInput
from proto_tools.utils.tool_io import Metrics

from proto_language import (
    AlphaFold2MultimerStructureConfig,
    MpnnPerplexityConfig,
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
    StructureBasedConstraintConfig,
    mpnn_perplexity_constraint,
    structure_contact_constraint,
    structure_helix_constraint,
    structure_interface_contact_constraint,
    structure_ipae_constraint,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_radius_gyration_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
)
from proto_language.optimizer import (
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)
from proto_language.utils import one_hot_protein_matrix
from proto_language.utils.ml_optimizers import AdamConfig

logger = logging.getLogger(__name__)

# Constants

PROTEINMPNN_GENERATOR_KEY = "proteinmpnn"
BINDER_CHAIN_COFOLD = "A"
TARGET_CHAIN_COFOLD = "B"
CLASH_CA_THRESHOLD = 2.5
MIN_INTERFACE_CONTACTS = 3
INTERFACE_CUTOFF = 4.0
PLDDT_GATE = 0.65
PLDDT_FINAL = 0.7
MAX_MPNN_PER_TRAJECTORY = 2
PYROSETTA_SCORE_FUNCTION = "beta_nov16"  # match BindCraft's -corrections::beta_nov16 energy

GRADIENT_OPTIMIZER_DEFAULTS: dict[str, object] = {
    "lr": 0.1,
    "merger": "weighted_sum",
    "normalize_gradients": True,
    "normalize_mode": "unit",
    "ml_optimizer": "adam",
    "adam_config": AdamConfig(),
}
LOGIT_A_GRADIENT_PARAMS: dict[str, object] = {
    "soft_start": 0.0,
    "soft_end": 0.9,
    "gumbel_logit_init": True,
}
LOGIT_B_GRADIENT_PARAMS: dict[str, object] = {
    "soft_start": 0.0,
    "soft_end": 1.0,
}
LOGITS_GRADIENT_PARAMS: dict[str, object] = {
    "soft_start": 0.0,
    "soft_end": 1.0,
    "gumbel_logit_init": True,
}
SOFTMAX_GRADIENT_PARAMS: dict[str, object] = {
    "soft_start": 1.0,
    "soft_end": 1.0,
    "temperature_start": 1.0,
    "temperature_end": 0.01,
    "softmax_schedule": "quadratic",
    "lr_schedule": "quadratic",  # required for scale_lr_by_temperature to decay the LR
    "scale_lr_by_temperature": True,
}
HARD_GRADIENT_PARAMS: dict[str, object] = {
    "soft_start": 1.0,
    "soft_end": 1.0,
    "hard_start": 1.0,
    "hard_end": 1.0,
    "temperature_start": 0.01,
    "temperature_end": 0.01,
    "scale_lr_by_temperature": True,
}
SEMIGREEDY_MCMC_PARAMS: dict[str, float] = {
    "max_temperature": 1e-6,
    "min_temperature": 1e-7,
}
MCMC_PARAMS: dict[str, float] = {
    "max_temperature": 0.01,
    "min_temperature": 0.0003,
}
BINDCRAFT_AF2_LOSS_FUNCTIONS = {
    "plddt": structure_plddt_constraint,
    "pae": structure_pae_constraint,
    "i_pae": structure_ipae_constraint,
    "con": structure_contact_constraint,
    "i_con": structure_interface_contact_constraint,
    "rg": structure_radius_gyration_constraint,
    "i_ptm": structure_iptm_constraint,
    "helix": structure_helix_constraint,
}

Algorithm = Literal["4stage", "3stage", "2stage", "greedy", "mcmc"]
ValidationTool = Literal["alphafold2", "esmfold", "boltz2", "chai1", "alphafold3"]


# Configuration


@dataclass(frozen=True)
class MetricRule:
    """Threshold rule for a single metric."""

    value: float
    operator: Literal[">", ">=", "<", "<="]

    def evaluate(self, observed: float) -> bool:
        """Return True if observed satisfies the rule."""
        ops = {">": float.__gt__, ">=": float.__ge__, "<": float.__lt__, "<=": float.__le__}
        return ops[self.operator](float(observed), self.value)


@dataclass
class BindCraftConfig:
    """Full pipeline configuration."""

    # Target
    target_pdb: Path
    target_chains: list[str]
    target_hotspot: str | None = None
    binder_length_min: int = 50
    binder_length_max: int = 120

    # Algorithm
    algorithm: Algorithm = "4stage"
    logit_steps: int = 75
    softmax_steps: int = 45
    hard_steps: int = 5
    semigreedy_steps: int = 15

    # Loss weights (BindCraft defaults)
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "plddt": 0.1,
            "pae": 0.4,
            "i_pae": 0.1,
            "con": 1.0,
            "i_con": 1.0,
            "rg": 0.3,
            "i_ptm": 0.05,
        }
    )
    helicity_range: tuple[float, float] = (-0.3, -0.3)
    omit_aas: str = "C"
    force_reject_aa: bool = False

    # MPNN
    mpnn_num_seqs: int = 20
    mpnn_temperature: float = 0.1
    max_mpnn_per_trajectory: int = MAX_MPNN_PER_TRAJECTORY

    # Validation (BindCraft uses AF2 binder cross-validation with the target template)
    validation_tool: ValidationTool = "alphafold2"
    num_validation_models: int = 2
    validation_recycles: int = 3

    # Beta-sheet optimization (BindCraft defaults)
    optimise_beta: bool = True
    beta_threshold: float = 15.0
    extra_softmax_steps: int = 0
    extra_hard_steps: int = 0
    beta_recycles: int = 3

    # Filtering (BindCraft defaults)
    filters: dict[str, MetricRule] = field(
        default_factory=lambda: {
            "plddt": MetricRule(0.8, ">="),
            "ptm": MetricRule(0.55, ">="),
            "iptm": MetricRule(0.5, ">="),
            "i_pae": MetricRule(0.35, "<="),
            "binder_energy": MetricRule(0.0, "<="),
            "surface_hydrophobicity": MetricRule(0.35, "<="),
            "interface_sc": MetricRule(0.6, ">="),
            "interface_dG": MetricRule(0.0, "<="),
            "interface_dSASA": MetricRule(1.0, ">="),
            "n_interface_residues": MetricRule(7, ">="),
            "interface_hbonds": MetricRule(3, ">="),
            "delta_unsat_hbonds": MetricRule(4, "<="),
            "hotspot_rmsd": MetricRule(6.0, "<="),
            "binder_plddt": MetricRule(0.8, ">="),
            "binder_rmsd": MetricRule(3.5, "<="),
            "binder_loop_pct": MetricRule(90, "<="),
            "interface_aa_K": MetricRule(3, "<="),
            "interface_aa_M": MetricRule(3, "<="),
        }
    )

    # Termination
    max_trajectories: int | None = None
    max_accepted: int = 100

    # Acceptance rate monitoring (BindCraft defaults)
    enable_rejection_check: bool = True
    start_monitoring: int = 600
    min_acceptance_rate: float = 0.01

    # Output
    output_dir: str = "./bindcraft_outputs"
    random_seed: int | None = None
    hallucination_only: bool = False


# AF2 Constraint


def _with_helicity_loss(loss_weights: dict[str, float], helicity_weight: float) -> dict[str, float]:
    """Add the upstream BindCraft helicity loss to the weight dict.

    Upstream ``add_helix_loss`` stores the signed weight directly on the
    ``helix`` key — ColabDesign multiplies ``loss_value * weight`` with no
    abs/clamping, so a negative weight discourages helix formation.
    """
    weights = dict(loss_weights)
    if helicity_weight != 0:
        weights["helix"] = helicity_weight
    return weights


def _make_af2_config(
    config: BindCraftConfig,
    target_pdb_text: str,
    seed: int,
    num_recycles: int = 1,
) -> AlphaFold2MultimerStructureConfig:
    """Create the base AF2 binder config used by one hallucination trajectory."""
    af2_cfg = AlphaFold2MultimerStructureConfig(
        target_pdb=target_pdb_text,
        target_chains=config.target_chains,
        binder_chain=None,  # BindCraft is de novo: hallucinate the binder, no template
        backend="base",
        omit_aas=[aa.strip().upper() for aa in config.omit_aas.split(",") if aa.strip()] or None,
        num_recycles=num_recycles,
        sample_models=True,
        use_multimer=True,
        rm_target_seq=False,
        rm_target_sc=False,
        rm_template_ic=False,
        seed=seed,
        intra_contact_num=2,
        intra_contact_cutoff=14.0,
        inter_contact_num=2,
        inter_contact_cutoff=20.0,
    )
    if config.target_hotspot:
        af2_cfg.target_hotspot = config.target_hotspot
    return af2_cfg


def _af2_constraints(
    binder: Segment,
    target: Segment,
    af2_cfg: AlphaFold2MultimerStructureConfig,
    loss_weights: dict[str, float],
) -> list[Constraint]:
    """Create public AF2 multimer constraints for BindCraft's weighted loss terms."""
    constraints: list[Constraint] = []
    for loss_key, weight in loss_weights.items():
        if weight == 0.0:
            continue
        function = BINDCRAFT_AF2_LOSS_FUNCTIONS.get(loss_key)
        if function is None:
            raise ValueError(f"Unsupported BindCraft AF2 loss key: {loss_key!r}")
        constraints.append(
            Constraint(
                inputs=[binder, target],
                function=function,
                function_config=StructureBasedConstraintConfig(
                    structure_tool="alphafold2_multimer",
                    alphafold2_multimer_config=copy.deepcopy(af2_cfg),
                ),
                label=f"af2_{loss_key}",
                weight=weight,
            )
        )
    return constraints


# Hallucination


def _has_4stage_work_after_logit_a(config: BindCraftConfig) -> bool:
    """Return whether a 4stage trajectory has stages left after the first logit pass."""
    return (
        max(0, config.logit_steps - 50) > 0
        or config.softmax_steps > 0
        or config.hard_steps > 0
        or config.semigreedy_steps > 0
    )


def _build_hallucination(
    config: BindCraftConfig,
    binder: Segment,
    target: Segment,
    construct: Construct,
    af2_cfg: AlphaFold2MultimerStructureConfig,
    af2_loss_weights: dict[str, float],
    binder_length: int,
    *,
    skip_logit_a: bool = False,
) -> tuple[Program, list[str]]:
    """Build the selected BindCraft hallucination stages as a proto-language Program."""
    stages: list[GradientOptimizer | MCMCOptimizer] = []
    stage_names: list[str] = []
    greedy_tries = max(1, math.ceil(binder_length * 0.01))

    if config.algorithm == "4stage":
        logit_a_steps = min(50, config.logit_steps)
        logit_b_steps = max(0, config.logit_steps - 50)
        if not skip_logit_a and logit_a_steps > 0:
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            stages.append(
                GradientOptimizer(
                    target_segment=binder,
                    constructs=[construct],
                    generators=[generator],
                    constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                    config=GradientOptimizerConfig(
                        **copy.deepcopy(GRADIENT_OPTIMIZER_DEFAULTS),
                        **LOGIT_A_GRADIENT_PARAMS,
                        num_steps=logit_a_steps,
                    ),
                )
            )
            stage_names.append("logit_a")
        if logit_b_steps > 0:
            generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
            generator.assign(binder)
            stages.append(
                GradientOptimizer(
                    target_segment=binder,
                    constructs=[construct],
                    generators=[generator],
                    constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                    config=GradientOptimizerConfig(
                        **copy.deepcopy(GRADIENT_OPTIMIZER_DEFAULTS),
                        **LOGIT_B_GRADIENT_PARAMS,
                        num_steps=logit_b_steps,
                    ),
                )
            )
            stage_names.append("logit_b")

    if config.algorithm in ("3stage", "2stage") and config.logit_steps > 0:
        generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
        generator.assign(binder)
        stages.append(
            GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                config=GradientOptimizerConfig(
                    **copy.deepcopy(GRADIENT_OPTIMIZER_DEFAULTS),
                    **LOGITS_GRADIENT_PARAMS,
                    num_steps=config.logit_steps,
                ),
            )
        )
        stage_names.append("logits")

    if config.algorithm in ("4stage", "3stage") and config.softmax_steps > 0:
        generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
        generator.assign(binder)
        stages.append(
            GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                config=GradientOptimizerConfig(
                    **copy.deepcopy(GRADIENT_OPTIMIZER_DEFAULTS),
                    **SOFTMAX_GRADIENT_PARAMS,
                    num_steps=config.softmax_steps,
                ),
            )
        )
        stage_names.append("softmax")

    if config.algorithm in ("4stage", "3stage") and config.hard_steps > 0:
        generator = PositionWeightGenerator(PositionWeightGeneratorConfig())
        generator.assign(binder)
        stages.append(
            GradientOptimizer(
                target_segment=binder,
                constructs=[construct],
                generators=[generator],
                constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                config=GradientOptimizerConfig(
                    **copy.deepcopy(GRADIENT_OPTIMIZER_DEFAULTS),
                    **HARD_GRADIENT_PARAMS,
                    num_steps=config.hard_steps,
                ),
            )
        )
        stage_names.append("hard")

    if config.algorithm == "4stage" and config.semigreedy_steps > 0:
        generator = SemigreedyMutationGenerator(
            SemigreedyMutationGeneratorConfig(
                position_weighting="plddt",
                exclude_current=True,
                clear_logits=True,
            )
        )
        generator.assign(binder)
        stages.append(
            MCMCOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                config=MCMCOptimizerConfig(
                    num_steps=config.semigreedy_steps,
                    proposals_per_result=greedy_tries,
                    **SEMIGREEDY_MCMC_PARAMS,
                ),
            )
        )
        stage_names.append("semigreedy")

    if config.algorithm == "2stage" and config.semigreedy_steps > 0:
        generator = SemigreedyMutationGenerator(
            SemigreedyMutationGeneratorConfig(
                position_weighting="plddt",
                exclude_current=True,
                clear_logits=False,
            )
        )
        generator.assign(binder)
        stages.append(
            MCMCOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                config=MCMCOptimizerConfig(
                    num_steps=config.semigreedy_steps,
                    proposals_per_result=greedy_tries,
                    **SEMIGREEDY_MCMC_PARAMS,
                ),
            )
        )
        stage_names.append("semigreedy")

    if config.algorithm == "greedy" and config.semigreedy_steps > 0:
        generator = SemigreedyMutationGenerator(
            SemigreedyMutationGeneratorConfig(
                position_weighting="plddt",
                exclude_current=True,
                clear_logits=True,
            )
        )
        generator.assign(binder)
        stages.append(
            MCMCOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                config=MCMCOptimizerConfig(
                    num_steps=config.semigreedy_steps,
                    proposals_per_result=greedy_tries,
                    **SEMIGREEDY_MCMC_PARAMS,
                ),
            )
        )
        stage_names.append("greedy")

    if config.algorithm == "mcmc" and config.semigreedy_steps > 0:
        generator = SemigreedyMutationGenerator(
            SemigreedyMutationGeneratorConfig(
                position_weighting="plddt",
                exclude_current=True,
                clear_logits=True,
            )
        )
        generator.assign(binder)
        stages.append(
            MCMCOptimizer(
                constructs=[construct],
                generators=[generator],
                constraints=_af2_constraints(binder, target, af2_cfg, af2_loss_weights),
                config=MCMCOptimizerConfig(
                    num_steps=config.semigreedy_steps,
                    proposals_per_result=greedy_tries,
                    **MCMC_PARAMS,
                ),
            )
        )
        stage_names.append("mcmc")

    if not stages:
        raise ValueError(f"No hallucination stages configured for algorithm={config.algorithm!r}.")

    seed = getattr(af2_cfg, "seed", None) or 0
    return Program(optimizers=stages, num_results=1, seed=seed), stage_names


# Quality Gates


def _passes_plddt_gate(binder: Segment, threshold: float) -> bool:
    """Check if binder pLDDT exceeds threshold."""
    result = binder.result_sequences[0]
    if result.structure is None or result.structure.per_residue_plddt is None:
        return False
    plddt = float(np.mean(result.structure.per_residue_plddt))
    logger.info(f"  pLDDT: {plddt:.3f} (gate: {threshold})")
    return plddt > threshold


def _output_chains(target_chains: list[str]) -> tuple[list[str], str]:
    """Positional AF2 output chain ids: (target chains A..N-1, de-novo binder chain N)."""
    n = len(target_chains)
    return [string.ascii_uppercase[i] for i in range(n)], string.ascii_uppercase[n]


def _passes_structural_gates(complex_struct: Structure, binder_struct: Structure, target_chains: list[str]) -> bool:
    """CA clashes, final pLDDT, then interface contacts (BindCraft gate order)."""
    clashes = complex_struct.ca_clash_score(threshold=CLASH_CA_THRESHOLD)
    if clashes > 0:
        logger.info(f"  rejected: {clashes} CA clashes")
        return False

    if binder_struct.per_residue_plddt is None:
        return False
    plddt = float(np.mean(binder_struct.per_residue_plddt))
    if plddt < PLDDT_FINAL:
        logger.info(f"  rejected: pLDDT {plddt:.3f} < {PLDDT_FINAL}")
        return False

    target_out, binder_out = _output_chains(target_chains)
    contacts = complex_struct.interface_contact_residues(
        binder_chain=binder_out,
        target_chains=target_out,
        cutoff=INTERFACE_CUTOFF,
    )
    if len(contacts) < MIN_INTERFACE_CONTACTS:
        logger.info(f"  rejected: {len(contacts)} contacts (need {MIN_INTERFACE_CONTACTS})")
        return False

    logger.info(f"  passed: {len(contacts)} contacts, pLDDT {plddt:.3f}")
    return True


# Scoring


MONOMER_FILTERS = {"binder_plddt", "binder_rmsd"}
AVERAGE_ONLY_FILTERS = {"interface_aa_K", "interface_aa_M"}
AF2_FILTERS = {"plddt", "ptm", "iptm", "i_pae"}
_PER_MODEL_OVERRIDES: dict[str, MetricRule] = {
    "interface_sc": MetricRule(0.55, ">="),
}


def _model_num(model_idx: int) -> int:
    """BindCraft validates with explicit AF2 model numbers."""
    return (model_idx % 5) + 1


def _metric(metrics: Metrics | Mapping[str, Any], name: str, default: float) -> float:
    """Read a numeric metric, returning ``default`` if the key is missing or stored as None."""
    value = metrics.get(name, default)
    return default if value is None else float(value)


def _dssp_secondary_structure_percentages(structure: Structure, chain_id: str) -> dict[str, float]:
    """Return DSSP helix/sheet/loop percentages for one chain."""
    result = run_dssp_secondary_structure(
        DSSPSecondaryStructureInput(inputs=[DSSPStructureInput(structure=structure, chain=chain_id)])
    )
    metrics = result.results[0]
    return {"helix": metrics.helix_pct, "sheet": metrics.sheet_pct, "loop": metrics.loop_pct}


def _passes_filters(
    metrics: dict[str, float | None],
    filters: dict[str, MetricRule],
    *,
    skip: set[str] | None = None,
) -> bool:
    """Apply filters, skipping missing/None metrics like upstream BindCraft."""
    skipped = skip or set()
    for name, rule in filters.items():
        if name in skipped:
            continue
        value = metrics.get(name)
        if value is None:
            continue
        if not rule.evaluate(value):
            return False
    return True


def _predict_validation_complex(
    config: BindCraftConfig,
    binder_seq: str,
    target_seq: str,
    target_pdb_text: str,
    model_idx: int,
) -> tuple[Structure, Metrics, str, list[str]]:
    """Predict the binder-target complex and return structure, metrics, binder chain, target chains."""
    model_num = _model_num(model_idx)
    if config.validation_tool == "alphafold2":
        output = run_alphafold2_binder(
            AlphaFold2BinderInput(
                logits=one_hot_protein_matrix(binder_seq),
                temperature=1.0,
                target_pdb=target_pdb_text,
                target_chain=",".join(config.target_chains),
                target_hotspot=config.target_hotspot,
                binder_chain=None,  # de-novo: fold the MPNN binder sequence, no template
            ),
            AlphaFold2BinderConfig(
                num_recycles=config.validation_recycles,
                model_num=model_num,
                loss_weights={"i_pae": 1.0},
                use_multimer=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=False,
                soft=0.0,
                hard=1.0,
                backend="base",
                compute_gradient=False,
            ),
        )
        target_out, binder_out = _output_chains(config.target_chains)
        return output.structure, output.metrics, binder_out, target_out

    cofold_result = predict_structures(
        complexes=[
            {
                "chains": [
                    {"sequence": binder_seq, "entity_type": "protein"},
                    {"sequence": target_seq, "entity_type": "protein"},
                ]
            }
        ],
        toolkit=config.validation_tool,
    )
    if not cofold_result.structures:
        raise RuntimeError(f"{config.validation_tool} cofold prediction returned no structures")
    cofold_struct = cofold_result.structures[0]
    return cofold_struct, cofold_struct.metrics, BINDER_CHAIN_COFOLD, [TARGET_CHAIN_COFOLD]


def _predict_validation_monomer(config: BindCraftConfig, binder_seq: str, model_idx: int) -> Structure:
    """Predict binder alone for the monomer RMSD/plDDT check."""
    tool_config = None
    if config.validation_tool == "alphafold2":
        tool_config = {"num_recycles": config.validation_recycles, "model_num": _model_num(model_idx)}
    monomer_result = predict_structures(
        complexes=[{"chains": [{"sequence": binder_seq, "entity_type": "protein"}]}],
        toolkit=config.validation_tool,
        tool_config=tool_config,
    )
    if not monomer_result.structures:
        raise RuntimeError(f"{config.validation_tool} monomer prediction returned no structures (model {model_idx})")
    return monomer_result.structures[0]


def _score_variant(
    config: BindCraftConfig,
    binder_seq: str,
    target_seq: str,
    target_pdb_text: str,
    trajectory_complex_struct: Structure,
    trajectory_binder_struct: Structure,
    target_struct: Structure,
) -> dict[str, float]:
    """Cofold, monomer prediction, and PyRosetta scoring.

    Returns the averaged metrics dict for variants that pass all per-model gates,
    or an empty dict on per-model fast-fail. The caller applies the final aggregate
    filter to decide accept/reject. Hard tool failures propagate.
    """
    all_model_metrics: list[dict[str, float]] = []
    binder_plddt_values: list[float] = []
    binder_rmsd_values: list[float] = []

    for model_idx in range(config.num_validation_models):
        model_metrics: dict[str, float] = {}

        cofold_struct, cofold_metrics, binder_chain, target_chains = _predict_validation_complex(
            config,
            binder_seq,
            target_seq,
            target_pdb_text,
            model_idx,
        )

        model_metrics["plddt"] = _metric(cofold_metrics, "avg_plddt", 0)
        model_metrics["ptm"] = _metric(cofold_metrics, "ptm", 0)
        model_metrics["iptm"] = _metric(cofold_metrics, "iptm", 0)
        model_metrics["pae"] = _metric(cofold_metrics, "avg_pae", 999)
        model_metrics["i_pae"] = _metric(cofold_metrics, "i_pae", model_metrics["pae"])

        af2_only = {k: v for k, v in config.filters.items() if k in AF2_FILTERS}
        if not _passes_filters(model_metrics, af2_only):
            logger.info(f"  model {model_idx} failed AF2 fast-fail filters")
            return {}

        model_metrics["hotspot_rmsd"] = trajectory_complex_struct.ca_rmsd_no_superposition(
            cofold_struct,
            self_chain_id=binder_chain,
            other_chain_id=binder_chain,
        )
        model_metrics["target_rmsd"] = float(cofold_struct.select_chains(target_chains).backbone_rmsd(target_struct))

        relax_result = run_pyrosetta_relax(
            PyRosettaRelaxInput(inputs=[ScoringStructureInput(structure=cofold_struct)]),
            PyRosettaRelaxConfig(
                scorefxn=PYROSETTA_SCORE_FUNCTION,
                relax_cycles=1,
                constrain_to_start=True,
                max_iter=200,
                disable_jumps=True,
                min_type="lbfgs_armijo_nonmonotone",
                align_to_start=True,
                copy_b_factors_from_start=True,
            ),
        )
        relaxed = relax_result.results[0].relax.relaxed_structure

        iface = run_pyrosetta_interface_analyzer(
            PyRosettaInterfaceAnalyzerInput(
                inputs=[
                    InterfaceStructureInput(
                        structure=relaxed,
                        binder_chain=binder_chain,
                        target_chains=target_chains,
                    )
                ]
            ),
            PyRosettaInterfaceAnalyzerConfig(scorefxn=PYROSETTA_SCORE_FUNCTION),
        ).results[0]

        model_metrics["interface_sc"] = float(iface.interface_sc)
        model_metrics["interface_dG"] = float(iface.interface_dG)
        model_metrics["interface_dSASA"] = float(iface.interface_dSASA)
        model_metrics["interface_hbonds"] = float(iface.interface_hbonds)
        model_metrics["surface_hydrophobicity"] = float(getattr(iface, "surface_hydrophobicity", 0))
        if getattr(iface, "delta_unsat_hbonds", None) is not None:
            model_metrics["delta_unsat_hbonds"] = float(iface.delta_unsat_hbonds)

        energy = run_pyrosetta_energy(
            PyRosettaEnergyInput(
                inputs=[ScoringStructureInput(structure=relaxed, chains_to_score=[binder_chain])],
            ),
            PyRosettaEnergyConfig(scorefxn=PYROSETTA_SCORE_FUNCTION),
        ).results[0]
        model_metrics["binder_energy"] = float(
            sum(res.total_energy for res in energy.per_residue if res.chain_id == binder_chain)
        )

        iface_residues = relaxed.interface_contact_residues(
            binder_chain=binder_chain,
            target_chains=target_chains,
            cutoff=INTERFACE_CUTOFF,
            include_hydrogens=True,
        )
        model_metrics["n_interface_residues"] = float(len(iface_residues))
        aa_counts: dict[str, int] = {}
        for aa_name in iface_residues.values():
            aa_counts[aa_name] = aa_counts.get(aa_name, 0) + 1
        model_metrics["interface_aa_K"] = float(aa_counts.get("K", aa_counts.get("LYS", 0)))
        model_metrics["interface_aa_M"] = float(aa_counts.get("M", aa_counts.get("MET", 0)))

        ss = _dssp_secondary_structure_percentages(cofold_struct, binder_chain)
        model_metrics["binder_loop_pct"] = ss["loop"]

        per_model_filters = {**config.filters, **_PER_MODEL_OVERRIDES}
        per_model_skip = MONOMER_FILTERS | AVERAGE_ONLY_FILTERS
        if not _passes_filters(model_metrics, per_model_filters, skip=per_model_skip):
            logger.info(f"  model {model_idx} failed per-model filters")
            return {}
        all_model_metrics.append(model_metrics)

        monomer_struct = _predict_validation_monomer(config, binder_seq, model_idx)
        binder_plddt = _metric(monomer_struct.metrics, "avg_plddt", 0)
        binder_rmsd = float(monomer_struct.backbone_rmsd(trajectory_binder_struct))
        monomer_metrics = {"binder_plddt": binder_plddt, "binder_rmsd": binder_rmsd}
        if not _passes_filters(monomer_metrics, {k: v for k, v in config.filters.items() if k in MONOMER_FILTERS}):
            logger.info(f"  model {model_idx} failed monomer filters")
            return {}
        binder_plddt_values.append(binder_plddt)
        binder_rmsd_values.append(binder_rmsd)

    # Average across models
    metrics: dict[str, float] = {}
    if all_model_metrics:
        all_keys = {k for m in all_model_metrics for k in m}
        for key in all_keys:
            vals = [m[key] for m in all_model_metrics if key in m]
            if vals:
                metrics[key] = sum(vals) / len(vals)
    metrics["binder_plddt"] = sum(binder_plddt_values) / len(binder_plddt_values) if binder_plddt_values else 0.0
    metrics["binder_rmsd"] = sum(binder_rmsd_values) / len(binder_rmsd_values) if binder_rmsd_values else 999.0

    return metrics


# Post-Hallucination: MPNN Redesign + Validation


def _redesign_and_validate(
    config: BindCraftConfig,
    binder: Segment,
    construct: Construct,
    complex_struct: Structure,
    mpnn_complex_struct: Structure,
    binder_struct: Structure,
    target_struct: Structure,
    target_pdb_text: str,
    target_seq: str,
    traj_idx: int,
    run_dir: Path,
    seen_sequences: set[str],
) -> int:
    """MPNN redesign, scoring, filtering. Returns accepted count."""
    # Upstream detects interface from the relaxed structure but feeds the
    # unrelaxed trajectory backbone to ProteinMPNN.
    # include_hydrogens=True matches upstream's `hotspot_residues` on a PyRosetta-
    # relaxed PDB, which carries explicit hydrogens that participate in the 4 Å contact set.
    target_out, binder_out = _output_chains(config.target_chains)
    interface_positions = list(
        mpnn_complex_struct.interface_contact_residues(
            binder_chain=binder_out,
            target_chains=target_out,
            cutoff=INTERFACE_CUTOFF,
            include_hydrogens=True,
        ).keys()
    )
    if not interface_positions:
        logger.info("  no relaxed interface residues; skipping MPNN redesign")
        return 0

    fixed_positions = {chain_id: complex_struct.get_chain_positions(chain_id) for chain_id in target_out}
    fixed_positions[binder_out] = interface_positions
    chains_to_redesign = [*target_out, binder_out]

    mpnn = ProteinMPNNGenerator(
        ProteinMPNNGeneratorConfig(
            model_choice="soluble",
            temperature=config.mpnn_temperature,
            excluded_amino_acids=[aa.strip().upper() for aa in config.omit_aas.split(",") if aa.strip()] or None,
            output_chain_id=binder_out,
            structure_inputs=[
                InverseFoldingStructureInput(
                    structure=complex_struct,
                    chains_to_redesign=chains_to_redesign,
                    fixed_positions=fixed_positions,
                )
            ],
            batch_size=config.mpnn_num_seqs,
        ),
    )
    mpnn.assign(binder)
    stage_redesign = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[mpnn],
        constraints=[
            Constraint(
                inputs=[binder],
                function=mpnn_perplexity_constraint,
                # Binder-only: proteinmpnn-gradient needs logits matching the parsed structure
                # length, and proposals are binder-only (ranking uses generator metadata).
                function_config=MpnnPerplexityConfig(
                    structure_input=InverseFoldingStructureInput(
                        structure=binder_struct,
                        chains_to_redesign=[binder_out],
                    ),
                ),
                label="proteinmpnn_perplexity",
            )
        ],
        config=RejectionSamplingOptimizerConfig(
            num_samples=config.mpnn_num_seqs,
            num_results=config.mpnn_num_seqs,
        ),
    )
    Program(optimizers=[stage_redesign], num_results=config.mpnn_num_seqs, seed=traj_idx).run_stage(0)

    forbidden = (
        {aa.strip().upper() for aa in config.omit_aas.split(",") if aa.strip()} if config.force_reject_aa else set()
    )
    candidates: list[tuple[int, float, float, str]] = []
    binder_length = len(binder_struct.get_chain_positions(binder_out))
    for i, seq_obj in enumerate(binder.result_sequences):
        binder_seq = seq_obj.sequence
        if len(binder_seq) != binder_length:
            logger.warning(
                "  skipping MPNN sequence with binder length %d (expected %d)", len(binder_seq), binder_length
            )
            continue
        if any(aa in forbidden for aa in binder_seq) or binder_seq in seen_sequences:
            continue
        seen_sequences.add(binder_seq)
        proteinmpnn_metadata = seq_obj.metadata.get("generators", {}).get(PROTEINMPNN_GENERATOR_KEY, {})
        mpnn_score = float(proteinmpnn_metadata.get("perplexity", float("inf")))
        seq_recovery = float(proteinmpnn_metadata.get("sequence_recovery", 0.0))
        candidates.append((i, mpnn_score, seq_recovery, binder_seq))

    candidates.sort(key=lambda c: c[1])

    accepted = 0
    for variant_idx, mpnn_score, seq_recovery, binder_seq in candidates:
        if accepted >= config.max_mpnn_per_trajectory:
            break

        metrics = _score_variant(
            config,
            binder_seq,
            target_seq,
            target_pdb_text,
            complex_struct,
            binder_struct,
            target_struct,
        )
        if not metrics:
            continue
        metrics["mpnn_score"] = mpnn_score
        metrics["mpnn_sequence_recovery"] = seq_recovery

        passed = _passes_filters(metrics, config.filters)
        status = "accepted" if passed else "rejected"

        stem = f"traj{traj_idx:04d}_v{variant_idx}_{status}"
        with (run_dir / f"{stem}.fasta").open("w") as f:
            f.write(f">binder\n{binder_seq}\n>target\n{target_seq}\n")
        with (run_dir / f"{stem}.json").open("w") as f:
            json.dump(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()} | {"status": status},
                f,
                indent=2,
            )

        logger.info(f"  variant {variant_idx}: {status} (iptm={metrics.get('iptm', 0):.3f})")
        if passed:
            accepted += 1

    return accepted


# Trajectory


def run_trajectory(
    config: BindCraftConfig,
    traj_idx: int,
    seed: int,
    target_pdb_text: str,
    target_seq: str,
    binder_length: int,
    helicity_weight: float,
    run_dir: Path,
    seen_sequences: set[str],
) -> int:
    """Run one full trajectory. Returns number of accepted designs."""
    logger.info(f"[Traj {traj_idx}] length={binder_length}, helicity={helicity_weight:.2f}, seed={seed}")

    binder = Segment(length=binder_length, sequence_type="protein", label="binder")
    target = Segment(sequence=target_seq, sequence_type="protein", label="target")
    construct = Construct([binder, target])

    af2_cfg = _make_af2_config(config, target_pdb_text, seed)
    af2_loss_weights = _with_helicity_loss(config.loss_weights, helicity_weight)
    effective_config = config
    _, binder_chain = _output_chains(config.target_chains)

    # Beta-sheet optimization: run logit_a first, check beta %, adjust remaining stages
    if config.optimise_beta and config.algorithm == "4stage" and config.logit_steps > 0:
        program_a, _ = _build_hallucination(
            config,
            binder,
            target,
            construct,
            af2_cfg,
            af2_loss_weights,
            binder_length,
        )
        logger.info(f"[Traj {traj_idx}] Stage 0: logit_a")
        program_a.run_stage(0)
        if not _passes_plddt_gate(binder, PLDDT_GATE):
            logger.info(f"[Traj {traj_idx}] Abandoned at logit_a gate")
            return 0

        binder_struct_check = binder.result_sequences[0].structure
        if binder_struct_check is not None:
            ss = _dssp_secondary_structure_percentages(binder_struct_check, binder_chain)
            if ss["sheet"] > config.beta_threshold:
                logger.info(
                    f"[Traj {traj_idx}] Beta {ss['sheet']:.1f}% > {config.beta_threshold}% — "
                    f"adding extra steps, increasing recycles to {config.beta_recycles}"
                )
                effective_config = replace(
                    config,
                    softmax_steps=config.softmax_steps + config.extra_softmax_steps,
                    hard_steps=config.hard_steps + config.extra_hard_steps,
                    validation_recycles=config.beta_recycles,
                )
                af2_cfg = _make_af2_config(config, target_pdb_text, seed, num_recycles=config.beta_recycles)

        # Continue with remaining stages (logit_b onward), if any were configured.
        if _has_4stage_work_after_logit_a(effective_config):
            program, stage_names = _build_hallucination(
                effective_config,
                binder,
                target,
                construct,
                af2_cfg,
                af2_loss_weights,
                binder_length,
                skip_logit_a=True,
            )
            for i, name in enumerate(stage_names):
                logger.info(f"[Traj {traj_idx}] Stage {i}: {name}")
                program.run_stage(i)
                if name in ("softmax", "hard") and not _passes_plddt_gate(binder, PLDDT_GATE):
                    logger.info(f"[Traj {traj_idx}] Abandoned at {name} gate")
                    return 0
    else:
        program, stage_names = _build_hallucination(
            effective_config,
            binder,
            target,
            construct,
            af2_cfg,
            af2_loss_weights,
            binder_length,
        )
        for i, name in enumerate(stage_names):
            logger.info(f"[Traj {traj_idx}] Stage {i}: {name}")
            program.run_stage(i)
            if (
                config.algorithm == "4stage"
                and name in ("logit_a", "softmax", "hard")
                and not _passes_plddt_gate(binder, PLDDT_GATE)
            ):
                logger.info(f"[Traj {traj_idx}] Abandoned at {name} gate")
                return 0

    binder_struct = binder.result_sequences[0].structure
    target_struct = target.result_sequences[0].structure
    if binder_struct is None or target_struct is None:
        logger.info(f"[Traj {traj_idx}] No structure produced")
        return 0

    complex_struct = Structure.concat([binder_struct, target_struct])
    if not _passes_structural_gates(complex_struct, binder_struct, config.target_chains):
        logger.info(f"[Traj {traj_idx}] Failed structural gates")
        return 0

    complex_struct.write_pdb(run_dir / f"traj{traj_idx:04d}_trajectory.pdb")

    if config.hallucination_only:
        logger.info(f"[Traj {traj_idx}] Hallucination-only run complete")
        return 1

    # Post-trajectory beta check (BindCraft bindcraft.py:197). Independent of the post-stage-1a
    # check above: re-measure binder SS on the FINAL trajectory structure and lift validation
    # recycles for sheet-rich trajectories. Runs for every algorithm, not just 4stage.
    if config.optimise_beta:
        ss_final = _dssp_secondary_structure_percentages(binder_struct, binder_chain)
        if ss_final["sheet"] > config.beta_threshold:
            logger.info(
                f"[Traj {traj_idx}] Post-trajectory beta {ss_final['sheet']:.1f}% > "
                f"{config.beta_threshold}% — using {config.beta_recycles} validation recycles"
            )
            effective_config = replace(effective_config, validation_recycles=config.beta_recycles)

    relax_result = run_pyrosetta_relax(
        PyRosettaRelaxInput(inputs=[ScoringStructureInput(structure=complex_struct)]),
        PyRosettaRelaxConfig(
            scorefxn=PYROSETTA_SCORE_FUNCTION,
            relax_cycles=1,
            constrain_to_start=True,
            max_iter=200,
            disable_jumps=True,
            min_type="lbfgs_armijo_nonmonotone",
            align_to_start=True,
            copy_b_factors_from_start=True,
        ),
    )
    mpnn_complex_struct = relax_result.results[0].relax.relaxed_structure

    logger.info(f"[Traj {traj_idx}] Starting MPNN redesign + validation")
    accepted = _redesign_and_validate(
        effective_config,
        binder,
        construct,
        complex_struct,
        mpnn_complex_struct,
        binder_struct,
        target_struct,
        target_pdb_text,
        target_seq,
        traj_idx,
        run_dir,
        seen_sequences,
    )
    logger.info(f"[Traj {traj_idx}] {accepted} designs accepted")
    return accepted


# Main


def run_bindcraft(config: BindCraftConfig) -> None:
    """Run the BindCraft pipeline."""
    if len(config.target_chains) > 1 and config.validation_tool != "alphafold2":
        raise ValueError(
            f"validation_tool={config.validation_tool!r} concatenates target_chains into a single chain; "
            f"use validation_tool='alphafold2' for multi-chain targets ({config.target_chains})."
        )
    if config.validation_tool != "alphafold2" and config.num_validation_models != 1:
        logger.warning(
            "validation_tool=%r ignores model_idx; forcing num_validation_models=1 (was %d).",
            config.validation_tool,
            config.num_validation_models,
        )
        config = replace(config, num_validation_models=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / config.target_pdb.stem / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    target_pdb_text = config.target_pdb.read_text()
    target_structure = Structure(structure=target_pdb_text)
    target_seq = "".join(
        target_structure.get_chain_sequence(chain_id, remove_non_standard=True) for chain_id in config.target_chains
    )

    rng = np.random.default_rng(config.random_seed)
    total_accepted = 0

    logger.info(
        f"BindCraft: {config.algorithm}, lengths [{config.binder_length_min}, {config.binder_length_max}], "
        f"max {config.max_trajectories or '∞'} trajectories, target {config.max_accepted} designs"
    )

    all_results: list[dict[str, object]] = []
    seen_sequences: set[str] = set()
    traj_iter = range(config.max_trajectories) if config.max_trajectories is not None else itertools.count()
    for traj_idx in traj_iter:
        seed = int(rng.integers(0, 999999))
        length = int(rng.integers(config.binder_length_min, config.binder_length_max + 1))
        helicity = float(rng.uniform(config.helicity_range[0], config.helicity_range[1]))

        accepted = run_trajectory(
            config,
            traj_idx,
            seed,
            target_pdb_text,
            target_seq,
            length,
            helicity,
            run_dir,
            seen_sequences,
        )
        total_accepted += accepted
        all_results.append(
            {"trajectory": traj_idx, "seed": seed, "length": length, "helicity": helicity, "accepted": accepted}
        )

        if total_accepted >= config.max_accepted:
            logger.info(f"Reached {config.max_accepted} accepted designs — stopping")
            break

        # Acceptance rate monitoring: stop if rate drops below threshold after warmup
        n_trajectories = traj_idx + 1
        if config.enable_rejection_check and n_trajectories >= config.start_monitoring:
            rate = total_accepted / n_trajectories
            if rate < config.min_acceptance_rate:
                logger.warning(
                    f"Acceptance rate {rate:.4f} < {config.min_acceptance_rate} after {n_trajectories} "
                    f"trajectories — consider adjusting settings. Stopping."
                )
                break

    # Rank accepted designs by i_pTM (BindCraft ranking metric)
    accepted_jsons = sorted(p for p in run_dir.iterdir() if p.name.endswith("_accepted.json"))
    ranked_designs = []
    for jf in accepted_jsons:
        with jf.open() as f:
            data = json.load(f)
        ranked_designs.append((data.get("iptm", 0), jf.name))
    ranked_designs.sort(key=lambda x: x[0], reverse=True)
    for rank, (iptm, jf_name) in enumerate(ranked_designs, 1):
        logger.info(f"  Rank {rank}: {jf_name} (i_pTM={iptm:.3f})")

    with (run_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "total_trajectories": len(all_results),
                "total_accepted": total_accepted,
                "ranked": [
                    {"rank": i + 1, "file": jf_name, "iptm": iptm} for i, (iptm, jf_name) in enumerate(ranked_designs)
                ],
                "trajectories": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Done: {total_accepted} accepted from {len(all_results)} trajectories. Output: {run_dir}")


# CLI


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BindCraft protein binder design pipeline.")
    parser.add_argument("--target-pdb", type=Path, required=True)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--target-hotspots", default=None)
    parser.add_argument("--algorithm", choices=["4stage", "3stage", "2stage", "greedy", "mcmc"], default="4stage")
    parser.add_argument("--binder-length-min", type=int, default=50)
    parser.add_argument("--binder-length-max", type=int, default=120)
    parser.add_argument("--logit-steps", type=int, default=75)
    parser.add_argument("--softmax-steps", type=int, default=45)
    parser.add_argument("--hard-steps", type=int, default=5)
    parser.add_argument("--semigreedy-steps", type=int, default=15)
    parser.add_argument("--mpnn-num-seqs", type=int, default=20)
    parser.add_argument("--mpnn-temperature", type=float, default=0.1)
    parser.add_argument("--max-mpnn-per-trajectory", type=int, default=MAX_MPNN_PER_TRAJECTORY)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-passing", type=int, default=100)
    parser.add_argument(
        "--validation-tool",
        choices=["alphafold2", "esmfold", "boltz2", "chai1", "alphafold3"],
        default="alphafold2",
    )
    parser.add_argument("--num-validation-models", type=int, default=2)
    parser.add_argument("--validation-recycles", type=int, default=3)
    parser.add_argument("--no-beta-opt", action="store_true", default=False)
    parser.add_argument(
        "--hallucination-only",
        action="store_true",
        default=False,
        help="Stop after AF2 hallucination and structural gates; useful for GPU smoke tests before MPNN.",
    )
    parser.add_argument("--force-reject-aa", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-rejection-check", action="store_true", default=False)
    parser.add_argument("--output-dir", default="./bindcraft_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    args = parse_args()

    cfg = BindCraftConfig(
        target_pdb=args.target_pdb,
        target_chains=args.target_chain.split(","),
        target_hotspot=args.target_hotspots,
        algorithm=args.algorithm,
        binder_length_min=args.binder_length_min,
        binder_length_max=args.binder_length_max,
        logit_steps=args.logit_steps,
        softmax_steps=args.softmax_steps,
        hard_steps=args.hard_steps,
        semigreedy_steps=args.semigreedy_steps,
        mpnn_num_seqs=args.mpnn_num_seqs,
        mpnn_temperature=args.mpnn_temperature,
        max_mpnn_per_trajectory=args.max_mpnn_per_trajectory,
        max_trajectories=args.max_trajectories,
        max_accepted=args.max_passing,
        validation_tool=args.validation_tool,
        num_validation_models=args.num_validation_models,
        validation_recycles=args.validation_recycles,
        optimise_beta=not args.no_beta_opt,
        hallucination_only=args.hallucination_only,
        force_reject_aa=args.force_reject_aa,
        random_seed=args.seed,
        enable_rejection_check=not args.no_rejection_check,
        output_dir=args.output_dir,
    )

    run_bindcraft(cfg)
