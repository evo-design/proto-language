"""AlphaFold2 binder-design gradient constraint."""

from typing import Any, Literal

import numpy as np
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2GradientConfig,
    AlphaFold2GradientInput,
    run_alphafold2_gradient,
)

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import PROTEIN_AMINO_ACIDS, Sequence
from proto_language.language.core.constraint import GradientResult


class AF2BinderGradientConfig(BaseConfig):
    """Configuration for the AlphaFold2 binder-design gradient constraint.

    Maps ColabDesign binder-gradient knobs to a constraint config so the
    gradient optimizer can drive AF2 binder redesign.

    The target structure is read from the second input segment's
    ``Sequence.structure`` field — no file path needed in config.

    Attributes:
        target_chain (str): Chain ID(s) of the frozen target in the PDB.
        binder_chain (str): Binder chain ID for template-based binder redesign.
        target_hotspot (str | None): Comma-separated target residue indices for interface contacts.
        design_positions (list[int] | None): Zero-based binder residue indices for loss focus.
        loss_weights (dict[str, float]): Binder-objective weights passed to ColabDesign.
        omit_aas (str | None): Amino acids to ban during optimization.
        num_recycles (int): Number of recycling iterations.
        model_num (int): AF2 model parameter set (1-5).
        intra_contact_num (int): Intra-molecular contacts per residue for contact loss.
        intra_contact_cutoff (float): Distance cutoff for intra-molecular contacts.
        inter_contact_num (int): Inter-molecular contacts per residue.
        inter_contact_cutoff (float): Distance cutoff for inter-molecular contacts.
        bias_redesign (float | None): Soft bias toward wildtype at non-design positions.
        sample_models (bool): Randomly sample AF2 model parameter sets each forward pass.
        backend (Literal["base", "germinal"]): ColabDesign backend.
    """

    target_chain: str = ConfigField(
        title="Target Chain",
        default="A",
        description="Chain ID(s) of the frozen target in the PDB.",
    )
    binder_chain: str = ConfigField(
        title="Binder Chain",
        default="H",
        description="Binder chain ID for template-based binder redesign.",
    )
    target_hotspot: str | None = ConfigField(
        title="Target Hotspot",
        default=None,
        description="Comma-separated hotspot residue indices on the target.",
    )
    design_positions: list[int] | None = ConfigField(
        title="Design Positions",
        default=None,
        description="Zero-based binder residue indices the losses focus on (e.g. CDR loops).",
    )
    loss_weights: dict[str, float] = ConfigField(
        title="Loss Weights",
        default_factory=lambda: {"plddt": 1.0, "i_pae": 1.0, "i_con": 1.0, "con": 0.5},
        description="Binder-objective weights (e.g. plddt, i_pae, i_con, con).",
    )
    omit_aas: str | None = ConfigField(
        title="Omit Amino Acids",
        default=None,
        description="Comma-separated amino acids to ban during optimization, e.g. 'C,W'.",
        advanced=True,
    )
    num_recycles: int = ConfigField(
        title="Number of Recycles", default=3, ge=0, le=48, description="Number of recycling iterations.", advanced=True
    )
    model_num: int = ConfigField(
        title="Model Number", default=1, ge=1, le=5, description="Which AF2 model parameter set (1-5).", advanced=True
    )
    intra_contact_num: int = ConfigField(
        title="Intra Contact Number",
        default=2,
        ge=1,
        advanced=True,
        description="Intra-molecular contacts per residue.",
    )
    intra_contact_cutoff: float = ConfigField(
        title="Intra Contact Cutoff",
        default=14.0,
        gt=0.0,
        advanced=True,
        description="Distance cutoff for intra contacts.",
    )
    inter_contact_num: int = ConfigField(
        title="Inter Contact Number",
        default=10,
        ge=1,
        advanced=True,
        description="Inter-molecular contacts per residue.",
    )
    inter_contact_cutoff: float = ConfigField(
        title="Inter Contact Cutoff",
        default=20.0,
        gt=0.0,
        advanced=True,
        description="Distance cutoff for inter contacts.",
    )
    bias_redesign: float | None = ConfigField(
        title="Bias Redesign",
        default=None,
        gt=0.0,
        description="Soft bias strength for non-design positions toward the wildtype template.",
        advanced=True,
    )
    sample_models: bool = ConfigField(
        title="Sample Models",
        default=False,
        description="Randomly sample from available AF2 model parameter sets each forward pass.",
        advanced=True,
    )
    backend: Literal["base", "germinal"] = ConfigField(
        title="Backend",
        default="base",
        description="ColabDesign backend: 'base' (upstream) or 'germinal' (with alpha, bias, IgLM).",
        advanced=True,
    )

    @classmethod
    def germinal_vhh_preset(cls, binder_chain: str = "H") -> "AF2BinderGradientConfig":
        """Germinal VHH preset matching vhh.yaml defaults."""
        return cls(
            binder_chain=binder_chain,
            loss_weights={
                "plddt": 1.0,
                "i_plddt": 1.0,
                "pae": 0.1,
                "i_pae": 0.5,
                "con": 0.1,
                "i_con": 0.2,
                "rg": 0.1,
                "i_ptm": 0.75,
                "helix": 0.1,
                "beta_strand": 0.2,
                "dgram_cce": 0.01,
            },
            bias_redesign=10.0,
            sample_models=True,
            backend="germinal",
        )


@constraint(
    key="af2-binder-gradient",
    label="AF2 Binder Structure Gradient",
    config=AF2BinderGradientConfig,
    description="Differentiable binder-design constraint using AlphaFold2/ColabDesign gradients",
    tools_called=["alphafold2-gradient"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
    input_labels=["Binder Chain", "Target Structure"],
)
def af2_binder_backward(
    inputs: tuple[Sequence, ...],
    *,
    config: AF2BinderGradientConfig,
    temperature: float,
    soft: float | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> GradientResult:
    """Compute AlphaFold2 binder-design gradient through ColabDesign."""
    binder_seq, target_seq = inputs[0], inputs[1]
    logits = binder_seq.logits
    if logits is None:
        raise RuntimeError("Binder segment has no logits set")
    if target_seq.structure is None:
        raise RuntimeError("Target segment has no structure set")

    target_pdb = target_seq.structure.structure_pdb

    output = run_alphafold2_gradient(
        AlphaFold2GradientInput(
            logits=logits.tolist(),
            temperature=temperature,
            target_pdb=target_pdb,
            target_chain=config.target_chain,
            target_hotspot=config.target_hotspot,
            binder_chain=config.binder_chain,
            design_positions=config.design_positions,
        ),
        AlphaFold2GradientConfig(
            omit_aas=config.omit_aas,
            num_recycles=config.num_recycles,
            model_num=config.model_num,
            loss_weights=config.loss_weights,
            intra_contact_num=config.intra_contact_num,
            intra_contact_cutoff=config.intra_contact_cutoff,
            inter_contact_num=config.inter_contact_num,
            inter_contact_cutoff=config.inter_contact_cutoff,
            bias_redesign=config.bias_redesign,
            sample_models=config.sample_models,
            backend=config.backend,
            soft=soft if soft is not None else 1.0,
        ),
    )
    binder_gradient = np.array(output.gradient, dtype=np.float64)
    target_gradient = np.zeros((len(target_seq.sequence), len(PROTEIN_AMINO_ACIDS)), dtype=np.float64)
    return GradientResult(
        gradient=(binder_gradient, target_gradient),
        loss=output.loss,
        metrics=output.metrics,
    )
