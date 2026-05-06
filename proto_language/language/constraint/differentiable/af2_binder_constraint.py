"""AlphaFold2 binder-design constraint (dual-mode: discrete scoring + gradient)."""

from typing import Any, Literal

import numpy as np
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2BinderConfig,
    AlphaFold2BinderInput,
    run_alphafold2_binder,
)
from proto_tools.utils import AminoAcid
from pydantic import field_validator, model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import InputSlot, constraint
from proto_language.language.core import PROTEIN_AMINO_ACIDS, ConstraintOutput, Sequence
from proto_language.language.core.constraint import GradientConstraintOutput
from proto_language.utils import one_hot_protein_matrix


class AF2BinderConstraintConfig(BaseConfig):
    """Configuration for the AlphaFold2 binder-design constraint.

    Target template is config-owned; each segment's ``.structure`` slot holds its own
    predicted chain after each call. Use ``Structure.concat`` to rejoin the complex
    for downstream clash / interface checks.

    Attributes:
        target_pdb (str): PDB content of the frozen target template (the "receptor"
            side of the binder-design task). Set once at construction.
        target_chains (list[str]): Chain ID(s) of the frozen target in the PDB.
        binder_chain (str): Binder chain ID for template-based binder redesign.
        target_hotspot (str | None): Comma-separated target residue indices for interface contacts.
        design_positions (list[int] | None): Zero-based binder residue indices for loss focus.
        loss_weights (dict[str, float]): Binder-objective weights passed to ColabDesign.
        omit_aas (list[AminoAcid] | None): Amino acids to ban during optimization.
        num_recycles (int): Number of recycling iterations.
        model_num (int): AF2 model parameter set (1-5).
        intra_contact_num (int): Intra-molecular contacts per residue for contact loss.
        intra_contact_cutoff (float): Distance cutoff for intra-molecular contacts.
        inter_contact_num (int): Inter-molecular contacts per residue.
        inter_contact_cutoff (float): Distance cutoff for inter-molecular contacts.
        framework_contact_offset (float): Framework contact penalty offset in the
            Germinal inter-chain contact loss. Germinal backend only.
        bias_redesign (float | None): Soft bias toward wildtype at non-design positions.
        sample_models (bool): Randomly sample AF2 model parameter sets each forward pass.
        use_multimer (bool): Use AlphaFold multimer parameters for binder protocol.
        rm_target_seq (bool): Mask target template sequence in ``prep_inputs``.
        rm_target_sc (bool): Mask target template side chains in ``prep_inputs``.
        rm_template_ic (bool): Mask inter-chain template contacts in ``prep_inputs``.
        backend (Literal["base", "germinal"]): ColabDesign backend.
        device (str): GPU device for AlphaFold2, for example ``"cuda"`` or
            ``"cuda:0"``.
        seed (int | None): Base AF2 seed for the evaluation stream. When set, the
            constraint derives a unique per-evaluation ColabDesign seed from it so
            repeated calls do not reset AF2 to the same RNG state.
    """

    target_pdb: str = ConfigField(
        title="Target PDB",
        default="",
        description="PDB content of the frozen target template.",
    )
    target_chains: list[str] = ConfigField(
        title="Target Chains",
        default_factory=lambda: ["A"],
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
    omit_aas: list[AminoAcid] | None = ConfigField(
        title="Omit Amino Acids",
        default=None,
        description="Amino acids to ban during optimization (e.g. ['C', 'W']).",
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
    framework_contact_offset: float = ConfigField(
        title="Framework Contact Offset",
        default=1.0,
        gt=0.0,
        advanced=True,
        description="Framework contact penalty offset in the Germinal i_con loss.",
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
    use_multimer: bool = ConfigField(
        title="Use Multimer",
        default=True,
        description="Use AlphaFold multimer parameters for binder protocol.",
        advanced=True,
    )
    rm_target_seq: bool = ConfigField(
        title="Mask Target Sequence",
        default=True,
        description="Mask target template sequence in ColabDesign prep_inputs.",
        advanced=True,
    )
    rm_target_sc: bool = ConfigField(
        title="Mask Target Side Chains",
        default=False,
        description="Mask target template side chains in ColabDesign prep_inputs.",
        advanced=True,
    )
    rm_template_ic: bool = ConfigField(
        title="Mask Template Contacts",
        default=True,
        description="Mask inter-chain template contacts in ColabDesign prep_inputs.",
        advanced=True,
    )
    backend: Literal["base", "germinal"] = ConfigField(
        title="Backend",
        default="base",
        description="ColabDesign backend: 'base' (upstream) or 'germinal' (with alpha, bias, IgLM).",
        advanced=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="GPU device for AlphaFold2 (e.g. 'cuda' or 'cuda:0').",
        hidden=True,
    )
    seed: int | None = ConfigField(
        title="Seed",
        default=None,
        ge=0,
        description="Base AF2 seed; the constraint derives a unique per-evaluation ColabDesign seed from it.",
        advanced=True,
    )
    _evaluation_seed_offset: int = 0

    @field_validator("target_chains", mode="before")
    @classmethod
    def _normalize_target_chains(cls, value: Any) -> list[str]:
        """Accept comma-separated strings or explicit lists; store a clean chain list."""
        raw_chain_ids = [value] if isinstance(value, str) else value
        if not isinstance(raw_chain_ids, (list, tuple)) or not all(isinstance(c, str) for c in raw_chain_ids):
            raise ValueError("target_chains must be a string or list of strings.")
        chains = [chain.strip() for raw in raw_chain_ids for chain in raw.split(",") if chain.strip()]
        if not chains:
            raise ValueError("target_chains must contain at least one chain ID.")
        return chains

    @model_validator(mode="after")
    def _require_target_pdb(self) -> "AF2BinderConstraintConfig":
        """Fail fast at config-time if target_pdb is empty; AF2 can't run without a template."""
        if not self.target_pdb:
            raise ValueError("AF2BinderConstraintConfig.target_pdb must be a non-empty PDB string or file path.")
        return self

    @model_validator(mode="after")
    def _reject_germinal_only_fields_on_base(self) -> "AF2BinderConstraintConfig":
        """Reject germinal-fork features that need the fork's ``prep_inputs`` or contact config."""
        if self.backend == "germinal":
            return self
        offenders = [
            name
            for name, value in (
                ("bias_redesign", self.bias_redesign),
                ("design_positions", self.design_positions),
            )
            if value is not None
        ]
        if self.framework_contact_offset != 1.0:
            offenders.append("framework_contact_offset")
        if offenders:
            raise ValueError(f"{offenders} require backend='germinal'; got {self.backend!r}.")
        return self

    @classmethod
    def germinal_vhh_preset(cls, target_pdb: str, binder_chain: str = "H") -> "AF2BinderConstraintConfig":
        """Germinal VHH preset matching vhh.yaml defaults.

        Args:
            target_pdb (str): PDB content (or file path) of the frozen target template.
            binder_chain (str): Binder chain ID in the template PDB. Defaults to 'H'.
        """
        return cls(
            target_pdb=target_pdb,
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
            omit_aas=["C"],
            bias_redesign=10.0,
            inter_contact_num=10,
            inter_contact_cutoff=20.0,
            framework_contact_offset=1.0,
            sample_models=True,
            backend="germinal",
        )


AF2BinderForwardConstraintConfig = AF2BinderConstraintConfig
AF2BinderBackwardConstraintConfig = AF2BinderConstraintConfig


def _next_af2_seed(config: AF2BinderConstraintConfig) -> int | None:
    """Derive deterministic per-evaluation seeds instead of replaying one fixed AF2 RNG state."""
    if config.seed is None:
        return None
    seed = config.seed + config._evaluation_seed_offset
    config._evaluation_seed_offset += 1
    return seed


def af2_binder_backward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AF2BinderBackwardConstraintConfig,
    temperature: float,
    soft: float,
    hard: float = 0.0,
    **kwargs: Any,  # noqa: ARG001
) -> list[GradientConstraintOutput]:
    """Compute AlphaFold2 binder-design gradient w.r.t. binder logits (batched)."""
    results: list[GradientConstraintOutput] = []
    for binder_seq, target_seq in input_sequences:
        logits = binder_seq.logits
        assert logits is not None  # noqa: S101 -- input_labels slot check guarantees logits on the binder
        evaluation_seed = _next_af2_seed(config)

        output = run_alphafold2_binder(
            AlphaFold2BinderInput(
                logits=logits.tolist(),
                temperature=temperature,
                target_pdb=config.target_pdb,
                target_chain=",".join(config.target_chains),
                target_hotspot=config.target_hotspot,
                binder_chain=config.binder_chain,
                design_positions=config.design_positions,
            ),
            AlphaFold2BinderConfig(
                omit_aas=config.omit_aas,
                num_recycles=config.num_recycles,
                model_num=config.model_num,
                loss_weights=config.loss_weights,
                intra_contact_num=config.intra_contact_num,
                intra_contact_cutoff=config.intra_contact_cutoff,
                inter_contact_num=config.inter_contact_num,
                inter_contact_cutoff=config.inter_contact_cutoff,
                framework_contact_offset=config.framework_contact_offset,
                bias_redesign=config.bias_redesign,
                sample_models=config.sample_models,
                use_multimer=config.use_multimer,
                rm_target_seq=config.rm_target_seq,
                rm_target_sc=config.rm_target_sc,
                rm_template_ic=config.rm_template_ic,
                backend=config.backend,
                device=config.device,
                seed=evaluation_seed,
                soft=soft,
                hard=hard,
                compute_gradient=True,
            ),
        )
        if output.gradient is None:
            raise RuntimeError("compute_gradient=True must populate output.gradient")
        binder_gradient = np.array(output.gradient, dtype=np.float64)
        target_gradient = np.zeros((len(target_seq.sequence), len(PROTEIN_AMINO_ACIDS)), dtype=np.float64)
        results.append(
            GradientConstraintOutput(
                gradient=(binder_gradient, target_gradient),
                loss=output.loss,
                metrics=output.metrics,
                structures=(
                    output.structure.select_chain(config.binder_chain),
                    output.structure.select_chains(config.target_chains),
                ),
            )
        )
    return results


@constraint(
    key="af2-binder",
    label="AF2 Binder Design",
    config=AF2BinderForwardConstraintConfig,
    description="AF2 binder design against a fixed target: scores binder sequences (discrete) or computes gradients w.r.t. logits (differentiable).",
    tools_called=["alphafold2-binder"],
    uses_gpu=True,
    category="differentiable",
    supported_sequence_types=["protein"],
    input_labels=[
        InputSlot(label="Binder Chain", requires_logits=True),
        InputSlot(label="Target"),  # Template lives on config.target_pdb; slot holds the predicted target chain.
    ],
    backward=af2_binder_backward,
    backward_config=AF2BinderBackwardConstraintConfig,
)
def af2_binder_forward(
    input_sequences: list[tuple[Sequence, ...]],
    *,
    config: AF2BinderForwardConstraintConfig,
) -> list[ConstraintOutput]:
    """Forward AF2 binder scoring for discrete optimizers.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal ``(binder_seq, target_seq)``.
        config (AF2BinderForwardConstraintConfig): Binder-design config.

    Returns:
        list[ConstraintOutput]: Per-proposal raw AF2 loss (lower is better) with AF2 metrics,
            ``complex_pdb``, and ``loss`` metadata, plus per-slot predicted chains.
    """
    results: list[ConstraintOutput] = []
    for binder_seq, _target_seq in input_sequences:
        # Forward-only scoring evaluates AF2 on the exact discrete proposal. Pass a true one-hot
        # matrix and force ColabDesign's STE (hard=1) so the argmax gets through unchanged.
        evaluation_seed = _next_af2_seed(config)
        output = run_alphafold2_binder(
            AlphaFold2BinderInput(
                logits=one_hot_protein_matrix(binder_seq.sequence),
                target_pdb=config.target_pdb,
                target_chain=",".join(config.target_chains),
                target_hotspot=config.target_hotspot,
                binder_chain=config.binder_chain,
                design_positions=config.design_positions,
            ),
            AlphaFold2BinderConfig(
                omit_aas=config.omit_aas,
                num_recycles=config.num_recycles,
                model_num=config.model_num,
                loss_weights=config.loss_weights,
                intra_contact_num=config.intra_contact_num,
                intra_contact_cutoff=config.intra_contact_cutoff,
                inter_contact_num=config.inter_contact_num,
                inter_contact_cutoff=config.inter_contact_cutoff,
                framework_contact_offset=config.framework_contact_offset,
                bias_redesign=config.bias_redesign,
                sample_models=config.sample_models,
                use_multimer=config.use_multimer,
                rm_target_seq=config.rm_target_seq,
                rm_target_sc=config.rm_target_sc,
                rm_template_ic=config.rm_template_ic,
                backend=config.backend,
                device=config.device,
                seed=evaluation_seed,
                soft=0.0,
                hard=1.0,
                compute_gradient=False,
            ),
        )
        # Metrics first so our explicit keys below win on collision.
        metadata = {**output.metrics, "complex_pdb": output.structure.structure_pdb, "loss": output.loss}
        results.append(
            ConstraintOutput(
                score=output.loss,
                metadata=metadata,
                metadata_recipient="binder",
                structures=(
                    output.structure.select_chain(config.binder_chain),
                    output.structure.select_chains(config.target_chains),
                ),
            )
        )

    return results


# Germinal semigreedy ranks proposals on the raw AF2 loss, so this intentional
# exception keeps discrete Stage-2 scoring aligned with the scientific objective.
af2_binder_forward._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
