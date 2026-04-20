"""AlphaFold2 binder-design constraint (dual-mode: discrete scoring + gradient)."""

from typing import Any, Literal

import numpy as np
from proto_tools.tools.structure_prediction.alphafold2 import (
    AlphaFold2BinderConfig,
    AlphaFold2BinderInput,
    run_alphafold2_binder,
)
from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import InputSlot, constraint
from proto_language.language.core import PROTEIN_AMINO_ACIDS, Sequence
from proto_language.language.core.constraint import GradientResult
from proto_language.utils import one_hot_protein_matrix

# Loss keys registered only under backend="germinal" (see proto-tools ``_configure_germinal_losses``).
_GERMINAL_ONLY_LOSS_KEYS = frozenset({"rg", "i_ptm", "NC", "helix", "beta_strand"})


class AF2BinderConstraintConfig(BaseConfig):
    """Configuration for the AlphaFold2 binder-design constraint.

    Target template is config-owned; each segment's ``.structure`` slot holds its own
    predicted chain after each call. Use ``Structure.concat`` to rejoin the complex
    for downstream clash / interface checks.

    Attributes:
        target_pdb (str): PDB content of the frozen target template (the "receptor"
            side of the binder-design task). Set once at construction.
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
        starting_binder_seq (str | None): Warm-start binder AA sequence. Germinal
            backend only; length must match the binder segment.
        seed (int | None): Seed forwarded to AF2 (ColabDesign ``set_seed``) for
            per-call reproducibility — e.g. a per-trajectory seed in the Germinal
            pipeline. Seeds both the JAX PRNGKey (recycle / Gumbel / uniform noise)
            and the ``np.random.choice(5)`` model picker when ``sample_models=True``.
    """

    target_pdb: str = ConfigField(
        title="Target PDB",
        default="",
        description="PDB content of the frozen target template.",
    )
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
    starting_binder_seq: str | None = ConfigField(
        title="Starting Binder Sequence",
        default=None,
        description="Warm-start binder AA sequence (Germinal backend only; length must match binder).",
        advanced=True,
    )
    seed: int | None = ConfigField(
        title="Seed",
        default=None,
        ge=0,
        description="Forwarded to ColabDesign ``set_seed`` for per-call AF2 reproducibility.",
        advanced=True,
    )

    @model_validator(mode="after")
    def _require_target_pdb(self) -> "AF2BinderConstraintConfig":
        """Fail fast at config-time if target_pdb is empty; AF2 can't run without a template."""
        if not self.target_pdb:
            raise ValueError("AF2BinderConstraintConfig.target_pdb must be a non-empty PDB string or file path.")
        return self

    @model_validator(mode="after")
    def _reject_germinal_only_fields_on_base(self) -> "AF2BinderConstraintConfig":
        """Reject germinal-only fields under ``backend="base"`` — they'd otherwise silently drop."""
        if self.backend == "germinal":
            return self
        offenders = [
            name
            for name, value in (
                ("bias_redesign", self.bias_redesign),
                ("design_positions", self.design_positions),
                ("starting_binder_seq", self.starting_binder_seq),
            )
            if value is not None
        ]
        offenders.extend(f"loss_weights[{k!r}]" for k in sorted(_GERMINAL_ONLY_LOSS_KEYS & self.loss_weights.keys()))
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
            omit_aas="C",
            bias_redesign=10.0,
            sample_models=True,
            backend="germinal",
        )


AF2BinderForwardConstraintConfig = AF2BinderConstraintConfig
AF2BinderBackwardConstraintConfig = AF2BinderConstraintConfig


def af2_binder_backward(
    inputs: tuple[Sequence, ...],
    *,
    config: AF2BinderBackwardConstraintConfig,
    temperature: float,
    soft: float,
    **kwargs: Any,  # noqa: ARG001
) -> GradientResult:
    """Compute AlphaFold2 binder-design gradient w.r.t. binder logits."""
    binder_seq, target_seq = inputs[0], inputs[1]
    logits = binder_seq.logits
    assert logits is not None  # noqa: S101 -- input_labels slot check guarantees logits on the binder

    output = run_alphafold2_binder(
        AlphaFold2BinderInput(
            logits=logits.tolist(),
            temperature=temperature,
            target_pdb=config.target_pdb,
            target_chain=config.target_chain,
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
            bias_redesign=config.bias_redesign,
            sample_models=config.sample_models,
            backend=config.backend,
            starting_binder_seq=config.starting_binder_seq,
            seed=config.seed,
            soft=soft,
            compute_gradient=True,
        ),
    )
    if output.gradient is None:
        raise RuntimeError("compute_gradient=True must populate output.gradient")
    binder_gradient = np.array(output.gradient, dtype=np.float64)
    target_gradient = np.zeros((len(target_seq.sequence), len(PROTEIN_AMINO_ACIDS)), dtype=np.float64)
    # Each slot gets its own predicted chain — rejoin via Structure.concat (shared AF2 frame).
    return GradientResult(
        gradient=(binder_gradient, target_gradient),
        loss=output.loss,
        metrics=output.metrics,
        structures=(
            output.structure.select_chain(config.binder_chain),
            output.structure.select_chain(config.target_chain),
        ),
    )


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
) -> list[float]:
    """Forward AF2 binder scoring for discrete optimizers.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal ``(binder_seq, target_seq)``.
        config (AF2BinderForwardConstraintConfig): Binder-design config.

    Returns:
        list[float]: Per-proposal raw AF2 loss; lower is better.
    """
    scores: list[float] = []
    for binder_seq, target_seq in input_sequences:
        # Forward-only scoring evaluates AF2 on the exact discrete proposal. Pass a true one-hot
        # matrix and force ColabDesign's STE (hard=1) so the argmax gets through unchanged.
        output = run_alphafold2_binder(
            AlphaFold2BinderInput(
                logits=one_hot_protein_matrix(binder_seq.sequence),
                target_pdb=config.target_pdb,
                target_chain=config.target_chain,
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
                bias_redesign=config.bias_redesign,
                sample_models=config.sample_models,
                backend=config.backend,
                starting_binder_seq=config.starting_binder_seq,
                seed=config.seed,
                soft=0.0,
                hard=1.0,
                compute_gradient=False,
            ),
        )

        binder_seq.structure = output.structure.select_chain(config.binder_chain)
        target_seq.structure = output.structure.select_chain(config.target_chain)
        # Metrics first so our explicit keys below win on collision.
        binder_seq._metadata.update(output.metrics)
        binder_seq._metadata["complex_pdb"] = output.structure.structure_pdb
        binder_seq._metadata["loss"] = output.loss
        scores.append(output.loss)

    return scores


# Germinal semigreedy ranks proposals on the raw AF2 loss, so this intentional
# exception keeps discrete Stage-2 scoring aligned with the scientific objective.
af2_binder_forward._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
