"""Base configuration classes for structure-based constraints.

This module provides standardized configuration classes for constraints that
use structure prediction tools (ESMFold, AlphaFold3, Boltz2, Chai1, Protenix,
AlphaFold2 multimer).
"""

from typing import Any, Literal

from proto_tools import (
    AlphaFold3Config,
    Boltz2Config,
    Chai1Config,
    ESMFold2Config,
    ESMFoldConfig,
    ProtenixConfig,
)
from proto_tools.utils import AminoAcid
from pydantic import PrivateAttr, field_validator, model_validator

from proto_language.utils.base import BaseConfig, ConfigField


class AlphaFold2MultimerStructureConfig(BaseConfig):
    """Configuration for AF2 multimer-backed structure constraints.

    Attributes:
        target_pdb (str): Target+binder template PDB content or path.
        binder_input_index (int): Input slot containing the optimizable binder.
        target_input_indices (list[int]): Input slots containing frozen target chains.
        target_chains (list[str]): Target chain IDs in the template PDB.
        binder_chain (str): Binder chain ID in the template PDB.
        target_hotspot (str | None): Comma-separated target hotspot residue IDs.
        design_positions (list[int] | None): Binder positions used by focused losses.
        omit_aas (list[AminoAcid] | None): Amino acids omitted during AF2 optimization.
        num_recycles (int): Number of AF2 recycle iterations.
        recycle_mode (Literal["last", "sample", "average", "first"]): Recycle output mode.
        model_num (int): AF2 model parameter set number.
        intra_contact_num (int): Intra-chain contacts per residue.
        intra_contact_cutoff (float): Intra-chain contact cutoff in Angstroms.
        inter_contact_num (int): Inter-chain contacts per residue.
        inter_contact_cutoff (float): Inter-chain contact cutoff in Angstroms.
        framework_contact_offset (float): Germinal framework contact penalty offset.
        bias_redesign (float | None): Bias strength toward template residues.
        sample_models (bool): Whether to sample AF2 model parameters.
        use_multimer (bool): Whether to use multimer model parameters.
        rm_target_seq (bool): Whether to mask the target template sequence.
        rm_target_sc (bool): Whether to mask target template side chains.
        rm_template_ic (bool): Whether to mask inter-chain template contacts.
        include_pae_matrix (bool): Whether to return the full PAE matrix.
        backend (Literal["base", "germinal"]): AF2 backend implementation.
        device (str): Device for AF2 execution.
        seed (int | None): Base seed for deterministic AF2 evaluations.
    """

    target_pdb: str = ConfigField(
        default="",
        title="Target PDB",
        description="Target+binder template PDB content or path.",
    )
    binder_input_index: int = ConfigField(
        default=0,
        title="Binder Input",
        description="Input slot containing the optimizable binder.",
        ge=0,
    )
    target_input_indices: list[int] = ConfigField(
        default_factory=lambda: [1],
        title="Target Inputs",
        description="Input slots containing frozen target chains.",
    )
    target_chains: list[str] = ConfigField(
        default_factory=lambda: ["A"],
        title="Target Chains",
        description="Target chain IDs in the template PDB.",
    )
    binder_chain: str = ConfigField(
        default="H",
        title="Binder Chain",
        description="Binder chain ID in the template PDB.",
    )
    target_hotspot: str | None = ConfigField(
        default=None,
        title="Target Hotspot",
        description="Comma-separated target hotspot residue IDs.",
    )
    design_positions: list[int] | None = ConfigField(
        default=None,
        title="Design Positions",
        description="Binder positions used by focused losses (zero-based, ColabDesign convention).",
    )
    omit_aas: list[AminoAcid] | None = ConfigField(
        default=None,
        title="Omit Amino Acids",
        description="Amino acids omitted during AF2 optimization.",
    )
    num_recycles: int = ConfigField(
        default=3,
        title="Recycle Count",
        description="Number of AF2 recycle iterations.",
        ge=0,
        le=48,
    )
    recycle_mode: Literal["last", "sample", "average", "first"] = ConfigField(
        default="last",
        title="Recycle Mode",
        description="How to combine AF2 recycle iterations: last, sample, average, or first.",
    )
    model_num: int = ConfigField(
        default=1,
        title="Model Number",
        description="AF2 model parameter set number.",
        ge=1,
        le=5,
    )
    intra_contact_num: int = ConfigField(
        default=2,
        title="Intra Contacts",
        description="Intra-chain contacts per residue.",
        ge=1,
    )
    intra_contact_cutoff: float = ConfigField(
        default=14.0,
        title="Intra Cutoff (Å)",
        description="Intra-chain contact distance cutoff in Ångströms.",
        gt=0.0,
    )
    inter_contact_num: int = ConfigField(
        default=1,
        title="Inter Contacts",
        description="Inter-chain contacts per residue.",
        ge=1,
    )
    inter_contact_cutoff: float = ConfigField(
        default=21.6875,
        title="Inter Cutoff (Å)",
        description="Inter-chain contact distance cutoff in Ångströms.",
        gt=0.0,
    )
    framework_contact_offset: float = ConfigField(
        default=1.0,
        title="Framework Offset",
        description="Germinal framework contact penalty offset.",
        gt=0.0,
    )
    bias_redesign: float | None = ConfigField(
        default=None,
        title="Bias Redesign",
        description="Bias strength toward template residues.",
        gt=0.0,
    )
    sample_models: bool = ConfigField(
        default=False,
        title="Sample Models",
        description="Whether to sample AF2 model parameters.",
    )
    use_multimer: bool = ConfigField(
        default=True,
        title="Use Multimer",
        description="Whether to use multimer model parameters.",
    )
    rm_target_seq: bool = ConfigField(
        default=True,
        title="Mask Target Seq",
        description="Whether to mask the target template sequence.",
    )
    rm_target_sc: bool = ConfigField(
        default=False,
        title="Mask Target Side Chains",
        description="Whether to mask target template side chains.",
    )
    rm_template_ic: bool = ConfigField(
        default=True,
        title="Mask Template Contacts",
        description="Whether to mask inter-chain template contacts.",
    )
    include_pae_matrix: bool = ConfigField(
        default=False,
        title="Include PAE",
        description="Whether to return the full PAE matrix.",
    )
    backend: Literal["base", "germinal"] = ConfigField(
        default="base",
        title="Backend",
        description="AF2 backend: 'base' (default) or 'germinal' (Germinal VHH extensions).",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for AF2 execution.",
    )
    seed: int | None = ConfigField(
        default=None,
        title="Seed",
        description="Base seed for deterministic AF2 evaluations.",
        ge=0,
    )
    _evaluation_seed_offset: int = PrivateAttr(default=0)

    @field_validator("target_chains", mode="before")
    @classmethod
    def _normalize_target_chains(cls, value: Any) -> list[str]:
        """Accept comma-separated chain IDs or explicit lists."""
        raw_chain_ids = [value] if isinstance(value, str) else value
        if not isinstance(raw_chain_ids, (list, tuple)) or not all(isinstance(c, str) for c in raw_chain_ids):
            raise ValueError("target_chains must be a string or list of strings.")
        chains = [chain.strip() for raw in raw_chain_ids for chain in raw.split(",") if chain.strip()]
        if not chains:
            raise ValueError("target_chains must contain at least one chain ID.")
        return chains

    @field_validator("target_input_indices", mode="before")
    @classmethod
    def _normalize_target_input_indices(cls, value: Any) -> list[int]:
        """Accept comma-separated indices or explicit integer lists."""
        if isinstance(value, str):
            value = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        elif isinstance(value, int):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("target_input_indices must be an int or list of ints.")
        try:
            indices = [int(v) for v in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("target_input_indices must contain integers.") from exc
        if not indices:
            raise ValueError("target_input_indices must contain at least one index.")
        if any(i < 0 for i in indices):
            raise ValueError("target_input_indices must be non-negative.")
        return indices

    @model_validator(mode="after")
    def _validate_roles(self) -> "AlphaFold2MultimerStructureConfig":
        """Validate static role configuration."""
        if len(set(self.target_input_indices)) != len(self.target_input_indices):
            raise ValueError("target_input_indices cannot contain duplicates.")
        if self.binder_input_index in self.target_input_indices:
            raise ValueError("binder_input_index cannot also be a target input.")
        if len(self.target_chains) != len(self.target_input_indices):
            raise ValueError("target_chains must match target_input_indices one-to-one.")
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
    def germinal_vhh_preset(
        cls, target_pdb: str, binder_chain: str = "H", target_chains: list[str] | str = "A"
    ) -> "AlphaFold2MultimerStructureConfig":
        """Create the Germinal VHH AF2 multimer config.

        Args:
            target_pdb (str): Target+binder template PDB content or path.
            binder_chain (str): Binder chain ID in the template PDB.
            target_chains (list[str] | str): Target chain ID list or comma string.
        """
        chains = cls._normalize_target_chains(target_chains)
        return cls(
            target_pdb=target_pdb,
            binder_chain=binder_chain,
            target_chains=chains,
            target_input_indices=list(range(1, 1 + len(chains))),
            omit_aas=["C"],
            bias_redesign=10.0,
            inter_contact_num=10,
            inter_contact_cutoff=20.0,
            framework_contact_offset=1.0,
            sample_models=True,
            backend="germinal",
        )


class StructureBasedConstraintConfig(BaseConfig):
    """Base configuration for constraints using structure prediction tools.

    This base class standardizes how structure prediction tools and their
    configurations are specified across all structure-based constraints.
    Each tool has its own dedicated config field.

    Subclasses can optionally restrict which tools are supported by overriding
    the structure_tool field with a narrower Literal type.

    Attributes:
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2_multimer']): Tool to use for structure prediction. Supported options:
            - "esmfold": ESMFold (Meta AI)
            - "esmfold2": ESMFold2 (Biohub) — all-atom, complex-capable
            - "alphafold3": AlphaFold 3 (Google DeepMind)
            - "boltz2": Boltz2 (MIT)
            - "chai1": Chai-1 (Chai Discovery)
            - "protenix": Protenix (ByteDance)
            - "alphafold2_multimer": AlphaFold2 multimer-design backend
            Default is "esmfold".

        esmfold_config (ESMFoldConfig): Configuration for ESMFold structure prediction.
            Used when ``structure_tool == "esmfold"``.

        esmfold2_config (ESMFold2Config): Configuration for ESMFold2 structure prediction.
            Used when ``structure_tool == "esmfold2"``.

        alphafold3_config (AlphaFold3Config): Configuration for AlphaFold3 structure prediction.
            Used when ``structure_tool == "alphafold3"``.

        boltz2_config (Boltz2Config): Configuration for Boltz2 structure prediction.
            Used when ``structure_tool == "boltz2"``.

        chai1_config (Chai1Config): Configuration for Chai1 structure prediction.
            Used when ``structure_tool == "chai1"``.

        protenix_config (ProtenixConfig): Configuration for Protenix structure prediction.
            Used when ``structure_tool == "protenix"``.

        alphafold2_multimer_config (AlphaFold2MultimerStructureConfig): Configuration
            for AF2 multimer-backed structure constraints.

    Example:
        >>> config = MyConstraintConfig(structure_tool="esmfold", esmfold_config=ESMFoldConfig(device="cuda"))
        >>>
        >>> config = MyConstraintConfig(structure_tool="alphafold3", alphafold3_config={"seeds": [0, 1]})
    """

    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2_multimer"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="esmfold",
        description=(
            "Structure predictor: esmfold, esmfold2, alphafold3, boltz2, chai1, protenix, or alphafold2_multimer."
        ),
    )

    esmfold_config: ESMFoldConfig = ConfigField(
        default_factory=ESMFoldConfig,
        title="ESMFold Configuration",
        description="Configuration for ESMFold structure prediction.",
    )
    esmfold2_config: ESMFold2Config = ConfigField(
        default_factory=ESMFold2Config,
        title="ESMFold2 Configuration",
        description="Configuration for ESMFold2 structure prediction.",
    )
    alphafold3_config: AlphaFold3Config = ConfigField(
        default_factory=AlphaFold3Config,
        title="AlphaFold3 Configuration",
        description="Configuration for AlphaFold3 structure prediction.",
    )
    boltz2_config: Boltz2Config = ConfigField(
        default_factory=Boltz2Config,
        title="Boltz2 Configuration",
        description="Configuration for Boltz2 structure prediction.",
    )
    chai1_config: Chai1Config = ConfigField(
        default_factory=Chai1Config,
        title="Chai1 Configuration",
        description="Configuration for Chai1 structure prediction.",
    )
    protenix_config: ProtenixConfig = ConfigField(
        default_factory=ProtenixConfig,
        title="Protenix Configuration",
        description="Configuration for Protenix structure prediction.",
    )
    alphafold2_multimer_config: AlphaFold2MultimerStructureConfig = ConfigField(
        default_factory=AlphaFold2MultimerStructureConfig,
        title="AF2 Multimer Config",
        description="Configuration for AF2 multimer constraints.",
    )

    @property
    def tool_config(
        self,
    ) -> (
        ESMFoldConfig
        | ESMFold2Config
        | AlphaFold3Config
        | Boltz2Config
        | Chai1Config
        | ProtenixConfig
        | AlphaFold2MultimerStructureConfig
    ):
        """Return the active tool configuration based on structure_tool."""
        configs = {
            "esmfold": self.esmfold_config,
            "esmfold2": self.esmfold2_config,
            "alphafold3": self.alphafold3_config,
            "boltz2": self.boltz2_config,
            "chai1": self.chai1_config,
            "protenix": self.protenix_config,
            "alphafold2_multimer": self.alphafold2_multimer_config,
        }
        return configs[self.structure_tool]

    @model_validator(mode="after")
    def _validate_active_tool(self) -> "StructureBasedConstraintConfig":
        """Validate selected tool config only when the tool is active."""
        if self.structure_tool == "alphafold2_multimer" and not self.alphafold2_multimer_config.target_pdb:
            raise ValueError("alphafold2_multimer_config.target_pdb must be a non-empty PDB string or file path.")
        return self
