"""FAMPNN structure-conditioned sequence and sidechain generation."""

from __future__ import annotations

from typing import Any, final

from proto_tools import (
    FAMPNNDesign,
    FAMPNNSampleConfig,
    FAMPNNSampleInput,
    FAMPNNStructureInput,
    Structure,
    run_fampnn_sample,
)
from pydantic import field_validator

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField


class FAMPNNGeneratorConfig(BaseConfig):
    """Configuration for FAMPNN inverse-folding generation.

    FAMPNN jointly designs amino-acid identities and sidechain conformations
    conditioned on an input backbone. The generator writes the designed protein
    sequence to the assigned segment and attaches FAMPNN's full-atom output
    structure to ``proposal.structure``.

    Attributes:
        structure_inputs (list[FAMPNNStructureInput] | None): Structures and redesign selections.
        output_chain_id (str | None): Chain whose sequence is written to the segment.
        model_variant (str): FAMPNN checkpoint variant used for design.
        temperature (float): Sampling temperature for amino-acid design.
        batch_size (int): Number of sequences processed together on GPU.
        num_steps (int): Number of iterative unmasking steps.
        seq_only (bool): Whether to skip sidechain generation.
        repack_last (bool): Whether to repack sidechains after sequence design.
        psce_threshold (float): Predicted-error cutoff for sidechain conditioning.
        scn_diffusion_steps (int): Number of sidechain denoising steps.
        scn_step_scale (float): Step scale for sidechain diffusion.
        device (str): Device used for model inference.
        verbose (bool): Whether to emit FAMPNN progress logs.
    """

    structure_inputs: list[FAMPNNStructureInput] | None = ConfigField(
        default=None,
        title="Structure Inputs",
        description="Structure(s) with optional chain, fixed-position, and fixed-sidechain selections.",
    )
    output_chain_id: str | None = ConfigField(
        default=None,
        title="Output Chain",
        description="When sampling a multi-chain structure, write only this chain's sequence to the target segment.",
    )
    model_variant: str = ConfigField(
        default="0.3",
        title="Model Variant",
        description="FAMPNN checkpoint variant for sequence design.",
    )
    temperature: float = ConfigField(
        default=0.1,
        ge=0.0,
        title="Temperature",
        description="Sampling temperature; lower is greedier and higher is more diverse.",
    )
    batch_size: int = ConfigField(
        default=1,
        ge=1,
        title="Batch Size",
        description="Number of sequences to process simultaneously on GPU.",
    )
    num_steps: int = ConfigField(
        default=100,
        ge=1,
        title="Unmasking Steps",
        description="Number of iterative unmasking steps for sequence design.",
    )
    seq_only: bool = ConfigField(
        default=False,
        title="Sequence Only",
        description="If true, skip sidechain generation during sampling.",
    )
    repack_last: bool = ConfigField(
        default=True,
        title="Repack Last",
        description="Repack sidechains after the final sequence is determined.",
    )
    psce_threshold: float = ConfigField(
        default=0.3,
        ge=0.0,
        title="pSCE Threshold",
        description="Only condition on sidechains below this predicted-error threshold during design.",
    )
    scn_diffusion_steps: int = ConfigField(
        default=50,
        ge=1,
        title="Sidechain Diffusion Steps",
        description="Number of sidechain diffusion denoising steps.",
    )
    scn_step_scale: float = ConfigField(
        default=1.5,
        gt=0.0,
        title="Sidechain Step Scale",
        description="Step scale for sidechain diffusion.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for model inference.",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print FAMPNN progress logs.",
    )

    @field_validator("structure_inputs", mode="before")
    @classmethod
    def normalize_structure_inputs(cls, value: Any) -> Any:
        """Convert shorthand structure inputs to ``FAMPNNStructureInput`` objects."""
        if value is None:
            return None
        if not isinstance(value, list):
            value = [value]
        normalized = []
        for item in value:
            if isinstance(item, FAMPNNStructureInput):
                normalized.append(item)
            elif isinstance(item, (str, Structure)):
                normalized.append(FAMPNNStructureInput(structure=item))
            elif isinstance(item, dict):
                normalized.append(FAMPNNStructureInput(**item))
            else:
                raise ValueError(f"Unsupported structure_inputs item type: {type(item)}")
        return normalized


@generator(
    key="fampnn",
    label="FAMPNN Inverse Folding",
    config=FAMPNNGeneratorConfig,
    description="FAMPNN structure-conditioned protein sequence design with full-atom sidechain co-generation.",
    uses_gpu=True,
    tools_called=["fampnn-sample"],
    supported_sequence_types=["protein"],
)
@final
class FAMPNNGenerator(Generator):
    """Protein sequence generator using FAMPNN."""

    input_type = GeneratorInputType.STRUCTURE

    def __init__(self, config: FAMPNNGeneratorConfig) -> None:
        """Initialize the FAMPNN generator with a validated config."""
        super().__init__()
        self.config = config
        self.structure_inputs = config.structure_inputs
        self.output_chain_id = config.output_chain_id
        self.batch_size = config.batch_size

    def _sample(self, structure_inputs: list[FAMPNNStructureInput] | None = None) -> None:
        self._validate_generator()
        num_proposals = self.segment.num_proposals
        sampling_structure_inputs = (
            FAMPNNGeneratorConfig.normalize_structure_inputs(structure_inputs)
            if structure_inputs is not None
            else self.structure_inputs
        )
        if sampling_structure_inputs is None:
            raise ValueError("FAMPNNGenerator requires structure_inputs in config or at sample() time.")

        if len(sampling_structure_inputs) == 1:
            num_sequences_per_structure = num_proposals
            batch_size = self.batch_size
        else:
            if len(sampling_structure_inputs) != num_proposals:
                raise ValueError(
                    f"Number of structure_inputs ({len(sampling_structure_inputs)}) must be 1 or match "
                    f"num_proposals ({num_proposals})."
                )
            num_sequences_per_structure = 1
            batch_size = 1

        tool_config = FAMPNNSampleConfig(
            model_variant=self.config.model_variant,
            num_sequences_per_structure=num_sequences_per_structure,
            batch_size=batch_size,
            temperature=self.config.temperature,
            seed=self._next_seed(),
            num_steps=self.config.num_steps,
            seq_only=self.config.seq_only,
            repack_last=self.config.repack_last,
            psce_threshold=self.config.psce_threshold,
            scn_diffusion_steps=self.config.scn_diffusion_steps,
            scn_step_scale=self.config.scn_step_scale,
            device=self.config.device,
            verbose=self.config.verbose,
        )
        result = run_fampnn_sample(
            inputs=FAMPNNSampleInput(inputs=sampling_structure_inputs),
            config=tool_config,
        )

        generated_sequences: list[str] = []
        full_sequences: list[str] = []
        structures: list[Structure] = []
        avg_psce_values: list[float] = []
        for design_set, struct_input in zip(result.design_sets, sampling_structure_inputs, strict=True):
            for design in design_set.complexes:
                designed_sequences = [
                    str(chain.sequence)
                    for chain, was_designed in zip(design.chains, design.designed, strict=True)
                    if was_designed
                ]
                full_sequences.append("/".join(designed_sequences))
                generated_sequences.append(self._select_output_sequence(design, struct_input))
                structures.append(design.structure)
                avg_psce_values.append(float(design.metrics.avg_psce))

        key = self._spec.key
        for proposal, sequence, full_sequence, structure, avg_psce in zip(
            self.segment.proposal_sequences,
            generated_sequences,
            full_sequences,
            structures,
            avg_psce_values,
            strict=True,
        ):
            proposal.sequence = sequence
            proposal.structure = structure
            proposal._generator_metadata[key] = {
                "avg_psce": avg_psce,
                "full_sequence": full_sequence,
            }

    def _select_output_sequence(self, design: FAMPNNDesign, struct_input: FAMPNNStructureInput) -> str:
        designed_chains = [
            chain for chain, was_designed in zip(design.chains, design.designed, strict=True) if was_designed
        ]
        if self.output_chain_id is None:
            return "/".join(str(chain.sequence) for chain in designed_chains)
        if self.output_chain_id not in struct_input.chain_ids_to_redesign:
            raise ValueError(
                f"output_chain_id {self.output_chain_id!r} not found in chain_ids_to_redesign "
                f"{struct_input.chain_ids_to_redesign}"
            )
        for chain in designed_chains:
            if chain.id == self.output_chain_id:
                return str(chain.sequence)
        raise ValueError(f"FAMPNN did not return designed chain {self.output_chain_id!r}.")
