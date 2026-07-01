"""MPNN-guided mutation generator for structure-conditioned sequence refinement."""

from __future__ import annotations

from typing import Any, Literal, final

import numpy as np
from proto_tools import (
    InverseFoldingStructureInput,
    LigandMPNNScoringConfig,
    LigandMPNNScoringInput,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    SequenceStructurePair,
    Structure,
    run_ligandmpnn_score,
    run_proteinmpnn_score,
)
from proto_tools.entities.structures import ResidueSelection
from pydantic import field_validator

from proto_language.core import Generator, GeneratorInputType
from proto_language.core.sequence import PROTEIN_AMINO_ACIDS
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField
from proto_language.utils.serialization import make_json_safe

MPNNMutationModel = Literal["ligandmpnn", "proteinmpnn"]
ReplacementStrategy = Literal["sample", "argmax"]
ProteinMPNNModelChoice = Literal["proteinmpnn", "v_48_002", "v_48_010", "v_48_030", "abmpnn", "soluble"]


class MPNNMutationGeneratorConfig(BaseConfig):
    """Configuration for structure-conditioned MPNN mutation.

    The generator scores the current sequence against a backbone, chooses
    mutable positions using the model's probability of the current residue,
    then replaces each chosen residue from the model's per-position amino-acid
    distribution.

    Attributes:
        model (MPNNMutationModel): MPNN scorer used to compute mutation probabilities.
        structure_inputs (list[InverseFoldingStructureInput] | None): Structures for scoring proposals.
        output_chain_id (str | None): Chain corresponding to the assigned segment.
        num_mutations (int): Number of positions to mutate per sequence.
        mutable_positions (ResidueSelection | None): Optional 1-indexed residue positions allowed to mutate.
        excluded_amino_acids (list[str] | None): One-letter amino acids forbidden as replacements.
        replacement_strategy (ReplacementStrategy): Sampling strategy for replacement residues.
        replacement_temperature (float): Temperature applied to MPNN logits before sampling.
        proteinmpnn_model_choice (ProteinMPNNModelChoice): ProteinMPNN checkpoint when model is proteinmpnn.
        ligand_mpnn_use_side_chain_context (bool): Whether LigandMPNN uses fixed-residue sidechain context.
        ligand_mpnn_cutoff_for_score (float): Ligand-residue cutoff for LigandMPNN scoring.
        device (str): Device used for MPNN scoring.
        verbose (bool): Whether to emit MPNN scoring logs.
    """

    model: MPNNMutationModel = ConfigField(
        default="ligandmpnn",
        title="MPNN Model",
        description="Structure-conditioned model used for mutation probabilities: ligandmpnn or proteinmpnn.",
    )
    structure_inputs: list[InverseFoldingStructureInput] | None = ConfigField(
        default=None,
        title="Structure Inputs",
        description="Structure(s) for MPNN scoring, with optional chains_to_redesign and fixed_positions.",
    )
    output_chain_id: str | None = ConfigField(
        default=None,
        title="Output Chain",
        description="Structure chain corresponding to the assigned sequence. Required for ambiguous multi-chain inputs.",
    )
    num_mutations: int = ConfigField(
        default=1,
        ge=1,
        title="Number of Mutations",
        description="Number of positions to resample per sequence.",
    )
    mutable_positions: ResidueSelection | None = ConfigField(
        default=None,
        title="Mutable Positions",
        description="Optional per-chain 1-indexed positions eligible for mutation. If unset, the output chain is mutable.",
    )
    excluded_amino_acids: list[str] | None = ConfigField(
        default=None,
        title="Excluded Amino Acids",
        description="Single-letter amino acids to forbid as replacement residues.",
    )
    replacement_strategy: ReplacementStrategy = ConfigField(
        default="sample",
        title="Replacement Strategy",
        description="'sample' draws from MPNN probabilities; 'argmax' chooses the highest-probability residue.",
    )
    replacement_temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Replacement Temperature",
        description="Temperature applied to MPNN logits before replacement sampling.",
    )
    proteinmpnn_model_choice: ProteinMPNNModelChoice = ConfigField(
        default="proteinmpnn",
        title="ProteinMPNN Weights",
        description="ProteinMPNN weights used when model='proteinmpnn'.",
    )
    ligand_mpnn_use_side_chain_context: bool = ConfigField(
        default=False,
        title="LigandMPNN Sidechain Context",
        description="Whether LigandMPNN scoring conditions on fixed-residue sidechain atoms.",
    )
    ligand_mpnn_cutoff_for_score: float = ConfigField(
        default=8.0,
        gt=0.0,
        title="LigandMPNN Ligand Cutoff",
        description="Ligand-residue distance cutoff (Å) used by LigandMPNN scoring.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device for MPNN scoring.",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print status messages during MPNN scoring.",
    )

    @field_validator("structure_inputs", mode="before")
    @classmethod
    def normalize_structure_inputs(cls, v: Any) -> Any:
        """Convert flexible structure input syntax to InverseFoldingStructureInput objects."""
        if v is None:
            return None
        if not isinstance(v, list):
            v = [v]

        result = []
        for item in v:
            if isinstance(item, InverseFoldingStructureInput):
                result.append(item)
            elif isinstance(item, (str, Structure)):
                result.append(InverseFoldingStructureInput(structure=item))
            elif isinstance(item, dict):
                result.append(InverseFoldingStructureInput(**item))
            else:
                raise ValueError(f"Unsupported structure_inputs item type: {type(item)}")
        return result

    @field_validator("excluded_amino_acids", mode="before")
    @classmethod
    def normalize_excluded_amino_acids(cls, v: Any) -> Any:
        """Normalize excluded residues to uppercase one-letter amino-acid codes."""
        if v is None:
            return None
        excluded = [str(aa).upper() for aa in v]
        invalid = sorted(set(excluded) - set(PROTEIN_AMINO_ACIDS))
        if invalid:
            raise ValueError(f"excluded_amino_acids contains invalid protein residues: {invalid}")
        return excluded


@generator(
    key="mpnn-mutation",
    label="MPNN Structure-Conditioned Mutation",
    config=MPNNMutationGeneratorConfig,
    description="LigandMPNN/ProteinMPNN-guided mutation of an existing protein sequence",
    uses_gpu=True,
    tools_called=["ligandmpnn-score", "proteinmpnn-score"],
    supported_sequence_types=["protein"],
)
@final
class MPNNMutationGenerator(Generator):
    """Mutate protein sequences using MPNN structure-conditioned probabilities."""

    input_type = GeneratorInputType.STARTING_SEQUENCE

    def __init__(self, config: MPNNMutationGeneratorConfig) -> None:
        """Initialize the MPNN mutation generator from its configuration."""
        super().__init__()
        self.config = config
        self.model = config.model
        self.structure_inputs = config.structure_inputs
        self.output_chain_id = config.output_chain_id
        self.num_mutations = config.num_mutations
        self.mutable_positions = config.mutable_positions
        self.excluded_amino_acids = set(config.excluded_amino_acids or [])
        self.replacement_strategy = config.replacement_strategy
        self.replacement_temperature = config.replacement_temperature
        self.proteinmpnn_model_choice = config.proteinmpnn_model_choice
        self.ligand_mpnn_use_side_chain_context = config.ligand_mpnn_use_side_chain_context
        self.ligand_mpnn_cutoff_for_score = config.ligand_mpnn_cutoff_for_score
        self.device = config.device
        self.verbose = config.verbose

    def _sample(self, structure_inputs: list[InverseFoldingStructureInput] | None = None) -> None:
        """Mutate proposal sequences in-place using MPNN score logits."""
        self._validate_generator()
        sampling_structure_inputs = (
            MPNNMutationGeneratorConfig.normalize_structure_inputs(structure_inputs)
            if structure_inputs is not None
            else self.structure_inputs
        )
        if sampling_structure_inputs is None:
            raise ValueError("No structure_inputs provided. Configure structure_inputs or pass them to sample().")

        proposals = self.segment.proposal_sequences
        if len(sampling_structure_inputs) == 1:
            paired_inputs = sampling_structure_inputs * len(proposals)
        elif len(sampling_structure_inputs) == len(proposals):
            paired_inputs = sampling_structure_inputs
        else:
            raise ValueError(
                f"Number of structure_inputs ({len(sampling_structure_inputs)}) must either be 1 "
                f"or match num_proposals ({len(proposals)})."
            )

        key = self._spec.key
        for proposal, struct_input in zip(proposals, paired_inputs, strict=True):
            seed = self._next_seed()
            rng = np.random.default_rng(seed)
            output_chain_id = self._resolve_output_chain(struct_input)
            full_sequence, chain_offset = self._build_scoring_sequence(
                struct_input=struct_input,
                output_chain_id=output_chain_id,
                output_sequence=proposal.sequence,
            )
            fixed_positions = self._scoring_fixed_positions(struct_input, output_chain_id)
            logits, vocab, score_metadata = self._score(full_sequence, struct_input, fixed_positions, seed)
            chain_logits = self._slice_chain_logits(
                logits, output_chain_id, struct_input, chain_offset, proposal.sequence
            )
            probabilities = self._softmax(chain_logits, self.replacement_temperature)
            vocab_index = {aa: idx for idx, aa in enumerate(vocab)}
            allowed_indices = self._allowed_vocab_indices(vocab)
            candidate_positions = self._candidate_positions(
                proposal.sequence, output_chain_id, struct_input, vocab_index
            )
            selected_positions = self._select_positions(
                sequence=proposal.sequence,
                probabilities=probabilities,
                vocab_index=vocab_index,
                candidate_positions=candidate_positions,
                rng=rng,
            )

            sequence_chars = list(proposal.sequence)
            mutations: list[dict[str, Any]] = []
            for idx in selected_positions:
                old = sequence_chars[idx]
                new = self._select_replacement(probabilities[idx], vocab, allowed_indices, rng)
                sequence_chars[idx] = new
                mutations.append({"position": idx + 1, "from": old, "to": new})

            proposal.sequence = "".join(sequence_chars)
            proposal._generator_metadata[key] = {
                "model": self.model,
                "output_chain_id": output_chain_id,
                "num_mutations": self.num_mutations,
                "selected_positions": [idx + 1 for idx in selected_positions],
                "mutations": mutations,
                "replacement_strategy": self.replacement_strategy,
                "score": make_json_safe(score_metadata),
            }

    def _resolve_output_chain(self, struct_input: InverseFoldingStructureInput) -> str:
        chain_ids = struct_input.structure.get_chain_ids()
        redesign_ids = struct_input.chain_ids_to_redesign

        if self.output_chain_id is not None:
            output_chain_id = self.output_chain_id
        elif len(redesign_ids) == 1:
            output_chain_id = redesign_ids[0]
        elif len(chain_ids) == 1:
            output_chain_id = chain_ids[0]
        else:
            raise ValueError(
                "output_chain_id is required when multiple chains are available for redesign "
                f"(chains_to_redesign={redesign_ids})."
            )

        if output_chain_id not in chain_ids:
            raise ValueError(f"output_chain_id {output_chain_id!r} not found in structure chains {chain_ids}.")
        if output_chain_id not in redesign_ids:
            raise ValueError(
                f"output_chain_id {output_chain_id!r} must be included in chains_to_redesign {redesign_ids}."
            )
        return output_chain_id

    def _build_scoring_sequence(
        self,
        *,
        struct_input: InverseFoldingStructureInput,
        output_chain_id: str,
        output_sequence: str,
    ) -> tuple[str, int]:
        chain_sequences = {
            chain_id: struct_input.structure.get_chain_sequence(chain_id)
            for chain_id in struct_input.structure.get_chain_ids()
        }
        expected_length = len(chain_sequences[output_chain_id])
        if len(output_sequence) != expected_length:
            raise ValueError(
                f"Sequence length {len(output_sequence)} does not match structure chain "
                f"{output_chain_id!r} length {expected_length}."
            )

        full_parts: list[str] = []
        chain_offset = 0
        for chain_id in struct_input.structure.get_chain_ids():
            if chain_id == output_chain_id:
                chain_offset = sum(len(part) for part in full_parts)
                full_parts.append(output_sequence)
            else:
                full_parts.append(chain_sequences[chain_id])
        return "".join(full_parts), chain_offset

    def _scoring_fixed_positions(
        self, struct_input: InverseFoldingStructureInput, output_chain_id: str
    ) -> ResidueSelection | None:
        fixed: dict[str, set[int]] = {}
        for chain_id in struct_input.structure.get_chain_ids():
            chain_positions = set(struct_input.structure.get_chain_positions(chain_id))
            if chain_id != output_chain_id:
                fixed[chain_id] = set(chain_positions)

        if struct_input.fixed_positions is not None:
            for chain_id, positions in struct_input.fixed_positions.chains.items():
                fixed.setdefault(chain_id, set()).update(positions)

        if self.mutable_positions is not None:
            self.mutable_positions.validate_against(struct_input.structure, label="mutable_positions")
            mutable = set(self.mutable_positions.chains.get(output_chain_id, []))
            if not mutable:
                raise ValueError(
                    f"mutable_positions does not include any positions for output chain {output_chain_id!r}."
                )
            output_positions = set(struct_input.structure.get_chain_positions(output_chain_id))
            fixed.setdefault(output_chain_id, set()).update(output_positions - mutable)

        fixed_dict = {chain_id: sorted(positions) for chain_id, positions in fixed.items() if positions}
        return ResidueSelection(chains=fixed_dict) if fixed_dict else None

    def _score(
        self,
        full_sequence: str,
        struct_input: InverseFoldingStructureInput,
        fixed_positions: ResidueSelection | None,
        seed: int | None,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        pair = SequenceStructurePair(
            sequence=full_sequence,
            structure=self._structure_for_scoring(struct_input.structure),
            fixed_positions=fixed_positions,
        )
        if self.model == "ligandmpnn":
            scoring_config: dict[str, Any] = {
                "return_logits": True,
                "scoring_mode": "single_aa",
                "seed": seed,
                "device": self.device,
                "verbose": self.verbose,
            }
            supported_fields = getattr(LigandMPNNScoringConfig, "model_fields", {})
            if "ligand_mpnn_use_side_chain_context" in supported_fields:
                scoring_config["ligand_mpnn_use_side_chain_context"] = self.ligand_mpnn_use_side_chain_context
            if "ligand_mpnn_cutoff_for_score" in supported_fields:
                scoring_config["ligand_mpnn_cutoff_for_score"] = self.ligand_mpnn_cutoff_for_score
            result = run_ligandmpnn_score(
                inputs=LigandMPNNScoringInput(sequence_structure_pairs=[pair]),
                config=LigandMPNNScoringConfig(**scoring_config),
            )
        else:
            result = run_proteinmpnn_score(
                inputs=ProteinMPNNScoringInput(sequence_structure_pairs=[pair]),
                config=ProteinMPNNScoringConfig(
                    return_logits=True,
                    model_choice=self.proteinmpnn_model_choice,
                    seed=seed,
                    device=self.device,
                    verbose=self.verbose,
                ),
            )

        score = result.scores[0]
        logits = getattr(score, "logits", None)
        vocab = getattr(score, "vocab", None)
        if logits is None or vocab is None:
            raise ValueError(f"{self.model} scoring did not return logits and vocab.")
        metadata = score.model_dump(exclude={"logits", "vocab"}) if hasattr(score, "model_dump") else {}
        return np.asarray(logits, dtype=np.float64), list(vocab), metadata

    def _structure_for_scoring(self, structure: Structure) -> Structure:
        """Normalize PDB records that inverse-folding scorers commonly drop.

        Some scaffolds contain alternate conformers with partial occupancy.
        LigandMPNN sampling handles the full chain, but the scoring path can
        drop those residues unless a single conformer is selected and
        occupancies are normalized.
        """
        if structure.structure_format not in (None, "pdb"):
            return structure

        sanitized = self._sanitize_pdb_for_scoring(structure.structure_pdb, set(structure.get_chain_ids()))
        if sanitized == structure.structure_pdb:
            return structure
        return Structure(
            structure=sanitized,
            structure_format="pdb",
            b_factor_type=structure.b_factor_type,
            source=structure.source,
            metrics=structure.metrics,
        )

    @staticmethod
    def _sanitize_pdb_for_scoring(pdb: str, polymer_chain_ids: set[str]) -> str:
        selected_altlocs: dict[tuple[str, str, str], str] = {}
        lines = pdb.splitlines()
        for line in lines:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
                continue
            altloc = line[16]
            if altloc == " ":
                continue
            key = (line[:6], line[21], line[22:27])
            if altloc == "A" or key not in selected_altlocs:
                selected_altlocs[key] = altloc

        ligand_chain_id = MPNNMutationGenerator._unused_pdb_chain_id(polymer_chain_ids)
        sanitized_lines = []
        changed = False
        for line in lines:
            sanitized_line = line
            if sanitized_line.startswith(("ATOM  ", "HETATM")) and len(sanitized_line) >= 60:
                altloc = sanitized_line[16]
                if altloc != " ":
                    key = (sanitized_line[:6], sanitized_line[21], sanitized_line[22:27])
                    if selected_altlocs.get(key) != altloc:
                        changed = True
                        continue
                    sanitized_line = f"{sanitized_line[:16]} {sanitized_line[17:]}"
                    changed = True
                if sanitized_line[54:60] != "  1.00":
                    sanitized_line = f"{sanitized_line[:54]}  1.00{sanitized_line[60:]}"
                    changed = True
                if (
                    sanitized_line.startswith("HETATM")
                    and sanitized_line[21] in polymer_chain_ids
                    and ligand_chain_id is not None
                ):
                    sanitized_line = f"{sanitized_line[:21]}{ligand_chain_id}{sanitized_line[22:]}"
                    changed = True
            sanitized_lines.append(sanitized_line)

        if not changed:
            return pdb
        return "\n".join(sanitized_lines) + ("\n" if pdb.endswith("\n") else "")

    @staticmethod
    def _unused_pdb_chain_id(used_chain_ids: set[str]) -> str | None:
        for chain_id in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
            if chain_id not in used_chain_ids:
                return chain_id
        return None

    def _slice_chain_logits(
        self,
        logits: np.ndarray,
        output_chain_id: str,
        struct_input: InverseFoldingStructureInput,
        chain_offset: int,
        output_sequence: str,
    ) -> np.ndarray:
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 2:
            raise ValueError(f"Expected MPNN logits with shape (L, vocab), got {logits.shape}.")
        total_length = sum(
            len(struct_input.structure.get_chain_sequence(chain_id))
            for chain_id in struct_input.structure.get_chain_ids()
        )
        if logits.shape[0] != total_length:
            raise ValueError(
                f"MPNN logits length {logits.shape[0]} does not match structure sequence length {total_length}."
            )
        chain_logits = logits[chain_offset : chain_offset + len(output_sequence)]
        if chain_logits.shape[0] != len(output_sequence):
            raise ValueError(f"Could not slice logits for output chain {output_chain_id!r}.")
        return chain_logits

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        scaled = logits / temperature
        scaled = scaled - np.nanmax(scaled, axis=1, keepdims=True)
        exp = np.exp(scaled)
        totals = np.sum(exp, axis=1, keepdims=True)
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
            raise ValueError("MPNN logits could not be normalized into probabilities.")
        return np.asarray(exp / totals, dtype=np.float64)

    def _allowed_vocab_indices(self, vocab: list[str]) -> list[int]:
        segment_vocab = set(self.segment.ordered_vocab())
        allowed = [
            idx
            for idx, aa in enumerate(vocab)
            if aa in segment_vocab and aa in PROTEIN_AMINO_ACIDS and aa not in self.excluded_amino_acids
        ]
        if not allowed:
            raise ValueError("No valid amino-acid replacements remain after applying exclusions.")
        return allowed

    def _candidate_positions(
        self,
        sequence: str,
        output_chain_id: str,
        struct_input: InverseFoldingStructureInput,
        vocab_index: dict[str, int],
    ) -> list[int]:
        candidates = [
            idx
            for idx in self._mutable_sequence_indices(output_chain_id, struct_input)
            if sequence[idx] in vocab_index and sequence[idx] not in self.excluded_amino_acids
        ]

        if len(candidates) < self.num_mutations:
            raise ValueError(
                f"MPNN mutation requested {self.num_mutations} mutations but only {len(candidates)} mutable "
                f"positions are available on chain {output_chain_id!r}."
            )
        return candidates

    def _mutable_sequence_indices(
        self,
        output_chain_id: str,
        struct_input: InverseFoldingStructureInput,
    ) -> list[int]:
        if self.mutable_positions is not None:
            positions = set(self.mutable_positions.chains.get(output_chain_id, []))
        else:
            positions = set(struct_input.structure.get_chain_positions(output_chain_id))

        if struct_input.fixed_positions is not None:
            positions -= set(struct_input.fixed_positions.chains.get(output_chain_id, []))

        position_to_index = self._chain_position_to_index(output_chain_id, struct_input)
        indices = []
        for position in sorted(positions):
            if position not in position_to_index:
                raise ValueError(f"Position {output_chain_id}{position} is not present in the structure.")
            indices.append(position_to_index[position])
        return indices

    @staticmethod
    def _chain_position_to_index(
        output_chain_id: str,
        struct_input: InverseFoldingStructureInput,
    ) -> dict[int, int]:
        return {
            position: idx for idx, position in enumerate(struct_input.structure.get_chain_positions(output_chain_id))
        }

    def _select_positions(
        self,
        *,
        sequence: str,
        probabilities: np.ndarray,
        vocab_index: dict[str, int],
        candidate_positions: list[int],
        rng: np.random.Generator,
    ) -> list[int]:
        weights = np.array(
            [probabilities[idx, vocab_index[sequence[idx]]] for idx in candidate_positions],
            dtype=np.float64,
        )
        weights = self._normalize_weights(weights, "MPNN current-residue probabilities")
        selected = rng.choice(
            np.asarray(candidate_positions, dtype=int),
            size=self.num_mutations,
            replace=False,
            p=weights,
        )
        return [int(idx) for idx in selected.tolist()]

    def _select_replacement(
        self,
        probabilities: np.ndarray,
        vocab: list[str],
        allowed_indices: list[int],
        rng: np.random.Generator,
    ) -> str:
        allowed_probs = probabilities[allowed_indices]
        if self.replacement_strategy == "argmax":
            return vocab[allowed_indices[int(np.argmax(allowed_probs))]]

        weights = self._normalize_weights(allowed_probs, "MPNN replacement probabilities")
        choice = rng.choice(np.asarray(allowed_indices, dtype=int), p=weights)
        return vocab[int(choice)]

    @staticmethod
    def _normalize_weights(weights: np.ndarray, label: str) -> np.ndarray:
        cleaned = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        total = float(cleaned.sum())
        if total <= 0.0:
            raise ValueError(f"{label} sum to zero; cannot sample.")
        return cleaned / total
