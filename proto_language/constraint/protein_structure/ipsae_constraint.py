"""Protein-DNA ipSAE interface constraint.

Scores protein-DNA interface confidence with ipSAE (Dunbrack 2025), a PAE-derived
interface-quality metric that is more selective than overall ipTM: it restricts to
residue pairs whose predicted aligned error is below a cutoff and rewards a tight,
low-error binder-target interface rather than an inflated whole-complex score.

The constraint folds each candidate's protein chain(s) (duplicated to
``num_protein_copies``) plus its DNA chain(s) (optionally appending the reverse
complement when a single DNA strand is supplied) into one complex, enabling the
PAE matrix on the structure tool, then runs the ``ipsae-scoring`` tool with the
protein chain as the binder and the DNA chain(s) as targets. The binder-target
ipSAE in ``[0, 1]`` (higher is better) is mapped to a cost where lower is better.

Constraints:
- protein-dna-ipsae: Binder protein vs. target DNA ipSAE interface scoring.

Examples:
    Reward strong protein-DNA interface confidence for a homodimer with AF3:

    >>> from proto_language.core import Constraint, Segment
    >>> protomer = Segment(length=120, sequence_type="protein")
    >>> operator = Segment(length=20, sequence_type="dna")
    >>> ipsae = Constraint(
    ...     inputs=[protomer, operator],
    ...     function=protein_dna_ipsae_constraint,
    ...     function_config={"structure_tool": "alphafold3", "num_protein_copies": 2},
    ... )
"""

from logging import getLogger
from typing import Literal

import numpy as np
from proto_tools import (
    Complex,
    IPSAEScoringConfig,
    IPSAEScoringInput,
    predict_structures,
    run_ipsae_scoring,
)
from proto_tools.entities.structures.selection import ChainSelection, SingleChainSelection
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import ConfigField

logger = getLogger(__name__)

_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence (ACGT alphabet)."""
    return seq.upper().translate(_DNA_COMPLEMENT)[::-1]


class ProteinDNAIpsaeConfig(StructureBasedConstraintConfig):
    """Config for the protein-DNA ipSAE interface constraint.

    Runs a structure prediction for the protein-DNA complex (with the PAE matrix
    enabled) and scores the binder protein vs. target DNA interface with ipSAE
    (Dunbrack 2025). ipSAE restricts to residue pairs with predicted aligned
    error below ``pae_cutoff`` (and CA-CA distance below ``distance_cutoff``),
    yielding a value in ``[0, 1]`` (higher = better) that is compared against
    ``desired_ipsae`` to produce a cost in ``[0, 1]`` (lower = better).

    Attributes:
        num_protein_copies (int): Number of protein monomer copies in the
            complex (2 = homodimer, 1 = monomer). If the input already contains
            this many protein chains, no duplication is performed.
        desired_ipsae (float): Target binder-target ipSAE. The score is 0.0
            (best) once this value is reached.
        include_reverse_complement (bool): If True, append the reverse-complement
            DNA strand when the input has only one DNA sequence.
        pae_cutoff (float): PAE threshold (Å) for interface residue detection,
            forwarded to the ipSAE tool.
        distance_cutoff (float): CA-CA distance cutoff (Å) for contact detection,
            forwarded to the ipSAE tool.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Structure-prediction tool; defaults to a DNA-capable predictor (alphafold3).
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    num_protein_copies: int = ConfigField(
        title="Protein Copies",
        default=2,
        ge=1,
        description="Protein monomer copies in the complex (2=homodimer, 1=monomer); reuses input chains first.",
    )
    desired_ipsae: float = ConfigField(
        title="Desired Prot-DNA ipSAE",
        default=0.5,
        gt=0.0,
        description="Target binder-target ipSAE in [0,1]. Score is 0 when achieved.",
    )
    include_reverse_complement: bool = ConfigField(
        title="Include Rev Complement",
        default=True,
        description="Add the reverse-complement DNA strand when the input has only one DNA sequence.",
    )
    pae_cutoff: float = ConfigField(
        title="PAE Cutoff",
        default=10.0,
        gt=0.0,
        description="PAE threshold (Angstrom) for ipSAE interface residue detection.",
    )
    distance_cutoff: float = ConfigField(
        title="Distance Cutoff",
        default=10.0,
        gt=0.0,
        description="CA-CA distance cutoff (Angstrom) for ipSAE contact detection.",
    )
    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )

    @field_validator("num_protein_copies")
    @classmethod
    def _validate_copies(cls, value: int) -> int:
        """Ensure at least one protein copy is requested."""
        if value < 1:
            raise ValueError("num_protein_copies must be >= 1")
        return value


def _build_chain_layout(
    candidate: tuple[Sequence, ...],
    config: ProteinDNAIpsaeConfig,
    cand_idx: int,
) -> tuple[list[dict[str, str]], list[int], list[int]]:
    """Build the prediction chain list and protein/DNA chain-index layout.

    Protein chains are duplicated up to ``num_protein_copies`` and the reverse
    complement DNA strand is appended when a single DNA strand is supplied and
    ``include_reverse_complement`` is set, mirroring the chain-pair ipTM
    constraint so chain IDs line up with the predicted structure.

    Args:
        candidate (tuple[Sequence, ...]): One candidate's input sequences.
        config (ProteinDNAIpsaeConfig): Validated constraint config.
        cand_idx (int): Candidate index, used only for error messages.

    Returns:
        tuple[list[dict[str, str]], list[int], list[int]]: The chain dicts to
            fold, the protein chain indices, and the DNA chain indices.

    Raises:
        ValueError: If the candidate lacks a protein or a DNA sequence.
    """
    proteins = [seq for seq in candidate if seq.sequence_type == "protein"]
    dnas = [seq for seq in candidate if seq.sequence_type == "dna"]

    if not proteins:
        raise ValueError(f"Candidate {cand_idx}: ipSAE constraint requires at least one protein sequence.")
    if not dnas:
        raise ValueError(f"Candidate {cand_idx}: ipSAE constraint requires at least one DNA sequence.")

    chains: list[dict[str, str]] = []
    protein_indices: list[int] = []
    dna_indices: list[int] = []
    chain_idx = 0

    # Add protein chains, duplicating to reach num_protein_copies.
    protein_seqs = [p.sequence for p in proteins]
    while len(protein_seqs) < config.num_protein_copies:
        protein_seqs.append(proteins[0].sequence)

    for pseq in protein_seqs:
        chains.append({"sequence": pseq, "entity_type": "protein"})
        protein_indices.append(chain_idx)
        chain_idx += 1

    # Add DNA chains, optionally adding the reverse complement strand.
    dna_seqs = [d.sequence for d in dnas]
    if len(dna_seqs) == 1 and config.include_reverse_complement:
        dna_seqs.append(_reverse_complement(dna_seqs[0]))

    for dseq in dna_seqs:
        chains.append({"sequence": dseq, "entity_type": "dna"})
        dna_indices.append(chain_idx)
        chain_idx += 1

    return chains, protein_indices, dna_indices


def _tool_config_with_pae(config: ProteinDNAIpsaeConfig) -> object:
    """Return the active tool config with the PAE matrix enabled when supported.

    ipSAE needs ``structure.metrics['pae']``, which predictors only emit when
    ``include_pae_matrix=True`` — and AlphaFold3, Boltz2, and Protenix all default
    it off, so this copies the active tool config and flips the field on when present.

    Args:
        config (ProteinDNAIpsaeConfig): Validated constraint config.

    Returns:
        object: The active tool config, with ``include_pae_matrix=True`` if the
            field exists on it; otherwise the unmodified active tool config.
    """
    tool_config = config.tool_config
    if "include_pae_matrix" in type(tool_config).model_fields:
        return tool_config.model_copy(update={"include_pae_matrix": True})
    return tool_config


def _ipsae_score(ipsae: float, desired_ipsae: float) -> float:
    """Map a binder-target ipSAE in ``[0, 1]`` to a cost in ``[0, 1]``.

    Returns ``MIN_ENERGY`` (best) once ``ipsae >= desired_ipsae`` and
    ``MAX_ENERGY`` (worst) when ipSAE is 0, linearly interpolating between.

    Args:
        ipsae (float): Binder-target ipSAE from the ipSAE tool (higher = better).
        desired_ipsae (float): Target ipSAE at which the cost reaches 0.

    Returns:
        float: Clipped cost in ``[MIN_ENERGY, MAX_ENERGY]`` (lower = better).
    """
    return float(np.clip((desired_ipsae - ipsae) / desired_ipsae, MIN_ENERGY, MAX_ENERGY))


@constraint(
    key="protein-dna-ipsae",
    label="Protein-DNA ipSAE",
    config=ProteinDNAIpsaeConfig,
    description=(
        "Score protein-DNA interface confidence with ipSAE (Dunbrack 2025), folding the protein-DNA "
        "complex and scoring the binder protein vs. target DNA interface."
    ),
    uses_gpu=True,
    tools_called=[
        "alphafold3-prediction",
        "boltz2-prediction",
        "protenix-prediction",
        "ipsae-scoring",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def protein_dna_ipsae_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: ProteinDNAIpsaeConfig,
) -> list[ConstraintOutput]:
    """Score protein-DNA interface confidence with ipSAE.

    For each candidate complex:

    1. Separate protein and DNA sequences from the input tuple.
    2. Build a complex with ``num_protein_copies`` protein chains and all DNA
       chains (optionally appending the reverse complement of a single strand).
    3. Run the configured structure predictor with the PAE matrix enabled and
       read ``structure.metrics['pae']``.
    4. Run ``ipsae-scoring`` with the first protein chain as the binder and the
       DNA chain(s) as the targets, reading the binder-target ipSAE.
    5. Score: 0.0 (best) when ipSAE >= ``desired_ipsae``, 1.0 (worst) when 0.

    When the predictor does not surface the PAE matrix (e.g. ``include_pae_matrix``
    is unsupported), this logs a warning and returns ``MAX_ENERGY`` with
    ``ipsae_error`` metadata for that candidate.

    **Supported tools**: any DNA-capable ``StructureBasedConstraintConfig``
    predictor that emits the per-residue PAE matrix (AlphaFold3 / Boltz2 /
    Protenix); ``include_pae_matrix`` is enabled automatically here.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of
            protein and DNA input sequences.
        config (ProteinDNAIpsaeConfig): Validated constraint config.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0, 1]`` (lower is
            better) with ``ipsae`` / ``desired_ipsae`` / ``binder_chain`` /
            ``target_chains`` / ``structure_tool`` / ``pdb_output`` metadata and
            the predicted Structure on slot 0.

    Examples:
        Programming a protein-DNA operator complex with AlphaFold3:

        >>> from proto_language.core import Constraint, Segment
        >>> protomer = Segment(length=120, sequence_type="protein")
        >>> operator = Segment(length=20, sequence_type="dna")
        >>> ipsae = Constraint(
        ...     inputs=[protomer, operator],
        ...     function=protein_dna_ipsae_constraint,
        ...     function_config={"structure_tool": "alphafold3", "num_protein_copies": 2},
        ... )
    """
    if not input_sequences:
        return []

    complexes: list[Complex] = []
    layouts: list[tuple[list[int], list[int]]] = []
    for cand_idx, candidate in enumerate(input_sequences):
        chains, protein_indices, dna_indices = _build_chain_layout(candidate, config, cand_idx)
        complexes.append(Complex(chains=chains))
        layouts.append((protein_indices, dna_indices))

    output = predict_structures(complexes, config.structure_tool, _tool_config_with_pae(config), msas=None)
    if len(output.structures) != len(input_sequences):
        raise RuntimeError(f"ipSAE: expected {len(input_sequences)} predictions, got {len(output.structures)}.")

    results: list[ConstraintOutput] = []
    for cand_idx, (structure_obj, candidate_tuple) in enumerate(zip(output.structures, input_sequences, strict=True)):
        protein_indices, dna_indices = layouts[cand_idx]
        n_segments = len(candidate_tuple)

        chain_ids = structure_obj.get_chain_ids()
        binder = chain_ids[protein_indices[0]]
        targets = [chain_ids[i] for i in dna_indices]

        if structure_obj.metrics.get("pae") is None:
            logger.warning(
                "Candidate %d: PAE matrix unavailable from %s; returning worst ipSAE score.",
                cand_idx,
                config.structure_tool,
            )
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"ipsae_error": "pae matrix unavailable"}))
            continue

        ipsae_out = run_ipsae_scoring(
            IPSAEScoringInput(
                structure=structure_obj,
                binder_chain=SingleChainSelection(chain=binder),
                target_chains=ChainSelection(chains=targets),
            ),
            IPSAEScoringConfig(
                pae_cutoff=config.pae_cutoff,
                distance_cutoff=config.distance_cutoff,
                device="cpu",
            ),
        )
        if ipsae_out.metrics.primary_value is None:
            logger.warning(
                "Candidate %d: ipSAE tool returned no primary value; returning worst ipSAE score.",
                cand_idx,
            )
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata={"ipsae_error": "ipsae value unavailable"}))
            continue

        ipsae = float(ipsae_out.metrics.primary_value)
        score = _ipsae_score(ipsae, config.desired_ipsae)

        metadata: dict[str, object] = {
            "ipsae": ipsae,
            "desired_ipsae": config.desired_ipsae,
            "binder_chain": binder,
            "target_chains": targets,
            "structure_tool": config.structure_tool,
            "pdb_output": structure_obj.structure_pdb,
        }
        results.append(
            ConstraintOutput(
                score=score,
                metadata=metadata,
                structures=(structure_obj,) + (None,) * (n_segments - 1),
            )
        )

    return results
