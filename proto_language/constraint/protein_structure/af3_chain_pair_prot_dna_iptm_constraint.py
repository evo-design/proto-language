"""AF3 chain-pair protein-DNA ipTM constraint.

Scores protein-DNA interface confidence by extracting the per-chain-pair ipTM
matrix from a structure prediction and aggregating across protein-DNA pairs.
This gives a much more targeted signal than overall ipTM, which can be inflated
by high protein-protein dimer confidence even when the DNA interface is weak.

The constraint folds each candidate's protein chain(s) (duplicated to
``num_protein_copies``) plus its DNA chain(s) (optionally adding the reverse
complement) into one complex, then reads the per-chain-pair ipTM matrix from the
predicted ``Structure.metrics`` (``chain_pair_iptm`` for AlphaFold3/Protenix, or
``pair_chains_iptm`` for Boltz-2) and selects the protein-to-DNA entries. A
predictor that does not surface the matrix is a hard error: falling back to
overall ipTM would reintroduce the inflated whole-complex signal this constraint
exists to avoid.

Constraints:
- af3-chain-pair-prot-dna-iptm: Max/mean protein-DNA chain-pair ipTM scoring.

Examples:
    Reward strong protein-DNA interface signal for a homodimer with Protenix:

    >>> from proto_language.core import Segment
    >>> protomer = Segment(length=120, sequence_type="protein")
    >>> operator = Segment(length=20, sequence_type="dna")
    >>> chain_pair_iptm = Constraint(
    ...     inputs=[protomer, operator],
    ...     function=af3_chain_pair_prot_dna_iptm_constraint,
    ...     function_config={"structure_tool": "protenix", "num_protein_copies": 2},
    ... )
"""

from typing import Any, Literal

import numpy as np
from proto_tools import Complex, predict_structures
from pydantic import field_validator

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import ConfigField

_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence (ACGT alphabet)."""
    return seq.upper().translate(_DNA_COMPLEMENT)[::-1]


class AF3ChainPairProtDNAIPTMConfig(StructureBasedConstraintConfig):
    """Config for the AF3 chain-pair protein-DNA ipTM constraint.

    Runs a structure prediction for the protein-DNA complex and extracts the
    pairwise chain ipTM matrix. Only the protein-to-DNA entries are kept; the
    result is aggregated via ``aggregation`` (default: max over all protein-DNA
    chain pairs — both dimer halves, and both DNA strands when
    ``include_reverse_complement`` is set), then scored against ``desired_iptm``.

    Note:
        The per-chain-pair ipTM matrix (``chain_pair_iptm``) is exposed by both
        AlphaFold3 and Protenix. Boltz-2 exposes the same matrix under
        ``pair_chains_iptm``. A tool that does not surface the matrix raises a
        ``RuntimeError`` — overall ipTM is not a valid substitute.

    Attributes:
        num_protein_copies (int): Number of protein monomer copies in the
            complex (2 = homodimer, 1 = monomer). If the input already contains
            this many protein chains, no duplication is performed.
        pair_type (Literal["protein-dna", "protein-protein"]): Which interface
            to score: ``"protein-dna"`` (default) scores the protein-DNA
            chain-pair ipTM, ``"protein-protein"`` scores the protein-protein
            chain-pair ipTM (homodimer interface).
        desired_iptm (float): Target protein-DNA chain-pair ipTM. The score is
            0.0 (best) once this value is reached.
        aggregation (Literal["max", "mean"]): How to combine protein-DNA
            chain-pair ipTM values: ``"max"`` keeps the best single pair,
            ``"mean"`` averages across all protein-DNA pairs.
        include_reverse_complement (bool): If True, add the reverse-complement
            DNA strand when the input has only one DNA sequence.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): Predictor for the protein-DNA complex; must be DNA-capable. Default "alphafold3".
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )
    num_protein_copies: int = ConfigField(
        title="Protein Copies",
        default=2,
        ge=1,
        description="Protein monomer copies in the complex (2=homodimer, 1=monomer); reuses input chains first.",
    )
    pair_type: Literal["protein-dna", "protein-protein"] = ConfigField(
        title="Pair Type",
        default="protein-dna",
        description="Interface scored: 'protein-dna' (protein-DNA ipTM) or 'protein-protein' (homodimer ipTM).",
    )
    desired_iptm: float = ConfigField(
        title="Desired Prot-DNA ipTM",
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Target protein-DNA chain-pair ipTM. Score is 0 when achieved.",
    )
    aggregation: Literal["max", "mean"] = ConfigField(
        title="Aggregation Method",
        default="max",
        description="Aggregate protein-DNA chain-pair ipTM: 'max' (best pair) or 'mean' (average of pairs).",
    )
    include_reverse_complement: bool = ConfigField(
        title="Include Rev Complement",
        default=False,
        description="Add the reverse-complement DNA strand when the input has only one DNA sequence.",
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
    config: AF3ChainPairProtDNAIPTMConfig,
    cand_idx: int,
) -> tuple[list[dict[str, str]], list[int], list[int]]:
    """Build the prediction chain list and protein/DNA chain-index layout.

    Protein chains are duplicated up to ``num_protein_copies`` and the reverse
    complement DNA strand is appended when requested, mirroring the original
    proto_language implementation so the matrix indices line up.

    Args:
        candidate (tuple[Sequence, ...]): One candidate's input sequences.
        config (AF3ChainPairProtDNAIPTMConfig): Validated constraint config.
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
        raise ValueError(f"Candidate {cand_idx}: chain-pair ipTM constraint requires at least one protein sequence.")
    if not dnas:
        raise ValueError(f"Candidate {cand_idx}: chain-pair ipTM constraint requires at least one DNA sequence.")

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


def _select_chain_pair_iptm(
    chain_pair_iptm: Any,
    protein_indices: list[int],
    dna_indices: list[int],
    aggregation: str,
) -> tuple[float | None, float | None]:
    """Select and aggregate protein-DNA (and protein-protein) ipTM from the matrix.

    Implements the original chain-pair selection + aggregation math exactly:
    iterate every protein-to-DNA index pair, then reduce by ``max`` or ``mean``;
    separately track the maximum protein-to-protein ipTM for metadata.

    Args:
        chain_pair_iptm (Any): The ``n_chains x n_chains`` ipTM matrix from
            the predictor (or ``None``/too-small when unavailable).
        protein_indices (list[int]): Protein chain indices into the matrix.
        dna_indices (list[int]): DNA chain indices into the matrix.
        aggregation (str): ``"max"`` or ``"mean"`` reducer for protein-DNA pairs.

    Returns:
        tuple[float | None, float | None]: ``(prot_dna_iptm, prot_prot_iptm)``.
            ``prot_dna_iptm`` is ``None`` when the matrix is unavailable or too
            small; ``prot_prot_iptm`` is ``None`` when there is a single protein
            copy.
    """
    n_chains = len(protein_indices) + len(dna_indices)
    if chain_pair_iptm is None or len(chain_pair_iptm) < n_chains:
        return None, None

    prot_dna_values: list[float] = [float(chain_pair_iptm[pi][di]) for pi in protein_indices for di in dna_indices]
    if not prot_dna_values:
        return None, None

    prot_dna_iptm = max(prot_dna_values) if aggregation == "max" else float(np.mean(prot_dna_values))

    prot_prot_values: list[float] = [
        float(chain_pair_iptm[pi][pj]) for i, pi in enumerate(protein_indices) for pj in protein_indices[i + 1 :]
    ]
    prot_prot_iptm = float(max(prot_prot_values)) if prot_prot_values else None

    return prot_dna_iptm, prot_prot_iptm


@constraint(
    key="af3-chain-pair-prot-dna-iptm",
    label="AF3 Chain-Pair Protein-DNA ipTM",
    config=AF3ChainPairProtDNAIPTMConfig,
    description=(
        "Score protein-DNA interface confidence using the chain-pair ipTM matrix, returning the max "
        "(or mean) ipTM across all protein-DNA chain pairs."
    ),
    uses_gpu=True,
    tools_called=[
        "alphafold3-prediction",
        "boltz2-prediction",
        "protenix-prediction",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def af3_chain_pair_prot_dna_iptm_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: AF3ChainPairProtDNAIPTMConfig,
) -> list[ConstraintOutput]:
    """Score protein-DNA ipTM from a per-chain-pair ipTM matrix.

    For each candidate complex:

    1. Separate protein and DNA sequences from the input tuple.
    2. Build a complex with ``num_protein_copies`` protein chains and all DNA
       chains (optionally adding the reverse complement).
    3. Run the configured structure predictor and read ``chain_pair_iptm``.
    4. Pick out the protein-to-DNA entries and aggregate (max or mean).
    5. Score: 0.0 (best) when aggregated ipTM >= ``desired_iptm``, 1.0 (worst)
       when ipTM is 0.

    When the predictor does not expose the per-chain-pair ipTM matrix this
    raises ``RuntimeError``: overall ipTM is not a valid substitute.

    **Supported tools**: any DNA-capable ``StructureBasedConstraintConfig``
    predictor; AlphaFold3 and Protenix emit the per-chain-pair ipTM matrix under
    ``chain_pair_iptm``, and Boltz-2 emits the same matrix under
    ``pair_chains_iptm``.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal tuples of
            protein and DNA input sequences.
        config (AF3ChainPairProtDNAIPTMConfig): Validated constraint config.

    Returns:
        list[ConstraintOutput]: Per-proposal score in ``[0, 1]`` (lower is
            better) with ``prot_dna_iptm`` / ``prot_prot_iptm`` / ``overall_iptm``
            metadata and the predicted Structure on slot 0.

    Raises:
        RuntimeError: If the predictor count mismatches, or if the configured
            tool does not surface the per-chain-pair ipTM matrix.

    Examples:
        Programming a protein-DNA operator complex with Protenix:

        >>> from proto_language.core import Segment
        >>> protomer = Segment(length=120, sequence_type="protein")
        >>> operator = Segment(length=20, sequence_type="dna")
        >>> chain_pair_iptm = Constraint(
        ...     inputs=[protomer, operator],
        ...     function=af3_chain_pair_prot_dna_iptm_constraint,
        ...     function_config={"structure_tool": "protenix", "num_protein_copies": 2},
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

    output = predict_structures(complexes, config.structure_tool, config.tool_config, msas=None)
    if len(output.structures) != len(input_sequences):
        raise RuntimeError(
            f"Chain-pair ipTM: expected {len(input_sequences)} predictions, got {len(output.structures)}."
        )

    results: list[ConstraintOutput] = []
    for cand_idx, (structure_obj, candidate_tuple) in enumerate(zip(output.structures, input_sequences, strict=True)):
        protein_indices, dna_indices = layouts[cand_idx]

        # AlphaFold3 and Protenix surface the per-chain-pair ipTM matrix under
        # "chain_pair_iptm"; Boltz-2 exposes the same matrix under
        # "pair_chains_iptm". Matrix availability is predictor-level, so a tool
        # that omits it is a hard error — overall ipTM is not a valid substitute.
        chain_pair_iptm = structure_obj.metrics.get("chain_pair_iptm")
        if chain_pair_iptm is None:
            chain_pair_iptm = structure_obj.metrics.get("pair_chains_iptm")
        overall_iptm = float(structure_obj.metrics.get("iptm") or 0.0)

        prot_dna_iptm, prot_prot_iptm = _select_chain_pair_iptm(
            chain_pair_iptm, protein_indices, dna_indices, config.aggregation
        )
        if prot_dna_iptm is None:
            raise RuntimeError(
                f"{config.structure_tool} does not surface the per-chain-pair ipTM matrix "
                "(chain_pair_iptm/pair_chains_iptm); use a DNA-capable predictor "
                "(alphafold3/boltz2/protenix)."
            )

        if config.pair_type == "protein-protein":
            if prot_prot_iptm is None:
                raise RuntimeError(
                    "pair_type='protein-protein' requires >= 2 protein copies "
                    f"(got num_protein_copies={config.num_protein_copies})."
                )
            scored_value = prot_prot_iptm
        else:
            scored_value = prot_dna_iptm

        # Score: 0.0 (best) when scored_value >= desired, 1.0 (worst) when 0.
        score = float(np.clip((config.desired_iptm - scored_value) / config.desired_iptm, MIN_ENERGY, MAX_ENERGY))

        metadata: dict[str, object] = {
            "prot_dna_iptm": prot_dna_iptm,
            "prot_prot_iptm": prot_prot_iptm,
            "overall_iptm": overall_iptm,
            "desired_iptm": config.desired_iptm,
            "pair_type": config.pair_type,
            "aggregation": config.aggregation,
            "num_protein_copies": config.num_protein_copies,
            "pdb_output": structure_obj.structure_pdb,
            "structure_tool": config.structure_tool,
        }
        n_segments = len(candidate_tuple)
        results.append(
            ConstraintOutput(
                score=score,
                metadata=metadata,
                structures=(structure_obj,) + (None,) * (n_segments - 1),
            )
        )

    return results
