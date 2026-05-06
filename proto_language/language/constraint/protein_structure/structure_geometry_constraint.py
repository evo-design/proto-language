"""AF2 multimer-backed structural geometry objectives.

These constraints expose geometric and fold-shaping objectives that can be used
for design, filtering, or ranking. They are separate from structure confidence
metrics: contact, compactness, secondary-structure, distogram, and
termini-distance losses describe desired structural features rather than
prediction confidence.
"""

from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils.alphafold2_multimer import (
    evaluate_af2_multimer_loss_constraint,
)


@constraint(
    key="structure-contact",
    label="Structure Contact Loss",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 intra-chain contact loss.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_contact_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer intra-chain contact loss.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Raw AF2 ``con`` loss outputs.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError("structure-contact requires structure_tool='alphafold2_multimer'.")
    return evaluate_af2_multimer_loss_constraint(input_sequences, config, "con")


@constraint(
    key="structure-interface-contact",
    label="Structure Interface Contact",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 interface contact loss.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_interface_contact_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer interface contact loss.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Raw AF2 ``i_con`` loss outputs.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError("structure-interface-contact requires structure_tool='alphafold2_multimer'.")
    return evaluate_af2_multimer_loss_constraint(input_sequences, config, "i_con")


@constraint(
    key="structure-radius-gyration",
    label="Structure Radius Gyration",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 radius-of-gyration loss.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_radius_gyration_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer radius-of-gyration loss.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Raw AF2 ``rg`` loss outputs.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError("structure-radius-gyration requires structure_tool='alphafold2_multimer'.")
    return evaluate_af2_multimer_loss_constraint(input_sequences, config, "rg")


@constraint(
    key="structure-helix",
    label="Structure Helix Loss",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 helical-content loss.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_helix_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer helical-content loss.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Raw AF2 ``helix`` loss outputs.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError("structure-helix requires structure_tool='alphafold2_multimer'.")
    return evaluate_af2_multimer_loss_constraint(input_sequences, config, "helix")


@constraint(
    key="structure-beta-strand",
    label="Structure Beta Strand",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 beta-strand-content loss.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_beta_strand_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer beta-strand-content loss.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Raw AF2 ``beta_strand`` loss outputs.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError("structure-beta-strand requires structure_tool='alphafold2_multimer'.")
    return evaluate_af2_multimer_loss_constraint(input_sequences, config, "beta_strand")


@constraint(
    key="structure-distogram-cce",
    label="Structure Distogram CCE",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 distogram CCE loss.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_distogram_cce_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer distogram cross-entropy loss.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Raw AF2 ``dgram_cce`` loss outputs.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError("structure-distogram-cce requires structure_tool='alphafold2_multimer'.")
    return evaluate_af2_multimer_loss_constraint(input_sequences, config, "dgram_cce")


@constraint(
    key="structure-termini-distance",
    label="Structure Termini Distance",
    config=StructureBasedConstraintConfig,
    description="Evaluate AF2 N-to-C termini distance loss.",
    uses_gpu=True,
    tools_called=["alphafold2-multimer"],
    category="protein_structure",
    supported_sequence_types=["protein"],
    input_labels=None,
)
def structure_termini_distance_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: StructureBasedConstraintConfig
) -> list[ConstraintOutput]:
    """Evaluate AF2 multimer N-to-C termini distance loss.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal input tuples.
        config (StructureBasedConstraintConfig): AF2 multimer structure config.

    Returns:
        list[ConstraintOutput]: Raw AF2 ``NC`` loss outputs.
    """
    if config.structure_tool != "alphafold2_multimer":
        raise ValueError("structure-termini-distance requires structure_tool='alphafold2_multimer'.")
    return evaluate_af2_multimer_loss_constraint(input_sequences, config, "NC")


_AF2_MULTIMER_RAW_SCORE_FUNCTIONS = (
    structure_contact_constraint,
    structure_interface_contact_constraint,
    structure_radius_gyration_constraint,
    structure_helix_constraint,
    structure_beta_strand_constraint,
    structure_distogram_cce_constraint,
    structure_termini_distance_constraint,
)

for _af2_fn in _AF2_MULTIMER_RAW_SCORE_FUNCTIONS:
    _af2_fn._constraint_allow_raw_scores = True  # type: ignore[attr-defined]
