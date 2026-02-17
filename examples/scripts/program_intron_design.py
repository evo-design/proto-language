import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tap import Tap

from proto_language.language.constraint import (
    splice_transformer_intron_boundary,
    splice_transformer_specificity,
)
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import (
    Evo2Generator,
    Evo2GeneratorConfig,
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
    TopKOptimizer,
    TopKOptimizerConfig,
)

# SpliceTransformer constants.
TARGET_LENGTH = 1000
CONTEXT_LENGTH = 4000

# Design defaults.
INTRON_LENGTH = 301
N_STEPS = 5_000


def _enable_mcmc_energy_logging() -> None:
    """
    Enable default per-iteration MCMC energy logs from the optimizer module.
    """
    mcmc_logger = logging.getLogger("proto_language.language.optimizer.mcmc_optimizer")
    mcmc_logger.setLevel(logging.DEBUG)
    if not mcmc_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        mcmc_logger.addHandler(handler)
    mcmc_logger.propagate = False


def _join_three_part_target_sequence(sequence_tuple: Tuple[Sequence, ...]) -> Sequence:
    """
    Concatenate left flank + intron core + right flank into the 1-kb target sequence
    required by SpliceTransformer constraints.
    """
    if len(sequence_tuple) != 3:
        raise ValueError(
            f"Expected 3 input sequences (left_flank, intron, right_flank), got {len(sequence_tuple)}."
        )
    left_flank, intron_core, right_flank = sequence_tuple
    return Sequence(
        sequence=left_flank.sequence + intron_core.sequence + right_flank.sequence,
        sequence_type=left_flank.sequence_type,
    )


def splice_transformer_intron_boundary_three_part(
    input_sequences: List[Tuple[Sequence, ...]],
    config: Any,
) -> List[float]:
    """
    Script-local compatibility wrapper.

    The current base constraint expects one concatenated target sequence tuple.
    This script historically passes three segments to keep the intron mutable while
    keeping flanks fixed, so we concatenate per tuple before scoring and propagate
    the resulting metadata back to each tuple member for optimizer logging.
    """
    concatenated_inputs = []
    for sequence_tuple in input_sequences:
        concatenated_target = _join_three_part_target_sequence(sequence_tuple)
        concatenated_inputs.append((concatenated_target,))

    scores = splice_transformer_intron_boundary(concatenated_inputs, config)

    for original_tuple, (concatenated_target,) in zip(input_sequences, concatenated_inputs):
        metadata_update = dict(concatenated_target._metadata)
        for sequence in original_tuple:
            sequence._metadata.update(metadata_update)

    return scores


def splice_transformer_specificity_three_part(
    input_sequences: List[Tuple[Sequence, ...]],
    config: Any,
) -> List[float]:
    """
    Script-local compatibility wrapper for tissue-specific SpliceTransformer scoring
    over three-part input tuples.
    """
    concatenated_inputs = []
    for sequence_tuple in input_sequences:
        concatenated_target = _join_three_part_target_sequence(sequence_tuple)
        concatenated_inputs.append((concatenated_target,))

    scores = splice_transformer_specificity(concatenated_inputs, config)

    for original_tuple, (concatenated_target,) in zip(input_sequences, concatenated_inputs):
        metadata_update = dict(concatenated_target._metadata)
        for sequence in original_tuple:
            sequence._metadata.update(metadata_update)

    return scores


# Keep config and type validation in Constraint while allowing three-part tuples.
splice_transformer_intron_boundary_three_part._constraint_config_class = (
    splice_transformer_intron_boundary._constraint_config_class
)
splice_transformer_intron_boundary_three_part._constraint_supported_sequence_types = (
    splice_transformer_intron_boundary._constraint_supported_sequence_types
)
splice_transformer_intron_boundary_three_part._constraint_num_input_sequences_per_tuple = 3

splice_transformer_specificity_three_part._constraint_config_class = (
    splice_transformer_specificity._constraint_config_class
)
splice_transformer_specificity_three_part._constraint_supported_sequence_types = (
    splice_transformer_specificity._constraint_supported_sequence_types
)
splice_transformer_specificity_three_part._constraint_num_input_sequences_per_tuple = 3


def _get_constraints_metadata(sequence: Sequence) -> Dict[str, Any]:
    """
    Return constraint metadata from a Sequence across old/new metadata layouts.
    """
    metadata_view = getattr(sequence, "metadata", None)
    if isinstance(metadata_view, dict):
        constraints = metadata_view.get("constraints")
        if isinstance(constraints, dict):
            return constraints

    constraints_metadata = getattr(sequence, "_constraints_metadata", None)
    if isinstance(constraints_metadata, dict):
        return constraints_metadata

    raw_metadata = getattr(sequence, "_metadata", None)
    if isinstance(raw_metadata, dict):
        constraints = raw_metadata.get("constraints")
        if isinstance(constraints, dict):
            return constraints

    return {}


class ProgramIntronDesignArgs(Tap):
    intron_length: int = INTRON_LENGTH
    n_steps: int = N_STEPS
    step_size: int = 1
    temperature: float = 1.
    temperature_min: float = 0.001
    plasmid_context_path: str = 'examples/data/intron_plasmid_context.txt'
    gene_sequence_path: str = 'examples/data/mscarlet.txt'
    gene_insertion_pos: int = 159*3  # Canonical mScarlet split complementation site.
    initialization: str = 'random'
    intron_generator: str = 'uniform'
    multicontext: bool = True
    specificity_type: str = 'max_brain_min_blood'


def get_initial_intron(args: ProgramIntronDesignArgs) -> str:
    if args.initialization == 'random':
        initial_intron = "GT" + "".join(random.choices('ACGT', k=(args.intron_length - 4))) + "AG"
    elif args.initialization == 'poly_a':
        initial_intron = "GT" + ("A" * (args.intron_length - 4)) + "AG"  # Keep for debugging.
    elif args.initialization == 'hbb1':
        # HBB intron 1:
        initial_intron = "GTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAG"
    elif args.initialization == 'hbb2c':
        # HBB intron 2 chimeric:
        initial_intron = "GTAAGTACCGCCTATAGAGTCTATAGGCCCACAAAAAATGCTTTCTTCTTTTAATATACTTTTTTGTTTATCTTATTTCTAATACTTTCCCTAATCTCTTTCTTTCAGGGCAATAATGATACAATGTATCATGCCTCTTTGCACCATTCTAAAGAATAACAGTGATAATTTCTGGGTTAAGGCAATAGCAATATTTCTGCATATAAATATTTCTGCATATAAATTGTAACTGATGTAAGAGGTTTCATATTGCTAATAGCAGCTACAATCCAGCTACCATTCTGCTTTTATTTTATGGTTGGGATAAGGCTGGATTATTCTGAGTCCAAGCTAGGCCCTTTTGCTAATCATGTTCATACCTCTTATCTTCCTCCCACAG"
    elif args.initialization == 'hbb2':
        # HBB intron 2 wildtype:
        initial_intron = "GTGAGTCTATGGGACGCTTGATGTTTTCTTTCCCCTTCTTTTCTATGGTTAAGTTCATGTCATAGGAAGGGGATAAGTAACAGGGTACAGTTTAGAATGGGAAACAGACGAATGATTGCATCAGTGTGGAAGTCTCAGGATCGTTTTAGTTTCTTTTATTTGCTGTTCATAACAATTGTTTTCTTTTGTTTAATTCTTGCTTTCTTTTTTTTTCTTCTCCGCAATTTTTACTATTATACTTAATGCCTTAACATTGTGTATAACAAAAGGAAATATCTCTGAGATACATTAAGTAACTTAAAAAAAAACTTTACACAGTCTGCCTAGTACATTACTATTTGGAATATATGTGTGCTTATTTGCATATTCATAATCTCCCTACTTTATTTTCTTTTATTTTTAATTGATACATAATCATTATACATATTTATGGGTTAAAGTGTAATGTTTTAATATGTGTACACATATTGACCAAATCAGGGTAATTTTGCATTTGTAATTTTAAAAAATGCTTTCTTCTTTTAATATACTTTTTTGTTTATCTTATTTCTAATACTTTCCCTAATCTCTTTCTTTCAGGGCAATAATGATACAATGTATCATGCCTCTTTGCACCATTCTAAAGAATAACAGTGATAATTTCTGGGTTAAGGCAATAGCAATATCTCTGCATATAAATATTTCTGCATATAAATTGTAACTGATGTAAGAGGTTTCATATTGCTAATAGCAGCTACAATCCAGCTACCATTCTGCTTTTATTTTATGGTTGGGATAAGGCTGGATTATTCTGAGTCCAAGCTAGGCCCTTTTGCTAATCATGTTCATACCTCTTATCTTCCTCCCACAG"
    else:
        raise ValueError(f'Invalid initialization type: "{args.initialization}"')

    return initial_intron


def process_splice_transformer_input(
    initial_intron: str,
    args: ProgramIntronDesignArgs,
) -> Tuple[str, str, str, str, str, str, str]:
    """
    Process the input to SpliceTransformer.

    SpliceTransformer has a very particular input/output architecture, that takes in a 1-kb target
    sequence on which the model makes a prediction for each position. The model also takes in a 4-kb
    context to the left and right of the target sequence.

    This script constructs a target sequence where the *intron* is centered.
    The gene (exons + intron) may be truncated if it is longer than the target length.
    The target sequence is padded with plasmid context.

    Returns:
        `left_context`: The sequence of the left context for SpliceTransformer.
        `right_context`: The sequence of the right context for SpliceTransformer.
        `target_seq`: The target sequence for SpliceTransfomer.
        `gene_start_pos_in_target`: 0-index into the target sequence where the (potentially truncated) CDS starts.
        `gene_end_pos_in_target`: 0-index into the target sequence where the (potentially truncated) CDS ends.
        `donor_start_pos_in_target`: 0-index into the target sequence where the intron starts.
        `acceptor_end_pos_in_target`:  0-index into the target sequence where the intron ends.
    """
    with open(args.plasmid_context_path) as f:
        plasmid_context = f.read().rstrip()
    with open(args.gene_sequence_path) as f:
        gene_seq = f.read().rstrip()

    left_exon = gene_seq[: args.gene_insertion_pos]
    right_exon = gene_seq[args.gene_insertion_pos :]

    # Center the intron within the target length.
    donor_start_pos_in_target = (TARGET_LENGTH - len(initial_intron)) // 2
    acceptor_end_pos_in_target = donor_start_pos_in_target + len(initial_intron) - 1
    right_exon_start_in_target = acceptor_end_pos_in_target + 1

    # Handle left side (exon + padding).
    space_on_left = donor_start_pos_in_target
    gene_start_pos_in_target_ideal = space_on_left - len(left_exon)

    if gene_start_pos_in_target_ideal >= 0:
        # Full left exon fits
        len_left_pad = gene_start_pos_in_target_ideal
        left_pad_seq = plasmid_context[-len_left_pad:] if len_left_pad > 0 else ""
        left_exon_in_target = left_exon
        left_exon_in_context = ""
        gene_start_pos_in_target = gene_start_pos_in_target_ideal
    else:
        # Left exon is truncated
        len_left_pad = 0
        left_pad_seq = ""
        left_exon_in_target = left_exon[-space_on_left:] # Take the end of the left exon.
        left_exon_in_context = left_exon[:-space_on_left]
        gene_start_pos_in_target = 0

    # Handle right side (exon + padding).
    space_on_right = TARGET_LENGTH - right_exon_start_in_target
    gene_end_pos_in_target_ideal = right_exon_start_in_target + len(right_exon) - 1

    if gene_end_pos_in_target_ideal < TARGET_LENGTH:
        # Full right exon fits
        len_right_pad = TARGET_LENGTH - (gene_end_pos_in_target_ideal + 1)
        right_pad_seq = plasmid_context[:len_right_pad] if len_right_pad > 0 else ""
        right_exon_in_target = right_exon
        right_exon_in_context = ""
        gene_end_pos_in_target = gene_end_pos_in_target_ideal
    else:
        # Right exon is truncated
        len_right_pad = 0
        right_pad_seq = ""
        right_exon_in_target = right_exon[:space_on_right] # Take the start of the right exon.
        right_exon_in_context = right_exon[space_on_right:]
        gene_end_pos_in_target = TARGET_LENGTH - 1

    target_seq = (
        left_pad_seq +
        left_exon_in_target +
        initial_intron +
        right_exon_in_target +
        right_pad_seq
    )

    left_context = plasmid_context[
        -len_left_pad - (CONTEXT_LENGTH - len(left_exon_in_context)) : -len_left_pad if len_left_pad > 0 else None
    ] + left_exon_in_context
    right_context = right_exon_in_context + plasmid_context[
        len_right_pad : len_right_pad + (CONTEXT_LENGTH - len(right_exon_in_context))
    ]

    assert len(target_seq) == TARGET_LENGTH, \
        f"Target seq length is {len(target_seq)}, expected {TARGET_LENGTH}"
    assert len(left_context) == CONTEXT_LENGTH, \
        f"Left context length is {len(left_context)}, expected {CONTEXT_LENGTH}"
    assert len(right_context) == CONTEXT_LENGTH, \
        f"Right context length is {len(right_context)}, expected {CONTEXT_LENGTH}"
    assert target_seq[donor_start_pos_in_target : donor_start_pos_in_target + 2] == "GT", \
        "Intron does not start with GT"
    assert target_seq[acceptor_end_pos_in_target - 1 : acceptor_end_pos_in_target + 1] == "AG", \
        "Intron does not end with AG"

    return (
        left_context,
        right_context,
        target_seq,
        gene_start_pos_in_target,
        gene_end_pos_in_target,
        donor_start_pos_in_target,
        acceptor_end_pos_in_target,
    )


if __name__ == '__main__':
    args = ProgramIntronDesignArgs(explicit_bool=True).parse_args()
    _enable_mcmc_energy_logging()

    print(args)

    initial_intron = get_initial_intron(args)

    if args.multicontext:
        plasmid_paths = [
            'examples/data/intron_plasmid_context.txt',
            'examples/data/plasmid_context_Ef1a.txt',
            'examples/data/plasmid_context_sffv.txt',
        ]
    else:
        plasmid_paths = [ 'examples/data/intron_plasmid_context.txt' ]

    intron = None
    intron_construct = None
    all_constraints = []

    for context_idx, plasmid_path in enumerate(plasmid_paths):

        args.plasmid_context_path = plasmid_path

        (
            left_context,
            right_context,
            target_seq,
            gene_start_pos_in_target,
            gene_end_pos_in_target,
            donor_start_pos_in_target,
            acceptor_end_pos_in_target,
        ) = process_splice_transformer_input(initial_intron, args)

        print('intron range (start, end):', (donor_start_pos_in_target, acceptor_end_pos_in_target + 1))

        #########################
        ## Sequence generation ##
        #########################

        assert acceptor_end_pos_in_target - donor_start_pos_in_target + 1 == len(initial_intron)

        if intron is None:
            intron = Segment(sequence=target_seq[
                    donor_start_pos_in_target + 2 : acceptor_end_pos_in_target - 1
                ],
                sequence_type="dna",
            )

            if args.intron_generator == "uniform":
                intron_gen_config = UniformMutationGeneratorConfig(
                    num_mutations=args.step_size,
                    #mutation_window=(2, len(initial_intron) - 2),
                )
                intron_gen = UniformMutationGenerator(intron_gen_config)
                intron_gen.assign(intron)

            elif args.intron_generator == "evo2":
                from Bio import SeqIO
                record = SeqIO.read("examples/data/hbb_intron2_prompt.txt", "fasta")
                hg38_prompt = str(record.seq)
                intron_gen_config = Evo2GeneratorConfig(
                    prompts=[hg38_prompt],
                    top_k=4,
                    temperature=1.,
                )
                intron_gen = Evo2Generator(intron_gen_config)
                intron_gen.sequence_length = args.intron_length - 4
                intron_gen.assign(intron)

            else:
                raise ValueError(f"Unsupported intron generator {args.intron_generator}")

        else:
            assert target_seq[donor_start_pos_in_target + 2 : acceptor_end_pos_in_target - 1] == intron.original_sequence._sequence

        left_flank = Segment(sequence=target_seq[: donor_start_pos_in_target + 2],
            sequence_type="dna",
        )

        right_flank = Segment(sequence=target_seq[acceptor_end_pos_in_target - 1 :],
            sequence_type="dna",
        )

        # New Program validation forbids reusing one Segment instance across multiple
        # constructs. Keep a single construct and apply additional context constraints
        # with separate flank segments.
        if intron_construct is None:
            intron_construct = Construct([left_flank, intron, right_flank])

        #################
        ## Constraints ##
        #################

        donor_eval_pos = donor_start_pos_in_target - 1
        acceptor_eval_pos = acceptor_end_pos_in_target + 1

        # SpliceTransformer donor is scored at the base immediately before "GT".
        assert 0 <= donor_eval_pos < TARGET_LENGTH, (
            f"Donor eval pos {donor_eval_pos} out of bounds for target length {TARGET_LENGTH}"
        )
        assert target_seq[donor_eval_pos + 1 : donor_eval_pos + 3] == "GT", (
            "Donor evaluation position is not immediately before GT."
        )

        # SpliceTransformer acceptor is scored at the base immediately after "AG".
        assert 0 <= acceptor_eval_pos < TARGET_LENGTH, (
            f"Acceptor eval pos {acceptor_eval_pos} out of bounds for target length {TARGET_LENGTH}"
        )
        assert target_seq[acceptor_eval_pos - 2 : acceptor_eval_pos] == "AG", (
            "Acceptor evaluation position is not immediately after AG."
        )

        donor_pos_all = [donor_eval_pos]
        acceptor_pos_all = [acceptor_eval_pos]
        context_label = Path(plasmid_path).stem

        intron_boundary = Constraint(
            inputs=[left_flank, intron, right_flank],
            function=splice_transformer_intron_boundary_three_part,
            function_config={
                "left_context": left_context,
                "right_context": right_context,
                "donor_pos": donor_pos_all,
                "acceptor_pos": acceptor_pos_all,
            },
            label=f"splice_boundary__{context_label}__{context_idx}",
        )
        all_constraints += [ intron_boundary ]
        if 'max_brain' in args.specificity_type:
            intron_brain_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity_three_part,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BRAIN",
                    "direction": "max",
                },
                label=f"splice_specificity_brain_max__{context_label}__{context_idx}",
            )
            all_constraints += [ intron_brain_specificity ]
        if 'min_brain' in args.specificity_type:
            intron_brain_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity_three_part,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BRAIN",
                    "direction": "min",
                },
                label=f"splice_specificity_brain_min__{context_label}__{context_idx}",
            )
            all_constraints += [ intron_brain_specificity ]
        if 'max_blood' in args.specificity_type:
            intron_brain_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity_three_part,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BLOOD",
                    "direction": "max",
                },
                label=f"splice_specificity_blood_max__{context_label}__{context_idx}",
            )
            all_constraints += [ intron_brain_specificity ]
        if 'min_blood' in args.specificity_type:
            intron_blood_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity_three_part,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BLOOD",
                    "direction": "min",
                },
                label=f"splice_specificity_blood_min__{context_label}__{context_idx}",
            )
            all_constraints += [ intron_blood_specificity ]

    #############
    ## Program ##
    #############

    def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
        left_flank_sequence: Sequence = outputs[0].selected_sequences[0]._sequence
        intron_sequence: Sequence = outputs[1].selected_sequences[0]._sequence
        right_flank_sequence: Sequence = outputs[2].selected_sequences[0]._sequence
        print(
            f"\tsequence (left_flank): {left_flank_sequence}\n"
            f"\tsequence (intron): {intron_sequence}\n"
            f"\tsequence (right_flank): {right_flank_sequence}"
        )

        constraints = _get_constraints_metadata(outputs[1].selected_sequences[0])
        for constraint_label in sorted(constraints):
            constraint_data = constraints[constraint_label]
            metric_data = constraint_data.get("data", {})
            for metric_name in sorted(metric_data):
                metric_value = metric_data[metric_name]
                if (
                    metric_name in {"donor_score", "acceptor_score", "total_splice_score"}
                    or metric_name.startswith("specificity_direction")
                    or metric_name.startswith("specificity_score")
                ):
                    print(f"\t{constraint_label}: {metric_name}: {metric_value}")


    if intron_construct is None:
        raise RuntimeError("No intron construct was created.")

    if args.intron_generator == "evo2":
        optimizer_config = TopKOptimizerConfig(
            num_samples=args.n_steps,
            k=args.n_steps,
            batch_size=1,
            verbose=True,
        )
        optimizer = TopKOptimizer(
            constructs=[intron_construct],
            generators=[intron_gen],
            constraints=all_constraints,
            config=optimizer_config,
            custom_logging=custom_logging,
            clear_tool_cache=True,
        )
    else:
        optimizer_config = MCMCOptimizerConfig(
            num_selected=1,
            num_steps=args.n_steps,
            max_temperature=args.temperature,
            min_temperature=args.temperature_min,
            verbose=True,
        )
        optimizer = MCMCOptimizer(
            constructs=[intron_construct],
            generators=[intron_gen],
            constraints=all_constraints,
            config=optimizer_config,
            custom_logging=custom_logging,
            clear_tool_cache=True,
        )

    program = Program(optimizers=[optimizer])

    program.run()
