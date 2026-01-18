from tap import Tap
from typing import Tuple
import random

from proto_language.language.core import (
    Constraint,
    Construct,
    Segment,
    Sequence,
)
from proto_language.language.constraint import (
    splice_transformer_intron_boundary,
    splice_transformer_specificity,
)
from proto_language.language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
    TopKOptimizer,
    TopKOptimizerConfig,
)
from proto_language.language.generator import (
    Evo2Generator,
    Evo2GeneratorConfig,
    UniformMutationGenerator,
    UniformMutationGeneratorConfig,
)
from proto_language.language.core import Program


# SpliceTransformer constants.
TARGET_LENGTH = 1000
CONTEXT_LENGTH = 4000

# Design defaults.
INTRON_LENGTH = 301
N_STEPS = 5_000


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
    intron_constructs = []
    all_constraints = []

    for plasmid_path in plasmid_paths:

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

        intron_construct = Construct([left_flank, intron, right_flank])

        intron_constructs.append(intron_construct)

        #################
        ## Constraints ##
        #################

        donor_pos_all = [donor_start_pos_in_target - 1]
        acceptor_pos_all = [acceptor_end_pos_in_target + 1]

        intron_boundary = Constraint(
            inputs=[left_flank, intron, right_flank],
            function=splice_transformer_intron_boundary,
            function_config={
                "left_context": left_context,
                "right_context": right_context,
                "donor_pos": donor_pos_all,
                "acceptor_pos": acceptor_pos_all,
            },
        )
        all_constraints += [ intron_boundary ]
        if 'max_brain' in args.specificity_type:
            intron_brain_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BRAIN",
                    "direction": "max",
                },
            )
            all_constraints += [ intron_brain_specificity ]
        if 'min_brain' in args.specificity_type:
            intron_brain_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BRAIN",
                    "direction": "min",
                },
            )
            all_constraints += [ intron_brain_specificity ]
        if 'max_blood' in args.specificity_type:
            intron_brain_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BLOOD",
                    "direction": "max",
                },
            )
            all_constraints += [ intron_brain_specificity ]
        if 'min_blood' in args.specificity_type:
            intron_blood_specificity = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_specificity,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": donor_pos_all + acceptor_pos_all,
                    "tissue": "BLOOD",
                    "direction": "min",
                },
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

        for idx, i in enumerate(range(
            0,
            len(intron_constructs) * len(intron_constructs[0].segments),
            len(intron_constructs[0].segments)
        )):
            metadata = outputs[i].selected_sequences[0]._metadata
            output_keys = [
                "specificity_direction",
                "specificity_score",
                "donor_score",
                "acceptor_score",
                "total_splice_score",
            ]
            metakeys = sorted(list(metadata.keys()))
            for output_key in output_keys:
                for key in metakeys:
                    if output_key in key:
                        print(f"\tConstruct {idx}: {key.split('.')[-1]}: {metadata[key]}")


    if args.intron_generator == "evo2":
        optimizer_config = TopKOptimizerConfig(
            num_samples=args.n_steps,
            k=args.n_steps,
            batch_size=1,
            verbose=True,
        )
        optimizer = TopKOptimizer(
            constructs=intron_constructs,
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
            track_step_size=1,
            verbose=True,
        )
        optimizer = MCMCOptimizer(
            constructs=intron_constructs,
            generators=[intron_gen],
            constraints=all_constraints,
            config=optimizer_config,
            custom_logging=custom_logging,
            clear_tool_cache=True,
        )

    program = Program(optimizers=[optimizer])

    program.run()
