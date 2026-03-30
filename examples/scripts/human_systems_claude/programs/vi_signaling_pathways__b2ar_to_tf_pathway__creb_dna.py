import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from proto_tools import (
    BORZOI_CONTEXT,  # 524,288 bp.
    BORZOI_OUTPUT,  # 6,144 dimensions.
    BorzoiConfig,
    BorzoiInput,
    run_borzoi,
)

try:
    from proto_tools.tools.causal_models.evo2 import clear_evo2_cache
except ImportError:
    from proto_tools.tools.causal_models.evo2._in_process_mode.evo2_cache import (
        clear_evo2_cache,
    )

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig

# Other Borzoi constants.
BORZOI_OUTPUT_RESOLUTION = 32
BORZOI_FLANK = 163_840
BORZOI_HUMAN_TARGETS = 'examples/data/borzoi_targets_human.txt'
BORZOI_N_HUMAN_TARGETS = 7_611

# Design constants.
N_SAMPLES = 300
DESIGN_SEQ_LENGTH = 512
PROMPT_FNAME = 'examples/data/creb_dna_design_prompt.fasta'
LEFT_FLANK_FNAME = 'examples/data/creb_dna_design_left_flank.fasta'
RIGHT_FLANK_FNAME = 'examples/data/creb_dna_design_right_flank.fasta'


def _borzoi_input_to_output_mask(input_mask: np.ndarray[bool]) -> np.ndarray[bool]:
    """
    Convert a mask over a single-nucleotide Borzoi input to a mask over a binned,
    32-bp resolution Borzoi output.
    """
    assert len(input_mask.shape) == 1
    assert input_mask.shape[0] == BORZOI_CONTEXT

    output_mask_bp = input_mask[BORZOI_FLANK:-BORZOI_FLANK]
    assert len(output_mask_bp) == BORZOI_OUTPUT * BORZOI_OUTPUT_RESOLUTION

    output_mask = output_mask_bp.reshape(-1, BORZOI_OUTPUT_RESOLUTION).any(axis=1).astype(bool)
    assert len(output_mask) == BORZOI_OUTPUT

    return output_mask


class BorzoiDNADesignConfig(BaseConfig):
    """
    Configuration for Borzoi DNA binding activity scoring.

    Attributes:
        borzoi_config (BorzoiConfig): Configuration for Borzoi sequence scoring.
            Import parameters include:
            - output_tracks (List[int]): Integer indices into Borzoi output tracks to use.
              The final activity score is the mean over all of these tracks.
            - species (str): Whether to use human or mouse tracks.
            - replicate (str): Which Borzoi replicate in the ensemble to use.

        output_mask (List[bool]): Boolean mask that is set to true if the output
            bin should contribute to the activity score, false if otherwise. Defaults
            to the entire output contributing to the activity score.
            Default: ``np.ones(BORZOI_OUTPUT).astype(bool)``.

        activity_threshold (float): Maximum activity threshold. If an activity is predicted
            as higher than this value, thresholds the activity to this value instead.
            Also defines the normalization, so activites >= this threshold are reported as a
            score of 0, and zero activity is reported as a score of 1.
            Default: 200.

        verbose (bool): Controls output verbosity.
    """

    borzoi_config: BorzoiConfig = ConfigField(
        title="Borzoi Config",
        description="Borzoi configuration parameters.",
        default_factory=BorzoiConfig,
    )

    output_mask: List[bool] = ConfigField(
        title="Output Mask",
        description="Which region of the Borzoi output to compute activity on.",
        default=np.ones(BORZOI_OUTPUT).astype(bool),
    )

    activity_threshold: float = ConfigField(
        title="Activity Threshold",
        description="Maximum activity threshold",
        default=200.,
    )

    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Control output verbosity",
    )


@ConstraintRegistry.register(
    key="borzoi_creb_dna_design",
    label="Borzoi CREB DNA Design",
    config=BorzoiDNADesignConfig,
    description="Compute activity of Borzoi-predicted CREB1 ChIP-seq",
    supported_sequence_types=["dna"],
    uses_gpu=False,
    tools_called=["borzoi"],
    category="epigenomic_sequence_scoring",
)
def _borzoi_creb_dna_design(
    complexes: List[Tuple[Sequence]],
    config: BorzoiDNADesignConfig,
) -> List[float]:
    """
    Returns Borzoi-predicted activity of DNA sequence.
    """
    clear_evo2_cache()  # Manually clear cache to free memory.

    config.borzoi_config.verbose = False  # Suppress output that prints full input sequence.

    # For this constraint, always take the mean over all tracks.
    config.borzoi_config.avg_output_tracks = True

    scores = []
    for comp in complexes:
        sequence = "".join(str(subseq) for subseq in comp)

        sequence = sequence.replace('N', 'A')  # Hack as Borzoi does not accept Ns.

        borzoi_input = BorzoiInput(sequence=str(sequence))

        borzoi_output = run_borzoi(borzoi_input, config.borzoi_config)

        assert len(borzoi_output.prediction) == 1
        assert len(borzoi_output.prediction[0]) == BORZOI_OUTPUT
        prediction = np.array(borzoi_output.prediction[0])
        activity_score = prediction[config.output_mask].mean()

        assert activity_score >= 0.
        score = 1. - (min(activity_score, config.activity_threshold) / config.activity_threshold)
        scores.append(score)

    return scores


def create_creb_dna_program() -> Program:
    """
    Make the CREB design program. Maximize Borzoi-predicted CREB ChIP-seq activity.
    """

    # =============================================================================
    # Input values and parameters.
    # =============================================================================

    # DNA prompts and contexts.
    creb_dna_prompt = str(SeqIO.read(PROMPT_FNAME, 'fasta').seq)

    left_flank_seq = str(SeqIO.read(LEFT_FLANK_FNAME, 'fasta').seq)
    assert len(left_flank_seq) >= BORZOI_CONTEXT // 2

    right_flank_seq = str(SeqIO.read(RIGHT_FLANK_FNAME, 'fasta').seq)
    assert len(right_flank_seq) >= BORZOI_CONTEXT // 2

    # Lengths and masks.
    len_left_flank = math.ceil((BORZOI_CONTEXT - DESIGN_SEQ_LENGTH) / 2.)
    len_right_flank = math.floor((BORZOI_CONTEXT - DESIGN_SEQ_LENGTH) / 2.)

    input_design_mask = np.concatenate([
        np.zeros(len_left_flank),
        np.ones(DESIGN_SEQ_LENGTH),
        np.zeros(len_right_flank),
    ]).astype(bool)
    assert len(input_design_mask) == BORZOI_CONTEXT
    output_design_mask = _borzoi_input_to_output_mask(input_design_mask)

    # Borzoi tracks.
    chip_seq_track = 'CHIP:CREB1:HepG2'
    borzoi_target_df = pd.read_csv(BORZOI_HUMAN_TARGETS, sep='\t')
    assert len(borzoi_target_df) == BORZOI_N_HUMAN_TARGETS
    all_tracks = list(borzoi_target_df['description'])
    creb1_tracks = [
        idx for idx, track in enumerate(all_tracks) if chip_seq_track in track
    ]

    # =============================================================================
    # Segments, Constructs, and Generators
    # =============================================================================

    creb_dna = Segment(
        length=DESIGN_SEQ_LENGTH,
        sequence_type="dna",
    )
    left_flank_borzoi = Segment(
        sequence=left_flank_seq[-len_left_flank:],
        sequence_type="dna",
    )
    right_flank_borzoi = Segment(
        sequence=right_flank_seq[:len_right_flank],
        sequence_type="dna",
    )

    borzoi_input_construct = Construct([
        left_flank_borzoi,
        creb_dna,
        right_flank_borzoi
    ])

    evo2_config = Evo2GeneratorConfig(
        prompts=[creb_dna_prompt] * N_SAMPLES,
        model_checkpoint='evo2_7b',
        top_k=4,
        top_p=1.,
        temperature=0.5,
        force_prompt_threshold=1,
        stop_at_eos=False,
        batched=True,
        batch_size=10,
        cached_generation=True,
        prepend_prompt=False,
        verbose=True,
    )
    evo2_generator = Evo2Generator(evo2_config)
    evo2_generator.assign(creb_dna)

    # =============================================================================
    # Constraints
    # =============================================================================

    constraints = []
    for i in range(4):
        borzoi_replicate = str(i)
        constraint_borzoi_replicate = Constraint(
            inputs=borzoi_input_construct.segments,
            function=_borzoi_creb_dna_design,
            function_config={
                'borzoi_config': {
                    'output_tracks': creb1_tracks,
                    'species': 'human',
                    'replicate': borzoi_replicate,
                },
                'output_mask': output_design_mask,
                'activity_threshold': 200.,
                'verbose': True,
            },
            label=f'borzoi_creb_dna_design_{i}',
        )
        constraints.append(constraint_borzoi_replicate)

    # =============================================================================
    # Program modification
    # =============================================================================

    top_k_optimizer_config = TopKOptimizerConfig(
        num_samples=N_SAMPLES,
        num_results=1,
        samples_per_round=N_SAMPLES,
        verbose=True,
    )
    creb_dna_optimizer = TopKOptimizer(
        constructs=[borzoi_input_construct],
        generators=[evo2_generator],
        constraints=constraints,
        config=top_k_optimizer_config,
    )

    creb_dna_program = Program(optimizers=[creb_dna_optimizer], num_results=1)

    return creb_dna_program


def generate_creb_dna_sequence() -> str:
    """
    Run the program and return the designed sequence.
    """
    program = create_creb_dna_program()
    program.run()
    creb_dna = str(program.optimizers[0].constructs[0].segments[1].result_sequences[0])
    assert len(creb_dna) == DESIGN_SEQ_LENGTH
    return creb_dna


if __name__ == '__main__':
    creb_dna = generate_creb_dna_sequence()

    print('Generated CREB sequence:', creb_dna)
