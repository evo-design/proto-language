"""CREB DNA design stage for the b2AR-to-TF pathway example."""

from __future__ import annotations

import argparse

import pandas as pd
from Bio import SeqIO

from proto_language.constraint import borzoi_track_activity_constraint
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig

# Design constants.
N_SAMPLES = 300
DESIGN_SEQ_LENGTH = 512
PROMPT_FNAME = "examples/data/creb_dna_design_prompt.fasta"
LEFT_FLANK_FNAME = "examples/data/creb_dna_design_left_flank.fasta"
RIGHT_FLANK_FNAME = "examples/data/creb_dna_design_right_flank.fasta"
BORZOI_HUMAN_TARGETS = "examples/data/borzoi_targets_human.txt"
BORZOI_CONTEXT = 524_288


def clean_dna(sequence: str) -> str:
    """Return uppercase DNA with ambiguous bases replaced for model inputs."""
    return sequence.upper().replace("N", "A")


def creb_track_ids() -> list[int]:
    """Return Borzoi human output tracks for CREB1 HepG2 ChIP-seq."""
    chip_seq_track = "CHIP:CREB1:HepG2"
    borzoi_target_df = pd.read_csv(BORZOI_HUMAN_TARGETS, sep="\t")
    all_tracks = list(borzoi_target_df["description"])
    tracks = [idx for idx, track in enumerate(all_tracks) if chip_seq_track in track]
    if not tracks:
        raise ValueError(f"No Borzoi tracks matched {chip_seq_track!r}.")
    return tracks


def creb_flank_lengths() -> tuple[int, int]:
    """Return left and right flank lengths that produce a full Borzoi context."""
    len_left_flank = (BORZOI_CONTEXT - DESIGN_SEQ_LENGTH + 1) // 2
    len_right_flank = BORZOI_CONTEXT - DESIGN_SEQ_LENGTH - len_left_flank
    return len_left_flank, len_right_flank


def create_creb_dna_program(profile: str = "full") -> Program:
    """Make the CREB design program."""
    smoke = profile == "smoke"
    creb_dna_prompt = clean_dna(str(SeqIO.read(PROMPT_FNAME, "fasta").seq))
    left_flank_seq = clean_dna(str(SeqIO.read(LEFT_FLANK_FNAME, "fasta").seq))
    right_flank_seq = clean_dna(str(SeqIO.read(RIGHT_FLANK_FNAME, "fasta").seq))
    len_left_flank, len_right_flank = creb_flank_lengths()

    left_flank_borzoi = Segment(sequence=left_flank_seq[-len_left_flank:], sequence_type="dna", label="Left Flank")
    creb_dna = Segment(length=DESIGN_SEQ_LENGTH, sequence_type="dna", label="CREB DNA")
    right_flank_borzoi = Segment(sequence=right_flank_seq[:len_right_flank], sequence_type="dna", label="Right Flank")
    borzoi_input_construct = Construct([left_flank_borzoi, creb_dna, right_flank_borzoi], label="CREB Borzoi Input")

    num_samples = 1 if smoke else N_SAMPLES
    evo2_config = Evo2GeneratorConfig(
        prompts=[creb_dna_prompt],
        model_checkpoint="evo2_7b",
        top_k=4,
        top_p=1.0,
        temperature=0.5,
        force_prompt_threshold=1,
        stop_at_eos=False,
        batched=True,
        batch_size=1 if smoke else 10,
        cached_generation=True,
        prepend_prompt=False,
        verbose=True,
    )
    evo2_generator = Evo2Generator(evo2_config)
    evo2_generator.assign(creb_dna)

    borzoi_constraint = Constraint(
        inputs=borzoi_input_construct.segments,
        function=borzoi_track_activity_constraint,
        function_config={
            "borzoi_output_tracks": creb_track_ids(),
            "organism": "human",
            "direction": "maximize",
            "activity_threshold": 200.0,
            "batch_size": 1,
        },
        label="borzoi_creb_track_activity",
    )

    creb_dna_optimizer = RejectionSamplingOptimizer(
        constructs=[borzoi_input_construct],
        generators=[evo2_generator],
        constraints=[borzoi_constraint],
        config=RejectionSamplingOptimizerConfig(num_samples=num_samples, num_results=1, verbose=True),
    )

    return Program(optimizers=[creb_dna_optimizer], num_results=1)


def generate_creb_dna_sequence(profile: str = "full") -> str:
    """Run the program and return the designed sequence."""
    program = create_creb_dna_program(profile=profile)
    program.run()
    creb_dna = str(program.optimizers[0].constructs[0].segments[1].result_sequences[0])
    assert len(creb_dna) == DESIGN_SEQ_LENGTH
    return creb_dna


def main() -> None:
    """Run CREB DNA design as a standalone script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    creb_dna = generate_creb_dna_sequence(profile=args.profile)
    print("Generated CREB sequence:", creb_dna)


if __name__ == "__main__":
    main()
