"""
Define a program for diversification of the U5 snRNA.
"""
from itertools import islice
from typing import Callable, List

from Bio import Align, SeqIO

from proto_language.base_config import BaseConfig, ConfigField
from proto_language.language.constraint.constraint_registry import ConstraintRegistry
from proto_language.language.constraint.rna_secondary_structure import (
    RNABasePairSimilarityConfig,
    RNAFeatureSimilarityConfig,
    RNAMotifSimilarityConfig,
    RNAPropertySimilarityConfig,
    rna_basepair_similarity_constraint,
    rna_feature_similarity_constraint,
    rna_motif_similarity_constraint,
    rna_property_similarity_constraint,
)
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.language.optimizer import TopKOptimizer, TopKOptimizerConfig

# Design constants.
U5_SNRNA = "AUACUCUGGUUUCUCUUCAGAUCGCAUAAAUCUUUCGCCUUUUACUAAAGAUUUCCGUGGAGAGGAACAACUCUGAGUCUUAACCCAAUUUUUUGAGGCCUUGCUUUGGCAAGGCUA"
U5_PROMPT_FASTA = "examples/data/u5_prompts.fasta"
SEP_SEQUENCE = "GGGGGGGG"
N_PROMPT_EXAMPLES = 32
N_SAMPLES = 150  # Number of sequences to sample with Evo 2.


def _compute_rna_similarity(seq1: str, seq2: str) -> float:
    """Compute normalized pairwise similarity between two RNA sequences."""
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0  # Penalize mismatches
    aligner.gap_score = -1.0

    score = aligner.score(seq1, seq2)

    # Normalize by the length of the longer sequence
    # Perfect match of equal-length seqs gives score = len, so normalized = 1.0
    max_len = max(len(seq1), len(seq2))
    normalized_score = score / max_len

    # Clamp to [0, 1] since negative scores are possible with many mismatches
    return max(0.0, normalized_score)


class Evo2RNAConstraintConfig(BaseConfig):
    """
    Let user pass in constraint function and config to Evo 2 constraint wrapper.

    Attributes:
        constraint_func (Callable): Constraint function to evaluate.
        constraint_config (BaseConfig): Constraint config to pass to constraint function.
    """
    constraint_func: Callable = ConfigField(
        title="Constraint Func",
        description="Constraint function to evaluate",
    )
    constraint_config: BaseConfig = ConfigField(
        title="Constraint Config",
        description="Constraint config to pass to constraint function"
    )

@ConstraintRegistry.register(
    key="evo2_rna_constraint_wrapper",
    label="Evo 2 RNA Constraint Wrapper",
    config=Evo2RNAConstraintConfig,
    description="Wrap RNA constraints to handle Evo 2 ICL outputs",
    supported_sequence_types=["dna"],
    batched=True,
    multi_input=False,
    gpu_required=False,
    tools_called=["viennarna"],
    category="rna_secondary_structure",
)
def _evo2_rna_constraint_wrapper(
    sequences: List[Sequence],
    config: Evo2RNAConstraintConfig,
) -> List[float]:
    """
    Wrap the RNA secondary structure constraints to ensure that the outputs of
    Evo 2 are compatible. For example, remove the ICL sep tokens and turn the DNA
    into an RNA sequence ('T' -> 'U').
    """
    # Process Evo 2 output into RNA.
    new_sequences = []
    for idx, sequence in enumerate(sequences):
        new_seq = sequence.sequence.split(SEP_SEQUENCE)[0].upper().replace('T', 'U')
        new_sequences.append(Sequence(
            sequence=new_seq,
            sequence_type='rna',
        ))

    # Actually evaluate the constraint.
    results = config.constraint_func(new_sequences, config.constraint_config)

    # Propagate metadata.
    for sequence, new_sequence in zip(sequences, new_sequences):
        sequence.metadata.update(new_sequence.metadata)

    return results

class RNAPairwiseSimilarityConfig(BaseConfig):
    """
    Compare RNA sequences against a reference sequence.

    Attributes:
        constraint_func (Callable): Constraint function to evaluate.
        constraint_config (BaseConfig): Constraint config to pass to constraint function.
    """
    reference_sequence: str = ConfigField(
        title="Reference Sequence",
        description="Compare RNA sequences to this sequence",
    )

@ConstraintRegistry.register(
    key="rna_pairwise_similarity",
    label="RNA pairwise similarity",
    config=RNAPairwiseSimilarityConfig,
    description="Compare RNA sequences to a reference sequence",
    supported_sequence_types=["rna"],
    batched=True,
    multi_input=False,
    gpu_required=False,
    category="sequence_composition",
)
def rna_pairwise_similarity_constraint(
    sequences: List[Sequence],
    config: RNAPairwiseSimilarityConfig,
) -> List[float]:
    """
    Compute similarity of candidate RNA sequences to a reference.

    Greater novelty (lower similarity) is considered better.
    """
    results = []
    for sequence in sequences:
        similarity = _compute_rna_similarity(sequence.sequence, config.reference_sequence)
        if similarity < 0.7:
            results.append(float('inf'))
        else:
            results.append(similarity)
    return results


def create_u5_snrna_program() -> Program:
    """
    Make the U5 diversification program. Balance diversification with structural similarity.
    """
    # Construct the prompt.

    records = islice(SeqIO.parse(U5_PROMPT_FASTA, "fasta"), N_PROMPT_EXAMPLES)
    prompt_seqs = [str(record.seq) for record in records]
    u5_icl_prompt = SEP_SEQUENCE.join(prompt_seqs) + SEP_SEQUENCE

    n_tokens_to_sample = max(len(seq) for seq in prompt_seqs) + len(SEP_SEQUENCE) + 2

    # =============================================================================
    # Segments, Constructs, and Generators
    # =============================================================================

    u5_snrna = Segment(
        length=n_tokens_to_sample,
        sequence_type="dna",  # Needed for Evo 2, will convert to RNA at constraint time.
    )

    u5_snrna_construct = Construct([u5_snrna])

    evo2_config = Evo2GeneratorConfig(
        prompts=[u5_icl_prompt] * N_SAMPLES,
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
    evo2_generator.assign(u5_snrna)

    # =============================================================================
    # Constraints
    # =============================================================================

    constraints = []

    constraint_property_similarity = Constraint(
        inputs=[u5_snrna],
        function=_evo2_rna_constraint_wrapper,
        function_config={
            "constraint_func": rna_property_similarity_constraint,
            "constraint_config": RNAPropertySimilarityConfig(reference_sequence=U5_SNRNA),
        },
        threshold=0.7,
        label='rna_property_similarity_constraint_evo2',
    )
    constraints.append(constraint_property_similarity)

    constraint_motif_similarity = Constraint(
        inputs=[u5_snrna],
        function=_evo2_rna_constraint_wrapper,
        function_config={
            "constraint_func": rna_motif_similarity_constraint,
            "constraint_config": RNAMotifSimilarityConfig(reference_sequence=U5_SNRNA),
        },
        threshold=0.8,
        label='rna_motif_similarity_constraint_evo2',
    )
    constraints.append(constraint_motif_similarity)

    constraint_feature_similarity = Constraint(
        inputs=[u5_snrna],
        function=_evo2_rna_constraint_wrapper,
        function_config={
            "constraint_func": rna_feature_similarity_constraint,
            "constraint_config": RNAFeatureSimilarityConfig(reference_sequence=U5_SNRNA),
        },
        threshold=0.6,
        label='rna_feature_similarity_constraint_evo2',
    )
    constraints.append(constraint_feature_similarity)

    constraint_basepair_similarity = Constraint(
        inputs=[u5_snrna],
        function=_evo2_rna_constraint_wrapper,
        function_config={
            "constraint_func": rna_basepair_similarity_constraint,
            "constraint_config": RNABasePairSimilarityConfig(reference_sequence=U5_SNRNA),
        },
        threshold=0.3,
        label='rna_basepair_similarity_constraint_evo2',
    )
    constraints.append(constraint_basepair_similarity)

    constraint_wt_similarity = Constraint(
        inputs=[u5_snrna],
        function=_evo2_rna_constraint_wrapper,
        function_config={
            "constraint_func": rna_pairwise_similarity_constraint,
            "constraint_config": RNAPairwiseSimilarityConfig(reference_sequence=U5_SNRNA),
        },
        weight=4,  # Equal weight as diversification constraints.
        label='rna_pairwise_similarity_constraint_evo2',
    )
    constraints.append(constraint_wt_similarity)

    # =============================================================================
    # Program modification
    # =============================================================================

    top_k_optimizer_config = TopKOptimizerConfig(
        num_samples=N_SAMPLES,
        num_results=1,
        batch_size=N_SAMPLES,
        verbose=True,
    )
    u5_snrna_optimizer = TopKOptimizer(
        constructs=[u5_snrna_construct],
        generators=[evo2_generator],
        constraints=constraints,
        config=top_k_optimizer_config,
    )

    u5_snrna_program = Program(optimizers=[u5_snrna_optimizer], num_results=1)

    return u5_snrna_program


if __name__ == '__main__':
    program = create_u5_snrna_program()
    program.run()
