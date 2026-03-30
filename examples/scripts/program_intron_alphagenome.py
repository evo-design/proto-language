import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from proto_tools.utils.tool_instance import ToolInstance
from tap import Tap

from proto_language.language.constraint import (
    splice_transformer_intron_boundary,
    splice_transformer_specificity,
)
from proto_language.language.constraint.rna_splicing.alphagenome_splice_site_usage import (
    AlphaGenomeSpliceSiteUsageConfig,
    alphagenome_splice_site_usage,
)
from proto_language.language.core import Constraint, Construct, Program, Segment
from proto_language.language.generator import (
    Evo2Generator,
    Evo2GeneratorConfig,
    MaskingStrategy,
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.language.optimizer import (
    MCMCOptimizer,
    MCMCOptimizerConfig,
    TopKOptimizer,
    TopKOptimizerConfig,
)
from examples.scripts.program_intron_design import (
    INTRON_LENGTH,
    N_STEPS,
    SAMPLES_PER_ROUND,
    _enable_mcmc_energy_logging,
    _get_constraints_metadata,
    get_initial_intron,
    process_splice_transformer_input,
)

DEFAULT_GENOMIC_CONTEXT_PATHS = ",".join(
    [
        "examples/data/alphagenome_context_aavs1.txt",
        "examples/data/alphagenome_context_ccr5.txt",
        "examples/data/alphagenome_context_clybl.txt",
        "examples/data/alphagenome_context_hrosa26.txt",
    ]
)

DEFAULT_CELL_ONTOLOGY_TERMS = {
    # SH-SY5Y / EFO:0002717 is the experimental target, but AlphaGenome currently
    # does not support it robustly for all relevant outputs, so we use CL:0002319
    # (neural cell) proxy terms.
    # Unknown aliases still fail early via _resolve_terms.
    "shsy5y": ["CL:0002319"],
    "sh-sy5y": ["CL:0002319"],
    "hepg2": ["EFO:0001187"],
    "k562": ["EFO:0002067"],
}


class ProgramIntronAlphaGenomeArgs(Tap):
    intron_length: int = INTRON_LENGTH
    n_steps: int = N_STEPS
    step_size: int = 1
    temperature: float = 1.0
    temperature_min: float = 0.001
    plasmid_context_path: str = "examples/data/plasmid_context_cmv_20260308.txt"
    gene_sequence_path: str = "examples/data/mscarlet_ires_zsgreen.txt"
    gene_insertion_pos: int = 159 * 3
    initialization: str = "random"
    intron_generator: str = "uniform"
    multicontext: bool = True
    specificity_type: str = "max_brain_min_blood"

    enable_alphagenome: bool = True
    enable_splice_transformer: bool = True
    enable_splice_specificity: bool = True

    target_cell: str = "shsy5y"
    offtarget_cell: str = "k562"
    target_ontology_terms: str = ""
    offtarget_ontology_terms: str = ""

    genomic_context_paths: str = DEFAULT_GENOMIC_CONTEXT_PATHS
    alphagenome_model_version: str = "all_folds"
    alphagenome_organism: Literal["human", "mouse"] = "human"
    alphagenome_device: str = "cuda"
    alphagenome_prediction_timeout: int = 3600
    alphagenome_track_strand: Literal["positive", "negative", "all"] = "positive"
    alphagenome_brain_weight: float = 1.0
    alphagenome_blood_weight: float = 1.0

    splice_transformer_device: str = "cuda"

    mcmc_num_results: int = 1
    mcmc_candidates_per_result: int = 1


def _split_csv(arg: str) -> List[str]:
    return [token.strip() for token in arg.split(",") if token.strip()]


def _resolve_terms(
    cell_name: str,
    override_terms_csv: str,
    defaults: Dict[str, List[str]],
) -> List[str]:
    override_terms = _split_csv(override_terms_csv)
    if override_terms:
        return override_terms
    key = cell_name.strip().lower()
    if key in defaults:
        return defaults[key]
    if ":" in cell_name:
        # Allow directly passing ontology terms through --target_cell/--offtarget_cell.
        return [cell_name.strip()]
    if key:
        raise ValueError(
            f"Unsupported cell alias '{cell_name}'. Provide a known alias or explicit ontology term."
        )
    raise ValueError("Cell name/ontology term cannot be empty.")


def _read_context_sequence(path: str) -> str:
    sequence = Path(path).read_text().strip().upper()
    if not sequence:
        raise ValueError(f"Context file is empty: {path}")
    invalid_chars = set(sequence) - set("ACGTN")
    if invalid_chars:
        raise ValueError(
            f"Context file contains invalid DNA characters {sorted(invalid_chars)}: {path}"
        )
    return sequence


if __name__ == "__main__":
    args = ProgramIntronAlphaGenomeArgs(explicit_bool=True).parse_args()
    _enable_mcmc_energy_logging()

    print(args)
    if args.mcmc_num_results < 1:
        raise ValueError(f"mcmc_num_results must be >= 1, got {args.mcmc_num_results}")
    if args.mcmc_candidates_per_result < 1:
        raise ValueError(
            "mcmc_candidates_per_result must be >= 1, "
            f"got {args.mcmc_candidates_per_result}"
        )
    if args.alphagenome_brain_weight <= 0:
        raise ValueError("alphagenome_brain_weight must be > 0")
    if args.alphagenome_blood_weight <= 0:
        raise ValueError("alphagenome_blood_weight must be > 0")

    initial_intron = get_initial_intron(args)

    if args.multicontext:
        plasmid_paths = [
            "examples/data/plasmid_context_cmv_20260308.txt",
            "examples/data/plasmid_context_Ef1a.txt",
            "examples/data/plasmid_context_sffv.txt",
        ]
    else:
        plasmid_paths = [args.plasmid_context_path]

    if args.multicontext:
        genomic_context_paths = _split_csv(args.genomic_context_paths)
    else:
        parsed_paths = _split_csv(args.genomic_context_paths)
        genomic_context_paths = [parsed_paths[0]] if parsed_paths else []
    genomic_contexts = [
        (Path(path).stem, path, _read_context_sequence(path)) for path in genomic_context_paths
    ]

    target_ontology_terms = _resolve_terms(
        args.target_cell,
        args.target_ontology_terms,
        DEFAULT_CELL_ONTOLOGY_TERMS,
    )
    offtarget_ontology_terms = _resolve_terms(
        args.offtarget_cell,
        args.offtarget_ontology_terms,
        DEFAULT_CELL_ONTOLOGY_TERMS,
    )

    intron = None
    intron_construct = None
    intron_gen = None
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

        print("intron range (start, end):", (donor_start_pos_in_target, acceptor_end_pos_in_target + 1))

        assert acceptor_end_pos_in_target - donor_start_pos_in_target + 1 == len(initial_intron)

        if intron is None:
            intron = Segment(
                sequence=target_seq[donor_start_pos_in_target + 2 : acceptor_end_pos_in_target - 1],
                sequence_type="dna",
            )

            if args.intron_generator == "uniform":
                intron_gen_config = RandomNucleotideGeneratorConfig(
                    masking_strategy=MaskingStrategy(num_mutations=args.step_size),
                )
                intron_gen = RandomNucleotideGenerator(intron_gen_config)
                intron_gen.assign(intron)
            elif args.intron_generator == "evo2":
                from Bio import SeqIO

                record = SeqIO.read("examples/data/hbb_intron2_prompt.txt", "fasta")
                hg38_prompt = str(record.seq)
                intron_gen_config = Evo2GeneratorConfig(
                    prompts=[hg38_prompt],
                    top_k=4,
                    temperature=1.0,
                )
                intron_gen = Evo2Generator(intron_gen_config)
                intron_gen.sequence_length = args.intron_length - 4
                intron_gen.assign(intron)
            else:
                raise ValueError(f"Unsupported intron generator {args.intron_generator}")
        else:
            assert (
                target_seq[donor_start_pos_in_target + 2 : acceptor_end_pos_in_target - 1]
                == intron.original_sequence._sequence
            )

        left_flank = Segment(
            sequence=target_seq[: donor_start_pos_in_target + 2],
            sequence_type="dna",
        )
        right_flank = Segment(
            sequence=target_seq[acceptor_end_pos_in_target - 1 :],
            sequence_type="dna",
        )

        if intron_construct is None:
            intron_construct = Construct([left_flank, intron, right_flank])

        donor_eval_pos = donor_start_pos_in_target - 1
        acceptor_eval_pos = acceptor_end_pos_in_target + 1
        splice_pos_all = [donor_eval_pos, acceptor_eval_pos]
        donor_pos_all = [donor_eval_pos]
        acceptor_pos_all = [acceptor_eval_pos]
        context_label = Path(plasmid_path).stem

        if args.enable_splice_transformer:
            intron_boundary = Constraint(
                inputs=[left_flank, intron, right_flank],
                function=splice_transformer_intron_boundary,
                function_config={
                    "left_context": left_context,
                    "right_context": right_context,
                    "donor_pos": donor_pos_all,
                    "acceptor_pos": acceptor_pos_all,
                    "splice_transformer_config": {
                        "device": args.splice_transformer_device,
                    },
                },
                label=f"splice_boundary__{context_label}__{context_idx}",
            )
            all_constraints.append(intron_boundary)

            if args.enable_splice_specificity:
                if "max_brain" in args.specificity_type:
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=splice_transformer_specificity,
                            function_config={
                                "left_context": left_context,
                                "right_context": right_context,
                                "splice_pos": splice_pos_all,
                                "tissue": "BRAIN",
                                "direction": "max",
                                "splice_transformer_config": {
                                    "device": args.splice_transformer_device,
                                },
                            },
                            label=f"splice_specificity_brain_max__{context_label}__{context_idx}",
                        )
                    )
                if "max_liver" in args.specificity_type:
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=splice_transformer_specificity,
                            function_config={
                                "left_context": left_context,
                                "right_context": right_context,
                                "splice_pos": splice_pos_all,
                                "tissue": "LIVER",
                                "direction": "max",
                                "splice_transformer_config": {
                                    "device": args.splice_transformer_device,
                                },
                            },
                            label=f"splice_specificity_liver_max__{context_label}__{context_idx}",
                        )
                    )
                if "min_brain" in args.specificity_type:
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=splice_transformer_specificity,
                            function_config={
                                "left_context": left_context,
                                "right_context": right_context,
                                "splice_pos": splice_pos_all,
                                "tissue": "BRAIN",
                                "direction": "min",
                                "splice_transformer_config": {
                                    "device": args.splice_transformer_device,
                                },
                            },
                            label=f"splice_specificity_brain_min__{context_label}__{context_idx}",
                        )
                    )
                if "min_liver" in args.specificity_type:
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=splice_transformer_specificity,
                            function_config={
                                "left_context": left_context,
                                "right_context": right_context,
                                "splice_pos": splice_pos_all,
                                "tissue": "LIVER",
                                "direction": "min",
                                "splice_transformer_config": {
                                    "device": args.splice_transformer_device,
                                },
                            },
                            label=f"splice_specificity_liver_min__{context_label}__{context_idx}",
                        )
                    )
                if "max_blood" in args.specificity_type:
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=splice_transformer_specificity,
                            function_config={
                                "left_context": left_context,
                                "right_context": right_context,
                                "splice_pos": splice_pos_all,
                                "tissue": "BLOOD",
                                "direction": "max",
                                "splice_transformer_config": {
                                    "device": args.splice_transformer_device,
                                },
                            },
                            label=f"splice_specificity_blood_max__{context_label}__{context_idx}",
                        )
                    )
                if "min_blood" in args.specificity_type:
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=splice_transformer_specificity,
                            function_config={
                                "left_context": left_context,
                                "right_context": right_context,
                                "splice_pos": splice_pos_all,
                                "tissue": "BLOOD",
                                "direction": "min",
                                "splice_transformer_config": {
                                    "device": args.splice_transformer_device,
                                },
                            },
                            label=f"splice_specificity_blood_min__{context_label}__{context_idx}",
                        )
                    )

        if args.enable_alphagenome:
            if not genomic_contexts:
                raise ValueError(
                    "AlphaGenome scoring enabled but no genomic context files were provided."
                )

            if not any(
                token in args.specificity_type
                for token in (
                    "max_brain",
                    "min_brain",
                    "max_liver",
                    "min_liver",
                    "max_blood",
                    "min_blood",
                )
            ):
                raise ValueError(
                    "specificity_type must include at least one of "
                    "max_brain/min_brain/max_liver/min_liver/max_blood/min_blood "
                    "when --enable_alphagenome true."
                )

            for genomic_idx, (genomic_label, _, genomic_context) in enumerate(genomic_contexts):
                common_kwargs = dict(
                    genomic_context=genomic_context,
                    cassette_left_context=left_context,
                    cassette_right_context=right_context,
                    splice_pos=splice_pos_all,
                    model_version=args.alphagenome_model_version,
                    organism=args.alphagenome_organism,
                    device=args.alphagenome_device,
                    prediction_timeout=args.alphagenome_prediction_timeout,
                    strand=args.alphagenome_track_strand,
                )

                if "max_brain" in args.specificity_type:
                    brain_max_cfg = AlphaGenomeSpliceSiteUsageConfig(
                        ontology_terms=target_ontology_terms,
                        direction="max",
                        **common_kwargs,
                    )
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=alphagenome_splice_site_usage,
                            function_config=brain_max_cfg,
                            weight=args.alphagenome_brain_weight,
                            label=(
                                f"alphagenome_ssu_brain_max__{context_label}__{genomic_label}"
                                f"__{context_idx}_{genomic_idx}"
                            ),
                        )
                    )

                if "min_brain" in args.specificity_type:
                    brain_min_cfg = AlphaGenomeSpliceSiteUsageConfig(
                        ontology_terms=target_ontology_terms,
                        direction="min",
                        **common_kwargs,
                    )
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=alphagenome_splice_site_usage,
                            function_config=brain_min_cfg,
                            weight=args.alphagenome_brain_weight,
                            label=(
                                f"alphagenome_ssu_brain_min__{context_label}__{genomic_label}"
                                f"__{context_idx}_{genomic_idx}"
                            ),
                        )
                    )

                if "max_liver" in args.specificity_type:
                    liver_max_cfg = AlphaGenomeSpliceSiteUsageConfig(
                        ontology_terms=target_ontology_terms,
                        direction="max",
                        **common_kwargs,
                    )
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=alphagenome_splice_site_usage,
                            function_config=liver_max_cfg,
                            weight=args.alphagenome_brain_weight,
                            label=(
                                f"alphagenome_ssu_liver_max__{context_label}__{genomic_label}"
                                f"__{context_idx}_{genomic_idx}"
                            ),
                        )
                    )

                if "min_liver" in args.specificity_type:
                    liver_min_cfg = AlphaGenomeSpliceSiteUsageConfig(
                        ontology_terms=target_ontology_terms,
                        direction="min",
                        **common_kwargs,
                    )
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=alphagenome_splice_site_usage,
                            function_config=liver_min_cfg,
                            weight=args.alphagenome_brain_weight,
                            label=(
                                f"alphagenome_ssu_liver_min__{context_label}__{genomic_label}"
                                f"__{context_idx}_{genomic_idx}"
                            ),
                        )
                    )

                if "max_blood" in args.specificity_type:
                    blood_max_cfg = AlphaGenomeSpliceSiteUsageConfig(
                        ontology_terms=offtarget_ontology_terms,
                        direction="max",
                        **common_kwargs,
                    )
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=alphagenome_splice_site_usage,
                            function_config=blood_max_cfg,
                            weight=args.alphagenome_blood_weight,
                            label=(
                                f"alphagenome_ssu_blood_max__{context_label}__{genomic_label}"
                                f"__{context_idx}_{genomic_idx}"
                            ),
                        )
                    )

                if "min_blood" in args.specificity_type:
                    blood_min_cfg = AlphaGenomeSpliceSiteUsageConfig(
                        ontology_terms=offtarget_ontology_terms,
                        direction="min",
                        **common_kwargs,
                    )
                    all_constraints.append(
                        Constraint(
                            inputs=[left_flank, intron, right_flank],
                            function=alphagenome_splice_site_usage,
                            function_config=blood_min_cfg,
                            weight=args.alphagenome_blood_weight,
                            label=(
                                f"alphagenome_ssu_blood_min__{context_label}__{genomic_label}"
                                f"__{context_idx}_{genomic_idx}"
                            ),
                        )
                    )

    if intron_construct is None or intron_gen is None:
        raise RuntimeError("No intron construct/generator was created.")
    if not all_constraints:
        raise RuntimeError("No constraints were configured. Enable at least one scoring path.")

    optimizer_for_logging: Dict[str, Optional[MCMCOptimizer]] = {"instance": None}

    def _format_intron_sequence(left: str, intron_core: str, right: str) -> str:
        if len(left) >= 2 and len(right) >= 2:
            return left[-2:] + intron_core + right[:2]
        return intron_core

    def _format_energy(value: float) -> str:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f"{float(value):.6f}"
        return str(value)

    def custom_logging(step: int, outputs: Tuple[Segment]) -> None:
        print(f"\tstep: {step}")
        optimizer_instance = optimizer_for_logging["instance"]
        num_results = len(outputs[1].result_sequences)
        candidate_energies = (
            list(getattr(optimizer_instance, "_candidate_energy_scores", []))
            if optimizer_instance is not None
            else []
        )
        candidate_outcomes = (
            list(getattr(optimizer_instance, "_candidate_outcomes", []))
            if optimizer_instance is not None
            else []
        )
        candidates_per_result = (
            int(getattr(optimizer_instance, "_proposals_per_result", 1))
            if optimizer_instance is not None
            else 1
        )

        for result_idx in range(num_results):
            left_flank_sequence = str(outputs[0].result_sequences[result_idx]._sequence)
            intron_core_sequence = str(outputs[1].result_sequences[result_idx]._sequence)
            right_flank_sequence = str(outputs[2].result_sequences[result_idx]._sequence)
            intron_sequence = _format_intron_sequence(
                left_flank_sequence, intron_core_sequence, right_flank_sequence
            )

            result_energy = float("nan")
            if optimizer_instance is not None and result_idx < len(optimizer_instance.energy_scores):
                result_energy = optimizer_instance.energy_scores[result_idx]

            print(
                f"\tresult[{result_idx}] energy: {_format_energy(result_energy)}\n"
                f"\tresult[{result_idx}] sequence (left_flank): {left_flank_sequence}\n"
                f"\tresult[{result_idx}] sequence (intron): {intron_sequence}\n"
                f"\tresult[{result_idx}] sequence (right_flank): {right_flank_sequence}"
            )

            proposal_pool_start = result_idx * candidates_per_result
            proposal_pool_end = min(
                proposal_pool_start + candidates_per_result,
                len(outputs[1].proposal_sequences),
            )
            print(
                f"\tresult[{result_idx}] candidate_pool_size: "
                f"{proposal_pool_end - proposal_pool_start}"
            )
            for candidate_idx in range(proposal_pool_start, proposal_pool_end):
                local_idx = candidate_idx - proposal_pool_start
                candidate_left = str(outputs[0].proposal_sequences[candidate_idx]._sequence)
                candidate_intron_core = str(
                    outputs[1].proposal_sequences[candidate_idx]._sequence
                )
                candidate_right = str(outputs[2].proposal_sequences[candidate_idx]._sequence)
                candidate_intron = _format_intron_sequence(
                    candidate_left,
                    candidate_intron_core,
                    candidate_right,
                )
                candidate_energy = (
                    candidate_energies[candidate_idx]
                    if candidate_idx < len(candidate_energies)
                    else float("nan")
                )
                candidate_outcome = (
                    candidate_outcomes[candidate_idx]
                    if candidate_idx < len(candidate_outcomes)
                    else "unknown"
                )
                print(
                    f"\t\tcandidate[{result_idx}:{local_idx}] "
                    f"global_idx={candidate_idx} "
                    f"outcome={candidate_outcome} "
                    f"energy={_format_energy(candidate_energy)}\n"
                    f"\t\tcandidate[{result_idx}:{local_idx}] sequence (intron): {candidate_intron}"
                )

            constraints = _get_constraints_metadata(outputs[1].result_sequences[result_idx])
            for constraint_label in sorted(constraints):
                constraint_data = constraints[constraint_label]
                metric_data = constraint_data.get("data", {})
                for metric_name in sorted(metric_data):
                    metric_value = metric_data[metric_name]
                    if (
                        metric_name
                        in {
                            "donor_score",
                            "acceptor_score",
                            "total_splice_score",
                            "ontology_terms",
                            "splice_pos",
                            "direction",
                            "strand",
                            "selected_track_count",
                            "selected_track_names",
                            "selected_track_strands",
                            "alphagenome_splice_site_usage_raw",
                            "alphagenome_splice_site_usage_score",
                        }
                        or metric_name.startswith("specificity_direction")
                        or metric_name.startswith("specificity_score")
                    ):
                        print(
                            f"\tresult[{result_idx}] {constraint_label}: "
                            f"{metric_name}: {metric_value}"
                        )

    if args.intron_generator == "evo2":
        optimizer_config = TopKOptimizerConfig(
            num_samples=args.n_steps,
            num_results=args.n_steps,
            samples_per_round=SAMPLES_PER_ROUND,
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
            num_steps=args.n_steps,
            num_results=args.mcmc_num_results,
            proposals_per_result=args.mcmc_candidates_per_result,
            max_temperature=args.temperature,
            min_temperature=args.temperature_min,
            track_proposals=True,
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
        optimizer_for_logging["instance"] = optimizer

    program_num_results = args.mcmc_num_results if args.intron_generator != "evo2" else 1
    program = Program(optimizers=[optimizer], num_results=program_num_results)

    with ToolInstance.persist():
        program.run()
