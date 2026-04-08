"""Run a proto-language program from a JSON specification file.

Usage:
    python examples/scripts/run_program.py examples/jsons/toy.json
    python examples/scripts/run_program.py examples/jsons/toy.json --verbose
"""

import argparse
import json
import sys
from typing import Any

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.core import Construct, Program, Segment
from proto_language.language.generator import GeneratorRegistry
from proto_language.language.optimizer import OptimizerRegistry


def parse_program(json_data: dict[str, Any]) -> Program:
    """Parse a JSON program specification into an executable Program."""
    segment_lookup: dict[str, Segment] = {}

    # Parse constructs
    constructs: list[Construct] = []
    for construct_json in json_data["constructs"]:
        segments: list[Segment] = []
        sequence_type = construct_json["type"].lower()
        if sequence_type not in ("dna", "rna", "protein"):
            raise ValueError(f"Invalid construct type: {sequence_type}")

        for segment_json in construct_json["segments"]:
            segment_id = segment_json["id"]
            if segment_id in segment_lookup:
                raise ValueError(f"Duplicate segment id: '{segment_id}'")

            seg = Segment(
                sequence=segment_json.get("sequence"),
                length=segment_json.get("length"),
                sequence_type=sequence_type,
                label=segment_json.get("label"),
            )
            segment_lookup[segment_id] = seg
            segments.append(seg)
        constructs.append(Construct(segments, label=construct_json.get("label")))

    # Parse optimization stages
    optimizers = []
    for idx, stage_json in enumerate(json_data["optimization_stages"]):
        opt_config = stage_json["optimizer"]
        method_key = opt_config["method"].lower()
        config_dict = opt_config.get("config", {})

        optimizer_spec = OptimizerRegistry.get(method_key)
        optimizer_config = optimizer_spec.config_model(**config_dict)

        # Parse generators
        stage_generators = []
        for gen_json in stage_json["generators"]:
            generator = GeneratorRegistry.create(
                key=gen_json["key"],
                config_dict=gen_json.get("config", {}),
            )
            target_id = gen_json["target"]
            if target_id not in segment_lookup:
                raise ValueError(f"Generator target '{target_id}' not found (stage {idx})")
            generator.assign(segment_lookup[target_id])
            stage_generators.append(generator)

        # Parse constraints
        stage_constraints = []
        for con_json in stage_json["constraints"]:
            target_ids = con_json["targets"]
            segments = []
            for tid in target_ids:
                if tid not in segment_lookup:
                    raise ValueError(f"Constraint target '{tid}' not found (stage {idx})")
                segments.append(segment_lookup[tid])

            constraint = ConstraintRegistry.create(
                key=con_json["key"],
                segments=segments,
                config_dict=con_json.get("config", {}),
                label=con_json.get("label"),
                threshold=con_json.get("threshold"),
                weight=con_json.get("weight"),
            )
            stage_constraints.append(constraint)

        # Build optimizer
        optimizer_kwargs: dict[str, Any] = {
            "constructs": constructs,
            "generators": stage_generators,
            "constraints": stage_constraints,
            "config": optimizer_config,
        }

        if optimizer_spec.targets_single_segment:
            target_segment_id = opt_config.get("target_segment")
            if not target_segment_id or target_segment_id not in segment_lookup:
                raise ValueError(
                    f"Optimizer '{method_key}' requires valid 'target_segment' (stage {idx})"
                )
            optimizer_kwargs["target_segment"] = segment_lookup[target_segment_id]

        optimizer = optimizer_spec.optimizer_class(**optimizer_kwargs)
        optimizers.append(optimizer)

    return Program(
        optimizers=optimizers,
        num_results=json_data.get("num_results", 1),
        verbose=json_data.get("verbose", False),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a proto-language JSON program")
    parser.add_argument("json_file", help="Path to the JSON program file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    with open(args.json_file) as f:
        json_data = json.load(f)

    program_name = json_data.get("name", "unnamed")
    print(f"Loading program: {program_name}")
    if json_data.get("description"):
        print(f"Description: {json_data['description']}")

    if args.verbose:
        json_data["verbose"] = True

    program = parse_program(json_data)
    print(f"Parsed {len(program.optimizers)} optimization stage(s)")

    print("\nRunning program...")
    program.run()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for i, construct in enumerate(program.constructs):
        for j, seq in enumerate(construct.joined_sequences):
            print(f"\nConstruct {i}, Result {j}:")
            print(f"  Sequence: {seq.sequence[:100]}{'...' if len(seq.sequence) > 100 else ''}")
            print(f"  Length: {len(seq.sequence)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
