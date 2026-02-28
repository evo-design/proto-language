#!/usr/bin/env python3
"""Run proto-language JSON programs locally."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from api.core.parser import ProtoParser


def main():
    arg_parser = argparse.ArgumentParser(description="Run JSON programs locally")
    arg_parser.add_argument("program", help="Path to JSON program file")
    arg_parser.add_argument("-o", "--output", help="Output file (JSON)")
    args = arg_parser.parse_args()

    # Load and parse
    with open(args.program) as f:
        json_data = json.load(f)

    print(f"Running: {json_data.get('name', 'Unnamed')}")

    proto_parser = ProtoParser(json_data)
    program = proto_parser.parse()
    program.run()

    # Extract results
    results = {
        "program": json_data.get("name", "Unnamed"),
        "constructs": [
            {
                "segments": [
                    {
                        "label": seg.label,
                        "result_sequences": [
                            {
                                "sequence": seq.sequence,
                                "length": len(seq),
                                "metadata": seq.metadata,
                            }
                            for seq in seg.result_sequences
                        ]
                    }
                    for seg in construct.segments
                ]
            }
            for construct in program.constructs
        ]
    }

    for i, construct in enumerate(results['constructs'], 1):
        print(f"\nConstruct {i}:")
        for seg in construct['segments']:
            print(f"  {seg['label']}:")
            for j, seq_data in enumerate(seg['result_sequences']):
                seq = seq_data['sequence']
                display = seq if len(seq) <= 80 else f"{seq[:80]}..."
                print(f"    [{j}] {display} ({seq_data['length']} bp)")

    # Optional save to output file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: str(x))
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
