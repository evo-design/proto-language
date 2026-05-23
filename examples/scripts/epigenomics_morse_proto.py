#!/usr/bin/env python3
"""Design a DNA insert whose predicted chromatin-accessibility profile spells Morse code.

This script builds a left-flank/target/right-flank construct, generates candidate
target segments with Evo2 beam search, and scores complete constructs with Borzoi
and Enformer Morse constraints. The objective rewards high predicted
accessibility in dash and dot windows and low accessibility in gap windows. It is
a full example of composing sequence generation, epigenomic scoring, and
window-level logging for a synthetic regulatory-design task.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_TOOLS_ROOT = REPO_ROOT / "proto-tools"
for path in (str(PROTO_TOOLS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from proto_language.constraint import (
    BorzoiChromatinAccessibilityMorseConfig,
    EnformerChromatinAccessibilityMorseConfig,
    borzoi_chromatin_accessibility_morse_constraint,
    enformer_chromatin_accessibility_morse_constraint,
)
from proto_language.constraint.sequence_annotation.chromatin_accessibility_morse_utils import (
    compute_morse_windows,
)
from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.optimizer import BeamSearchOptimizer, BeamSearchOptimizerConfig

logger = logging.getLogger(__name__)

Organism = Literal["mouse", "human"]
ScoreBy = Literal["mean", "last"]
ReduceMethod = Literal["mean", "min", "std", "lcb"]
PatternNormalization = Literal["global_max", "region_max", "none"]
WindowStatTransform = Literal["log1p", "identity"]

ORGANISMS: tuple[Organism, ...] = ("mouse", "human")
SCORE_BY_OPTIONS: tuple[ScoreBy, ...] = ("mean", "last")
REDUCE_METHODS: tuple[ReduceMethod, ...] = ("mean", "min", "std", "lcb")
PATTERN_NORMALIZATIONS: tuple[PatternNormalization, ...] = ("global_max", "region_max", "none")
WINDOW_STAT_TRANSFORMS: tuple[WindowStatTransform, ...] = ("log1p", "identity")


@dataclass(frozen=True)
class RuntimeConfig:
    left_flank_path: Path
    right_flank_path: Path
    output_dir: Path
    intermediate_output_dir: Path | None
    organism: Organism
    pattern: str
    dot_bp: int
    dash_bp: int
    intra_symbol_gap_bp: int
    inter_letter_gap_bp: int
    pattern_start_bp: int
    target_length: int
    evo_device: str
    scoring_device: str
    evo_prompt: str | None
    evo_prompt_context_bp: int
    evo_temperature: float
    evo_top_k: int
    evo_top_p: float
    stop_at_eos: bool
    use_kv_caching: bool
    force_prompt_threshold: int | None
    beam_length: int
    beam_width: int
    candidates_per_beam: int
    beam_batch_size: int
    score_by: ScoreBy
    borzoi_tracks: tuple[int, ...] | None
    enformer_tracks: tuple[int, ...] | None
    borzoi_ensemble_reduce_method: ReduceMethod
    enformer_track_reduce_method: ReduceMethod
    scoring_batch_size: int
    pattern_normalization: PatternNormalization
    contrast_margin: float
    contrast_weight: float
    raw_amplitude_weight: float
    high_window_reward_weight: float
    low_window_penalty_weight: float
    window_stat_transform: WindowStatTransform


def parse_args(argv: list[str] | None = None) -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Run the epigenomics Morse beam-search pipeline.")

    io = parser.add_argument_group("I/O")
    io.add_argument("--left-flank", type=Path, required=True, help="Left flank FASTA path.")
    io.add_argument("--right-flank", type=Path, required=True, help="Right flank FASTA path.")
    io.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    io.add_argument(
        "--intermediate-output-dir",
        type=Path,
        default=None,
        help="Optional directory for selected beam target FASTAs written at each tracked beam.",
    )

    seq = parser.add_argument_group("Sequence geometry")
    seq.add_argument("--organism", choices=ORGANISMS, default="mouse")
    seq.add_argument("--pattern", default=".--. .-. --- - ---")
    seq.add_argument("--dot-bp", type=int, default=384)
    seq.add_argument("--dash-bp", type=int, default=1152)
    seq.add_argument("--intra-gap-bp", type=int, default=384)
    seq.add_argument("--inter-gap-bp", type=int, default=1152)
    seq.add_argument("--pattern-start-bp", type=int, default=0)
    seq.add_argument("--target-length", type=int, default=21120)

    evo = parser.add_argument_group("Evo2")
    evo.add_argument("--evo-device", default="cuda:0")
    evo.add_argument("--scoring-device", default=None, help="Defaults to --evo-device if omitted.")
    evo.add_argument("--evo-prompt", default=None, help="Optional explicit Evo2 prompt.")
    evo.add_argument("--evo-prompt-context-bp", type=int, default=8192)
    evo.add_argument("--evo-temperature", type=float, default=1.0)
    evo.add_argument("--evo-top-k", type=int, default=4)
    evo.add_argument("--evo-top-p", type=float, default=1.0)
    evo.add_argument("--stop-at-eos", action=argparse.BooleanOptionalAction, default=False)
    evo.add_argument("--use-kv-caching", action=argparse.BooleanOptionalAction, default=True)
    evo.add_argument("--force-prompt-threshold", type=int, default=1)

    beam = parser.add_argument_group("Beam search")
    beam.add_argument("--beam-length", type=int, default=128)
    beam.add_argument("--beam-width", type=int, default=2)
    beam.add_argument("--candidates-per-beam", type=int, default=18)
    beam.add_argument("--beam-batch-size", type=int, default=18)
    beam.add_argument("--score-by", choices=SCORE_BY_OPTIONS, default="last")

    scoring = parser.add_argument_group("Scoring")
    scoring.add_argument("--borzoi-tracks", type=int, nargs="+", default=None)
    scoring.add_argument("--enformer-tracks", type=int, nargs="+", default=None)
    scoring.add_argument("--borzoi-reduce", choices=REDUCE_METHODS, default="lcb")
    scoring.add_argument("--enformer-reduce", choices=REDUCE_METHODS, default="mean")
    scoring.add_argument("--scoring-batch-size", type=int, default=15)

    objective = parser.add_argument_group("MORSE objective")
    objective.add_argument("--pattern-normalization", choices=PATTERN_NORMALIZATIONS, default="global_max")
    objective.add_argument("--contrast-margin", type=float, default=0.2)
    objective.add_argument("--contrast-weight", type=float, default=1.0)
    objective.add_argument("--raw-amplitude-weight", type=float, default=0.2)
    objective.add_argument("--high-window-reward-weight", type=float, default=0.4)
    objective.add_argument("--low-window-penalty-weight", type=float, default=0.4)
    objective.add_argument("--window-stat-transform", choices=WINDOW_STAT_TRANSFORMS, default="log1p")

    args = parser.parse_args(argv)
    return RuntimeConfig(
        left_flank_path=args.left_flank.resolve(),
        right_flank_path=args.right_flank.resolve(),
        output_dir=args.output_dir.resolve(),
        intermediate_output_dir=(
            None if args.intermediate_output_dir is None else args.intermediate_output_dir.resolve()
        ),
        organism=args.organism,
        pattern=args.pattern,
        dot_bp=args.dot_bp,
        dash_bp=args.dash_bp,
        intra_symbol_gap_bp=args.intra_gap_bp,
        inter_letter_gap_bp=args.inter_gap_bp,
        pattern_start_bp=args.pattern_start_bp,
        target_length=args.target_length,
        evo_device=args.evo_device,
        scoring_device=args.scoring_device or args.evo_device,
        evo_prompt=args.evo_prompt.upper() if args.evo_prompt else None,
        evo_prompt_context_bp=args.evo_prompt_context_bp,
        evo_temperature=args.evo_temperature,
        evo_top_k=args.evo_top_k,
        evo_top_p=args.evo_top_p,
        stop_at_eos=args.stop_at_eos,
        use_kv_caching=args.use_kv_caching,
        force_prompt_threshold=args.force_prompt_threshold,
        beam_length=args.beam_length,
        beam_width=args.beam_width,
        candidates_per_beam=args.candidates_per_beam,
        beam_batch_size=args.beam_batch_size,
        score_by=args.score_by,
        borzoi_tracks=None if args.borzoi_tracks is None else tuple(args.borzoi_tracks),
        enformer_tracks=None if args.enformer_tracks is None else tuple(args.enformer_tracks),
        borzoi_ensemble_reduce_method=args.borzoi_reduce,
        enformer_track_reduce_method=args.enformer_reduce,
        scoring_batch_size=args.scoring_batch_size,
        pattern_normalization=args.pattern_normalization,
        contrast_margin=args.contrast_margin,
        contrast_weight=args.contrast_weight,
        raw_amplitude_weight=args.raw_amplitude_weight,
        high_window_reward_weight=args.high_window_reward_weight,
        low_window_penalty_weight=args.low_window_penalty_weight,
        window_stat_transform=args.window_stat_transform,
    )


def _read_fasta_sequence(path: Path) -> str:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(">"):
            continue
        lines.append(line.strip())
    sequence = "".join(lines).upper()
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return sequence


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    lines = [f">{name}\n{sequence}\n" for name, sequence in records]
    path.write_text("".join(lines), encoding="utf-8")


def _make_intermediate_state_logger(
    output_dir: Path,
    *,
    optimizer: BeamSearchOptimizer,
    target_label: str = "morse_target",
) -> Callable[[int, tuple[Segment, ...]], None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "intermediate_states.tsv"
    if not summary_path.exists():
        summary_path.write_text("beam\trank\tenergy\ttarget_length\ttarget_fasta\n", encoding="utf-8")

    def log_intermediate_state(beam_num: int, segments: tuple[Segment, ...]) -> None:
        target_segment = next((segment for segment in segments if segment.label == target_label), None)
        if target_segment is None:
            raise ValueError(f"Could not find target segment labeled {target_label!r}.")

        beam_dir = output_dir / f"beam_{beam_num:04d}"
        beam_dir.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        records: list[tuple[str, str]] = []
        # strict=True: energy/sequence pairing is invariant; fail loud on drift.
        for rank, (sequence, energy) in enumerate(
            zip(target_segment.result_sequences, optimizer.energy_scores, strict=True)
        ):
            record_name = f"beam_{beam_num:04d}_rank_{rank}_energy_{energy:.6g}"
            records.append((record_name, sequence.sequence))
            rows.append(
                f"{beam_num}\t{rank}\t{energy:.12g}\t{len(sequence.sequence)}\t"
                f"{beam_dir / f'rank_{rank:02d}_target.fa'}\n"
            )
            _write_fasta(beam_dir / f"rank_{rank:02d}_target.fa", [(record_name, sequence.sequence)])

        _write_fasta(beam_dir / "selected_targets.fa", records)
        (beam_dir / "selected_targets.tsv").write_text(
            "beam\trank\tenergy\ttarget_length\ttarget_fasta\n" + "".join(rows),
            encoding="utf-8",
        )
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.writelines(rows)

    return log_intermediate_state


def build_program(cfg: RuntimeConfig) -> Program:
    left_flank_seq = _read_fasta_sequence(cfg.left_flank_path)
    right_flank_seq = _read_fasta_sequence(cfg.right_flank_path)
    evo_prompt = cfg.evo_prompt or left_flank_seq[-cfg.evo_prompt_context_bp :]
    highs, lows = compute_morse_windows(
        pattern=cfg.pattern,
        pattern_start_bp=cfg.pattern_start_bp,
        dot_bp=cfg.dot_bp,
        dash_bp=cfg.dash_bp,
        intra_symbol_gap_bp=cfg.intra_symbol_gap_bp,
        inter_letter_gap_bp=cfg.inter_letter_gap_bp,
    )
    pattern_end_bp = max((end for _, end in highs + lows), default=cfg.pattern_start_bp)
    if pattern_end_bp > cfg.target_length:
        raise ValueError(
            "Morse pattern does not fit target segment: "
            f"pattern_end_bp={pattern_end_bp}, target_length={cfg.target_length}."
        )

    logger.info("Output directory: %s", cfg.output_dir)
    logger.info(
        "Sequence lengths | left=%d target=%d right=%d total=%d",
        len(left_flank_seq),
        cfg.target_length,
        len(right_flank_seq),
        len(left_flank_seq) + cfg.target_length + len(right_flank_seq),
    )
    logger.info(
        "Pattern geometry | dot=%d dash=%d intra_gap=%d inter_gap=%d start=%d end=%d",
        cfg.dot_bp,
        cfg.dash_bp,
        cfg.intra_symbol_gap_bp,
        cfg.inter_letter_gap_bp,
        cfg.pattern_start_bp,
        pattern_end_bp,
    )
    logger.info(
        "Beam config | length=%d width=%d candidates=%d batch=%d score_by=%s use_kv_caching=%s",
        cfg.beam_length,
        cfg.beam_width,
        cfg.candidates_per_beam,
        cfg.beam_batch_size,
        cfg.score_by,
        cfg.use_kv_caching,
    )

    left_flank = Segment(sequence=left_flank_seq, sequence_type="dna", label="left_flank")
    target = Segment(length=cfg.target_length, sequence_type="dna", label="morse_target")
    right_flank = Segment(sequence=right_flank_seq, sequence_type="dna", label="right_flank")
    construct = Construct([left_flank, target, right_flank], label="morse_construct")

    generator = Evo2Generator(
        Evo2GeneratorConfig(
            prompts=[evo_prompt],
            top_k=cfg.evo_top_k,
            top_p=cfg.evo_top_p,
            temperature=cfg.evo_temperature,
            batch_size=cfg.beam_batch_size,
            prepend_prompt=False,
            force_prompt_threshold=cfg.force_prompt_threshold,
            stop_at_eos=cfg.stop_at_eos,
            device=cfg.evo_device,
        ),
    )
    generator.assign(target)

    morse_config = {
        "organism": cfg.organism,
        "pattern": cfg.pattern,
        "dot_bp": cfg.dot_bp,
        "dash_bp": cfg.dash_bp,
        "intra_symbol_gap_bp": cfg.intra_symbol_gap_bp,
        "inter_letter_gap_bp": cfg.inter_letter_gap_bp,
        "pattern_start_bp": cfg.pattern_start_bp,
        "pattern_normalization": cfg.pattern_normalization,
        "contrast_margin": cfg.contrast_margin,
        "contrast_weight": cfg.contrast_weight,
        "raw_amplitude_weight": cfg.raw_amplitude_weight,
        "high_window_reward_weight": cfg.high_window_reward_weight,
        "low_window_penalty_weight": cfg.low_window_penalty_weight,
        "window_stat_transform": cfg.window_stat_transform,
        "device": cfg.scoring_device,
        "batch_size": cfg.scoring_batch_size,
        "trim_prefix_bp": len(evo_prompt),
    }
    borzoi_config = {
        **morse_config,
        "borzoi_ensemble_reduce_method": cfg.borzoi_ensemble_reduce_method,
    }
    enformer_config = {
        **morse_config,
        "enformer_track_reduce_method": cfg.enformer_track_reduce_method,
    }
    if cfg.borzoi_tracks is not None:
        borzoi_config["borzoi_output_tracks"] = list(cfg.borzoi_tracks)
    if cfg.enformer_tracks is not None:
        enformer_config["enformer_output_tracks"] = list(cfg.enformer_tracks)

    constraints = [
        Constraint(
            inputs=[left_flank, target, right_flank],
            function=borzoi_chromatin_accessibility_morse_constraint,
            function_config=BorzoiChromatinAccessibilityMorseConfig(**borzoi_config),
            weight=1.0,
            label="chromatin_accessibility_morse_borzoi",
        ),
        Constraint(
            inputs=[left_flank, target, right_flank],
            function=enformer_chromatin_accessibility_morse_constraint,
            function_config=EnformerChromatinAccessibilityMorseConfig(**enformer_config),
            weight=1.0,
            label="chromatin_accessibility_morse_enformer",
        ),
    ]

    beam_config = BeamSearchOptimizerConfig(
        prompt=evo_prompt,
        beam_length=cfg.beam_length,
        num_results=cfg.beam_width,
        proposals_per_result=cfg.candidates_per_beam,
        score_by=cfg.score_by,
        use_kv_caching=cfg.use_kv_caching,
        prepend_prompt=False,
        verbose=True,
        tracking_interval=1,
        track_proposals=True,
    )
    beam_optimizer = BeamSearchOptimizer(
        target_segment=target,
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=beam_config,
    )
    if cfg.intermediate_output_dir is not None:
        beam_optimizer.custom_logging = _make_intermediate_state_logger(
            cfg.intermediate_output_dir,
            optimizer=beam_optimizer,
        )
        logger.info("Intermediate selected beam targets will be written to %s", cfg.intermediate_output_dir)

    return Program(
        optimizers=[beam_optimizer],
        num_results=cfg.beam_width,
        verbose=True,
    )


def _run_epigenomics_morse_pipeline(cfg: RuntimeConfig) -> None:
    from proto_tools.utils import DeviceManager

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "run_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str),
        encoding="utf-8",
    )
    if cfg.evo_device == cfg.scoring_device:
        DeviceManager.get_instance().configure(allow_multiple_per_device=True)
    program = build_program(cfg)
    program.run()
    program.export(cfg.output_dir, format="csv", include_proposals=True)

    construct = program.constructs[0]
    best_sequence = construct.joined_sequences[0].sequence
    (cfg.output_dir / "best_sequence.fa").write_text(
        f">{construct.label}\n{best_sequence}\n",
        encoding="utf-8",
    )
    (cfg.output_dir / "best_energy.txt").write_text(f"{program.energy_scores[0]}\n", encoding="utf-8")
    target = construct.segments[1]
    (cfg.output_dir / "best_target.fa").write_text(
        f">{target.label}\n{target.result_sequences[0].sequence}\n",
        encoding="utf-8",
    )
    logger.info("All outputs saved to %s", cfg.output_dir)


def main(argv: list[str] | None = None) -> None:
    from proto_tools.utils.tool_instance import ToolInstance

    cfg = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    with ToolInstance.persist():
        _run_epigenomics_morse_pipeline(cfg)


if __name__ == "__main__":
    main()
