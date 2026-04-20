"""Run a Germinal-style PD-L1 antibody redesign pipeline.

By default this example stitches:

- ``examples/germinal/pdbs/pdl1.pdb`` chain ``A`` as the PD-L1 target
- ``examples/germinal/pdbs/nb.pdb`` chain ``A`` as the VHH scaffold
- Companion Germinal-style YAML configs from ``examples/germinal/configs/``

The program has three stages:

1. Germinal logit phase (AF2 germinal backend + external AbLang gradient)
2. Germinal softmax refinement (AF2 germinal backend + external AbLang gradient)
3. pLDDT-weighted semigreedy MCMC refinement with forward AF2 + AbLang scoring

The default step counts are intentionally small so the script doubles as a smoke
test. Increase them for more realistic optimization runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from Bio.PDB import PDBIO, PDBParser, PPBuilder
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Structure import Structure as BioPDBStructure
from proto_tools.entities.structures import Structure
from proto_tools.utils.device_manager import DeviceManager
from proto_tools.utils.tool_pool import ToolPool

from proto_language.language.constraint.differentiable.ablang_naturalness_constraint import (
    AbLangConstraintConfig,
    ablang_naturalness_forward,
    ablang_naturalness_gradient_backward,
)
from proto_language.language.constraint.differentiable.af2_binder_constraint import (
    AF2BinderConstraintConfig,
    af2_binder_backward,
    af2_binder_forward,
)
from proto_language.language.core import (
    PROTEIN_AMINO_ACIDS,
    Constraint,
    Construct,
    GradientResult,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGET_PDB = Path(__file__).resolve().parents[2] / "examples" / "germinal" / "pdbs" / "pdl1.pdb"
DEFAULT_NANOBODY_TEMPLATE_PDB = Path(__file__).resolve().parents[2] / "examples" / "germinal" / "pdbs" / "nb.pdb"
DEFAULT_VHH_CDR_LENGTHS = (11, 8, 18)
DEFAULT_VHH_FRAMEWORK_LENGTHS = (25, 17, 38, 14)
DEFAULT_SCFV_CDR_LENGTHS = (8, 8, 13, 6, 6, 9)
DEFAULT_SCFV_FRAMEWORK_LENGTHS = (25, 17, 38, 52, 17, 33, 10)
DEFAULT_PDL1_TARGET_HOTSPOT = "37,39,41,96,98"
_HOTSPOT_TOKEN_RE = re.compile(r"^(?P<chain>[A-Za-z]+)?(?P<residue>\d+[A-Za-z]?)$")

YAML_LOSS_WEIGHT_MAP = {
    "weights_plddt": "plddt",
    "weights_i_plddt": "i_plddt",
    "weights_pae_intra": "pae",
    "weights_pae_inter": "i_pae",
    "weights_con_intra": "con",
    "weights_con_inter": "i_con",
    "weights_rg": "rg",
    "weights_iptm": "i_ptm",
    "weights_helix": "helix",
    "weights_beta": "beta_strand",
    "dgram_cce": "dgram_cce",
}


def build_parser(
    *,
    description: str | None = None,
    default_config_yaml: Path | None = None,
    default_target_pdb: Path = DEFAULT_TARGET_PDB,
    default_binder_template_pdb: Path = DEFAULT_NANOBODY_TEMPLATE_PDB,
    default_binder_type: str = "vhh",
    default_cdr_lengths: tuple[int, ...] = DEFAULT_VHH_CDR_LENGTHS,
    default_framework_lengths: tuple[int, ...] = DEFAULT_VHH_FRAMEWORK_LENGTHS,
    default_vh_len: int | None = None,
    default_vl_len: int | None = None,
    default_vh_first: bool = True,
) -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=default_config_yaml,
        help="Optional Germinal-style YAML config to map onto the local PD-L1 runner.",
    )
    parser.add_argument(
        "--binder-type",
        choices=["vhh", "scfv"],
        default=default_binder_type,
        help="Antibody binder format for the scaffold and external AbLang constraint.",
    )
    parser.add_argument("--target-pdb", type=Path, default=default_target_pdb, help="Target PDB.")
    parser.add_argument("--target-chain", default="A", help="Frozen target chain ID in --target-pdb.")
    parser.add_argument(
        "--target-hotspot",
        default=DEFAULT_PDL1_TARGET_HOTSPOT,
        help="Comma-separated target hotspot residues for AF2 binder design, e.g. '37,39,41,96,98'.",
    )
    parser.add_argument(
        "--binder-template-pdb",
        "--nanobody-template-pdb",
        dest="binder_template_pdb",
        type=Path,
        default=default_binder_template_pdb,
        help="Binder template/scaffold PDB.",
    )
    parser.add_argument(
        "--binder-template-chain",
        "--nanobody-template-chain",
        dest="binder_template_chain",
        default="A",
        help="Binder template chain ID inside --binder-template-pdb.",
    )
    parser.add_argument(
        "--binder-chain",
        default="B",
        help="Binder chain ID assigned inside the stitched AF2 template PDB.",
    )
    parser.add_argument(
        "--binder-sequence",
        default=None,
        help="Optional starting binder sequence. Defaults to the sequence extracted from --binder-template-pdb.",
    )
    parser.add_argument(
        "--proto-home",
        type=Path,
        default=None,
        help="Optional PROTO_HOME override for tool envs and model caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for summary JSON, FASTA, and binder PDB exports.",
    )
    parser.add_argument("--num-results", type=int, default=2, help="Parallel design trajectories / final candidates.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of independent seeds to run sequentially.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment between sequential seeds in a multi-seed sweep.",
    )
    parser.add_argument("--logit-steps", type=int, default=3, help="Stage 1 Germinal logit steps.")
    parser.add_argument("--softmax-steps", type=int, default=2, help="Stage 2 Germinal softmax steps.")
    parser.add_argument("--mcmc-steps", type=int, default=3, help="Stage 3 semigreedy MCMC steps.")
    parser.add_argument(
        "--proposals-per-result",
        type=int,
        default=2,
        help="Semigreedy MCMC proposals per trajectory.",
    )
    parser.add_argument("--num-recycles", type=int, default=1, help="AF2 recycles for binder scoring.")
    parser.add_argument(
        "--sample-models",
        action="store_true",
        help="Randomly sample AF2 model weights on each AF2 binder call.",
    )
    parser.add_argument(
        "--position-weighting",
        choices=["uniform", "entropy", "plddt"],
        default="plddt",
        help="Semigreedy mutation position weighting.",
    )
    parser.add_argument(
        "--semigreedy-temperature",
        type=float,
        default=0.6,
        help="Temperature for semigreedy amino-acid sampling from logits.",
    )
    parser.add_argument(
        "--mcmc-max-temperature",
        type=float,
        default=0.01,
        help="Initial MCMC temperature for semigreedy refinement.",
    )
    parser.add_argument(
        "--mcmc-min-temperature",
        type=float,
        default=0.001,
        help="Final MCMC temperature for semigreedy refinement.",
    )
    parser.add_argument(
        "--ablang-weight",
        type=float,
        default=0.4,
        help="External AbLang weight for the semigreedy scoring stage.",
    )
    parser.add_argument(
        "--ablang-device",
        default="cuda",
        help="Execution device for the external AbLang semigreedy stage, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--allow-multiple-per-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow AF2 and AbLang persistent workers to coexist on the same GPU.",
    )
    parser.add_argument(
        "--ablang-temperature",
        "--ablm-temperature",
        dest="ablang_temperature",
        type=float,
        default=0.6,
        help="Temperature for the external AbLang naturalness constraint.",
    )
    parser.add_argument(
        "--cdr-lengths",
        type=int,
        nargs="+",
        default=default_cdr_lengths,
        metavar="CDR_LEN",
        help="Sequential CDR lengths used to derive AF2 design positions.",
    )
    parser.add_argument(
        "--framework-lengths",
        type=int,
        nargs="+",
        default=default_framework_lengths,
        metavar="FW_LEN",
        help="Sequential framework lengths paired with --cdr-lengths to derive AF2 design positions.",
    )
    parser.add_argument(
        "--vh-len",
        type=int,
        default=default_vh_len,
        help="scFv-only: VH domain length used to split the single-chain scaffold for AbLang.",
    )
    parser.add_argument(
        "--vl-len",
        type=int,
        default=default_vl_len,
        help="scFv-only: VL domain length used to split the single-chain scaffold for AbLang.",
    )
    parser.add_argument(
        "--vh-first",
        action=argparse.BooleanOptionalAction,
        default=default_vh_first,
        help="scFv-only: whether the single-chain scaffold is ordered VH-linker-VL.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Program/tool seed.")
    parser.add_argument("--verbose", action="store_true", help="Enable optimizer-level verbose logging.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build the program and print the config without running."
    )
    return parser


def _parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Return parser defaults keyed by destination."""
    return {action.dest: action.default for action in parser._actions if action.dest is not argparse.SUPPRESS}


def normalize_binder_type(binder_type: str | None) -> str | None:
    """Normalize binder type aliases to the local runner vocabulary."""
    if binder_type is None:
        return None
    normalized = binder_type.strip().lower()
    if normalized in {"vhh", "nb", "nanobody"}:
        return "vhh"
    if normalized == "scfv":
        return "scfv"
    raise ValueError(f"Unsupported binder type {binder_type!r}; expected one of: vhh, nb, nanobody, scfv.")


def _apply_yaml_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Map supported Germinal YAML fields onto local runner arguments."""
    args.loss_weights_override = None
    args.bias_redesign_override = None
    args.intra_contact_num_override = None
    args.intra_contact_cutoff_override = None
    args.inter_contact_num_override = None
    args.inter_contact_cutoff_override = None
    args.yaml_applied = {}

    if args.config_yaml is None:
        return args

    config = yaml.safe_load(args.config_yaml.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {args.config_yaml}, got {type(config).__name__}.")

    defaults = _parser_defaults(parser)

    def maybe_set(name: str, value: Any) -> None:
        if value is None:
            return
        if getattr(args, name) == defaults.get(name):
            setattr(args, name, value)
            args.yaml_applied[name] = value

    maybe_set("binder_type", normalize_binder_type(config.get("type")))
    maybe_set("logit_steps", config.get("logits_steps"))
    maybe_set("softmax_steps", config.get("softmax_steps"))
    maybe_set("mcmc_steps", config.get("search_steps"))
    maybe_set("num_seeds", config.get("num_seqs"))
    maybe_set("num_recycles", config.get("num_recycles_design"))
    maybe_set("sample_models", config.get("sample_models"))
    maybe_set("ablang_temperature", config.get("ablm_temp"))
    maybe_set("target_hotspot", normalize_target_hotspot(config.get("target_hotspots"), args.target_chain))
    maybe_set("cdr_lengths", tuple(config.get("cdr_lengths", [])) or None)
    maybe_set("framework_lengths", tuple(config.get("fw_lengths", [])) or None)
    maybe_set("vh_len", config.get("vh_len"))
    maybe_set("vl_len", config.get("vl_len"))
    maybe_set("vh_first", config.get("vh_first"))
    maybe_set("semigreedy_temperature", config.get("sampling_temp"))

    if "ablm_model" in config:
        ablm_model = str(config["ablm_model"]).strip().lower()
        if ablm_model != "ablang":
            raise ValueError(
                "This local PD-L1 runner uses the external AbLang constraint; only ablm_model='ablang' is supported."
            )
        args.yaml_applied["ablang_backend"] = "external_ablang"

    search_mutation_rate = config.get("search_mutation_rate")
    if args.proposals_per_result == defaults.get("proposals_per_result") and search_mutation_rate is not None:
        cdr_lengths = config.get("cdr_lengths") or list(args.cdr_lengths)
        fw_lengths = config.get("fw_lengths") or list(args.framework_lengths)
        length = sum(cdr_lengths) + sum(fw_lengths)
        args.proposals_per_result = max(1, math.ceil(length * float(search_mutation_rate)))
        args.yaml_applied["proposals_per_result"] = args.proposals_per_result

    loss_weights: dict[str, float] = {}
    for yaml_key, dst_key in YAML_LOSS_WEIGHT_MAP.items():
        if yaml_key in config:
            loss_weights[dst_key] = float(config[yaml_key])
    if not config.get("use_rg_loss", True):
        loss_weights.pop("rg", None)
    if not config.get("use_i_ptm_loss", True):
        loss_weights.pop("i_ptm", None)
    if not config.get("use_helix_loss", True):
        loss_weights.pop("helix", None)
    if not config.get("use_beta_loss", True):
        loss_weights.pop("beta_strand", None)
    if config.get("use_termini_distance_loss", False):
        loss_weights["NC"] = float(config.get("weights_termini_loss", 0.1))
    if loss_weights:
        args.loss_weights_override = loss_weights
        args.yaml_applied["loss_weights"] = loss_weights

    if "bias_redesign" in config:
        args.bias_redesign_override = float(config["bias_redesign"])
        args.yaml_applied["bias_redesign"] = args.bias_redesign_override
    if "intra_contact_number" in config:
        args.intra_contact_num_override = int(config["intra_contact_number"])
        args.yaml_applied["intra_contact_num"] = args.intra_contact_num_override
    if "intra_contact_distance" in config:
        args.intra_contact_cutoff_override = float(config["intra_contact_distance"])
        args.yaml_applied["intra_contact_cutoff"] = args.intra_contact_cutoff_override
    if "inter_contact_number" in config:
        args.inter_contact_num_override = int(config["inter_contact_number"])
        args.yaml_applied["inter_contact_num"] = args.inter_contact_num_override
    if "inter_contact_distance" in config:
        args.inter_contact_cutoff_override = float(config["inter_contact_distance"])
        args.yaml_applied["inter_contact_cutoff"] = args.inter_contact_cutoff_override

    return args


def parse_args(parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = parser or build_parser()
    args = _apply_yaml_config(parser.parse_args(), parser)
    args.binder_type = normalize_binder_type(args.binder_type)
    args.target_hotspot = normalize_target_hotspot(args.target_hotspot, args.target_chain)
    return args


def configure_logging(debug: bool) -> None:
    """Set up process-wide logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


def normalize_target_hotspot(hotspot_spec: str | None, target_chain: str) -> str | None:
    """Normalize hotspot specs to the residue-only format expected by the AF2 wrapper."""
    if hotspot_spec is None:
        return None

    normalized_tokens: list[str] = []
    for raw_token in hotspot_spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = _HOTSPOT_TOKEN_RE.fullmatch(token)
        if match is None:
            normalized_tokens.append(token)
            continue
        chain = match.group("chain")
        residue = match.group("residue")
        if chain is not None and chain != target_chain:
            raise ValueError(
                f"Hotspot residue {token!r} targets chain {chain!r}, but this run uses target_chain={target_chain!r}."
            )
        normalized_tokens.append(residue)
    return ",".join(normalized_tokens) or None


def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    """Extract the one-letter sequence for a PDB chain."""
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id!r} not found in {pdb_path}.")
    peptides = PPBuilder().build_peptides(model[chain_id])
    sequence = "".join(str(peptide.get_sequence()) for peptide in peptides)
    if not sequence:
        raise ValueError(f"Could not extract a protein sequence for chain {chain_id!r} in {pdb_path}.")
    return sequence


def _sequence_views(segment: Segment) -> list[Sequence]:
    """Return all sequence views held by a segment."""
    return [segment.original_sequence, *segment.result_sequences, *segment.proposal_sequences]


def attach_structure(segment: Segment, structure: Structure) -> None:
    """Attach the same structure object to every sequence view of a segment."""
    for sequence in _sequence_views(segment):
        sequence.structure = structure


def build_target_segment(pdb_path: Path, target_chain: str) -> Segment:
    """Build the fixed target segment from the template PDB."""
    target_sequence = extract_chain_sequence(pdb_path, target_chain)
    target = Segment(sequence=target_sequence, sequence_type="protein", label="target")
    attach_structure(target, Structure(structure=pdb_path.read_text(), structure_format="pdb"))
    return target


def build_design_positions(
    cdr_lengths: tuple[int, ...] | list[int],
    framework_lengths: tuple[int, ...] | list[int],
    binder_length: int,
) -> tuple[list[int], list[int]]:
    """Return zero-based CDR and framework positions from alternating region lengths."""
    if not cdr_lengths:
        raise ValueError("cdr_lengths must be non-empty.")
    if len(framework_lengths) != len(cdr_lengths) + 1:
        raise ValueError(
            f"Expected len(framework_lengths) == len(cdr_lengths) + 1, got {framework_lengths} vs {cdr_lengths}."
        )

    cdr_positions: list[int] = []
    framework_positions: list[int] = []
    position = 0
    for index, cdr_length in enumerate(cdr_lengths):
        framework_length = framework_lengths[index]
        framework_positions.extend(range(position, position + framework_length))
        position += framework_length
        cdr_positions.extend(range(position, position + cdr_length))
        position += cdr_length

    framework_positions.extend(range(position, position + framework_lengths[-1]))
    position += framework_lengths[-1]

    if position != binder_length:
        raise ValueError(
            f"CDR/framework lengths imply binder length {position}, but scaffold sequence length is {binder_length}."
        )
    return cdr_positions, framework_positions


def _validate_scfv_split(
    binder_length: int,
    cdr_positions: list[int],
    *,
    vh_len: int | None,
    vl_len: int | None,
    vh_first: bool,
) -> None:
    """Validate an scFv single-chain split: positive chain lengths, fits in binder, no CDR/linker overlap."""
    if vh_len is None or vl_len is None:
        raise ValueError("scFv runs require both --vh-len and --vl-len (or YAML vh_len / vl_len).")
    if vh_len <= 0 or vl_len <= 0:
        raise ValueError(f"scFv domain lengths must be positive, got vh_len={vh_len}, vl_len={vl_len}.")
    if vh_len + vl_len > binder_length:
        raise ValueError(
            f"scFv vh_len + vl_len = {vh_len + vl_len}, but scaffold sequence length is only {binder_length}."
        )
    linker_start = vh_len if vh_first else vl_len
    linker_end = binder_length - (vl_len if vh_first else vh_len)
    overlap = sorted(set(range(linker_start, linker_end)).intersection(cdr_positions))
    if overlap:
        raise ValueError(
            f"CDR positions {overlap} overlap the scFv linker; the CDR/framework YAML does not match the scaffold split."
        )


def _template_workspace(output_dir: Path | None) -> Path:
    """Return the directory used for stitched target+binder templates."""
    if output_dir is not None:
        workspace = output_dir.resolve() / "_templates"
    else:
        workspace = Path.cwd() / ".tmp" / "stitched_templates"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def stitch_template_complex(
    target_pdb_path: Path,
    *,
    target_chain: str,
    binder_template_pdb: Path,
    binder_template_chain: str,
    binder_chain: str,
    output_dir: Path | None,
) -> Path:
    """Create a two-chain AF2 template from separate target and binder template inputs."""
    if binder_chain == target_chain:
        raise ValueError("binder_chain must differ from target_chain when stitching the AF2 template.")

    output_path = (
        _template_workspace(output_dir)
        / f"{target_pdb_path.stem}_{target_chain}_{binder_template_pdb.stem}_{binder_template_chain}_as_{binder_chain}.pdb"
    )
    if output_path.exists():
        return output_path

    parser = PDBParser(QUIET=True)
    target_structure = parser.get_structure(target_pdb_path.stem, str(target_pdb_path))
    binder_structure = parser.get_structure(binder_template_pdb.stem, str(binder_template_pdb))
    target_model = next(target_structure.get_models())
    binder_model = next(binder_structure.get_models())

    if target_chain not in target_model:
        raise ValueError(f"Target chain {target_chain!r} not found in {target_pdb_path}.")
    if binder_template_chain not in binder_model:
        raise ValueError(f"Binder template chain {binder_template_chain!r} not found in {binder_template_pdb}.")

    stitched_structure = BioPDBStructure(output_path.stem)
    stitched_model = BioModel(0)
    stitched_structure.add(stitched_model)

    stitched_model.add(copy.deepcopy(target_model[target_chain]))
    binder_chain_obj = copy.deepcopy(binder_model[binder_template_chain])
    binder_chain_obj.id = binder_chain
    stitched_model.add(binder_chain_obj)

    io = PDBIO()
    io.set_structure(stitched_structure)
    io.save(str(output_path))
    return output_path


def resolve_template_inputs(args: argparse.Namespace) -> tuple[Path, str, list[int]]:
    """Resolve the stitched AF2 template PDB, binder seed sequence, and CDR design positions."""
    target_pdb_path = args.target_pdb.resolve()
    binder_template_pdb = args.binder_template_pdb.resolve()
    template_pdb_path = stitch_template_complex(
        target_pdb_path,
        target_chain=args.target_chain,
        binder_template_pdb=binder_template_pdb,
        binder_template_chain=args.binder_template_chain,
        binder_chain=args.binder_chain,
        output_dir=args.output_dir,
    )
    binder_seed = args.binder_sequence or extract_chain_sequence(binder_template_pdb, args.binder_template_chain)
    cdr_positions, _ = build_design_positions(args.cdr_lengths, args.framework_lengths, len(binder_seed))
    if args.binder_type == "scfv":
        _validate_scfv_split(
            len(binder_seed),
            cdr_positions,
            vh_len=args.vh_len,
            vl_len=args.vl_len,
            vh_first=args.vh_first,
        )
    return template_pdb_path, binder_seed, cdr_positions


def make_af2_config(
    args: argparse.Namespace,
    *,
    template_pdb: Path,
    starting_binder_seq: str | None,
    design_positions: list[int] | None,
) -> AF2BinderConstraintConfig:
    """Create the AF2 germinal binder config used across stages."""
    config = AF2BinderConstraintConfig.germinal_vhh_preset(
        target_pdb=str(template_pdb.resolve()),
        binder_chain=args.binder_chain,
    )
    config.target_chain = args.target_chain
    config.target_hotspot = args.target_hotspot
    config.num_recycles = args.num_recycles
    config.sample_models = args.sample_models
    config.starting_binder_seq = starting_binder_seq
    config.design_positions = design_positions
    if args.loss_weights_override is not None:
        config.loss_weights = dict(args.loss_weights_override)
    if args.bias_redesign_override is not None:
        config.bias_redesign = args.bias_redesign_override
    if args.intra_contact_num_override is not None:
        config.intra_contact_num = args.intra_contact_num_override
    if args.intra_contact_cutoff_override is not None:
        config.intra_contact_cutoff = args.intra_contact_cutoff_override
    if args.inter_contact_num_override is not None:
        config.inter_contact_num = args.inter_contact_num_override
    if args.inter_contact_cutoff_override is not None:
        config.inter_contact_cutoff = args.inter_contact_cutoff_override
    return config


def build_germinal_semigreedy_bias(
    binder_seed: str,
    *,
    design_positions: list[int] | None,
    bias_redesign: float | None,
    omit_aas: str | None,
) -> list[list[float]] | None:
    """Build Germinal's persistent bias matrix for decoding and semigreedy sampling."""
    bias = np.zeros((len(binder_seed), len(PROTEIN_AMINO_ACIDS)), dtype=np.float64)
    aa_index = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
    design_idx = np.asarray(design_positions, dtype=int) if design_positions else np.array([], dtype=int)

    if bias_redesign is not None:
        for pos, aa in enumerate(binder_seed):
            bias[pos, aa_index[aa]] += bias_redesign
        if design_idx.size > 0:
            bias[design_idx] = 0.0

    if omit_aas:
        omit_indices = [aa_index[token.strip()] for token in omit_aas.split(",") if token.strip()]
        if omit_indices:
            target_positions = design_idx if design_idx.size > 0 else np.arange(len(binder_seed))
            bias[np.ix_(target_positions, omit_indices)] -= 1e6

    return bias.tolist() if np.any(bias) else None


def make_cdr_masked_ablang_backward(cdr_positions: list[int]) -> Any:
    """Return an AbLang backward wrapper that zeroes framework gradients.

    Germinal's scaffold preservation in the VHH YAML is soft on the AF2 side
    (``bias_redesign`` + framework-contact loss). When AbLang runs externally,
    its full-chain gradient can overpower that and drag the framework. This
    wrapper limits its gradient support to the CDR positions passed to the AF2
    binder objective.
    """
    cdr_index = np.asarray(sorted(set(cdr_positions)), dtype=int)

    def backward(inputs: tuple[Sequence, ...], *, config: AbLangConstraintConfig, **kwargs: Any) -> Any:
        result = ablang_naturalness_gradient_backward(inputs, config=config, **kwargs)
        gradient = np.array(result.gradient[0], dtype=np.float64, copy=True)
        mask = np.zeros(gradient.shape[0], dtype=bool)
        mask[cdr_index] = True
        gradient[~mask] = 0.0
        return GradientResult(
            gradient=(gradient,),
            loss=result.loss,
            metrics=result.metrics,
            structures=result.structures,
        )

    backward.__name__ = "ablang_naturalness_cdr_gradient_backward"
    backward._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]
    backward._constraint_num_input_sequences_per_tuple = 1  # type: ignore[attr-defined]
    return backward


def build_ablang_constraint(
    args: argparse.Namespace,
    binder: Segment,
    *,
    cdr_positions: list[int],
    weight: float,
    include_backward: bool,
) -> Constraint:
    """Create the external AbLang constraint; scFv runs set ``heavy_slice``/``light_slice``."""
    heavy_slice: tuple[int, int] | None = None
    light_slice: tuple[int, int] | None = None
    if args.binder_type == "scfv":
        binder_length, vh_len, vl_len = binder.sequence_length, args.vh_len, args.vl_len
        linker_len = binder_length - vh_len - vl_len
        if args.vh_first:
            heavy_slice, light_slice = (0, vh_len), (vh_len + linker_len, binder_length)
        else:
            heavy_slice, light_slice = (vl_len + linker_len, binder_length), (0, vl_len)
    ablang_config = AbLangConstraintConfig(
        temperature=args.ablang_temperature,
        device=args.ablang_device,
        heavy_slice=heavy_slice,
        light_slice=light_slice,
    )

    backward_fn = make_cdr_masked_ablang_backward(cdr_positions) if include_backward else None
    return Constraint(
        inputs=[binder],
        function=ablang_naturalness_forward,
        function_config=ablang_config,
        backward=backward_fn,
        backward_config=copy.deepcopy(ablang_config) if include_backward else None,
        label="ablang",
        weight=weight,
    )


def make_gradient_stage(
    construct: Construct,
    binder: Segment,
    target: Segment,
    optimizer_config: GradientOptimizerConfig,
    args: argparse.Namespace,
    *,
    template_pdb: Path,
    binder_seed: str,
    af2_starting_binder_seq: str | None,
    cdr_positions: list[int],
    ablang_weight: float = 1.0,
) -> GradientOptimizer:
    """Build one gradient-optimization stage."""
    af2_config = make_af2_config(
        args,
        template_pdb=template_pdb,
        starting_binder_seq=af2_starting_binder_seq,
        design_positions=cdr_positions,
    )
    generator = PositionWeightGenerator(
        PositionWeightGeneratorConfig(
            logit_bias=build_germinal_semigreedy_bias(
                binder_seed,
                design_positions=cdr_positions,
                bias_redesign=af2_config.bias_redesign,
                omit_aas=af2_config.omit_aas,
            ),
        ),
    )
    generator.assign(binder)
    constraints = [
        Constraint(
            inputs=[binder, target],
            function=af2_binder_forward,
            function_config=af2_config,
            backward=af2_binder_backward,
            backward_config=copy.deepcopy(af2_config),
            label="af2",
        ),
        build_ablang_constraint(
            args,
            binder,
            cdr_positions=cdr_positions,
            weight=ablang_weight,
            include_backward=True,
        ),
    ]
    optimizer_config.verbose = args.verbose
    return GradientOptimizer(
        target_segment=binder,
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=optimizer_config,
    )


def make_semigreedy_stage(
    construct: Construct,
    binder: Segment,
    target: Segment,
    args: argparse.Namespace,
    *,
    template_pdb: Path,
    binder_seed: str,
    cdr_positions: list[int],
) -> MCMCOptimizer:
    """Build the discrete semigreedy sampling stage."""
    af2_config = make_af2_config(
        args,
        template_pdb=template_pdb,
        starting_binder_seq=None,
        design_positions=cdr_positions,
    )
    generator = SemigreedyMutationGenerator(
        SemigreedyMutationGeneratorConfig(
            position_weighting=args.position_weighting,
            temperature=args.semigreedy_temperature,
            logit_bias=build_germinal_semigreedy_bias(
                binder_seed,
                design_positions=cdr_positions,
                bias_redesign=af2_config.bias_redesign,
                omit_aas=af2_config.omit_aas,
            ),
        )
    )
    generator.assign(binder)
    constraints = [
        Constraint(
            inputs=[binder, target],
            function=af2_binder_forward,
            function_config=af2_config,
            label="af2",
        ),
        build_ablang_constraint(
            args,
            binder,
            cdr_positions=cdr_positions,
            weight=args.ablang_weight,
            include_backward=False,
        ),
    ]
    return MCMCOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=MCMCOptimizerConfig(
            num_steps=args.mcmc_steps,
            proposals_per_result=args.proposals_per_result,
            max_temperature=args.mcmc_max_temperature,
            min_temperature=args.mcmc_min_temperature,
            verbose=args.verbose,
        ),
    )


def build_program(
    args: argparse.Namespace,
    *,
    compute: ToolPool | None = None,
) -> tuple[Program, Segment, Segment, str, Path]:
    """Build the full three-stage PD-L1 redesign program."""
    template_pdb_path, binder_seed, cdr_positions = resolve_template_inputs(args)
    binder = Segment(sequence=binder_seed, sequence_type="protein", label="binder")
    target = build_target_segment(args.target_pdb.resolve(), args.target_chain)
    construct = Construct([binder, target])

    logit_config = GradientOptimizerConfig.germinal_logit_preset()
    logit_config.num_steps = args.logit_steps
    softmax_config = GradientOptimizerConfig.germinal_softmax_preset()
    softmax_config.num_steps = args.softmax_steps

    stage1 = make_gradient_stage(
        construct,
        binder,
        target,
        logit_config,
        args,
        template_pdb=template_pdb_path,
        binder_seed=binder_seed,
        af2_starting_binder_seq=binder_seed,
        cdr_positions=cdr_positions,
    )
    stage2 = make_gradient_stage(
        construct,
        binder,
        target,
        softmax_config,
        args,
        template_pdb=template_pdb_path,
        binder_seed=binder_seed,
        af2_starting_binder_seq=None,
        cdr_positions=cdr_positions,
        ablang_weight=0.4,
    )
    stage3 = make_semigreedy_stage(
        construct,
        binder,
        target,
        args,
        template_pdb=template_pdb_path,
        binder_seed=binder_seed,
        cdr_positions=cdr_positions,
    )

    program = Program(
        optimizers=[stage1, stage2, stage3],
        num_results=args.num_results,
        compute=compute,
        seed=args.seed,
    )
    return program, binder, target, binder_seed, template_pdb_path


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    """Return a shallow-cloned argparse namespace with selected field overrides."""
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def build_seed_values(args: argparse.Namespace) -> list[int]:
    """Return the concrete seed values for this run."""
    return [args.seed + offset * args.seed_step for offset in range(args.num_seeds)]


def summarize_candidates(binder: Segment, energies: list[float]) -> list[dict[str, Any]]:
    """Return ranked final candidates with key metrics."""
    ranked_indices = sorted(range(len(energies)), key=energies.__getitem__)
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        result = binder.result_sequences[idx]
        constraints = result.metadata.get("constraints", {})
        af2_data = constraints.get("af2", {}).get("data", {})
        ablang_data = constraints.get("ablang", {}).get("data", {})
        rows.append(
            {
                "rank": rank,
                "result_index": idx,
                "energy": energies[idx],
                "sequence": result.sequence,
                "avg_plddt": af2_data.get("avg_plddt"),
                "ptm": af2_data.get("ptm"),
                "iptm": af2_data.get("iptm"),
                "avg_pae": af2_data.get("avg_pae"),
                "af2_loss": af2_data.get("loss"),
                "ablang_log_likelihood": ablang_data.get("ablang_log_likelihood"),
                "ablang_loss": ablang_data.get("ablang_loss"),
            }
        )
    return rows


def export_results(output_dir: Path, binder: Segment, summary: list[dict[str, Any]]) -> None:
    """Write summary JSON, FASTA, and binder PDBs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    fasta_lines: list[str] = []
    for row in summary:
        idx = int(row["result_index"])
        result = binder.result_sequences[idx]
        constraints = result.metadata.get("constraints", {})
        af2_data = constraints.get("af2", {}).get("data", {})
        fasta_lines.append(f">candidate_{row['rank']:02d}_energy_{row['energy']:.4f}")
        fasta_lines.append(result.sequence)
        complex_pdb = af2_data.get("complex_pdb")
        if complex_pdb is not None:
            (output_dir / f"candidate_{row['rank']:02d}.pdb").write_text(complex_pdb)
        elif result.structure is not None:
            (output_dir / f"candidate_{row['rank']:02d}.pdb").write_text(result.structure.structure_pdb)
        if result.structure is not None:
            (output_dir / f"candidate_{row['rank']:02d}_binder.pdb").write_text(result.structure.structure_pdb)
    (output_dir / "candidates.fasta").write_text("\n".join(fasta_lines) + "\n")


def export_sweep_results(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    """Write aggregated summary/FASTA/PDB outputs for a multi-seed sweep."""
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict[str, Any]] = []
    fasta_lines: list[str] = []
    for row in summary:
        row_copy = dict(row)
        candidate_pdb = row_copy.get("seed_candidate_pdb")
        if candidate_pdb is not None:
            source_pdb = Path(candidate_pdb)
            if source_pdb.exists():
                dest_pdb = output_dir / f"candidate_{row_copy['rank']:02d}_seed_{row_copy['seed']:04d}.pdb"
                dest_pdb.write_text(source_pdb.read_text())
                row_copy["candidate_pdb"] = str(dest_pdb)
        candidate_binder_pdb = row_copy.get("seed_candidate_binder_pdb")
        if candidate_binder_pdb is not None:
            source_binder_pdb = Path(candidate_binder_pdb)
            if source_binder_pdb.exists():
                dest_binder_pdb = (
                    output_dir / f"candidate_{row_copy['rank']:02d}_seed_{row_copy['seed']:04d}_binder.pdb"
                )
                dest_binder_pdb.write_text(source_binder_pdb.read_text())
                row_copy["candidate_binder_pdb"] = str(dest_binder_pdb)
        fasta_lines.append(
            f">candidate_{row_copy['rank']:02d}_seed_{row_copy['seed']:04d}_energy_{row_copy['energy']:.4f}"
        )
        fasta_lines.append(row_copy["sequence"])
        exported_rows.append(row_copy)

    (output_dir / "summary.json").write_text(json.dumps(exported_rows, indent=2))
    (output_dir / "candidates.fasta").write_text("\n".join(fasta_lines) + "\n")


def log_run_configuration(
    args: argparse.Namespace,
    binder_seed: str,
    target: Segment,
    template_pdb_path: Path,
) -> None:
    """Log the resolved run configuration."""
    logger.info("Target PDB: %s (chain %s)", args.target_pdb.resolve(), args.target_chain)
    logger.info(
        "Binder template PDB: %s (chain %s) -> binder chain %s",
        args.binder_template_pdb.resolve(),
        args.binder_template_chain,
        args.binder_chain,
    )
    logger.info("AF2 template: %s", template_pdb_path.resolve())
    if args.config_yaml is not None:
        logger.info("Config YAML: %s", args.config_yaml.resolve())
    logger.info(
        "Binder type=%s | target length=%d | binder length=%d | target_hotspot=%s",
        args.binder_type,
        len(target.original_sequence.sequence),
        len(binder_seed),
        args.target_hotspot,
    )
    if args.binder_type == "scfv":
        linker_len = len(binder_seed) - args.vh_len - args.vl_len
        logger.info(
            "scFv split: vh_len=%d vl_len=%d linker_len=%d vh_first=%s",
            args.vh_len,
            args.vl_len,
            linker_len,
            args.vh_first,
        )
    logger.info(
        "Stages: logit=%d softmax=%d semigreedy=%d | num_results=%d num_seeds=%d proposals_per_result=%d recycles=%d | ablang_temp=%.2f | ablang_device=%s | share_gpu=%s | seed=%d",
        args.logit_steps,
        args.softmax_steps,
        args.mcmc_steps,
        args.num_results,
        args.num_seeds,
        args.proposals_per_result,
        args.num_recycles,
        args.ablang_temperature,
        args.ablang_device,
        args.allow_multiple_per_device,
        args.seed,
    )
    if args.yaml_applied:
        logger.info("Applied YAML fields: %s", json.dumps(args.yaml_applied, sort_keys=True, default=list))
    if args.proto_home is not None:
        logger.info("PROTO_HOME=%s", args.proto_home)


def log_dry_run_configuration(args: argparse.Namespace) -> None:
    """Build the program once and log the resolved configuration without running it."""
    preview_program, _, target, binder_seed, template_pdb_path = build_program(args)
    del preview_program
    log_run_configuration(args, binder_seed, target, template_pdb_path)


def log_seed_sweep_configuration(args: argparse.Namespace, seeds: list[int]) -> None:
    """Log a concise overview for a multi-seed sweep."""
    logger.info(
        "Seed sweep: %d sequential seeds from %d to %d (step=%d) | expected final candidates=%d",
        len(seeds),
        seeds[0],
        seeds[-1],
        args.seed_step,
        len(seeds) * args.num_results,
    )
    if args.output_dir is not None:
        logger.info("Sweep output root: %s", args.output_dir.resolve())


def annotate_seed_summary(
    summary: list[dict[str, Any]],
    *,
    seed: int,
    seed_output_dir: Path | None,
) -> list[dict[str, Any]]:
    """Attach seed-level metadata to one seed's ranked candidates."""
    annotated: list[dict[str, Any]] = []
    for row in summary:
        row_copy = dict(row)
        row_copy["seed"] = seed
        row_copy["seed_rank"] = row["rank"]
        if seed_output_dir is not None:
            row_copy["seed_output_dir"] = str(seed_output_dir)
            row_copy["seed_candidate_pdb"] = str(seed_output_dir / f"candidate_{row['rank']:02d}.pdb")
            row_copy["seed_candidate_binder_pdb"] = str(seed_output_dir / f"candidate_{row['rank']:02d}_binder.pdb")
        annotated.append(row_copy)
    return annotated


def run_single_seed(
    args: argparse.Namespace,
    *,
    compute: ToolPool | None = None,
    log_config: bool = True,
) -> list[dict[str, Any]]:
    """Execute one seed and return its ranked summary rows."""
    program, binder, target, binder_seed, template_pdb_path = build_program(args, compute=compute)
    if log_config:
        log_run_configuration(args, binder_seed, target, template_pdb_path)
    program.run()
    summary = summarize_candidates(binder, program.energy_scores)

    for row in summary:
        logger.info(
            "rank=%d energy=%.4f avg_plddt=%s ptm=%s iptm=%s ablang_ll=%s seq=%s",
            row["rank"],
            row["energy"],
            row["avg_plddt"],
            row["ptm"],
            row["iptm"],
            row["ablang_log_likelihood"],
            row["sequence"],
        )

    if args.output_dir is not None:
        export_results(args.output_dir.resolve(), binder, summary)
        logger.info("Wrote outputs to %s", args.output_dir.resolve())

    return summary


def main(parser: argparse.ArgumentParser | None = None) -> None:
    """Build and optionally run the program."""
    args = parse_args(parser)
    configure_logging(args.debug)

    if args.proto_home is not None:
        os.environ["PROTO_HOME"] = str(args.proto_home.resolve())
    DeviceManager.get_instance().configure(allow_multiple_per_device=args.allow_multiple_per_device)
    seed_values = build_seed_values(args)

    if len(seed_values) == 1:
        if args.dry_run:
            log_dry_run_configuration(args)
            logger.info("Dry run complete. Program constructed successfully.")
            return
        run_single_seed(args)
        return

    preview_args = clone_args(args, seed=seed_values[0])
    log_dry_run_configuration(preview_args)
    log_seed_sweep_configuration(args, seed_values)

    if args.dry_run:
        logger.info("Dry run complete. Multi-seed sweep constructed successfully.")
        return

    all_rows: list[dict[str, Any]] = []
    shared_compute = ToolPool()
    with shared_compute:
        for seed_index, seed in enumerate(seed_values, start=1):
            seed_output_dir = args.output_dir.resolve() / f"seed_{seed:04d}" if args.output_dir is not None else None
            seed_args = clone_args(
                args,
                seed=seed,
                num_seeds=1,
                output_dir=seed_output_dir,
                dry_run=False,
            )
            logger.info("Starting seed %d/%d (seed=%d)", seed_index, len(seed_values), seed)
            summary = run_single_seed(seed_args, compute=shared_compute, log_config=False)
            all_rows.extend(annotate_seed_summary(summary, seed=seed, seed_output_dir=seed_output_dir))

    all_rows.sort(key=lambda row: row["energy"])
    for global_rank, row in enumerate(all_rows, start=1):
        row["rank"] = global_rank

    if args.output_dir is not None:
        export_sweep_results(args.output_dir.resolve(), all_rows)
        logger.info("Wrote aggregated sweep outputs to %s", args.output_dir.resolve())

    best = all_rows[0]
    logger.info(
        "Best overall: rank=%d seed=%d energy=%.4f avg_plddt=%s iptm=%s seq=%s",
        best["rank"],
        best["seed"],
        best["energy"],
        best["avg_plddt"],
        best["iptm"],
        best["sequence"],
    )


if __name__ == "__main__":
    main()
