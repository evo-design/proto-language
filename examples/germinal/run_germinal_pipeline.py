r"""Germinal antibody binder design pipeline — VHH (nanobody) and scFv modes.

Mirrors ``germinal/run_germinal.py`` + ``germinal/design/design.py`` for antibody
binder design against a target PDB. Three hallucination stages with post-stage
confidence gates, then an external pre-redesign cofold filter pass, then AbMPNN
redesign ranked + filtered by the YAML-selected cofold model.

Supported modes (``--mode``):
- ``vhh``  — single heavy-chain nanobody scaffold (``pdbs/nb.pdb``, 131 residues)
- ``scfv`` — single-chain Fv (VH + (G4S)3 linker + VL, ``pdbs/scfv.pdb``, 242 residues)

Supported Germinal presets (``--preset``):
- ``vhh`` / ``scfv`` for the generic Germinal antibody presets
- ``vhh_pdl1`` / ``scfv_pdl1`` for the PD-L1-specific Germinal overrides
- if ``--preset`` is omitted, the script defaults to ``{mode}_pdl1`` for backward compatibility
- generic ``vhh`` / ``scfv`` presets do not imply a target; this script requires explicit
  ``--target-pdb``, ``--target-chain``, and ``--target-hotspots`` for them, matching
  Germinal's separate ``target=...`` Hydra config group

Pipeline (same for both modes):

1. Stage 0 — ``GradientOptimizer`` (logit hallucination, mode-specific steps, gumbel init)
2. Gate — ``plddt > threshold`` AND ``iptm > threshold``
3. Stage 1 — ``GradientOptimizer`` (softmax refinement, 35 steps, temp 1 → 0.01)
4. Gate — adds ``i_pae < threshold``
5. Stage 2 — ``MCMCOptimizer`` + ``SemigreedyMutationGenerator`` (10 iters, greedy)
6. Gate — stage-2 confidence + hallucinated-structure checks
7. Gate — external cofold + FastRelax + initial Germinal filters (4/5 implemented)
8. Stage 3 — ``RejectionSamplingOptimizer`` + AbMPNN (40 samples) ranked by
   ``structure-composite`` with the YAML-selected cofold model, keeps top mode-specific candidates
9. Per-variant final filter: ``external_*`` thresholds + ``pdockq2 > 0.23``; PDB/FASTA/JSON saved

All numeric defaults come from the colocated script-owned preset file
``antibody_presets.yaml``.

Stage-3 divergence (quality > compute): cofolds all 40 AbMPNN samples and ranks
by ``structure-composite`` directly, instead of MPNN-log-prob pre-screening.

Known parity gaps (intentional):
- VHH external cofold uses Chai-1/AF3 fallbacks instead of Germinal's Protenix until
  proto-language/proto-tools can pass Protenix full-PAE outputs through final pDockQ2.
- Post-softmax entropy gate dropped (moot — Stage-3 AbMPNN rebuilds sequence).
- No global-best tracking across optimizer steps (near-monotonic for gradient
  phases; small impact for semigreedy MCMC at near-zero temperature).

Usage:
    # VHH against PD-L1 (default)
    python examples/germinal/run_germinal_pipeline.py --mode vhh --max-trajectories 10

    # scFv against PD-L1
    python examples/germinal/run_germinal_pipeline.py --mode scfv --max-trajectories 10

    # Generic VHH preset against an explicit target
    python examples/germinal/run_germinal_pipeline.py --preset vhh \
        --target-pdb path/to/target.pdb --target-chain A \
        --binder-scaffold-pdb path/to/scaffold.pdb --binder-scaffold-chain A \
        --target-hotspots A45,A47,A50 --max-trajectories 10
"""

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from Bio.PDB import PDBIO, PDBParser, Superimposer
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Structure import Structure as BioStructure
from proto_tools.utils.device_manager import DeviceManager
from proto_tools import (
    AlphaFold3Config,
    Chai1Config,
    InverseFoldingStructureInput,
    PDockQ2Config,
    PDockQ2Input,
    Structure,
    run_pdockq2,
)
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_interface_analyzer import (
    InterfaceStructureInput,
    PyRosettaInterfaceAnalyzerConfig,
    PyRosettaInterfaceAnalyzerInput,
    run_pyrosetta_interface_analyzer,
)
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import (
    PyRosettaRelaxConfig,
    PyRosettaRelaxInput,
    run_pyrosetta_relax,
)
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import ScoringStructureInput
from scipy.spatial import cKDTree

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
from proto_language.language.constraint.protein_structure.structure_confidence_constraint import (
    PAE_MAXIMUM,
    structure_composite_constraint,
)
from proto_language.language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.language.core import PROTEIN_AMINO_ACIDS, Constraint, Construct, Program, Segment, Sequence
from proto_language.utils import one_hot_protein_matrix
from proto_language.language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
)
from proto_language.language.optimizer import (
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)
from proto_language.language.optimizer.gradient_optimizer import ConstraintWeightSchedule

# =============================================================================
# Preset configuration (loaded from a consolidated script-owned YAML)
# =============================================================================

_PDB_DIR = Path(__file__).resolve().parent / "pdbs"
_PRESET_CONFIG_PATH = Path(__file__).resolve().with_name("antibody_presets.yaml")
_PRESET_NAMES = ("vhh", "scfv", "vhh_pdl1", "scfv_pdl1")


@dataclass(frozen=True)
class MetricRule:
    value: float
    operator: str

    def evaluate(self, observed: float) -> bool:
        if self.operator == ">":
            return observed > self.value
        if self.operator == ">=":
            return observed >= self.value
        if self.operator == "<":
            return observed < self.value
        if self.operator == "<=":
            return observed <= self.value
        if self.operator in ("==", "="):
            return observed == self.value
        raise ValueError(f"Unsupported operator {self.operator!r}.")


@dataclass
class BinderGeometry:
    preset_name: str
    vh_fw_lengths: list[int]
    vh_cdr_lengths: list[int]
    linker_length: int
    vl_fw_lengths: list[int]
    vl_cdr_lengths: list[int]
    plddt_threshold: float
    iptm_threshold: float
    ipae_threshold: float
    logits_steps: int
    softmax_steps: int
    search_steps: int
    search_mutation_rate: float
    plddt_loss_weight: float
    iptm_loss_weight: float
    beta_strand_loss_weight: float
    framework_contact_offset: float
    ablm_logit_start_weight: float
    ablm_logit_end_weight: float
    min_lr_scale: float
    zero_norm_eps: float
    lr_schedule: str | None
    ban_cysteine: bool
    cofold_tool: str
    num_seqs: int
    max_mpnn_sequences: int
    sampling_temp: float
    initial_filters: dict[str, MetricRule]
    final_filters: dict[str, MetricRule]
    default_scaffold_pdb: Path
    default_target_pdb: Path | None
    default_target_chain: str | None
    default_target_hotspots: str | None

    @property
    def vh_length(self) -> int:
        return sum(self.vh_fw_lengths) + sum(self.vh_cdr_lengths)

    @property
    def vl_length(self) -> int:
        return sum(self.vl_fw_lengths) + sum(self.vl_cdr_lengths)

    @property
    def binder_length(self) -> int:
        return self.vh_length + self.linker_length + self.vl_length

    @property
    def heavy_slice(self) -> tuple[int, int] | None:
        """AbLang VH slice; None for VHH."""
        return None if not self.vl_fw_lengths else (0, self.vh_length)

    @property
    def light_slice(self) -> tuple[int, int] | None:
        """AbLang VL slice; None for VHH."""
        if not self.vl_fw_lengths:
            return None
        start = self.vh_length + self.linker_length
        return (start, self.binder_length)

    def cdr_positions(self) -> list[int]:
        """Zero-indexed CDR positions across the full binder."""
        positions: list[int] = []
        offset = 0
        for fw_len, cdr_len in zip(self.vh_fw_lengths, self.vh_cdr_lengths, strict=False):
            offset += fw_len
            positions.extend(range(offset, offset + cdr_len))
            offset += cdr_len
        if self.vl_fw_lengths:
            offset = self.vh_length + self.linker_length
            for fw_len, cdr_len in zip(self.vl_fw_lengths, self.vl_cdr_lengths, strict=False):
                offset += fw_len
                positions.extend(range(offset, offset + cdr_len))
                offset += cdr_len
        return positions

    def vh_cdr3_positions(self) -> list[int]:
        """Zero-indexed VH CDR3 positions (3rd CDR for scFv)."""
        offset = sum(self.vh_fw_lengths[:3]) + sum(self.vh_cdr_lengths[:2])
        return list(range(offset, offset + self.vh_cdr_lengths[2]))


@dataclass(frozen=True)
class PreRedesignFilterMetrics:
    clashes: int
    cdr_hotspot_contacts: int
    cdr3_hotspot_contacts: int
    percent_interface_cdr: float
    interface_sc: float

    @property
    def binder_near_hotspot(self) -> bool:
        return self.cdr_hotspot_contacts >= MIN_CDR_HOTSPOT_CONTACTS


@dataclass
class StageMetrics:
    plddt: float = 0.0
    iptm: float = 0.0
    ipae: float = 0.0
    ablang_loss: float = 0.0


@dataclass
class TrajectoryRecord:
    traj_idx: int
    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    rejected_at: str | None = None
    accepted: bool = False


def _extract_stage_metrics(binder: "Segment") -> StageMetrics:
    """Extract current metrics from binder segment after a stage run."""
    result = binder.result_sequences[0]
    assert result.structure is not None  # noqa: S101 -- af2 always populates it on stages 0/1/2
    af2_data = result._constraints_metadata["af2"]["data"]
    return StageMetrics(
        plddt=float(np.mean(result.structure.per_residue_plddt)),
        iptm=float(af2_data["iptm"]),
        ipae=float(af2_data["i_pae"]),
        ablang_loss=float(result._constraints_metadata["ablang"]["score"]),
    )


def _plot_trajectory_dynamics(records: list[TrajectoryRecord], run_dir: str) -> None:
    """Generate per-metric charts with individual trajectory lines and bolded average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_names = ["stage0", "stage1", "stage2"]
    metric_names = ["plddt", "iptm", "ipae", "ablang_loss"]
    metric_labels = {"plddt": "pLDDT", "iptm": "iPTM", "ipae": "iPAE", "ablang_loss": "AbLang Loss"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_names):
        all_values: list[list[float]] = []
        for rec in records:
            vals = []
            for stage in stage_names:
                if stage in rec.stage_metrics:
                    vals.append(getattr(rec.stage_metrics[stage], metric))
                else:
                    break
            if vals:
                all_values.append(vals)
                x = list(range(len(vals)))
                alpha = 0.15 if len(records) > 10 else 0.3
                ax.plot(x, vals, color="steelblue", alpha=alpha, linewidth=0.8)

        if all_values:
            max_len = max(len(v) for v in all_values)
            avg_line = []
            for stage_i in range(max_len):
                stage_vals = [v[stage_i] for v in all_values if len(v) > stage_i]
                avg_line.append(np.nanmean(stage_vals))
            ax.plot(range(len(avg_line)), avg_line, color="black", linewidth=2.5, label="Average")
            ax.legend()

        ax.set_xticks(range(len(stage_names)))
        ax.set_xticklabels(["Stage 0\n(logit)", "Stage 1\n(softmax)", "Stage 2\n(MCMC)"])
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric])
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Trajectory Dynamics ({len(records)} trajectories)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plot_path = os.path.join(run_dir, "trajectory_dynamics.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory dynamics plot to {plot_path}")


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}.")
    return data


def _load_metric_rule(final_cfg: dict[str, Any], key: str) -> MetricRule:
    rule = final_cfg[key]
    if not isinstance(rule, dict) or "value" not in rule or "operator" not in rule:
        raise ValueError(f"Expected final filter {key!r} to define value and operator.")
    return MetricRule(value=float(rule["value"]), operator=str(rule["operator"]))


def _preset_mode(preset_name: str) -> str:
    if preset_name.startswith("vhh"):
        return "vhh"
    if preset_name.startswith("scfv"):
        return "scfv"
    raise ValueError(f"Unsupported preset {preset_name!r}.")


def _resolve_preset_name(args: argparse.Namespace) -> str:
    if args.preset is not None:
        preset_mode = _preset_mode(args.preset)
        if args.mode is not None and args.mode != preset_mode:
            raise ValueError(f"--mode={args.mode!r} is incompatible with --preset={args.preset!r}.")
        return args.preset
    return f"{(args.mode or 'vhh')}_pdl1"


def _resolve_target_inputs(args: argparse.Namespace, geom: BinderGeometry) -> tuple[Path, str, str]:
    if geom.default_target_pdb is None or geom.default_target_chain is None or geom.default_target_hotspots is None:
        missing = [
            flag
            for flag, value in (
                ("--target-pdb", args.target_pdb),
                ("--target-chain", args.target_chain),
                ("--target-hotspots", args.target_hotspots),
            )
            if value is None
        ]
        if missing:
            if len(missing) == 1:
                missing_flags = missing[0]
            elif len(missing) == 2:
                missing_flags = " and ".join(missing)
            else:
                missing_flags = ", ".join(missing[:-1]) + f", and {missing[-1]}"
            raise ValueError(
                f"--preset={geom.preset_name} is target-agnostic, like Germinal's generic run presets. "
                f"Pass explicit {missing_flags}."
            )
        return args.target_pdb, args.target_chain, args.target_hotspots
    return (
        args.target_pdb or geom.default_target_pdb,
        args.target_chain or geom.default_target_chain,
        args.target_hotspots or geom.default_target_hotspots,
    )


def _load_geometry(preset_name: str, preset_cfg: dict[str, Any]) -> BinderGeometry:
    # ``antibody_presets.yaml`` is already normalized to the subset this example
    # consumes, so the loader can map fields directly without recreating
    # Germinal's original Hydra/run/filter config stitching.
    default_target_cfg = preset_cfg.get("default_target")
    if default_target_cfg is None:
        default_target_pdb = None
        default_target_chain = None
        default_target_hotspots = None
    else:
        if not isinstance(default_target_cfg, dict):
            raise ValueError(f"{preset_name}.default_target must be a mapping or null.")
        default_target_pdb = _PDB_DIR / str(default_target_cfg["pdb"])
        default_target_chain = str(default_target_cfg["chain"])
        default_target_hotspots = str(default_target_cfg["hotspots"])

    initial_filters = {k: _load_metric_rule(preset_cfg["initial_filter"], k) for k in preset_cfg["initial_filter"]}
    final_filters = {k: _load_metric_rule(preset_cfg["final_filter"], k) for k in preset_cfg["final_filter"]}
    return BinderGeometry(
        preset_name=preset_name,
        vh_fw_lengths=[int(x) for x in preset_cfg["vh_fw_lengths"]],
        vh_cdr_lengths=[int(x) for x in preset_cfg["vh_cdr_lengths"]],
        linker_length=int(preset_cfg["linker_length"]),
        vl_fw_lengths=[int(x) for x in preset_cfg["vl_fw_lengths"]],
        vl_cdr_lengths=[int(x) for x in preset_cfg["vl_cdr_lengths"]],
        plddt_threshold=float(preset_cfg["plddt_threshold"]),
        iptm_threshold=float(preset_cfg["iptm_threshold"]),
        ipae_threshold=float(preset_cfg["ipae_threshold"]),
        logits_steps=int(preset_cfg["logits_steps"]),
        softmax_steps=int(preset_cfg["softmax_steps"]),
        search_steps=int(preset_cfg["search_steps"]),
        search_mutation_rate=float(preset_cfg["search_mutation_rate"]),
        plddt_loss_weight=float(preset_cfg["plddt_loss_weight"]),
        iptm_loss_weight=float(preset_cfg["iptm_loss_weight"]),
        beta_strand_loss_weight=float(preset_cfg["beta_strand_loss_weight"]),
        framework_contact_offset=float(preset_cfg["framework_contact_offset"]),
        ablm_logit_start_weight=float(preset_cfg["ablm_logit_start_weight"]),
        ablm_logit_end_weight=float(preset_cfg["ablm_logit_end_weight"]),
        min_lr_scale=float(preset_cfg["min_lr_scale"]),
        zero_norm_eps=float(preset_cfg.get("zero_norm_eps", 0.0)),
        lr_schedule=None if preset_cfg["lr_schedule"] is None else str(preset_cfg["lr_schedule"]),
        ban_cysteine=bool(preset_cfg["ban_cysteine"]),
        cofold_tool=str(preset_cfg["cofold_tool"]),
        num_seqs=int(preset_cfg["num_seqs"]),
        max_mpnn_sequences=int(preset_cfg["max_mpnn_sequences"]),
        sampling_temp=float(preset_cfg["sampling_temp"]),
        initial_filters=initial_filters,
        final_filters=final_filters,
        default_scaffold_pdb=_PDB_DIR / str(preset_cfg["default_scaffold_pdb"]),
        default_target_pdb=default_target_pdb,
        default_target_chain=default_target_chain,
        default_target_hotspots=default_target_hotspots,
    )


def _load_presets() -> dict[str, BinderGeometry]:
    config = _load_yaml_dict(_PRESET_CONFIG_PATH)
    presets = config.get("presets")
    if not isinstance(presets, dict):
        raise ValueError(f"{_PRESET_CONFIG_PATH} must define a top-level 'presets' mapping.")

    missing = [name for name in _PRESET_NAMES if name not in presets]
    if missing:
        raise ValueError(f"{_PRESET_CONFIG_PATH} is missing presets: {missing}.")

    return {name: _load_geometry(name, presets[name]) for name in _PRESET_NAMES}


GERMINAL_PRESETS = _load_presets()

# =============================================================================
# Constants
# =============================================================================

ATOM_DISTANCE_CUTOFF = 3.0
CLASH_THRESHOLD = 2.5
HOTSPOT_DISTANCE_THRESHOLD = 5.3
RESIDUE_CONTACT_DISTANCE = 6.0
MIN_CDR_HOTSPOT_CONTACTS = 3
GERMINAL_ABLANG_TEMPERATURE = 0.6
GERMINAL_LOGIT_SCALE = 2.0
SCAFFOLD_OFFSET = np.array([30.0, 30.0, 0.0])
STITCHED_BINDER_CHAIN = "B"
COFOLD_BINDER_CHAIN = "A"
COFOLD_TARGET_CHAIN = "B"
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}

_COFOLD_CONFIGS: dict[str, dict[str, Any]] = {
    "chai1": {"structure_tool": "chai1", "chai1_config": Chai1Config(include_pae_matrix=True).model_dump()},
    "alphafold3": {
        "structure_tool": "alphafold3",
        "alphafold3_config": AlphaFold3Config(include_pae_matrix=True).model_dump(),
    },
}


def _cofold_config(tool: str) -> dict[str, Any]:
    if tool in _COFOLD_CONFIGS:
        return _COFOLD_CONFIGS[tool]
    return {"structure_tool": tool}


# =============================================================================
# Main pipeline
# =============================================================================


def run_germinal_antibody(
    geom: BinderGeometry,
    target_pdb: Path,
    target_chain: str,
    target_hotspots: str | None,
    scaffold_pdb: Path,
    scaffold_chain: str,
    max_trajectories: int,
    max_passing: int,
    output_dir: str,
    no_filter: bool = False,
) -> None:
    """Run the Germinal antibody pipeline until ``max_passing`` designs are accepted."""
    target_chains = [chain_id.strip() for chain_id in target_chain.split(",") if chain_id.strip()]
    target_chain_label = ",".join(target_chains)
    binder_chain = STITCHED_BINDER_CHAIN
    target_name = target_pdb.stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"germinal/{target_name}/run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    cdr_str = "_".join(str(c) for c in geom.vh_cdr_lengths + geom.vl_cdr_lengths)
    stitched_pdb = Path(run_dir) / f"{target_name}_{cdr_str}_binder.pdb"
    stitch_starting_complex(
        target_pdb=target_pdb,
        target_chains=target_chains,
        scaffold_pdb=scaffold_pdb,
        scaffold_chain=scaffold_chain,
        binder_chain=binder_chain,
        save_path=stitched_pdb,
    )

    with stitched_pdb.open() as f:
        target_structure = Structure(structure=f.read(), structure_format="pdb")
    target_seq = "".join(
        target_structure.get_chain_sequence(chain_id, remove_non_standard=True) for chain_id in target_chains
    )
    binder_template = target_structure.get_chain_sequence(binder_chain, remove_non_standard=True)
    if len(binder_template) != geom.binder_length:
        raise ValueError(
            f"Scaffold chain {scaffold_chain!r} in {scaffold_pdb} has length {len(binder_template)}, "
            f"expected {geom.binder_length}."
        )

    print(
        f"Germinal[{geom.preset_name}] → {run_dir} | target={target_pdb}:{target_chain_label} "
        f"scaffold={scaffold_pdb}:{scaffold_chain}->{binder_chain} hotspots={target_hotspots} "
        f"len={geom.binder_length} (vh={geom.vh_length} linker={geom.linker_length} vl={geom.vl_length})"
    )

    num_accepted = 0
    all_records: list[TrajectoryRecord] = []
    for traj_idx in range(max_trajectories):
        print(f"\n--- Trajectory {traj_idx + 1}/{max_trajectories} ---")
        record = TrajectoryRecord(traj_idx=traj_idx)
        num_accepted += run_trajectory(
            geom,
            traj_idx,
            target_structure,
            target_seq,
            binder_template,
            target_chains,
            target_hotspots,
            binder_chain,
            run_dir,
            no_filter=no_filter,
            record=record,
        )
        all_records.append(record)
        if num_accepted >= max_passing:
            break

    def _nan_safe(v: float) -> float | None:
        return None if math.isnan(v) else v

    summary = {
        "num_trajectories": len(all_records),
        "num_accepted": num_accepted,
        "trajectories": [
            {
                "traj_idx": r.traj_idx,
                "rejected_at": r.rejected_at,
                "accepted": r.accepted,
                "stages": {
                    stage: {
                        "plddt": _nan_safe(m.plddt),
                        "iptm": _nan_safe(m.iptm),
                        "ipae": _nan_safe(m.ipae),
                        "ablang_loss": _nan_safe(m.ablang_loss),
                    }
                    for stage, m in r.stage_metrics.items()
                },
            }
            for r in all_records
        ],
    }
    with open(os.path.join(run_dir, "trajectory_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved trajectory summary to {run_dir}/trajectory_summary.json")

    _plot_trajectory_dynamics(all_records, run_dir)

    print(f"Done. {num_accepted} accepted design(s) in {run_dir}")


def run_trajectory(
    geom: BinderGeometry,
    traj_idx: int,
    target_structure: Structure,
    target_seq: str,
    binder_template: str,
    target_chains: list[str],
    target_hotspots: str | None,
    binder_chain: str,
    run_dir: str,
    no_filter: bool = False,
    record: TrajectoryRecord | None = None,
) -> int:
    """Run one Germinal trajectory; returns number of accepted variants saved."""
    np.random.seed(traj_idx)

    # CDR index sets
    cdr_positions = geom.cdr_positions()
    cdr_set = set(cdr_positions)
    cdr_positions_1idx = {p + 1 for p in cdr_positions}
    # VH CDR3 only — for scFv, Germinal uses cdr_lengths[:3]
    cdr3_positions_1idx = {p + 1 for p in geom.vh_cdr3_positions()}

    # --- Segments + construct ---
    binder = Segment(length=geom.binder_length, sequence_type="protein", label="binder")
    target = Segment(sequence=target_seq, sequence_type="protein", label="target")
    construct = Construct([binder, target])

    # ── AF2 configuration ──
    af2_cfg = AF2BinderConstraintConfig.germinal_vhh_preset(
        target_pdb=target_structure.structure_pdb, binder_chain=binder_chain
    )
    af2_cfg.seed = traj_idx
    af2_cfg.target_chains = target_chains
    af2_cfg.design_positions = cdr_positions
    af2_cfg.framework_contact_offset = geom.framework_contact_offset
    if target_hotspots:
        af2_cfg.target_hotspot = target_hotspots
    af2_cfg.loss_weights = {
        **af2_cfg.loss_weights,
        "plddt": geom.plddt_loss_weight,
        "i_ptm": geom.iptm_loss_weight,
        "beta_strand": geom.beta_strand_loss_weight,
    }
    if not geom.ban_cysteine:
        af2_cfg.omit_aas = None

    def af2() -> Constraint:
        return Constraint(
            inputs=[binder, target],
            label="af2",
            function=af2_binder_forward,
            backward=af2_binder_backward,
            function_config=af2_cfg,
            backward_config=af2_cfg,
        )

    handoff_bias = _germinal_handoff_bias(binder_template, cdr_set, ban_cysteine=geom.ban_cysteine)

    def ablang(weight: float | None = None) -> Constraint:
        cfg = AbLangConstraintConfig(
            temperature=GERMINAL_ABLANG_TEMPERATURE,
            heavy_slice=geom.heavy_slice,
            light_slice=geom.light_slice,
            logit_scale=GERMINAL_LOGIT_SCALE,
            logit_bias=handoff_bias,
        )
        return Constraint(
            inputs=[binder],
            label="ablang",
            weight=weight,
            function=ablang_naturalness_forward,
            backward=ablang_naturalness_gradient_backward,
            function_config=cfg,
            backward_config=cfg,
        )

    # ── STAGE 0 / 1: gradient hallucination ──
    logit_cfg = GradientOptimizerConfig.germinal_logit_preset()
    logit_cfg.num_steps = geom.logits_steps
    logit_cfg.zero_norm_eps = geom.zero_norm_eps
    logit_cfg.initial_logits = one_hot_protein_matrix(binder_template)
    logit_cfg.softmax_init_positions = cdr_positions
    logit_cfg.constraint_weight_schedules = [
        ConstraintWeightSchedule(
            constraint_label="ablang",
            start_weight=geom.ablm_logit_start_weight,
            end_weight=geom.ablm_logit_end_weight,
            schedule="hinge",
        )
    ]

    pwg_stage0 = PositionWeightGenerator(
        PositionWeightGeneratorConfig(
            logit_bias=handoff_bias,
            logit_scale=GERMINAL_LOGIT_SCALE,
        )
    )
    stage0 = GradientOptimizer(
        target_segment=binder,
        constructs=[construct],
        generators=[pwg_stage0],
        constraints=[af2(), ablang()],
        config=logit_cfg,
    )

    softmax_cfg = GradientOptimizerConfig.germinal_softmax_preset()
    softmax_cfg.num_steps = geom.softmax_steps
    softmax_cfg.min_lr_scale = geom.min_lr_scale
    softmax_cfg.zero_norm_eps = geom.zero_norm_eps
    if geom.lr_schedule is not None:
        softmax_cfg.lr_schedule = geom.lr_schedule
    pwg_stage1 = PositionWeightGenerator(
        PositionWeightGeneratorConfig(
            logit_bias=handoff_bias,
            logit_scale=GERMINAL_LOGIT_SCALE,
        )
    )
    stage1 = GradientOptimizer(
        target_segment=binder,
        constructs=[construct],
        generators=[pwg_stage1],
        constraints=[af2(), ablang(weight=0.4)],
        config=softmax_cfg,
    )

    # ── STAGE 2: semigreedy MCMC ──
    # Discards gradient logits, samples from bias alone (clear_logits=True).
    # Near-greedy argmin via max_temperature=1e-3.
    semigreedy = SemigreedyMutationGenerator(
        SemigreedyMutationGeneratorConfig(
            position_weighting="plddt",
            exclude_current=True,
            clear_logits=True,
            logit_bias=handoff_bias,
        ),
    )
    semigreedy.assign(binder)
    stage2 = MCMCOptimizer(
        constructs=[construct],
        generators=[semigreedy],
        constraints=[af2(), ablang(weight=1.0)],
        config=MCMCOptimizerConfig(
            num_steps=geom.search_steps,
            proposals_per_result=max(1, math.ceil(geom.binder_length * geom.search_mutation_rate)),
            max_temperature=2e-6,
            min_temperature=1e-6,
        ),
    )

    def _check_gate(gate_name: str, passed: bool, detail: str = "") -> bool:
        """Check a gate; returns True to continue, False to abort (when filters are on)."""
        if passed:
            print(f"[Traj {traj_idx}] {gate_name} passed" + (f" ({detail})" if detail else ""))
            return True
        if no_filter:
            print(f"[Traj {traj_idx}] {gate_name} FAILED (continuing, --no-filter)" + (f" ({detail})" if detail else ""))
            return True
        if record is not None:
            record.rejected_at = f"{gate_name}: {detail}" if detail else gate_name
        print(f"[Traj {traj_idx}] rejected at {gate_name}" + (f": {detail}" if detail else ""))
        return False

    hallucination = Program(optimizers=[stage0, stage1, stage2], num_results=1, seed=traj_idx)

    print(f"[Traj {traj_idx}] Stage 0: logit hallucination ({geom.logits_steps} steps)...")
    hallucination.run_stage(0)
    if record is not None:
        record.stage_metrics["stage0"] = _extract_stage_metrics(binder)
    if not _check_gate("stage0_gate", passes_gate(binder, geom=geom, include_ipae=False)):
        return 0

    print(f"[Traj {traj_idx}] Stage 1: softmax refinement ({geom.softmax_steps} steps)...")
    hallucination.run_stage(1)
    if record is not None:
        record.stage_metrics["stage1"] = _extract_stage_metrics(binder)
    if not _check_gate("stage1_gate", passes_gate(binder, geom=geom, include_ipae=True)):
        return 0

    print(f"[Traj {traj_idx}] Stage 2: semigreedy MCMC ({geom.search_steps} steps)...")
    hallucination.run_stage(2)
    if record is not None:
        record.stage_metrics["stage2"] = _extract_stage_metrics(binder)
    if not _check_gate("stage2_gate", passes_gate(binder, geom=geom, include_ipae=True)):
        return 0

    # ── POST-STAGE-2: structural gates ──
    binder_struct = binder.result_sequences[0].structure
    target_struct = target.result_sequences[0].structure
    assert binder_struct is not None and target_struct is not None  # noqa: S101 -- populated by af2 backward/forward
    complex_struct = Structure.concat([binder_struct, target_struct])

    print(f"[Traj {traj_idx}] Post-stage-2 structural checks...")
    clashes = complex_struct.ca_clash_score(threshold=CLASH_THRESHOLD)
    general_contacts = complex_struct.interface_contact_residues(
        binder_chain=binder_chain, target_chains=target_chains, cutoff=4.0, include_hydrogens=True
    )
    if not _check_gate(
        "structural_gate",
        clashes == 0 and len(general_contacts) >= 3,
        f"{clashes} clashes, {len(general_contacts)} contacts",
    ):
        return 0

    # Map target hotspots onto the external cofold chain layout.
    cofold_hotspots = remap_hotspots_to_cofold(
        target_hotspots,
        target_chains,
        COFOLD_TARGET_CHAIN,
        target_structure,
    )

    # MPNN fix-positions: non-CDR framework + interface residues
    interface_contacts = complex_struct.interface_contact_residues(
        binder_chain=binder_chain,
        target_chains=target_chains,
        cutoff=ATOM_DISTANCE_CUTOFF,
        include_hydrogens=True,
    )
    interface_residues = set(interface_contacts)

    print(f"[Traj {traj_idx}] Pre-redesign: external cofold ({geom.cofold_tool}) + FastRelax + filters...")
    pre_redesign_metrics = run_pre_redesign_external_filters(
        binder_sequence=binder.result_sequences[0].sequence,
        target_sequence=target.result_sequences[0].sequence,
        cofold_tool=geom.cofold_tool,
        cofold_hotspots=cofold_hotspots,
        cdr_positions_1idx=cdr_positions_1idx,
        cdr3_positions_1idx=cdr3_positions_1idx,
    )
    if not _check_gate(
        "pre_redesign_external_gate",
        passes_pre_redesign_external_gate(pre_redesign_metrics, geom=geom, traj_idx=traj_idx),
    ):
        return 0

    # ── STAGE 3: AbMPNN redesign ──
    # External pre-redesign filters are now applied on the separate cofolded structure.
    non_cdr_one_indexed = {i + 1 for i in range(geom.binder_length) if i not in cdr_set}
    fixed_positions = sorted(non_cdr_one_indexed | interface_residues)
    print(
        f"[Traj {traj_idx}] Stage 3 fixed: {len(non_cdr_one_indexed)} framework + "
        f"{len(interface_residues)} interface = {len(fixed_positions)} positions"
    )
    abmpnn = ProteinMPNNGenerator(
        ProteinMPNNGeneratorConfig(
            model_choice="abmpnn",
            temperature=geom.sampling_temp,
            excluded_amino_acids=["C"] if geom.ban_cysteine else [],
            structure_inputs=[
                InverseFoldingStructureInput(
                    structure=complex_struct,
                    chain_ids=[binder_chain],
                    fixed_positions={binder_chain: fixed_positions},
                )
            ],
        ),
    )
    abmpnn.assign(binder)
    stage3 = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[abmpnn],
        constraints=[
            Constraint(
                inputs=[binder, target],
                function=structure_composite_constraint,
                function_config=_cofold_config(geom.cofold_tool),
                label="cofold",
            )
        ],
        config=RejectionSamplingOptimizerConfig(
            num_samples=geom.num_seqs,
            num_results=geom.max_mpnn_sequences,
            samples_per_round=geom.num_seqs,
        ),
    )
    print(f"[Traj {traj_idx}] Stage 3: AbMPNN redesign ({geom.num_seqs} samples, keep top {geom.max_mpnn_sequences})...")
    Program(optimizers=[stage3], num_results=geom.max_mpnn_sequences, seed=traj_idx).run_stage(0)

    print(f"[Traj {traj_idx}] Final filter: relax + evaluate {len(geom.final_filters)} gates per variant...")
    # ── FINAL FILTER: relax + evaluate configured Germinal gates per variant ──
    accepted = 0
    for variant_idx, variant in enumerate(binder.result_sequences):
        data = variant._constraints_metadata["cofold"]["data"]
        plddt = float(data["composite_avg_plddt"])
        iptm = float(data["composite_iptm"])
        ptm = float(data["composite_ptm"])
        pae_norm = float(data["composite_avg_pae"])
        pae_angstroms = pae_norm * PAE_MAXIMUM
        assert variant.structure is not None  # noqa: S101 -- structure_composite_constraint always populates it
        cofold_struct = variant.structure

        pdockq2 = float(
            run_pdockq2(
                PDockQ2Input(
                    structure=cofold_struct,
                    binder_chain=COFOLD_BINDER_CHAIN,
                    target_chains=[COFOLD_TARGET_CHAIN],
                ),
                PDockQ2Config(),
            ).metrics.pdockq2
        )

        # FastRelax (1 cycle, matching Germinal)
        relax_result = run_pyrosetta_relax(
            PyRosettaRelaxInput(inputs=[ScoringStructureInput(structure=cofold_struct)]),
            PyRosettaRelaxConfig(relax_cycles=1, constrain_to_start=True),
        )
        relaxed_struct = relax_result.results[0].relax.relaxed_structure

        clashes = compute_interchain_clash_score(relaxed_struct, threshold=2.4)

        # RMSD on unrelaxed cofold (Germinal convention)
        sc_rmsd = compute_sc_rmsd(
            hallucinated_struct=complex_struct,
            cofolded_struct=cofold_struct,
            hall_target_chains=target_chains,
            hall_binder_chain=binder_chain,
            cofold_target_chain=COFOLD_TARGET_CHAIN,
            cofold_binder_chain=COFOLD_BINDER_CHAIN,
        )

        cofold_binder_near_hotspot = False
        cofold_cdr3_contacts = 0
        if cofold_hotspots:
            cofold_hotspot_hits = set(
                relaxed_struct.hotspot_contacts(
                    binder_chain=COFOLD_BINDER_CHAIN,
                    target_hotspots=cofold_hotspots,
                    expansion_cutoff=HOTSPOT_DISTANCE_THRESHOLD,
                    contact_cutoff=RESIDUE_CONTACT_DISTANCE,
                    germinal_mode=True,
                )
            )
            cofold_cdr_contacts = cofold_hotspot_hits & cdr_positions_1idx
            cofold_binder_near_hotspot = len(cofold_cdr_contacts) >= MIN_CDR_HOTSPOT_CONTACTS
            cofold_cdr3_contacts = len(cofold_hotspot_hits & cdr3_positions_1idx)

        cofold_interface_res = relaxed_struct.interface_contact_residues(
            binder_chain=COFOLD_BINDER_CHAIN, target_chains=[COFOLD_TARGET_CHAIN], cutoff=4.0, include_hydrogens=True
        )
        cofold_pct_iface_cdr = (
            len(set(cofold_interface_res) & cdr_positions_1idx) / len(cofold_interface_res)
            if cofold_interface_res
            else 0.0
        )

        iface_analysis = run_pyrosetta_interface_analyzer(
            PyRosettaInterfaceAnalyzerInput(
                inputs=[InterfaceStructureInput(
                    structure=relaxed_struct,
                    binder_chain=COFOLD_BINDER_CHAIN,
                    target_chain=COFOLD_TARGET_CHAIN,
                )]
            ),
            PyRosettaInterfaceAnalyzerConfig(),
        ).results[0]
        filter_values: dict[str, float] = {
            "external_plddt": plddt,
            "external_iptm": iptm,
            "external_ptm": ptm,
            "external_pae": pae_angstroms,
            "pdockq2": pdockq2,
            "clashes": float(clashes),
            "sc_rmsd": sc_rmsd,
            "binder_near_hotspot": 1.0 if cofold_binder_near_hotspot else 0.0,
            "cdr3_hotspot_contacts": float(cofold_cdr3_contacts),
            "percent_interface_cdr": cofold_pct_iface_cdr,
            "interface_hydrophobicity": float(iface_analysis.interface_hydrophobicity),
            "interface_sc": float(iface_analysis.interface_sc),
            "interface_hbonds": float(iface_analysis.interface_hbonds),
            "surface_hydrophobicity": float(iface_analysis.surface_hydrophobicity),
        }
        ok = all(rule.evaluate(filter_values[k]) for k, rule in geom.final_filters.items())
        if no_filter:
            ok = True
        status = "accepted" if ok else "redesign_candidate"
        if ok and record is not None:
            record.accepted = True
        metrics_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in filter_values.items())
        print(f"[Traj {traj_idx} / variant {variant_idx + 1}] {metrics_str} -> {status}")

        stem = f"traj{traj_idx:04d}_variant{variant_idx + 1}_{status}"
        with open(os.path.join(run_dir, f"{stem}.fasta"), "w") as f:
            f.write(f">binder\n{variant.sequence}\n>target\n{target_seq}\n")
        with open(os.path.join(run_dir, f"{stem}.json"), "w") as f:
            json.dump(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in filter_values.items()}
                | {"status": status},
                f,
                indent=2,
            )
        relaxed_struct.write_pdb(os.path.join(run_dir, f"{stem}.pdb"))
        accepted += int(ok)

    if accepted == 0 and record is not None and record.rejected_at is None:
        record.rejected_at = "final_filter"

    return accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Germinal antibody binder design (VHH or scFv).")
    parser.add_argument(
        "--mode",
        choices=["vhh", "scfv"],
        default=None,
        help="Binder mode. Used to choose the default preset when --preset is omitted.",
    )
    parser.add_argument(
        "--preset",
        choices=list(_PRESET_NAMES),
        default=None,
        help="Exact Germinal preset name. If omitted, defaults to {mode}_pdl1 (or vhh_pdl1 when --mode is unset).",
    )
    parser.add_argument(
        "--target-pdb",
        type=Path,
        default=None,
        help="Target PDB. Required for generic vhh/scfv presets; defaults to pdbs/pdl1.pdb for *_pdl1 presets.",
    )
    parser.add_argument(
        "--target-chain",
        default=None,
        help="Target chain(s). Required for generic vhh/scfv presets; defaults to A for *_pdl1 presets.",
    )
    parser.add_argument(
        "--target-hotspots",
        default=None,
        help="Comma-separated target residues. Required for generic vhh/scfv presets; defaults to PD-L1 hotspots.",
    )
    parser.add_argument(
        "--binder-scaffold-pdb",
        type=Path,
        default=None,
        help="Free-standing scaffold PDB. Defaults to pdbs/nb.pdb (vhh) or pdbs/scfv.pdb (scfv).",
    )
    parser.add_argument("--binder-scaffold-chain", default="A", help="Scaffold chain inside --binder-scaffold-pdb.")
    parser.add_argument("--max-trajectories", type=int, default=10)
    parser.add_argument("--max-passing", type=int, default=1)
    parser.add_argument(
        "--logits-steps",
        type=int,
        default=None,
        help="Optional override for the preset's Stage-0 gradient steps.",
    )
    parser.add_argument(
        "--softmax-steps",
        type=int,
        default=None,
        help="Optional override for the preset's Stage-1 gradient steps.",
    )
    parser.add_argument(
        "--search-steps",
        type=int,
        default=None,
        help="Optional override for the preset's Stage-2 semigreedy steps.",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=None,
        help="Optional override for the preset's number of AbMPNN redesign samples.",
    )
    parser.add_argument(
        "--max-mpnn-sequences",
        type=int,
        default=None,
        help="Optional override for the preset's number of top redesign variants kept.",
    )
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument(
        "--share-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep AF2 (~3GB) and AbLang (~3GB) loaded simultaneously. Disable with --no-share-gpu on low-memory GPUs.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable all inter-stage gates and final filters. All trajectories run through all stages (significantly slower since no trajectories are short-circuited); metrics are still collected and saved.",
    )
    return parser.parse_args()


def _apply_runtime_overrides(args: argparse.Namespace, geom: BinderGeometry) -> BinderGeometry:
    """Apply optional CLI overrides without mutating the shared preset object."""
    updates: dict[str, int] = {}
    if args.logits_steps is not None:
        updates["logits_steps"] = args.logits_steps
    if args.softmax_steps is not None:
        updates["softmax_steps"] = args.softmax_steps
    if args.search_steps is not None:
        updates["search_steps"] = args.search_steps
    if args.num_seqs is not None:
        updates["num_seqs"] = args.num_seqs
    if args.max_mpnn_sequences is not None:
        updates["max_mpnn_sequences"] = args.max_mpnn_sequences
    return replace(geom, **updates) if updates else geom


def main() -> None:
    args = parse_args()
    preset_name = _resolve_preset_name(args)
    geom = _apply_runtime_overrides(args, GERMINAL_PRESETS[preset_name])
    target_pdb, target_chain, target_hotspots = _resolve_target_inputs(args, geom)
    scaffold_pdb = args.binder_scaffold_pdb or geom.default_scaffold_pdb
    DeviceManager.get_instance().configure(allow_multiple_per_device=args.share_gpu)
    run_germinal_antibody(
        geom=geom,
        target_pdb=target_pdb,
        target_chain=target_chain,
        target_hotspots=target_hotspots,
        scaffold_pdb=scaffold_pdb,
        scaffold_chain=args.binder_scaffold_chain,
        max_trajectories=args.max_trajectories,
        max_passing=args.max_passing,
        output_dir=args.output_dir,
        no_filter=args.no_filter,
    )


# =============================================================================
# Pipeline helpers
# =============================================================================


def passes_gate(binder: Segment, *, geom: BinderGeometry, include_ipae: bool) -> bool:
    """Confidence gate: binder-only pLDDT + iPTM, optionally iPAE."""
    result = binder.result_sequences[0]
    data = result._constraints_metadata["af2"]["data"]
    assert result.structure is not None  # noqa: S101 -- af2 backward always populates it
    plddt = float(np.mean(result.structure.per_residue_plddt))
    iptm = float(data["iptm"])
    ipae = float(data["i_pae"])
    if plddt <= geom.plddt_threshold or iptm <= geom.iptm_threshold:
        print(f"  gate: plddt={plddt:.3f} (need>{geom.plddt_threshold}) iptm={iptm:.3f} (need>{geom.iptm_threshold}) ipae={ipae:.2f}")
        return False
    if include_ipae and ipae >= geom.ipae_threshold:
        print(f"  gate: ipae={ipae:.2f} (need<{geom.ipae_threshold}) plddt={plddt:.3f} iptm={iptm:.3f}")
        return False
    return True


def _germinal_handoff_bias(template_seq: str, cdr_set: set[int], *, ban_cysteine: bool) -> list[list[float]]:
    """Germinal static bias: keep framework near template and optionally ban Cys in CDRs."""
    vocab = list(PROTEIN_AMINO_ACIDS)
    aa_to_idx = {aa: idx for idx, aa in enumerate(vocab)}
    bias = np.zeros((len(template_seq), len(vocab)), dtype=np.float64)
    for pos, aa in enumerate(template_seq):
        if pos in cdr_set:
            if ban_cysteine:
                bias[pos, aa_to_idx["C"]] -= 1e6
        else:
            bias[pos, aa_to_idx[aa]] = 10.0
    return bias.tolist()


def stitch_starting_complex(
    *,
    target_pdb: Path,
    target_chains: list[str],
    scaffold_pdb: Path,
    scaffold_chain: str,
    binder_chain: str,
    save_path: Path,
) -> None:
    """Graft scaffold onto target at +30 Å offset, renaming scaffold to ``binder_chain``."""
    if binder_chain in target_chains:
        raise ValueError(
            f"Binder chain {binder_chain!r} must not also be a target chain. "
            "Rename the colliding target chain in the input PDB or choose a different stitched binder chain."
        )
    parser = PDBParser(QUIET=True)
    target = parser.get_structure("target", str(target_pdb))
    scaffold = parser.get_structure("scaffold", str(scaffold_pdb))
    missing_target_chains = [chain_id for chain_id in target_chains if chain_id not in target[0]]
    if missing_target_chains:
        raise ValueError(f"Target chain(s) {missing_target_chains} not found in {target_pdb}.")
    if scaffold_chain not in scaffold[0]:
        raise ValueError(f"Scaffold chain {scaffold_chain!r} not found in {scaffold_pdb}.")

    merged = BioStructure("merged")
    model = BioModel(0)
    merged.add(model)
    for chain_id in target_chains:
        model.add(target[0][chain_id].copy())
    binder = scaffold[0][scaffold_chain].copy()
    binder.id = binder_chain
    for atom in binder.get_atoms():
        atom.set_coord(atom.coord + SCAFFOLD_OFFSET)
    model.add(binder)
    io = PDBIO()
    io.set_structure(merged)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(save_path))


# =============================================================================
# Filter helpers
# =============================================================================


def run_pre_redesign_external_filters(
    *,
    binder_sequence: str,
    target_sequence: str,
    cofold_tool: str,
    cofold_hotspots: str,
    cdr_positions_1idx: set[int],
    cdr3_positions_1idx: set[int],
) -> PreRedesignFilterMetrics:
    """Run Germinal's extra external cofold + relax + initial-filter stage."""
    eval_binder = Sequence(sequence=binder_sequence, sequence_type="protein")
    eval_target = Sequence(sequence=target_sequence, sequence_type="protein")
    structure_composite_constraint(
        [(eval_binder, eval_target)],
        StructureBasedConstraintConfig.model_validate(_cofold_config(cofold_tool)),
    )
    assert eval_binder.structure is not None  # noqa: S101 -- structure_composite_constraint always populates it
    cofold_struct = eval_binder.structure

    relax_result = run_pyrosetta_relax(
        PyRosettaRelaxInput(inputs=[ScoringStructureInput(structure=cofold_struct)]),
        PyRosettaRelaxConfig(relax_cycles=1, constrain_to_start=True),
    )
    relaxed_struct = relax_result.results[0].relax.relaxed_structure

    hotspot_hits = set(
        relaxed_struct.hotspot_contacts(
            binder_chain=COFOLD_BINDER_CHAIN,
            target_hotspots=cofold_hotspots,
            expansion_cutoff=HOTSPOT_DISTANCE_THRESHOLD,
            contact_cutoff=RESIDUE_CONTACT_DISTANCE,
            germinal_mode=True,
        )
    )
    interface_res = relaxed_struct.interface_contact_residues(
        binder_chain=COFOLD_BINDER_CHAIN,
        target_chains=[COFOLD_TARGET_CHAIN],
        cutoff=4.0,
        include_hydrogens=True,
    )
    percent_interface_cdr = (
        len(set(interface_res) & cdr_positions_1idx) / len(interface_res) if interface_res else 0.0
    )

    iface_result = run_pyrosetta_interface_analyzer(
        PyRosettaInterfaceAnalyzerInput(
            inputs=[InterfaceStructureInput(
                structure=relaxed_struct,
                binder_chain=COFOLD_BINDER_CHAIN,
                target_chain=COFOLD_TARGET_CHAIN,
            )]
        ),
        PyRosettaInterfaceAnalyzerConfig(),
    )

    return PreRedesignFilterMetrics(
        clashes=compute_interchain_clash_score(relaxed_struct, threshold=2.4),
        cdr_hotspot_contacts=len(hotspot_hits & cdr_positions_1idx),
        cdr3_hotspot_contacts=len(hotspot_hits & cdr3_positions_1idx),
        percent_interface_cdr=percent_interface_cdr,
        interface_sc=float(iface_result.results[0].interface_sc),
    )


def _apply_filter_gates(
    rules: dict[str, MetricRule], values: dict[str, float], traj_idx: int, stage: str
) -> bool:
    """Evaluate a dict of MetricRules against observed values; print and return False on first failure."""
    for name, rule in rules.items():
        if name not in values:
            raise ValueError(f"Filter '{name}' has no corresponding value in the {stage} metrics.")
        if not rule.evaluate(values[name]):
            print(f"[Traj {traj_idx}] rejected at {stage} {name} gate: {values[name]}")
            return False
    return True


def passes_pre_redesign_external_gate(metrics: PreRedesignFilterMetrics, *, geom: BinderGeometry, traj_idx: int) -> bool:
    """Apply Germinal's initial external filters."""
    values = {
        "clashes": float(metrics.clashes),
        "binder_near_hotspot": 1.0 if metrics.binder_near_hotspot else 0.0,
        "cdr3_hotspot_contacts": float(metrics.cdr3_hotspot_contacts),
        "percent_interface_cdr": metrics.percent_interface_cdr,
        "interface_sc": metrics.interface_sc,
    }
    return _apply_filter_gates(geom.initial_filters, values, traj_idx, "external")


def compute_interchain_clash_score(structure: Structure, threshold: float = 2.4) -> int:
    """Inter-chain heavy-atom clash count (excludes same-chain pairs)."""
    gs = structure.gemmi_struct
    coords: list[list[float]] = []
    chain_ids: list[str] = []
    for model in gs:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == "H" or atom.element.name == "D":
                        continue
                    pos = atom.pos
                    coords.append([pos.x, pos.y, pos.z])
                    chain_ids.append(chain.name)
    if len(coords) < 2:
        return 0
    tree = cKDTree(coords)
    pairs = tree.query_pairs(threshold)
    return sum(1 for i, j in pairs if chain_ids[i] != chain_ids[j])


def compute_sc_rmsd(
    hallucinated_struct: Structure,
    cofolded_struct: Structure,
    hall_target_chains: list[str],
    hall_binder_chain: str,
    cofold_target_chain: str,
    cofold_binder_chain: str,
) -> float:
    """Binder backbone RMSD after target-CA superposition."""
    parser = PDBParser(QUIET=True)

    def _parse(struct: Structure, label: str) -> BioStructure:
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(struct.structure_pdb)
            tmp = f.name
        try:
            return parser.get_structure(label, tmp)
        finally:
            os.unlink(tmp)

    hall_bio = _parse(hallucinated_struct, "hall")
    cofold_bio = _parse(cofolded_struct, "cofold")

    def _ca_atoms(bio_struct: BioStructure, chain_ids: list[str]) -> list:
        atoms = []
        chains = {chain.id: chain for chain in bio_struct[0]}
        for chain_id in chain_ids:
            chain = chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found in structure (available: {list(chains.keys())})")
            atoms.extend(
                atom for residue in chain for atom in residue if atom.get_name() == "CA"
            )
        return atoms

    def _bb_atoms(bio_struct: BioStructure, chain_id: str) -> list:
        return [
            atom
            for chain in bio_struct[0]
            if chain.id == chain_id
            for residue in chain
            for atom in residue
            if atom.get_name() in _BACKBONE_ATOMS
        ]

    hall_target_ca = _ca_atoms(hall_bio, hall_target_chains)
    cofold_target_ca = _ca_atoms(cofold_bio, [cofold_target_chain])
    n_align = min(len(hall_target_ca), len(cofold_target_ca))
    if n_align < 3:
        raise ValueError(f"Too few CA atoms for superposition: hall={len(hall_target_ca)}, cofold={len(cofold_target_ca)}")

    sup = Superimposer()
    sup.set_atoms(hall_target_ca[:n_align], cofold_target_ca[:n_align])

    hall_bb = _bb_atoms(hall_bio, hall_binder_chain)
    cofold_bb = _bb_atoms(cofold_bio, cofold_binder_chain)
    n_bb = min(len(hall_bb), len(cofold_bb))
    if n_bb == 0:
        raise ValueError(f"No backbone atoms found: hall chain {hall_binder_chain}={len(hall_bb)}, cofold chain {cofold_binder_chain}={len(cofold_bb)}")

    sup.apply(cofold_bb[:n_bb])
    diff = np.array([h.get_vector().get_array() - c.get_vector().get_array()
                     for h, c in zip(hall_bb[:n_bb], cofold_bb[:n_bb])])
    return round(float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))), 2)


def remap_hotspots_to_cofold(
    hotspots: str,
    hall_target_chains: list[str],
    cofold_target_chain: str,
    hall_target_structure: Structure,
) -> str:
    """Remap hallucinated target hotspots onto the cofold target chain."""
    residue_map = hall_target_structure.get_residue_position_map()
    cofold_positions: dict[tuple[str, int], int] = {}
    next_position = 1
    for chain_id in hall_target_chains:
        for aa, residue_number in residue_map.get(chain_id, []):
            if aa not in {"X", "-"}:
                cofold_positions[(chain_id, residue_number)] = next_position
                next_position += 1

    tokens = []
    for raw in hotspots.split(","):
        token = raw.strip()
        if not token:
            continue
        chain_id, residue_text = token[0], token[1:]
        if chain_id not in hall_target_chains:
            raise ValueError(f"Hotspot {token!r} is not on target chains {hall_target_chains}.")
        if not residue_text.lstrip("-").isdigit():
            raise ValueError(f"Hotspot {token!r} must be chain-prefixed like 'A45'.")
        cofold_position = cofold_positions.get((chain_id, int(residue_text)))
        if cofold_position is None:
            raise ValueError(f"Hotspot {token!r} is not present in target chains {hall_target_chains}.")
        tokens.append(f"{cofold_target_chain}{cofold_position}")
    return ",".join(tokens)


if __name__ == "__main__":
    main()
