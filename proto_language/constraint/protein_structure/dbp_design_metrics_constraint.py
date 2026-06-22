"""dbp_design-driven structural metrics constraint for protein-DNA binders.

Scores a predicted protein-operator complex with the heavy ``dbp_design`` metric
suite (PyRosetta H-bond typing, compactness, interface ddG / CMS / shape
complementarity / packstat / buried unsats / RotamerBoltzmann, and notebook-
aligned charge ratios). The complex PDB is resolved once per candidate via the
shared structure resolver, handed to PyRosetta + ``dbp_design`` scripts, and the
extracted metrics are turned into a single weighted-composite penalty in
``[0, 1]`` where ``0`` is best and ``1`` is worst. Every metric and per-component
penalty is preserved verbatim in metadata for post-hoc thresholding.

This is the heaviest DNA-binding constraint: it requires PyRosetta and a local
``dbp_design`` checkout, so it only executes where PyRosetta is provisioned. The
pure scoring / threshold / H-bond-weighting helpers are PyRosetta-free.

Constraints:
- dbp-design-metrics: Score protein-DNA designs with the dbp_design metric suite.

Examples:
    Score an AF3 protein-operator complex with dbp_design metrics:

    >>> from proto_language.core import Segment
    >>> protein = Segment(length=100, sequence_type="protein")
    >>> operator = Segment(length=20, sequence_type="dna")
    >>> dbp_metrics = Constraint(
    ...     inputs=[protein, operator],
    ...     function=dbp_design_metrics_constraint,
    ...     function_config={
    ...         "structure_tool": "alphafold3",
    ...         "min_base_score": 10.0,
    ...         "max_ddg": -15.0,
    ...     },
    ... )
"""

import contextlib
import importlib.util
import logging
import math
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np

from proto_language.constraint.constraint_registry import constraint
from proto_language.constraint.protein_structure.dna_binding_structure_helper import (
    resolve_structure_paths,
)
from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils.base import ConfigField

logger = logging.getLogger(__name__)

_BASE_SCORE_WEIGHTS = {
    "ARG_g_hbonds": 5.0,
    "LYS_g_hbonds": 3.0,
    "ASP_g_hbonds": -3.0,
    "GLU_g_hbonds": -3.0,
    "ASN_g_hbonds": 1.0,
    "GLN_g_hbonds": 2.0,
    "SER_g_hbonds": 0.0,
    "THR_g_hbonds": -3.0,
    "TYR_g_hbonds": -3.0,
    "HIS_g_hbonds": 1.0,
    "ARG_c_hbonds": -3.0,
    "LYS_c_hbonds": 0.0,
    "ASP_c_hbonds": 7.0,
    "GLU_c_hbonds": 4.0,
    "ASN_c_hbonds": 3.0,
    "GLN_c_hbonds": 0.0,
    "SER_c_hbonds": 0.0,
    "THR_c_hbonds": 0.0,
    "TYR_c_hbonds": -3.0,
    "HIS_c_hbonds": 0.0,
    "ARG_a_hbonds": -3.0,
    "LYS_a_hbonds": -3.0,
    "ASP_a_hbonds": -3.0,
    "GLU_a_hbonds": -3.0,
    "ASN_a_hbonds": 5.0,
    "GLN_a_hbonds": 5.0,
    "SER_a_hbonds": 0.0,
    "THR_a_hbonds": -3.0,
    "TYR_a_hbonds": 0.0,
    "HIS_a_hbonds": 0.0,
    "ARG_t_hbonds": 1.0,
    "LYS_t_hbonds": 1.0,
    "ASP_t_hbonds": -3.0,
    "GLU_t_hbonds": -3.0,
    "ASN_t_hbonds": 1.0,
    "GLN_t_hbonds": 1.0,
    "SER_t_hbonds": -3.0,
    "THR_t_hbonds": -3.0,
    "TYR_t_hbonds": -3.0,
    "HIS_t_hbonds": -3.0,
}
_PHOSPHATE_SCORE_WEIGHTS = {
    "ARG_phosphate_hbonds": 7.0,
    "LYS_phosphate_hbonds": 0.0,
    "GLN_phosphate_hbonds": 10.0,
    "TYR_phosphate_hbonds": 4.0,
    "ASN_phosphate_hbonds": 1.0,
    "SER_phosphate_hbonds": 1.0,
    "THR_phosphate_hbonds": 1.0,
    "HIS_phosphate_hbonds": 1.0,
}
_BIDENTATE_SCORE_WEIGHTS = {
    "ARG_g_bidentates": 1.0,
    "ASN_a_bidentates": 1.0,
    "GLN_a_bidentates": 1.0,
}

_MODULE_CACHE: dict[tuple[str, str], object] = {}
_INTERFACE_FILTER_CACHE: dict[str, dict[str, Any]] = {}

# Shipped, calibrated ddG/CMS maximum-likelihood prefilter assets (dbp_design).
_DEFAULT_PREFILTER_EQ_PATH = str(Path(__file__).resolve().parents[3] / "examples" / "data" / "dbp_prefilter_eq.txt")
_DEFAULT_PREFILTER_CUT_PATH = str(Path(__file__).resolve().parents[3] / "examples" / "data" / "dbp_prefilter_cut.txt")


class DBPDesignMetricsConfig(StructureBasedConstraintConfig):
    """Config for the dbp-design-metrics constraint.

    Predicts (or reuses) a protein-DNA complex and scores it with the heavy
    ``dbp_design`` metric suite (PyRosetta H-bond typing, compactness, interface
    ddG / CMS / shape complementarity / packstat / buried unsats /
    RotamerBoltzmann, and notebook-aligned charge ratios). The score is the
    weight-normalized sum of per-metric penalties clamped to ``[0, 1]``, where
    ``0`` means every threshold is satisfied and ``1`` means none are.

    Attributes:
        dbp_design_repo_path (str): Path to the local dbp_design repository root (required, resolved from this config).
        count_hbond_script (str): Relative path to the count_hbond_types.py script.
        compactness_script (str): Relative path to the compactness_filter.py script.
        pyrosetta_site_packages (str): Optional site-packages path containing pyrosetta (resolved from this config).
        fail_hard (bool): If true, raise on dbp metric tool failures instead of soft-failing.
        failure_score (float): Penalty score returned when evaluation fails in fail-soft mode.
        prefilter_eq_path (str | None): Path to the calibrated ddG/CMS sigmoid prefilter equation (None skips prefilter).
        prefilter_cut_path (str | None): Path to the prefilter log-prob cutoff file (used when prefilter_cut is None).
        prefilter_cut (float | None): Explicit prefilter log-prob cutoff; when None it is loaded from prefilter_cut_path.
        hbond_energy_cutoff (float): HBond energy cutoff passed to count_hbond_types.
        min_base_score (float): Minimum weighted base score required.
        min_phosphate_score (float): Minimum weighted phosphate score required.
        min_bidentate_score (float): Minimum weighted bidentate score required.
        min_backbone_phosphate_contacts (int): Minimum backbone-phosphate H-bond contacts.
        min_compactness_contacts (int): Minimum per-SSE contacts from the compactness filter.
        max_loop_length (int): Maximum allowed loop length from the compactness filter.
        require_motif_in_rec_helix (bool): Require the motif_in_rec_helix metric to be true.
        require_rifres_in_rec_helix (bool): Require the rifres_in_rec_helix metric to be true.
        max_buried_unsats (float): Maximum allowed buried unsatisfied polar atoms at interface.
        min_shape_complementarity (float): Minimum interface shape complementarity (Sc) score.
        min_packstat (float): Minimum packstat score for foldability/packing quality.
        min_max_rboltz_rkqe (float): Minimum max RotamerBoltzmann score over interface RKQE residues.
        min_avg_top_two_rboltz (float): Minimum avg_top_two_rboltz score.
        max_ddg (float): Maximum allowed interface ddG (more negative is better).
        min_contact_molecular_surface (float): Minimum required contact molecular surface (CMS).
        min_net_charge_over_sasa (float): Lower bound of the allowed net_charge_over_sasa window.
        max_net_charge_over_sasa (float): Upper bound of the allowed net_charge_over_sasa window.
        max_ddg_over_cms (float): Maximum allowed ddg/contact_molecular_surface ratio.
        substantial_divergence_ratio (float): Normalized distance at which heavy penalties saturate.
        base_weight (float): Weight for the base score penalty component.
        phosphate_weight (float): Weight for the phosphate score penalty component.
        bidentate_weight (float): Weight for the bidentate score penalty component.
        backbone_weight (float): Weight for the backbone phosphate contact penalty.
        compactness_weight (float): Weight for the compactness contact penalty.
        loop_weight (float): Weight for the loop-length penalty component.
        motif_weight (float): Weight for the motif-in-recognition-helix check.
        rifres_weight (float): Weight for the rifres-in-recognition-helix check.
        buried_unsat_weight (float): Weight for the buried-unsatisfied-polars penalty.
        shape_complementarity_weight (float): Weight for the shape-complementarity penalty.
        packstat_weight (float): Weight for the packstat penalty.
        rboltz_weight (float): Weight for the max-RKQE RotamerBoltzmann preorganization penalty.
        avg_top_two_rboltz_weight (float): Weight for the avg-top-two RotamerBoltzmann preorganization penalty.
        ddg_weight (float): Weight for the ddG notebook-alignment penalty.
        contact_molecular_surface_weight (float): Weight for the CMS notebook-alignment penalty.
        net_charge_over_sasa_weight (float): Weight for the net_charge_over_sasa penalty.
        ddg_over_cms_weight (float): Weight for the ddg_over_cms notebook-alignment penalty.
        max_mpnn_score (float): Maximum allowed MPNN score when available in candidate metadata.
        missing_mpnn_score_penalty (float): Penalty used when MPNN score is unavailable.
        mpnn_score_weight (float): Weight for the MPNN score threshold penalty.
        structure_tool (Literal['esmfold', 'esmfold2', 'alphafold3', 'boltz2', 'chai1', 'protenix', 'alphafold2', 'alphafold2_binder']): DNA-capable structure-prediction tool (default alphafold3).
        esmfold_config (ESMFoldConfig): ESMFold config (used when structure_tool="esmfold").
        esmfold2_config (ESMFold2Config): ESMFold2 config (used when structure_tool="esmfold2").
        alphafold3_config (AlphaFold3Config): AlphaFold3 config (used when structure_tool="alphafold3").
        boltz2_config (Boltz2Config): Boltz2 config (used when structure_tool="boltz2").
        chai1_config (Chai1Config): Chai1 config (used when structure_tool="chai1").
        protenix_config (ProtenixConfig): Protenix config (used when structure_tool="protenix").
        alphafold2_config (AlphaFold2Config): AlphaFold2 config (used when structure_tool="alphafold2").
        alphafold2_binder_config (AlphaFold2BinderStructureConfig): AF2 binder config (alphafold2_binder).
    """

    structure_tool: Literal[
        "esmfold", "esmfold2", "alphafold3", "boltz2", "chai1", "protenix", "alphafold2", "alphafold2_binder"
    ] = ConfigField(
        title="Structure Prediction Tool",
        default="alphafold3",
        description="Predictor for the protein-DNA complex; must be DNA-capable (alphafold3/boltz2/protenix).",
    )
    dbp_design_repo_path: str = ConfigField(
        title="dbp_design Repo",
        default="",
        description="Path to local dbp_design repository root; required (no default).",
    )
    count_hbond_script: str = ConfigField(
        title="Count HBond Script",
        default="2b_design_mpnn/count_hbond_types.py",
        description="Relative path to count_hbond_types.py script.",
    )
    compactness_script: str = ConfigField(
        title="Compactness Script",
        default="2b_design_mpnn/compactness_filter.py",
        description="Relative path to compactness_filter.py script.",
    )
    pyrosetta_site_packages: str = ConfigField(
        title="PyRosetta Site-Packages",
        default="",
        description="Optional site-packages path containing pyrosetta when not in the active env.",
    )
    fail_hard: bool = ConfigField(
        title="Fail Hard",
        default=False,
        description="If true, raise on dbp metric tool failures instead of returning failure_score.",
    )
    failure_score: float = ConfigField(
        title="Failure Score",
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Penalty score returned when dbp metric evaluation fails in fail-soft mode.",
    )
    prefilter_eq_path: str | None = ConfigField(
        title="Prefilter Eq Path",
        default=_DEFAULT_PREFILTER_EQ_PATH,
        description="Path to calibrated ddG/CMS sigmoid prefilter equation file; None skips the prefilter.",
    )
    prefilter_cut_path: str | None = ConfigField(
        title="Prefilter Cut Path",
        default=_DEFAULT_PREFILTER_CUT_PATH,
        description="Path to prefilter log-prob cutoff file; used when prefilter_cut is None.",
    )
    prefilter_cut: float | None = ConfigField(
        title="Prefilter Cut",
        default=None,
        description="Explicit prefilter log-prob cutoff; when None it is loaded from prefilter_cut_path.",
    )
    hbond_energy_cutoff: float = ConfigField(
        title="HBond Energy Cutoff",
        default=-0.5,
        description="HBond energy cutoff passed to count_hbond_types.",
    )
    min_base_score: float = ConfigField(
        title="Min Base Score",
        default=10.0,
        description="Minimum weighted base score required.",
    )
    min_phosphate_score: float = ConfigField(
        title="Min Phosphate Score",
        default=0.0,
        description="Minimum weighted phosphate score required.",
    )
    min_bidentate_score: float = ConfigField(
        title="Min Bidentate Score",
        default=1.0,
        description="Minimum weighted bidentate score required.",
    )
    min_backbone_phosphate_contacts: int = ConfigField(
        title="Min BB-Phosphate",
        default=0,
        ge=0,
        description="Minimum backbone-phosphate H-bond contacts.",
    )
    min_compactness_contacts: int = ConfigField(
        title="Min Compactness",
        default=0,
        ge=0,
        description="Minimum per-SSE contacts from compactness filter (default makes this penalty component inactive).",
    )
    max_loop_length: int = ConfigField(
        title="Max Loop Length",
        default=1000000,
        ge=0,
        description="Maximum allowed loop length from compactness filter (default makes this penalty component inactive).",
    )
    require_motif_in_rec_helix: bool = ConfigField(
        title="Require Motif in RH",
        default=False,
        description="Require motif_in_rec_helix metric to be true.",
    )
    require_rifres_in_rec_helix: bool = ConfigField(
        title="Require RIFRES in RH",
        default=False,
        description="Require rifres_in_rec_helix metric to be true.",
    )
    max_buried_unsats: float = ConfigField(
        title="Max Buried Unsats",
        default=2.0,
        ge=0.0,
        description="Maximum allowed buried unsatisfied polar atoms at the interface.",
    )
    min_shape_complementarity: float = ConfigField(
        title="Min Shape Complementarity",
        default=0.65,
        ge=0.0,
        description="Minimum interface shape complementarity score (Sc).",
    )
    min_packstat: float = ConfigField(
        title="Min PackStat",
        default=0.55,
        ge=0.0,
        description="Minimum packstat score for foldability/packing quality.",
    )
    min_max_rboltz_rkqe: float = ConfigField(
        title="Min Max Rotamer Boltzmann",
        default=0.15,
        ge=0.0,
        description="Minimum max RotamerBoltzmann-style score over interface RKQE residues.",
    )
    min_avg_top_two_rboltz: float = ConfigField(
        title="Min Avg Top-Two RBoltz",
        default=0.1,
        ge=0.0,
        description="Minimum avg_top_two_rboltz score.",
    )
    max_ddg: float = ConfigField(
        title="Max ddG",
        default=-15.0,
        description="Maximum allowed interface ddG (more negative is better).",
    )
    min_contact_molecular_surface: float = ConfigField(
        title="Min Contact Mol Surface",
        default=225.0,
        ge=0.0,
        description="Minimum required contact molecular surface (CMS).",
    )
    min_net_charge_over_sasa: float = ConfigField(
        title="Min Net Charge Over SASA",
        default=-10.0,
        description="Lower bound of the allowed net_charge_over_sasa window.",
    )
    max_net_charge_over_sasa: float = ConfigField(
        title="Max Net Charge Over SASA",
        default=10.0,
        description="Upper bound of the allowed net_charge_over_sasa window.",
    )
    max_ddg_over_cms: float = ConfigField(
        title="Max ddG Over CMS",
        default=-0.06,
        description="Maximum allowed ddg/contact_molecular_surface ratio.",
    )
    substantial_divergence_ratio: float = ConfigField(
        title="Substantial Divergence",
        default=0.5,
        gt=0.0,
        description="Normalized distance from threshold at which heavy penalties reach full penalty.",
    )
    base_weight: float = ConfigField(
        title="Base Weight",
        default=1.0,
        ge=0.0,
        description="Weight for base score penalty component.",
    )
    phosphate_weight: float = ConfigField(
        title="Phosphate Weight",
        default=1.0,
        ge=0.0,
        description="Weight for phosphate score penalty component.",
    )
    bidentate_weight: float = ConfigField(
        title="Bidentate Weight",
        default=1.0,
        ge=0.0,
        description="Weight for bidentate score penalty component.",
    )
    backbone_weight: float = ConfigField(
        title="Backbone Weight",
        default=1.0,
        ge=0.0,
        description="Weight for backbone phosphate contact penalty.",
    )
    compactness_weight: float = ConfigField(
        title="Compactness Weight",
        default=1.0,
        ge=0.0,
        description="Weight for compactness contact penalty.",
    )
    loop_weight: float = ConfigField(
        title="Loop Weight",
        default=1.0,
        ge=0.0,
        description="Weight for loop-length penalty component.",
    )
    motif_weight: float = ConfigField(
        title="Motif Weight",
        default=1.0,
        ge=0.0,
        description="Weight for motif-in-recognition-helix check.",
    )
    rifres_weight: float = ConfigField(
        title="RIFRES Weight",
        default=1.0,
        ge=0.0,
        description="Weight for rifres-in-recognition-helix check.",
    )
    buried_unsat_weight: float = ConfigField(
        title="Buried Unsat Weight",
        default=1.0,
        ge=0.0,
        description="Weight for buried-unsatisfied-polars penalty.",
    )
    shape_complementarity_weight: float = ConfigField(
        title="Shape Complement Weight",
        default=1.0,
        ge=0.0,
        description="Weight for shape-complementarity penalty.",
    )
    packstat_weight: float = ConfigField(
        title="PackStat Weight",
        default=1.0,
        ge=0.0,
        description="Weight for packstat penalty.",
    )
    rboltz_weight: float = ConfigField(
        title="Rotamer Boltzmann Weight",
        default=1.0,
        ge=0.0,
        description="Weight for max-RKQE RotamerBoltzmann preorganization penalty.",
    )
    avg_top_two_rboltz_weight: float = ConfigField(
        title="Avg Top-Two RBoltz Weight",
        default=1.0,
        ge=0.0,
        description="Weight for avg-top-two RotamerBoltzmann preorganization penalty.",
    )
    ddg_weight: float = ConfigField(
        title="ddG Weight",
        default=2.0,
        ge=0.0,
        description="Weight for ddG notebook-alignment penalty.",
    )
    contact_molecular_surface_weight: float = ConfigField(
        title="CMS Weight",
        default=2.0,
        ge=0.0,
        description="Weight for CMS notebook-alignment penalty.",
    )
    net_charge_over_sasa_weight: float = ConfigField(
        title="Net Charge Over SASA Weight",
        default=2.0,
        ge=0.0,
        description="Weight for net_charge_over_sasa notebook-alignment penalty.",
    )
    ddg_over_cms_weight: float = ConfigField(
        title="ddG/CMS Weight",
        default=2.0,
        ge=0.0,
        description="Weight for ddg_over_cms notebook-alignment penalty.",
    )
    max_mpnn_score: float = ConfigField(
        title="Max MPNN Score",
        default=2.0,
        description="Maximum allowed MPNN score when available in candidate metadata.",
    )
    missing_mpnn_score_penalty: float = ConfigField(
        title="Missing MPNN Penalty",
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Penalty used when MPNN score is unavailable in candidate metadata.",
    )
    mpnn_score_weight: float = ConfigField(
        title="MPNN Score Weight",
        default=1.0,
        ge=0.0,
        description="Weight for MPNN score threshold penalty.",
    )


def _split_search_paths(path_string: str) -> list[str]:
    """Split OS path-list style strings into non-empty entries."""
    return [path for path in str(path_string or "").split(os.pathsep) if path]


def _resolve_repo_root(config: DBPDesignMetricsConfig) -> Path:
    """Resolve the dbp_design repository root, raising when it is unset or missing."""
    if not str(config.dbp_design_repo_path).strip():
        raise ValueError("dbp-design-metrics requires 'dbp_design_repo_path'; set it to a local dbp_design checkout.")
    repo_root = Path(config.dbp_design_repo_path).expanduser().resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"dbp_design repo not found at 'dbp_design_repo_path': {repo_root}")
    return repo_root


def _candidate_pyrosetta_paths(config: DBPDesignMetricsConfig) -> list[str]:
    """Resolve candidate sys.path entries for locating pyrosetta."""
    candidates: list[str] = []
    candidates.extend(_split_search_paths(config.pyrosetta_site_packages))
    candidates.extend(_split_search_paths(os.environ.get("PYROSETTA_SITE_PACKAGES", "")))

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        normalized = str(Path(candidate).expanduser())
        if normalized in seen:
            continue
        if Path(normalized).exists():
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


@contextmanager
def _prepend_sys_path(paths: list[str]) -> Iterator[None]:
    """Temporarily prepend import paths during module loading."""
    inserted: list[str] = []
    for path in reversed(paths):
        if path and path not in sys.path:
            sys.path.insert(0, path)
            inserted.append(path)
    try:
        yield
    finally:
        for path in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


def _load_module(module_path: Path, module_name: str) -> Any:
    """Load a module from a concrete file path."""
    cache_key = (str(module_path.resolve()), module_name)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    if not module_path.exists():
        raise FileNotFoundError(f"Required dbp_design module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def _load_dbp_modules(config: DBPDesignMetricsConfig) -> tuple[Any, Any]:
    """Load dbp_design helper scripts with fail-hard behavior."""
    repo_root = _resolve_repo_root(config)
    count_hbond_path = repo_root / config.count_hbond_script
    compactness_path = repo_root / config.compactness_script

    with _prepend_sys_path(_candidate_pyrosetta_paths(config)):
        count_hbond = _load_module(count_hbond_path, "dbp_design_count_hbond_types")
        compactness = _load_module(compactness_path, "dbp_design_compactness_filter")
    return count_hbond, compactness


def _load_interface_filters(config: DBPDesignMetricsConfig) -> dict[str, Any]:
    """Load/cached dbp_design interface filters used for ddG/CMS and related terms."""
    repo_root = _resolve_repo_root(config)
    cache_key = str(repo_root)
    if cache_key in _INTERFACE_FILTER_CACHE:
        return _INTERFACE_FILTER_CACHE[cache_key]

    xml_loader_path = repo_root / "2b_design_mpnn" / "xml_loader.py"
    weights_file = repo_root / "2b_design_mpnn" / "flags_and_weights" / "RM8B_torsional.wts"
    relax_script = (
        repo_root / "2b_design_mpnn" / "flags_and_weights" / "no_ref.rosettacon2018.beta_nov16_constrained.txt"
    )
    if not xml_loader_path.exists():
        raise FileNotFoundError(f"dbp_design xml_loader missing: {xml_loader_path}")
    if not weights_file.exists():
        raise FileNotFoundError(f"dbp_design weights missing: {weights_file}")
    if not relax_script.exists():
        raise FileNotFoundError(f"dbp_design relax script missing: {relax_script}")

    with _prepend_sys_path(_candidate_pyrosetta_paths(config)):
        import pyrosetta  # type: ignore[import-not-found]

        xml_loader = _load_module(xml_loader_path, "dbp_design_xml_loader")
        (
            _pack_no_design,
            _softish_min,
            _hard_min,
            _fast_relax,
            ddg_filter,
            cms_filter,
            xml_interface_sc_filter,
            xml_packstat_filter,
            _vbuns_filter_unused,
            _sbuns_filter_unused,
            net_charge_filter,
            net_charge_over_sasa_filter,
        ) = xml_loader.fast_relax(
            pyrosetta.rosetta.protocols,
            str(weights_file),
            "/software/psipred4/runpsipred_single",
            str(relax_script),
        )

        # Prefer xml-loaded filters if available, otherwise create fresh
        sc_filter = (
            xml_interface_sc_filter
            if xml_interface_sc_filter is not None
            else (pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter())
        )
        if hasattr(sc_filter, "jump_id"):
            sc_filter.jump_id(1)
        packstat_filter = (
            xml_packstat_filter
            if xml_packstat_filter is not None
            else (pyrosetta.rosetta.protocols.simple_filters.PackStatFilter())
        )
        buns_filter = pyrosetta.rosetta.protocols.simple_filters.BuriedUnsatHbondFilter()
        buns_filter.set_report_all_heavy_atom_unsats(True)
        buns_filter.set_use_reporter_behavior(True)

        scorefxn = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(str(weights_file))

    filters: dict[str, Any] = {
        "ddg_filter": ddg_filter,
        "cms_filter": cms_filter,
        "net_charge_filter": net_charge_filter,
        "net_charge_over_sasa_filter": net_charge_over_sasa_filter,
        "sc_filter": sc_filter,
        "packstat_filter": packstat_filter,
        "buns_filter": buns_filter,
        "scorefxn": scorefxn,
    }
    _INTERFACE_FILTER_CACHE[cache_key] = filters
    return filters


def _is_dna_residue_name(name: str) -> bool:
    """Return true if residue name token corresponds to DNA/RNA nucleotides."""
    token = str(name).upper().strip()
    return token in {"DA", "DC", "DG", "DT", "A", "C", "G", "T", "U"}


def _interface_protein_selector(pyrosetta_module: Any) -> Any:
    """Build selector for interface residues on protein chain (assumed chain 1)."""
    rs = pyrosetta_module.rosetta.core.select.residue_selector
    chain_a = rs.ChainSelector("1")
    chain_b = rs.NotResidueSelector(chain_a)
    interface_ch_a = rs.NeighborhoodResidueSelector(chain_b, 14.0, False)
    return rs.AndResidueSelector(interface_ch_a, chain_a)


def _interface_protein_hbond_residues(
    pose: Any,
    hbond_energy_cutoff: float,
) -> list[int]:
    """Collect protein residue indices making protein-DNA HBonds."""
    hbonds = pose.get_hbonds()
    residue_indices = set()
    for hbond_idx in range(1, hbonds.nhbonds() + 1):
        hbond = hbonds.hbond(hbond_idx)
        if float(hbond.energy()) > float(hbond_energy_cutoff):
            continue
        donor_res = hbond.don_res()
        acceptor_res = hbond.acc_res()
        donor_is_dna = _is_dna_residue_name(pose.residue(donor_res).name().split(":")[0])
        acceptor_is_dna = _is_dna_residue_name(pose.residue(acceptor_res).name().split(":")[0])
        if donor_is_dna == acceptor_is_dna:
            continue
        protein_res = acceptor_res if donor_is_dna else donor_res
        if pose.residue(protein_res).is_protein():
            residue_indices.add(int(protein_res))
    return sorted(residue_indices)


def _rboltz_summary(
    pose: Any,
    scorefxn: Any,
    residue_indices: list[int],
) -> dict[str, float]:
    """Compute dbp-style RotamerBoltzmann summary metrics."""
    if not residue_indices:
        return {"max_rboltz_rkqe": 0.0, "avg_top_two_rboltz": 0.0}

    import pyrosetta

    rkqe = {"R", "K", "Q", "E"}
    rb_by_residue: dict[int, float] = {}
    for res_idx in residue_indices:
        residue_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(str(int(res_idx)))
        not_selector = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(residue_selector)
        prevent_repack = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
        lock_non_target = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
            prevent_repack,
            not_selector,
        )

        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(lock_non_target)

        rboltz = pyrosetta.rosetta.protocols.calc_taskop_filters.RotamerBoltzmannWeight()
        rboltz.scorefxn(scorefxn)
        rboltz.task_factory(tf)
        rboltz.skip_ala_scan(1)
        rboltz.no_modified_ddG(1)
        rb_by_residue[int(res_idx)] = float(rboltz.compute(pose))

    if not rb_by_residue:
        return {"max_rboltz_rkqe": 0.0, "avg_top_two_rboltz": 0.0}

    all_values = sorted(rb_by_residue.values())
    top_n = all_values[: min(2, len(all_values))]
    avg_top_two_rboltz = float(-np.mean(top_n))

    rkqe_values = [value for res_idx, value in rb_by_residue.items() if pose.residue(res_idx).name1() in rkqe]
    max_rboltz_rkqe = float(-min(rkqe_values)) if rkqe_values else 0.0
    return {
        "max_rboltz_rkqe": max_rboltz_rkqe,
        "avg_top_two_rboltz": avg_top_two_rboltz,
    }


def _load_pose_from_pdb(pdb_path: str, config: DBPDesignMetricsConfig) -> Any:
    """Load a PyRosetta pose from a PDB path."""
    with _prepend_sys_path(_candidate_pyrosetta_paths(config)):
        try:
            import pyrosetta
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyrosetta is required for dbp-design-metrics but is not importable. "
                "Install pyrosetta in the active env or set "
                "'pyrosetta_site_packages' (or PYROSETTA_SITE_PACKAGES)."
            ) from exc

    if hasattr(pyrosetta, "is_initialized") and not pyrosetta.is_initialized():
        pyrosetta.init("-mute all")
    return pyrosetta.pose_from_pdb(str(pdb_path))


def _protein_only_pose(pose: Any) -> Any:
    """Return a clone of ``pose`` with non-protein (DNA/RNA) residues removed.

    Protein-only filters (compactness / secondary-structure) call ``residue.xyz("CA")``,
    which raises on nucleotide residues (e.g. ``GUA`` has no ``CA``). Those filters must
    therefore run on the protein chains alone, while protein-DNA H-bond and
    phosphate-contact counters keep using the full complex.
    """
    protein_pose = pose.clone()
    for i in range(protein_pose.size(), 0, -1):
        if not protein_pose.residue(i).is_protein():
            protein_pose.delete_residue_slow(i)
    return protein_pose


def _count_backbone_phosphate_contacts(pose: Any) -> int:
    """Count backbone donor hydrogens contacting DNA phosphate oxygens."""
    hbonds = pose.get_hbonds()
    contacts = []
    for hbond_idx in range(1, hbonds.nhbonds() + 1):
        hbond = hbonds.hbond(hbond_idx)
        donor_res = hbond.don_res()
        acceptor_res = hbond.acc_res()
        donor_hatm = hbond.don_hatm()
        acceptor_atm = hbond.acc_atm()
        don_atom = pose.residue(donor_res).atom_name(donor_hatm).strip()
        acc_atom = pose.residue(acceptor_res).atom_name(acceptor_atm).strip()
        if acc_atom in {"OP1", "OP2", "O5'"} and don_atom == "H":
            contacts.append(donor_res)
    return len(contacts)


def _weighted_sum(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted sum over available metrics."""
    return float(sum(float(metrics.get(key, 0.0)) * w for key, w in weights.items()))


def _compute_dbp_metrics_for_pdb(
    pdb_path: str,
    config: DBPDesignMetricsConfig,
) -> dict[str, float]:
    """Compute dbp_design metrics for one structure."""
    count_hbond, compactness = _load_dbp_modules(config)
    pose = _load_pose_from_pdb(pdb_path, config)
    interface_filters = _load_interface_filters(config)
    columns, result = count_hbond.count_hbonds_protein_dna(
        pose,
        Path(pdb_path).stem,
        config.hbond_energy_cutoff,
    )
    hbond_metrics = dict(zip(columns, result, strict=False))

    # Compactness/secondary-structure analysis is protein-only (it reads CA atoms), so it
    # must run on the protein chains alone — DNA residues (e.g. GUA) have no CA.
    min_ncontacts, max_loop_length = compactness.filter(_protein_only_pose(pose))
    bb_contacts = _count_backbone_phosphate_contacts(pose)
    with _prepend_sys_path(_candidate_pyrosetta_paths(config)):
        import pyrosetta
    interface_selector = _interface_protein_selector(pyrosetta)
    buns_filter = interface_filters["buns_filter"]
    buns_filter.set_residue_selector(interface_selector)
    sc_results = interface_filters["sc_filter"].compute(pose)
    interface_hbond_resis = _interface_protein_hbond_residues(pose, config.hbond_energy_cutoff)
    rboltz_summary = _rboltz_summary(
        pose,
        interface_filters["scorefxn"],
        interface_hbond_resis,
    )
    ddg = float(interface_filters["ddg_filter"].compute(pose))
    contact_molecular_surface = float(interface_filters["cms_filter"].compute(pose))
    ddg_over_cms = float("inf") if abs(contact_molecular_surface) < 1e-8 else float(ddg / contact_molecular_surface)

    return {
        "base_score": _weighted_sum(hbond_metrics, _BASE_SCORE_WEIGHTS),
        "phosphate_score": _weighted_sum(hbond_metrics, _PHOSPHATE_SCORE_WEIGHTS),
        "bidentate_score": _weighted_sum(hbond_metrics, _BIDENTATE_SCORE_WEIGHTS),
        "total_hbonds": float(hbond_metrics.get("total_hbonds", 0.0)),
        "n_backbone_phosphate_contacts": float(bb_contacts),
        "min_compactness_contacts": float(min_ncontacts),
        "max_loop_length": float(max_loop_length),
        "motif_in_rec_helix": float(bool(hbond_metrics.get("motif_in_rec_helix", False))),
        "rifres_in_rec_helix": float(bool(hbond_metrics.get("rifres_in_rec_helix", False))),
        "ddg": ddg,
        "contact_molecular_surface": contact_molecular_surface,
        "net_charge": float(interface_filters["net_charge_filter"].compute(pose)),
        "net_charge_over_sasa": float(interface_filters["net_charge_over_sasa_filter"].compute(pose)),
        "ddg_over_cms": ddg_over_cms,
        "shape_complementarity": float(sc_results.sc),
        "packstat": float(interface_filters["packstat_filter"].compute(pose)),
        "buried_unsats": float(buns_filter.compute(pose)),
        "max_rboltz_rkqe": float(rboltz_summary["max_rboltz_rkqe"]),
        "avg_top_two_rboltz": float(rboltz_summary["avg_top_two_rboltz"]),
    }


def _read_first_line(path: str) -> str:
    """Return the first line of a small trusted text asset (newline preserved)."""
    with open(path) as handle:
        return handle.readline()


def _prefilter_logp(eq_text: str, ddg_over_cms: float, contact_molecular_surface: float) -> float:
    """Evaluate the calibrated ddG/CMS sigmoid prefilter and return ``log10(p)``.

    Mirrors dbp_design ``prefilter_preemption``: the shipped equation is the
    negated product of two ``1/(1+exp(...))`` sigmoids over the truncated feature
    names ``ddg_ove`` (ddg_over_cms) and ``contact_molecular_su`` (CMS). The first
    line of the equation file is quote-wrapped, so we strip the surrounding quotes,
    substitute the two feature values (space-delimited, as in the source), and
    evaluate ``log10(-eval(eq))`` over a constrained namespace exposing only
    ``exp``/``EXP``. Non-finite results (overflow / divide-by-zero) collapse to
    ``-100.0``, matching the source's fail-closed fallback.

    Args:
        eq_text (str): First line of the shipped prefilter equation file.
        ddg_over_cms (float): The candidate ``ddg_over_cms`` ratio.
        contact_molecular_surface (float): The candidate contact molecular surface.

    Returns:
        float: ``log10(p)`` for the design, or ``-100.0`` on evaluation failure.
    """
    # Non-finite features can't be substituted into the equation (the 'inf'/'nan'
    # tokens raise inside eval); fail closed explicitly, matching the source.
    if not math.isfinite(ddg_over_cms) or not math.isfinite(contact_molecular_surface):
        return -100.0
    # The shipped first line is wrapped in double quotes (and a trailing newline);
    # strip the leading quote and the trailing quote (+ optional newline), matching
    # dbp_design's ``get_first_line(f)[1:-2]``.
    eq = eq_text.rstrip("\n").removeprefix('"').removesuffix('"')
    # KEEP THE SPACES: the source substitutes ``f' {feature} '`` with the value so
    # truncated feature names never partially match each other.
    eq = eq.replace(" ddg_ove ", f" {float(ddg_over_cms)} ")
    eq = eq.replace(" contact_molecular_su ", f" {float(contact_molecular_surface)} ")
    namespace: dict[str, Any] = {"__builtins__": {}, "exp": np.exp, "EXP": np.exp}
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        try:
            product = float(eval(eq, namespace))  # noqa: S307 - trusted shipped asset, sandboxed namespace
            log_prob = float(np.log10(-product))
        except Exception:
            return -100.0
    if not math.isfinite(log_prob):
        return -100.0
    return log_prob


def _resolve_prefilter_cut(config: DBPDesignMetricsConfig) -> float:
    """Resolve the prefilter log-prob cutoff from config or the shipped cut file."""
    if config.prefilter_cut is not None:
        return float(config.prefilter_cut)
    if config.prefilter_cut_path is None:
        raise ValueError(
            "dbp-design-metrics prefilter is enabled but neither 'prefilter_cut' nor 'prefilter_cut_path' is set."
        )
    return float(_read_first_line(config.prefilter_cut_path).strip())


def _evaluate_prefilter(
    config: DBPDesignMetricsConfig,
    ddg_over_cms: float,
    contact_molecular_surface: float,
) -> tuple[bool, float, float]:
    """Run the ddG/CMS prefilter, returning ``(passed, log_prob, log_prob_cut)``.

    A design passes iff ``log_prob >= log_prob_cut`` (equivalently
    ``-log10(p) <= -log_prob_cut``), matching dbp_design ``prefilter_preemption``.
    """
    if config.prefilter_eq_path is None:
        raise ValueError("prefilter is disabled; _evaluate_prefilter must not be called.")
    eq_text = _read_first_line(config.prefilter_eq_path)
    log_prob = _prefilter_logp(eq_text, ddg_over_cms, contact_molecular_surface)
    log_prob_cut = _resolve_prefilter_cut(config)
    return (log_prob >= log_prob_cut, log_prob, log_prob_cut)


def _failed_dbp_metrics() -> dict[str, float]:
    """Placeholder metrics used when dbp evaluation fails in fail-soft mode."""
    nan = float("nan")
    return {
        "base_score": nan,
        "phosphate_score": nan,
        "bidentate_score": nan,
        "total_hbonds": nan,
        "n_backbone_phosphate_contacts": nan,
        "min_compactness_contacts": nan,
        "max_loop_length": nan,
        "motif_in_rec_helix": 0.0,
        "rifres_in_rec_helix": 0.0,
        "ddg": nan,
        "contact_molecular_surface": nan,
        "net_charge": nan,
        "net_charge_over_sasa": nan,
        "ddg_over_cms": nan,
        "shape_complementarity": nan,
        "packstat": nan,
        "buried_unsats": nan,
        "max_rboltz_rkqe": nan,
        "avg_top_two_rboltz": nan,
    }


def _failed_penalties() -> dict[str, float]:
    """Per-component penalties used when dbp evaluation fails in fail-soft mode."""
    return {
        "base": 1.0,
        "phosphate": 1.0,
        "bidentate": 1.0,
        "backbone": 1.0,
        "compactness": 1.0,
        "loop": 1.0,
        "motif": 1.0,
        "rifres": 1.0,
        "buried_unsats": 1.0,
        "shape_complementarity": 1.0,
        "packstat": 1.0,
        "rboltz": 1.0,
        "avg_top_two_rboltz": 1.0,
        "ddg": 1.0,
        "contact_molecular_surface": 1.0,
        "net_charge_over_sasa": 1.0,
        "ddg_over_cms": 1.0,
        "mpnn_score": 1.0,
    }


def _penalty_min(value: float, threshold: float) -> float:
    """Penalty for a minimum threshold metric."""
    if not math.isfinite(value):
        return 1.0
    if value >= threshold:
        return 0.0
    denom = max(abs(threshold), 1.0)
    return float(np.clip((threshold - value) / denom, 0.0, 1.0))


def _penalty_max(value: float, threshold: float) -> float:
    """Penalty for a maximum threshold metric."""
    if not math.isfinite(value):
        return 1.0
    if value <= threshold:
        return 0.0
    denom = max(abs(threshold), 1.0)
    return float(np.clip((value - threshold) / denom, 0.0, 1.0))


def _heavy_penalty(normalized_delta: float, saturation_ratio: float) -> float:
    """Quadratic penalty that saturates to 1 for substantial threshold divergence."""
    if normalized_delta <= 0.0:
        return 0.0
    scaled = float(normalized_delta / max(saturation_ratio, 1e-8))
    return float(np.clip(scaled * scaled, 0.0, 1.0))


def _heavy_penalty_min(value: float, threshold: float, saturation_ratio: float) -> float:
    """Strongly penalize values that fall below a minimum threshold."""
    if not math.isfinite(value):
        return 1.0
    if value >= threshold:
        return 0.0
    denom = max(abs(threshold), 1.0)
    normalized_delta = float((threshold - value) / denom)
    return _heavy_penalty(normalized_delta, saturation_ratio)


def _heavy_penalty_max(value: float, threshold: float, saturation_ratio: float) -> float:
    """Strongly penalize values that exceed a maximum threshold."""
    if not math.isfinite(value):
        return 1.0
    if value <= threshold:
        return 0.0
    denom = max(abs(threshold), 1.0)
    normalized_delta = float((value - threshold) / denom)
    return _heavy_penalty(normalized_delta, saturation_ratio)


def _compute_penalties(
    metrics: dict[str, float],
    config: DBPDesignMetricsConfig,
) -> dict[str, float]:
    """Convert extracted dbp metrics into per-component penalties in ``[0, 1]``.

    ``metrics`` must include an ``mpnn_score`` key (``nan`` when unavailable).
    This is a pure function of metrics and config; it never touches PyRosetta.
    """
    return {
        "base": _penalty_min(metrics["base_score"], config.min_base_score),
        "phosphate": _penalty_min(metrics["phosphate_score"], config.min_phosphate_score),
        "bidentate": _penalty_min(metrics["bidentate_score"], config.min_bidentate_score),
        "backbone": _penalty_min(
            metrics["n_backbone_phosphate_contacts"],
            float(config.min_backbone_phosphate_contacts),
        ),
        "compactness": _penalty_min(
            metrics["min_compactness_contacts"],
            float(config.min_compactness_contacts),
        ),
        "loop": _penalty_max(metrics["max_loop_length"], float(config.max_loop_length)),
        "motif": (1.0 if config.require_motif_in_rec_helix and not bool(metrics["motif_in_rec_helix"]) else 0.0),
        "rifres": (1.0 if config.require_rifres_in_rec_helix and not bool(metrics["rifres_in_rec_helix"]) else 0.0),
        "buried_unsats": _penalty_max(
            metrics["buried_unsats"],
            float(config.max_buried_unsats),
        ),
        "shape_complementarity": _penalty_min(
            metrics["shape_complementarity"],
            float(config.min_shape_complementarity),
        ),
        "packstat": _penalty_min(
            metrics["packstat"],
            float(config.min_packstat),
        ),
        "rboltz": _penalty_min(
            metrics["max_rboltz_rkqe"],
            float(config.min_max_rboltz_rkqe),
        ),
        "avg_top_two_rboltz": _penalty_min(
            metrics["avg_top_two_rboltz"],
            float(config.min_avg_top_two_rboltz),
        ),
        "ddg": _heavy_penalty_max(
            metrics["ddg"],
            float(config.max_ddg),
            float(config.substantial_divergence_ratio),
        ),
        "contact_molecular_surface": _heavy_penalty_min(
            metrics["contact_molecular_surface"],
            float(config.min_contact_molecular_surface),
            float(config.substantial_divergence_ratio),
        ),
        "net_charge_over_sasa": max(
            _heavy_penalty_min(
                metrics["net_charge_over_sasa"],
                float(config.min_net_charge_over_sasa),
                float(config.substantial_divergence_ratio),
            ),
            _heavy_penalty_max(
                metrics["net_charge_over_sasa"],
                float(config.max_net_charge_over_sasa),
                float(config.substantial_divergence_ratio),
            ),
        ),
        "ddg_over_cms": _heavy_penalty_max(
            metrics["ddg_over_cms"],
            float(config.max_ddg_over_cms),
            float(config.substantial_divergence_ratio),
        ),
        "mpnn_score": (
            float(config.missing_mpnn_score_penalty)
            if np.isnan(metrics["mpnn_score"])
            else _heavy_penalty_max(
                metrics["mpnn_score"],
                float(config.max_mpnn_score),
                float(config.substantial_divergence_ratio),
            )
        ),
    }


def _component_weights(config: DBPDesignMetricsConfig) -> dict[str, float]:
    """Return the per-component penalty weights from config."""
    return {
        "base": config.base_weight,
        "phosphate": config.phosphate_weight,
        "bidentate": config.bidentate_weight,
        "backbone": config.backbone_weight,
        "compactness": config.compactness_weight,
        "loop": config.loop_weight,
        "motif": config.motif_weight,
        "rifres": config.rifres_weight,
        "buried_unsats": config.buried_unsat_weight,
        "shape_complementarity": config.shape_complementarity_weight,
        "packstat": config.packstat_weight,
        "rboltz": config.rboltz_weight,
        "avg_top_two_rboltz": config.avg_top_two_rboltz_weight,
        "ddg": config.ddg_weight,
        "contact_molecular_surface": config.contact_molecular_surface_weight,
        "net_charge_over_sasa": config.net_charge_over_sasa_weight,
        "ddg_over_cms": config.ddg_over_cms_weight,
        "mpnn_score": config.mpnn_score_weight,
    }


def _composite_score(penalties: dict[str, float], config: DBPDesignMetricsConfig) -> float:
    """Combine per-component penalties into a single weighted score in ``[0, 1]``."""
    weights = _component_weights(config)
    total_weight = float(sum(weights.values()))
    if total_weight <= 0.0:
        return 0.0
    weighted = sum(weights[key] * penalties[key] for key in penalties)
    return float(np.clip(weighted / total_weight, 0.0, 1.0))


def _extract_candidate_mpnn_score(candidate: tuple[Sequence, ...]) -> float | None:
    """Extract MPNN score from candidate metadata if present."""
    candidate_level_keys = (
        "mpnn_score",
        "protein_mpnn_score",
        "binder_mpnn_score",
    )
    per_sequence_keys = (
        "mpnn_score",
        "protein_mpnn_score",
        "binder_mpnn_score",
    )
    for seq in candidate:
        metadata = getattr(seq, "_metadata", {})
        if not isinstance(metadata, dict):
            continue
        for key in candidate_level_keys:
            value = metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        proteinmpnn_perplexity = metadata.get("proteinmpnn_perplexity")
        if proteinmpnn_perplexity is not None:
            try:
                return float(proteinmpnn_perplexity)
            except (TypeError, ValueError):
                pass
        ligandmpnn_metrics = metadata.get("ligandmpnn_metrics")
        if isinstance(ligandmpnn_metrics, dict):
            for key in (
                "mpnn_score",
                "score",
                "perplexity",
            ):
                value = ligandmpnn_metrics.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
        nested = metadata.get("dbp_metrics")
        if isinstance(nested, dict):
            for key in per_sequence_keys:
                value = nested.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
    return None


@constraint(
    key="dbp-design-metrics",
    label="DBP Design Metrics",
    config=DBPDesignMetricsConfig,
    description="Score protein-DNA designs with dbp_design metrics and configurable failure behavior",
    uses_gpu=True,
    tools_called=[
        "alphafold3-prediction",
        "boltz2-prediction",
        "protenix-prediction",
    ],
    category="protein_structure",
    supported_sequence_types=["protein", "dna"],
    input_labels=None,
)
def dbp_design_metrics_constraint(
    input_sequences: list[tuple[Sequence, ...]],
    config: DBPDesignMetricsConfig,
) -> list[ConstraintOutput]:
    """Evaluate dbp_design metrics and return normalized penalties in ``[0, 1]``.

    Resolves (reuses or predicts) one protein-operator complex PDB per candidate
    tuple, scores it with the heavy ``dbp_design`` PyRosetta metric suite, and
    converts the extracted metrics into a weighted-composite penalty where ``0``
    is best and ``1`` is worst. When evaluation fails and ``fail_hard`` is
    disabled, returns ``failure_score`` and records the error in metadata.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-candidate tuples of
            protein and DNA sequences forming the complex.
        config (DBPDesignMetricsConfig): Tool, threshold, and weight parameters.

    Returns:
        list[ConstraintOutput]: Per-candidate score in ``[0, 1]`` (lower is
            better) with ``dbp_metrics``, ``dbp_penalties``, ``pdb_path`` (and
            ``dbp_metrics_error`` on soft failure) metadata.

    Raises:
        RuntimeError: If dbp metric evaluation fails and ``fail_hard`` is enabled.
    """
    if not input_sequences:
        return []

    pdb_paths = resolve_structure_paths(
        input_sequences,
        structure_tool=config.structure_tool,
        tool_config=config.tool_config,
    )

    # Resolve and validate the shared dbp_design + PyRosetta setup once. These
    # raise on missing repo scripts / weights / pyrosetta, which are batch-level
    # failures that invalidate the whole call (never soft-failed per proposal).
    _load_dbp_modules(config)
    _load_interface_filters(config)

    results: list[ConstraintOutput] = []
    for candidate, pdb_path in zip(input_sequences, pdb_paths, strict=True):
        mpnn_score = _extract_candidate_mpnn_score(candidate)
        try:
            metrics = _compute_dbp_metrics_for_pdb(pdb_path, config)
        except Exception as exc:
            if config.fail_hard:
                raise RuntimeError("dbp_design metrics evaluation failed; this constraint is fail-hard.") from exc
            logger.warning("dbp-design-metrics: evaluation failed for %s: %s", pdb_path, exc)
            metrics = _failed_dbp_metrics()
            metrics["mpnn_score"] = float(mpnn_score) if mpnn_score is not None else float("nan")
            penalties = _failed_penalties()
            score = float(np.clip(float(config.failure_score), 0.0, 1.0))
            error_message = f"{type(exc).__name__}: {exc}"
            results.append(
                ConstraintOutput(
                    score=score,
                    metadata={
                        "pdb_path": pdb_path,
                        "dbp_metrics": metrics,
                        "dbp_penalties": penalties,
                        "dbp_metrics_error": error_message,
                    },
                )
            )
            continue

        metrics["mpnn_score"] = float(mpnn_score) if mpnn_score is not None else float("nan")

        # Calibrated ddG/CMS maximum-likelihood prefilter (dbp_design): short-circuit
        # designs that fail before paying for the full composite. Skipped entirely
        # when prefilter_eq_path is None (preserving the pre-prefilter behavior).
        if config.prefilter_eq_path is not None:
            try:
                prefilter_passed, prefilter_logp, prefilter_cut = _evaluate_prefilter(
                    config,
                    float(metrics["ddg_over_cms"]),
                    float(metrics["contact_molecular_surface"]),
                )
            except Exception as exc:
                if config.fail_hard:
                    raise RuntimeError(
                        "dbp_design ddG/CMS prefilter evaluation failed; this constraint is fail-hard."
                    ) from exc
                logger.warning("dbp-design-metrics: prefilter failed for %s: %s", pdb_path, exc)
                penalties = _failed_penalties()
                results.append(
                    ConstraintOutput(
                        score=float(np.clip(float(config.failure_score), 0.0, 1.0)),
                        metadata={
                            "pdb_path": pdb_path,
                            "dbp_metrics": metrics,
                            "dbp_penalties": penalties,
                            "dbp_metrics_error": f"prefilter: {type(exc).__name__}: {exc}",
                        },
                    )
                )
                continue

            if not prefilter_passed:
                penalties = _failed_penalties()
                results.append(
                    ConstraintOutput(
                        score=float(np.clip(float(config.failure_score), 0.0, 1.0)),
                        metadata={
                            "pdb_path": pdb_path,
                            "dbp_metrics": metrics,
                            "dbp_penalties": penalties,
                            "prefilter_passed": False,
                            "prefilter_logp": prefilter_logp,
                            "prefilter_cut": prefilter_cut,
                        },
                    )
                )
                continue

            prefilter_metadata: dict[str, Any] = {
                "prefilter_passed": True,
                "prefilter_logp": prefilter_logp,
                "prefilter_cut": prefilter_cut,
            }
        else:
            prefilter_metadata = {}

        penalties = _compute_penalties(metrics, config)
        score = _composite_score(penalties, config)
        results.append(
            ConstraintOutput(
                score=score,
                metadata={
                    "pdb_path": pdb_path,
                    "dbp_metrics": metrics,
                    "dbp_penalties": penalties,
                    **prefilter_metadata,
                },
            )
        )

    return results
