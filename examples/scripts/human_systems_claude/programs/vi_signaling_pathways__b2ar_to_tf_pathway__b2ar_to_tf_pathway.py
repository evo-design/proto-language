#!/usr/bin/env python3
"""B2AR-to-TF pathway design and client JSON export."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR))

from vi_signaling_pathways__b2ar_to_tf_pathway__creb_dna import (  # noqa: E402
    DESIGN_SEQ_LENGTH as CREB_DNA_LENGTH,
    LEFT_FLANK_FNAME,
    PROMPT_FNAME,
    RIGHT_FLANK_FNAME,
    clean_dna,
    creb_flank_lengths,
    creb_track_ids,
    generate_creb_dna_sequence,
)

from proto_language.constraint import (  # noqa: E402
    overall_protein_quality_constraint,
    structure_ensemble_rmsd_constraint,
    structure_iptm_constraint,
    structure_pae_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.core import Constraint, Construct, Program, Segment  # noqa: E402
from proto_language.generator import ESM3Generator, ESM3GeneratorConfig, MaskingStrategy  # noqa: E402
from proto_language.optimizer import (  # noqa: E402
    MCMCOptimizer,
    MCMCOptimizerConfig,
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)

PROJECT_SLUG = "vi_signaling_pathways__b2ar_to_tf_pathway__b2ar_to_tf_pathway"

DEFAULT_HUMAN_GENES_TSV = Path("examples/data/human_genes.tsv")
DEFAULT_HUMAN_GENES_FASTA = Path("examples/data/human_genes.fasta")
DEFAULT_PDB_CACHE_DIR = Path("examples/data/pdb_cache")

EPINEPHRINE_SMILES = "CNC[C@@H](c1ccc(c(c1)O)O)O"
ATP_SMILES = "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N"
CREBBP_KIX_SEQUENCE = "GVRKGWHEHVTQDLRSHLVHKLVQAIFPTPDPAALKDRRMENLVAYAKKVEGDMYESANSRDEYYHLLAEKIYKIQKELE"

CREB_MOTIF_START = (CREB_DNA_LENGTH // 2) - 25
CREB_MOTIF_END = (CREB_DNA_LENGTH // 2) + 25
CREB_MOTIF_LENGTH = CREB_MOTIF_END - CREB_MOTIF_START

BIOEMU_JOBS = (
    ("GNAS inactive ensemble RMSD", "GNAS", "6au6", "A", (85, 394), "gnas"),
    ("GNAS exchange ensemble RMSD", "GNAS", "3sn6", "A", (85, 394), "gnas"),
    ("PRKAR1A homodimer ensemble RMSD", "PRKAR1A", "1rl3", "A", (119, 379), "prkar1a"),
    ("PRKAR1A tetramer ensemble RMSD", "PRKAR1A", "2qcs", "B", (119, 379), "prkar1a"),
)
BIOEMU_TIMEOUT_SECONDS = 2 * 60 * 60
PROTEIN_QUALITY_THRESHOLD = 0.15


def _repo_path(path: Path) -> Path:
    return path if path.exists() else REPO_ROOT / path


@cache
def _cached_pdb_text(pdb_id: str) -> str:
    return _repo_path(DEFAULT_PDB_CACHE_DIR / f"{pdb_id}.pdb").read_text()


def _paths() -> tuple[Path, Path, Path]:
    config_path = PROJECT_DIR / "configs" / f"{PROJECT_SLUG}.json"
    output_dir = PROJECT_DIR / "outputs" / PROJECT_SLUG
    json_path = Path("examples/jsons") / f"{PROJECT_SLUG}.json"
    return config_path, output_dir, json_path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_wildtype_sequences(gene_ids: list[str]) -> dict[str, str]:
    gene_df = pd.read_csv(DEFAULT_HUMAN_GENES_TSV, sep="\t")
    gene_to_uniprot = {row["From"]: row["Entry"] for _, row in gene_df.iterrows()}

    missing_genes = [gene_id for gene_id in gene_ids if gene_id not in gene_to_uniprot]
    if missing_genes:
        raise ValueError(f"Gene IDs not found in mapping: {missing_genes}")

    uniprot_to_sequence: dict[str, str] = {}
    for record in SeqIO.parse(DEFAULT_HUMAN_GENES_FASTA, "fasta"):
        uniprot_to_sequence[record.id.split("|")[1]] = str(record.seq)

    sequences = {}
    for gene_id in gene_ids:
        uniprot_id = gene_to_uniprot[gene_id]
        if uniprot_id not in uniprot_to_sequence:
            raise ValueError(f"UniProt ID not found in FASTA for {gene_id}: {uniprot_id}")
        sequences[gene_id] = uniprot_to_sequence[uniprot_id]
    return sequences


def _profile_values(profile: str) -> dict[str, Any]:
    smoke = profile == "smoke"
    return {
        "creb_samples": 1 if smoke else 300,
        "protein_steps_per_generator": 1 if smoke else 5,
        "protein_num_steps": 10 if smoke else None,
        "esm2_model": "esm2_t6_8M_UR50D" if smoke else "esm2_t33_650M_UR50D",
        "esmfold_config": {"num_recycles": 1, "max_batch_residues": 1200, "verbose": 0} if smoke else None,
        "bioemu_samples": {"GNAS": 1 if smoke else 3000, "PRKAR1A": 1 if smoke else 1000},
        "bioemu_batch_size": 1 if smoke else 100,
        "protenix_config": _protenix_config(smoke=smoke),
    }


def _protenix_config(smoke: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_name": "protenix_base_default_v1.0.0",
        "use_msa": False,
        "seed": 0,
        "verbose": 1,
    }
    if smoke:
        config.update(
            {
                "model_name": "protenix_mini_default_v0.5.0",
                "num_diffusion_samples": 1,
                "num_diffusion_steps": 5,
                "num_pairformer_cycles": 1,
                "timeout": 600,
            }
        )
    return config


def _protein_quality_config() -> dict[str, Any]:
    return {
        "enable_length": False,
        "enable_complexity": True,
        "complexity_max_low_complexity": 0.12,
        "enable_repetitiveness": True,
        "repetitiveness_max_repetitiveness": 0.08,
        "repetitiveness_min_repeat_length": 2,
        "enable_diversity": False,
        "enable_balanced_aas": False,
    }


def _copy_label(component_id: str, index: int, count: int) -> str:
    return component_id if count == 1 else f"{component_id} copy {index + 1}"


def _segment_id(component_id: str, index: int, count: int) -> str:
    base = component_id.lower()
    return base if count == 1 else f"{base}_{index + 1}"


def _pathway_complexes(base_complexes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complexes = [
        {
            **complex_info,
            "gene_ids": list(complex_info["gene_ids"]),
            "stoichiometry": dict(complex_info["stoichiometry"]),
        }
        for complex_info in base_complexes
    ]
    for complex_info in complexes:
        if complex_info["complex_id"] == "MONOMER::GPCR":
            complex_info["gene_ids"].append("L_epinephrine")
            complex_info["stoichiometry"]["L_epinephrine"] = 1
        elif complex_info["complex_id"] == "MONOMER::Adenylyl_cyclase":
            complex_info["gene_ids"].append("ATP")
            complex_info["stoichiometry"]["ATP"] = 1
        elif complex_info["complex_id"] == "HOMOMER::CREB_dimer":
            complex_info["gene_ids"] += ["CREBBP_KIX", "CREB_TF_motif1", "CREB_TF_motif2"]
            complex_info["stoichiometry"]["CREBBP_KIX"] = 2
            complex_info["stoichiometry"]["CREB_TF_motif1"] = 1
            complex_info["stoichiometry"]["CREB_TF_motif2"] = 1
    return complexes


def _max_stoichiometry(complexes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for complex_info in complexes:
        for component_id in complex_info["gene_ids"]:
            counts[component_id] = max(counts.get(component_id, 1), int(complex_info["stoichiometry"][component_id]))
    return counts


def _complex_segments(complex_info: dict[str, Any], component_segments: dict[str, list[Segment]]) -> list[Segment]:
    inputs = []
    for component_id in complex_info["gene_ids"]:
        count = int(complex_info["stoichiometry"][component_id])
        inputs.extend(component_segments[component_id][:count])
    return inputs


def _complex_segment_ids(complex_info: dict[str, Any], segment_ids: dict[str, list[str]]) -> list[str]:
    targets = []
    for component_id in complex_info["gene_ids"]:
        count = int(complex_info["stoichiometry"][component_id])
        targets.extend(segment_ids[component_id][:count])
    return targets


def _add_construct(
    constructs: list[Construct],
    component_segments: dict[str, list[Segment]],
    component_id: str,
    segment: Segment,
    label: str,
) -> None:
    component_segments.setdefault(component_id, []).append(segment)
    constructs.append(Construct([segment], label=label))


def _add_static_constructs(
    constructs: list[Construct],
    component_segments: dict[str, list[Segment]],
    creb_dna: str,
) -> None:
    creb_motif = creb_dna[CREB_MOTIF_START:CREB_MOTIF_END]
    creb_motif_revcomp = str(Seq(creb_motif).reverse_complement())

    _add_construct(
        constructs,
        component_segments,
        "L_epinephrine",
        Segment(sequence=EPINEPHRINE_SMILES, sequence_type="ligand", label="L-epinephrine"),
        "L-epinephrine",
    )
    _add_construct(
        constructs,
        component_segments,
        "ATP",
        Segment(sequence=ATP_SMILES, sequence_type="ligand", label="ATP"),
        "ATP",
    )
    for idx in range(2):
        label = _copy_label("CREBBP_KIX", idx, 2)
        _add_construct(
            constructs,
            component_segments,
            "CREBBP_KIX",
            Segment(sequence=CREBBP_KIX_SEQUENCE, sequence_type="protein", label=label),
            label,
        )
    _add_construct(
        constructs,
        component_segments,
        "CREB_TF_motif1",
        Segment(sequence=creb_motif, sequence_type="dna", label="CREB TF motif 1"),
        "CREB TF motif 1",
    )
    _add_construct(
        constructs,
        component_segments,
        "CREB_TF_motif2",
        Segment(sequence=creb_motif_revcomp, sequence_type="dna", label="CREB TF motif 2"),
        "CREB TF motif 2",
    )


def _build_protein_design(
    gene_ids: list[str],
    sequences: dict[str, str],
    complexes: list[dict[str, Any]],
    profile: str,
) -> tuple[list[Construct], dict[str, list[Segment]], list[ESM3Generator], list[Constraint]]:
    profile_values = _profile_values(profile)
    max_stoich = _max_stoichiometry(complexes)
    structure_config: dict[str, Any] = {"structure_tool": "esmfold"}
    if profile_values["esmfold_config"] is not None:
        structure_config["esmfold_config"] = profile_values["esmfold_config"]

    constructs: list[Construct] = []
    component_segments: dict[str, list[Segment]] = {}
    generators: list[ESM3Generator] = []
    constraints: list[Constraint] = []

    for gene_id in gene_ids:
        sequence = sequences[gene_id]
        copy_count = max_stoich.get(gene_id, 1)
        tied_segments = [
            Segment(
                sequence=sequence,
                sequence_type="protein",
                label=_copy_label(gene_id, idx, copy_count),
            )
            for idx in range(copy_count)
        ]
        for idx, segment in enumerate(tied_segments):
            _add_construct(constructs, component_segments, gene_id, segment, _copy_label(gene_id, idx, copy_count))

        generator = ESM3Generator(
            ESM3GeneratorConfig(
                model_checkpoint="esm3_sm_open_v1",
                temperature=0.3,
                masking_strategy=MaskingStrategy(num_mutations=max(1, int(0.25 * len(sequence)))),
                batch_size=1,
            )
        )
        generator.assign(tied_segments)
        generators.append(generator)

        primary_segment = tied_segments[0]
        constraints.extend(
            [
                Constraint(
                    inputs=[primary_segment],
                    function=structure_plddt_constraint,
                    function_config=structure_config,
                    label=f"{gene_id}_esmfold_plddt",
                ),
                Constraint(
                    inputs=[primary_segment],
                    function=structure_ptm_constraint,
                    function_config=structure_config,
                    label=f"{gene_id}_esmfold_ptm",
                ),
                Constraint(
                    inputs=[primary_segment],
                    function=overall_protein_quality_constraint,
                    function_config={"protein_quality_config": _protein_quality_config()},
                    threshold=PROTEIN_QUALITY_THRESHOLD,
                    label=f"{gene_id}_protein_quality",
                ),
            ]
        )

    return constructs, component_segments, generators, constraints


def _bioemu_constraints(
    component_segments: dict[str, list[Segment]],
    output_dir: Path,
    profile: str,
) -> list[Constraint]:
    profile_values = _profile_values(profile)
    run_dir = os.environ.get("RUN_OUTPUT_DIR")
    if run_dir is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = str(output_dir / f"run_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
    bioemu_dir = Path(run_dir) / "bioemu_outputs"

    jobs = BIOEMU_JOBS[:1] if profile == "smoke" else BIOEMU_JOBS
    constraints = []
    for label, gene_id, pdb_id, chain_id, residue_range, output_suffix in jobs:
        constraints.append(
            Constraint(
                inputs=[component_segments[gene_id][0]],
                function=structure_ensemble_rmsd_constraint,
                function_config={
                    "target_structure": _cached_pdb_text(pdb_id),
                    "target_chain_id": chain_id,
                    "target_residue_range": residue_range,
                    "proposal_residue_range": residue_range,
                    "bioemu_config": {
                        "num_samples": profile_values["bioemu_samples"][gene_id],
                        "output_dir": str(bioemu_dir) + f"_{output_suffix}",
                        "batch_size": profile_values["bioemu_batch_size"],
                        "filter_samples": False,
                        "seed": 0,
                        "timeout": BIOEMU_TIMEOUT_SECONDS,
                    },
                    "rmsd_aggregation": "min",
                    "inflection_point_angstroms": 3.0,
                    "sigmoid_slope": 3.0,
                    "verbose": True,
                },
                label=label.lower().replace(" ", "_"),
            )
        )
    return constraints


def _protenix_constraints(
    component_segments: dict[str, list[Segment]],
    complexes: list[dict[str, Any]],
    profile: str,
) -> list[Constraint]:
    profile_values = _profile_values(profile)
    structure_config = {
        "structure_tool": "protenix",
        "protenix_config": profile_values["protenix_config"],
    }
    metric_constraints = [
        ("structure_plddt", structure_plddt_constraint),
        ("structure_ptm", structure_ptm_constraint),
        ("structure_iptm", structure_iptm_constraint),
        ("structure_pae", structure_pae_constraint),
    ]
    selected_complexes = complexes[:1] if profile == "smoke" else complexes

    constraints = []
    for complex_info in selected_complexes:
        inputs = _complex_segments(complex_info, component_segments)
        for metric_name, metric_fn in metric_constraints:
            constraints.append(
                Constraint(
                    inputs=inputs,
                    function=metric_fn,
                    function_config=structure_config,
                    label=f"{complex_info['complex_id']}_{metric_name}",
                )
            )
    return constraints


def create_b2ar_to_tf_pathway_program(creb_dna: str, profile: str = "full") -> tuple[Program, dict[str, list[Segment]]]:
    """Build the local executable Proto program."""
    config_path, output_dir, _json_path = _paths()
    config = _load_config(config_path)
    gene_ids = list(config["all_gene_ids"])
    sequences = _load_wildtype_sequences(gene_ids)
    complexes = _pathway_complexes(config["complexes"])
    profile_values = _profile_values(profile)

    constructs, component_segments, protein_generators, protein_constraints = _build_protein_design(
        gene_ids,
        sequences,
        complexes,
        profile,
    )
    _add_static_constructs(constructs, component_segments, creb_dna)

    protein_optimizer = MCMCOptimizer(
        constructs=constructs,
        generators=protein_generators,
        constraints=protein_constraints,
        config=MCMCOptimizerConfig(
            num_results=1,
            proposals_per_result=1,
            num_steps=profile_values["protein_num_steps"]
            or len(protein_generators) * profile_values["protein_steps_per_generator"],
            max_temperature=0.1,
            min_temperature=0.01,
            tracking_interval=1,
            verbose=True,
        ),
        custom_logging=_protein_logger(gene_ids, component_segments),
        clear_tool_cache=4 * 1024 * 1024 * 1024,
    )

    scoring_optimizer = RejectionSamplingOptimizer(
        constructs=constructs,
        generators=[],
        constraints=[
            *_bioemu_constraints(component_segments, output_dir, profile),
            *_protenix_constraints(component_segments, complexes, profile),
        ],
        config=RejectionSamplingOptimizerConfig(
            num_samples=1,
            num_results=1,
            proposal_source="existing_results",
            verbose=True,
        ),
    )

    program = Program(
        optimizers=[protein_optimizer, scoring_optimizer],
        num_results=1,
        verbose=True,
    )
    return program, component_segments


def _protein_logger(gene_ids: list[str], component_segments: dict[str, list[Segment]]):
    design_segments = [(gene_id, component_segments[gene_id][0]) for gene_id in gene_ids]

    def custom_logging(step: int, outputs: tuple[Segment, ...]) -> None:
        del outputs
        print(f"Protein design step {step}:")
        for gene_id, segment in design_segments:
            print(f"\t{gene_id}: {segment.result_sequences[0].sequence}")

    return custom_logging


def _json_construct(
    construct_id: str,
    label: str,
    sequence_type: str,
    sequence: str | None = None,
    length: int | None = None,
) -> dict[str, Any]:
    segment: dict[str, Any] = {"id": construct_id, "label": label}
    if sequence is None:
        segment["length"] = length
    else:
        segment["sequence"] = sequence
    return {
        "id": f"{construct_id}_construct",
        "type": sequence_type,
        "label": label,
        "segments": [segment],
    }


def _json_structure_constraint(
    label: str,
    key: str,
    targets: list[str],
    config: dict[str, Any],
    weight: float = 1.0,
) -> dict[str, Any]:
    item: dict[str, Any] = {"key": key, "label": label, "targets": targets, "config": config}
    if weight != 1.0:
        item["weight"] = weight
    return item


def _json_bioemu_constraint(
    label: str,
    target: str,
    pdb_id: str,
    chain_id: str,
    residue_range: tuple[int, int],
    samples: int,
) -> dict[str, Any]:
    return {
        "key": "structure-ensemble-rmsd",
        "label": label,
        "targets": [target],
        "config": {
            "target_structure": str(DEFAULT_PDB_CACHE_DIR / f"{pdb_id}.pdb"),
            "target_chain_id": chain_id,
            "target_residue_range": list(residue_range),
            "proposal_residue_range": list(residue_range),
            "bioemu_config": {
                "num_samples": samples,
                "batch_size": 1,
                "filter_samples": False,
                "seed": 0,
                "timeout": BIOEMU_TIMEOUT_SECONDS,
            },
            "rmsd_aggregation": "min",
            "inflection_point_angstroms": 3.0,
            "sigmoid_slope": 3.0,
            "verbose": True,
        },
    }


def build_frontend_program_json(profile: str = "full") -> dict[str, Any]:
    """Build client/API-compatible JSON for the complete pathway."""
    config_path, _output_dir, _json_path = _paths()
    config = _load_config(config_path)
    gene_ids = list(config["all_gene_ids"])
    sequences = _load_wildtype_sequences(gene_ids)
    complexes = _pathway_complexes(config["complexes"])
    for complex_info in complexes:
        if complex_info["complex_id"] == "HOMOMER::CREB_dimer":
            complex_info["gene_ids"].remove("CREB_TF_motif2")
            del complex_info["stoichiometry"]["CREB_TF_motif2"]
    max_stoich = _max_stoichiometry(complexes)
    profile_values = _profile_values(profile)

    segment_ids: dict[str, list[str]] = {}
    constructs: list[dict[str, Any]] = []
    for gene_id in gene_ids:
        copy_count = max_stoich.get(gene_id, 1)
        segment_ids[gene_id] = []
        for idx in range(copy_count):
            sid = _segment_id(gene_id, idx, copy_count)
            segment_ids[gene_id].append(sid)
            constructs.append(
                _json_construct(
                    sid,
                    _copy_label(gene_id, idx, copy_count),
                    "protein",
                    sequence=sequences[gene_id] if idx == 0 else None,
                    length=None if idx == 0 else len(sequences[gene_id]),
                )
            )
        if gene_id == "ADRB2":
            constructs.append(_json_construct("l_epinephrine", "L-epinephrine", "ligand", sequence=EPINEPHRINE_SMILES))
        elif gene_id == "ADCY9":
            constructs.append(_json_construct("atp", "ATP", "ligand", sequence=ATP_SMILES))

    left_flank_len, right_flank_len = creb_flank_lengths()
    creb_prompt = clean_dna(str(SeqIO.read(PROMPT_FNAME, "fasta").seq))
    left_flank = clean_dna(str(SeqIO.read(LEFT_FLANK_FNAME, "fasta").seq))[-left_flank_len:]
    right_flank = clean_dna(str(SeqIO.read(RIGHT_FLANK_FNAME, "fasta").seq))[:right_flank_len]

    constructs.extend(
        [
            _json_construct("creb_left_flank", "CREB left Borzoi flank", "dna", sequence=left_flank),
            _json_construct("creb_dna", "CREB TF motif", "dna", length=CREB_MOTIF_LENGTH),
            _json_construct("creb_right_flank", "CREB right Borzoi flank", "dna", sequence=right_flank),
            _json_construct("crebbp_kix_1", "CREBBP_KIX copy 1", "protein", sequence=CREBBP_KIX_SEQUENCE),
            _json_construct("crebbp_kix_2", "CREBBP_KIX copy 2", "protein", sequence=CREBBP_KIX_SEQUENCE),
        ]
    )
    segment_ids.update(
        {
            "L_epinephrine": ["l_epinephrine"],
            "ATP": ["atp"],
            "CREBBP_KIX": ["crebbp_kix_1", "crebbp_kix_2"],
            "CREB_TF_motif1": ["creb_dna"],
        }
    )

    creb_stage = {
        "generators": [
            {
                "key": "evo2",
                "label": "CREB motif generator",
                "targets": ["creb_dna"],
                "config": {
                    "prompts": [creb_prompt],
                    "model_checkpoint": "evo2_7b",
                    "top_k": 4,
                    "top_p": 1.0,
                    "temperature": 0.5,
                    "force_prompt_threshold": 1,
                    "stop_at_eos": False,
                    "batched": True,
                    "batch_size": 1 if profile == "smoke" else 10,
                    "cached_generation": True,
                    "prepend_prompt": False,
                },
            }
        ],
        "constraints": [
            {
                "key": "borzoi-track-activity",
                "label": "Borzoi CREB track activity",
                "targets": ["creb_left_flank", "creb_dna", "creb_right_flank"],
                "config": {
                    "organism": "human",
                    "borzoi_output_tracks": creb_track_ids(),
                    "direction": "maximize",
                    "activity_threshold": 200.0,
                    "batch_size": 1,
                },
            }
        ],
        "optimizer": {
            "method": "rejection-sampling",
            "config": {"num_samples": profile_values["creb_samples"], "num_results": 1, "verbose": True},
        },
    }

    protein_generators = []
    protein_constraints = []
    esmfold_config: dict[str, Any] = {"structure_tool": "esmfold"}
    if profile_values["esmfold_config"] is not None:
        esmfold_config["esmfold_config"] = profile_values["esmfold_config"]

    for gene_id in gene_ids:
        primary_sid = segment_ids[gene_id][0]
        if gene_id != "ADCY9":
            protein_generators.append(
                {
                    "key": "esm2",
                    "label": f"{gene_id} tied-copy ESM2 generator",
                    "targets": segment_ids[gene_id],
                    "config": {
                        "model_checkpoint": profile_values["esm2_model"],
                        "temperature": 0.3,
                        "masking_strategy": MaskingStrategy(
                            num_mutations=max(1, int(0.25 * len(sequences[gene_id])))
                        ).model_dump(),
                        "batch_size": 1,
                    },
                }
            )
            protein_constraints.extend(
                [
                    _json_structure_constraint(
                        f"{gene_id} ESMFold pLDDT", "structure-plddt", [primary_sid], esmfold_config
                    ),
                    _json_structure_constraint(
                        f"{gene_id} ESMFold pTM", "structure-ptm", [primary_sid], esmfold_config
                    ),
                    {
                        "key": "overall-protein-quality",
                        "label": f"{gene_id} protein quality",
                        "targets": [primary_sid],
                        "threshold": PROTEIN_QUALITY_THRESHOLD,
                        "config": {"protein_quality_config": _protein_quality_config()},
                    },
                ]
            )

    protein_stage = {
        "generators": protein_generators,
        "constraints": protein_constraints,
        "optimizer": {
            "method": "mcmc",
            "config": {
                "num_results": 1,
                "proposals_per_result": 1,
                "num_steps": profile_values["protein_num_steps"]
                or len(gene_ids) * profile_values["protein_steps_per_generator"],
                "max_temperature": 0.1,
                "min_temperature": 0.01,
                "tracking_interval": 1,
                "verbose": True,
            },
        },
    }

    scoring_constraints = [
        _json_bioemu_constraint(
            label,
            segment_ids[gene_id][0],
            pdb_id,
            chain_id,
            residue_range,
            profile_values["bioemu_samples"][gene_id],
        )
        for label, gene_id, pdb_id, chain_id, residue_range, _ in BIOEMU_JOBS
    ]
    structure_config = {
        "structure_tool": "protenix",
        "protenix_config": profile_values["protenix_config"],
    }
    for complex_info in complexes:
        targets = _complex_segment_ids(complex_info, segment_ids)
        for metric_key in ("structure-plddt", "structure-ptm", "structure-iptm", "structure-pae"):
            scoring_constraints.append(
                _json_structure_constraint(
                    f"{complex_info['complex_name']} {metric_key}",
                    metric_key,
                    targets,
                    structure_config,
                )
            )

    scoring_stage = {
        "generators": [],
        "constraints": scoring_constraints,
        "optimizer": {
            "method": "rejection-sampling",
            "config": {"num_samples": 1, "num_results": 1, "proposal_source": "existing_results", "verbose": True},
        },
    }

    return {
        "name": "B2AR-to-TF pathway",
        "description": "End-to-end human beta-2 adrenergic receptor signaling-pathway design, from ligand sensing at ADRB2 through Gs activation, adenylyl cyclase engagement, PKA holoenzyme assembly, and CREB-mediated transcriptional readout. The program includes epinephrine-bound ADRB2, the GNAS/GNB1/GNG2 heterotrimer, ADCY9 with ATP, a 2:2 PRKACA:PRKAR1A PKA holoenzyme, and a CREB1 dimer bound to a CRE-family regulatory DNA motif with two CREBBP KIX domains. Stage 1 designs the CREB target motif inside fixed Borzoi flanks with Evo2 and a Borzoi CREB-track activity objective. Stage 2 diversifies non-ADCY9 pathway proteins with tied-copy protein generators for stoichiometric assemblies while preserving ESMFold pLDDT/pTM and protein-quality constraints; the client JSON holds ADCY9 fixed to avoid the very large ESM2 diversification step. Stage 3 rescoring uses existing-results rejection sampling to rank the complete pathway candidate with BioEmu ensemble RMSD checks for GNAS and PRKAR1A conformational states plus Protenix v1 confidence metrics for the receptor, G-protein, cyclase, PKA, and CREB transcription-factor complexes. This checked-in JSON uses the smoke profile so it can render and compile as a system-design example; increase sample counts, BioEmu samples, and Protenix settings for a full design campaign.",
        "version": "1.0",
        "num_results": 1,
        "verbose": True,
        "constructs": constructs,
        "optimization_stages": [creb_stage, protein_stage, scoring_stage],
    }


def write_frontend_program_json(path: Path, profile: str = "full") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_frontend_program_json(profile=profile), indent=2) + "\n")
    print(f"Wrote client program JSON to {path}")


def _run_output_dir(output_dir: Path, profile: str) -> Path:
    env_run_dir = os.environ.get("RUN_OUTPUT_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"e2e_{run_timestamp}_{profile}"
        os.environ["RUN_OUTPUT_DIR"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _compact_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "stage",
        "stage_name",
        "result_idx",
        "energy_score",
        "constraint",
        "score",
        "weighted_score",
        "avg_constraint_score",
        "protein_quality_scores",
        "low_complexity_fraction",
        "repetitiveness_score",
        "avg_plddt",
        "ptm",
        "iptm",
        "avg_pae",
        "ensemble_rmsd_summary",
        "ensemble_rmsd_aggregation",
        "ensemble_rmsd_min",
        "ensemble_rmsd_mean",
        "ensemble_rmsd_median",
        "ensemble_rmsd_p10",
        "ensemble_rmsd_std",
        "ensemble_size",
        "ensemble_score",
        "pct_within_2A",
        "pct_within_3A",
        "structure_tool",
    ]
    columns = [column for column in preferred_columns if column in df.columns]
    compact = df[columns].copy()
    return compact.drop_duplicates()


def _format_score_value(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _print_score_summary(stage_name: str, score_df: pd.DataFrame) -> None:
    if score_df.empty:
        print(f"\nScore summary ({stage_name}): no constraint scores exported")
        return

    display_columns = [
        "score",
        "weighted_score",
        "avg_constraint_score",
        "low_complexity_fraction",
        "repetitiveness_score",
        "avg_plddt",
        "ptm",
        "iptm",
        "avg_pae",
        "ensemble_rmsd_summary",
        "ensemble_rmsd_min",
        "ensemble_rmsd_mean",
        "ensemble_size",
        "pct_within_2A",
        "pct_within_3A",
    ]
    print(f"\nScore summary ({stage_name}):")
    for _, row in score_df.iterrows():
        fields = []
        for column in display_columns:
            if column in row and _format_score_value(row[column]) != "n/a":
                fields.append(f"{column}={_format_score_value(row[column])}")
        print(f"\t{row['constraint']}: {', '.join(fields)}")


def _write_run_artifacts(program: Program, component_segments: dict[str, list[Segment]], run_dir: Path) -> None:
    results_dir = run_dir / "results"
    stage_names = ["protein_design", "pathway_rescore"]
    score_frames = []

    for stage_index, optimizer in enumerate(program.optimizers):
        stage_name = stage_names[stage_index] if stage_index < len(stage_names) else optimizer.__class__.__name__
        stage_dir = results_dir / f"stage_{stage_index + 1}_{stage_name}"
        program.export(stage_dir, format="csv", stage=stage_index, include_proposals=True)

        constraints_df = program.to_dataframe(table="constraints", stage=stage_index)
        if constraints_df.empty:
            continue
        constraints_df.insert(0, "stage", stage_index + 1)
        constraints_df.insert(1, "stage_name", stage_name)
        score_df = _compact_score_frame(constraints_df)
        score_df.to_csv(run_dir / f"score_summary_stage_{stage_index + 1}_{stage_name}.tsv", sep="\t", index=False)
        _print_score_summary(stage_name, score_df)
        score_frames.append(score_df)

    if score_frames:
        pd.concat(score_frames, ignore_index=True).to_csv(run_dir / "score_summary.tsv", sep="\t", index=False)

    program.export(results_dir / "final", format="csv", include_proposals=True)
    program.to_fasta(run_dir / "final_sequences.fasta")

    final_sequences = {
        segment.label or component_id: segment.result_sequences[0].sequence
        for component_id, segments in component_segments.items()
        for segment in segments
        if segment.sequence_type != "ligand"
    }
    (run_dir / "final_sequences.json").write_text(json.dumps(final_sequences, indent=2) + "\n")
    print(f"\nWrote score and sequence artifacts to {run_dir}")


def run_local(profile: str = "full") -> int:
    _config_path, output_dir, _json_path = _paths()
    run_dir = _run_output_dir(output_dir, profile)
    print(f"RUN_OUTPUT_DIR={run_dir}")

    creb_dna = generate_creb_dna_sequence(profile=profile)
    gc.collect()
    torch.cuda.empty_cache()

    program, component_segments = create_b2ar_to_tf_pathway_program(creb_dna, profile=profile)
    program.run()
    _write_run_artifacts(program, component_segments, run_dir)

    print("\nFinal design sequences:")
    for component_id, segments in component_segments.items():
        for segment in segments:
            if segment.sequence_type != "ligand":
                print(f"\t{segment.label or component_id}: {segment.result_sequences[0].sequence}")
    return 0


def main() -> int:
    _config_path, _output_dir, json_path = _paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--emit-json", type=Path, nargs="?", const=json_path)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    if args.emit_json is not None:
        write_frontend_program_json(args.emit_json, profile=args.profile)
    if args.skip_run:
        return 0
    return run_local(profile=args.profile)


if __name__ == "__main__":
    sys.exit(main())
