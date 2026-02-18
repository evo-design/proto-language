"""
Cas9 generation pipeline using a single TopK optimizer with filter constraints.

Expresses the multi-stage Cas9 generation pipeline as a proto-language Program
with one TopKOptimizer. All filtering steps are expressed as constraints ordered
cheap -> expensive. The optimizer's built-in filter short-circuiting (score_energy
mask propagation) ensures expensive filters (AF3) only run on candidates that pass
all cheaper ones.

Architecture:
    1 TopK optimizer with 1 Evo1Generator + 8 filter constraints:
        1. orf_filter          - ORFipy, ORF >= 3000 bp
        2. cas9_phmm_filter    - PyHmmer vs cas9.hmm
        3. crispr_array_filter - MinCED, >= 3 repeats
        4. identity_filter     - MMseqs2, identity < 90%
        5. gap_gini_filter     - MAFFT + gap Gini < 0.1
        6. domain_filter       - PyHmmer vs cas9_domains.hmm
        7. tracr_filter        - CRISPRtracrRNA + IntaRNA
        8. structure_filter    - AF3 + structure metrics

Usage:
    python evocas9_topk.py --n-samples 10
    python evocas9_topk.py --n-samples 150 --batch-size 150
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Generation
PROMPT = "`A"  # Backtick = Cas9 class token + one nucleotide
NUM_TOKENS = 8000
MODEL_NAME = "evo-1-8k-crispr"
# Temperatures [0.1, 0.3, 0.5] and top_k [2, 4] were all tested in the paper;
# subsequent analysis showed temp=0.5 dominates the top structural hits.
TEMPERATURES = [0.5]
TOP_KS = [2, 4]

# Data paths (all relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _REPO_ROOT / "examples" / "data"
CAS9_HMM_PATH = str(_DATA_DIR / "cas_hmms" / "cas9.hmm")
DOMAIN_HMM_PATH = str(_DATA_DIR / "cas_hmms" / "cas9_domains.hmm")
TRAINING_FASTA_DIR = _DATA_DIR / "cas_training_proteins"
TRAINING_FASTA_CACHE = _REPO_ROOT / "data" / "cas_training_combined.fasta"

# Stage 1 thresholds
ORF_MIN_LEN = 3000  # Nucleotides
CAS9_PHMM_EVALUE = 1e-3

# Stage 2 thresholds
IDENTITY_THRESHOLD = 0.90
GAP_GINI_THRESHOLD = 0.1

# Stage 3 thresholds
DOMAIN_EVALUE_THRESHOLD = 1e-10
REQUIRED_DOMAINS = {"RuvC_1", "RuvC_2", "RuvC_3", "HNH"}

# Stage 4 thresholds
PLDDT_THRESHOLD = 75.0
GYRATION_RADIUS_THRESHOLD = 45.0
LONGEST_ALPHA_THRESHOLD = 50

# ============================================================================
# Module-level cache
# ============================================================================

# Key: DNA sequence string
# Values populated incrementally by successive filter constraints
CACHE: Dict[str, Dict[str, Any]] = {}

# Training sequences loaded once and reused across combos
_TRAINING_SEQS: Optional[dict] = None

# ============================================================================
# Helpers
# ============================================================================


def _get_training_fasta() -> Path:
    """Build combined training FASTA from individual .gz files (cached)."""
    if TRAINING_FASTA_CACHE.exists():
        logger.info(
            f"Using cached combined training FASTA: {TRAINING_FASTA_CACHE}"
        )
        return TRAINING_FASTA_CACHE

    if not TRAINING_FASTA_DIR.exists():
        raise FileNotFoundError(
            f"Training FASTA directory not found: {TRAINING_FASTA_DIR}"
        )

    fasta_files = sorted(TRAINING_FASTA_DIR.glob("*.fasta.gz"))
    if not fasta_files:
        raise FileNotFoundError(
            f"No .fasta.gz files found in {TRAINING_FASTA_DIR}"
        )

    TRAINING_FASTA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    total_seqs = 0
    with open(TRAINING_FASTA_CACHE, "w") as out:
        for fasta in fasta_files:
            with gzip.open(fasta, "rt") as f:
                content = f.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                total_seqs += content.count(">")

    logger.info(
        f"Built combined training FASTA ({total_seqs} sequences "
        f"from {len(fasta_files)} files): {TRAINING_FASTA_CACHE}"
    )
    return TRAINING_FASTA_CACHE


def _load_training_sequences(fasta_path: Path) -> dict:
    """Load FASTA into {id: sequence} dict."""
    sequences = {}
    current_id = None
    current_seq: List[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)

    logger.info(f"Loaded {len(sequences)} training sequences from {fasta_path}")
    return sequences


def _get_training_seqs() -> dict:
    """Get training sequences, loading them once on first call."""
    global _TRAINING_SEQS
    if _TRAINING_SEQS is None:
        training_fasta = _get_training_fasta()
        _TRAINING_SEQS = _load_training_sequences(training_fasta)
    return _TRAINING_SEQS


def _get_protein(dna: str) -> Optional[str]:
    """Look up cached protein for a DNA sequence."""
    entry = CACHE.get(dna)
    if entry:
        return entry.get("protein")
    return None


def _parse_seq_index(target_name: str) -> Optional[int]:
    """Parse integer index from HMM/mmseqs target names like 'seq_42' or '42'."""
    try:
        if target_name.startswith("seq_"):
            return int(target_name[4:])
        return int(target_name)
    except (ValueError, IndexError):
        return None


# ============================================================================
# Constraint functions
# ============================================================================


def orf_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by ORF length. Caches protein translation.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import OrfipyConfig, OrfipyInput, run_orfipy_prediction

    min_len = config.get("min_len", ORF_MIN_LEN)
    sequences = [seq_tuple[0].sequence for seq_tuple in input_sequences]

    orf_result = run_orfipy_prediction(
        OrfipyInput(sequences=sequences),
        OrfipyConfig(min_len=min_len, strand="b"),
    )

    scores = []
    for i, seq_orfs in enumerate(orf_result.predicted_orfs):
        dna = sequences[i]
        if dna not in CACHE:
            CACHE[dna] = {}

        if seq_orfs:
            best = max(seq_orfs, key=lambda o: o.amino_acid_length)
            CACHE[dna]["protein"] = best.amino_acid_sequence
            scores.append(0.0)
        else:
            scores.append(1.0)

    logger.info(
        f"orf_filter: {sum(1 for s in scores if s == 0.0)}/{len(scores)} "
        f"have ORFs >= {min_len} nt"
    )
    return scores


def cas9_phmm_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by Cas9 profile HMM hit.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import (
        PyHmmsearchConfig,
        PyHmmsearchInput,
        run_pyhmmer_hmmsearch,
    )

    evalue = config.get("evalue", CAS9_PHMM_EVALUE)
    hmm_path = config.get("hmm_path", CAS9_HMM_PATH)

    if not Path(hmm_path).exists():
        logger.warning(f"Cas9 HMM not found: {hmm_path} — passing all")
        return [0.0] * len(input_sequences)

    # Collect proteins from cache
    dna_seqs = [seq_tuple[0].sequence for seq_tuple in input_sequences]
    proteins = []
    valid_indices = []
    for i, dna in enumerate(dna_seqs):
        protein = _get_protein(dna)
        if protein:
            proteins.append(protein)
            valid_indices.append(i)

    if not proteins:
        return [1.0] * len(input_sequences)

    hmm_result = run_pyhmmer_hmmsearch(
        PyHmmsearchInput(sequences=proteins, hmm=hmm_path),
        PyHmmsearchConfig(evalue_threshold=evalue),
    )

    hmm_hits = set()
    if (
        hmm_result.sequence_hits_df is not None
        and not hmm_result.sequence_hits_df.empty
    ):
        for _, row in hmm_result.sequence_hits_df.iterrows():
            j = _parse_seq_index(row.get("target_name", ""))
            if j is not None and 0 <= j < len(proteins):
                hmm_hits.add(j)

    scores = [1.0] * len(input_sequences)
    for protein_idx in hmm_hits:
        original_idx = valid_indices[protein_idx]
        scores[original_idx] = 0.0

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info(
        f"cas9_phmm_filter: {n_pass}/{len(scores)} have Cas9 pHMM hit "
        f"(E < {evalue})"
    )
    return scores


def crispr_array_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by CRISPR array detection. Caches repeat sequence.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import MincedConfig, MincedInput, run_minced

    sequences = [seq_tuple[0].sequence for seq_tuple in input_sequences]

    minced_result = run_minced(
        MincedInput(sequences=sequences),
        MincedConfig(min_num_repeats=3, min_repeat_length=23),
    )

    scores = []
    for i, dna in enumerate(sequences):
        if dna not in CACHE:
            CACHE[dna] = {}

        if i < len(minced_result.results):
            res = minced_result.results[i]
            if res.has_crispr:
                if res.crispr_arrays:
                    CACHE[dna]["crispr_repeat"] = (
                        res.crispr_arrays[0].repeats_and_spacers[0].repeat
                    )
                scores.append(0.0)
            else:
                scores.append(1.0)
        else:
            scores.append(1.0)

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info(
        f"crispr_array_filter: {n_pass}/{len(scores)} have CRISPR arrays"
    )
    return scores


def identity_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by sequence identity to training set. Caches identity and nearest hit.

    Returns 0.0 for PASS (identity < threshold or no hit), 1.0 for FAIL.
    """
    from proto_tools import (
        MmseqsSearchProteinsConfig,
        MmseqsSearchProteinsInput,
        run_mmseqs_search_proteins,
    )

    threshold = config.get("threshold", IDENTITY_THRESHOLD)
    training_fasta = _get_training_fasta()
    training_seqs = _get_training_seqs()

    dna_seqs = [seq_tuple[0].sequence for seq_tuple in input_sequences]
    proteins = []
    for dna in dna_seqs:
        proteins.append(_get_protein(dna) or "")

    # Batch MMseqs2 search
    mmseqs_result = run_mmseqs_search_proteins(
        MmseqsSearchProteinsInput(
            query_sequences=proteins,
            mmseqs_db=str(training_fasta),
        ),
        MmseqsSearchProteinsConfig(only_top_hits=True),
    )

    scores = []
    for i, dna in enumerate(dna_seqs):
        if dna not in CACHE:
            CACHE[dna] = {}

        result = mmseqs_result.results[i]

        if not result.has_hits:
            CACHE[dna]["identity"] = 0.0
            CACHE[dna]["nearest_hit_seq"] = None
            scores.append(0.0)
            continue

        top_hit = result.top_hit
        identity = top_hit.pident / 100.0
        CACHE[dna]["identity"] = identity

        target_seq = training_seqs.get(top_hit.target_id)
        CACHE[dna]["nearest_hit_seq"] = target_seq

        if identity >= threshold:
            scores.append(1.0)
        else:
            scores.append(0.0)

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info(
        f"identity_filter: {n_pass}/{len(scores)} have identity < "
        f"{threshold:.0%}"
    )
    return scores


def gap_gini_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by gap Gini on MAFFT alignment vs nearest training hit.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import MafftConfig, MafftInput, run_mafft_align

    from proto_language.language.constraint.sequence_alignment.gap_gini_constraint import (
        _gap_gini_single,
        _trim_alignment,
    )

    threshold = config.get("threshold", GAP_GINI_THRESHOLD)

    scores = []
    for seq_tuple in input_sequences:
        dna = seq_tuple[0].sequence
        if dna not in CACHE:
            CACHE[dna] = {}

        protein = _get_protein(dna)
        nearest_hit = CACHE.get(dna, {}).get("nearest_hit_seq")

        # No nearest hit or no protein -> passes (novel sequence)
        if not protein or nearest_hit is None:
            CACHE[dna]["gap_gini"] = 0.0
            scores.append(0.0)
            continue

        align_result = run_mafft_align(
            MafftInput(sequences=[protein, nearest_hit]),
            MafftConfig(),
        )

        if align_result.msa and len(align_result.msa) >= 2:
            al1, al2 = _trim_alignment(
                align_result.msa[0], align_result.msa[1]
            )
            if al1 is not None:
                gini = _gap_gini_single(al1, al2)
            else:
                gini = 0.0
        else:
            gini = 0.0

        CACHE[dna]["gap_gini"] = gini

        if gini < threshold:
            scores.append(0.0)
        else:
            scores.append(1.0)

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info(
        f"gap_gini_filter: {n_pass}/{len(scores)} have gap Gini < {threshold}"
    )
    return scores


def domain_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by presence of all required Cas9 domains.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import (
        PyHmmsearchConfig,
        PyHmmsearchInput,
        run_pyhmmer_hmmsearch,
    )

    hmm_path = config.get("hmm_path", DOMAIN_HMM_PATH)
    required = config.get("required_domains", REQUIRED_DOMAINS)
    evalue = config.get("evalue", DOMAIN_EVALUE_THRESHOLD)

    if not Path(hmm_path).exists():
        logger.warning(f"Domain HMM not found: {hmm_path} — passing all")
        return [0.0] * len(input_sequences)

    dna_seqs = [seq_tuple[0].sequence for seq_tuple in input_sequences]
    proteins = []
    valid_indices = []
    for i, dna in enumerate(dna_seqs):
        protein = _get_protein(dna)
        if protein:
            proteins.append(protein)
            valid_indices.append(i)

    if not proteins:
        return [1.0] * len(input_sequences)

    hmm_result = run_pyhmmer_hmmsearch(
        PyHmmsearchInput(sequences=proteins, hmm=hmm_path),
        PyHmmsearchConfig(domain_evalue_threshold=evalue),
    )

    # Build per-protein domain sets
    protein_domains: Dict[int, List[str]] = {i: [] for i in range(len(proteins))}
    if (
        hmm_result.domain_hits_df is not None
        and not hmm_result.domain_hits_df.empty
    ):
        for _, row in hmm_result.domain_hits_df.iterrows():
            j = _parse_seq_index(row.get("target_name", ""))
            hmm_name = row.get("query_name", "")
            if j is not None and j in protein_domains:
                for domain in required:
                    if domain.lower() in hmm_name.lower():
                        if domain not in protein_domains[j]:
                            protein_domains[j].append(domain)

    scores = [1.0] * len(input_sequences)
    for protein_idx, original_idx in enumerate(valid_indices):
        dna = dna_seqs[original_idx]
        if dna not in CACHE:
            CACHE[dna] = {}
        CACHE[dna]["domains_found"] = protein_domains[protein_idx]

        if set(protein_domains[protein_idx]) >= set(required):
            scores[original_idx] = 0.0

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info(
        f"domain_filter: {n_pass}/{len(scores)} have all required domains"
    )
    return scores


def tracr_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by tracrRNA prediction. Caches tracrRNA and interaction energy.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import (
        CrisprTracrConfig,
        CrisprTracrInput,
        run_crispr_tracr,
    )

    sequences = [seq_tuple[0].sequence for seq_tuple in input_sequences]

    tracr_workers = len(os.sched_getaffinity(0)) or 1
    logger.info(
        f"tracr_filter: CRISPRtracrRNA prediction ({tracr_workers} workers)..."
    )
    tracr_result = run_crispr_tracr(
        CrisprTracrInput(sequences=sequences),
        CrisprTracrConfig(model_type="II", num_workers=tracr_workers),
    )
    if tracr_result.success is False:
        raise RuntimeError(
            f"tracrRNA prediction failed: {tracr_result.errors}"
        )

    scores = []
    for i, dna in enumerate(sequences):
        if dna not in CACHE:
            CACHE[dna] = {}

        if i < len(tracr_result.predictions):
            pred = tracr_result.predictions[i]
            CACHE[dna]["tracr_sequence"] = pred.tracr_hit
            CACHE[dna]["interaction_energy"] = pred.interaction_energy

            has_tracr = pred.has_tracr
            has_intarna = pred.intarna_anti_repeat_interaction is not None

            if has_tracr and has_intarna:
                scores.append(0.0)
            else:
                scores.append(1.0)
        else:
            scores.append(1.0)

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info(
        f"tracr_filter: {n_pass}/{len(scores)} have tracrRNA + IntaRNA"
    )
    return scores


def structure_filter(
    input_sequences: List[Tuple[Any, ...]],
    config: dict,
) -> List[float]:
    """Filter by AF3 structure prediction + metrics.

    Returns 0.0 for PASS, 1.0 for FAIL.
    """
    from proto_tools import (
        AlphaFold3Config,
        AlphaFold3Input,
        ColabfoldSearchConfig,
        StructurePredictionComplex,
        run_alphafold3,
    )

    plddt_threshold = config.get("plddt_threshold", PLDDT_THRESHOLD)
    rg_threshold = config.get("rg_threshold", GYRATION_RADIUS_THRESHOLD)
    alpha_threshold = config.get("alpha_threshold", LONGEST_ALPHA_THRESHOLD)
    af3_dir = config.get("af3_output_dir", "af3_pdbs")
    af3_name = "cas9"

    dna_seqs = [seq_tuple[0].sequence for seq_tuple in input_sequences]
    scores = [1.0] * len(input_sequences)

    for i, dna in enumerate(dna_seqs):
        if dna not in CACHE:
            CACHE[dna] = {}

        protein = _get_protein(dna)
        if not protein:
            continue

        # Skip AF3 if we already have a PDB for this sequence
        if "pdb_path" in CACHE[dna]:
            pdb_file = Path(CACHE[dna]["pdb_path"])
            if pdb_file.exists():
                plddt = CACHE[dna].get("plddt")
                rg = CACHE[dna].get("gyration_radius")
                alpha = CACHE[dna].get("longest_alpha")
                plddt_ok = plddt is not None and plddt >= plddt_threshold
                rg_ok = rg is not None and rg < rg_threshold
                alpha_ok = alpha is not None and alpha < alpha_threshold
                if plddt_ok and rg_ok and alpha_ok:
                    scores[i] = 0.0
                continue

        # AF3 structure prediction
        af3_idx = structure_filter._next_idx
        structure_filter._next_idx += 1
        candidate_dir = f"{af3_dir}/{af3_name}_{af3_idx}"
        try:
            af3_result = run_alphafold3(
                AlphaFold3Input(
                    complexes=[StructurePredictionComplex(chains=[protein])]
                ),
                AlphaFold3Config(
                    name=af3_name,
                    output_dir=candidate_dir,
                    use_msa=True,
                    colabfold_search_config=ColabfoldSearchConfig(
                        search_mode="local"
                    ),
                ),
            )
        except Exception as e:
            logger.error(f"  structure_filter: AF3 failed for candidate: {e}")
            continue

        structure = af3_result.structures[0]
        plddt = structure.metrics.get("avg_plddt")
        CACHE[dna]["plddt"] = plddt

        # Find PDB path
        output_dir = Path(f"{candidate_dir}_af3_results")
        pdb_file = output_dir / f"{af3_name}_0_af3.pdb"
        if not pdb_file.exists():
            # Try to find any PDB in the output directory
            pdb_files = list(output_dir.glob("*.pdb")) if output_dir.exists() else []
            if pdb_files:
                pdb_file = pdb_files[0]
            else:
                logger.error(
                    f"  structure_filter: PDB not found for candidate"
                )
                continue

        CACHE[dna]["pdb_path"] = str(pdb_file)

        # Structure metrics
        from proto_tools import (
            StructureMetricsConfig,
            StructureMetricsInput,
            run_structure_metrics,
        )

        metrics_result = run_structure_metrics(
            StructureMetricsInput(pdb_paths=[str(pdb_file)]),
            StructureMetricsConfig(),
        )

        if metrics_result.metrics:
            m = metrics_result.metrics[0]
            CACHE[dna]["gyration_radius"] = m.gyration_radius
            CACHE[dna]["longest_alpha"] = m.longest_alpha_helix

            # Combined filter
            plddt_ok = plddt is not None and plddt >= plddt_threshold
            rg_ok = (
                m.gyration_radius is not None
                and m.gyration_radius < rg_threshold
            )
            alpha_ok = (
                m.longest_alpha_helix is not None
                and m.longest_alpha_helix < alpha_threshold
            )

            if plddt_ok and rg_ok and alpha_ok:
                scores[i] = 0.0
            else:
                reasons = []
                if not plddt_ok:
                    reasons.append(f"pLDDT={plddt}")
                if not rg_ok:
                    reasons.append(f"Rg={m.gyration_radius}")
                if not alpha_ok:
                    reasons.append(f"alpha={m.longest_alpha_helix}")
                logger.info(
                    f"  structure_filter: FAIL — {'; '.join(reasons)}"
                )

    n_pass = sum(1 for s in scores if s == 0.0)
    logger.info(
        f"structure_filter: {n_pass}/{len(scores)} passed structure checks"
    )
    return scores


structure_filter._next_idx = 0


# Suppress validation warnings by setting supported sequence types
for _fn in [
    orf_filter,
    cas9_phmm_filter,
    crispr_array_filter,
    identity_filter,
    gap_gini_filter,
    domain_filter,
    tracr_filter,
    structure_filter,
]:
    _fn._constraint_supported_sequence_types = ["dna"]
    _fn._constraint_num_input_sequences_per_tuple = 1


# ============================================================================
# Program builder
# ============================================================================


def build_program(
    n_samples: int,
    temperature: float,
    top_k_val: int,
    batch_size: int,
    verbose: bool = False,
    af3_output_dir: str = "af3_pdbs",
) -> Tuple[Any, Any]:
    """Build a Program with a single TopK optimizer for one (temp, top_k) combo.

    Returns:
        (program, segment) tuple for result collection.
    """
    from proto_language.language.core import Constraint, Construct, Program, Segment
    from proto_language.language.generator.evo1_generator import (
        Evo1Generator,
        Evo1GeneratorConfig,
    )
    from proto_language.language.optimizer.topk_optimizer import (
        TopKOptimizer,
        TopKOptimizerConfig,
    )

    # Segment: DNA sequence of length NUM_TOKENS (Evo1 output length)
    # prepend_prompt=False means num_tokens = segment.sequence_length
    segment = Segment(
        length=NUM_TOKENS,
        sequence_type="dna",
        label="crispr_locus",
    )
    construct = Construct([segment], label="cas9_construct")

    # Generator
    gen_config = Evo1GeneratorConfig(
        prompts=[PROMPT],
        model_name=MODEL_NAME,
        top_k=top_k_val,
        temperature=temperature,
        prepend_prompt=False,
        batch_size=batch_size,
        verbose=verbose,
    )
    generator = Evo1Generator(gen_config)
    generator.assign(segment)

    # Filter constraints (ordered cheap -> expensive)
    constraints = [
        Constraint(
            inputs=[segment],
            function=orf_filter,
            function_config={"min_len": ORF_MIN_LEN},
            label="orf_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=cas9_phmm_filter,
            function_config={
                "evalue": CAS9_PHMM_EVALUE,
                "hmm_path": CAS9_HMM_PATH,
            },
            label="cas9_phmm_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=crispr_array_filter,
            function_config={},
            label="crispr_array_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=identity_filter,
            function_config={"threshold": IDENTITY_THRESHOLD},
            label="identity_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=gap_gini_filter,
            function_config={"threshold": GAP_GINI_THRESHOLD},
            label="gap_gini_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=domain_filter,
            function_config={
                "hmm_path": DOMAIN_HMM_PATH,
                "required_domains": REQUIRED_DOMAINS,
                "evalue": DOMAIN_EVALUE_THRESHOLD,
            },
            label="domain_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=tracr_filter,
            function_config={},
            label="tracr_filter",
            threshold=0.5,
        ),
        Constraint(
            inputs=[segment],
            function=structure_filter,
            function_config={
                "plddt_threshold": PLDDT_THRESHOLD,
                "rg_threshold": GYRATION_RADIUS_THRESHOLD,
                "alpha_threshold": LONGEST_ALPHA_THRESHOLD,
                "af3_output_dir": af3_output_dir,
            },
            label="structure_filter",
            threshold=0.5,
        ),
    ]

    # TopK optimizer: num_results = num_samples to keep all survivors
    optimizer_config = TopKOptimizerConfig(
        num_samples=n_samples,
        num_results=n_samples,
        batch_size=batch_size,
        verbose=verbose,
    )
    optimizer = TopKOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=optimizer_config,
    )

    program = Program(optimizers=[optimizer], num_results=n_samples, verbose=verbose)
    return program, segment


# ============================================================================
# Result collection
# ============================================================================


def collect_results(
    segment: Any,
    cache: Dict[str, Dict[str, Any]],
    temperature: float,
    top_k_val: int,
) -> List[dict]:
    """Collect passing candidates from segment selected_sequences and cache."""
    results = []
    for i, seq in enumerate(segment.selected_sequences):
        dna = seq.sequence
        if not dna:
            continue  # Skip empty padding

        entry = cache.get(dna, {})
        results.append(
            {
                "candidate_idx": i,
                "temperature": temperature,
                "top_k": top_k_val,
                "score": seq._metadata.get("evo1_score"),
                "identity": entry.get("identity"),
                "gap_gini": entry.get("gap_gini"),
                "domains_found": entry.get("domains_found", []),
                "interaction_energy": entry.get("interaction_energy"),
                "plddt": entry.get("plddt"),
                "gyration_radius": entry.get("gyration_radius"),
                "longest_alpha": entry.get("longest_alpha"),
                "pdb_path": entry.get("pdb_path"),
                "dna_sequence": dna,
                "crispr_repeat": entry.get("crispr_repeat"),
                "tracr_rna_sequence": entry.get("tracr_sequence"),
                "protein_sequence": entry.get("protein"),
            }
        )

    return results


def save_results(
    results: List[dict],
    output_tsv: Path,
    output_fasta: Path,
) -> None:
    """Save results to TSV and FASTA files."""
    columns = [
        "candidate_idx",
        "temperature",
        "top_k",
        "score",
        "identity",
        "gap_gini",
        "domains_found",
        "interaction_energy",
        "plddt",
        "gyration_radius",
        "longest_alpha_helix",
        "pdb_path",
        "dna_sequence",
        "crispr_repeat",
        "tracr_rna_sequence",
        "protein_sequence",
    ]

    with open(output_tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "candidate_idx": r["candidate_idx"],
                    "temperature": r["temperature"],
                    "top_k": r["top_k"],
                    "score": (
                        f"{r['score']:.4f}"
                        if r["score"] is not None
                        else ""
                    ),
                    "identity": (
                        f"{r['identity']:.4f}"
                        if r["identity"] is not None
                        else ""
                    ),
                    "gap_gini": (
                        f"{r['gap_gini']:.4f}"
                        if r["gap_gini"] is not None
                        else ""
                    ),
                    "domains_found": (
                        ",".join(r["domains_found"])
                        if r["domains_found"]
                        else ""
                    ),
                    "interaction_energy": (
                        f"{r['interaction_energy']:.2f}"
                        if r["interaction_energy"] is not None
                        else ""
                    ),
                    "plddt": (
                        f"{r['plddt']:.1f}"
                        if r["plddt"] is not None
                        else ""
                    ),
                    "gyration_radius": (
                        f"{r['gyration_radius']:.1f}"
                        if r["gyration_radius"] is not None
                        else ""
                    ),
                    "longest_alpha_helix": (
                        r["longest_alpha"]
                        if r["longest_alpha"] is not None
                        else ""
                    ),
                    "pdb_path": r["pdb_path"] or "",
                    "dna_sequence": r["dna_sequence"],
                    "crispr_repeat": r["crispr_repeat"] or "",
                    "tracr_rna_sequence": r["tracr_rna_sequence"] or "",
                    "protein_sequence": r["protein_sequence"] or "",
                }
            )

    logger.info(f"Summary TSV written to: {output_tsv}")

    with open(output_fasta, "w") as f:
        for r in results:
            plddt_str = (
                f" plddt={r['plddt']:.1f}" if r["plddt"] is not None else ""
            )
            header = (
                f">cas9_candidate_{r['candidate_idx']} "
                f"temp={r['temperature']} top_k={r['top_k']}"
                f"{plddt_str}"
            )
            f.write(f"{header}\n{r['dna_sequence']}\n")

    logger.info(f"FASTA written to: {output_fasta}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Cas9 generation pipeline (TopK optimizer version)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Samples per sweep combination (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for generation (default: same as n-samples)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cas9_candidates.fasta",
        help="Output FASTA for passing candidates (default: cas9_candidates.fasta)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    batch_size = args.batch_size or args.n_samples

    # Enable tool caching
    from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache

    _program_tool_cache.set(ToolCache())

    n_combos = len(TEMPERATURES) * len(TOP_KS)
    total_seqs = args.n_samples * n_combos
    logger.info("=" * 60)
    logger.info("Cas9 Generation Pipeline (TopK Optimizer)")
    logger.info("=" * 60)
    logger.info(
        f"Sweep: {len(TEMPERATURES)} temps x {len(TOP_KS)} top_k "
        f"= {n_combos} combos"
    )
    logger.info(f"Samples per combo: {args.n_samples}")
    logger.info(f"Total sequences: {total_seqs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 60)

    all_results: List[dict] = []
    combo_idx = 0

    for temp in TEMPERATURES:
        for top_k_val in TOP_KS:
            combo_idx += 1
            logger.info(
                f"\nCombo {combo_idx}/{n_combos}: "
                f"temp={temp}, top_k={top_k_val}"
            )

            CACHE.clear()
            structure_filter._next_idx = 0

            output_base = Path(args.output).stem.replace("_candidates", "")
            af3_output_dir = (
                f"{output_base}_af3_pdbs/temp{temp}_topk{top_k_val}"
            )
            program, segment = build_program(
                n_samples=args.n_samples,
                temperature=temp,
                top_k_val=top_k_val,
                batch_size=batch_size,
                verbose=args.verbose,
                af3_output_dir=af3_output_dir,
            )
            program.run()

            results = collect_results(segment, CACHE, temp, top_k_val)
            all_results.extend(results)

            logger.info(
                f"Combo {combo_idx}: {len(results)} candidates passed "
                f"all filters"
            )

    # Release Evo1 model from GPU memory
    try:
        from proto_tools.tools.causal_models.evo1._in_process_mode import (
            clear_evo1_cache,
        )

        clear_evo1_cache()
        logger.info("Released Evo1 model from GPU memory")
    except Exception:
        pass

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total combos: {n_combos}")
    logger.info(f"Total sequences generated: {total_seqs}")
    logger.info(f"Total passing candidates: {len(all_results)}")
    logger.info("=" * 60)

    if all_results:
        output_fasta = Path(args.output)
        output_tsv = output_fasta.with_suffix(".tsv")
        save_results(all_results, output_tsv, output_fasta)
        logger.info(f"\nPassing candidates written to: {output_fasta}")
        logger.info(f"Summary TSV written to: {output_tsv}")
    else:
        logger.info("\nNo candidates passed all filters.")

    return all_results


if __name__ == "__main__":
    main()
