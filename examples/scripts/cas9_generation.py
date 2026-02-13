"""
Cas9 generation pipeline matching the published paper methods.

Generates candidate CRISPR-Cas9 loci using Evo1 (CRISPR fine-tuned) with the
paper's full sampling sweep, then filters through a multi-stage pipeline:

  Stage 1: ORF prediction + Cas9 pHMM search + CRISPR array detection
  Stage 2: Alignment-based identity + gap Gini filtering
  Stage 3: Domain HMM + tracrRNA (cheap filters)
  Stage 4: AF3 structure prediction + metrics (expensive)

Usage:
    python cas9_generation.py --n-samples 10
    python cas9_generation.py --n-samples 2 --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Generation
PROMPT = "`A"  # Backtick = Cas9 class token + one nucleotide
NUM_TOKENS = 8000
MODEL_NAME = "evo-1-8k-crispr"
TEMPERATURES = [0.1, 0.3, 0.5]
TOP_KS = [2, 4]

# Data paths (all relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _REPO_ROOT / "examples" / "data"
CAS9_HMM_PATH = str(_DATA_DIR / "cas_hmms" / "cas9.hmm")
DOMAIN_HMM_PATH = str(_DATA_DIR / "cas_hmms" / "cas9_domains.hmm")
TRAINING_FASTA_DIR = _DATA_DIR / "cas_training_proteins"
TRAINING_FASTA_CACHE = _REPO_ROOT / "data" / "cas_training_combined.fasta"

# Stage 1 thresholds
ORF_MIN_LEN = 3000  # Nucleotides (user override; paper uses 1800)
CAS9_PHMM_EVALUE = 1e-3

# Stage 2 thresholds
IDENTITY_THRESHOLD = 0.90  # Paper: < 90% to nearest training sequence
GAP_GINI_THRESHOLD = 0.1

# Stage 3 thresholds
DOMAIN_EVALUE_THRESHOLD = 1e-10
REQUIRED_DOMAINS = {"RuvC_1", "RuvC_2", "RuvC_3", "HNH"}

# Stage 4 thresholds
PLDDT_THRESHOLD = 75.0
GYRATION_RADIUS_THRESHOLD = 45.0
LONGEST_ALPHA_THRESHOLD = 50


@dataclass
class Candidate:
    """A generated CRISPR locus candidate tracked through the pipeline."""

    # Core identity
    sequence: str
    temperature: float
    top_k: int
    score: float
    idx: int

    # Stage 1: ORF + pHMM + CRISPR
    protein: Optional[str] = None
    has_crispr: bool = False
    has_cas9_hmm: bool = False
    crispr_repeat: Optional[str] = None

    # Stage 2: alignment
    identity: Optional[float] = None
    gap_gini: Optional[float] = None

    # Stage 3: domains + tracrRNA
    domains_found: List[str] = field(default_factory=list)
    has_tracr: bool = False
    interaction_energy: Optional[float] = None
    tracr_sequence: Optional[str] = None
    intarna_interaction: Optional[str] = None

    # Stage 4: structure
    plddt: Optional[float] = None
    gyration_radius: Optional[float] = None
    longest_alpha: Optional[int] = None
    pdb_path: Optional[str] = None

    # Pass/fail flags
    passed_stage1: bool = False
    passed_stage2: bool = False
    passed_stage3: bool = False
    passed_stage4: bool = False


# ============================================================================
# Helpers
# ============================================================================

def _get_training_fasta() -> Path:
    """Build combined training FASTA from individual .gz files (cached)."""
    if TRAINING_FASTA_CACHE.exists():
        logger.info(f"Using cached combined training FASTA: {TRAINING_FASTA_CACHE}")
        return TRAINING_FASTA_CACHE

    if not TRAINING_FASTA_DIR.exists():
        raise FileNotFoundError(f"Training FASTA directory not found: {TRAINING_FASTA_DIR}")

    fasta_files = sorted(TRAINING_FASTA_DIR.glob("*.fasta.gz"))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta.gz files found in {TRAINING_FASTA_DIR}")

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
    """Load FASTA into {id: sequence} dict. IDs are first whitespace-delimited word."""
    sequences = {}
    current_id = None
    current_seq = []

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


def _trim_alignment(al1: str, al2: str) -> tuple:
    """Center-crop to 80% and strip end gaps (matches evocas9 pipeline).

    Returns (trimmed_al1, trimmed_al2) or (None, None) if no overlap remains.
    """
    align_len = len(al1)
    start, end = int(0.1 * align_len), int(0.9 * align_len)
    al1, al2 = al1[start:end], al2[start:end]

    def _end_gaps(seq: str) -> tuple:
        first = next((i for i, c in enumerate(seq) if c != "-"), None)
        last = next((i for i, c in enumerate(reversed(seq)) if c != "-"), None)
        return first, last

    g1_start, g1_end = _end_gaps(al1)
    g2_start, g2_end = _end_gaps(al2)
    if g1_start is None or g2_start is None:
        return None, None

    trim_start = max(g1_start, g2_start)
    trim_end = max(g1_end, g2_end)
    al1 = al1[trim_start : len(al1) - trim_end]
    al2 = al2[trim_start : len(al2) - trim_end]
    if len(al1) == 0:
        return None, None

    return al1, al2


def _parse_seq_index(target_name: str) -> Optional[int]:
    """Parse integer index from HMM/mmseqs target names like 'seq_42' or '42'."""
    try:
        if target_name.startswith("seq_"):
            return int(target_name[4:])
        return int(target_name)
    except (ValueError, IndexError):
        return None


# ============================================================================
# Stage 0: Generation
# ============================================================================

def stage_generation(n_samples: int, device: str) -> List[Candidate]:
    """Generate sequences using Evo1 CRISPR model with paper's sampling sweep."""
    from proto_tools.tools.causal_models.evo1._in_process_mode import (
        Evo1SampleConfig,
        Evo1SampleInput,
        clear_evo1_cache,
        run_evo1_sample,
    )

    candidates = []
    global_idx = 0
    for temp in TEMPERATURES:
        for top_k in TOP_KS:
            logger.info(f"Generating {n_samples} sequences: temp={temp}, top_k={top_k}")
            result = run_evo1_sample(
                Evo1SampleInput(prompts=[PROMPT] * n_samples),
                Evo1SampleConfig(
                    model_name=MODEL_NAME,
                    num_tokens=NUM_TOKENS,
                    top_k=top_k,
                    temperature=temp,
                    prepend_prompt=True,
                    device=device,
                    batch_size=150,
                    keep_on_gpu=True,
                    verbose=True,
                ),
            )
            for i, seq in enumerate(result.sequences):
                score = result.scores[i] if result.scores else 0.0
                candidates.append(Candidate(
                    sequence=seq, temperature=temp, top_k=top_k,
                    score=score, idx=global_idx,
                ))
                global_idx += 1

    clear_evo1_cache()
    logger.info("Released Evo1 model from GPU memory")

    n_combos = len(TEMPERATURES) * len(TOP_KS)
    logger.info(f"GENERATION: {len(candidates)} sequences across {n_combos} sweep combos")
    return candidates


# ============================================================================
# Stage 1: ORF + Cas9 pHMM + CRISPR array
# ============================================================================

def stage1_orf_phmm_crispr(candidates: List[Candidate]) -> List[Candidate]:
    """Filter by ORF prediction, Cas9 pHMM hit, and CRISPR array presence."""
    if not candidates:
        return []

    from proto_tools import (
        MincedConfig,
        MincedInput,
        OrfipyConfig,
        OrfipyInput,
        PyHmmsearchConfig,
        PyHmmsearchInput,
        run_minced,
        run_orfipy_prediction,
        run_pyhmmer_hmmsearch,
    )

    sequences = [c.sequence for c in candidates]

    # ORF prediction
    logger.info(f"Stage 1: ORF prediction on {len(sequences)} sequences...")
    orf_result = run_orfipy_prediction(
        OrfipyInput(sequences=sequences),
        OrfipyConfig(min_len=ORF_MIN_LEN, strand="b"),
    )

    proteins, has_orf = [], []
    for seq_orfs in orf_result.predicted_orfs:
        if seq_orfs:
            best = max(seq_orfs, key=lambda o: o.amino_acid_length)
            proteins.append(best.amino_acid_sequence)
            has_orf.append(True)
        else:
            proteins.append("")
            has_orf.append(False)

    for i, c in enumerate(candidates):
        if has_orf[i]:
            c.protein = proteins[i]

    orf_count = sum(has_orf)
    logger.info(f"Stage 1: {orf_count}/{len(candidates)} have ORFs >= {ORF_MIN_LEN} nt")

    # Cas9 pHMM search
    valid_proteins = [p for p, h in zip(proteins, has_orf) if h]
    valid_indices = [i for i, h in enumerate(has_orf) if h]
    hmm_hits = set()

    if not Path(CAS9_HMM_PATH).exists():
        logger.warning(f"Cas9 HMM not found: {CAS9_HMM_PATH} — skipping pHMM filter")
    elif valid_proteins:
        logger.info(f"Stage 1: Cas9 pHMM search on {len(valid_proteins)} proteins...")
        hmm_result = run_pyhmmer_hmmsearch(
            PyHmmsearchInput(sequences=valid_proteins, hmm=CAS9_HMM_PATH),
            PyHmmsearchConfig(evalue_threshold=CAS9_PHMM_EVALUE),
        )
        if hmm_result.sequence_hits_df is not None and not hmm_result.sequence_hits_df.empty:
            for _, row in hmm_result.sequence_hits_df.iterrows():
                j = _parse_seq_index(row.get("target_name", ""))
                if j is not None and 0 <= j < len(valid_indices):
                    hmm_hits.add(valid_indices[j])

    for idx in hmm_hits:
        candidates[idx].has_cas9_hmm = True
    logger.info(f"Stage 1: {len(hmm_hits)}/{orf_count} ORFs have Cas9 pHMM hit (E < {CAS9_PHMM_EVALUE})")

    # CRISPR array detection
    logger.info("Stage 1: MinCED CRISPR array detection...")
    minced_result = run_minced(
        MincedInput(sequences=sequences),
        MincedConfig(min_num_repeats=3, min_repeat_length=23),
    )

    crispr_count = 0
    for i, c in enumerate(candidates):
        if i < len(minced_result.results):
            res = minced_result.results[i]
            c.has_crispr = res.has_crispr
            if c.has_crispr:
                crispr_count += 1
                if res.crispr_arrays:
                    c.crispr_repeat = res.crispr_arrays[0].repeats_and_spacers[0].repeat
    logger.info(f"Stage 1: {crispr_count}/{len(candidates)} have CRISPR arrays")

    # Filter
    passed = [c for c in candidates if c.has_cas9_hmm and c.has_crispr]
    for c in passed:
        c.passed_stage1 = True

    logger.info(f"STAGE 1: {len(passed)}/{len(candidates)} passed (ORF + pHMM + CRISPR)")
    return passed


# ============================================================================
# Stage 2: Alignment identity + gap Gini
# ============================================================================

def stage2_alignment(candidates: List[Candidate]) -> List[Candidate]:
    """Filter by sequence identity to training set and gap distribution."""
    if not candidates:
        return []

    from proto_tools import (
        GapGiniConfig,
        GapGiniInput,
        MafftConfig,
        MafftInput,
        MmseqsSearchProteinsConfig,
        MmseqsSearchProteinsInput,
        run_gap_gini,
        run_mafft_align,
        run_mmseqs_search_proteins,
    )

    training_fasta = _get_training_fasta()
    training_seqs = _load_training_sequences(training_fasta)
    proteins = [c.protein for c in candidates]

    # MMseqs2: find nearest training sequence for each candidate
    logger.info(f"Stage 2: MMseqs2 search for {len(proteins)} proteins...")
    mmseqs_result = run_mmseqs_search_proteins(
        MmseqsSearchProteinsInput(
            query_sequences=proteins,
            mmseqs_db=str(training_fasta),
        ),
        MmseqsSearchProteinsConfig(only_top_hits=True),
    )

    # Per-candidate: identity filter, then MAFFT + trim + gap Gini
    passed = []
    for i, c in enumerate(candidates):
        result = mmseqs_result.results[i]

        # No hit → novel sequence, passes
        if not result.has_hits:
            logger.info(f"  Candidate {c.idx}: no MMseqs2 hit — passes")
            c.identity = 0.0
            c.gap_gini = 0.0
            c.passed_stage2 = True
            passed.append(c)
            continue

        top_hit = result.top_hit
        c.identity = top_hit.pident / 100.0

        # Too similar to training set
        if c.identity >= IDENTITY_THRESHOLD:
            logger.info(f"  Candidate {c.idx}: identity={c.identity:.2%} — FILTERED")
            continue

        # Look up target protein for pairwise alignment
        target_seq = training_seqs.get(top_hit.target_id)
        if target_seq is None:
            logger.warning(
                f"  Candidate {c.idx}: target '{top_hit.target_id}' not in "
                f"training FASTA — skipping gap Gini, passing"
            )
            c.gap_gini = 0.0
            c.passed_stage2 = True
            passed.append(c)
            continue

        # MAFFT pairwise alignment → trim → gap Gini
        align_result = run_mafft_align(
            MafftInput(sequences=[c.protein, target_seq]),
            MafftConfig(),
        )
        if align_result.msa:
            al1, al2 = _trim_alignment(align_result.msa[0], align_result.msa[1])
            if al1 is not None:
                trimmed_fasta = f">query\n{al1}\n>target\n{al2}"
                gini_result = run_gap_gini(
                    GapGiniInput(alignments=[trimmed_fasta]),
                    GapGiniConfig(),
                )
                c.gap_gini = gini_result.gini_scores[0]
            else:
                c.gap_gini = 0.0
        else:
            c.gap_gini = 0.0

        if c.gap_gini is not None and c.gap_gini < GAP_GINI_THRESHOLD:
            c.passed_stage2 = True
            passed.append(c)
        else:
            logger.info(f"  Candidate {c.idx}: gap_gini={c.gap_gini:.3f} — FILTERED")

    logger.info(
        f"STAGE 2: {len(passed)}/{len(candidates)} passed "
        f"(identity < {IDENTITY_THRESHOLD:.0%}, gap_gini < {GAP_GINI_THRESHOLD})"
    )
    return passed


# ============================================================================
# Stage 3: Domain HMM + tracrRNA
# ============================================================================

def stage3_domains_tracr(candidates: List[Candidate]) -> List[Candidate]:
    """Filter by Cas9 domain presence and tracrRNA prediction."""
    if not candidates:
        return []

    from proto_tools import (
        CrisprTracrConfig,
        CrisprTracrInput,
        PyHmmsearchConfig,
        PyHmmsearchInput,
        run_crispr_tracr,
        run_pyhmmer_hmmsearch,
    )

    # Domain HMM search
    proteins_for_domain = [c.protein for c in candidates if c.protein]
    if proteins_for_domain and Path(DOMAIN_HMM_PATH).exists():
        logger.info(f"Stage 3: Domain HMM search on {len(proteins_for_domain)} proteins...")
        hmm_result = run_pyhmmer_hmmsearch(
            PyHmmsearchInput(sequences=proteins_for_domain, hmm=DOMAIN_HMM_PATH),
            PyHmmsearchConfig(domain_evalue_threshold=DOMAIN_EVALUE_THRESHOLD),
        )
        if hmm_result.domain_hits_df is not None and not hmm_result.domain_hits_df.empty:
            protein_idx_map = {
                j: c for j, c in enumerate(c for c in candidates if c.protein)
            }
            for _, row in hmm_result.domain_hits_df.iterrows():
                j = _parse_seq_index(row.get("target_name", ""))
                hmm_name = row.get("query_name", "")
                if j is not None and j in protein_idx_map:
                    c = protein_idx_map[j]
                    for domain in REQUIRED_DOMAINS:
                        if domain.lower() in hmm_name.lower():
                            if domain not in c.domains_found:
                                c.domains_found.append(domain)
    elif not Path(DOMAIN_HMM_PATH).exists():
        logger.warning(f"Domain HMM not found: {DOMAIN_HMM_PATH} — skipping")

    # tracrRNA prediction
    tracr_workers = len(os.sched_getaffinity(0)) or 1
    logger.info(f"Stage 3: CRISPRtracrRNA prediction ({tracr_workers} workers)...")
    tracr_result = run_crispr_tracr(
        CrisprTracrInput(sequences=[c.sequence for c in candidates]),
        CrisprTracrConfig(model_type="II", num_workers=tracr_workers),
    )
    if tracr_result.success is False:
        raise RuntimeError(f"tracrRNA prediction failed: {tracr_result.errors}")

    for i, c in enumerate(candidates):
        if i < len(tracr_result.predictions):
            pred = tracr_result.predictions[i]
            c.has_tracr = pred.has_tracr
            c.interaction_energy = pred.interaction_energy
            c.tracr_sequence = pred.tracr_hit
            c.intarna_interaction = pred.intarna_anti_repeat_interaction

    # Filter
    passed = []
    for c in candidates:
        reasons = []

        if Path(DOMAIN_HMM_PATH).exists():
            missing = REQUIRED_DOMAINS - set(c.domains_found)
            if missing:
                reasons.append(f"missing domains: {missing}")

        if not c.has_tracr:
            reasons.append("no tracrRNA detected")

        if c.intarna_interaction is None:
            reasons.append("no IntaRNA anti-repeat interaction")

        if reasons:
            logger.info(f"  Candidate {c.idx}: FILTERED — {'; '.join(reasons)}")
        else:
            c.passed_stage3 = True
            passed.append(c)

    logger.info(f"STAGE 3: {len(passed)}/{len(candidates)} passed")
    return passed


# ============================================================================
# Stage 4: AF3 structure + metrics
# ============================================================================

def stage4_structure(candidates: List[Candidate]) -> List[Candidate]:
    """Filter by AF3 pLDDT, gyration radius, and longest alpha helix."""
    if not candidates:
        return []

    from proto_tools import (
        AlphaFold3Config,
        AlphaFold3Input,
        ColabfoldSearchConfig,
        StructurePredictionComplex,
        run_alphafold3,
    )

    # AF3 structure prediction — each candidate gets a unique output_dir
    # so PDB files land at predictable paths (AF3 renames shared dirs on collision).
    af3_name = "cas9"
    logger.info(f"Stage 4: AF3 prediction on {len(candidates)} proteins...")
    af3_failures = 0
    for c in candidates:
        try:
            af3_result = run_alphafold3(
                AlphaFold3Input(
                    complexes=[StructurePredictionComplex(chains=[c.protein])]
                ),
                AlphaFold3Config(
                    name=af3_name,
                    output_dir=f"af3_pdbs/{af3_name}_{c.idx}",
                    use_msa=True,
                    colabfold_search_config=ColabfoldSearchConfig(search_mode="local"),
                ),
            )
            structure = af3_result.structures[0]
            c.plddt = structure.metrics.get("avg_plddt")
            pdb_file = Path(f"af3_pdbs/{af3_name}_{c.idx}_af3_results/{af3_name}_0_af3.pdb")
            if pdb_file.exists():
                c.pdb_path = str(pdb_file)
            else:
                logger.error(f"  Candidate {c.idx}: PDB file not found: {pdb_file}")
                af3_failures += 1
        except Exception as e:
            logger.error(f"  Candidate {c.idx}: AF3 prediction failed: {e}")
            af3_failures += 1

    if af3_failures == len(candidates):
        raise RuntimeError(f"AF3 failed for all {len(candidates)} candidates")

    # Structure metrics
    pdb_paths = [c.pdb_path for c in candidates if c.pdb_path]
    if pdb_paths:
        logger.info(f"Stage 4: Structure metrics for {len(pdb_paths)} structures...")
        from proto_tools import (
            StructureMetricsConfig,
            StructureMetricsInput,
            run_structure_metrics,
        )

        metrics_result = run_structure_metrics(
            StructureMetricsInput(pdb_paths=pdb_paths),
            StructureMetricsConfig(),
        )
        pdb_to_metrics = {m.pdb_path: m for m in metrics_result.metrics}
        metrics_failures = 0
        for c in candidates:
            if c.pdb_path:
                if c.pdb_path in pdb_to_metrics:
                    m = pdb_to_metrics[c.pdb_path]
                    c.gyration_radius = m.gyration_radius
                    c.longest_alpha = m.longest_alpha_helix
                else:
                    logger.error(f"  Candidate {c.idx}: no metrics for {c.pdb_path}")
                    metrics_failures += 1

        if metrics_failures == len(pdb_paths):
            raise RuntimeError(f"Structure metrics failed for all {len(pdb_paths)} structures")

    # Filter
    passed = []
    for c in candidates:
        reasons = []
        if c.plddt is None:
            reasons.append("AF3 prediction failed")
        elif c.plddt < PLDDT_THRESHOLD:
            reasons.append(f"pLDDT={c.plddt:.1f} < {PLDDT_THRESHOLD}")
        if c.gyration_radius is not None and c.gyration_radius >= GYRATION_RADIUS_THRESHOLD:
            reasons.append(f"Rg={c.gyration_radius:.1f} >= {GYRATION_RADIUS_THRESHOLD}")
        if c.longest_alpha is not None and c.longest_alpha >= LONGEST_ALPHA_THRESHOLD:
            reasons.append(f"longest_alpha={c.longest_alpha} >= {LONGEST_ALPHA_THRESHOLD}")

        if reasons:
            logger.info(f"  Candidate {c.idx}: FILTERED — {'; '.join(reasons)}")
        else:
            c.passed_stage4 = True
            passed.append(c)

    logger.info(f"STAGE 4: {len(passed)}/{len(candidates)} passed")
    return passed


# ============================================================================
# Output
# ============================================================================

def save_summary_tsv(candidates: List[Candidate], path: Path) -> None:
    """Save a TSV summary of passing designs."""
    columns = [
        "candidate_idx", "temperature", "top_k", "score",
        "identity", "gap_gini", "domains_found", "interaction_energy",
        "plddt", "gyration_radius", "longest_alpha_helix",
        "pdb_path", "dna_sequence", "crispr_repeat", "tracr_rna_sequence", "protein_sequence",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "candidate_idx": c.idx,
                "temperature": c.temperature,
                "top_k": c.top_k,
                "score": f"{c.score:.4f}",
                "identity": f"{c.identity:.4f}" if c.identity is not None else "",
                "gap_gini": f"{c.gap_gini:.4f}" if c.gap_gini is not None else "",
                "domains_found": ",".join(c.domains_found) if c.domains_found else "",
                "interaction_energy": f"{c.interaction_energy:.2f}" if c.interaction_energy is not None else "",
                "plddt": f"{c.plddt:.1f}" if c.plddt is not None else "",
                "gyration_radius": f"{c.gyration_radius:.1f}" if c.gyration_radius is not None else "",
                "longest_alpha_helix": c.longest_alpha if c.longest_alpha is not None else "",
                "pdb_path": c.pdb_path or "",
                "dna_sequence": c.sequence,
                "crispr_repeat": c.crispr_repeat or "",
                "tracr_rna_sequence": c.tracr_sequence or "",
                "protein_sequence": c.protein or "",
            })
    logger.info(f"Summary TSV written to: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cas9 generation pipeline (paper methods)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Samples per sweep combination (default: 10)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference (default: cuda)")
    parser.add_argument("--output", type=str, default="cas9_candidates.fasta",
                        help="Output FASTA for passing candidates (default: cas9_candidates.fasta)")
    args = parser.parse_args()

    # Enable tool caching so duplicate inputs are not recomputed
    from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
    _program_tool_cache.set(ToolCache())

    n_combos = len(TEMPERATURES) * len(TOP_KS)
    total_seqs = args.n_samples * n_combos
    print(f"{'='*60}")
    print(f"Cas9 Generation Pipeline")
    print(f"{'='*60}")
    print(f"Sweep: {len(TEMPERATURES)} temps x {len(TOP_KS)} top_k = {n_combos} combos")
    print(f"Samples per combo: {args.n_samples}")
    print(f"Total sequences: {total_seqs}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    candidates = stage_generation(args.n_samples, args.device)
    stage1_passed = stage1_orf_phmm_crispr(candidates)
    stage2_passed = stage2_alignment(stage1_passed)
    stage3_passed = stage3_domains_tracr(stage2_passed)
    stage4_passed = stage4_structure(stage3_passed)

    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Generated:      {len(candidates)}")
    print(f"Stage 1 passed: {len(stage1_passed)}")
    print(f"Stage 2 passed: {len(stage2_passed)}")
    print(f"Stage 3 passed: {len(stage3_passed)}")
    print(f"Stage 4 passed: {len(stage4_passed)}")
    print(f"{'='*60}")

    if stage4_passed:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            for c in stage4_passed:
                plddt_str = f" plddt={c.plddt:.1f}" if c.plddt is not None else ""
                header = (
                    f">cas9_candidate_{c.idx} "
                    f"temp={c.temperature} top_k={c.top_k} score={c.score:.4f}"
                    f"{plddt_str}"
                )
                f.write(f"{header}\n{c.sequence}\n")
        print(f"\nPassing candidates written to: {output_path}")

        summary_path = output_path.with_suffix(".tsv")
        save_summary_tsv(stage4_passed, summary_path)
        print(f"Summary TSV written to: {summary_path}")
    else:
        print("\nNo candidates passed all stages.")

    return stage4_passed


if __name__ == "__main__":
    main()
