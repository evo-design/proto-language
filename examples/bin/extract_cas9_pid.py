#!/usr/bin/env python3
"""Extract PAM Interacting Domains (PIDs) from Cas9 proposals.

Reads Cas9 proposal TSVs (from evocas9_topk), aligns each protein to
SpCas9 (UniProt Q99ZW2) using BioPython pairwise alignment, extracts the
PID subsequence (residues 1099-1368 of SpCas9, 270 aa), and computes
PID identity to the SpCas9 PID.

Optionally merges cofold metrics (AF3 + USalign) from a cofold output
directory.

Usage:
    # TSVs only:
    python examples/bin/extract_cas9_pid.py \\
        --proposal-tsvs cas9_topk_2000_168954{5,6,7,8}_proposals.tsv

    # With cofold metrics:
    python examples/bin/extract_cas9_pid.py \\
        --proposal-tsvs cas9_topk_2000_168954{5,6,7,8}_proposals.tsv \\
        --cofold-dir cofold_cas9_grna_output_v2 \\
        --output-dir pid_extraction_output
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from Bio.Align import PairwiseAligner, substitution_matrices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpCas9 reference (UniProt Q99ZW2, 1371 aa)
# ---------------------------------------------------------------------------
SPCAS9_SEQUENCE = (
    "MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAE"
    "ATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGN"
    "IVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSDVD"
    "KLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLIA"
    "LSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLS"
    "DILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGGYA"
    "GYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFDNGSIPHQIHLGELH"
    "HAILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFE"
    "EVVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAF"
    "LSGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLK"
    "IIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWG"
    "RLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSL"
    "HEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRER"
    "MKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDH"
    "IVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDN"
    "LTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKS"
    "KLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKM"
    "IAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDTAT"
    "VRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTAVAY"
    "SVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYS"
    "LFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQH"
    "KHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPA"
    "AFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD"
)

# PID boundaries (1-indexed, inclusive) from Nishimasu et al. 2014
PID_START = 1099
PID_END = 1368
SPCAS9_PID = SPCAS9_SEQUENCE[PID_START - 1 : PID_END]  # 270 aa

# sgRNA construction constants (matching cofold_cas9_grna.py)
SGRNA_SPACER = "GGAAAUUAGGUGCGCUUGGC"
TETRALOOP_LINKER = "GAAA"
DEFAULT_MAX_TRACR_LENGTH = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dna_to_rna(seq: str) -> str:
    """Convert a DNA sequence to RNA (T -> U)."""
    return seq.replace("T", "U").replace("t", "u")


def construct_sgrna(
    crispr_repeat: str,
    tracr_rna_sequence: str,
    max_tracr_length: int = DEFAULT_MAX_TRACR_LENGTH,
) -> str:
    """Construct chimeric sgRNA from crRNA repeat and tracrRNA."""
    crRNA = dna_to_rna(crispr_repeat)
    tracrRNA = dna_to_rna(tracr_rna_sequence)
    if max_tracr_length and len(tracrRNA) > max_tracr_length:
        tracrRNA = tracrRNA[:max_tracr_length]
    return SGRNA_SPACER + crRNA + TETRALOOP_LINKER + tracrRNA


def extract_pid(
    proposal_seq: str,
    pid_start: int = PID_START,
    pid_end: int = PID_END,
) -> Dict:
    """Align proposal to SpCas9 and extract the PID region.

    Returns dict with keys: pid_sequence, pid_length, pid_identity, n_matches.
    """
    aligner = PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-10,
        extend_gap_score=-0.5,
    )

    alignments = aligner.align(SPCAS9_SEQUENCE, proposal_seq)
    aln = alignments[0]

    # aln.indices is (2, alignment_length) where -1 means gap.
    # Row 0 = SpCas9 positions (0-indexed), Row 1 = proposal positions.
    indices = aln.indices  # shape (2, L)

    # Extract proposal residues corresponding to SpCas9 PID positions
    pid_residues = []
    n_matches = 0
    ref_pid = SPCAS9_SEQUENCE[pid_start - 1 : pid_end]

    for col in range(indices.shape[1]):
        ref_pos = int(indices[0, col])
        cand_pos = int(indices[1, col])
        # Skip gap columns
        if ref_pos < 0 or cand_pos < 0:
            continue
        # Check if this SpCas9 position falls within the PID
        if pid_start - 1 <= ref_pos < pid_end:
            cand_aa = proposal_seq[cand_pos]
            pid_residues.append(cand_aa)
            ref_aa = SPCAS9_SEQUENCE[ref_pos]
            if cand_aa == ref_aa:
                n_matches += 1

    pid_sequence = "".join(pid_residues)
    pid_length = len(pid_sequence)
    pid_identity = n_matches / len(ref_pid) if len(ref_pid) > 0 else 0.0

    return {
        "pid_sequence": pid_sequence,
        "pid_length": pid_length,
        "pid_identity": pid_identity,
        "n_matches": n_matches,
    }


# ---------------------------------------------------------------------------
# Load proposals
# ---------------------------------------------------------------------------
def load_proposals(tsv_paths: List[str]) -> List[Dict]:
    """Load proposal rows from TSVs, sorted by pLDDT descending."""
    rows = []
    for tsv_path in tsv_paths:
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
        logger.info(f"Loaded {len(rows)} proposals so far (added {tsv_path})")

    logger.info(f"Total: {len(rows)} proposals from {len(tsv_paths)} TSV(s)")

    # Sort by pLDDT descending (same ordering as cofold script)
    rows.sort(key=lambda r: float(r.get("plddt") or 0), reverse=True)

    # Assign global index
    for i, row in enumerate(rows):
        row["global_idx"] = i

    return rows


# ---------------------------------------------------------------------------
# Load cofold metrics
# ---------------------------------------------------------------------------
def load_cofold_metrics(cofold_dir: str) -> Dict[int, Dict]:
    """Load cofold metrics from summary.tsv in the cofold output directory.

    Returns dict mapping proposal_idx -> metrics dict.
    """
    summary_path = Path(cofold_dir) / "summary.tsv"
    if not summary_path.exists():
        logger.warning(f"Cofold summary not found: {summary_path}")
        return {}

    metrics = {}
    with open(summary_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["proposal_idx"])
            metrics[idx] = {
                "ptm": row.get("ptm", ""),
                "iptm": row.get("iptm", ""),
                "ranking_score": row.get("ranking_score", ""),
                "tm_score_proposal": row.get("tm_score_proposal", ""),
                "tm_score_reference": row.get("tm_score_reference", ""),
                "rmsd": row.get("rmsd", ""),
            }

    logger.info(f"Loaded cofold metrics for {len(metrics)} proposals")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract PAM Interacting Domains (PIDs) from Cas9 proposals.",
    )
    parser.add_argument(
        "--proposal-tsvs",
        nargs="+",
        required=True,
        help="One or more Cas9 proposal TSVs (from evocas9_topk)",
    )
    parser.add_argument(
        "--cofold-dir",
        default=None,
        help="Cofold output directory containing summary.tsv (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default="pid_extraction_output",
        help="Output directory (default: pid_extraction_output/)",
    )
    parser.add_argument(
        "--pid-start",
        type=int,
        default=PID_START,
        help=f"PID start position in SpCas9 (1-indexed, default: {PID_START})",
    )
    parser.add_argument(
        "--pid-end",
        type=int,
        default=PID_END,
        help=f"PID end position in SpCas9 (1-indexed, inclusive, default: {PID_END})",
    )
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load proposals
    rows = load_proposals(parsed.proposal_tsvs)

    # Load cofold metrics if provided
    cofold_metrics: Dict[int, Dict] = {}
    if parsed.cofold_dir:
        cofold_metrics = load_cofold_metrics(parsed.cofold_dir)

    # Output directory
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each proposal
    results = []
    for row in rows:
        global_idx = row["global_idx"]
        protein_seq = row["protein_sequence"]

        logger.info(
            f"Proposal {global_idx}: extracting PID "
            f"(protein length={len(protein_seq)})"
        )

        # Extract PID via pairwise alignment
        pid_result = extract_pid(protein_seq, parsed.pid_start, parsed.pid_end)

        # Reconstruct sgRNA
        crispr_repeat = row.get("crispr_repeat", "")
        tracr_rna = row.get("tracr_rna_sequence", "")
        sgrna = ""
        if crispr_repeat and tracr_rna:
            sgrna = construct_sgrna(crispr_repeat, tracr_rna)

        # Merge cofold metrics
        cofold = cofold_metrics.get(global_idx, {})

        result = {
            "global_idx": global_idx,
            "temperature": row.get("temperature", ""),
            "top_k": row.get("top_k", ""),
            "score": row.get("score", ""),
            "identity": row.get("identity", ""),
            "gap_gini": row.get("gap_gini", ""),
            "domains_found": row.get("domains_found", ""),
            "interaction_energy": row.get("interaction_energy", ""),
            "plddt": row.get("plddt", ""),
            "gyration_radius": row.get("gyration_radius", ""),
            "longest_alpha_helix": row.get("longest_alpha_helix", ""),
            "protein_length": len(protein_seq),
            "sgrna_length": len(sgrna) if sgrna else "",
            "pid_length": pid_result["pid_length"],
            "pid_identity": f"{pid_result['pid_identity']:.4f}",
            "ptm": cofold.get("ptm", ""),
            "iptm": cofold.get("iptm", ""),
            "ranking_score": cofold.get("ranking_score", ""),
            "tm_score_proposal": cofold.get("tm_score_proposal", ""),
            "tm_score_reference": cofold.get("tm_score_reference", ""),
            "rmsd": cofold.get("rmsd", ""),
            "protein_sequence": protein_seq,
            "pid_sequence": pid_result["pid_sequence"],
            "sgrna_sequence": sgrna,
            "dna_sequence": row.get("dna_sequence", ""),
            "crispr_repeat": crispr_repeat,
            "tracr_rna_sequence": tracr_rna,
        }
        results.append(result)

    # Sort results by PID identity descending for output
    results.sort(key=lambda r: float(r["pid_identity"]), reverse=True)

    # Write FASTA
    fasta_path = output_dir / "pid_sequences.fasta"
    with open(fasta_path, "w") as f:
        for r in results:
            header = (
                f">proposal_{r['global_idx']} "
                f"pid_identity={r['pid_identity']} "
                f"pid_length={r['pid_length']} "
                f"full_identity={r['identity']} "
                f"plddt={r['plddt']}"
            )
            f.write(header + "\n")
            # Wrap sequence at 80 chars
            seq = r["pid_sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")
    logger.info(f"Wrote PID FASTA: {fasta_path}")

    # Write TSV
    tsv_path = output_dir / "pid_summary.tsv"
    fieldnames = [
        "global_idx",
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
        "protein_length",
        "sgrna_length",
        "pid_length",
        "pid_identity",
        "ptm",
        "iptm",
        "ranking_score",
        "tm_score_proposal",
        "tm_score_reference",
        "rmsd",
        "protein_sequence",
        "pid_sequence",
        "sgrna_sequence",
        "dna_sequence",
        "crispr_repeat",
        "tracr_rna_sequence",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    logger.info(f"Wrote PID summary TSV: {tsv_path}")

    # Print ranked table to stdout
    print(f"\n{'='*100}")
    print("Cas9 PID Extraction Summary (ranked by PID identity)")
    print(f"{'='*100}")
    print(
        f"{'Idx':>4}  {'Temp':>5}  {'TopK':>4}  "
        f"{'pLDDT':>6}  {'FullID':>6}  {'PID_ID':>6}  {'PIDLen':>6}  "
        f"{'ProtLen':>7}  {'Domains'}"
    )
    print(f"{'-'*100}")
    for r in results:
        print(
            f"{r['global_idx']:>4}  "
            f"{r['temperature']:>5}  "
            f"{r['top_k']:>4}  "
            f"{float(r['plddt'] or 0):>6.1f}  "
            f"{float(r['identity'] or 0):>6.4f}  "
            f"{r['pid_identity']:>6}  "
            f"{r['pid_length']:>6}  "
            f"{r['protein_length']:>7}  "
            f"{r['domains_found']}"
        )

    # Print cofold metrics if available
    if cofold_metrics:
        print(f"\n{'='*100}")
        print("Cofold Metrics")
        print(f"{'='*100}")
        print(
            f"{'Idx':>4}  {'PID_ID':>6}  "
            f"{'pTM':>5}  {'ipTM':>5}  {'Rank':>5}  "
            f"{'TM(c)':>6}  {'TM(r)':>6}  {'RMSD':>6}"
        )
        print(f"{'-'*100}")
        for r in results:
            print(
                f"{r['global_idx']:>4}  "
                f"{r['pid_identity']:>6}  "
                f"{float(r['ptm'] or 0):>5.2f}  "
                f"{float(r['iptm'] or 0):>5.2f}  "
                f"{float(r['ranking_score'] or 0):>5.2f}  "
                f"{float(r['tm_score_proposal'] or 0):>6.3f}  "
                f"{float(r['tm_score_reference'] or 0):>6.3f}  "
                f"{float(r['rmsd'] or 0):>6.2f}"
            )

    print(f"\n{'='*100}")
    print(f"Output directory: {output_dir}")
    print(f"  PID FASTA:   {fasta_path}")
    print(f"  PID summary: {tsv_path}")
    print(f"  Total proposals: {len(results)}")
    ref_pid_len = parsed.pid_end - parsed.pid_start + 1
    print(f"  Reference PID: SpCas9 residues {parsed.pid_start}-{parsed.pid_end} ({ref_pid_len} aa)")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
