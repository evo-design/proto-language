"""Cofold Cas9 proteins with their predicted sgRNAs using AlphaFold3.

Reads one or more Cas9 proposal TSVs (from evocas9_rejection_sampling),
constructs chimeric sgRNAs from crRNA repeat + tracrRNA components,
cofolds each Cas9-sgRNA complex with AF3, and aligns the resulting
structures to the SpCas9-sgRNA reference (PDB 4OO8) using USalign.

Usage:
    python examples/bin/cofold_cas9_grna.py proposals.tsv
    python examples/bin/cofold_cas9_grna.py a.tsv b.tsv c.tsv --output-dir my_output/
    python examples/bin/cofold_cas9_grna.py cas9_rs_2000_*_proposals.tsv --no-msa
"""

import argparse
import csv
import logging
import re
import shutil
import subprocess
from pathlib import Path

import requests
from proto_tools import Mmseqs2HomologySearchConfig
from proto_tools.tools.structure_prediction.alphafold3 import (
    AlphaFold3Config,
    AlphaFold3Input,
    run_alphafold3,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
)

logger = logging.getLogger(__name__)

# 4OO8 spacer: 20-nt guide from Nishimasu et al. 2014
SGRNA_SPACER = "GGAAAUUAGGUGCGCUUGGC"
TETRALOOP_LINKER = "GAAA"
REFERENCE_PDB_ID = "4OO8"
RCSB_PDB_URL = f"https://files.rcsb.org/download/{REFERENCE_PDB_ID}.pdb"

# 4OO8 sgRNA is 98nt: 20nt spacer + 12nt crRNA + 4nt GAAA + 62nt tracrRNA.
# CRISPRtracrRNA outputs full-length tracrRNA (~170nt), but only the 5' ~62-80nt
# is functionally relevant (anti-repeat + stem-loops 1-3). Truncating to 80nt
# gives some margin for novel Cas9s with longer anti-repeat duplexes.
DEFAULT_MAX_TRACR_LENGTH = 80


def dna_to_rna(seq: str) -> str:
    """Convert a DNA sequence to RNA (T -> U)."""
    return seq.replace("T", "U").replace("t", "u")


def construct_sgrna(
    crispr_repeat: str,
    tracr_rna_sequence: str,
    max_tracr_length: int | None = DEFAULT_MAX_TRACR_LENGTH,
) -> str:
    """Construct chimeric sgRNA from crRNA repeat and tracrRNA.

    Architecture (matching 4OO8):
        spacer + crRNA_repeat + GAAA_tetraloop + tracrRNA

    The tracrRNA is truncated to max_tracr_length (default 80nt) to match
    the functional region used in 4OO8 (62nt). Set to None for full-length.
    """
    crRNA = dna_to_rna(crispr_repeat)
    tracrRNA = dna_to_rna(tracr_rna_sequence)
    if max_tracr_length is not None and len(tracrRNA) > max_tracr_length:
        logger.info(f"Truncating tracrRNA from {len(tracrRNA)}nt to {max_tracr_length}nt")
        tracrRNA = tracrRNA[:max_tracr_length]
    return SGRNA_SPACER + crRNA + TETRALOOP_LINKER + tracrRNA


def download_reference_pdb(output_dir: Path) -> Path:
    """Download 4OO8.pdb from RCSB if not already cached.

    4OO8 contains a biological dimer (chains A-C and D-F).  We keep only
    chains A (protein), B (sgRNA), and C (target DNA) so that USalign
    compares against a single monomer complex.
    """
    pdb_path = output_dir / f"{REFERENCE_PDB_ID}.pdb"
    if pdb_path.exists():
        logger.info(f"Reference PDB already cached: {pdb_path}")
        return pdb_path

    logger.info(f"Downloading {REFERENCE_PDB_ID}.pdb from RCSB...")
    response = requests.get(RCSB_PDB_URL, timeout=60)
    response.raise_for_status()

    # Keep only monomer chains A, B, C (drop dimer mate D, E, F).
    keep_chains = {"A", "B", "C"}
    filtered_lines = []
    for line in response.text.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM", "TER", "ANISOU")):
            chain_id = line[21] if len(line) > 21 else ""
            if chain_id not in keep_chains:
                continue
        filtered_lines.append(line)
    pdb_path.write_text("".join(filtered_lines))
    logger.info(f"Saved reference PDB (chains {','.join(sorted(keep_chains))}) to {pdb_path}")
    return pdb_path


def run_usalign(
    proposal_pdb: Path,
    reference_pdb: Path,
    output_prefix: Path,
) -> dict[str, float]:
    """Run USalign on proposal vs reference and return alignment metrics.

    Returns dict with keys: tm_score_1, tm_score_2, rmsd.
    Also saves the superposed PDB and raw USalign output.
    """
    usalign_path = shutil.which("USalign")
    if not usalign_path:
        raise ImportError(
            "The 'USalign' binary is required for structural alignment. Install via: conda install -c bioconda usalign"
        )

    cmd = [
        usalign_path,
        str(proposal_pdb),
        str(reference_pdb),
        "-mm",
        "1",  # multimeric alignment mode
        "-ter",
        "1",  # treat each chain as separate entity
        "-o",
        str(output_prefix),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout

    # Save raw output
    usalign_output_path = output_prefix.parent / "usalign_output.txt"
    usalign_output_path.write_text(output)

    # Parse TM-scores
    tm_score_1 = None
    tm_score_2 = None

    match_struct1 = re.search(
        r"TM-score=\s*([0-9.]+)\s+\(normalized by length of Structure_1",
        output,
    )
    match_struct2 = re.search(
        r"TM-score=\s*([0-9.]+)\s+\(normalized by length of Structure_2",
        output,
    )

    if match_struct1:
        tm_score_1 = float(match_struct1.group(1))
    if match_struct2:
        tm_score_2 = float(match_struct2.group(1))

    # Fallback parsing
    if tm_score_1 is None or tm_score_2 is None:
        matches = re.findall(r"TM-score=\s*([0-9.]+)", output)
        if len(matches) >= 2:
            tm_score_1 = tm_score_1 if tm_score_1 is not None else float(matches[0])
            tm_score_2 = tm_score_2 if tm_score_2 is not None else float(matches[1])
        elif len(matches) == 1:
            tm_score_1 = tm_score_2 = float(matches[0])
        else:
            logger.warning("Could not find TM-score in USalign output")
            tm_score_1 = tm_score_2 = 0.0

    # Parse RMSD
    rmsd = None
    rmsd_match = re.search(r"RMSD=\s*([0-9.]+)", output)
    if rmsd_match:
        rmsd = float(rmsd_match.group(1))
    else:
        logger.warning("Could not find RMSD in USalign output")
        rmsd = float("nan")

    return {
        "tm_score_1": tm_score_1,
        "tm_score_2": tm_score_2,
        "rmsd": rmsd,
    }


def cofold_proposal(
    proposal_idx: int,
    protein_sequence: str,
    sgrna_sequence: str,
    output_dir: Path,
    seeds: list[int],
    use_msa: bool,
    verbose: bool,
) -> dict | None:
    """Run AF3 cofolding for a single Cas9-sgRNA proposal.

    Returns dict with AF3 metrics and output paths, or None on failure.
    """
    proposal_dir = output_dir / f"proposal_{proposal_idx}"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Proposal {proposal_idx}: protein={len(protein_sequence)}aa, sgRNA={len(sgrna_sequence)}nt")

    complex_input = Complex(
        chains=[
            Chain(sequence=protein_sequence, entity_type="protein"),
            Chain(sequence=sgrna_sequence, entity_type="rna"),
        ]
    )
    inputs = AlphaFold3Input(complexes=[complex_input])
    config = AlphaFold3Config(
        name=f"cas9_grna_{proposal_idx}",
        seeds=seeds,
        use_msa=use_msa,
        msa_search_config=Mmseqs2HomologySearchConfig(search_mode="local"),
        output_dir=str(proposal_dir),
        verbose=verbose,
    )

    af3_output = run_alphafold3(inputs, config)

    if not af3_output.structures:
        logger.warning(f"Proposal {proposal_idx}: AF3 returned no structures")
        return None

    structure = af3_output.structures[0]

    # Write structure to a standard PDB file
    standard_pdb = proposal_dir / "cas9_grna_af3.pdb"
    if not standard_pdb.exists():
        structure.write_pdb(standard_pdb)

    metrics = structure.metrics or {}
    return {
        "avg_plddt": metrics.get("avg_plddt"),
        "ptm": metrics.get("ptm"),
        "iptm": metrics.get("iptm"),
        "ranking_score": metrics.get("ranking_score"),
        "pdb_path": str(standard_pdb),
    }


def main(args: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(
        description="Cofold Cas9 proteins with sgRNAs using AF3 and align to 4OO8.",
    )
    parser.add_argument(
        "input_tsvs",
        nargs="+",
        help="One or more Cas9 proposal TSVs (from evocas9_rejection_sampling)",
    )
    parser.add_argument(
        "--output-dir",
        default="cofold_cas9_grna_output",
        help="Output directory (default: cofold_cas9_grna_output/)",
    )
    parser.add_argument(
        "--reference-pdb",
        default=None,
        help="Path to 4OO8.pdb (downloaded automatically if not provided)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="AF3 random seeds (default: 0)",
    )
    parser.add_argument(
        "--max-tracr-length",
        type=int,
        default=DEFAULT_MAX_TRACR_LENGTH,
        help=f"Max tracrRNA length in nt (default: {DEFAULT_MAX_TRACR_LENGTH}). "
        "4OO8 uses 62nt; 80nt gives margin for novel Cas9s.",
    )
    parser.add_argument(
        "--full-length-tracr",
        action="store_true",
        help="Use full-length tracrRNA (no truncation)",
    )
    parser.add_argument(
        "--no-msa",
        action="store_true",
        help="Disable MSA generation (faster, less accurate)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose AF3 logging",
    )
    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get reference PDB
    if parsed.reference_pdb:
        reference_pdb = Path(parsed.reference_pdb)
        if not reference_pdb.exists():
            raise FileNotFoundError(f"Reference PDB not found: {reference_pdb}")
    else:
        reference_pdb = download_reference_pdb(output_dir)

    # Read input TSV(s)
    rows = []
    for input_tsv in parsed.input_tsvs:
        with open(input_tsv) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows.extend(reader)
        logger.info(f"Loaded {len(rows)} proposals so far (added {input_tsv})")

    logger.info(f"Total: {len(rows)} proposals from {len(parsed.input_tsvs)} TSV(s)")

    # Process highest-pLDDT proposals first (most promising from generation).
    rows.sort(key=lambda r: float(r.get("plddt") or 0), reverse=True)

    max_tracr = None if parsed.full_length_tracr else parsed.max_tracr_length

    results = []
    for global_idx, row in enumerate(rows):
        orig_idx = row.get("proposal_idx", "")
        protein_sequence = row["protein_sequence"]
        crispr_repeat = row["crispr_repeat"]
        tracr_rna_sequence = row["tracr_rna_sequence"]

        if not crispr_repeat or not tracr_rna_sequence:
            logger.warning(f"Proposal {global_idx} (orig {orig_idx}): missing crRNA or tracrRNA, skipping")
            continue

        sgrna = construct_sgrna(crispr_repeat, tracr_rna_sequence, max_tracr)
        tracr_used_len = len(sgrna) - len(SGRNA_SPACER) - len(dna_to_rna(crispr_repeat)) - len(TETRALOOP_LINKER)
        logger.info(
            f"Proposal {global_idx} (orig {orig_idx}): constructed sgRNA "
            f"({len(sgrna)}nt = {len(SGRNA_SPACER)}sp + "
            f"{len(dna_to_rna(crispr_repeat))}cr + "
            f"{len(TETRALOOP_LINKER)}loop + "
            f"{tracr_used_len}tracr)"
        )

        # AF3 cofolding
        af3_result = cofold_proposal(
            proposal_idx=global_idx,
            protein_sequence=protein_sequence,
            sgrna_sequence=sgrna,
            output_dir=output_dir,
            seeds=parsed.seeds,
            use_msa=not parsed.no_msa,
            verbose=parsed.verbose,
        )

        if af3_result is None:
            logger.warning(f"Proposal {global_idx}: AF3 cofolding failed")
            continue

        # USalign against reference
        proposal_pdb = Path(af3_result["pdb_path"])
        proposal_dir = proposal_pdb.parent
        superposed_prefix = proposal_dir / "superposed_to_4OO8"

        try:
            usalign_metrics = run_usalign(proposal_pdb, reference_pdb, superposed_prefix)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Proposal {global_idx}: USalign failed: {e}")
            usalign_metrics = {
                "tm_score_1": None,
                "tm_score_2": None,
                "rmsd": None,
            }

        # Find superposed PDB (USalign writes {prefix}.pdb or {prefix}_all_atm.pdb)
        superposed_pdb = None
        for suffix in ["_all_atm.pdb", ".pdb", "_atm.pdb"]:
            proposal_path = proposal_dir / f"superposed_to_4OO8{suffix}"
            if proposal_path.exists():
                superposed_pdb = str(proposal_path)
                break

        result_row = {
            "proposal_idx": global_idx,
            "temperature": row.get("temperature", ""),
            "top_k": row.get("top_k", ""),
            "protein_length": len(protein_sequence),
            "sgrna_length": len(sgrna),
            "avg_plddt": af3_result.get("avg_plddt"),
            "ptm": af3_result.get("ptm"),
            "iptm": af3_result.get("iptm"),
            "ranking_score": af3_result.get("ranking_score"),
            "tm_score_proposal": usalign_metrics.get("tm_score_1"),
            "tm_score_reference": usalign_metrics.get("tm_score_2"),
            "rmsd": usalign_metrics.get("rmsd"),
            "pdb_path": af3_result.get("pdb_path"),
            "superposed_path": superposed_pdb,
        }
        results.append(result_row)

        logger.info(
            f"Proposal {global_idx}: "
            f"pLDDT={result_row['avg_plddt']}, "
            f"ipTM={result_row['iptm']}, "
            f"TM(cand)={result_row['tm_score_proposal']}, "
            f"TM(ref)={result_row['tm_score_reference']}, "
            f"RMSD={result_row['rmsd']}"
        )

    # Write summary TSV
    if results:
        summary_path = output_dir / "summary.tsv"
        fieldnames = [
            "proposal_idx",
            "temperature",
            "top_k",
            "protein_length",
            "sgrna_length",
            "avg_plddt",
            "ptm",
            "iptm",
            "ranking_score",
            "tm_score_proposal",
            "tm_score_reference",
            "rmsd",
            "pdb_path",
            "superposed_path",
        ]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Summary written to {summary_path}")

        # Print summary table to stdout
        print(f"\n{'=' * 80}")
        print(f"Cas9-sgRNA Cofolding Summary ({len(results)} proposals)")
        print(f"{'=' * 80}")
        header = (
            f"{'Idx':>4s}  {'Prot':>5s}  {'sgRNA':>5s}  "
            f"{'pLDDT':>6s}  {'pTM':>5s}  {'ipTM':>5s}  "
            f"{'Rank':>6s}  {'TM(c)':>6s}  {'TM(r)':>6s}  {'RMSD':>6s}"
        )
        print(header)
        print("-" * len(header))
        for r in results:

            def fmt(v, w=5, d=3):
                return f"{v:{w}.{d}f}" if v is not None else f"{'N/A':>{w}s}"

            print(
                f"{r['proposal_idx']:>4}  "
                f"{r['protein_length']:>5d}  "
                f"{r['sgrna_length']:>5d}  "
                f"{fmt(r['avg_plddt'], 6, 1)}  "
                f"{fmt(r['ptm'])}  "
                f"{fmt(r['iptm'])}  "
                f"{fmt(r['ranking_score'], 6, 3)}  "
                f"{fmt(r['tm_score_proposal'], 6, 3)}  "
                f"{fmt(r['tm_score_reference'], 6, 3)}  "
                f"{fmt(r['rmsd'], 6, 2)}"
            )
        print()
        print("TM(c) = TM-score normalized by proposal length")
        print("TM(r) = TM-score normalized by reference (4OO8) length")
        print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    main()
