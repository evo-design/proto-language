import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# --- DNA/RNA Scan Configuration ---
_DNA_RESNAMES = {"DA", "DT", "DG", "DC", "DI"}
_RNA_RESNAMES = {"A", "U", "G", "C", "I"}
# Mapping 3-letter/2-letter PDB codes to 1-letter sequence
_NA_MAP = {
    "DA": "A", "DT": "T", "DG": "G", "DC": "C", "DI": "I",
    "A": "A", "U": "U", "G": "G", "C": "C", "I": "I"
}

def scan_pdb_for_nucleic_acids(pdb_path: Path) -> Dict[str, Any]:
    """
    Scans a PDB file for DNA/RNA residues and extracts their sequences.
    Robustness features:
      - Reads only the first MODEL (avoids duplicating sequences in NMR files).
      - Handles HETATM and ATOM records.
      - Skips alternate conformations (only reads the first instance of a residue).
    """
    if not pdb_path.exists():
        return {'has_dna': None, 'has_rna': None, 'dna_seqs': {}, 'rna_seqs': {}}

    dna_seqs = {}
    rna_seqs = {}

    # State tracking
    current_chain = None
    last_resi = None

    try:
        with pdb_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # Stop if we hit the end of the first model (common in NMR structures)
                if line.startswith("ENDMDL"):
                    break

                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if len(line) < 27: continue

                # PDB Column parsing (0-indexed)
                chain_id = line[21]
                resi = line[22:27]
                resn = line[17:20].strip()

                # Check if we moved to a new residue
                if chain_id != current_chain or resi != last_resi:
                    current_chain = chain_id
                    last_resi = resi

                    if resn in _DNA_RESNAMES:
                        if chain_id not in dna_seqs: dna_seqs[chain_id] = []
                        dna_seqs[chain_id].append(_NA_MAP.get(resn, 'X'))
                    elif resn in _RNA_RESNAMES:
                        if chain_id not in rna_seqs: rna_seqs[chain_id] = []
                        rna_seqs[chain_id].append(_NA_MAP.get(resn, 'X'))

    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return {'has_dna': None, 'has_rna': None, 'dna_seqs': {}, 'rna_seqs': {}}

    return {
        'has_dna': bool(dna_seqs),
        'has_rna': bool(rna_seqs),
        'dna_seqs': {k: "".join(v) for k, v in dna_seqs.items()},
        'rna_seqs': {k: "".join(v) for k, v in rna_seqs.items()}
    }

def find_pdb_file(pdb_dir: Path, pdb_id: str) -> Optional[Path]:
    """Tries to find the PDB file using common naming conventions."""
    if not pdb_id:
        return None
    clean_id = pdb_id.strip().split('_')[0]
    proposals = [
        f"{clean_id}.pdb", f"{clean_id}.cif",
        f"{clean_id.lower()}.pdb", f"pdb{clean_id.lower()}.ent",
        f"{clean_id.upper()}.pdb"
    ]
    for name in proposals:
        p = pdb_dir / name
        if p.exists():
            return p
    return None

def load_concatenated_json(file_path: str):
    """Robustly parses concatenated JSON objects."""
    data = []
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []

    with open(file_path, 'r') as f:
        content = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, idx = decoder.raw_decode(content[pos:])
            data.append(obj)
            pos += idx
        except json.JSONDecodeError:
            pos += 1
    return data

def parse_error_for_oom(error_list: List[str]) -> bool:
    """Checks for OOM/Resource Exhausted errors."""
    if not error_list:
        return False
    combined = " ".join(str(e) for e in error_list)
    keywords = ["RESOURCE_EXHAUSTED", "OOM", "Out of memory", "Failed to allocate request", "xla_extension.XlaRuntimeError"]
    return any(k in combined for k in keywords)

def parse_error_for_chain_limit(error_list: List[str]) -> bool:
    """Checks for the >26 chains ValueError."""
    if not error_list:
        return False
    combined = " ".join(str(e) for e in error_list)
    return "Cannot provide more than 26 chains" in combined

def load_metadata(file_path):
    """Loads metadata from Excel or CSV."""
    if not os.path.exists(file_path):
        return None
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, sheet_name='PDB IDs')
        else:
            df = pd.read_csv(file_path)

        if 'Complex label (yours)' in df.columns:
            df.set_index('Complex label (yours)', inplace=True)
            return df
        return None
    except Exception as e:
        print(f"Warning: Failed to load metadata: {e}")
        return None

def extract_complex_data(json_objects, pdb_metadata=None, pdb_dir: Optional[Path] = None):
    rows = []
    pdb_scan_cache = {}

    for run in json_objects:
        sequences = run.get('final_sequences', {})

        complexes = run.get('complex_scores', [])
        if not complexes and 'complex_id' in run:
            complexes = [run]

        run_dir = run.get('run_dir', "")

        for complex_idx, complex_info in enumerate(complexes):
            complex_id = complex_info.get('complex_id', 'Unknown')

            chain_ids = complex_info.get('expanded_gene_ids')
            if not chain_ids:
                chain_ids = complex_info.get('gene_ids', [])

            total_residues = 0
            complex_seqs = []
            for gid in chain_ids:
                seq = sequences.get(gid, "")
                total_residues += len(seq)
                complex_seqs.append(seq)
            concatenated_sequences = ";".join(complex_seqs)

            errors = complex_info.get('errors', [])
            is_oom = parse_error_for_oom(errors)
            is_chain_limit = parse_error_for_chain_limit(errors)

            status = "Failed" if errors else "Success"

            # Determine Failure Reason
            if is_oom:
                failure_reason = "OOM"
            elif is_chain_limit:
                failure_reason = "ChainLimit"
            elif errors:
                failure_reason = "Other"
            else:
                failure_reason = "None"

            af3 = complex_info.get('af3_confidence', {})
            pdb_comps = complex_info.get('pdb_comparisons', {})

            ref_pdb = None
            tm_score = None
            rmsd = None
            for ref_pdb in list(pdb_comps.keys()):
                stats = pdb_comps[ref_pdb]
                tm_score = stats.get('tmscore')
                rmsd = stats.get('rmsd')

                if complex_idx == 0:
                    af3_pdb_suffix = ""
                else:
                    af3_pdb_suffix = f".{complex_idx}"
                af3_pdb_path = (
                    f"{run_dir}/af3_outputs_af3_results{af3_pdb_suffix}/"
                    "af3_job_0_af3.pdb"
                )

                # Scan PDB for nucleic acids.
                scan_res = {
                    'has_dna': None,
                    'has_rna': None,
                    'dna_seqs': {},
                    'rna_seqs': {},
                }
                if ref_pdb and pdb_dir:
                    if ref_pdb not in pdb_scan_cache:
                        p_path = find_pdb_file(pdb_dir, ref_pdb)
                        if p_path:
                            pdb_scan_cache[ref_pdb] = scan_pdb_for_nucleic_acids(p_path)
                        else:
                            pdb_scan_cache[ref_pdb] = scan_res

                    scan_res = pdb_scan_cache[ref_pdb]

                desc = "N/A"
                if pdb_metadata is not None and complex_id in pdb_metadata.index:
                    desc = pdb_metadata.loc[complex_id, 'What the structure corresponds to']

                rows.append({
                    'run_dir': run.get('run_dir', ''),
                    'complex_id': complex_id,
                    'n_subunits': len(chain_ids),
                    'total_residues': total_residues,
                    'status': status,
                    'failure_reason': failure_reason,
                    'plddt': af3.get('plddt'),
                    'ptm': af3.get('ptm'),
                    'iptm': af3.get('iptm'),
                    'tm_score': tm_score,
                    'rmsd': rmsd,
                    'ref_pdb': ref_pdb,
                    'ref_has_dna': scan_res['has_dna'],
                    'ref_has_rna': scan_res['has_rna'],
                    'ref_dna_seqs': (
                        str(scan_res['dna_seqs']) if scan_res['has_dna'] else "")
                    ,
                    'ref_rna_seqs': (
                        str(scan_res['rna_seqs']) if scan_res['has_rna'] else ""
                    ),
                    'af3_pdb_path': af3_pdb_path,
                    'description': desc,
                    'sequences': concatenated_sequences,
                })

    return pd.DataFrame(rows)

def analyze_and_report(df, used_pdb_dir=False):
    print("\n" + "="*80)
    print("ANALYSIS REPORT")
    print("="*80)

    # 1. OOM Analysis
    oom_runs = df[df['failure_reason'] == 'OOM']
    print(f"\n1. OOM ANALYSIS")
    print(f"Total Runs: {len(df)}")
    print(f"OOM Failures: {len(oom_runs)} ({len(oom_runs)/len(df) if len(df) > 0 else 0:.1%})")

    if not oom_runs.empty:
        print("\n  OOM Failures (Sorted by Total Residues):")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print(oom_runs[['complex_id', 'n_subunits', 'total_residues']]
              .sort_values('total_residues', ascending=False)
              .to_string(index=False))

    # 2. Chain Limit Analysis
    chain_limit_runs = df[df['failure_reason'] == 'ChainLimit']
    print(f"\n2. CHAIN LIMIT ANALYSIS (>26 Chains)")
    print(f"Failures: {len(chain_limit_runs)} ({len(chain_limit_runs)/len(df) if len(df) > 0 else 0:.1%})")

    if not chain_limit_runs.empty:
        print("\n  Chain Limit Failures (Sorted by Subunit Count):")
        print(chain_limit_runs[['complex_id', 'n_subunits', 'total_residues']]
              .sort_values('n_subunits', ascending=False)
              .to_string(index=False))

    # 3. Structural Accuracy
    success = df[df['status'] == 'Success']
    low_tm = success[success['tm_score'] < 0.5].sort_values(
        'tm_score', ascending=True,
    )

    print(f"\n" + "="*80)
    print(f"3. STRUCTURAL ACCURACY (TM < 0.5)")
    print(f"Count: {len(low_tm)}")
    if not low_tm.empty:
        print(low_tm[[
            'complex_id', 'tm_score', 'rmsd', 'ref_pdb', 'af3_pdb_path',
        ]].to_string(index=False))

    # 4. Nucleic Acid Analysis
    print(f"\n" + "="*80)
    print(f"4. MISSING NUCLEIC ACIDS ANALYSIS")

    confirmed_ids = set()

    if used_pdb_dir:
        print("[Method: Checking Reference PDB Files]")
        has_na = df[ (df['ref_has_dna'] == True) | (df['ref_has_rna'] == True) ]
        problematic = has_na[ (has_na['status'] != 'Success') | (has_na['tm_score'] < 0.6) ]

        if not problematic.empty:
            confirmed_ids = set(problematic['complex_id'])
            print(f"CONFIRMED: {len(problematic)} complexes have DNA/RNA in PDB but failed/scored low.")
            print("Listing Reference Nucleic Acid Sequences found in PDB:")
            print("-" * 80)

            for idx, row in problematic.iterrows():
                print(f"Complex: {row['complex_id']} (Ref: {row['ref_pdb']}, TM: {row['tm_score']})")
                if row['ref_has_dna']:
                    print(f"  DNA Sequences: {row['ref_dna_seqs']}")
                if row['ref_has_rna']:
                    print(f"  RNA Sequences: {row['ref_rna_seqs']}")
                print("-" * 40)
        else:
            print("  No problematic runs found that match PDBs with DNA/RNA.")
    else:
        print("[Skipped PDB Scan: No --pdb_dir provided]")

    print("\n[Method: Description Keyword Search (Supplementary)]")
    keywords = [r'\bDNA\b', r'\bRNA\b', r'nucleosome', r'promoter', r'replication', r'telomere']

    proposals = []
    for idx, row in df.iterrows():
        if row['complex_id'] in confirmed_ids: continue
        desc = str(row.get('description', ''))
        if any(re.search(k, desc, re.IGNORECASE) for k in keywords):
            if row['status'] != 'Success' or (row['tm_score'] is not None and row['tm_score'] < 0.6):
                proposals.append(row)

    if proposals:
        cand_df = pd.DataFrame(proposals)
        print(f"Found {len(proposals)} additional proposals based on text description:")
        print(cand_df[['complex_id', 'description']].to_string(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="all_json.txt")
    parser.add_argument("--excel", default="Major pathways and complexes in human biology.xlsx")
    parser.add_argument("--pdb_dir", default=None, type=str)
    parser.add_argument("--output", default="af3_analysis_summary.tsv")

    args = parser.parse_args()

    # Load
    print("Loading data...")
    json_data = load_concatenated_json(args.json)
    metadata = load_metadata(args.excel)
    pdb_path = Path(args.pdb_dir) if args.pdb_dir else None

    # Process
    print(f"Processing (PDB Scan: {pdb_path is not None})...")
    df = extract_complex_data(json_data, metadata, pdb_path)

    # Save & Report
    df.to_csv(args.output, sep='\t', index=False)
    print(f"Saved summary to {args.output}")
    analyze_and_report(df, used_pdb_dir=(pdb_path is not None))

if __name__ == "__main__":
    main()
