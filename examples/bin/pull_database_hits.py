"""
Usage: pull_database_hits.py input_fasta output_dir --database-path mmseqs_db

Save all of the near-length sequences in an input UniProt FASTA retrieved from an
mmseqs database.
"""
import argparse
import os
import shutil
import subprocess
from typing import List

from Bio import SeqIO

DB_PATH = "/large_storage/hielab/brianhie/datasets/uniref50/mmseqs/uniref50_db"
TMP_DIR = "./mmseqs_tmp"


def run_pipeline(sequences: List[str], uniprot_ids: List[str], output_dir: str) -> None:
    """
    Iterates over a list of sequences, queries UniRef50 locally,
    filters by length (10%), and saves individual FASTA files.
    """
    if len(sequences) != len(uniprot_ids):
        raise ValueError("Number of sequences must be the same as the number of UniProt IDs")

    # Ensure directories exist.
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    for i, (seq, uniprot_id) in enumerate(zip(sequences, uniprot_ids)):
        query_id = uniprot_id
        print(f"[{i+1}/{len(sequences)}] Processing {query_id} (Length: {len(seq)})...")

        # 1. Create a temporary query FASTA for this single sequence.
        query_fasta = os.path.join(TMP_DIR, "query.fasta")
        with open(query_fasta, "w") as f:
            f.write(f">{query_id}\n{seq}\n")

        # 2. Define output path for MMseqs result.
        # We use a custom format to grab the sequence (tseq) directly.
        mmseqs_out = os.path.join(TMP_DIR, "result.m8")

        # 3. Construct MMseqs2 command.
        # --format-output allows us to fetch the target sequence and lengths immediately
        # -s 6.0 is a sensitivity setting (higher is slower/more sensitive, default is 5.7)
        # --max-seqs 2000 prevents massive files if the protein is common
        cmd = [
            "mmseqs", "easy-search",
            query_fasta,
            DB_PATH,
            mmseqs_out,
            TMP_DIR,
            "--format-output", "query,target,qlen,tlen,tseq",
            "-s", "6.0",
            "-e", "1e-5"
        ]

        try:
            # Suppress stdout to keep console clean, capture stderr for errors.
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error running MMseqs for {query_id}: {e}")
            continue

        # 4. Parse results and filter.
        # If no hits found, mmseqs might not create the file or it might be empty.
        if not os.path.exists(mmseqs_out) or os.path.getsize(mmseqs_out) == 0:
            print(f"  -> No hits found for {query_id}")
            continue

        hits_found = 0
        final_fasta_path = os.path.join(output_dir, f"{query_id}_hits.fasta")

        with open(mmseqs_out, "r") as fin, open(final_fasta_path, "w") as fout:
            # Write the original query first.
            fout.write(f">{query_id}_original\n{seq}\n")

            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) != 5:
                    continue

                # Unpack based on --format-output "query,target,qlen,tlen,tseq".
                q_name, t_name, q_len_str, t_len_str, t_seq = parts

                q_len = len(seq)
                t_len = len(t_seq)

                # Filter to 10% of length.
                lower_bound = 0.9 * q_len
                upper_bound = 1.1 * q_len

                if lower_bound <= t_len <= upper_bound:
                    fout.write(f">{t_name}\n{t_seq}\n")
                    hits_found += 1

        print(f"  -> Saved {hits_found} filtered hits to {final_fasta_path}")

    # Cleanup temp dir after all are done.
    shutil.rmtree(TMP_DIR)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get database hits for a list of sequences"
    )
    parser.add_argument(
        "input_fasta",
        type=str,
        help="UniProt FASTA file containing protein sequences and UniProt IDs in the record",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for results",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default=DB_PATH,
        help=f"mmseqs database to query against, defaults to UniRef50 ({DB_PATH})",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=TMP_DIR,
        help=f"mmseqs temp director, defaults to {TMP_DIR}",
    )
    args = parser.parse_args()

    sequences, uniprot_ids = [], []
    for record in SeqIO.parse(args.input_fasta, 'fasta'):
        uniprot_id = record.id.split('|')[1]
        seq = str(record.seq)
        sequences.append(seq)
        uniprot_ids.append(uniprot_id)

    run_pipeline(sequences, uniprot_ids, args.output_dir)
