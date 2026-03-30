#!/usr/bin/env python3
"""
Map UniProt IDs to UniRef clusters and retrieve all member sequences.

Usage: python pull_uniref_clusters.py input_ids.txt --cluster-type UniRef50 --download-sequences --output-dir results/

Input file should have one UniProt ID per line.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

# UniProt API endpoints
UNIPROT_API_BASE = "https://rest.uniprot.org"
ID_MAPPING_RUN = f"{UNIPROT_API_BASE}/idmapping/run"
ID_MAPPING_STATUS = f"{UNIPROT_API_BASE}/idmapping/status"
ID_MAPPING_RESULTS = f"{UNIPROT_API_BASE}/idmapping/results"
UNIREF_MEMBERS = f"{UNIPROT_API_BASE}/uniref"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests


def submit_id_mapping_job(uniprot_ids: list[str], cluster_type: str) -> str:
    """
    Submit a batch ID mapping job to UniProt.

    Args:
        uniprot_ids: List of UniProt accessions
        cluster_type: One of UniRef50, UniRef90, UniRef100

    Returns:
        Job ID for polling
    """
    payload = {
        "ids": ",".join(uniprot_ids),
        "from": "UniProtKB_AC-ID",
        "to": cluster_type,
    }

    response = requests.post(ID_MAPPING_RUN, data=payload)
    response.raise_for_status()

    return response.json()["jobId"]


def poll_job_status(job_id: str, poll_interval: float = 3.0, max_wait: float = 300.0) -> bool:
    """
    Poll the ID mapping job until completion.

    Args:
        job_id: Job ID from submit_id_mapping_job
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait before timeout

    Returns:
        True if job completed successfully

    Raises:
        TimeoutError: If job doesn't complete within max_wait
        RuntimeError: If job fails
    """
    url = f"{ID_MAPPING_STATUS}/{job_id}"
    elapsed = 0.0

    while elapsed < max_wait:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        if "jobStatus" in data:
            status = data["jobStatus"]
            if status == "RUNNING":
                print(f"  Job {job_id} still running... ({elapsed:.0f}s elapsed)")
                time.sleep(poll_interval)
                elapsed += poll_interval
            elif status == "FAILED":
                raise RuntimeError(f"ID mapping job {job_id} failed")
            else:
                print(f"  Job status: {status}")
                time.sleep(poll_interval)
                elapsed += poll_interval
        else:
            # No jobStatus means results are ready
            return True

    raise TimeoutError(f"Job {job_id} did not complete within {max_wait} seconds")


def get_mapping_results(job_id: str) -> dict[str, str]:
    """
    Retrieve results from a completed ID mapping job.

    Args:
        job_id: Job ID from submit_id_mapping_job

    Returns:
        Dictionary mapping UniProt ID -> UniRef cluster ID
    """
    url = f"{ID_MAPPING_RESULTS}/{job_id}"
    results = {}

    while url:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        for result in data.get("results", []):
            uniprot_id = result["from"]
            # Handle both formats: "to" can be a string or a dict with "id" key
            to_field = result["to"]
            if isinstance(to_field, dict):
                uniref_id = to_field["id"]
            else:
                uniref_id = to_field
            results[uniprot_id] = uniref_id

        # Handle pagination via Link header
        url = None
        link_header = response.headers.get("Link")
        if link_header and 'rel="next"' in link_header:
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    break

        if url:
            time.sleep(REQUEST_DELAY)

    return results


def get_cluster_members(cluster_id: str, max_members: int | None = None) -> list[dict]:
    """
    Retrieve all member sequences for a UniRef cluster.

    Args:
        cluster_id: UniRef cluster ID (e.g., UniRef50_P12345)
        max_members: Optional limit on number of members to retrieve

    Returns:
        List of member dictionaries with accession and sequence info
    """
    members = []
    params = {"size": 500}  # Page size
    url = f"{UNIREF_MEMBERS}/{cluster_id}/members?{urlencode(params)}"

    while url:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        for member in data.get("results", []):
            member_info = {
                "memberId": member.get("memberId"),
                "memberIdType": member.get("memberIdType"),
                "organismName": member.get("organismName"),
                "organismTaxId": member.get("organismTaxId"),
                "sequenceLength": member.get("sequenceLength"),
                "proteinName": member.get("proteinName"),
            }
            members.append(member_info)

            if max_members and len(members) >= max_members:
                return members

        # Handle pagination via Link header
        url = None
        link_header = response.headers.get("Link")
        if link_header and 'rel="next"' in link_header:
            # Parse next URL from Link header
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    break

        if url:
            time.sleep(REQUEST_DELAY)

    return members


def get_member_sequences_fasta(cluster_id: str, output_path: Path, members: list[dict]) -> int:
    """
    Download member sequences as FASTA file by querying UniProtKB for the member IDs.

    Args:
        cluster_id: UniRef cluster ID
        output_path: Path to write FASTA file
        members: List of member dicts from get_cluster_members()

    Returns:
        Number of sequences written
    """
    # Extract member IDs (filter out UniParc-only entries)
    member_ids = []
    for member in members:
        member_id = member.get("memberId", "")
        # Skip UniParc entries (start with UPI)
        if member_id and not member_id.startswith("UPI"):
            member_ids.append(member_id)

    if not member_ids:
        print(f"    No UniProtKB members found for {cluster_id}")
        return 0

    # Batch download sequences from UniProtKB using search/stream endpoint
    # Process in chunks to avoid URL length limits
    seq_count = 0
    chunk_size = 50  # Smaller chunks to avoid URL length issues

    with open(output_path, "w") as f:
        for i in range(0, len(member_ids), chunk_size):
            chunk = member_ids[i:i + chunk_size]
            # Build query using "id" field which matches entry names like ATP5E_HUMAN
            # and also matches accessions like P56381
            query = " OR ".join(f"id:{mid}" for mid in chunk)
            url = f"{UNIPROT_API_BASE}/uniprotkb/stream?query={query}&format=fasta"

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        f.write(line + "\n")
                        if line.startswith(">"):
                            seq_count += 1
            except requests.HTTPError as e:
                print(f"    Warning: Failed to fetch chunk {i//chunk_size + 1}: {e}")

            if i + chunk_size < len(member_ids):
                time.sleep(REQUEST_DELAY)

    return seq_count


def load_uniprot_ids(input_path: Path) -> list[str]:
    """Load UniProt IDs from a file, one per line."""
    ids = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Map UniProt IDs to UniRef clusters and retrieve member sequences"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="File containing UniProt IDs, one per line"
    )
    parser.add_argument(
        "--cluster-type",
        choices=["UniRef50", "UniRef90", "UniRef100"],
        default="UniRef50",
        help="UniRef cluster type (default: UniRef50)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("uniref_results"),
        help="Output directory for results (default: uniref_results/)"
    )
    parser.add_argument(
        "--download-sequences",
        action="store_true",
        help="Download FASTA sequences for each cluster"
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=None,
        help="Maximum members to retrieve per cluster (for member info only)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of IDs per batch for ID mapping (default: 500)"
    )

    args = parser.parse_args()

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load input IDs
    print(f"Loading UniProt IDs from {args.input_file}...")
    uniprot_ids = load_uniprot_ids(args.input_file)
    print(f"  Loaded {len(uniprot_ids)} IDs")

    # Deduplicate
    uniprot_ids = list(set(uniprot_ids))
    print(f"  {len(uniprot_ids)} unique IDs after deduplication")

    # Step 1: Map UniProt IDs to UniRef clusters in batches
    print(f"\nMapping IDs to {args.cluster_type} clusters...")
    all_mappings = {}

    for i in range(0, len(uniprot_ids), args.batch_size):
        batch = uniprot_ids[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(uniprot_ids) + args.batch_size - 1) // args.batch_size

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} IDs)...")

        # Submit job
        job_id = submit_id_mapping_job(batch, args.cluster_type)
        print(f"    Submitted job: {job_id}")

        # Poll until complete
        poll_job_status(job_id)

        # Get results
        mappings = get_mapping_results(job_id)
        all_mappings.update(mappings)
        print(f"    Retrieved {len(mappings)} mappings")

        # Be nice to the API
        time.sleep(REQUEST_DELAY)

    # Save mappings
    mappings_file = args.output_dir / "id_mappings.json"
    with open(mappings_file, "w") as f:
        json.dump(all_mappings, f, indent=2)
    print(f"\nSaved ID mappings to {mappings_file}")

    # Report unmapped IDs
    mapped_ids = set(all_mappings.keys())
    unmapped_ids = set(uniprot_ids) - mapped_ids
    if unmapped_ids:
        unmapped_file = args.output_dir / "unmapped_ids.txt"
        with open(unmapped_file, "w") as f:
            for uid in sorted(unmapped_ids):
                f.write(uid + "\n")
        print(f"  Warning: {len(unmapped_ids)} IDs could not be mapped (saved to {unmapped_file})")

    # Get unique clusters
    unique_clusters = set(all_mappings.values())
    print(f"\nFound {len(unique_clusters)} unique {args.cluster_type} clusters")

    # Step 2: Retrieve cluster members
    print("\nRetrieving cluster members...")
    cluster_info = {}

    for idx, cluster_id in enumerate(sorted(unique_clusters), 1):
        print(f"  [{idx}/{len(unique_clusters)}] {cluster_id}...")

        try:
            # Get member metadata
            members = get_cluster_members(cluster_id, max_members=args.max_members)
            cluster_info[cluster_id] = {
                "member_count": len(members),
                "members": members
            }
            print(f"    Retrieved {len(members)} members")

            # Optionally download sequences
            if args.download_sequences:
                fasta_path = args.output_dir / "fasta" / f"{cluster_id}.fasta"
                fasta_path.parent.mkdir(parents=True, exist_ok=True)
                seq_count = get_member_sequences_fasta(cluster_id, fasta_path, members)
                print(f"    Downloaded {seq_count} sequences to {fasta_path}")

        except requests.HTTPError as e:
            print(f"    Error: {e}")
            cluster_info[cluster_id] = {"error": str(e)}

        time.sleep(REQUEST_DELAY)

    # Save cluster info
    cluster_file = args.output_dir / "cluster_members.json"
    with open(cluster_file, "w") as f:
        json.dump(cluster_info, f, indent=2)
    print(f"\nSaved cluster member info to {cluster_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Input IDs: {len(uniprot_ids)}")
    print(f"  Mapped IDs: {len(mapped_ids)}")
    print(f"  Unmapped IDs: {len(unmapped_ids)}")
    print(f"  Unique clusters: {len(unique_clusters)}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
