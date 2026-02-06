import Bio
from Bio import SeqIO
from Bio.Seq import Seq
from dataclasses import dataclass
from glob import glob
import numpy as np
import os
import pandas as pd
import random
import shutil
import subprocess
import tempfile
from tqdm import tqdm
from typing import Any, Dict, List, Optional


def collect_uniprot_data(uniprot_id: str, cluster_name: str = 'UniRef50') -> Dict[str, Any]:
    """
    For a given UniProt ID, get the wildtype sequence and all sequences from
    its UniRef cluster.
    """
    wt_seq = None
    for record in SeqIO.parse('examples/data/human_genes.fasta', 'fasta'):
        if record.id.startswith(f'sp|{uniprot_id}|'):
            wt_seq = str(record.seq)
            break
    if wt_seq is None:
        raise ValueError(f'Could not find sequence of {uniprot_id} in reference FASTA')

    fasta_fname = f'examples/data/human_gene_{cluster_name.lower()}/fasta/{cluster_name}_{uniprot_id}.fasta'
    if not os.path.exists(fasta_fname):
        raise FileNotFoundError(f'Could not find {fasta_fname}')
    uniref_seqs = [ str(record.seq) for record in SeqIO.parse(fasta_fname, 'fasta') ]

    return {
        'uniprot_id': uniprot_id,
        'wt_seq': wt_seq,
        'uniref_seqs': uniref_seqs,
    }


def sample_progen2(uniprot_data: Dict[str, Any], n_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Run and store the outputs of a basic sweep over ProGen2 configurations.
    """
    from proto_language.tools.language_models.progen2 import ProGen2Model

    model_checkpoints = [ 'progen2-medium', 'progen2-large', 'progen2-xlarge', ]
    prompt_lengths = [ 16, 32, 64 ]

    print('Benchmarking ProGen2...')

    data: List[Dict[str, Any]] = []  # Metadata for sampled sequences.

    for model_checkpoint in model_checkpoints:
        print(f'\tTesting ProGen2 checkpoint {model_checkpoint}...')

        progen2_model = ProGen2Model(model_checkpoint)

        for prompt_length in prompt_lengths:
            print(f'\t\tTesting ProGen2 prompt length {prompt_length}...')

            namespace = f'{model_checkpoint}:{prompt_length}'
            prompts = [ uniprot_data['wt_seq'][:prompt_length] ]

            for sample_idx in range(n_samples):
                outputs = progen2_model(
                    prompts,
                    temperature=0.2,
                    top_p=0.95,
                    top_k=0,
                    max_length=len(uniprot_data['wt_seq']) + 2,
                    truncate_at_stop=True,
                    strip_special_tokens=True,
                    prepend_prompt=True,
                    device='cuda',
                    verbose=False,
                )

                data.append({
                    'model_namespace': namespace,
                    'sample_idx': sample_idx,
                    'uniprot_id': uniprot_data['uniprot_id'],
                    'sample_seq': outputs['sequences'][0]
                })

    return data


def sample_esm3(uniprot_data: Dict[str, Any], n_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Mutate the existing wildtype sequence with ESM3.
    """

    from proto_language.tools.language_models.esm3.standalone.inference import ESM3Model

    mutation_fractions = [ 0.25, 0.5, 0.75, 1.0 ]
    temperatures = [ 0.3, 0.7 ]

    print('Benchmarking ESM3...')

    model_checkpoint = 'esm3_sm_open_v1'
    esm3_model = ESM3Model(model_checkpoint=model_checkpoint)

    data: List[Dict[str, Any]] = []  # Metadata for sampled sequences.

    for mutation_fraction in mutation_fractions:
        for temperature in temperatures:
            namespace = f'{model_checkpoint}:{mutation_fraction}:{temperature}'

            seq_len = len(uniprot_data['wt_seq'])
            num_mutations = int(mutation_fraction * seq_len)

            for sample_idx in range(n_samples):
                results = esm3_model.sample(
                    sequences=[ uniprot_data['wt_seq'] ],
                    temperature=temperature,
                    decoding_method='entropy',
                    num_mutations=num_mutations,
                    device='cuda',
                    verbose=False,
                )

                data.append({
                    'model_namespace': namespace,
                    'sample_idx': sample_idx,
                    'uniprot_id': uniprot_data['uniprot_id'],
                    'sample_seq': results[0],
                })

    return data


def human_codon_optimize(aa_seqs: List[str]) -> List[str]:
    """
    Use CodonTransformer to human-codon optimize proteins.
    """
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    import torch
    from transformers import AutoTokenizer, BigBirdForMaskedLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(device)

    nt_seqs = []
    for aa_seq in aa_seqs:
        output = predict_dna_sequence(
            protein=aa_seq,
            organism="Homo sapiens",
            device=device,
            tokenizer=tokenizer,
            model=model,
            attention_type="original_full",
            deterministic=True,
        )
        nt_seqs.append(output.predicted_dna)

    return nt_seqs


def sample_diverse_subset(sequences: List[str], k: int, kmer_size: int = 3, random_seed: int = 1337) -> List[str]:
    """
    Selects k maximally diverse sequences using Farthest Point Sampling (FPS)
    on k-mer counts.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import pairwise_distances

    N = len(sequences)
    if k >= N:
        return sequences

    # 1. Vectorize sequences into k-mer counts (Bag of Words).
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(kmer_size, kmer_size))
    X = vectorizer.fit_transform(sequences)

    # 2. Farthest Point Sampling (Greedy MaxMin).
    np.random.seed(random_seed)

    # Step A: Pick the first sequence randomly.
    selected_indices = [np.random.randint(0, N)]

    # Step B: Initialize minimum distances array.
    # This array tracks the distance from every point to the *closest* selected point so far.
    # We initialize it with the distances to our first selection.
    # We use 'cosine' metric (1 - cosine_similarity).
    min_dists = pairwise_distances(
        X[selected_indices[0]], X, metric='cosine'
    ).flatten()

    # Step C: Greedily pick the farthest point K-1 times.
    for _ in range(k - 1):
        # Find the point with the maximum distance to the current set.
        farthest_idx = np.argmax(min_dists)
        selected_indices.append(farthest_idx)

        # Calculate distances from the new point to all other points.
        new_dists = pairwise_distances(
            X[farthest_idx], X, metric='cosine'
        ).flatten()

        # Update the minimum distances.
        # For each point, is the new point closer than the previously selected ones?.
        min_dists = np.minimum(min_dists, new_dists)

    return [ sequences[i] for i in selected_indices ]


def sample_evo2(uniprot_data: Dict[str, Any], n_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Sample variation with in-context Evo 2 diversification.
    """
    from proto_language.tools.language_models.evo2 import Evo2Model

    sep_sequence = "GGGGGGGG"  # Synthetic sequence that delimits generated CDSs.
    model_checkpoint = 'evo2_7b'

    n_prompt_examples = [ 8, 16, 32 ]

    print('Benchmarking Evo 2...')

    evo2_model = Evo2Model(model_checkpoint)

    # Codon-optimize and only consider sequences within 10% of the wildtype sequence length.
    wt_seq_len = len(uniprot_data['wt_seq'])
    uniref_seqs_nt = human_codon_optimize([
        seq for seq in uniprot_data['uniref_seqs']
        if (0.9 * wt_seq_len) < len(seq) < (1.1 * wt_seq_len)
    ])

    data: List[Dict[str, Any]] = []  # Metadata for sampled sequences.

    for n_prompt_example in n_prompt_examples:
        print(f'\tTesting Evo 2 with {n_prompt_example} in-context examples...')

        if n_prompt_example > len(uniref_seqs_nt):
            print(f'\tSkipping because cluster size {len(uniprot_data["uniref_seqs"])} is smaller '
                  f'than the number of in-context examples {n_prompt_example}...')
            continue

        prompt_seqs = sample_diverse_subset(uniref_seqs_nt, n_prompt_example)
        n_tokens = max(max(len(seq) for seq in prompt_seqs), wt_seq_len * 3) + len(sep_sequence)

        batch_size = 10
        base_prompt = sep_sequence + sep_sequence.join(prompt_seqs) + sep_sequence

        for temperature in [ 0.3, 0.6, 1. ]:
            namespace = f'{model_checkpoint}:{n_prompt_example}:{temperature}'

            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)
                batch_prompts = [ base_prompt ] * current_batch_size

                output = evo2_model(
                    batch_prompts,
                    top_k=4,
                    temperature=temperature,
                    device='cuda',
                    num_tokens=n_tokens,
                    force_prompt_threshold=1,
                    print_generation=False,
                )

                for j, nt_seq in enumerate(output['sequences']):
                    try:
                        aa_seq = Seq(nt_seq.split(sep_sequence)[0]).translate()
                    except Bio.Data.CodonTable.TranslationError:
                        aa_seq = ""
                    data.append({
                        'model_namespace': namespace,
                        'sample_idx': i + j,  # Global index calculation.
                        'uniprot_id': uniprot_data['uniprot_id'],
                        'sample_seq': aa_seq,
                    })

    return data


structure_cache: Dict[str, str] = {}
esmfold_model = None
def cached_esmfold(seq: str) -> Dict[str, str]:
    """Basic caching and lazy loading mechanism for ESMFold."""
    global structure_cache
    global esmfold_model

    if esmfold_model is None:
        from proto_language.tools.structure_prediction.esmfold.inference import ESMFoldModel
        esmfold_model = ESMFoldModel()

    if seq in structure_cache:
        results = structure_cache[seq]
    else:
        results = esmfold_model(
            batch_data=[{
                #"complex_idx": 0,
                "chains": [ seq ],
                "linked_seq": seq,
                "seq_lengths": [ len(seq) ],
                "total_residues": len(seq),
                "num_chains": 1,
            }],
            residue_idx_offset=0,
            chain_linker=None,
        )
        results = results[0]
        structure_cache[seq] = results
    return results


def compute_ce_aligned_rmsd(pdb_text1: str, pdb_text2: str) -> Dict[str, Any]:
    """
    Compute CE-aligned RMSD using PyMOL's cealign.
    """
    import pymol
    from pymol import cmd

    pymol.finish_launching(['pymol', '-qc'])
    cmd.reinitialize()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1:
        f1.write(pdb_text1)
        tmp1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
        f2.write(pdb_text2)
        tmp2 = f2.name

    try:
        cmd.load(tmp1, "ref")
        cmd.load(tmp2, "mobile")

        result = cmd.cealign("ref", "mobile")

        return {
            'rmsd': result['RMSD'],
            'aligned_length': result['alignment_length'],
            'alignment_score': result.get('raw_score', None)
        }
    finally:
        os.unlink(tmp1)
        os.unlink(tmp2)
        cmd.delete("all")


def score_esmfold(seq: str, wt_seq: str, plddt_threshold: float = 0.6) -> Dict[str, float]:
    """ESMFold a sequence and optionally compare it to the wildtype structure."""
    if not isinstance(seq, str):
        return {
            'avg_plddt': None,
            'ptm': None,
            'rmsd_to_wt': None,
        }

    seq = ''.join(c for c in seq.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')

    wt_folding_results = cached_esmfold(wt_seq)

    if len(seq) > 20 and wt_folding_results['avg_plddt'] > plddt_threshold:
        folding_results = cached_esmfold(seq)
        data = {
            'avg_plddt': folding_results['avg_plddt'],
            'ptm': folding_results['ptm'],
        }
        cealign_results = compute_ce_aligned_rmsd(
            folding_results['pdb'],
            wt_folding_results['pdb'],
        )
        data['rmsd_to_wt'] = cealign_results['rmsd']
    else:
        # Do not attempt to fold or align if wildtype structure is low confidence.
        data = {
            'avg_plddt': None,
            'ptm': None,
            'rmsd_to_wt': None,
        }

    return data


@dataclass
class SequenceMatch:
    """Result of an mmseqs sequence search."""
    query_id: str
    query_seq: str
    target_id: str
    target_seq: Optional[str]
    evalue: float
    pident: float  # percent identity (0-100)
    qcov: float    # query coverage (0-100)
    score: float   # qcov * pident / 100
    alignment_length: int


def _run_mmseqs(mmseqs_path: str, args: List[str]) -> subprocess.CompletedProcess:
    """Run an MMseqs2 command."""
    cmd = [mmseqs_path] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"MMseqs2 failed: {result.stderr}\nCommand: {' '.join(cmd)}")
    return result


def find_nearest_sequences(
    query_sequences: List[str],
    fasta_files: List[str],
    evalue_threshold: float = 1e-3,
    min_qcov: float = 0.0,
    min_pident: float = 0.0,
    threads: int = -1,
    sensitivity: float = 7.5,
    mmseqs_path: str = "mmseqs",
    keep_tmp: bool = False,
) -> Dict[int, Optional[SequenceMatch]]:
    """
    Find the nearest sequence in FASTA files for each query sequence.

    Proximity is scored as: query_coverage * percent_identity / 100

    Dictionary mapping query index to best match (or None if no hit).
    """
    tmp_dir = tempfile.mkdtemp(prefix="mmseqs_")

    # -1 means use all available CPUs.
    if threads == -1:
        threads = os.cpu_count()

    try:
        # Paths
        query_fasta = os.path.join(tmp_dir, "queries.fasta")
        target_fasta = os.path.join(tmp_dir, "targets.fasta")
        query_db = os.path.join(tmp_dir, "query_db")
        target_db = os.path.join(tmp_dir, "target_db")
        result_db = os.path.join(tmp_dir, "result_db")
        result_tsv = os.path.join(tmp_dir, "results.tsv")
        tmp_folder = os.path.join(tmp_dir, "tmp")

        os.makedirs(tmp_folder, exist_ok=True)

        with open(query_fasta, 'w') as f:
            for i, seq in enumerate(query_sequences):
                f.write(f">query_{i}\n{seq}\n")

        # Concatenate target FASTA files.
        with open(target_fasta, 'w') as outf:
            for fasta_path in fasta_files:
                with open(fasta_path, 'r') as inf:
                    content = inf.read()
                    outf.write(content)
                    if not content.endswith('\n'):
                        outf.write('\n')

        # Create MMseqs2 databases.
        _run_mmseqs(mmseqs_path, ["createdb", query_fasta, query_db])
        _run_mmseqs(mmseqs_path, ["createdb", target_fasta, target_db])

        # Run search.
        _run_mmseqs(mmseqs_path, [
            "search",
            query_db,
            target_db,
            result_db,
            tmp_folder,
            "-e", str(evalue_threshold),
            "--threads", str(threads),
            "-s", str(sensitivity),
        ])

        # Convert results to TSV with all relevant columns.
        # Format: query, target, pident, alnlen, mismatch, gapopen, qstart, qend, tstart, tend, evalue, bits, qcov, tcov.
        _run_mmseqs(mmseqs_path, [
            "convertalis",
            query_db,
            target_db,
            result_db,
            result_tsv,
            "--format-output", "query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov",
        ])

        results: Dict[int, Optional[SequenceMatch]] = {i: None for i in range(len(query_sequences))}
        best_scores: Dict[int, float] = {}

        with open(result_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 14:
                    continue

                query_id = parts[0]
                target_id = parts[1]
                pident = float(parts[2])
                alnlen = int(parts[3])
                qcov = float(parts[12]) * 100  # mmseqs returns as fraction
                evalue = float(parts[10])

                # Apply filters
                if qcov < min_qcov or pident < min_pident:
                    continue

                # Calculate combined score
                score = qcov * pident / 100

                # Extract query index
                query_idx = int(query_id.replace("query_", ""))

                # Update if this is the best match
                if query_idx not in best_scores or score > best_scores[query_idx]:
                    best_scores[query_idx] = score
                    results[query_idx] = SequenceMatch(
                        query_id=query_id,
                        query_seq=query_sequences[query_idx],
                        target_id=target_id,
                        target_seq=None,  # Could fetch if needed
                        evalue=evalue,
                        pident=pident,
                        qcov=qcov,
                        score=score,
                        alignment_length=alnlen,
                    )

        return results

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def sample_sequences(uniprot_ids: List[str], output_fname: str, n_samples_per_condition: int = 10) -> None:
    """
    Conduct all of the sequence sampling.
    """
    data = []
    for uniprot_id in uniprot_ids:
        uniprot_data = collect_uniprot_data(uniprot_id, 'UniRef50')

        data += sample_progen2(uniprot_data, n_samples=n_samples_per_condition)
        data += sample_esm3(uniprot_data, n_samples=n_samples_per_condition)
        data += sample_evo2(uniprot_data, n_samples=n_samples_per_condition)

    df = pd.DataFrame(data)
    df.to_csv(output_fname, index=False, sep='\t')


def score_sequences(
    df: pd.DataFrame,
    output_fname: str,
    run_esmfold_scoring: bool = True,
    run_mmseqs_scoring: bool = True,
) -> None:
    """
    Score sequences with ESMFold (structure) and mmseqs (sequence novelty).
    """
    designed_seqs = list(df['sample_seq'])
    seq_to_scores: Dict[str, Dict[str, float]] = { seq: {} for seq in designed_seqs }

    # Get all of the wildtype sequences.
    uniprot_id_to_seq = {}
    for record in SeqIO.parse('examples/data/human_genes.fasta', 'fasta'):
        uniprot_id = record.id.split('|')[1]
        uniprot_id_to_seq[uniprot_id] = str(record.seq)

    if run_esmfold_scoring:
        print(f'Scoring sequences with ESMFold...')

        for _, row in tqdm(df.iterrows(), total=len(df)):
            wt_seq = uniprot_id_to_seq[row['uniprot_id']]
            designed_seq = row['sample_seq']

            esmfold_results = score_esmfold(designed_seq, wt_seq)

            seq_to_scores[designed_seq]['avg_plddt'] = esmfold_results['avg_plddt']
            seq_to_scores[designed_seq]['ptm'] = esmfold_results['ptm']
            seq_to_scores[designed_seq]['rmsd_to_wt'] = esmfold_results['rmsd_to_wt']

        for score_name in ['avg_plddt', 'ptm', 'rmsd_to_wt' ]:
            df[score_name] = [
                seq_to_scores[designed_seq][score_name] for designed_seq in designed_seqs
            ]

    if run_mmseqs_scoring:
        print(f'Scoring sequences with mmseqs...')

        uniref_fastas = glob('examples/data/human_gene_uniref50/fasta/UniRef50_*.fasta')

        nearest_results = find_nearest_sequences(designed_seqs, uniref_fastas)

        for i, designed_seq in enumerate(designed_seqs):
            if nearest_result := nearest_results.get(i):
                seq_to_scores[designed_seq]['evalue'] = nearest_result.evalue
                seq_to_scores[designed_seq]['identity'] = nearest_result.score
            else:
                seq_to_scores[designed_seq]['evalue'] = 0.
                seq_to_scores[designed_seq]['identity'] = 0.

        for score_name in [ 'evalue', 'identity' ]:
            df[score_name] = [
                seq_to_scores[designed_seq][score_name] for designed_seq in designed_seqs
            ]

    df.to_csv(output_fname, index=False, sep='\t')


if __name__ == '__main__':
    random.seed(1337)

    sampled_seq_fname = 'benchmark_human_protein_gen_seqs.tsv'
    sampled_stats_fname = 'benchmark_human_protein_gen_stats.tsv'

    n_samples_per_condition = 10
    uniprot_ids = [
        'O75380',
        'P01116',
        'P12074',
        'P18859',
        'P37108',
        'P39019',
        'P62314',
        'P62829',
        'Q07812',
        'Q9Y4Z0',
    ]

    sample_sequences(uniprot_ids, sampled_seq_fname, n_samples_per_condition)

    df = pd.read_csv(sampled_seq_fname, sep='\t')

    score_sequences(df, sampled_stats_fname, run_esmfold_scoring=True)
