"""
Usage: python examples/scripts/proteinmpnn_ensemble_baseline.py

Sample sequences from ProteinMPNN given a backbone, and use BioEmu to predict the resulting
structural ensemble of the lowest perplexity sample.
"""
import copy

from proto_tools import (
    BioEmuConfig,
    BioEmuInput,
    InverseFoldingStructureInput,
    run_bioemu,
)

from proto_language.language.core import (
    Segment,
)
from proto_language.language.generator import (
    ProteinMPNNGenerator,
    ProteinMPNNGeneratorConfig,
)

if __name__ == '__main__':
    n_samples = 200
    pdb_files = [
        'examples/data/pdb_cache/6au6.pdb',
        'examples/data/pdb_cache/3sn6.pdb',
        'examples/data/pdb_cache/1rl3.pdb',
        'examples/data/pdb_cache/2qcs.pdb',
    ]
    chains = [ 'A', 'A', 'A', 'B' ]  # Corresponds to ``pdb_files``, respectively.

    for pdb_file, chain in zip(pdb_files, chains):
        pdb_id = pdb_file.split('/')[-1].split('.')[0].upper()

        proteinmpnn_config = ProteinMPNNGeneratorConfig(
            structure_inputs=InverseFoldingStructureInput(
                structure=pdb_file,
                chain_ids=[chain],
            ),
            temperature=0.1,
        )
        proteinmpnn = ProteinMPNNGenerator(proteinmpnn_config)

        seq_len = len(
            proteinmpnn_config.structure_inputs[0].structure.get_chain_sequence(chain)
        )

        protein_segment = Segment(
            length=seq_len,
            sequence_type='protein',
        )
        protein_segment.proposal_sequences = [
            copy.deepcopy(protein_segment.original_sequence)
            for _ in range(n_samples)
        ]
        proteinmpnn.assign(protein_segment)

        print(f"Sampling {n_samples} sequences with ProteinMPNN off of {pdb_id}...")

        proteinmpnn.sample()

        # Pick the best sequence by perplexity.
        best_ppl = float('inf')
        best_seq = None
        for proposal in protein_segment.proposal_sequences:
            ppl = proposal._metadata['proteinmpnn_perplexity']
            if ppl < best_ppl:
                best_ppl = ppl
                best_seq = str(proposal.sequence)

        bioemu_input = BioEmuInput(
            complexes=[best_seq],
        )
        bioemu_config = BioEmuConfig(
            num_samples=1000,
            batch_size=100,
            output_dir=f"bioemu_ensemble_baseline_{pdb_id}",
            verbose=True,
        )

        print(f"Running BioEmu on ProteinMPNN designed seq from {pdb_id}...")

        run_bioemu(bioemu_input, bioemu_config)
