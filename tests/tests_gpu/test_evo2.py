import sys
sys.path.append('.')
from language.generator import Evo2Generator


def test_evo2_sampling():
    prompts = ['ATCG', 'AAAA']
    evo2_generator = Evo2Generator(prompt_seqs=prompts, n_tokens=100, batch_size=2)

    evo2_outputs = evo2_generator.register()
    assert len(evo2_outputs) == 1  # Generator returns one BatchedProgramSequence

    evo2_generator.sample()

    # Check that each individual sequence is not None
    for i in range(len(prompts)):
        assert evo2_outputs[0][i].sequence is not None
