import sys
sys.path.append('.')
from language.generator import ESM2Generator


def test_esm2_entropy_sampling():
    esm2_generator = ESM2Generator(
        esm2_type="esm2_t33_650M_UR50D",
        sequence_length=20,
        temperature=1.,
        decoding_method="entropy",
        top_k=5,
    )

    esm2_outputs = esm2_generator.register()
    assert len(esm2_outputs) == 1

    esm2_generator.sample()

    assert esm2_outputs[0][0].sequence is not None


def test_esm2_max_logit_sampling():
    esm2_generator = ESM2Generator(
        esm2_type="esm2_t33_650M_UR50D",
        sequence_length=20,
        temperature=1.,
        decoding_method="max_logit",
        top_k=5,
    )

    esm2_outputs = esm2_generator.register()
    assert len(esm2_outputs) == 1

    esm2_generator.sample()

    assert esm2_outputs[0][0].sequence is not None


def test_esm2_random_sampling():
    esm2_generator = ESM2Generator(
        esm2_type="esm2_t33_650M_UR50D",
        sequence_length=20,
        temperature=1.,
        decoding_method="random",
        top_k=5,
    )

    esm2_outputs = esm2_generator.register()
    assert len(esm2_outputs) == 1

    esm2_generator.sample()

    assert esm2_outputs[0][0].sequence is not None
