"""Generator registry and all registered sequence generators."""

from proto_tools.transforms.masking import MaskingStrategy

from proto_language.core.generator import GeneratorInputType
from proto_language.generator.esm2_generator import ESM2Generator, ESM2GeneratorConfig
from proto_language.generator.esm3_generator import ESM3Generator, ESM3GeneratorConfig
from proto_language.generator.evo1_generator import Evo1Generator, Evo1GeneratorConfig
from proto_language.generator.evo2_generator import Evo2Generator, Evo2GeneratorConfig
from proto_language.generator.fampnn_generator import FAMPNNGenerator, FAMPNNGeneratorConfig
from proto_language.generator.freebindcraft_generator import FreeBindCraftGenerator, FreeBindCraftGeneratorConfig
from proto_language.generator.generator_registry import GeneratorRegistry, GeneratorSpec, generator
from proto_language.generator.ligandmpnn_generator import LigandMPNNGenerator, LigandMPNNGeneratorConfig
from proto_language.generator.mpnn_mutation_generator import MPNNMutationGenerator, MPNNMutationGeneratorConfig
from proto_language.generator.msa_generator import MSAGenerator, MSAGeneratorConfig
from proto_language.generator.position_weight_generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
)
from proto_language.generator.progen2_generator import ProGen2Generator, ProGen2GeneratorConfig
from proto_language.generator.proteinmpnn_generator import ProteinMPNNGenerator, ProteinMPNNGeneratorConfig
from proto_language.generator.random_nucleotide_generator import (
    RandomNucleotideGenerator,
    RandomNucleotideGeneratorConfig,
)
from proto_language.generator.random_protein_generator import (
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
)
from proto_language.generator.rfdiffusion_mpnn_binder_generator import (
    RFdiffusionMPNNBinderGenerator,
    RFdiffusionMPNNBinderGeneratorConfig,
)
from proto_language.generator.semigreedy_mutation_generator import (
    SemigreedyMutationGenerator,
    SemigreedyMutationGeneratorConfig,
)
from proto_language.utils.sequence_matrices import SequenceLogitBiasConfig

__all__ = [
    # Masking strategies
    "MaskingStrategy",
    # Registry
    "GeneratorRegistry",
    "GeneratorSpec",
    "generator",
    # Generator input contract enum
    "GeneratorInputType",
    # Mutation generators
    "ESM2Generator",
    "ESM2GeneratorConfig",
    "ESM3Generator",
    "ESM3GeneratorConfig",
    "MPNNMutationGenerator",
    "MPNNMutationGeneratorConfig",
    "MSAGenerator",
    "MSAGeneratorConfig",
    "RandomNucleotideGenerator",
    "RandomNucleotideGeneratorConfig",
    "RandomProteinGenerator",
    "RandomProteinGeneratorConfig",
    "SemigreedyMutationGenerator",
    "SemigreedyMutationGeneratorConfig",
    # Autoregressive (language model) generators
    "Evo1Generator",
    "Evo1GeneratorConfig",
    "Evo2Generator",
    "Evo2GeneratorConfig",
    "ProGen2Generator",
    "ProGen2GeneratorConfig",
    # Inverse folding generators
    "FAMPNNGenerator",
    "FAMPNNGeneratorConfig",
    "LigandMPNNGenerator",
    "LigandMPNNGeneratorConfig",
    "ProteinMPNNGenerator",
    "ProteinMPNNGeneratorConfig",
    # Binder design generators
    "FreeBindCraftGenerator",
    "FreeBindCraftGeneratorConfig",
    # Gradient generators
    "PositionWeightGenerator",
    "PositionWeightGeneratorConfig",
    "SequenceLogitBiasConfig",
    # Binder design generators
    "RFdiffusionMPNNBinderGenerator",
    "RFdiffusionMPNNBinderGeneratorConfig",
]
