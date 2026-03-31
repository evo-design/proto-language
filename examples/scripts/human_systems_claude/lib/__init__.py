"""
Shared library for human complex diversification programs.

Note: Heavy dependencies (pymol, proto_language) are loaded lazily
when base_program functions are called.
"""

from .stoichiometry import expand_gene_ids_by_stoichiometry, get_stoichiometry


# Lazy imports for heavy modules
def load_config(config_path):
    from .base_program import load_config as _load_config
    return _load_config(config_path)

def load_wildtype_seqs(*args, **kwargs):
    from .base_program import load_wildtype_seqs as _load_wildtype_seqs
    return _load_wildtype_seqs(*args, **kwargs)

def gene_ids_to_program(*args, **kwargs):
    from .base_program import gene_ids_to_program as _gene_ids_to_program
    return _gene_ids_to_program(*args, **kwargs)

def score_complexes_in_program_with_af3(*args, **kwargs):
    from .base_program import score_complexes_in_program_with_af3 as _score
    return _score(*args, **kwargs)

def get_remote_pdb_contents(*args, **kwargs):
    from .base_program import get_remote_pdb_contents as _get_pdb
    return _get_pdb(*args, **kwargs)

__all__ = [
    "load_config",
    "load_wildtype_seqs",
    "gene_ids_to_program",
    "score_complexes_in_program_with_af3",
    "get_remote_pdb_contents",
    "get_stoichiometry",
    "expand_gene_ids_by_stoichiometry",
]
