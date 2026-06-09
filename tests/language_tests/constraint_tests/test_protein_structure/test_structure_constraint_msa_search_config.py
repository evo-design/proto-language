"""Regression tests for the colabfold-search -> mmseqs2-homology-search rename.

proto-tools #1179/#1193 consolidated ``colabfold-search`` into
``mmseqs2-homology-search`` and renamed every structure predictor's MSA-search
sub-config from ``colabfold_search_config`` (``ColabfoldSearchConfig``) to
``msa_search_config`` (``Mmseqs2HomologySearchConfig``). Predictor configs are
``extra="forbid"``, so any program still carrying the legacy field hard-fails
validation. In prod this surfaced as
``constraint 'structure-distogram-cce' config invalid —
esmfold2_config.colabfold_search_config: Extra inputs are not permitted`` (every
predictor sub-config rejected), with a misleading "Optimization timed out"
headline.
"""

import pytest
from proto_tools import Mmseqs2HomologySearchConfig
from pydantic import ValidationError

from proto_language.constraint.protein_structure.structure_constraint_config import (
    StructureBasedConstraintConfig,
)

# MSA-capable predictor sub-config fields named in the prod error panel.
MSA_PREDICTOR_FIELDS = [
    "esmfold2_config",
    "alphafold3_config",
    "boltz2_config",
    "chai1_config",
    "protenix_config",
    "alphafold2_config",
]


@pytest.mark.parametrize("predictor_field", MSA_PREDICTOR_FIELDS)
def test_legacy_colabfold_search_config_is_rejected(predictor_field):
    """Reproduce the prod rejection: the renamed-away field is now forbidden."""
    with pytest.raises(ValidationError, match="colabfold_search_config"):
        StructureBasedConstraintConfig(
            structure_tool="alphafold2_binder",
            alphafold2_binder_config={"target_pdb": "MOCK_PDB"},
            # use_msa=False isolates the failure to the extra legacy key, rather
            # than relying on the extra-forbid error firing before any predictor
            # use_msa-capability guard.
            **{predictor_field: {"use_msa": False, "colabfold_search_config": {"search_mode": "remote"}}},
        )


@pytest.mark.parametrize("predictor_field", MSA_PREDICTOR_FIELDS)
def test_msa_search_config_is_accepted(predictor_field):
    """The renamed field validates and parses into a Mmseqs2HomologySearchConfig."""
    config = StructureBasedConstraintConfig(
        structure_tool="alphafold2_binder",
        alphafold2_binder_config={"target_pdb": "MOCK_PDB"},
        # use_msa=False sidesteps predictor-specific MSA-capability guards (e.g.
        # the esmfold2-fast checkpoint); the field itself must still validate.
        **{predictor_field: {"use_msa": False, "msa_search_config": {"search_mode": "remote"}}},
    )
    sub_config = getattr(config, predictor_field)
    assert isinstance(sub_config.msa_search_config, Mmseqs2HomologySearchConfig)
    assert sub_config.msa_search_config.search_mode == "remote"
