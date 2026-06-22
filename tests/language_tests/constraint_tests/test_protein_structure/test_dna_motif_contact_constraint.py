"""tests/language_tests/constraint_tests/test_protein_structure/test_dna_motif_contact_constraint.py.

Config validation, pure PDB-geometry/scoring, and constraint-level soft-fail
tests for the ``dna-motif-contact-count`` constraint. These exercise the
internal parsing and scoring helpers on hand-built synthetic PDB strings; the
GPU/AF3 structure prediction path (``resolve_structure_paths``) is always mocked.
"""

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from proto_language.constraint.protein_structure.dna_motif_contact_constraint import (
    DNAMotifContactCountConfig,
    _dna_atom_allowed,
    _is_hydrogen,
    _pair_has_contact,
    _parse_pdb_atoms,
    _score_pdb_motif_contacts,
    dna_motif_contact_count_constraint,
)
from proto_language.core import Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY

PATCH_TARGET = "proto_language.constraint.protein_structure.dna_motif_contact_constraint.resolve_structure_paths"


def _atom_line(serial, name, resname, chain, resseq, x, y, z):
    """Build a single fixed-width PDB ATOM record."""
    return f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"


def _write_pdb(tmp_path, lines):
    """Write PDB lines to a temp file and return its path."""
    pdb_path = tmp_path / "complex.pdb"
    pdb_path.write_text("".join(lines) + "END\n", encoding="utf-8")
    return str(pdb_path)


def _complex_pdb_lines(protein_xyz, dna_atoms):
    """Build a minimal protein chain A + DNA chain B PDB.

    ``protein_xyz`` is a list of (resseq, x, y, z) CA atoms; ``dna_atoms`` is a
    list of (resseq, atom_name, resname, x, y, z) records for the DNA chain.
    """
    lines = []
    serial = 1
    for resseq, x, y, z in protein_xyz:
        lines.append(_atom_line(serial, "CA", "ALA", "A", resseq, x, y, z))
        serial += 1
    for resseq, atom_name, resname, x, y, z in dna_atoms:
        lines.append(_atom_line(serial, atom_name, resname, "B", resseq, x, y, z))
        serial += 1
    return lines


class TestConfigValidation:
    """Config field validation for DNAMotifContactCountConfig."""

    def test_minimal_valid_config(self):
        config = DNAMotifContactCountConfig(dna_indices=[0, 1, 2])
        assert config.dna_indices == [0, 1, 2]
        assert config.dna_chain_label == 0
        assert config.min_contacts == 1
        assert config.contact_distance_angstrom == 4.0
        assert config.dna_atom_scope == "base"

    def test_default_structure_tool_is_dna_capable(self):
        # The inherited "esmfold" default cannot fold DNA; this constraint must
        # default to a DNA-capable predictor.
        assert DNAMotifContactCountConfig(dna_indices=[0]).structure_tool == "alphafold3"

    def test_empty_dna_indices_rejected(self):
        with pytest.raises(ValidationError, match="dna_indices cannot be empty"):
            DNAMotifContactCountConfig(dna_indices=[])

    def test_negative_dna_indices_rejected(self):
        with pytest.raises(ValidationError, match="dna_indices must be non-negative"):
            DNAMotifContactCountConfig(dna_indices=[0, -1])

    def test_duplicate_dna_indices_rejected(self):
        with pytest.raises(ValidationError, match="dna_indices must be unique"):
            DNAMotifContactCountConfig(dna_indices=[1, 1, 2])

    def test_negative_distance_rejected(self):
        with pytest.raises(ValidationError):
            DNAMotifContactCountConfig(dna_indices=[0], contact_distance_angstrom=-1.0)

    def test_bad_atom_scope_rejected(self):
        with pytest.raises(ValidationError):
            DNAMotifContactCountConfig(dna_indices=[0], dna_atom_scope="sidechain")


class TestGeometryHelpers:
    """Pure geometry/parsing helpers on hand-built coordinates."""

    def test_is_hydrogen(self):
        assert _is_hydrogen("H")
        assert _is_hydrogen(" HG1")
        assert _is_hydrogen("1H")
        assert not _is_hydrogen("CA")
        assert not _is_hydrogen("N1")

    def test_dna_atom_scope_selection(self):
        # Backbone atom.
        assert _dna_atom_allowed("P", "backbone")
        assert not _dna_atom_allowed("P", "base")
        assert _dna_atom_allowed("P", "any")
        # Base atom (not in backbone set).
        assert _dna_atom_allowed("N1", "base")
        assert not _dna_atom_allowed("N1", "backbone")
        assert _dna_atom_allowed("N1", "any")

    def test_pair_has_contact_within_cutoff(self):
        protein = [np.array([0.0, 0.0, 0.0])]
        dna = [np.array([3.0, 0.0, 0.0])]
        assert _pair_has_contact(protein, dna, cutoff=4.0)

    def test_pair_no_contact_outside_cutoff(self):
        protein = [np.array([0.0, 0.0, 0.0])]
        dna = [np.array([10.0, 0.0, 0.0])]
        assert not _pair_has_contact(protein, dna, cutoff=4.0)

    def test_pair_no_contact_when_empty(self):
        assert not _pair_has_contact([], [np.array([0.0, 0.0, 0.0])], cutoff=4.0)
        assert not _pair_has_contact([np.array([0.0, 0.0, 0.0])], [], cutoff=4.0)

    def test_parse_pdb_skips_hydrogens(self, tmp_path):
        lines = [
            _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "H", "ALA", "A", 1, 0.5, 0.0, 0.0),
            _atom_line(3, "N1", "DA", "B", 1, 5.0, 0.0, 0.0),
        ]
        pdb_path = _write_pdb(tmp_path, lines)
        chain_order, _residues_by_chain, residue_atoms = _parse_pdb_atoms(pdb_path)
        assert chain_order == ["A", "B"]
        # Hydrogen excluded: ALA residue keeps only CA.
        ala_atoms = residue_atoms[("A", "1", "", "ALA")]
        assert [name for name, _ in ala_atoms] == ["CA"]


class TestScoring:
    """End-to-end scoring of _score_pdb_motif_contacts on synthetic complexes."""

    def _config(self, **kwargs):
        base = {
            "dna_indices": [0],
            "min_contacts": 1,
            "min_unique_protein_residues": 1,
            "min_unique_dna_positions": 1,
        }
        base.update(kwargs)
        return DNAMotifContactCountConfig(**base)

    def test_perfect_score_when_contact_present(self, tmp_path):
        # Protein CA at origin; DNA base atom 3A away -> within 4A cutoff.
        lines = _complex_pdb_lines(
            protein_xyz=[(1, 0.0, 0.0, 0.0)],
            dna_atoms=[(1, "N1", "DA", 3.0, 0.0, 0.0)],
        )
        pdb_path = _write_pdb(tmp_path, lines)
        score, meta = _score_pdb_motif_contacts(pdb_path, self._config())
        assert score == MIN_ENERGY
        assert meta["motif_contact_count"] == 1
        assert meta["motif_contacting_protein_residue_count"] == 1
        assert meta["motif_contacting_dna_position_count"] == 1
        assert meta["motif_contact_dna_chain_id"] == "B"

    def test_worst_score_when_no_contact(self, tmp_path):
        # DNA base atom far from protein -> no contact.
        lines = _complex_pdb_lines(
            protein_xyz=[(1, 0.0, 0.0, 0.0)],
            dna_atoms=[(1, "N1", "DA", 50.0, 0.0, 0.0)],
        )
        pdb_path = _write_pdb(tmp_path, lines)
        score, meta = _score_pdb_motif_contacts(pdb_path, self._config())
        assert score == MAX_ENERGY
        assert meta["motif_contact_count"] == 0

    def test_base_scope_excludes_backbone_contact(self, tmp_path):
        # Only a backbone atom (P) is near the protein; base-scope ignores it.
        lines = _complex_pdb_lines(
            protein_xyz=[(1, 0.0, 0.0, 0.0)],
            dna_atoms=[(1, "P", "DA", 3.0, 0.0, 0.0), (1, "N1", "DA", 50.0, 0.0, 0.0)],
        )
        pdb_path = _write_pdb(tmp_path, lines)
        score_base, meta_base = _score_pdb_motif_contacts(pdb_path, self._config(dna_atom_scope="base"))
        assert score_base == MAX_ENERGY
        assert meta_base["motif_contact_count"] == 0
        # Backbone scope counts the P contact.
        score_bb, meta_bb = _score_pdb_motif_contacts(pdb_path, self._config(dna_atom_scope="backbone"))
        assert score_bb == MIN_ENERGY
        assert meta_bb["motif_contact_count"] == 1

    def test_partial_deficit_score(self, tmp_path):
        # One contact present but min_contacts=2 -> deficit 1/2 = 0.5.
        lines = _complex_pdb_lines(
            protein_xyz=[(1, 0.0, 0.0, 0.0)],
            dna_atoms=[(1, "N1", "DA", 3.0, 0.0, 0.0)],
        )
        pdb_path = _write_pdb(tmp_path, lines)
        config = self._config(min_contacts=2, min_unique_protein_residues=1, min_unique_dna_positions=1)
        score, meta = _score_pdb_motif_contacts(pdb_path, config)
        assert score == pytest.approx(0.5)
        assert meta["motif_contact_count"] == 1

    def test_no_dna_chain_raises(self, tmp_path):
        lines = [
            _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CA", "GLY", "A", 2, 1.0, 0.0, 0.0),
        ]
        pdb_path = _write_pdb(tmp_path, lines)
        with pytest.raises(ValueError, match="No DNA chains"):
            _score_pdb_motif_contacts(pdb_path, self._config())

    def test_dna_chain_label_out_of_range_raises(self, tmp_path):
        lines = _complex_pdb_lines(
            protein_xyz=[(1, 0.0, 0.0, 0.0)],
            dna_atoms=[(1, "N1", "DA", 3.0, 0.0, 0.0)],
        )
        pdb_path = _write_pdb(tmp_path, lines)
        with pytest.raises(ValueError, match="out of range"):
            _score_pdb_motif_contacts(pdb_path, self._config(dna_chain_label=5))

    def test_dna_index_out_of_range_raises(self, tmp_path):
        lines = _complex_pdb_lines(
            protein_xyz=[(1, 0.0, 0.0, 0.0)],
            dna_atoms=[(1, "N1", "DA", 3.0, 0.0, 0.0)],
        )
        pdb_path = _write_pdb(tmp_path, lines)
        with pytest.raises(ValueError, match="outside selected DNA chain length"):
            _score_pdb_motif_contacts(pdb_path, self._config(dna_indices=[10]))


class TestConstraintSoftFail:
    """Constraint-level batch behavior with the structure resolver mocked."""

    def _candidate(self):
        protein = Sequence(sequence="MKQ", sequence_type="protein")
        dna = Sequence(sequence="ACGT", sequence_type="dna")
        return (protein, dna)

    def _config(self, **kwargs):
        return DNAMotifContactCountConfig(dna_indices=kwargs.pop("dna_indices", [0]), **kwargs)

    def test_empty_input(self):
        assert dna_motif_contact_count_constraint([], self._config()) == []

    def test_no_dna_chain_soft_fails(self, tmp_path, caplog):
        # A protein-only structure cannot be scored; soft-fail this proposal.
        lines = [
            _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CA", "GLY", "A", 2, 1.0, 0.0, 0.0),
        ]
        pdb_path = _write_pdb(tmp_path, lines)
        logger_name = "proto_language.constraint.protein_structure.dna_motif_contact_constraint"
        with patch(PATCH_TARGET, return_value=[pdb_path]), caplog.at_level("WARNING", logger=logger_name):
            results = dna_motif_contact_count_constraint([self._candidate()], self._config())
        assert len(results) == 1
        assert results[0].score == MAX_ENERGY
        assert "No DNA chains" in results[0].metadata["motif_contact_error"]
        assert results[0].metadata["motif_contact_pdb_path"] == pdb_path
        assert "dna-motif-contact" in caplog.text

    def test_partial_batch_soft_fails_only_bad_proposal(self, tmp_path):
        # _write_pdb uses a fixed filename, so write each PDB in its own subdir.
        good_dir = tmp_path / "good"
        bad_dir = tmp_path / "bad"
        good_dir.mkdir()
        bad_dir.mkdir()
        good_pdb = _write_pdb(
            good_dir,
            _complex_pdb_lines(
                protein_xyz=[(1, 0.0, 0.0, 0.0)],
                dna_atoms=[(1, "N1", "DA", 3.0, 0.0, 0.0)],
            ),
        )
        bad_pdb = _write_pdb(bad_dir, [_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)])
        with patch(PATCH_TARGET, return_value=[good_pdb, bad_pdb]):
            results = dna_motif_contact_count_constraint([self._candidate(), self._candidate()], self._config())
        good, bad = results
        assert good.score == MIN_ENERGY
        assert good.metadata["motif_contact_count"] == 1
        assert "motif_contact_error" not in good.metadata
        assert bad.score == MAX_ENERGY
        assert "No DNA chains" in bad.metadata["motif_contact_error"]
