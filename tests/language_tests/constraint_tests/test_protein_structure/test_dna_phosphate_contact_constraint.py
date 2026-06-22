"""Tests for the DNA phosphate contact constraint.

Exercises config validation and the internal PDB geometry / scoring helper on
small synthetic PDB inputs, without GPU or structure prediction (the structure
resolver is mocked so no AF3 call is made).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from proto_language.constraint.protein_structure.dna_phosphate_contact_constraint import (
    DNAPhosphateContactConfig,
    _analyze_phosphate_contacts,
    dna_phosphate_contact_constraint,
)
from proto_language.core import Sequence
from proto_language.utils import MAX_ENERGY

PATCH_TARGET = "proto_language.constraint.protein_structure.dna_phosphate_contact_constraint.resolve_structure_paths"


def _atom(serial: int, atom: str, resname: str, chain: str, resseq: int, xyz):
    """Build a single fixed-width PDB ATOM record with exact column placement."""
    x, y, z = xyz
    # Atom names < 4 chars are left-padded one space (cols 13-16); 4-char names fill 13-16.
    name_field = atom if len(atom) >= 4 else f" {atom:<3}"
    record = "ATOM  "  # cols 1-6
    record += f"{serial:>5}"  # cols 7-11
    record += " "  # col 12
    record += f"{name_field:<4}"  # cols 13-16
    record += " "  # col 17 (altLoc)
    record += f"{resname:>3}"  # cols 18-20
    record += " "  # col 21
    record += f"{chain:1}"  # col 22
    record += f"{resseq:>4}"  # cols 23-26
    record += " " * 4  # cols 27-30 (insertion code + pad)
    record += f"{x:8.3f}{y:8.3f}{z:8.3f}"  # cols 31-54
    record += "  1.00  0.00\n"
    return record


def _write_pdb(tmp_path: Path, lines: list[str], name: str = "complex") -> str:
    """Write PDB lines to a uniquely named temp file and return its path."""
    pdb = tmp_path / f"{name}_{len(list(tmp_path.glob('*.pdb')))}.pdb"
    pdb.write_text("".join(lines) + "END\n", encoding="utf-8")
    return str(pdb)


def _contact_pdb(tmp_path: Path) -> str:
    """PDB where an ARG side-chain contacts a DNA phosphate oxygen.

    ARG NH1 sits within the default 3.5 A cutoff of guanine OP1, yielding one
    phosphate contact from ARG (weight 7). A base atom and a far ARG atom are
    included to confirm only phosphate oxygens within cutoff count.
    """
    lines = [
        # ARG polar sidechain atom near the phosphate oxygen.
        _atom(1, "NH1", "ARG", "A", 10, (0.0, 0.0, 0.0)),
        _atom(2, "NH2", "ARG", "A", 10, (0.0, 0.0, 50.0)),
        # DNA backbone phosphate oxygen (OP1) within cutoff.
        _atom(3, "OP1", "DG", "B", 5, (1.0, 0.0, 0.0)),
        # A DNA base atom that must be ignored (not a phosphate atom).
        _atom(4, "O6", "DG", "B", 5, (1.0, 0.0, 0.5)),
    ]
    return _write_pdb(tmp_path, lines)


def _no_contact_pdb(tmp_path: Path) -> str:
    """PDB where protein and DNA phosphate atoms are far apart (no contacts)."""
    lines = [
        _atom(1, "NH1", "ARG", "A", 10, (0.0, 0.0, 0.0)),
        _atom(2, "OP1", "DG", "B", 5, (50.0, 50.0, 50.0)),
    ]
    return _write_pdb(tmp_path, lines)


class TestConfigValidation:
    """Config field defaults and validation bounds."""

    def test_defaults(self):
        """Default config carries the ported scoring targets."""
        config = DNAPhosphateContactConfig()
        assert config.contact_cutoff == 3.5
        assert config.desired_phosphate_contacts == 2
        assert config.phosphate_atoms == ["OP1", "OP2", "O1P", "O2P"]

    def test_default_structure_tool_is_dna_capable(self):
        """Default predictor is overridden to a DNA-capable tool (esmfold can't fold DNA)."""
        assert DNAPhosphateContactConfig().structure_tool == "alphafold3"

    def test_contact_cutoff_must_be_positive(self):
        """contact_cutoff has a gt=0 bound."""
        with pytest.raises(ValueError):
            DNAPhosphateContactConfig(contact_cutoff=0.0)

    def test_negative_desired_rejected(self):
        """desired_phosphate_contacts has a ge=0 bound."""
        with pytest.raises(ValueError):
            DNAPhosphateContactConfig(desired_phosphate_contacts=-1)

    def test_custom_phosphate_atoms(self):
        """phosphate_atoms can be overridden to include bridging atoms."""
        config = DNAPhosphateContactConfig(phosphate_atoms=["OP1", "OP2", "P"])
        assert config.phosphate_atoms == ["OP1", "OP2", "P"]

    def test_legacy_phosphate_atom_names_in_default(self):
        """Legacy O1P/O2P aliases are scored by default so old-convention PDBs aren't missed."""
        assert {"O1P", "O2P"} <= set(DNAPhosphateContactConfig().phosphate_atoms)


class TestAnalyzePhosphateContacts:
    """Internal geometry helper on synthetic PDBs."""

    def test_contact_detected(self, tmp_path):
        """An ARG contacting a phosphate oxygen is one contact; weighted sum reports ARG=7."""
        result = _analyze_phosphate_contacts(
            _contact_pdb(tmp_path),
            cutoff=3.5,
            phosphate_atoms={"OP1", "OP2"},
        )
        assert result["n_phosphate_contacts"] == 1
        assert result["n_unique_residues"] == 1
        assert result["weighted_phosphate_score"] == pytest.approx(7.0)
        assert result["contacting_types"] == {"ARG": 1}

    def test_same_resseq_on_two_dna_chains_not_collapsed(self, tmp_path):
        """Phosphates with equal resseq on different DNA chains count as distinct contacts."""
        lines = [
            _atom(1, "NH1", "ARG", "A", 10, (0.0, 0.0, 0.0)),
            _atom(2, "OP1", "DG", "B", 5, (1.0, 0.0, 0.0)),  # chain B, residue 5
            _atom(3, "OP1", "DG", "C", 5, (-1.0, 0.0, 0.0)),  # chain C, residue 5 (same number)
        ]
        result = _analyze_phosphate_contacts(
            _write_pdb(tmp_path, lines),
            cutoff=3.5,
            phosphate_atoms={"OP1", "OP2"},
        )
        # Keyed by (chain, resseq), the two strands' residue-5 phosphates stay distinct.
        assert result["n_phosphate_contacts"] == 2
        # One ARG residue weighted (7) across its two distinct phosphate contacts = 14.
        assert result["weighted_phosphate_score"] == pytest.approx(14.0)
        assert result["contacting_types"] == {"ARG": 1}

    def test_legacy_atom_name_detected(self, tmp_path):
        """A phosphate oxygen named with the legacy O1P convention is still counted."""
        lines = [
            _atom(1, "NH1", "ARG", "A", 10, (0.0, 0.0, 0.0)),
            _atom(2, "O1P", "DG", "B", 5, (1.0, 0.0, 0.0)),
        ]
        result = _analyze_phosphate_contacts(
            _write_pdb(tmp_path, lines),
            cutoff=3.5,
            phosphate_atoms={"OP1", "OP2", "O1P", "O2P"},
        )
        assert result["n_phosphate_contacts"] == 1

    def test_no_contacts(self, tmp_path):
        """Far-apart atoms (no contacts) produce a zeroed contact summary."""
        result = _analyze_phosphate_contacts(
            _no_contact_pdb(tmp_path),
            cutoff=3.5,
            phosphate_atoms={"OP1", "OP2"},
        )
        assert result["n_phosphate_contacts"] == 0
        assert result["weighted_phosphate_score"] == 0.0
        assert result["n_unique_residues"] == 0

    def test_missing_phosphate_atoms_early_return(self, tmp_path):
        """A PDB with no DNA phosphate atoms hits the empty-input early return."""
        pdb = _write_pdb(tmp_path, [_atom(1, "NH1", "ARG", "A", 10, (0.0, 0.0, 0.0))])
        result = _analyze_phosphate_contacts(pdb, cutoff=3.5, phosphate_atoms={"OP1", "OP2"})
        assert result["n_phosphate_contacts"] == 0
        assert "contacting_types" not in result

    def test_cutoff_excludes_distant_atoms(self, tmp_path):
        """A tight cutoff drops contacts that a loose cutoff would count."""
        pdb = _contact_pdb(tmp_path)
        tight = _analyze_phosphate_contacts(pdb, cutoff=0.5, phosphate_atoms={"OP1", "OP2"})
        assert tight["n_phosphate_contacts"] == 0

    def test_base_atoms_excluded(self, tmp_path):
        """Restricting phosphate_atoms to a name not present yields no contacts."""
        result = _analyze_phosphate_contacts(
            _contact_pdb(tmp_path),
            cutoff=3.5,
            phosphate_atoms={"OP2"},  # PDB only has OP1
        )
        assert result["n_phosphate_contacts"] == 0


class TestConstraintScoring:
    """End-to-end scoring with the structure resolver mocked."""

    def _candidate(self):
        protein = Sequence(sequence="MKQ", sequence_type="protein")
        dna = Sequence(sequence="ACGT", sequence_type="dna")
        return (protein, dna)

    def test_empty_input(self):
        """No candidates returns an empty result list."""
        assert dna_phosphate_contact_constraint([], DNAPhosphateContactConfig()) == []

    def test_missing_pdb_gets_max_energy(self):
        """An unresolved structure (empty path) soft-fails to MAX_ENERGY with a reason."""
        config = DNAPhosphateContactConfig(structure_tool="alphafold3")
        with patch(PATCH_TARGET, return_value=[""]):
            results = dna_phosphate_contact_constraint([self._candidate()], config)
        assert len(results) == 1
        assert results[0].score == MAX_ENERGY
        assert results[0].metadata["phosphate_contact_error"] == "structure_unresolved"

    def test_contact_design_scores_well(self, tmp_path):
        """A phosphate-contacting complex scores better than no contacts."""
        config = DNAPhosphateContactConfig(structure_tool="alphafold3")
        good_pdb = _contact_pdb(tmp_path)
        bad_pdb = _no_contact_pdb(tmp_path)
        with patch(PATCH_TARGET, return_value=[good_pdb, bad_pdb]):
            results = dna_phosphate_contact_constraint([self._candidate(), self._candidate()], config)
        assert len(results) == 2
        good, bad = results
        assert 0.0 <= good.score <= 1.0
        # One integer contact against the default target of 2: (2 - 1) / 2 = 0.5.
        assert good.score == pytest.approx(0.5)
        assert bad.score == pytest.approx(1.0)
        assert good.metadata["n_phosphate_contacts"] == 1
        # Residue weighting is reported as auxiliary metadata, not in the score.
        assert good.metadata["weighted_phosphate_score"] == pytest.approx(7.0)
        assert good.metadata["pdb_path"] == good_pdb
        assert bad.metadata["n_phosphate_contacts"] == 0

    def test_scoring_math_matches_penalty(self, tmp_path):
        """Reproduce the exact integer-count penalty mapping for a known geometry."""
        config = DNAPhosphateContactConfig(structure_tool="alphafold3", desired_phosphate_contacts=4)
        pdb = _contact_pdb(tmp_path)
        with patch(PATCH_TARGET, return_value=[pdb]):
            results = dna_phosphate_contact_constraint([self._candidate()], config)
        out = results[0]
        n_contacts = out.metadata["n_phosphate_contacts"]  # 1 unique contact
        expected = max(0.0, (4 - n_contacts) / 4)  # (4 - 1) / 4 = 0.75
        assert out.score == pytest.approx(expected)
        assert out.score == pytest.approx(0.75)
