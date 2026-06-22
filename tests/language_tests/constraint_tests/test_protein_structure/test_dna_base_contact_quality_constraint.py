"""Tests for the DNA base contact quality constraint.

Exercises config validation and the internal PDB geometry / scoring helpers on
small synthetic PDB inputs, without GPU or structure prediction (the structure
resolver is mocked so no AF3/Boltz2 call is made).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from proto_language.constraint.protein_structure.dna_base_contact_quality_constraint import (
    DNABaseContactQualityConfig,
    _analyze_base_contacts,
    dna_base_contact_quality_constraint,
)
from proto_language.core import Sequence
from proto_language.utils import MAX_ENERGY

PATCH_TARGET = "proto_language.constraint.protein_structure.dna_base_contact_quality_constraint.resolve_structure_paths"


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


def _bidentate_pdb(tmp_path: Path) -> str:
    """PDB where one GLN reads two H-bond atoms of a single guanine (bidentate).

    GLN NE2 and OE1 sit within the default 3.5 A cutoff of guanine O6 and N7,
    which both polar SC atoms and base H-bond atoms, so this yields one
    bidentate contact from a specific (diverse) residue.
    """
    lines = [
        # GLN polar sidechain atoms near the base
        _atom(1, "OE1", "GLN", "A", 10, (0.0, 0.0, 0.0)),
        _atom(2, "NE2", "GLN", "A", 10, (0.0, 0.0, 2.0)),
        # Guanine base H-bond atoms (DG) on chain B
        _atom(3, "O6", "DG", "B", 5, (1.0, 0.0, 0.0)),
        _atom(4, "N7", "DG", "B", 5, (1.0, 0.0, 2.0)),
        # A backbone phosphate atom that must be ignored
        _atom(5, "P", "DG", "B", 5, (1.0, 0.0, 50.0)),
    ]
    return _write_pdb(tmp_path, lines)


def _no_contact_pdb(tmp_path: Path) -> str:
    """PDB where protein and DNA atoms are far apart (no contacts)."""
    lines = [
        _atom(1, "NE2", "GLN", "A", 10, (0.0, 0.0, 0.0)),
        _atom(2, "O6", "DG", "B", 5, (50.0, 50.0, 50.0)),
    ]
    return _write_pdb(tmp_path, lines)


class TestConfigValidation:
    """Config field defaults and validation bounds."""

    def test_defaults(self):
        """Default config carries the ported scoring targets and a DNA-capable predictor."""
        config = DNABaseContactQualityConfig()
        assert config.contact_cutoff == 3.5
        assert config.desired_bidentate == 2
        assert config.desired_base_contacts == 8
        assert config.desired_unique_residues == 4
        assert config.diversity_bonus_weight == 0.3
        # Default must be DNA-capable, not the inherited "esmfold" (cannot fold DNA).
        assert config.structure_tool == "alphafold3"

    def test_contact_cutoff_must_be_positive(self):
        """contact_cutoff has a gt=0 bound."""
        with pytest.raises(ValueError):
            DNABaseContactQualityConfig(contact_cutoff=0.0)

    def test_negative_desired_rejected(self):
        """Desired counts have ge=0 bounds."""
        with pytest.raises(ValueError):
            DNABaseContactQualityConfig(desired_bidentate=-1)

    def test_diversity_weight_bounded(self):
        """diversity_bonus_weight is bounded to [0, 1]."""
        with pytest.raises(ValueError):
            DNABaseContactQualityConfig(diversity_bonus_weight=1.5)


class TestAnalyzeBaseContacts:
    """Internal geometry helper on synthetic PDBs."""

    def test_bidentate_detected(self, tmp_path):
        """A GLN reading two H-bond atoms of one base is one bidentate contact."""
        result = _analyze_base_contacts(_bidentate_pdb(tmp_path), cutoff=3.5)
        assert result["n_bidentate"] == 1
        assert result["n_base_contacts"] >= 2
        assert result["n_unique_residues"] == 1
        # GLN is a specific (diverse) readout residue.
        assert result["n_specific_residues"] == 1
        assert result["diversity_score"] == 1.0
        assert result["pct_arg"] == 0.0

    def test_no_contacts(self, tmp_path):
        """Far-apart atoms (no contacts) produce a zeroed contact summary."""
        result = _analyze_base_contacts(_no_contact_pdb(tmp_path), cutoff=3.5)
        assert result["n_base_contacts"] == 0
        assert result["n_bidentate"] == 0
        assert result["n_unique_residues"] == 0
        assert result["diversity_score"] == 0.0

    def test_missing_atoms_early_return(self, tmp_path):
        """A PDB with protein atoms but no DNA is a real zero-contact structure, not a parse failure."""
        # Only a protein polar atom, no DNA: triggers the pct_arg=100.0 sentinel path.
        pdb = _write_pdb(tmp_path, [_atom(1, "NE2", "GLN", "A", 10, (0.0, 0.0, 0.0))])
        result = _analyze_base_contacts(pdb, cutoff=3.5)
        assert result["n_base_contacts"] == 0
        assert result["pct_arg"] == 100.0
        assert result["parse_failed"] is False
        assert "contacting_types" not in result

    def test_empty_pdb_flags_parse_failure(self, tmp_path):
        """A PDB with no protein polar atoms AND no DNA base atoms is a parse failure."""
        pdb = _write_pdb(tmp_path, [])
        result = _analyze_base_contacts(pdb, cutoff=3.5)
        assert result["parse_failed"] is True

    def test_cutoff_excludes_distant_atoms(self, tmp_path):
        """A tight cutoff drops contacts that a loose cutoff would count."""
        pdb = _bidentate_pdb(tmp_path)
        tight = _analyze_base_contacts(pdb, cutoff=0.5)
        assert tight["n_bidentate"] == 0
        assert tight["n_base_contacts"] == 0


class TestConstraintScoring:
    """End-to-end scoring with the structure resolver mocked."""

    def _candidate(self):
        protein = Sequence(sequence="MKQ", sequence_type="protein")
        dna = Sequence(sequence="ACGT", sequence_type="dna")
        return (protein, dna)

    def test_empty_input(self):
        """No candidates returns an empty result list."""
        assert dna_base_contact_quality_constraint([], DNABaseContactQualityConfig()) == []

    def test_missing_pdb_gets_max_energy(self):
        """An unresolved structure (empty path) scores MAX_ENERGY."""
        config = DNABaseContactQualityConfig(structure_tool="alphafold3")
        with patch(PATCH_TARGET, return_value=[""]):
            results = dna_base_contact_quality_constraint([self._candidate()], config)
        assert len(results) == 1
        assert results[0].score == MAX_ENERGY

    def test_empty_pdb_soft_fails_to_max_energy(self, tmp_path, caplog):
        """An empty/unparseable PDB scores MAX_ENERGY with a named reason, not a misleading deficit."""
        config = DNABaseContactQualityConfig(structure_tool="alphafold3")
        empty_pdb = _write_pdb(tmp_path, [], name="empty")
        with patch(PATCH_TARGET, return_value=[empty_pdb]):
            with caplog.at_level(
                "WARNING",
                logger="proto_language.constraint.protein_structure.dna_base_contact_quality_constraint",
            ):
                results = dna_base_contact_quality_constraint([self._candidate()], config)
        assert results[0].score == MAX_ENERGY
        assert "empty or unparseable PDB" in results[0].metadata["dna_base_contact_quality_error"]
        assert "no atoms parsed" in caplog.text

    def test_bidentate_design_scores_well(self, tmp_path):
        """A bidentate, diverse-readout complex scores better than no contacts."""
        config = DNABaseContactQualityConfig(structure_tool="alphafold3")
        good_pdb = _bidentate_pdb(tmp_path)
        bad_pdb = _no_contact_pdb(tmp_path)
        with patch(PATCH_TARGET, return_value=[good_pdb, bad_pdb]):
            results = dna_base_contact_quality_constraint([self._candidate(), self._candidate()], config)
        assert len(results) == 2
        good, bad = results
        assert 0.0 <= good.score <= 1.0
        assert good.score < bad.score
        # Metadata round-trips the geometry summary.
        assert good.metadata["n_bidentate"] == 1
        assert good.metadata["pdb_path"] == good_pdb
        assert bad.metadata["n_base_contacts"] == 0

    def test_scoring_math_matches_components(self, tmp_path):
        """Reproduce the exact weighted combination for a known geometry."""
        config = DNABaseContactQualityConfig(
            structure_tool="alphafold3",
            desired_bidentate=2,
            desired_base_contacts=8,
            desired_unique_residues=4,
            diversity_bonus_weight=0.3,
        )
        pdb = _bidentate_pdb(tmp_path)
        with patch(PATCH_TARGET, return_value=[pdb]):
            results = dna_base_contact_quality_constraint([self._candidate()], config)
        out = results[0]
        n_contacts = out.metadata["n_base_contacts"]
        bidentate_deficit = (2 - 1) / 2  # 0.5
        contacts_deficit = max(0.0, (8 - n_contacts) / 8)
        residues_deficit = (4 - 1) / 4  # 0.75
        diversity_deficit = 1.0 - 1.0  # diversity_score == 1.0 -> 0.0
        expected = (
            0.35 * bidentate_deficit + 0.30 * contacts_deficit + 0.15 * residues_deficit + 0.3 * diversity_deficit
        )
        assert out.score == pytest.approx(expected)
