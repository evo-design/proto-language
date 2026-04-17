"""mock_structure.py."""

from pathlib import Path
from typing import Any

from proto_tools import BFactorType, Structure, load_structure_file

MOCK_PDB = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
MOCK_CIF = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin.cif")

# Target+binder complex from proto-tools (submodule) used by AF2 binder-gradient tests.
# Kept as a Path so callers can choose between .read_text() and load_structure_file().
PDL1_PDB: Path = Path(__file__).parent.parent.parent / "proto-tools" / "tests" / "dummy_data" / "pdl1.pdb"


class MockStructure(Structure):
    """Mock version of Structure for testing with custom metrics."""

    def __init__(
        self,
        structure_content: str | None = None,
        structure_format: str = "pdb",
        b_factor_type: BFactorType = BFactorType.UNSPECIFIED,
        metrics: dict[str, Any] | None = None,
        source: str = "mock",
    ) -> None:
        """Construct a Structure with optional mock metrics."""
        structure = (
            structure_content
            if structure_content is not None
            else (MOCK_PDB if structure_format == "pdb" else MOCK_CIF)
        )
        super().__init__(
            structure=structure,
            structure_format=structure_format,
            b_factor_type=b_factor_type,
            source=source if "mock" in source else f"mock.{source}",
            metrics=metrics or {},
        )

    @classmethod
    def with_plddt(
        cls,
        per_residue: list[float],
        b_factor_type: BFactorType = BFactorType.NORMALIZED_PLDDT,
        chain: str = "A",
        source: str = "mock",
    ) -> "MockStructure":
        """Build a mock whose B-factor column encodes per-residue pLDDT.

        Emits a minimal single-chain PDB with one CA atom per residue; each atom's
        B-factor is the supplied pLDDT value. Exercises ``Structure.per_residue_plddt``
        in tests without running a real structure predictor.

        Args:
            per_residue (list[float]): Per-residue pLDDT values. Scale should match
                ``b_factor_type``: 0-1 for ``NORMALIZED_PLDDT``, 0-100 for ``PLDDT``.
            b_factor_type (BFactorType): B-factor column semantics. Defaults to
                ``NORMALIZED_PLDDT`` (0-1 scale, matches ESMFold / AlphaFold2 outputs).
            chain (str): Chain ID for all residues. Defaults to ``"A"``.
            source (str): Source identifier; prefixed with ``"mock."`` if not already present.

        Returns:
            MockStructure: A pLDDT-bearing structure whose ``per_residue_plddt`` property
                returns values matching the input (normalized to 0-1).
        """
        lines: list[str] = []
        for i, plddt in enumerate(per_residue, start=1):
            x = float((i - 1) * 3.8)
            # PDB ATOM record (cols 1-6 record, 7-11 serial, 13-16 atom, 18-20 resname,
            # 22 chain, 23-26 resseq, 31-38/39-46/47-54 xyz, 55-60 occ, 61-66 B-factor, 77-78 element).
            lines.append(
                f"ATOM  {i:5d}  CA  ALA {chain}{i:4d}    {x:8.3f}   0.000   0.000  1.00{plddt:6.2f}           C  "
            )
        lines.append("END")
        return cls(
            structure_content="\n".join(lines),
            structure_format="pdb",
            b_factor_type=b_factor_type,
            source=source,
        )
