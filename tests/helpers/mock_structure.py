"""mock_structure.py."""

from pathlib import Path
from typing import Any

from proto_tools import BFactorType, Structure, load_structure_file

MOCK_PDB = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
MOCK_CIF = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin.cif")


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
