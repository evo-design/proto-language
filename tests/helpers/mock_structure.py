"""
mock_structure.py
"""
"""
mock_structure.py
"""

from typing import Optional, Dict
from pathlib import Path
from proto_language.bio_tools.entities.structures import Structure, BFactorType, load_structure_file


MOCK_PDB = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
MOCK_CIF = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin.cif")


class MockStructure(Structure):
    """Mock version of Structure that bypasses file loading for testing."""

    def __init__(
        self,
        structure_content: Optional[str] = None,
        structure_format: str = "pdb",
        b_factor_type: BFactorType = BFactorType.UNSPECIFIED,
        metrics: Optional[Dict[str, float]] = None,
        source: str = "mock",
    ) -> None:
        """
        Mocked Structure class for testing. Bypasses __init__ validation
        and detection of structure format.
        """
        # Save the structure content and format directly
        self.structure_format = structure_format

        if structure_content is not None:
            self.structure = structure_content
        else:
            self.structure = MOCK_PDB if structure_format == "pdb" else MOCK_CIF

        # Save other attributes
        self.b_factor_type = b_factor_type
        self.source = source if "mock" in source else f"mock.{source}"

        # Set up metrics
        self.metrics = metrics if metrics is not None else {}

        # Set up placeholder for lazy loading of gemmi structure object
        self._gemmi_struct = None