"""mock_structure.py."""

from pathlib import Path

from proto_tools import BFactorType, Structure, load_structure_file

MOCK_PDB = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")
MOCK_CIF = load_structure_file(Path(__file__).parent.parent / "dummy_data" / "renin.cif")


class MockStructure(Structure):
    """Mock version of Structure that bypasses file loading for testing."""

    def __init__(
        self,
        structure_content: str | None = None,
        structure_format: str = "pdb",
        b_factor_type: BFactorType = BFactorType.UNSPECIFIED,
        metrics: dict[str, float] | None = None,
        source: str = "mock",
    ) -> None:
        """Mocked Structure class for testing. Bypasses validation via model_construct."""
        structure = (
            structure_content
            if structure_content is not None
            else (MOCK_PDB if structure_format == "pdb" else MOCK_CIF)
        )
        constructed = Structure.model_construct(
            structure=structure,
            structure_format=structure_format,
            b_factor_type=b_factor_type,
            source=source if "mock" in source else f"mock.{source}",
            metrics=metrics if metrics is not None else {},
        )
        # Copy all Pydantic internal state from the constructed instance
        self.__dict__.update(constructed.__dict__)
        self.__pydantic_fields_set__ = constructed.__pydantic_fields_set__
        self._gemmi_struct = None
