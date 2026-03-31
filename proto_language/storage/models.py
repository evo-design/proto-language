"""
proto_language/storage/models.py

Data models for file storage references.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


class FileType(str, Enum):
    """Supported file types for storage."""

    PDB = "pdb"
    CIF = "cif"
    HMM = "hmm"
    FASTA = "fasta"
    CSV = "csv"
    JSON = "json"
    BINARY = "binary"

    @property
    def content_type(self) -> str:
        """HTTP content type for this file type."""
        return _CONTENT_TYPES.get(self, "application/octet-stream")


_CONTENT_TYPES: dict[FileType, str] = {
    FileType.PDB: "chemical/x-pdb",
    FileType.CIF: "chemical/x-mmcif",
    FileType.CSV: "text/csv",
    FileType.JSON: "application/json",
    FileType.FASTA: "text/plain",
    FileType.HMM: "application/octet-stream",
    FileType.BINARY: "application/octet-stream",
}


# Marker key used to identify file references in dictionaries
FILE_REF_MARKER = "__file_ref__"


class FileReference(BaseModel):
    """Reference to a file stored in the file store.

    Attributes:
        id (str): Content-addressed SHA-256 hash of the file content.
        file_type (FileType): Type of the file (pdb, cif, hmm, etc.).
        size (int): Size of the file in bytes.
        url (str): URL or path to the file (gs:// for GCS, local path for development).
    """

    id: str = Field(description="Content-addressed SHA-256 hash ID")
    file_type: FileType = Field(description="Type of the stored file")
    size: int = Field(description="Size of the file in bytes", ge=0)
    url: str = Field(description="URL or path to the stored file")

    @classmethod
    def is_file_ref(cls, data: Any) -> bool:
        """Check if a dictionary represents a file reference.

        Args:
            data (Any): Value to check.

        Returns:
            bool: True if data is a dict with the file reference marker.
        """
        if not isinstance(data, dict):
            return False
        return data.get(FILE_REF_MARKER, False) is True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary with the file reference marker.

        Returns:
            dict[str, Any]: Dictionary representation suitable for storage in metadata.
        """
        return {
            FILE_REF_MARKER: True,
            "id": self.id,
            "file_type": self.file_type.value,
            "size": self.size,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileReference:
        """Create a FileReference from a dictionary.

        Args:
            data (dict[str, Any]): Dictionary with file reference data.

        Returns:
            FileReference: FileReference instance.

        Raises:
            ValueError: If data is not a valid file reference dict.
        """
        if not cls.is_file_ref(data):
            raise ValueError("Data is not a valid file reference dictionary")

        return cls(
            id=data["id"],
            file_type=FileType(data["file_type"]),
            size=data["size"],
            url=data["url"],
        )
