"""
proto_language/storage/helpers.py

Convenience functions for storing and retrieving file content.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from proto_language.storage.models import FileReference, FileType
from proto_language.storage.store import get_file_store


def store_file(
    content: Union[str, bytes],
    file_type: FileType,
) -> Dict[str, Any]:
    """Store content to file store and return a reference dictionary.

    This is the primary function for constraints to use when storing
    large file content (PDB, CIF, HMM, etc.) to avoid bloating metadata.

    Args:
        content (str | bytes): File content as string or bytes.
        file_type (FileType): Type of the file being stored.

    Returns:
        dict[str, Any]: Dictionary representation of the FileReference, suitable for
            storing in sequence metadata.

    Example:
        >>> from proto_language.storage import store_file, FileType
        >>> # In a constraint:
        >>> seq._metadata["pdb_output"] = store_file(
        ...     structure.structure_pdb,
        ...     FileType.PDB
        ... )
    """
    store = get_file_store()
    ref = store.put(content, file_type)
    return ref.to_dict()


def get_file_content(ref_or_content: Union[Dict[str, Any], str]) -> str:
    """Get content from a file reference or return as-is if already content.

    This function handles both inline strings and file references transparently,
    making it easy to work with metadata that may contain either format.

    Args:
        ref_or_content (dict[str, Any] | str): Either a file reference dictionary or an inline string.

    Returns:
        str: The file content as a string.

    Raises:
        ValueError: If the input is neither a valid file reference nor a string.

    Example:
        >>> from proto_language.storage import get_file_content
        >>> # Works with both inline content and file references:
        >>> pdb_content = get_file_content(seq._metadata["pdb_output"])
    """
    # If it's already a string, return it directly
    if isinstance(ref_or_content, str):
        return ref_or_content

    # Check if it's a file reference dictionary
    if FileReference.is_file_ref(ref_or_content):
        store = get_file_store()
        file_id = ref_or_content["id"]
        return store.get(file_id).decode("utf-8")

    raise ValueError(
        f"Expected a string or file reference dictionary, got {type(ref_or_content)}"
    )


def get_file_content_bytes(ref_or_content: Union[Dict[str, Any], bytes]) -> bytes:
    """Get content as bytes from a file reference or return as-is if already bytes.

    Similar to get_file_content but for binary content.

    Args:
        ref_or_content (dict[str, Any] | bytes): Either a file reference dictionary or inline bytes.

    Returns:
        bytes: The file content as bytes.

    Raises:
        ValueError: If the input is neither a valid file reference nor bytes.
    """
    # If it's already bytes, return directly
    if isinstance(ref_or_content, bytes):
        return ref_or_content

    # Check if it's a file reference dictionary
    if FileReference.is_file_ref(ref_or_content):
        store = get_file_store()
        file_id = ref_or_content["id"]
        return store.get(file_id)

    raise ValueError(
        f"Expected bytes or file reference dictionary, got {type(ref_or_content)}"
    )


def is_file_reference(value: Any) -> bool:
    """Check if a value is a file reference dictionary.

    Args:
        value (Any): Value to check.

    Returns:
        bool: True if value is a file reference dictionary.
    """
    return FileReference.is_file_ref(value)
