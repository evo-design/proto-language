"""
Storage module for large file content and GCS utilities.

Provides:
- File storage backends (local and GCS) for externalizing large file content
  (PDB, CIF, HMM, etc.) from sequence metadata.
- GCS file resolution utilities for downloading and caching remote files.

Configuration via environment variables:
    FILE_STORE_TYPE: "local" (default) or "gcs"
    FILE_STORE_PATH: Base path for local storage (default: ./file_store)
    GCS_FILE_BUCKET: GCS bucket name (required if FILE_STORE_TYPE=gcs)

Example usage in constraints:
    >>> from proto_language.storage import store_file, FileType
    >>> seq._metadata["pdb_output"] = store_file(structure.structure_pdb, FileType.PDB)

Example usage to retrieve content:
    >>> from proto_language.storage import get_file_content
    >>> pdb_content = get_file_content(seq._metadata["pdb_output"])

Example usage for GCS file resolution:
    >>> from proto_language.storage import resolve_paths
    >>> resolved = resolve_paths({"db": "gcs://bucket/mmseqs_db"})
"""

from proto_language.storage.helpers import (
    get_file_content,
    get_file_content_bytes,
    is_file_reference,
    store_file,
)
from proto_language.storage.models import FILE_REF_MARKER, FileReference, FileType
from proto_language.storage.resolver import (
    VOLUME_PATH,
    download_gcs_file,
    get_cache_path,
    resolve_file,
    resolve_paths,
)
from proto_language.storage.store import (
    FileStore,
    GCSFileStore,
    LocalFileStore,
    get_file_store,
    reset_file_store,
)

__all__ = [
    # Models
    "FileType",
    "FileReference",
    "FILE_REF_MARKER",
    # Stores
    "FileStore",
    "LocalFileStore",
    "GCSFileStore",
    "get_file_store",
    "reset_file_store",
    # Helpers
    "store_file",
    "get_file_content",
    "get_file_content_bytes",
    "is_file_reference",
    # Resolver
    "VOLUME_PATH",
    "download_gcs_file",
    "get_cache_path",
    "resolve_file",
    "resolve_paths",
]
