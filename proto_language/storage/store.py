"""
proto_language/storage/store.py

File storage backends for large file content.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

from proto_language.storage.models import FileReference, FileType

logger = logging.getLogger(__name__)


class FileStore(ABC):
    """Abstract base class for file storage backends."""

    @abstractmethod
    def put(self, content: Union[str, bytes], file_type: FileType) -> FileReference:
        """Store content and return a reference.

        Args:
            content (str | bytes): File content as string or bytes.
            file_type (FileType): Type of the file being stored.

        Returns:
            FileReference: FileReference to the stored content.
        """
        pass

    @abstractmethod
    def get(self, file_id: str) -> bytes:
        """Retrieve content by file ID.

        Args:
            file_id (str): Content-addressed ID of the file.

        Returns:
            bytes: File content as bytes.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        pass

    @abstractmethod
    def exists(self, file_id: str) -> bool:
        """Check if a file exists.

        Args:
            file_id (str): Content-addressed ID of the file.

        Returns:
            bool: True if file exists.
        """
        pass

    # Whether get_url() returns a redirect URL (True) or a local path (False).
    # Used by the file-serving endpoint to decide between redirect and stream.
    serves_redirect: bool = False

    @abstractmethod
    def get_url(self, file_id: str) -> str:
        """Get an accessible URL or path for the file.

        Args:
            file_id (str): Content-addressed ID of the file.

        Returns:
            str: URL (for remote stores) or local file path (for local stores).

        Raises:
            FileNotFoundError: If file does not exist.
        """
        pass

    def get_content_type(self, file_id: str) -> str:
        """Get the MIME content type for a stored file.

        Args:
            file_id (str): Content-addressed ID of the file.

        Returns:
            str: MIME type string, defaults to ``application/octet-stream``.
        """
        return "application/octet-stream"

    @staticmethod
    def compute_hash(content: Union[str, bytes]) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content (str | bytes): Content as string or bytes.

        Returns:
            str: Hex-encoded SHA-256 hash.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def get_batch(
        self, file_ids: set[str], *, max_workers: int = 10
    ) -> dict[str, bytes]:
        """Retrieve multiple files concurrently.

        Dispatches to :meth:`get` in a thread pool. Subclasses may override
        with native batch APIs for better performance.

        Args:
            file_ids (set[str]): Set of content-addressed file IDs to fetch.
            max_workers (int): Maximum concurrent fetches (default 10).

        Returns:
            dict[str, bytes]: Dict mapping each file ID to its content bytes.

        Raises:
            FileNotFoundError: If any file does not exist.
        """
        if not file_ids:
            return {}
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(max_workers, len(file_ids))) as pool:
            futures = {fid: pool.submit(self.get, fid) for fid in file_ids}
            return {fid: fut.result() for fid, fut in futures.items()}

    @staticmethod
    def _get_sharded_path(file_id: str) -> str:
        """Get sharded path components for a file ID.

        Uses first 4 characters to create 2 levels of directory sharding
        to avoid too many files in a single directory.

        Args:
            file_id (str): Content-addressed ID.

        Returns:
            str: Path like "ab/cd/abcdef1234..."
        """
        return f"{file_id[:2]}/{file_id[2:4]}/{file_id}"


class LocalFileStore(FileStore):
    """Local filesystem-based file store for development.

    Stores files in a sharded directory structure:
        {base_path}/{hash[:2]}/{hash[2:4]}/{hash}

    Attributes:
        serves_redirect (bool): Whether this store serves redirect URLs instead of direct content.
    """

    def __init__(self, base_path: Union[str, Path]):
        """Initialize local file store.

        Args:
            base_path (str | Path): Base directory for file storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalFileStore initialized at {self.base_path}")

    def _get_file_path(self, file_id: str) -> Path:
        """Get the full file path for a given ID."""
        return self.base_path / self._get_sharded_path(file_id)

    def put(self, content: Union[str, bytes], file_type: FileType) -> FileReference:
        """Store content to local filesystem."""
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        file_id = self.compute_hash(content_bytes)
        file_path = self._get_file_path(file_id)

        # Content-addressed: if file exists, it's already the same content
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content_bytes)
            # Write file type sidecar for content-type lookups
            file_path.with_suffix(".type").write_text(file_type.value)
            logger.debug(f"Stored file {file_id} ({len(content_bytes)} bytes)")
        else:
            logger.debug(f"File {file_id} already exists, skipping write")

        return FileReference(
            id=file_id,
            file_type=file_type,
            size=len(content_bytes),
            url=str(file_path),
        )

    def get(self, file_id: str) -> bytes:
        """Retrieve content from local filesystem."""
        file_path = self._get_file_path(file_id)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_id}")
        return file_path.read_bytes()

    def exists(self, file_id: str) -> bool:
        """Check if file exists in local filesystem."""
        return self._get_file_path(file_id).exists()

    def get_url(self, file_id: str) -> str:
        """Return the local file path as a string."""
        file_path = self._get_file_path(file_id)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_id}")
        return str(file_path)

    def get_content_type(self, file_id: str) -> str:
        """Read content type from sidecar file."""
        type_path = self._get_file_path(file_id).with_suffix(".type")
        if type_path.exists():
            try:
                return FileType(type_path.read_text().strip()).content_type
            except ValueError:
                pass
        return "application/octet-stream"


class GCSFileStore(FileStore):
    """Google Cloud Storage-based file store for production.

    Stores files in a sharded path structure:
        gs://{bucket}/files/{hash[:2]}/{hash[2:4]}/{hash}

    Attributes:
        serves_redirect (bool): Whether this store serves redirect URLs instead of direct content.
    """

    serves_redirect = True

    def __init__(self, bucket_name: str, signed_url_expiration_minutes: int = 60):
        """Initialize GCS file store.

        Args:
            bucket_name (str): Name of the GCS bucket.
            signed_url_expiration_minutes (int): Validity period for signed URLs
                returned by :meth:`get_url` (default 60 minutes).
        """
        self.bucket_name = bucket_name
        self.signed_url_expiration_minutes = signed_url_expiration_minutes
        self._client = None
        self._bucket = None
        logger.info(f"GCSFileStore initialized for bucket {bucket_name}")

    @property
    def client(self):
        """Lazy-load GCS client.

        Supports credentials via:
        - GOOGLE_APPLICATION_CREDENTIALS_JSON: JSON string (for containers)
        - GOOGLE_APPLICATION_CREDENTIALS: File path (standard approach)
        - Default credentials (GCE, Workload Identity, etc.)
        """
        if self._client is None:
            import json

            from google.cloud import storage

            # Check for JSON credentials string (avoids temp file writing)
            creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if creds_json:
                from google.oauth2 import service_account

                creds_info = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(
                    creds_info
                )
                self._client = storage.Client(credentials=credentials)
                logger.info(
                    "GCS client initialized from GOOGLE_APPLICATION_CREDENTIALS_JSON"
                )
            else:
                # Fall back to default credentials
                self._client = storage.Client()
                logger.info("GCS client initialized with default credentials")

        return self._client

    @property
    def bucket(self):
        """Lazy-load GCS bucket."""
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def _get_blob_path(self, file_id: str) -> str:
        """Get the blob path for a given file ID."""
        return f"files/{self._get_sharded_path(file_id)}"

    def put(self, content: Union[str, bytes], file_type: FileType) -> FileReference:
        """Store content to GCS."""
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        file_id = self.compute_hash(content_bytes)
        blob_path = self._get_blob_path(file_id)
        blob = self.bucket.blob(blob_path)

        # Content-addressed: if blob exists, it's already the same content
        if not blob.exists():
            blob.upload_from_string(content_bytes, content_type=file_type.content_type)
            logger.debug(f"Uploaded file {file_id} to GCS ({len(content_bytes)} bytes)")
        else:
            logger.debug(f"File {file_id} already exists in GCS, skipping upload")

        return FileReference(
            id=file_id,
            file_type=file_type,
            size=len(content_bytes),
            url=f"gs://{self.bucket_name}/{blob_path}",
        )

    def get(self, file_id: str) -> bytes:
        """Retrieve content from GCS."""
        blob_path = self._get_blob_path(file_id)
        blob = self.bucket.blob(blob_path)

        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {file_id}")

        return blob.download_as_bytes()

    def exists(self, file_id: str) -> bool:
        """Check if file exists in GCS."""
        blob_path = self._get_blob_path(file_id)
        return self.bucket.blob(blob_path).exists()

    def get_url(self, file_id: str) -> str:
        """Generate a time-limited signed URL for the file.

        Args:
            file_id (str): Unique identifier for the stored file.

        Returns:
            str: Signed HTTPS URL for direct browser access.

        Raises:
            FileNotFoundError: If file does not exist in GCS.
        """
        blob_path = self._get_blob_path(file_id)
        blob = self.bucket.blob(blob_path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {file_id}")
        return blob.generate_signed_url(
            expiration=timedelta(minutes=self.signed_url_expiration_minutes)
        )


# Module-level singleton for the configured file store
_file_store: Optional[FileStore] = None


def get_file_store() -> FileStore:
    """Get the configured file store instance.

    Configuration via environment variables:
        FILE_STORE_TYPE: "local" (default) or "gcs"
        FILE_STORE_PATH: Base path for local storage (default: ./file_store)
        GCS_FILE_BUCKET: GCS bucket name (required if FILE_STORE_TYPE=gcs)

    Returns:
        FileStore: Configured FileStore instance.

    Raises:
        ValueError: If GCS is configured but GCS_FILE_BUCKET is not set.
    """
    global _file_store

    if _file_store is None:
        store_type = os.environ.get("FILE_STORE_TYPE", "local").lower()

        if store_type == "gcs":
            bucket_name = os.environ.get("GCS_FILE_BUCKET")
            if not bucket_name:
                raise ValueError(
                    "GCS_FILE_BUCKET environment variable is required when "
                    "FILE_STORE_TYPE=gcs"
                )
            _file_store = GCSFileStore(bucket_name)
        else:
            # Default to local storage
            base_path = os.environ.get("FILE_STORE_PATH", "./file_store")
            _file_store = LocalFileStore(base_path)

    return _file_store


def reset_file_store() -> None:
    """Reset the file store singleton (useful for testing)."""
    global _file_store
    _file_store = None
