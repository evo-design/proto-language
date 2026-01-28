"""
store.py

File storage backends for large file content.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
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
            content: File content as string or bytes.
            file_type: Type of the file being stored.

        Returns:
            FileReference to the stored content.
        """
        pass

    @abstractmethod
    def get(self, file_id: str) -> bytes:
        """Retrieve content by file ID.

        Args:
            file_id: Content-addressed ID of the file.

        Returns:
            File content as bytes.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        pass

    @abstractmethod
    def exists(self, file_id: str) -> bool:
        """Check if a file exists.

        Args:
            file_id: Content-addressed ID of the file.

        Returns:
            True if file exists.
        """
        pass

    @staticmethod
    def compute_hash(content: Union[str, bytes]) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Content as string or bytes.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _get_sharded_path(file_id: str) -> str:
        """Get sharded path components for a file ID.

        Uses first 4 characters to create 2 levels of directory sharding
        to avoid too many files in a single directory.

        Args:
            file_id: Content-addressed ID.

        Returns:
            Path like "ab/cd/abcdef1234..."
        """
        return f"{file_id[:2]}/{file_id[2:4]}/{file_id}"


class LocalFileStore(FileStore):
    """Local filesystem-based file store for development.

    Stores files in a sharded directory structure:
        {base_path}/{hash[:2]}/{hash[2:4]}/{hash}
    """

    def __init__(self, base_path: Union[str, Path]):
        """Initialize local file store.

        Args:
            base_path: Base directory for file storage.
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


class GCSFileStore(FileStore):
    """Google Cloud Storage-based file store for production.

    Stores files in a sharded path structure:
        gs://{bucket}/files/{hash[:2]}/{hash[2:4]}/{hash}
    """

    def __init__(self, bucket_name: str):
        """Initialize GCS file store.

        Args:
            bucket_name: Name of the GCS bucket.
        """
        self.bucket_name = bucket_name
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
                credentials = service_account.Credentials.from_service_account_info(creds_info)
                self._client = storage.Client(credentials=credentials)
                logger.info("GCS client initialized from GOOGLE_APPLICATION_CREDENTIALS_JSON")
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
            blob.upload_from_string(content_bytes)
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


# Module-level singleton for the configured file store
_file_store: Optional[FileStore] = None


def get_file_store() -> FileStore:
    """Get the configured file store instance.

    Configuration via environment variables:
        FILE_STORE_TYPE: "local" (default) or "gcs"
        FILE_STORE_PATH: Base path for local storage (default: ./file_store)
        GCS_FILE_BUCKET: GCS bucket name (required if FILE_STORE_TYPE=gcs)

    Returns:
        Configured FileStore instance.

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
