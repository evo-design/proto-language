"""
Tests for the storage module.
"""

import os
from pathlib import Path

import pytest

from proto_language.storage import (
    FILE_REF_MARKER,
    FileReference,
    FileType,
    LocalFileStore,
    get_file_content,
    get_file_store,
    is_file_reference,
    reset_file_store,
    store_file,
)


@pytest.fixture
def temp_file_store(tmp_path):
    """Create a temporary file store for testing."""
    # Reset the global file store before each test
    reset_file_store()
    # Set environment to use local storage in temp directory
    os.environ["FILE_STORE_TYPE"] = "local"
    os.environ["FILE_STORE_PATH"] = str(tmp_path / "file_store")
    yield tmp_path / "file_store"
    # Clean up
    reset_file_store()
    os.environ.pop("FILE_STORE_TYPE", None)
    os.environ.pop("FILE_STORE_PATH", None)


class TestFileReference:
    """Tests for FileReference model."""

    def test_is_file_ref_with_valid_ref(self):
        """Test is_file_ref returns True for valid file reference dict."""
        ref_dict = {
            FILE_REF_MARKER: True,
            "id": "abc123",
            "file_type": "pdb",
            "size": 100,
            "url": "/path/to/file",
        }
        assert FileReference.is_file_ref(ref_dict) is True

    def test_is_file_ref_with_missing_marker(self):
        """Test is_file_ref returns False when marker is missing."""
        ref_dict = {
            "id": "abc123",
            "file_type": "pdb",
            "size": 100,
            "url": "/path/to/file",
        }
        assert FileReference.is_file_ref(ref_dict) is False

    def test_is_file_ref_with_false_marker(self):
        """Test is_file_ref returns False when marker is False."""
        ref_dict = {
            FILE_REF_MARKER: False,
            "id": "abc123",
        }
        assert FileReference.is_file_ref(ref_dict) is False

    def test_is_file_ref_with_non_dict(self):
        """Test is_file_ref returns False for non-dict values."""
        assert FileReference.is_file_ref("not a dict") is False
        assert FileReference.is_file_ref(123) is False
        assert FileReference.is_file_ref(None) is False

    def test_to_dict(self):
        """Test to_dict creates proper dictionary with marker."""
        ref = FileReference(
            id="abc123",
            file_type=FileType.PDB,
            size=100,
            url="/path/to/file",
        )
        ref_dict = ref.to_dict()

        assert ref_dict[FILE_REF_MARKER] is True
        assert ref_dict["id"] == "abc123"
        assert ref_dict["file_type"] == "pdb"
        assert ref_dict["size"] == 100
        assert ref_dict["url"] == "/path/to/file"

    def test_from_dict(self):
        """Test from_dict creates FileReference from valid dict."""
        ref_dict = {
            FILE_REF_MARKER: True,
            "id": "abc123",
            "file_type": "pdb",
            "size": 100,
            "url": "/path/to/file",
        }
        ref = FileReference.from_dict(ref_dict)

        assert ref.id == "abc123"
        assert ref.file_type == FileType.PDB
        assert ref.size == 100
        assert ref.url == "/path/to/file"

    def test_from_dict_invalid(self):
        """Test from_dict raises error for invalid dict."""
        invalid_dict = {"id": "abc123"}  # Missing marker
        with pytest.raises(ValueError):
            FileReference.from_dict(invalid_dict)


class TestLocalFileStore:
    """Tests for LocalFileStore."""

    def test_put_creates_file(self, tmp_path):
        """Test put creates file in sharded directory."""
        store = LocalFileStore(tmp_path / "store")
        content = "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0"

        ref = store.put(content, FileType.PDB)

        assert ref.id is not None
        assert ref.file_type == FileType.PDB
        assert ref.size == len(content.encode("utf-8"))
        assert Path(ref.url).exists()

    def test_put_bytes_content(self, tmp_path):
        """Test put works with bytes content."""
        store = LocalFileStore(tmp_path / "store")
        content = b"\x00\x01\x02\x03"

        ref = store.put(content, FileType.BINARY)

        assert ref.size == 4
        assert store.get(ref.id) == content

    def test_put_deduplicates_content(self, tmp_path):
        """Test same content returns same ID (content-addressed)."""
        store = LocalFileStore(tmp_path / "store")
        content = "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0"

        ref1 = store.put(content, FileType.PDB)
        ref2 = store.put(content, FileType.PDB)

        assert ref1.id == ref2.id

    def test_get_retrieves_content(self, tmp_path):
        """Test get retrieves stored content."""
        store = LocalFileStore(tmp_path / "store")
        content = "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0"

        ref = store.put(content, FileType.PDB)
        retrieved = store.get(ref.id)

        assert retrieved.decode("utf-8") == content

    def test_get_nonexistent_raises(self, tmp_path):
        """Test get raises FileNotFoundError for missing file."""
        store = LocalFileStore(tmp_path / "store")

        with pytest.raises(FileNotFoundError):
            store.get("nonexistent_id")

    def test_exists(self, tmp_path):
        """Test exists returns correct values."""
        store = LocalFileStore(tmp_path / "store")
        content = "test content"

        ref = store.put(content, FileType.PDB)

        assert store.exists(ref.id) is True
        assert store.exists("nonexistent_id") is False

    def test_sharded_directory_structure(self, tmp_path):
        """Test files are stored in sharded directories."""
        store = LocalFileStore(tmp_path / "store")
        content = "test content"

        ref = store.put(content, FileType.PDB)
        file_path = Path(ref.url)

        # Should be in sharded path: store/{id[:2]}/{id[2:4]}/{id}
        assert file_path.parent.parent.name == ref.id[:2]
        assert file_path.parent.name == ref.id[2:4]
        assert file_path.name == ref.id


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_store_file(self, temp_file_store):
        """Test store_file helper function."""
        content = "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0"

        ref_dict = store_file(content, FileType.PDB)

        assert FileReference.is_file_ref(ref_dict)
        assert ref_dict["file_type"] == "pdb"
        assert ref_dict["size"] == len(content.encode("utf-8"))

    def test_get_file_content_with_file_ref(self, temp_file_store):
        """Test get_file_content retrieves content from file reference."""
        content = "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0"
        ref_dict = store_file(content, FileType.PDB)

        retrieved = get_file_content(ref_dict)

        assert retrieved == content

    def test_get_file_content_with_inline_string(self, temp_file_store):
        """Test get_file_content returns inline string as-is."""
        content = "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0"

        retrieved = get_file_content(content)

        assert retrieved == content

    def test_get_file_content_invalid_input(self, temp_file_store):
        """Test get_file_content raises error for invalid input."""
        invalid_dict = {"not": "a file ref"}

        with pytest.raises(ValueError):
            get_file_content(invalid_dict)

    def test_is_file_reference(self, temp_file_store):
        """Test is_file_reference helper."""
        content = "test"
        ref_dict = store_file(content, FileType.PDB)

        assert is_file_reference(ref_dict) is True
        assert is_file_reference({"regular": "dict"}) is False
        assert is_file_reference("string") is False


class TestGetFileStore:
    """Tests for get_file_store factory function."""

    def test_default_local_store(self, tmp_path):
        """Test default creates local file store."""
        reset_file_store()
        os.environ["FILE_STORE_TYPE"] = "local"
        os.environ["FILE_STORE_PATH"] = str(tmp_path)

        store = get_file_store()

        assert isinstance(store, LocalFileStore)

        reset_file_store()
        os.environ.pop("FILE_STORE_TYPE", None)
        os.environ.pop("FILE_STORE_PATH", None)

    def test_gcs_store_requires_bucket(self):
        """Test GCS store requires bucket name."""
        reset_file_store()
        os.environ["FILE_STORE_TYPE"] = "gcs"
        os.environ.pop("GCS_FILE_BUCKET", None)

        with pytest.raises(ValueError, match="GCS_FILE_BUCKET"):
            get_file_store()

        reset_file_store()
        os.environ.pop("FILE_STORE_TYPE", None)

    def test_singleton_behavior(self, tmp_path):
        """Test file store is singleton."""
        reset_file_store()
        os.environ["FILE_STORE_TYPE"] = "local"
        os.environ["FILE_STORE_PATH"] = str(tmp_path)

        store1 = get_file_store()
        store2 = get_file_store()

        assert store1 is store2

        reset_file_store()
        os.environ.pop("FILE_STORE_TYPE", None)
        os.environ.pop("FILE_STORE_PATH", None)


class TestGetUrl:
    """Tests for get_url and serves_redirect."""

    def test_get_url_returns_path(self, tmp_path):
        """Test get_url returns the file path for existing files."""
        store = LocalFileStore(tmp_path / "store")
        content = "ATOM 1 N ALA A 1 0.0 0.0 0.0 1.0 0.0"
        ref = store.put(content, FileType.PDB)

        url = store.get_url(ref.id)

        assert url == ref.url
        assert Path(url).exists()

    def test_get_url_nonexistent_raises(self, tmp_path):
        """Test get_url raises FileNotFoundError for missing files."""
        store = LocalFileStore(tmp_path / "store")

        with pytest.raises(FileNotFoundError):
            store.get_url("a" * 64)

    def test_local_serves_redirect_is_false(self, tmp_path):
        """Test LocalFileStore.serves_redirect is False."""
        store = LocalFileStore(tmp_path / "store")
        assert store.serves_redirect is False

    def test_gcs_serves_redirect_is_true(self):
        """Test GCSFileStore.serves_redirect is True."""
        from proto_language.storage import GCSFileStore

        # GCSFileStore sets serves_redirect at the class level
        assert GCSFileStore.serves_redirect is True


class TestGetContentType:
    """Tests for get_content_type and .type sidecar files."""

    def test_content_type_from_sidecar(self, tmp_path):
        """Test get_content_type reads the .type sidecar written by put()."""
        store = LocalFileStore(tmp_path / "store")
        ref = store.put("ATOM 1 N ALA", FileType.PDB)

        assert store.get_content_type(ref.id) == "chemical/x-pdb"

    def test_content_type_csv(self, tmp_path):
        """Test CSV content type."""
        store = LocalFileStore(tmp_path / "store")
        ref = store.put("a,b,c\n1,2,3", FileType.CSV)

        assert store.get_content_type(ref.id) == "text/csv"

    def test_content_type_json(self, tmp_path):
        """Test JSON content type."""
        store = LocalFileStore(tmp_path / "store")
        ref = store.put('{"key": "value"}', FileType.JSON)

        assert store.get_content_type(ref.id) == "application/json"

    def test_content_type_missing_sidecar(self, tmp_path):
        """Test fallback to octet-stream when sidecar is missing."""
        store = LocalFileStore(tmp_path / "store")
        ref = store.put("content", FileType.PDB)

        # Delete the sidecar to simulate a file stored before sidecar support
        sidecar = Path(ref.url).with_suffix(".type")
        sidecar.unlink()

        assert store.get_content_type(ref.id) == "application/octet-stream"

    def test_content_type_nonexistent_file(self, tmp_path):
        """Test fallback for nonexistent file ID."""
        store = LocalFileStore(tmp_path / "store")
        assert store.get_content_type("a" * 64) == "application/octet-stream"

    def test_sidecar_written_on_put(self, tmp_path):
        """Test that put() creates the .type sidecar file."""
        store = LocalFileStore(tmp_path / "store")
        ref = store.put("content", FileType.FASTA)

        sidecar = Path(ref.url).with_suffix(".type")
        assert sidecar.exists()
        assert sidecar.read_text() == "fasta"


class TestGetBatch:
    """Tests for FileStore.get_batch concurrent retrieval."""

    def test_get_batch_multiple_files(self, tmp_path):
        """Batch retrieval returns all requested files."""
        store = LocalFileStore(tmp_path / "store")
        refs = [store.put(f"content_{i}", FileType.PDB) for i in range(5)]
        file_ids = {r.id for r in refs}

        result = store.get_batch(file_ids)

        assert len(result) == 5
        for i, ref in enumerate(refs):
            assert result[ref.id].decode("utf-8") == f"content_{i}"

    def test_get_batch_empty(self, tmp_path):
        """Empty file_ids returns empty dict without creating threads."""
        store = LocalFileStore(tmp_path / "store")
        assert store.get_batch(set()) == {}

    def test_get_batch_missing_file_raises(self, tmp_path):
        """Missing file in batch raises FileNotFoundError."""
        store = LocalFileStore(tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            store.get_batch({"nonexistent_id"})


class TestComputeHash:
    """Tests for hash computation."""

    def test_compute_hash_deterministic(self):
        """Test hash is deterministic for same content."""
        content = "test content"

        hash1 = LocalFileStore.compute_hash(content)
        hash2 = LocalFileStore.compute_hash(content)

        assert hash1 == hash2

    def test_compute_hash_different_content(self):
        """Test different content produces different hash."""
        hash1 = LocalFileStore.compute_hash("content1")
        hash2 = LocalFileStore.compute_hash("content2")

        assert hash1 != hash2

    def test_compute_hash_bytes(self):
        """Test hash works with bytes."""
        content_str = "test"
        content_bytes = b"test"

        hash_str = LocalFileStore.compute_hash(content_str)
        hash_bytes = LocalFileStore.compute_hash(content_bytes)

        assert hash_str == hash_bytes
