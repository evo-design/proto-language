# Storage Module

Content-addressed file storage for externalizing large file content (PDB structures, CIF files, HMM profiles, etc.) from sequence metadata. Supports local filesystem for development and Google Cloud Storage for production.

## Architecture

```
proto_language/storage/
├── models.py      # FileType enum, FileReference model
├── store.py       # FileStore ABC, LocalFileStore, GCSFileStore, singleton factory
├── helpers.py     # store_file(), get_file_content(), is_file_reference()
├── resolver.py    # GCS download/caching utilities for pre-existing remote files
└── __init__.py    # Public API re-exports
```

Two distinct subsystems live here:

1. **File storage** (`models.py`, `store.py`, `helpers.py`) — Store and retrieve generated files (constraint outputs like PDB structures). Content-addressed by SHA-256 hash.
2. **File resolution** (`resolver.py`) — Download and cache pre-existing GCS-hosted databases (e.g., MMseqs DBs) to local paths for tool execution. Unrelated to the storage backends.

## Content-Addressed Storage

Every file is identified by the SHA-256 hash of its content. This gives three properties:

- **Deduplication**: Identical content (even from different sequences) is stored once.
- **Immutability**: A file ID always refers to the same content. Changed content = new hash = new file.
- **Idempotent writes**: `put()` is safe to call repeatedly — if the file already exists, it's a no-op.

### Storage layout

Files are stored in a two-level sharded directory structure to avoid filesystem bottlenecks:

```
{base}/{hash[0:2]}/{hash[2:4]}/{hash}
```

For example, a file with hash `abcdef1234...` is stored at:

```
# Local
./file_store/ab/cd/abcdef1234...

# GCS
gs://my-bucket/files/ab/cd/abcdef1234...
```

## Configuration

Set via environment variables:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `FILE_STORE_TYPE` | `"local"`, `"gcs"` | `"local"` | Storage backend |
| `FILE_STORE_PATH` | Any path | `"./file_store"` | Base directory for local storage |
| `GCS_FILE_BUCKET` | Bucket name | *(required if gcs)* | GCS bucket name |

GCS credentials are resolved in order:

1. `GOOGLE_APPLICATION_CREDENTIALS_JSON` — JSON string (for containers)
2. `GOOGLE_APPLICATION_CREDENTIALS` — File path (standard)
3. Default credentials (GCE, Workload Identity, etc.)

## Usage

### Storing files (in constraints)

```python
from proto_language.storage import store_file, FileType

# Store a PDB file, get back a reference dict
ref = store_file(structure.structure_pdb, FileType.PDB)
seq._metadata["pdb_output"] = ref
```

`store_file()` returns a dictionary like:

```python
{
    "__file_ref__": True,
    "id": "abcdef1234...",       # SHA-256 hash
    "file_type": "pdb",
    "size": 48201,
    "url": "gs://bucket/files/ab/cd/abcdef1234..."
}
```

The `__file_ref__` marker distinguishes file references from regular metadata dicts.

### Retrieving content

```python
from proto_language.storage import get_file_content

# Works transparently with both inline strings and file references
pdb_content = get_file_content(seq._metadata["pdb_output"])
```

`get_file_content()` handles two cases:
- If the value is already a string → returns it as-is (backward compatible with inline content)
- If it's a file reference dict → fetches from the store and decodes as UTF-8

For binary content, use `get_file_content_bytes()`.

### Checking if a value is a file reference

```python
from proto_language.storage import is_file_reference

if is_file_reference(seq._metadata.get("pdb_output")):
    # It's stored externally
    ...
```

### Direct store access

```python
from proto_language.storage import get_file_store

store = get_file_store()  # Returns the singleton FileStore instance

# Low-level operations
ref = store.put(content, FileType.PDB)   # Store content, get FileReference
data = store.get(file_id)                 # Retrieve bytes by hash ID
exists = store.exists(file_id)            # Check existence
url = store.get_url(file_id)              # Get accessible URL/path
```

## Supported File Types

| FileType | Extension | Content-Type |
|----------|-----------|-------------|
| `PDB` | `.pdb` | `chemical/x-pdb` |
| `CIF` | `.cif` | `chemical/x-mmcif` |
| `CSV` | `.csv` | `text/csv` |
| `JSON` | `.json` | `application/json` |
| `FASTA` | `.fasta` | `text/plain` |
| `HMM` | `.hmm` | `application/octet-stream` |
| `BINARY` | — | `application/octet-stream` |

Content types are set on GCS uploads so browsers can handle files correctly.

## File Serving API Endpoint

`GET /files/{file_id}` serves stored files over HTTP.

Behavior depends on the backend:

- **GCS** (`serves_redirect = True`): Returns a `307 Redirect` to a time-limited signed URL (default 60-minute expiry).
- **Local** (`serves_redirect = False`): Streams the file content directly via `FileResponse`.
- Returns `404` if the file doesn't exist.

```python
# api/main.py
@app.get("/files/{file_id}")
def get_file(file_id: str):
    if not _FILE_ID_RE.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID format")

    store = get_file_store()
    try:
        url = store.get_url(file_id)
        if store.serves_redirect:
            return RedirectResponse(url=url)
        return FileResponse(url, media_type=store.get_content_type(file_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
```

This endpoint is what the client uses to fetch PDB content for `StructureViewer`, download CSVs, etc.

## Export Integration

The export system (`proto_language/utils/export.py`) handles file references in two modes:

### Default: URLs in exports

```python
rows = flatten_sequences(results)
# metadata.pdb_output column → "gs://bucket/files/ab/cd/abcdef1234..."
```

File references are serialized to their URL string, keeping exports compact.

### Resolved: inline content

```python
rows = flatten_sequences(results, resolve_files=True)
# metadata.pdb_output column → actual PDB file content as a string
```

When `resolve_files=True`, file references are resolved to their actual content. Useful for self-contained exports where the consumer needs the data inline (e.g., a Python script processing PDB content from a CSV).

The `resolve_files` parameter is available on all flatten functions: `flatten_sequences()`, `flatten_constraints()`, `flatten_constructs()`, `flatten_optimization()`, and `flatten_table()`.

## File Resolution (resolver.py)

Separate from the storage system. Used for downloading and caching pre-existing GCS-hosted files (like tool databases) to local paths.

```python
from proto_language.storage import resolve_file, resolve_paths

# Single file
local_path = resolve_file("gcs://bucket/mmseqs_db.tar.gz")
# → /data/a1b2c3d4e5f6g7h8  (cached in storage volume)

# Recursive resolution in nested data structures
config = resolve_paths({
    "database": "gcs://bucket/mmseqs_db",
    "threads": 4
})
# → {"database": "/data/a1b2c3d4e5f6g7h8", "threads": 4}
```

Cache location is controlled by `STORAGE_VOLUME_PATH` (default: `/data`). Files are cached by a SHA-256 hash of the reference URL.

## Storage Backends in Detail

### LocalFileStore

- Stores files as plain files on the local filesystem
- `get_url()` returns the absolute file path as a string
- `serves_redirect = False` — API endpoint streams content directly
- Suitable for development and local testing

### GCSFileStore

- Stores files as blobs in a Google Cloud Storage bucket
- `get_url()` generates a time-limited signed HTTPS URL (configurable expiry, default 60 min)
- `serves_redirect = True` — API endpoint redirects clients to the signed URL
- Uploads include proper `content_type` headers derived from `FileType`
- Lazy-loads GCS client and bucket on first use
- Credentials support JSON strings (containers), file paths, and default credentials

### Singleton Pattern

`get_file_store()` returns a module-level singleton, initialized on first call based on environment variables. Use `reset_file_store()` in tests to clear the singleton between test cases.

## Testing

```bash
pytest tests/tests_cpu/test_storage.py    # All storage tests
```

`reset_file_store()` is available for clearing the singleton in test fixtures:

```python
from proto_language.storage import reset_file_store

@pytest.fixture(autouse=True)
def clean_store():
    reset_file_store()
    yield
    reset_file_store()
```
