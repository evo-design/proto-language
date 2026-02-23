"""
Test configuration and fixtures for the proto-language test suite.
"""

import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from proto_language import setup_logging


def is_on_chimera() -> bool:
    """Check if running on the Chimera (arc-slurm) cluster."""
    return os.environ.get("SLURM_CLUSTER_NAME") == "arc-slurm"


# Helper to create a mock generator spec for patching
def _create_mock_generator_spec(
    category: str = "autoregressive", sequence_types: list = None
):
    """Create a mock GeneratorSpec with the given category."""
    mock_spec = Mock()
    mock_spec.category = category
    mock_spec.supported_sequence_types = sequence_types or ["dna"]
    return mock_spec


# Fixture to patch GeneratorRegistry for mock generators
@pytest.fixture(autouse=True)
def mock_generator_registry(monkeypatch):
    """Patch GeneratorRegistry to return autoregressive category for mock generators."""
    from proto_language.language.generator import generator_registry

    original_get_key = generator_registry.GeneratorRegistry.get_key
    original_get = generator_registry.GeneratorRegistry.get

    def patched_get_key(generator):
        # For mock generators, return a fake key
        if generator.__class__.__name__ in (
            "MockAutoregressiveGenerator",
            "MockAutoregressiveGeneratorNoKVCache",
            "ControlledMockGenerator",
            "SegmentAwareMockGenerator",
            "AccumulativeTrackingGenerator",
        ):
            return f"mock-{generator.__class__.__name__}"
        if generator.__class__.__name__ == "MockMutationGenerator":
            return "mock-mutation"
        if generator.__class__.__name__ == "MockInverseFoldingGenerator":
            return "mock-inverse-folding"
        return original_get_key(generator)

    def patched_get(key):
        # For mock generator keys, return appropriate mock spec
        if key.startswith("mock-"):
            if key == "mock-mutation":
                return _create_mock_generator_spec("mutation")
            if key == "mock-inverse-folding":
                return _create_mock_generator_spec("inverse_folding", ["protein"])
            return _create_mock_generator_spec("autoregressive")
        return original_get(key)

    monkeypatch.setattr(
        generator_registry.GeneratorRegistry,
        "get_key",
        classmethod(lambda cls, gen: patched_get_key(gen)),
    )
    monkeypatch.setattr(
        generator_registry.GeneratorRegistry,
        "get",
        classmethod(lambda cls, key: patched_get(key)),
    )


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--cpu",
        action="store_true",
        default=False,
        help="Run only CPU tests, skip GPU tests",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run only GPU tests, skip CPU tests",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests including slow tests",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run only slow tests",
    )
    parser.addoption(
        "--skip-ci",
        action="store_true",
        default=False,
        help="Skip tests marked with skip_ci (mimics CI environment behavior)",
    )
    parser.addoption(
        "--no-log-console",
        action="store_true",
        default=False,
        help="Disable console logging during tests",
    )


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    config.addinivalue_line("markers", "uses_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "uses_cpu: mark test as CPU-only")

    # Set environment variable to indicate we're in pytest
    # This prevents setup_logging() from creating timestamped files during test imports
    os.environ["PYTEST_RUNNING"] = "1"

    # Hide CUDA devices when --skip-ci is specified to simulate CI environment
    if config.getoption("--skip-ci"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Note: We don't configure pytest's log file here. Instead, we rely on
    # setup_logging() in the setup_test_logging fixture which already has
    # the ProtoLanguageOnlyFilter applied


def pytest_runtest_logstart(nodeid, location):
    """Log when a test starts (DEBUG level, file only)."""
    logger = logging.getLogger("proto_language.tests")
    logger.debug(f"TEST START: {nodeid}")


def pytest_runtest_logreport(report):
    """Log test results (DEBUG level to avoid console output)."""
    logger = logging.getLogger("proto_language.tests")

    # Only log on the call phase (not setup/teardown)
    if report.when == "call":
        if report.passed:
            logger.debug(f"TEST PASSED: {report.nodeid}")
        elif report.failed:

            logger.error(f"TEST FAILED: {report.nodeid}")
            if report.longrepr:
                # Use DEBUG level to keep it file-only, but prefix with ERROR for visibility in logs
                logger.debug(f"Error Traceback:\n{report.longreprtext}")


def pytest_sessionfinish(session, exitstatus):
    """Log test session summary at the end."""
    logger = logging.getLogger("proto_language.tests")

    # Get test statistics from the session
    test_reports = session.items
    num_collected = len(test_reports)

    # Count passed and failed tests from the terminal reporter
    if hasattr(session.config, 'pluginmanager'):
        terminalreporter = session.config.pluginmanager.get_plugin('terminalreporter')
        if terminalreporter:
            stats = terminalreporter.stats

            passed = len(stats.get('passed', []))
            failed = len(stats.get('failed', []))
            skipped = len(stats.get('skipped', []))
            errors = len(stats.get('error', []))

            # Build summary message
            summary_lines = [
                "\n" + "=" * 80,
                "TEST SESSION SUMMARY",
                "=" * 80,
                f"Tests collected: {num_collected}",
                f"Tests passed:    {passed}",
                f"Tests failed:    {failed}",
                f"Tests skipped:   {skipped}",
                f"Tests errors:    {errors}",
            ]

            # Add list of failed tests if any
            if failed > 0:
                summary_lines.append("\nFailed tests:")
                failed_reports = stats.get('failed', [])
                for report in failed_reports:
                    summary_lines.append(f"  - {report.nodeid}")

            # Add list of error tests if any
            if errors > 0:
                summary_lines.append("\nTests with errors:")
                error_reports = stats.get('error', [])
                for report in error_reports:
                    summary_lines.append(f"  - {report.nodeid}")

            summary_lines.append("=" * 80)

            # Log the summary at INFO level so it appears in both console and file
            summary_message = "\n".join(summary_lines)
            logger.info(summary_message)


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options and auto-mark tests."""
    # Auto-mark all tests as CPU-only unless explicitly marked as GPU
    for item in items:
        # If no GPU marker found, mark as CPU
        if not any(mark.name == "uses_gpu" for mark in item.iter_markers()):
            item.add_marker(pytest.mark.uses_cpu)

    # Skip tests marked with skip_ci when running in GitHub Actions or --skip-ci is specified
    if os.getenv("GITHUB_ACTIONS") == "true" or config.getoption("--skip-ci"):
        skip_ci = pytest.mark.skip(
            reason="Skipped in CI environment (GitHub Actions or --skip-ci)"
        )
        for item in items:
            if "skip_ci" in item.keywords:
                item.add_marker(skip_ci)

    # Skip only_chimera tests when not on Chimera cluster
    if not is_on_chimera():
        skip_not_chimera = pytest.mark.skip(
            reason="Test requires Chimera cluster (SLURM_CLUSTER_NAME != 'arc-slurm')"
        )
        for item in items:
            if "only_chimera" in item.keywords:
                item.add_marker(skip_not_chimera)

    # Skip GPU tests when --cpu is specified
    if config.getoption("--cpu"):
        skip_gpu = pytest.mark.skip(reason="--cpu specified")
        for item in items:
            if "uses_gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip CPU tests when --gpu is specified
    elif config.getoption("--gpu"):
        skip_cpu = pytest.mark.skip(reason="--gpu specified")
        for item in items:
            if "uses_cpu" in item.keywords and "uses_gpu" not in item.keywords:
                item.add_marker(skip_cpu)

    # Handle slow test filtering
    run_all = config.getoption("--all")
    run_slow_only = config.getoption("--slow")

    if run_slow_only:
        # When --slow is specified, skip tests NOT marked as slow
        skip_non_slow = pytest.mark.skip(
            reason="--slow specified, skipping non-slow tests"
        )
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_non_slow)
    elif not run_all:
        # By default (no --all flag), skip slow tests
        skip_slow = pytest.mark.skip(
            reason="slow test (use --all to run, or --slow to run only slow tests)"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging(request):
    """Set up logging for the test session. Runs early to prevent timestamped log files."""
    # Use same log directory as application logs (logs/ in project root)
    project_root = Path(__file__).parent.parent
    log_dir = os.environ.get(
        "PROTO_LANGUAGE_LOG_DIR",
        str(project_root / "logs")
    )

    # Get options from command line
    no_log_console = request.config.getoption("--no-log-console")
    k_expression = request.config.getoption("-k", default=None)

    # Clear any existing handlers first to prevent duplicate log files
    bio_prog_logger = logging.getLogger("proto_language")
    bio_prog_logger.handlers.clear()

    # Create header with pytest command and timestamp
    pytest_command = " ".join(sys.argv)
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    datestamp = now.strftime("%m/%d/%Y")
    header = f"Pytest Run Command: `{pytest_command}`\nRun Started: {timestamp} on {datestamp}\n{'=' * 80}\n\n"

    # Create log filename based on -k parameter or timestamp
    if k_expression:
        # Sanitize the -k expression to make it filename-safe
        # Replace spaces with underscores, remove special characters
        sanitized = re.sub(r'[^\w\s-]', '', k_expression)  # Remove special chars except spaces, hyphens, underscores
        sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
        sanitized = sanitized.strip('_')  # Remove leading/trailing underscores
        log_filename = f"pytest_{sanitized}.log"
    else:
        # Use timestamp for the log file
        file_timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_filename = f"pytest_{file_timestamp}.log"

    # Configure logging (use pytest's --log-cli-level for level control)
    setup_logging(
        level=logging.INFO,
        log_dir=log_dir,
        log_filename=log_filename,
        log_to_file=True,
        log_to_console=not no_log_console,
        log_file_header=header,
    )

    # Suppress noisy third-party loggers that aren't suppressed by setup_logging
    # (setup_logging only suppresses proto_language's child loggers)
    noisy_test_loggers = [
        "httpcore",
        "httpx",
        "LiteLLM",
        "openai",
        "asyncio",
        "urllib3",
        "requests",
    ]
    for logger_name in noisy_test_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    yield


@pytest.fixture(scope="session", autouse=True)
def setup_cloud_environment():
    """Ensure cloud can find credentials in pytest context."""
    import toml

    logger = logging.getLogger(__name__)

    cloud_config_path = Path.home() / ".cloud.toml"
    if cloud_config_path.exists():
        config = toml.load(cloud_config_path)

        # Find active profile (proto-language)
        if "proto-language" in config:
            os.environ["CLOUD_TOKEN_ID"] = config["proto-language"]["token_id"]
            os.environ["CLOUD_TOKEN_SECRET"] = config["proto-language"]["token_secret"]
            os.environ["CLOUD_ENVIRONMENT"] = "main"
            logger.info("Loaded cloud credentials for proto-language workspace")

    yield

    # Cleanup
    for key in ["CLOUD_TOKEN_ID", "CLOUD_TOKEN_SECRET", "CLOUD_ENVIRONMENT"]:
        if key in os.environ:
            del os.environ[key]


# Sample PDB content for testing (minimal valid structure)
SAMPLE_PDB_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.326   1.562   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.941   2.877   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.449   2.831   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.074   1.772   0.000  1.00  0.00           O
ATOM      9  N   SER A   3       6.032   4.027   0.000  1.00  0.00           N
ATOM     10  CA  SER A   3       7.476   4.180   0.000  1.00  0.00           C
ATOM     11  C   SER A   3       8.064   5.572   0.000  1.00  0.00           C
ATOM     12  O   SER A   3       7.337   6.562   0.000  1.00  0.00           O
ATOM     13  OG  SER A   3       7.929   3.453   1.135  1.00  0.00           O
ATOM     14  N   VAL A   4       9.377   5.660   0.000  1.00  0.00           N
ATOM     15  CA  VAL A   4      10.044   6.955   0.000  1.00  0.00           C
ATOM     16  C   VAL A   4      11.548   6.820   0.000  1.00  0.00           C
ATOM     17  O   VAL A   4      12.101   5.720   0.000  1.00  0.00           O
ATOM     18  CB  VAL A   4       9.566   7.867  -1.140  1.00  0.00           C
ATOM     19  CG1 VAL A   4      10.238   9.235  -1.043  1.00  0.00           C
ATOM     20  CG2 VAL A   4       8.050   8.008  -1.071  1.00  0.00           C
ATOM     21  N   LEU A   5      12.207   7.978   0.000  1.00  0.00           N
ATOM     22  CA  LEU A   5      13.655   8.068   0.000  1.00  0.00           C
ATOM     23  C   LEU A   5      14.195   9.485   0.000  1.00  0.00           C
ATOM     24  O   LEU A   5      13.424  10.440   0.000  1.00  0.00           O
ATOM     25  CB  LEU A   5      14.232   7.264  -1.171  1.00  0.00           C
ATOM     26  CG  LEU A   5      13.781   7.730  -2.561  1.00  0.00           C
ATOM     27  CD1 LEU A   5      14.329   6.786  -3.630  1.00  0.00           C
ATOM     28  CD2 LEU A   5      14.248   9.152  -2.857  1.00  0.00           C
END
"""


@pytest.fixture
def sample_pdb_content():
    """Return sample PDB content string for tests that need it directly."""
    return SAMPLE_PDB_CONTENT


@pytest.fixture
def temp_pdb_file():
    """Create a temporary PDB file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(SAMPLE_PDB_CONTENT)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for tests."""
    try:
        from proto_language.utils import is_gpu_available

        return is_gpu_available()
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def mock_background_tasks():
    """Mock background task dispatch for all tests.

    Prevents tests from dispatching to cloud or spawning background threads.
    """
    with patch("api.main._dispatch_stage"):
        yield


@pytest.fixture(autouse=True)
def mock_redis():
    """Mock a cache connections for both sync and async operations."""
    # Sync a cache mock
    mock_redis_sync = Mock()
    mock_redis_sync.ping.return_value = True
    mock_redis_sync.publish.return_value = 1
    mock_redis_sync.setex.return_value = True
    mock_redis_sync.exists.return_value = 0
    mock_redis_sync.get.return_value = None

    # Async a cache mock
    from unittest.mock import AsyncMock

    mock_redis_async = AsyncMock()
    mock_redis_async.publish = AsyncMock(return_value=1)
    mock_redis_async.pubsub_numsub = AsyncMock(return_value=[("run:sse:test", 0)])
    mock_redis_async.pubsub_channels = AsyncMock(return_value=[])

    with (
        patch("cache.a cache", return_value=mock_redis_sync),
        patch("cache.StrictRedis", return_value=mock_redis_sync),
        patch(
            "api.core.sse.SSEManager.get_sync_redis_client",
            return_value=mock_redis_sync,
        ),
        patch(
            "api.core.sse.SSEManager.get_async_redis_client",
            return_value=mock_redis_async,
        ),
    ):
        yield mock_redis_sync


@pytest.fixture(autouse=True)
def mock_database():
    """Mock database dependencies for tests."""
    # Mock the database session
    mock_session = Mock()

    # Mock SQLAlchemy text execution for health check
    mock_result = Mock()
    mock_result.fetchone.return_value = (1,)
    mock_session.execute.return_value = mock_result

    # Mock a simple run object with all required attributes
    from datetime import datetime, timezone

    mock_run = Mock()
    mock_run.id = uuid4()  # Changed back to id
    mock_run.user_id = "test_user"  # Added user_id
    mock_run.status = "pending"
    mock_run.created_at = datetime.now(timezone.utc)
    mock_run.updated_at = datetime.now(timezone.utc)
    mock_run.started_at = None
    mock_run.completed_at = None
    mock_run.gpl_request = {}
    mock_run.task_id = None
    mock_run.stage_task_ids = []
    mock_run.num_steps = None
    mock_run.current_stage = 0
    mock_run.total_stages = 1
    mock_run.intermediate_states = []
    mock_run.auto_continue = False
    mock_run.result = None
    mock_run.error_message = None
    mock_run.stage_results = []

    # Mock session dependency (generator function, not async)
    def mock_get_session():
        yield mock_session

    # Check if API modules are available before patching
    patches = []

    try:
        import api.db.database

        patches.extend(
            [
                patch(
                    "api.db.database.DatabaseManager.create_run", return_value=mock_run
                ),
                patch("api.db.database.DatabaseManager.get_run", return_value=mock_run),
                patch(
                    "api.db.database.DatabaseManager.update_run", return_value=mock_run
                ),
                patch(
                    "api.db.database.DatabaseManager.get_timepoints", return_value=[]
                ),
            ]
        )
    except ImportError:
        pass

    try:
        import api.main  # noqa

        patches.extend(
            [
                patch("api.main.get_session", side_effect=mock_get_session),
                patch("api.main.create_db_and_tables"),
            ]
        )
    except ImportError:
        pass

    # Apply patches if any are available
    if patches:
        # Use contextlib.ExitStack to handle multiple patches
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield mock_session
    else:
        yield mock_session
