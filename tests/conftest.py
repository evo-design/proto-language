"""Test configuration and fixtures for the proto-language test suite."""

import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from proto_language import setup_logging


# Fixture to patch GeneratorRegistry for mock generators
@pytest.fixture(autouse=True)
def mock_generator_registry(monkeypatch):
    """Patch GeneratorRegistry to return autoregressive category for mock generators."""
    from proto_language.generator import generator_registry

    original_get_key = generator_registry.GeneratorRegistry.get_key

    # Map mock generators to real registry keys so they pass compatible_generators checks.
    # Autoregressive mocks → "evo1", mutation → "random-protein", inverse_folding → "proteinmpnn".
    _MOCK_KEY_MAP = {
        "MockAutoregressiveGenerator": "evo1",
        "MockAutoregressiveGeneratorNoKVCache": "evo1",
        "ControlledMockGenerator": "evo1",
        "SegmentAwareMockGenerator": "evo1",
        "AccumulativeTrackingGenerator": "evo1",
        "TrackingKVCacheGenerator": "evo1",
        "MockMutationGenerator": "random-protein",
        "MockCyclingGenerator": "random-protein",
        "MockInverseFoldingGenerator": "proteinmpnn",
    }

    original_find_key = generator_registry.GeneratorRegistry.find_key

    def patched_key_lookup(generator):
        key = _MOCK_KEY_MAP.get(generator.__class__.__name__)
        return key if key is not None else original_get_key(generator)

    def patched_find_key(generator):
        key = _MOCK_KEY_MAP.get(generator.__class__.__name__)
        return key if key is not None else original_find_key(generator)

    monkeypatch.setattr(
        generator_registry.GeneratorRegistry,
        "get_key",
        classmethod(lambda cls, gen: patched_key_lookup(gen)),
    )
    monkeypatch.setattr(
        generator_registry.GeneratorRegistry,
        "find_key",
        classmethod(lambda cls, gen: patched_find_key(gen)),
    )


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Run only CPU tests, skip GPU tests",
    )
    parser.addoption(
        "--gpu-only",
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
        "--integration",
        action="store_true",
        default=False,
        help="Include integration tests (require external tools like MAFFT). Skipped by default.",
    )
    parser.addoption(
        "--extensive",
        "--ext",
        action="store_true",
        default=False,
        dest="extensive",
        help="Include extensive combinatorial tests (e.g., every constraint x optimizer). Skipped by default.",
    )
    parser.addoption(
        "--no-log-console",
        action="store_true",
        default=False,
        help="Disable console logging during tests",
    )
    parser.addoption(
        "--random-order",
        action="store_true",
        default=False,
        help="Shuffle test order (pytest-randomly's reordering, which this suite disables by default)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    # Set environment variable to indicate we're in pytest
    # This prevents setup_logging() from creating timestamped files during test imports
    os.environ["PYTEST_RUNNING"] = "1"

    # Hide CUDA devices when --skip-ci is specified to simulate CI environment
    if config.getoption("--skip-ci"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Deterministic order by default, so a failure reproduces from the same command that found it.
    # Integration CI passes --random-order, which is where order-dependence is worth surfacing.
    if hasattr(config.option, "randomly_reorganize"):
        config.option.randomly_reorganize = bool(config.getoption("--random-order"))

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
    if hasattr(session.config, "pluginmanager"):
        terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            stats = terminalreporter.stats

            passed = len(stats.get("passed", []))
            failed = len(stats.get("failed", []))
            skipped = len(stats.get("skipped", []))
            errors = len(stats.get("error", []))

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
                failed_reports = stats.get("failed", [])
                summary_lines.extend(f"  - {report.nodeid}" for report in failed_reports)

            # Add list of error tests if any
            if errors > 0:
                summary_lines.append("\nTests with errors:")
                error_reports = stats.get("error", [])
                summary_lines.extend(f"  - {report.nodeid}" for report in error_reports)

            summary_lines.append("=" * 80)

            # Log the summary at INFO level so it appears in both console and file
            summary_message = "\n".join(summary_lines)
            logger.info(summary_message)


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options and auto-mark tests."""
    # Auto-mark CPU tests — skip if already tagged (uses_gpu or explicit uses_cpu(n))
    # so we don't stack a zero-arg duplicate on top of an explicit count.
    for item in items:
        markers = list(item.iter_markers())
        if not any(m.name == "uses_gpu" for m in markers) and not any(m.name == "uses_cpu" for m in markers):
            item.add_marker(pytest.mark.uses_cpu)

    # Skip tests marked with skip_ci when running in GitHub Actions or --skip-ci is specified
    if os.getenv("GITHUB_ACTIONS") == "true" or config.getoption("--skip-ci"):
        skip_ci = pytest.mark.skip(reason="Skipped in CI environment (GitHub Actions or --skip-ci)")
        for item in items:
            if "skip_ci" in item.keywords:
                item.add_marker(skip_ci)

    # Skip GPU tests when --cpu-only is specified
    if config.getoption("--cpu-only"):
        skip_gpu = pytest.mark.skip(reason="--cpu-only specified")
        for item in items:
            if "uses_gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip CPU tests when --gpu-only is specified
    elif config.getoption("--gpu-only"):
        skip_cpu = pytest.mark.skip(reason="--gpu-only specified")
        for item in items:
            if "uses_cpu" in item.keywords and "uses_gpu" not in item.keywords:
                item.add_marker(skip_cpu)

    # Handle slow test filtering
    run_all = config.getoption("--all")
    run_slow_only = config.getoption("--slow")

    if run_slow_only:
        # When --slow is specified, skip tests NOT marked as slow
        skip_non_slow = pytest.mark.skip(reason="--slow specified, skipping non-slow tests")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_non_slow)
    elif not run_all and not config.getoption("extensive"):
        # By default (no --all / no --ext flag), skip slow tests
        skip_slow = pytest.mark.skip(reason="slow test (use --all to run, or --slow to run only slow tests)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip integration tests unless --integration or --all
    if not config.getoption("--integration") and not run_all:
        skip_integration = pytest.mark.skip(reason="integration test (use --integration or --all to run)")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip extensive tests unless --ext (or --extensive) is specified
    if not config.getoption("extensive"):
        skip_extensive = pytest.mark.skip(reason="extensive test (use --ext to run)")
        for item in items:
            if "extensive" in item.keywords:
                item.add_marker(skip_extensive)

    # Skip uses_gpu(n) tests when fewer than n GPUs are visible. Bare uses_gpu = count=1.
    # Helpers come from proto-tools (shared runtime), keeping count detection identical across layers.
    # Skipped under --cpu-only — the --cpu-only block above already skip-marks every uses_gpu test,
    # and number_of_visible_gpus shells out to nvidia-smi, which we shouldn't pay for on CPU runs.
    if not config.getoption("--cpu-only"):
        from proto_tools.utils.device import number_of_visible_gpus

        visible_gpus = number_of_visible_gpus()
        for item in items:
            for marker in item.iter_markers("uses_gpu"):
                required = marker.args[0] if marker.args else 1
                if visible_gpus < required:
                    unit = "GPU" if required == 1 else "GPUs"
                    item.add_marker(pytest.mark.skip(reason=f"Requires {required} {unit}, only {visible_gpus} visible"))

    # Skip uses_cpu(n) tests when fewer than n CPUs are visible. Bare uses_cpu = count=1.
    # _detect_cpus has no subprocess cost — fine to run unconditionally.
    from proto_tools.utils.tool_pool import _detect_cpus

    visible_cpus = _detect_cpus()
    for item in items:
        for marker in item.iter_markers("uses_cpu"):
            required = marker.args[0] if marker.args else 1
            if visible_cpus < required:
                unit = "CPU" if required == 1 else "CPUs"
                item.add_marker(pytest.mark.skip(reason=f"Requires {required} {unit}, only {visible_cpus} visible"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging(request):
    """Set up logging for the test session. Runs early to prevent timestamped log files."""
    # Use same log directory as application logs (logs/ in project root)
    project_root = Path(__file__).parent.parent
    log_dir = str(project_root / "logs")

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
        sanitized = re.sub(r"[^\w\s-]", "", k_expression)  # Remove special chars except spaces, hyphens, underscores
        sanitized = re.sub(r"\s+", "_", sanitized)  # Replace spaces with underscores
        sanitized = sanitized.strip("_")  # Remove leading/trailing underscores
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
        "asyncio",
        "urllib3",
    ]
    for logger_name in noisy_test_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    yield


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
def toy_json():
    with open(os.path.join(os.path.dirname(__file__), "../examples/jsons/toy.json")) as f:
        return json.load(f)
