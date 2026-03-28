"""
proto_language/logging_config.py

Provides centralized logging setup with file and console handlers,
automatic log directory management, suppression of noisy third-party loggers,
and integration with Python's warnings system.
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class ProtoLanguageOnlyFilter(logging.Filter):
    """Filter to only allow logs from proto_language project packages."""
    def filter(self, record):
        # Include logs from all project packages: proto_language, api, agent, deployment, tests
        allowed_prefixes = ("proto_language", "api", "agent", "deployment", "tests")
        return record.name.startswith(allowed_prefixes)


class SelectiveLevelFormatter(logging.Formatter):
    """Formatter that shows level prefix only for WARNING and above."""

    def format(self, record):
        # For WARNING, ERROR, CRITICAL: add level prefix
        if record.levelno >= logging.WARNING:
            # Save original format
            original_fmt = self._style._fmt
            # Temporarily use format with level
            self._style._fmt = "%(levelname)s: %(message)s"
            result = super().format(record)
            # Restore original format
            self._style._fmt = original_fmt
            return result
        else:
            # For DEBUG and INFO: just the message
            return record.getMessage()


def _parse_log_level(level: Union[int, str]) -> int:
    """
    Parse log level from string or int, case-insensitive.

    Args:
        level (int | str): Log level as int (e.g., logging.INFO) or string (e.g., "INFO", "info", "Info")

    Returns:
        int: Integer log level

    Raises:
        ValueError: If string level is not recognized
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_upper = level.upper()
        if hasattr(logging, level_upper):
            return getattr(logging, level_upper)
        raise ValueError(
            f"Unknown log level: '{level}'. "
            f"Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    raise TypeError(f"level must be int or str, got {type(level)}")


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    console_level: Optional[Union[int, str]] = None,
    file_level: Optional[Union[int, str]] = None,
    console_output_formatted: bool = False,
    log_file_header: Optional[str] = None,
) -> None:
    """
    Configure logging for proto_language.

    Args:
        level (int | str): Default logging level for all handlers. Can be an int (e.g., logging.INFO)
            or a case-insensitive string (e.g., "INFO", "info", "Debug"). Default: INFO.
        log_dir (str | None): Directory for log files. Defaults to logs/ in project root.
        log_filename (str | None): Custom filename for the log file. If None, uses timestamped filename
            like proto_language_YYYYMMDD_HHMMSS.log. If provided, uses this exact filename.
        log_to_file (bool): Whether to enable file logging (default: True).
        log_to_console (bool): Whether to enable console logging to stdout (default: True).
        console_level (int | str | None): Override level for console handler. Accepts int or string (default: uses `level`).
        file_level (int | str | None): Override level for file handler. Accepts int or string (default: DEBUG for full capture).
        console_output_formatted (bool): If True, console output includes timestamp and metadata like file logs.
            If False (default), console output is print-like: DEBUG/INFO show only the message,
            while WARNING/ERROR/CRITICAL include the level prefix (e.g., "WARNING: message").
        log_file_header (str | None): Optional header text to write at the top of the log file before logging starts.
    """
    # Parse levels (supports case-insensitive strings)
    level = _parse_log_level(level)
    if console_level is not None:
        console_level = _parse_log_level(console_level)
    if file_level is not None:
        file_level = _parse_log_level(file_level)

    # During pytest, disable file logging unless explicitly requested with log_filename
    # This prevents timestamped log files from being created during test imports
    if os.environ.get("PYTEST_RUNNING") == "1" and log_filename is None:
        log_to_file = False
    # Determine log directory
    if log_dir is None:
        # Default to logs/ in project root
        # Find project root by looking for pyproject.toml or use current directory
        current = Path.cwd()
        project_root = current
        # Search up the directory tree for pyproject.toml
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        log_dir = str(project_root / "logs")
    log_path = Path(log_dir)

    # Create log directory if needed
    if log_to_file:
        log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger for proto_language
    root_logger = logging.getLogger("proto_language")
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers filter

    # Clear any existing handlers to prevent duplicates on reconfiguration
    root_logger.handlers.clear()

    # File handler format (with line numbers for debugging)
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler format (default: selective level display, or full formatted output)
    if console_output_formatted:
        console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        console_formatter = logging.Formatter(console_format, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        # Use SelectiveLevelFormatter: shows level prefix only for WARNING and above
        console_formatter = SelectiveLevelFormatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stdout for print-like behavior, works in Jupyter)
    console_handler = None
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level or level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler (timestamped or custom filename)
    file_handler = None
    if log_to_file:
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"proto_language_{timestamp}.log"
        else:
            log_file = log_path / log_filename

        # Write header to log file if provided
        if log_file_header:
            with open(log_file, 'w') as f:
                f.write(log_file_header)
            # Use append mode for FileHandler so we don't overwrite the header
            file_handler = logging.FileHandler(log_file, mode='a')
        else:
            # Use write mode as before
            file_handler = logging.FileHandler(log_file, mode='w')

        file_handler.setLevel(file_level or logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        # Add filter to only log proto_language messages to file
        file_handler.addFilter(ProtoLanguageOnlyFilter())
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    noisy_loggers = [
        "transformers",
        "vortex",
        "StripedHyena",
        "esm",
        "torch",
        "urllib3",
        "httpx",
        "cloud",
        "matplotlib",
        "PIL",
        "h5py",
        "numba",
        "filelock",
        "huggingface_hub",
        "datasets",
        "accelerate",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Capture warnings through the logging system
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)

    # Add handlers to py.warnings logger so warnings go to same destinations
    if log_to_console and console_handler:
        warnings_logger.addHandler(console_handler)
    if log_to_file and file_handler:
        warnings_logger.addHandler(file_handler)

    # Filter noisy third-party deprecation/future warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"transformers.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"huggingface_hub.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"huggingface_hub.*")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a proto_language module.

    Args:
        name (str): Logger name, typically __name__ from the calling module.
            Will be prefixed with 'proto_language.' if not already.

    Returns:
        logging.Logger: Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Detailed debug info")
    """
    if not name.startswith("proto_language"):
        name = f"proto_language.{name}"
    return logging.getLogger(name)
