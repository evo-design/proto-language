"""Tests for proto_language.utils.logging_config and spinner re-exports.

Covers the bar-aware console handler (cooperation with proto-tools spinners) and
that the spinner helpers are reused from proto-tools rather than reimplemented.
"""

import logging
from unittest.mock import patch

import proto_tools.utils.progress as pt_progress

from proto_language.utils import spinner
from proto_language.utils.logging_config import (
    _BarAwareStreamHandler,
    setup_logging,
)


def _make_record(message: str = "hello") -> logging.LogRecord:
    """Build a minimal INFO LogRecord under the proto_language namespace."""
    return logging.LogRecord(
        name="proto_language.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=None,
        exc_info=None,
    )


class TestBarAwareStreamHandler:
    """The console handler must cooperate with an active proto-tools spinner."""

    def test_routes_through_tqdm_write_when_bar_active(self) -> None:
        """With a progress bar active, the record is written via tqdm.write."""
        handler = _BarAwareStreamHandler()
        record = _make_record("spinner-active")
        with (
            patch.object(pt_progress, "has_active_progress_bar", return_value=True),
            patch("tqdm.tqdm.write") as mock_write,
        ):
            handler.emit(record)
        mock_write.assert_called_once()
        assert "spinner-active" in mock_write.call_args[0][0]

    def test_falls_back_to_plain_emit_when_no_bar(self) -> None:
        """With no active bar, the handler defers to StreamHandler.emit and never calls tqdm."""
        handler = _BarAwareStreamHandler()
        record = _make_record("no-bar")
        with (
            patch.object(pt_progress, "has_active_progress_bar", return_value=False),
            patch("tqdm.tqdm.write") as mock_write,
            patch.object(logging.StreamHandler, "emit") as mock_super_emit,
        ):
            handler.emit(record)
        mock_write.assert_not_called()
        mock_super_emit.assert_called_once_with(record)

    def test_setup_logging_installs_bar_aware_console_handler(self) -> None:
        """setup_logging wires the console handler as a _BarAwareStreamHandler."""
        # setup_logging mutates global logging state: it clears+rebuilds the
        # proto_language root handlers and *appends* the console handler to the
        # py.warnings logger. Snapshot both so this test doesn't leak a handler
        # into later tests under pytest-randomly. (Noisy-logger levels and
        # captureWarnings() are left as-is; the session already configures them.)
        root = logging.getLogger("proto_language")
        warnings_logger = logging.getLogger("py.warnings")
        saved_root = root.handlers[:]
        saved_warnings = warnings_logger.handlers[:]
        try:
            setup_logging(log_to_file=False, log_to_console=True)
            assert any(isinstance(h, _BarAwareStreamHandler) for h in root.handlers)
        finally:
            root.handlers[:] = saved_root
            warnings_logger.handlers[:] = saved_warnings


class TestSpinnerReexports:
    """Spinner helpers are reused from proto-tools, not reimplemented."""

    def test_reexports_are_the_same_objects_as_proto_tools(self) -> None:
        """Each re-exported helper is identical to the proto-tools original."""
        assert spinner.progress_bar is pt_progress.progress_bar
        assert spinner.set_substatus is pt_progress.set_substatus
        assert spinner.has_active_progress_bar is pt_progress.has_active_progress_bar

    def test_internal_status_indicator_is_not_reexported(self) -> None:
        """proto-tools treats status_indicator as internal; we don't promote it."""
        assert "status_indicator" not in spinner.__all__
        assert not hasattr(spinner, "status_indicator")

    def test_helpers_exposed_on_utils_namespace(self) -> None:
        """The helpers are also reachable from the proto_language.utils package."""
        from proto_language import utils

        for name in ("progress_bar", "set_substatus", "has_active_progress_bar"):
            assert hasattr(utils, name)
