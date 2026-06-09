"""Spinner / progress-bar helpers, reused from proto-tools.

``proto_language`` reuses proto-tools' renderer (``proto_tools.utils.progress``)
rather than maintaining its own implementation, and re-exports the public helpers
here for convenience and for future call sites.

The language layer does **not** currently drive the spinner: the proto-tools
tools own the status channel and emit their own status updates, so adding
language-layer updates on top would overwhelm the shared status line. These names
are provided so callers can render progress when appropriate, and so that
:class:`proto_language.utils.logging_config._BarAwareStreamHandler` can query
whether a tool spinner is active.

Only the helpers proto-tools treats as public are re-exported. ``status_indicator``
is deliberately omitted: proto-tools documents it as an internal fallback and does
not export it, and the language layer has no call site for it.

Examples:
    >>> from proto_language.utils.spinner import has_active_progress_bar
    >>> has_active_progress_bar()  # True only while a progress bar is open
    False
"""

from proto_tools.utils.progress import (
    has_active_progress_bar,
    progress_bar,
    set_substatus,
)

__all__ = [
    "progress_bar",
    "set_substatus",
    "has_active_progress_bar",
]
