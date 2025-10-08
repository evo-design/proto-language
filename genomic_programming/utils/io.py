"""
io.py
"""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
import sys


@contextmanager
def suppress_console_output(stdout=True, stderr=True):
    """
    Suppress stdout and/or stderr output.

    Args:
        stdout (bool): Whether to suppress stdout
        stderr (bool): Whether to suppress stderr
    """
    stdout_redirect = (
        redirect_stdout(StringIO()) if stdout else redirect_stdout(sys.stdout)
    )
    stderr_redirect = (
        redirect_stderr(StringIO()) if stderr else redirect_stderr(sys.stderr)
    )

    with stdout_redirect, stderr_redirect:
        yield
