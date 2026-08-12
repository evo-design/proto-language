"""Run-scoped, on-demand Modal dispatch for language optimizations."""

import contextvars
import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock

_ACTIVE = contextvars.ContextVar("proto_language_modal_dispatch_active", default=False)


def _deployment_lock(app_name: str, environment: str) -> FileLock:
    """Coordinate the first deployment across local processes without storing credentials."""
    workspace = os.environ.get("MODAL_TOKEN_ID", "default-profile")
    identity = hashlib.sha256(f"{workspace}\0{environment}\0{app_name}".encode()).hexdigest()
    root = Path(tempfile.gettempdir()) / "proto-language-modal-deployments"
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(root / f"{identity}.lock")


def _dispatch_with_deployment(key: str, inputs: Any, config: Any, environment: str) -> Any:
    """Dispatch one actual tool call, deploying only its missing app before retrying."""
    from proto_tools.modal import ToolNotDeployedError, dispatch_to_modal
    from proto_tools.modal.deploy import deploy_app

    try:
        return dispatch_to_modal(key, inputs, config, environment=environment)
    except ToolNotDeployedError as exc:
        app_name = exc.app_name

    with _deployment_lock(app_name, environment):
        # Another process may have completed the same app while this call waited.
        try:
            return dispatch_to_modal(key, inputs, config, environment=environment)
        except ToolNotDeployedError:
            if not deploy_app(app_name, environment):
                raise RuntimeError(f"Failed to deploy Modal app {app_name!r} for tool {key!r}.") from None
        return dispatch_to_modal(key, inputs, config, environment=environment)


@contextmanager
def on_demand_modal_tools() -> Iterator[None]:
    """Route GPU tool calls to Modal and deploy only missing apps they reach.

    Entering this context is the cost-bearing approval: a missing app may build
    and run its normal warmup. CPU and local-only tools continue to run locally,
    and deployable GPU tools that are never called deploy nothing.
    """
    from proto_tools.modal.app import resolve_environment
    from proto_tools.tools import ToolRegistry

    if ToolRegistry.dispatch_backend_configured():
        raise RuntimeError("Cannot enable on-demand Modal tools while another dispatch backend is configured.")

    resolved_environment = resolve_environment(None)

    def dispatch(key: str, inputs: Any, config: Any) -> Any:
        # ToolRegistry's hook is process-global; the ContextVar keeps unrelated
        # threads and tasks on their existing execution path.
        if not _ACTIVE.get():
            return None
        spec = ToolRegistry.get(key)
        if not spec.uses_gpu or spec.local_only is not None:
            return None
        return _dispatch_with_deployment(key, inputs, config, resolved_environment)

    ToolRegistry.configure_dispatch_backend(dispatch)
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        ToolRegistry.clear_dispatch_backend()
        _ACTIVE.reset(token)
