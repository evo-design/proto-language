"""Focused tests for language-owned, on-demand Modal deployment."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

from proto_language.modal import _dispatch_with_deployment, on_demand_modal_tools


def test_current_tool_dispatches_without_deploying(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    deployed: list[str] = []
    monkeypatch.setattr("proto_tools.modal.dispatch_to_modal", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr("proto_tools.modal.deploy.deploy_app", lambda app, _env: deployed.append(app) or True)

    result = _dispatch_with_deployment("boltz2-prediction", object(), object(), "proto-env")

    assert result is expected
    assert deployed == []


def test_missing_tool_deploys_only_its_app_then_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from proto_tools.modal import ToolNotDeployedError

    calls = 0
    deployed: list[tuple[str, str]] = []
    expected = object()

    def dispatch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ToolNotDeployedError("boltz2-prediction", "proto-tools-boltz2")
        return expected

    monkeypatch.setattr("proto_tools.modal.dispatch_to_modal", dispatch)
    monkeypatch.setattr(
        "proto_tools.modal.deploy.deploy_app",
        lambda app, env: deployed.append((app, env)) or True,
    )
    monkeypatch.setattr("proto_language.modal._deployment_lock", lambda *_args: nullcontext())

    result = _dispatch_with_deployment("boltz2-prediction", object(), object(), "hackathon")

    assert result is expected
    assert deployed == [("proto-tools-boltz2", "hackathon")]


def test_failed_deploy_never_falls_back_to_local_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from proto_tools.modal import ToolNotDeployedError

    calls = 0

    def missing(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ToolNotDeployedError("esmfold-prediction", "proto-tools-esmfold")

    monkeypatch.setattr("proto_tools.modal.dispatch_to_modal", missing)
    monkeypatch.setattr("proto_tools.modal.deploy.deploy_app", lambda *_args: False)
    monkeypatch.setattr("proto_language.modal._deployment_lock", lambda *_args: nullcontext())

    with pytest.raises(RuntimeError, match="Failed to deploy Modal app 'proto-tools-esmfold'"):
        _dispatch_with_deployment("esmfold-prediction", object(), object(), "proto-env")

    assert calls == 2


def test_context_installs_and_restores_one_dispatch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from proto_tools.tools import ToolRegistry

    monkeypatch.setattr("proto_tools.modal.app.resolve_environment", lambda value: value or "proto-env")
    ToolRegistry.clear_dispatch_backend()

    with on_demand_modal_tools():
        assert ToolRegistry.dispatch_backend_configured()

    assert not ToolRegistry.dispatch_backend_configured()


def test_context_does_not_overwrite_an_existing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from proto_tools.tools import ToolRegistry

    def existing(*_args):
        return None

    ToolRegistry.configure_dispatch_backend(existing)
    try:
        with pytest.raises(RuntimeError, match="another dispatch backend"):
            with on_demand_modal_tools():
                pass
        assert ToolRegistry.dispatch_backend_configured()
    finally:
        ToolRegistry.clear_dispatch_backend()


def test_unrelated_thread_is_not_captured_by_modal_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-global registry hook must remain scoped to the initiating context."""
    from proto_tools.tools import ToolRegistry

    monkeypatch.setattr("proto_tools.modal.app.resolve_environment", lambda value: value or "proto-env")
    monkeypatch.setattr(
        "proto_language.modal._dispatch_with_deployment",
        lambda *_args: pytest.fail("an unrelated thread was routed to Modal"),
    )

    with on_demand_modal_tools():
        backend = ToolRegistry._dispatch_backend
        assert backend is not None
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(backend, "any-tool", object(), object()).result()

    assert result is None


def test_program_modal_scope_skips_local_tool_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from proto_tools.tools import ToolRegistry

    from proto_language.constraint import ConstraintRegistry
    from proto_language.core import Construct, Program, Segment
    from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
    from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig

    segment = Segment(sequence="ATGC", sequence_type="dna")
    generator = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    generator.assign(segment)
    constraint = ConstraintRegistry.create(
        key="gc-content",
        segments=[segment],
        config_dict={"min_gc": 0, "max_gc": 100},
    )
    optimizer = RejectionSamplingOptimizer(
        constructs=[Construct([segment])],
        generators=[generator],
        constraints=[constraint],
        config=RejectionSamplingOptimizerConfig(num_samples=1, num_results=1),
    )
    program = Program([optimizer], num_results=1)

    class RefusingCompute:
        def __enter__(self):
            pytest.fail("device='modal' entered the local ToolPool")

        def __exit__(self, *_args):
            return None

    program.compute = RefusingCompute()
    observed: list[bool] = []
    monkeypatch.setattr("proto_tools.modal.app.resolve_environment", lambda value: value or "proto-env")
    monkeypatch.setattr(
        program, "run_stage", lambda _stage: observed.append(ToolRegistry.dispatch_backend_configured())
    )

    program.run(device="modal")

    assert observed == [True]
    assert not ToolRegistry.dispatch_backend_configured()
