"""
Test configuration and fixtures for the proto-language test suite.
"""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4
import os


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


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    config.addinivalue_line("markers", "uses_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "uses_cpu: mark test as CPU-only")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options and auto-mark tests."""
    # Auto-mark all tests as CPU-only unless explicitly marked as GPU
    for item in items:
        # If no GPU marker found, mark as CPU
        if not any(mark.name == "uses_gpu" for mark in item.iter_markers()):
            item.add_marker(pytest.mark.uses_cpu)

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
        skip_non_slow = pytest.mark.skip(reason="--slow specified, skipping non-slow tests")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_non_slow)
    elif not run_all:
        # By default (no --all flag), skip slow tests
        skip_slow = pytest.mark.skip(reason="slow test (use --all to run, or --slow to run only slow tests)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def setup_cloud_environment():
    """Ensure cloud can find credentials in pytest context."""
    # Read cloud credentials from ~/.cloud.toml and set as environment variables
    import toml
    from pathlib import Path
    
    cloud_config_path = Path.home() / '.cloud.toml'
    if cloud_config_path.exists():
        config = toml.load(cloud_config_path)
        
        # Find active profile (proto-language)
        if 'proto-language' in config:
            os.environ['CLOUD_TOKEN_ID'] = config['proto-language']['token_id']
            os.environ['CLOUD_TOKEN_SECRET'] = config['proto-language']['token_secret']
            os.environ['CLOUD_ENVIRONMENT'] = 'main'
            print("✓ Loaded cloud credentials for proto-language workspace")
    
    yield
    
    # Cleanup
    for key in ['CLOUD_TOKEN_ID', 'CLOUD_TOKEN_SECRET', 'CLOUD_ENVIRONMENT']:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for tests."""
    try:
        from proto_language.utils import is_gpu_available

        return is_gpu_available()
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def mock_celery():
    """Mock the task queue and a cache dependencies for all tests."""
    
    # Mock the task queue task
    mock_task = Mock()
    mock_task.delay.return_value = Mock(id=str(uuid4()))
    
    # Mock the task queue app
    mock_celery_app = Mock()
    mock_celery_app.control.inspect.return_value.active.return_value = {"worker1": []}
    mock_celery_app.AsyncResult.return_value = Mock(status="PENDING")
    
    # Check if API modules are available before patching
    patches = []
    
    try:
        import api.main
        patches.extend([
            patch("api.main.celery_app", mock_celery_app),
            patch("api.main.run_program_task", mock_task)
        ])
    except ImportError:
        pass
    
    try:
        import api.workers.celery_config# noqa
        patches.append(patch("api.workers.celery_config.celery_app", mock_celery_app))
    except ImportError:
        pass
    
    try:
        import api.workers.tasks# noqa
        patches.extend([
            patch("api.workers.tasks.celery_app", mock_celery_app),
            patch("api.workers.tasks.run_program_task", mock_task)
        ])
    except ImportError:
        pass
    
    # Apply patches if any are available
    if patches:
        # Use contextlib.ExitStack to handle multiple patches
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield {
                "celery_app": mock_celery_app,
                "task": mock_task
            }
    else:
        yield {
            "celery_app": mock_celery_app,
            "task": mock_task
        }


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
    
    with patch("cache.a cache", return_value=mock_redis_sync), \
         patch("cache.StrictRedis", return_value=mock_redis_sync), \
         patch("api.core.sse.SSEManager.get_sync_redis_client", return_value=mock_redis_sync), \
         patch("api.core.sse.SSEManager.get_async_redis_client", return_value=mock_redis_async):
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
    mock_run.celery_task_id = None
    mock_run.num_steps = None
    mock_run.track_step_size = None
    mock_run.current_stage = 0
    mock_run.total_stages = 1
    mock_run.intermediate_state = None
    mock_run.result = None
    mock_run.error_message = None
    
    # Mock session dependency (generator function, not async)
    def mock_get_session():
        yield mock_session
    
    # Check if API modules are available before patching
    patches = []
    
    try:
        import api.db.database
        patches.extend([
            patch("api.db.database.DatabaseManager.create_run", return_value=mock_run),
            patch("api.db.database.DatabaseManager.get_run", return_value=mock_run),
            patch("api.db.database.DatabaseManager.update_run", return_value=mock_run),
            patch("api.db.database.DatabaseManager.get_timepoints", return_value=[])
        ])
    except ImportError:
        pass
    
    try:
        import api.main # noqa
        patches.extend([
            patch("api.main.get_session", side_effect=mock_get_session),
            patch("api.main.create_db_and_tables")
        ])
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
