"""
Test configuration and fixtures for the proto-language test suite.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4


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
    
    # Patch imports in the API module and tasks
    with patch("api.main.celery_app", mock_celery_app), \
         patch("api.main.run_program_task", mock_task), \
         patch("api.celery_config.celery_app", mock_celery_app), \
         patch("api.tasks.celery_app", mock_celery_app), \
         patch("api.tasks.run_program_task", mock_task):
        yield {
            "celery_app": mock_celery_app,
            "task": mock_task
        }


@pytest.fixture(autouse=True) 
def mock_redis():
    """Mock a cache connections."""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    
    with patch("cache.a cache", return_value=mock_redis), \
         patch("cache.StrictRedis", return_value=mock_redis):
        yield mock_redis


@pytest.fixture(autouse=True)
def mock_database():
    """Mock database dependencies for tests."""
    # Mock the database session
    mock_session = AsyncMock()
    
    # Mock SQLAlchemy text execution for health check
    mock_result = Mock()
    mock_result.fetchone.return_value = (1,)
    mock_session.execute.return_value = mock_result
    
    # Mock a simple run object with all required attributes
    from datetime import datetime
    mock_run = Mock()
    mock_run.id = uuid4()
    mock_run.status = "pending"
    mock_run.created_at = datetime.utcnow()
    mock_run.updated_at = datetime.utcnow()
    mock_run.started_at = None
    mock_run.completed_at = None
    mock_run.total_steps = None
    mock_run.track_step_size = None
    mock_run.result = None
    mock_run.error_message = None
    
    # Mock session dependency  
    async def mock_get_session():
        yield mock_session
    
    # Mock DatabaseManager methods
    with patch("api.database.DatabaseManager.create_run", new_callable=AsyncMock, return_value=mock_run), \
         patch("api.database.DatabaseManager.get_run", new_callable=AsyncMock, return_value=mock_run), \
         patch("api.database.DatabaseManager.update_run", new_callable=AsyncMock, return_value=mock_run), \
         patch("api.database.DatabaseManager.get_timepoints", new_callable=AsyncMock, return_value=[]), \
         patch("api.main.get_session", side_effect=mock_get_session), \
         patch("api.main.create_db_and_tables", new_callable=AsyncMock):
        yield mock_session