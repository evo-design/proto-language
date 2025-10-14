"""
Base configuration class for all pydantic configs.
"""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """
    Base configuration class for consistent behavior across all configs (tools, constraints, and generators).
    
    Example:
        >>> class MyToolConfig(BaseConfig):
        ...     param1: int
        ...     param2: str
    """
    
    model_config = ConfigDict(
        extra='forbid',              # Catch typos in config keys
        validate_assignment=True,    # Validate on field updates
        use_enum_values=True,        # Serialize enums as values
        validate_default=True,       # Validate default values
    )
