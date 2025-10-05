"""
Base configuration class for all configs in proto_language.
"""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """
    Base configuration class for tools, constraints, and generators.
    
    Provides consistent behavior across all configs:
    - Typo detection via extra='forbid'
    - Validation on assignment
    - Proper JSON serialization
    
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
        str_strip_whitespace=True,   # Strip whitespace from strings
    )

