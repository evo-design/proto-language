"""
Base configuration class for all pydantic configs.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field as PydanticField


def ConfigField(
    default: Any = ...,
    *,
    title: str = None,
    description: str = None,
    advanced: bool = False,
    hidden: bool = False,
    **kwargs,
) -> Any:
    """
    Custom Field wrapper that automatically adds metadata flags to json_schema_extra.

    Args:
        advanced: If True, field appears in "Advanced" section of UI
        hidden: If True, field is hidden from UI completely

        **kwargs: All other standard Pydantic Field arguments

    Usage:
        param: int = Field(default=42, title="Param", description="...", advanced=True)
    """
    # Pull the existing json_schema_extra
    json_schema_extra = kwargs.get("json_schema_extra", {})

    # Add the advanced and hidden flags to the json_schema_extra
    json_schema_extra["advanced"] = advanced
    json_schema_extra["hidden"] = hidden

    # Update the kwargs with the new json_schema_extra
    kwargs["json_schema_extra"] = json_schema_extra

    return PydanticField(default, title=title, description=description, **kwargs)


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
