"""
proto_language/base_config.py

Base configuration classes for all pydantic configs.
"""
from __future__ import annotations

from typing import Any, TypedDict, Union

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField


class DependsOn(TypedDict, total=False):
    """Declares that a field's UI visibility depends on a sibling field's value.

    Evaluation rules (for any consumer):
      - ``value`` present (single): show when ``parent[field] == value``
      - ``value`` present (list):   show when ``parent[field] in value``
      - ``not_null`` is True:       show when ``parent[field]`` is not None
      - Neither ``value`` nor ``not_null``: show when ``parent[field]`` is truthy

    Attributes:
        field (str): Name of the field this dependency targets.
        value (Union[str, int, float, bool, list]): Required value of the target field for this field to be visible.
        not_null (bool): If True, the dependency is satisfied when the target field is not None.
    """

    field: str  # Required: sibling field key to watch
    value: Union[str, int, float, bool, list]  # Optional: value(s) to match
    not_null: bool  # Optional: True means "show when not None"


def ConfigField(
    default: Any = ...,
    *,
    title: str = None,
    description: str = None,
    advanced: bool = False,
    hidden: bool = False,
    depends_on: DependsOn | None = None,
    **kwargs,
) -> Any:
    """
    Custom Field wrapper that automatically adds metadata flags to json_schema_extra.

    Args:
        default (Any): Default value for the configuration field.
        title (str): Human-readable display title for the field.
        description (str): Short description shown in the client UI.
        advanced (bool): If True, field appears in "Advanced" section of UI.
        hidden (bool): If True, field is hidden from UI completely.
        depends_on (DependsOn | None): If set, field is only visible when the
            sibling field identified by ``depends_on["field"]`` satisfies the
            condition. See :class:`DependsOn` for evaluation rules.
        kwargs: All other standard Pydantic Field arguments (passed through
            to ``pydantic.Field``).

    Usage:
        param: int = ConfigField(default=42, title="Param", advanced=True)
        nested: Optional[Config] = ConfigField(
            default=None,
            depends_on={"field": "mode", "value": "advanced"},
        )
    """
    json_schema_extra = kwargs.get("json_schema_extra", {})

    json_schema_extra["advanced"] = advanced
    json_schema_extra["hidden"] = hidden

    if depends_on is not None:
        if "field" not in depends_on:
            raise ValueError("depends_on must include a 'field' key")
        if "value" in depends_on and "not_null" in depends_on:
            raise ValueError(
                "depends_on cannot specify both 'value' and 'not_null'"
            )
        json_schema_extra["x-depends-on"] = depends_on

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
        extra='ignore',              # Ignore unknown fields
        validate_assignment=True,    # Validate on field updates
        use_enum_values=True,        # Serialize enums as values
        validate_default=True,       # Validate default values
    )


# ---------------------------------------------------------------------------
# Optimizer configs
# ---------------------------------------------------------------------------


class BaseOptimizerConfig(BaseConfig):
    """Shared base config for all optimizers."""
    tracking_interval: int = ConfigField(
        default=1,
        ge=1,
        title="Tracking Interval",
        description="Save history and log progress every N steps. Step 0 and final step always saved.",
        advanced=True,
    )
    track_proposals: bool = ConfigField(
        default=False,
        title="Track Proposals",
        description="Save granular per-proposal results (accept/reject) in history snapshots.",
        advanced=True,
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
        hidden=True,
    )
