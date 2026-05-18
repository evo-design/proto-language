"""Base configuration classes for all pydantic configs."""

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

_REMOVED_UI_KWARGS = frozenset({"advanced", "hidden", "depends_on"})


def ConfigField(
    default: Any = ...,
    *,
    title: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Any:
    """Custom Field wrapper for proto-language configs.

    Thin alias over ``pydantic.Field`` preserved so call sites keep using
    ``ConfigField(...)`` consistently across constraints, generators, and
    optimizers. UI-presentation hints (``advanced``/``hidden``/``depends_on``)
    live in client's overlay infra (``src/data/ui-overlays/``) and are no
    longer carried on the schema.

    Args:
        default (Any): Default value for the configuration field.
        title (str | None): Human-readable display title for the field.
        description (str | None): Short description shown as a UI tooltip.
        kwargs: All other standard Pydantic Field arguments (passed through
            to ``pydantic.Field``).

    Usage:
        param: int = ConfigField(default=42, title="Param", ge=0)
    """
    removed_kwargs = _REMOVED_UI_KWARGS & kwargs.keys()
    if removed_kwargs:
        removed = ", ".join(sorted(removed_kwargs))
        raise TypeError(f"ConfigField no longer accepts UI-presentation kwargs: {removed}")

    return PydanticField(default, title=title, description=description, **kwargs)


class BaseConfig(BaseModel):
    """Base configuration class for consistent behavior across all configs (tools, constraints, and generators).

    Example:
        >>> class MyToolConfig(BaseConfig):
        ...     param1: int
        ...     param2: str
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True,  # Validate on field updates
        use_enum_values=True,  # Serialize enums as values
        validate_default=True,  # Validate default values
    )


# ---------------------------------------------------------------------------
# Optimizer configs
# ---------------------------------------------------------------------------


class BaseOptimizerConfig(BaseConfig):
    """Shared base config for all optimizers.

    Optimizer instances single-source their effective ``seed`` from this config.
    Program-level seeds overwrite this field with optimizer-specific child
    seeds during program initialization.
    """

    seed: int | None = ConfigField(
        default=None,
        title="Random Seed",
        description="Random seed for reproducible optimization, generator, and constraint tool streams.",
        ge=0,
    )
    tracking_interval: int = ConfigField(
        default=1,
        ge=1,
        title="Tracking Interval",
        description="Save history and log progress every N steps. Step 0 and final step always saved.",
    )
    track_proposals: bool = ConfigField(
        default=False,
        title="Track Proposals",
        description="Save granular per-proposal results (accept/reject) in history snapshots.",
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Whether to print progress information.",
    )
