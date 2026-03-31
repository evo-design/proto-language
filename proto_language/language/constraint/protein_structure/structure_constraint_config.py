"""
proto_language/language/constraint/protein_structure/structure_constraint_config.py

Base configuration classes for structure-based constraints.

This module provides standardized configuration classes for constraints that
use structure prediction tools (ESMFold, AlphaFold3, Boltz2, Chai1).
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from proto_tools import (
    AlphaFold3Config,
    Boltz2Config,
    Chai1Config,
    ESMFoldConfig,
)
from proto_tools.tools.structure_prediction.dispatch import SP_TOOL_MAP
from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField


class StructureBasedConstraintConfig(BaseConfig):
    """Base configuration for constraints using structure prediction tools.

    This base class standardizes how structure prediction tools and their
    configurations are specified across all structure-based constraints.

    Subclasses can optionally restrict which tools are supported by overriding
    the structure_tool field with a narrower Literal type.

    Attributes:
        structure_tool (Literal['esmfold', 'alphafold3', 'boltz2', 'chai1']): Tool to use for structure prediction. Supported options:
            - "esmfold": ESMFold (Meta AI)
            - "alphafold3": AlphaFold 3 (Google DeepMind)
            - "boltz2": Boltz2 (MIT)
            - "chai1": Chai-1 (Chai Discovery)
            Default is "esmfold".

        tool_config (dict[str, Any] | ESMFoldConfig | AlphaFold3Config | Boltz2Config | Chai1Config | None): Tool-specific configuration parameters. Can be provided as:
            - A typed config object (ESMFoldConfig, AlphaFold3Config, etc.)
            - A dictionary that will be automatically converted to the appropriate config type
            Default is an empty dictionary.

    Example:
        >>> # Using dict (will be converted to ESMFoldConfig)
        >>> config = MyConstraintConfig(
        ...     structure_tool="esmfold",
        ...     tool_config={"device": "cuda"}
        ... )
        >>>
        >>> # Using typed config
        >>> config = MyConstraintConfig(
        ...     structure_tool="esmfold",
        ...     tool_config=ESMFoldConfig(device="cuda")
        ... )

    Note:
        The tool_config is automatically validated to ensure it matches the
        selected structure_tool. If a dict is provided, it will be converted
        to the appropriate typed config class (with full Pydantic validation).
    """

    structure_tool: Literal["esmfold", "alphafold3", "boltz2", "chai1"] = ConfigField(
        title="Structure Prediction Tool",
        default="esmfold",
        description="Tool to use for structure prediction.",
    )

    tool_config: Optional[Union[
        Dict[str, Any],
        ESMFoldConfig,
        AlphaFold3Config,
        Boltz2Config,
        Chai1Config,
    ]] = ConfigField(
        title="Tool Configuration",
        default=None,
        description="Tool-specific configuration parameters. Can be a typed config, dict, or None (uses defaults).",
        advanced=True,
    )

    @model_validator(mode="before")
    @classmethod
    def convert_and_validate_tool_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Converts dict/None to tool-specific config and validates type consistency.

        Handles all possible input formats:
        - tool_config=None → default config for structure_tool
        - tool_config=dict → instantiate config for structure_tool
        - tool_config=<typed config> → validate it matches structure_tool

        Args:
            values (dict[str, Any]): Dict of tool configuration parameters.

        Returns:
            dict[str, Any]: Dict of tool configuration parameters with tool_config converted.

        Raises:
            ValueError: If structure_tool is unknown or tool_config type doesn't match.
        """
        if not isinstance(values, dict):
            return values

        structure_tool = values.get("structure_tool", "esmfold")
        tool_config = values.get("tool_config")

        # Validate structure_tool is known
        if structure_tool not in SP_TOOL_MAP:
            raise ValueError(
                f"Unknown structure prediction tool: '{structure_tool}'. "
                f"Supported tools: {', '.join(SP_TOOL_MAP.keys())}"
            )

        expected_type = SP_TOOL_MAP[structure_tool]["config"]

        # Convert dict or None to appropriate config object
        if tool_config is None:
            values["tool_config"] = expected_type()
        elif isinstance(tool_config, dict):
            values["tool_config"] = expected_type(**tool_config)
        # Validate that already-typed config matches the structure_tool
        elif not isinstance(tool_config, expected_type):
            raise ValueError(
                f"tool_config type {type(tool_config).__name__} doesn't match "
                f"structure_tool '{structure_tool}' (expected {expected_type.__name__})"
            )
        # else: tool_config is already the correct type, leave it as-is

        return values
