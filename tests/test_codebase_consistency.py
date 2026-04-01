"""consistency."""

from __future__ import annotations

import inspect
import types
from typing import Union, get_args, get_origin

import pytest
from proto_tools import BaseToolInput, BaseToolOutput, ToolRegistry
from proto_tools.utils import BaseConfig as ToolsBaseConfig

from proto_language.base_config import BaseConfig as LanguageBaseConfig
from proto_language.base_config import ConfigField
from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.generator import GeneratorRegistry
from proto_language.language.optimizer import OptimizerRegistry

# Defines the maximum length of a field title in characters
MAX_FIELD_TITLE_LENGTH = 31

# Defines the maximum length of a field description in characters
MAX_FIELD_DESCRIPTION_LENGTH = 100


def list_of_all_config_models() -> list[type]:
    """List of all config models of registered components."""
    return [
        spec.config_model
        for spec in [
            *ConstraintRegistry.list_all(),
            *GeneratorRegistry.list_all(),
            *OptimizerRegistry.list_all(),
            *ToolRegistry.list_all(),
        ]
    ]


@pytest.mark.parametrize("config_model", list_of_all_config_models())
def test_config_consistency(config_model: type):
    """Determines if config models are defined consistently throughout the codebase.

    for consistency of the API and client.
    """
    # Check if config_model is subclass of BaseConfig (language or tools)
    assert issubclass(config_model, (LanguageBaseConfig, ToolsBaseConfig)), (
        f"Config model {config_model} is not a subclass of BaseConfig"
    )

    # Pull the model schema and ensure fields are defined consistently
    schema = config_model.model_json_schema()
    required_fields = set(schema.get("required", []))

    # Pull the docstring for the config model
    docstring = config_model.__doc__
    assert docstring is not None, f"{config_model.__name__} is missing docstring. "
    assert len(docstring) > 0, f"{config_model.__name__} docstring is empty. "

    # Ensure all fields are defined consistently
    for field_name, field_info in config_model.model_fields.items():
        # TITLE: Ensure title is explicitly provided and is under 45 characters
        title = field_info.title
        assert title is not None, f"{config_model.__name__}.{field_name} is missing title. "
        assert len(title) <= MAX_FIELD_TITLE_LENGTH, (
            f"{config_model.__name__}.{field_name} title is too long (currently {len(title)} characters, must be under {MAX_FIELD_TITLE_LENGTH} characters). "
        )

        # DESCRIPTION: Must exist and be concise (~15 words / ~90 chars for tooltip)
        description_error = _field_description_is_valid(field_info.description)
        assert description_error == "", (
            f"{config_model.__name__}.{field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

        # OPTIONALITY: Check for Optional types (should be rare)
        # If the default value is None, the field must have annotation Optional[type]
        if field_info.default is None:
            annotation = field_info.annotation
            origin = get_origin(annotation)
            ann_args = get_args(annotation)

            # Optional[...] is Union[..., None]; X | None is types.UnionType
            is_optional = origin in (Union, types.UnionType) and type(None) in ann_args

            if not is_optional:
                raise TypeError(
                    f"{config_model.__name__}.{field_name} default value is None but annotation is not Optional[...]"
                )

        # ADVANCED FLAG: Must exist and be a boolean
        json_schema_extra = field_info.json_schema_extra or {}
        assert "advanced" in json_schema_extra, (
            f"{config_model.__name__}.{field_name} missing 'advanced' flag. "
            "Add: Field(..., json_schema_extra={{'advanced': False}})"
        )
        assert isinstance(json_schema_extra["advanced"], bool), (
            f"{config_model.__name__}.{field_name} 'advanced' flag is not a boolean. "
            "Add: Field(..., json_schema_extra={{'advanced': False}})"
        )

        assert "hidden" in json_schema_extra, (
            f"{config_model.__name__}.{field_name} missing 'hidden' flag. "
            "Add: Field(..., json_schema_extra={{'hidden': False}})"
        )
        assert isinstance(json_schema_extra["hidden"], bool), (
            f"{config_model.__name__}.{field_name} 'hidden' flag is not a boolean. "
            "Add: Field(..., json_schema_extra={{'hidden': False}})"
        )

        # Pull advanced and hidden flags
        advanced = json_schema_extra.get("advanced", False)
        hidden = json_schema_extra.get("hidden", False)

        # Advanced and hidden flags must be false if the field is required
        if field_name in required_fields:
            assert not advanced, (
                f"{config_model.__name__}.{field_name} 'advanced' flag cannot be True if the field is required. "
                "Remove the 'advanced' flag."
            )
            assert not hidden, (
                f"{config_model.__name__}.{field_name} 'hidden' flag cannot be True if the field is required. "
                "Remove the 'hidden' flag."
            )

    # DOCUMENTATION CHECK: Ensure that all fields are mentioned in the docstring
    # Exclude inherited BaseConfig fields (documented once in BaseConfig)
    # and don't need to be re-documented in every subclass.
    standard_base_config_fields = ("verbose", "device", "timeout", "tracking_interval", "track_proposals")
    missing_fields = _find_missing_fields_in_docstring(docstring, config_model.model_fields.keys())
    missing_fields = [f for f in missing_fields if f not in standard_base_config_fields]
    assert len(missing_fields) == 0, (
        f"{config_model.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Add: Field(..., description='Brief explanation for tooltip')"
    )


def list_tool_inputs_and_outputs() -> list[tuple[str, str]]:
    """List of all tool inputs and outputs."""
    return [(tool.input_model, tool.output_model) for tool in ToolRegistry.list_all()]


@pytest.mark.parametrize(
    "tool_input, tool_output",
    list_tool_inputs_and_outputs(),
)
def test_tool_input_and_output_consistency(tool_input: type, tool_output: type):
    """Test if tool inputs and outputs are defined consistently throughout the codebase.

    for consistency of the API and client.
    """
    # Ensure tool input inherits from BaseToolInput
    assert issubclass(tool_input, BaseToolInput), f"Tool input {tool_input} is not a subclass of BaseToolInput"
    # Ensure tool output inherits from BaseToolOutput
    assert issubclass(tool_output, BaseToolOutput), f"Tool output {tool_output} is not a subclass of BaseToolOutput"

    # Ensure docstring exists and is not empty for both tool input and output
    input_docstring = tool_input.__doc__
    assert input_docstring is not None, f"Tool input {tool_input.__name__} is missing docstring. "
    assert len(input_docstring) > 0, f"Tool input {tool_input.__name__} docstring is empty. "
    output_docstring = tool_output.__doc__
    assert output_docstring is not None, f"Tool output {tool_output.__name__} is missing docstring. "
    assert len(output_docstring) > 0, f"Tool output {tool_output.__name__} docstring is empty. "

    # Iterate through input fields and ensure they are defined consistently
    for field_name, field_info in tool_input.model_fields.items():
        description_error = _field_description_is_valid(field_info.description)
        assert description_error == "", (
            f"Tool input {tool_input.__name__} has field {field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

    # Iterate through output fields and ensure they are defined consistently
    for field_name, field_info in tool_output.model_fields.items():
        description_error = _field_description_is_valid(field_info.description)
        assert description_error == "", (
            f"Tool output {tool_output.__name__} has field {field_name} {description_error}. "
            "Ensure: Field(..., description='Brief explanation for tooltip')"
        )

    # DOCUMENTATION CHECK: Ensure that all fields are mentioned in the docstring
    missing_fields = _find_missing_fields_in_docstring(input_docstring, tool_input.model_fields.keys())
    assert len(missing_fields) == 0, (
        f"Tool input {tool_input.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Ensure: Field(..., description='Brief explanation for tooltip')"
    )
    missing_fields = _find_missing_fields_in_docstring(output_docstring, tool_output.model_fields.keys())
    # Remove standardized output fields
    standard_tool_output_fields = (
        "tool_id",
        "execution_time",
        "timestamp",
        "success",
        "warnings",
        "errors",
        "metadata",
    )
    missing_fields = [field for field in missing_fields if field not in standard_tool_output_fields]
    assert len(missing_fields) == 0, (
        f"Tool output {tool_output.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Ensure: Field(..., description='Brief explanation for tooltip')"
    )

    # Ensure tool output is concrete (all abstract methods implemented)
    assert not inspect.isabstract(tool_output), (
        f"Tool output {tool_output.__name__} is abstract. "
        f"Missing implementations for abstract methods: "
        f"{sorted(tool_output.__abstractmethods__)}"
    )


@pytest.mark.parametrize("config_model", list_of_all_config_models())
def test_depends_on_references_valid_field(config_model: type):
    """Every x-depends-on in a config schema must reference an existing sibling.

    Validates that all x-depends-on references point to existing, non-hidden
    sibling properties in the schema.
    """
    schema = config_model.model_json_schema()
    properties = schema.get("properties", {})

    for field_name, field_schema in properties.items():
        x_dep = field_schema.get("x-depends-on")
        if x_dep is None:
            continue

        ref_field = x_dep.get("field")
        assert ref_field is not None, f"{config_model.__name__}.{field_name}: x-depends-on is missing the 'field' key."

        assert ref_field in properties, (
            f"{config_model.__name__}.{field_name}: "
            f"x-depends-on references '{ref_field}' which does not exist "
            f"in the schema properties. "
            f"Available: {sorted(properties.keys())}"
        )

        ref_hidden = properties[ref_field].get("hidden", False)
        assert not ref_hidden, (
            f"{config_model.__name__}.{field_name}: "
            f"x-depends-on references '{ref_field}' which is hidden. "
            "A dependency on a hidden field is not visible to the user."
        )


# ---------------------------------------------------------------------------
# ConfigField depends_on serialization
# ---------------------------------------------------------------------------


class _DependsOnModel(LanguageBaseConfig):
    """Shared test model for depends_on parametrized tests."""

    mode: str = ConfigField(default="basic", title="Mode", description="Operating mode.")
    target: str | None = ConfigField(default=None, title="Target", description="Optional target.")
    enabled: bool = ConfigField(default=False, title="Enabled", description="Toggle.")
    with_value: str = ConfigField(
        default="off",
        title="With Value",
        description="Single value.",
        depends_on={"field": "mode", "value": "advanced"},
    )
    with_list: int = ConfigField(
        default=1,
        title="With List",
        description="List value.",
        depends_on={"field": "mode", "value": ["a", "b"]},
    )
    with_not_null: str = ConfigField(
        default="x",
        title="With Not Null",
        description="Not null.",
        depends_on={"field": "target", "not_null": True},
    )
    with_truthy: str = ConfigField(
        default="",
        title="With Truthy",
        description="Truthy check.",
        depends_on={"field": "enabled"},
    )
    plain: int = ConfigField(default=0, title="Plain", description="No depends_on.")


@pytest.mark.parametrize(
    "field_name, expected",
    [
        ("with_value", {"field": "mode", "value": "advanced"}),
        ("with_list", {"field": "mode", "value": ["a", "b"]}),
        ("with_not_null", {"field": "target", "not_null": True}),
        ("with_truthy", {"field": "enabled"}),
        ("plain", None),
    ],
)
def test_depends_on_schema_output(field_name: str, expected):
    """depends_on produces the correct x-depends-on in JSON schema."""
    schema = _DependsOnModel.model_json_schema()
    props = schema["properties"][field_name]
    if expected is None:
        assert "x-depends-on" not in props
    else:
        assert props["x-depends-on"] == expected


@pytest.mark.parametrize(
    "depends_on, match",
    [
        ({"value": "bar"}, "must include a 'field' key"),
        ({"field": "x", "value": "y", "not_null": True}, "cannot specify both"),
    ],
)
def test_depends_on_invalid_raises(depends_on, match):
    """Invalid depends_on dicts raise ValueError at class definition."""
    with pytest.raises(ValueError, match=match):

        class _Bad(LanguageBaseConfig):
            """Test model."""

            bad: str = ConfigField(
                default="",
                title="Bad",
                description="Should fail.",
                depends_on=depends_on,
            )


def _field_description_is_valid(description: str) -> str:
    """Check if the description is under MAX_FIELD_DESCRIPTION_LENGTH characters."""
    if description is None:
        return "is None"
    if len(description) > MAX_FIELD_DESCRIPTION_LENGTH:
        return f"is too long (currently {len(description)} characters, must be under {MAX_FIELD_DESCRIPTION_LENGTH} characters)"
    if not description.strip():
        return "description is empty or just whitespace"
    if "\n" in description:
        return "description contains newline characters. Please use single line descriptions."
    return ""


def _find_missing_fields_in_docstring(docstring: str, field_names: list[str]) -> list[str]:
    """Find missing fields in the docstring."""
    return [field_name for field_name in field_names if field_name not in docstring]
