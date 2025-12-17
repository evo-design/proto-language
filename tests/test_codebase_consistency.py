"""
Pulls all codebase configs of registered components and checks for field definition
consistency.
"""
from __future__ import annotations
import pytest
from typing import Type, List, get_origin, get_args, Tuple, Union
from proto_language.base_config import BaseConfig
from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.generator import GeneratorRegistry
from proto_language.language.optimizer import OptimizerRegistry
from proto_language.tools.tool_registry import ToolRegistry
from proto_language.tools.tool_io import BaseToolInput, BaseToolOutput

# Defines the maximum length of a field title in characters
MAX_FIELD_TITLE_LENGTH = 31

# Defines the maximum length of a field description in characters
MAX_FIELD_DESCRIPTION_LENGTH = 100

def list_of_all_config_models() -> List[Type[BaseConfig]]:
    """
    List of all config models of registered components.
    """
    return [
        spec.config_model
        for spec in [
            *ConstraintRegistry.list_all(),
            *GeneratorRegistry.list_all(),
            *OptimizerRegistry.list_all(),
            *ToolRegistry.list_all(),
        ]
    ]

@pytest.mark.parametrize("config_model", [
    config_model for config_model in list_of_all_config_models()
])
def test_config_consistency(config_model: Type[BaseConfig]):
    """
    Determines if config models are defined consistently throughout the codebase
    for consistency of the API and client.
    """
    # Check if config_model is subclass of BaseModel
    assert issubclass(config_model, BaseConfig), f"Config model {config_model} is not a subclass of BaseConfig"

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
        assert (
            len(title) <= MAX_FIELD_TITLE_LENGTH
        ), f"{config_model.__name__}.{field_name} title is too long (currently {len(title)} characters, must be under {MAX_FIELD_TITLE_LENGTH} characters). "

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

            # Optional[...] is Union[..., None]
            is_optional = origin is Union and type(None) in ann_args

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
    missing_fields = _find_missing_fields_in_docstring(docstring, config_model.model_fields.keys())
    assert len(missing_fields) == 0, (
        f"{config_model.__name__} is missing the following fields in the docstring: {missing_fields}. "
        "Add: Field(..., description='Brief explanation for tooltip')"
    )


def list_tool_inputs_and_outputs() -> List[Tuple[str, str]]:
    """
    List of all tool inputs and outputs.
    """
    full_list = []
    for tool in ToolRegistry.list_all():
        full_list.append((tool.input_model, tool.output_model))
    return full_list


@pytest.mark.parametrize(
    "tool_input, tool_output",
    [(tool_input, tool_output) for (tool_input, tool_output) in list_tool_inputs_and_outputs()],
)
def test_tool_input_and_output_consistency(tool_input: type, tool_output: type):
    """
    Test if tool inputs and outputs are defined consistently throughout the codebase
    for consistency of the API and client.
    """

    # Ensure tool input inherits from BaseToolInput
    assert issubclass(
        tool_input, BaseToolInput
    ), f"Tool input {tool_input} is not a subclass of BaseToolInput"
    # Ensure tool output inherits from BaseToolOutput
    assert issubclass(
        tool_output, BaseToolOutput
    ), f"Tool output {tool_output} is not a subclass of BaseToolOutput"

    # Ensure docstring exists and is not empty for both tool input and output
    input_docstring = tool_input.__doc__
    assert input_docstring is not None, f"Tool input {tool_input.__name__} is missing docstring. "
    assert len(input_docstring) > 0, f"Tool input {tool_input.__name__} docstring is empty. "
    output_docstring = tool_output.__doc__
    assert (
        output_docstring is not None
    ), f"Tool output {tool_output.__name__} is missing docstring. "
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


def _field_description_is_valid(description: str) -> str:
    """
    Check if the description is under MAX_FIELD_DESCRIPTION_LENGTH characters.
    """
    if description is None:
        return "is None"
    if len(description) > MAX_FIELD_DESCRIPTION_LENGTH:
        return f"is too long (currently {len(description)} characters, must be under {MAX_FIELD_DESCRIPTION_LENGTH} characters)"
    if not description.strip():
        return "description is empty or just whitespace"
    if "\n" in description:
        return "description contains newline characters. Please use single line descriptions."
    return ""

def _find_missing_fields_in_docstring(docstring: str, field_names: List[str]) -> List[str]:
    """
    Find missing fields in the docstring.
    """
    missing_fields = []
    for field_name in field_names:
        if field_name not in docstring:
            missing_fields.append(field_name)
    return missing_fields
