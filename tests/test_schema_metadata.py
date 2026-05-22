"""tests/test_schema_metadata.py.

For every registered constraint / generator / optimizer spec, walk the JSON
Schema of the spec itself and of its ``config_model`` (including every
nested ``$defs`` model) and assert each property has a non-empty ``title``
and ``description``.

Complements ``test_codebase_consistency.test_config_consistency``, which
checks ``model_fields`` on each ``config_model`` directly. This test walks
the resolved JSON Schema and follows ``$ref`` into shared submodels, so
nested data classes (e.g. ``InputSlot``) get the same enforcement.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from proto_language.constraint import ConstraintRegistry
from proto_language.generator import GeneratorRegistry
from proto_language.optimizer import OptimizerRegistry

_MAX_FIELD_TITLE_LENGTH = 31
_MAX_FIELD_DESCRIPTION_LENGTH = 100


def _walk_schema(schema: dict[str, Any], path: str) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(path, prop_name, prop_dict)`` for every property in a JSON schema and its ``$defs``."""
    for name, prop in schema.get("properties", {}).items():
        yield path, name, prop
    for defname, defschema in schema.get("$defs", {}).items():
        for name, prop in defschema.get("properties", {}).items():
            yield f"{path}::$defs::{defname}", name, prop


def _check_property(model_label: str, path: str, name: str, prop: dict[str, Any]) -> list[str]:
    """Return a list of error messages describing missing/invalid metadata on ``prop``."""
    errors: list[str] = []
    title = prop.get("title")
    if not title:
        errors.append(f"{model_label} {path}.{name}: missing title")
    elif len(title) > _MAX_FIELD_TITLE_LENGTH:
        errors.append(
            f"{model_label} {path}.{name}: title is too long ({len(title)} chars, must be ≤ {_MAX_FIELD_TITLE_LENGTH})"
        )
    description = prop.get("description")
    if not description:
        errors.append(f"{model_label} {path}.{name}: missing description")
    elif len(description) > _MAX_FIELD_DESCRIPTION_LENGTH:
        errors.append(
            f"{model_label} {path}.{name}: description is too long "
            f"({len(description)} chars, must be ≤ {_MAX_FIELD_DESCRIPTION_LENGTH})"
        )
    elif "\n" in description:
        errors.append(f"{model_label} {path}.{name}: description contains a newline")
    return errors


def _all_specs() -> list:
    """Return every registered spec across the three proto-language registries."""
    return [
        *ConstraintRegistry.list_all(),
        *GeneratorRegistry.list_all(),
        *OptimizerRegistry.list_all(),
    ]


_SPECS = _all_specs()


@pytest.mark.parametrize("spec", _SPECS, ids=[s.key for s in _SPECS])
def test_spec_schema_has_title_and_description(spec):
    """Every property in the spec + config_model JSON Schemas must have title + description."""
    errors: list[str] = []

    spec_schema = type(spec).model_json_schema()
    for path, name, prop in _walk_schema(spec_schema, "spec"):
        errors.extend(_check_property(type(spec).__name__, path, name, prop))

    config_schema = spec.config_model.model_json_schema()
    for path, name, prop in _walk_schema(config_schema, "config"):
        errors.extend(_check_property(spec.config_model.__name__, path, name, prop))

    if errors:
        message = f"Spec {spec.key!r} has {len(errors)} field-metadata violation(s):\n  " + "\n  ".join(errors)
        message += (
            "\n\nFix: add title= and description= on every Pydantic field exposed via the JSON Schema. "
            "Use ConfigField for direct Config classes; bare pydantic.Field(title=..., description=...) "
            "is acceptable on shared nested submodels."
        )
        pytest.fail(message)
