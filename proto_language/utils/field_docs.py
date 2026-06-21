"""Per-field documentation extracted from config-model docstrings.

Config models document each field twice: a terse ``ConfigField(description=...)``
for compact display, and a much richer Google-style ``Attributes:`` entry in the
class docstring (enforced by ``tests/test_docstring_consistency.py``). Only the
terse description was ever exposed programmatically. This module parses the full
``Attributes:`` text and injects it into JSON schemas under the ``x-proto-doc``
extension key (matching the existing ``x-proto-*`` convention), so consumers can
render help richer than the one-liner.
"""

import inspect
import logging
from typing import Any, get_args

from docstring_parser import DocstringStyle
from docstring_parser import parse as parse_docstring
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def field_docs_from_docstrings(model_class: type[BaseModel]) -> dict[str, str]:
    """Map each field of ``model_class`` to its full docstring description.

    Walks the MRO most-derived first and parses each class's own Google-style
    ``Attributes:`` section, but only trusts a class's text for the fields it
    actually declares (those in its own ``__annotations__``). So an inherited
    field resolves to its defining model's description rather than a subclass's
    "Inherited from X" re-mention, while a subclass that redefines a field
    (declaring it anew) still overrides the parent's text.

    Args:
        model_class (type[BaseModel]): A Pydantic model whose docstrings document
            fields in a Google-style ``Attributes:`` section.

    Returns:
        dict[str, str]: Field name to docstring description, restricted to the
            model's own fields.
    """
    field_docs: dict[str, str] = {}
    for cls in model_class.__mro__:
        own_doc = cls.__doc__
        if not own_doc:
            continue
        # Fields declared in this class's own body (never inherited), so an
        # inherited field's text comes from its defining class, not a re-mention.
        own_fields = cls.__dict__.get("__annotations__", {})
        try:
            parsed = parse_docstring(inspect.cleandoc(own_doc), style=DocstringStyle.GOOGLE)
        except Exception:
            logger.debug("Could not parse docstring for %s", cls.__name__, exc_info=True)
            continue
        for param in parsed.params:
            name = param.arg_name
            if name.startswith("*") or not param.description or name not in own_fields:
                continue
            field_docs.setdefault(name, param.description.strip())
    return {name: doc for name, doc in field_docs.items() if name in model_class.model_fields}


def _models_in_annotation(annotation: Any) -> list[type[BaseModel]]:
    """Return the ``BaseModel`` subclasses nested anywhere in a type annotation.

    Args:
        annotation (Any): A type annotation (possibly generic, e.g. ``list[X]``).

    Returns:
        list[type[BaseModel]]: Model classes found at any nesting depth.
    """
    models: list[type[BaseModel]] = []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        models.append(annotation)
    for arg in get_args(annotation):
        models.extend(_models_in_annotation(arg))
    return models


def _collect_nested_models(model_class: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Map ``$defs`` names to the nested ``BaseModel`` classes reachable from fields.

    Args:
        model_class (type[BaseModel]): The model whose fields are walked.

    Returns:
        dict[str, type[BaseModel]]: Class name to model class for every nested model.
    """
    found: dict[str, type[BaseModel]] = {}

    def _walk(cls: type[BaseModel]) -> None:
        for info in cls.model_fields.values():
            for nested in _models_in_annotation(info.annotation):
                if nested.__name__ not in found:
                    found[nested.__name__] = nested
                    _walk(nested)

    _walk(model_class)
    return found


def _apply_field_docs(schema: dict[str, Any], model_class: type[BaseModel]) -> None:
    """Set ``x-proto-doc`` on each property of ``schema`` from ``model_class`` docstrings.

    Args:
        schema (dict[str, Any]): A JSON Schema object with a ``properties`` map.
        model_class (type[BaseModel]): The model the properties were generated from.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    docs = field_docs_from_docstrings(model_class)
    for name, prop in properties.items():
        doc = docs.get(name)
        if doc and isinstance(prop, dict):
            prop["x-proto-doc"] = doc


def inject_field_docs(schema: dict[str, Any], model_class: type[BaseModel]) -> dict[str, Any]:
    """Add ``x-proto-doc`` per-field documentation to a model's JSON schema, in place.

    Each property gets an ``x-proto-doc`` key carrying the field's full
    Google-style docstring description, so consumers can render help richer than
    the terse ``description``. Nested model definitions under ``$defs`` are
    annotated too, matching how clients recurse into nested objects.

    Args:
        schema (dict[str, Any]): Output of ``model_class.model_json_schema()``.
        model_class (type[BaseModel]): The model the schema was generated from.

    Returns:
        dict[str, Any]: The same ``schema`` dict, mutated in place.
    """
    _apply_field_docs(schema, model_class)
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        nested = _collect_nested_models(model_class)
        for name, def_schema in defs.items():
            nested_cls = nested.get(name)
            if nested_cls is not None and isinstance(def_schema, dict):
                _apply_field_docs(def_schema, nested_cls)
    return schema
