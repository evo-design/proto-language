"""Command-line entry point for proto-language discovery and component docs.

Reachable as the ``proto-language`` shell command after ``pip install``, or as
``python -m proto_language`` without one. Verbs mirror proto-tools where the
shape matches and split by registry kind where it does not.

Defaults to human-readable text output; most verbs that return structured
data accept ``--json`` for machine-readable output via Pydantic
``model_dump_json()`` (the ``schema`` verb always emits JSON Schema).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Protocol

from pydantic import BaseModel

from proto_language.constraint.constraint_registry import ConstraintRegistry, ConstraintSpec
from proto_language.generator.generator_registry import GeneratorRegistry, GeneratorSpec
from proto_language.optimizer.optimizer_registry import OptimizerRegistry
from proto_language.utils.base import BaseSpec
from proto_language.utils.docs_api import (
    _CORE_TYPES,
    ComponentDoc,
    ComponentKind,
    ConfigModelDoc,
    get_compatibility,
    get_core_type_doc,
    list_categories,
    list_specs,
)


class _DocRegistry(Protocol):
    """Subset of registry surface the CLI calls."""

    def get(self, key: str) -> Any: ...
    def get_docs(self, identifier: str) -> ComponentDoc: ...
    def get_config_doc(self, identifier: str) -> ConfigModelDoc: ...


logger = logging.getLogger(__name__)


_KIND_TO_REGISTRY: dict[ComponentKind, _DocRegistry] = {
    "constraint": ConstraintRegistry,
    "generator": GeneratorRegistry,
    "optimizer": OptimizerRegistry,
}


# =============================================================================
# Output helpers
# =============================================================================


def _dump_json(value: Any) -> str:
    """Render any value as pretty-printed JSON."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, list) and value and isinstance(value[0], BaseModel):
        return json.dumps([v.model_dump() for v in value], indent=2, default=str)
    return json.dumps(value, indent=2, default=str)


def _strip_duplicate_lead(docstring: str, description: str) -> str:
    """Drop the docstring's first line when it duplicates ``spec.description``.

    Spec descriptions are almost always near-copies of the docstring's opening
    line. Skipping the duplicate keeps text-mode docs readable without losing
    the docstring's richer body. Comparison is whitespace- and trailing-
    period-insensitive.
    """
    if not docstring or not description:
        return docstring
    first, _, rest = docstring.partition("\n")
    if first.rstrip(".").strip() == description.rstrip(".").strip():
        return rest.lstrip("\n")
    return docstring


def _tool_summary(key: str) -> dict[str, Any]:
    """One-line summary of a proto-tools key: label, category, short description.

    Falls back to ``{"key": key, "error": ...}`` when proto-tools can't resolve
    the key so one bad key in ``tools_called`` doesn't sink the call.
    """
    from proto_tools import ToolRegistry  # lazy: keeps `--help` cheap and decouples import order.

    try:
        spec = ToolRegistry.get(key)
    except Exception as e:
        return {"key": key, "error": f"{type(e).__name__}: {e}"}
    return {
        "key": spec.key,
        "label": spec.label,
        "category": getattr(spec, "category", "") or "",
        "description": spec.description,
    }


def _tool_doc_payload(key: str) -> dict[str, Any]:
    """Return the proto-tools docs entry for ``key`` as a JSON-friendly dict."""
    from proto_tools import ToolRegistry

    try:
        entry = ToolRegistry.get_tool_docs(key)
    except Exception as e:
        return {"key": key, "error": f"{type(e).__name__}: {e}"}
    if entry is None:
        return {"key": key, "error": "no README entry"}
    payload: dict[str, Any] = json.loads(entry.model_dump_json())
    return payload


def _render_tool_section(tool_keys: list[str], full_keys: set[str]) -> None:
    """Print the trailing "Tools" section: summaries by default, full README for ``full_keys``.

    Header surfaces a shared category when all listed tools share one (the
    runtime-dispatched structure-predictor pattern), so readers see "8
    interchangeable structure_prediction predictors" instead of guessing why one
    constraint lists eight tools.
    """
    if not tool_keys:
        return

    summaries = [_tool_summary(k) for k in tool_keys]
    categories = {s["category"] for s in summaries if "error" not in s and s["category"]}
    shared = next(iter(categories)) if len(categories) == 1 and len(tool_keys) > 1 else None

    print()
    header = f"### Tools ({len(tool_keys)})"
    if shared:
        header = f"### Tools ({len(tool_keys)}, all `{shared}`)"
    print(header)
    print()

    for s in summaries:
        if "error" in s:
            print(f"  {s['key']:30s}  ⚠ {s['error']}")
            continue
        cat_str = f"  [{s['category']}]" if s["category"] and not shared else ""
        print(f"  {s['key']:30s}  {s['label']}{cat_str}")
        if s.get("description"):
            print(f"  {'':30s}  {s['description']}")

    if full_keys:
        for key in tool_keys:
            if key in full_keys:
                _render_full_tool_docs(key)
    elif len(tool_keys) > 1:
        print()
        print("  (use `--with-tool=<key>` for one tool's full docs, or `--with-tools-full` for all)")


def _render_full_tool_docs(key: str) -> None:
    """Append a full ``proto-tools docs <key>``-style block after the summary table."""
    from proto_tools import ToolRegistry

    print()
    try:
        entry = ToolRegistry.get_tool_docs(key)
    except Exception as e:
        print(f"### Tool docs unavailable for `{key}`: {type(e).__name__}: {e}")
        return
    if entry is None:
        print(f"### Tool docs unavailable for `{key}`: no README entry")
        return
    print(f"## Tool: {entry.label} (`{entry.key}`)\n")
    print(entry.intro)
    if entry.applications:
        print("\n### Applications\n")
        print(entry.applications)
    if entry.usage_tips:
        print("\n### Usage Tips\n")
        print(entry.usage_tips)
    if entry.toolkit_notes:
        print("\n### Toolkit Notes\n")
        print(entry.toolkit_notes)
    if entry.license:
        print("\n### License\n")
        print(_dump_json(entry.license))


def _spec_summary(spec: BaseSpec) -> str:
    """One-line text summary of a spec for list-style output."""
    gpu = " (GPU)" if spec.uses_gpu else ""
    category = getattr(spec, "category", None) or ""
    cat_str = f"  [{category}]" if category else ""
    return f"{spec.key:40s}{cat_str}{gpu}  {spec.description}"


def _filter_specs(
    specs: list[BaseSpec],
    *,
    category: str | None,
    gpu: bool,
    cpu: bool,
    mode: str | None,
    input_type: str | None,
) -> list[BaseSpec]:
    """Apply common CLI filters across all kinds; unsupported filters are no-ops."""
    out = list(specs)
    if category:
        wanted = category.replace(" ", "_").lower()
        out = [s for s in out if str(getattr(s, "category", "") or "").lower() == wanted]
    if gpu:
        out = [s for s in out if s.uses_gpu]
    if cpu:
        out = [s for s in out if not s.uses_gpu]
    if mode is not None:
        out = [s for s in out if isinstance(s, ConstraintSpec) and s.mode == mode]
    if input_type is not None:
        out = [s for s in out if isinstance(s, GeneratorSpec) and s.input_type.value == input_type]
    return out


# =============================================================================
# Verb handlers
# =============================================================================


def _cmd_list(args: argparse.Namespace) -> int:
    """``proto-language <kind> list [filters]`` or ``proto-language list``."""
    if args.kind == "all":
        kinds: list[ComponentKind] = ["constraint", "generator", "optimizer"]
    else:
        kinds = [args.kind]

    if args.json:
        out: dict[str, list[dict[str, Any]]] = {}
        for k in kinds:
            specs = _filter_specs(
                list_specs(k),
                category=args.category,
                gpu=args.gpu,
                cpu=args.cpu,
                mode=getattr(args, "mode", None),
                input_type=getattr(args, "input_type", None),
            )
            out[k] = [s.model_dump(mode="json") for s in specs]
        print(_dump_json(out if len(kinds) > 1 else out[kinds[0]]))
        return 0

    for k in kinds:
        specs = _filter_specs(
            list_specs(k),
            category=args.category,
            gpu=args.gpu,
            cpu=args.cpu,
            mode=getattr(args, "mode", None),
            input_type=getattr(args, "input_type", None),
        )
        if len(kinds) > 1:
            print(f"\n## {k}s ({len(specs)})")
        for spec in specs:
            print(_spec_summary(spec))
        if not specs and args.category:
            cats = list_categories(k)
            # Mirror _filter_specs's normalization (spaces->underscores, lowercase).
            wanted = args.category.replace(" ", "_").lower()
            hits = [c for c in cats if wanted in c.lower()]
            if hits:
                print(f"# no matches for --category {args.category!r}; did you mean: {', '.join(hits)}?")
            elif cats:
                print(f"# no matches for --category {args.category!r}; available: {', '.join(cats)}")
    return 0


def _cmd_categories(args: argparse.Namespace) -> int:
    """``proto-language <kind> categories``."""
    cats = list_categories(args.kind)
    if args.json:
        print(_dump_json(cats))
    else:
        for c in cats:
            print(c)
    return 0


def _cmd_docs(args: argparse.Namespace) -> int:
    """``proto-language <kind> docs <name>``."""
    registry = _KIND_TO_REGISTRY[args.kind]
    doc = registry.get_docs(args.name)
    tool_keys: list[str] = []
    if not getattr(args, "no_tools", False):
        tool_keys = list(getattr(doc.spec_metadata, "tools_called", []) or [])
    explicit_full = set(getattr(args, "with_tool", None) or [])
    full_keys: set[str] = set(tool_keys) if getattr(args, "with_tools_full", False) else explicit_full
    if args.json:
        payload: dict[str, Any] = json.loads(doc.model_dump_json())
        if tool_keys:
            payload["tool_summaries"] = [_tool_summary(k) for k in tool_keys]
        if full_keys:
            payload["tool_docs"] = [_tool_doc_payload(k) for k in tool_keys if k in full_keys]
        print(_dump_json(payload))
        return 0

    print(f"## {doc.label}  (`{doc.key}`, {doc.kind})")
    gpu = " (requires GPU)" if doc.uses_gpu else ""
    print(f"{doc.description}{gpu}\n")

    spec_meta = doc.spec_metadata.model_dump()
    if spec_meta:
        print("### Spec")
        _cross_ref_hints = {
            "tools_called": "(inspect via `proto-tools docs <key>`)",
            "requires_generators": "(inspect via `proto-language generator docs <key>`)",
        }
        for k, v in spec_meta.items():
            if v is None or v == [] or v == "":
                continue
            print(f"  {k:30s}  {v}")
            hint = _cross_ref_hints.get(k)
            if hint and v:
                print(f"  {'':30s}  {hint}")
        print()

    body = _strip_duplicate_lead(doc.docstring, doc.description)
    if body:
        print("### Description\n")
        print(body)
        print()

    print(f"### Config: {doc.config.name}\n")
    if doc.config.docstring:
        print(doc.config.docstring)
        print()
    for f in doc.config.fields:
        marker = "required" if f.required else f"default={f.default!r}"
        print(f"  {f.name:24s}  {f.type_str:30s}  ({marker})")
        if f.title:
            print(f"  {'':24s}  title: {f.title}")
        if f.description:
            print(f"  {'':24s}  {f.description}")

    _render_tool_section(tool_keys, full_keys)
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    """``proto-language <kind> config <name>``."""
    registry = _KIND_TO_REGISTRY[args.kind]
    cfg = registry.get_config_doc(args.name)
    if args.json:
        print(_dump_json(cfg))
        return 0
    print(f"Config: {cfg.name}\n")
    if cfg.docstring:
        print(cfg.docstring)
        print()
    for f in cfg.fields:
        marker = "required" if f.required else f"default={f.default!r}"
        print(f"  {f.name:24s}  {f.type_str:30s}  ({marker})")
        if f.description:
            print(f"  {'':24s}  {f.description}")
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    """``proto-language <kind> schema <name>``."""
    from proto_language.utils.docs_api import resolve_key

    registry = _KIND_TO_REGISTRY[args.kind]
    spec = registry.get(resolve_key(args.kind, args.name))
    print(json.dumps(spec.config_model.model_json_schema(), indent=2, default=str))
    return 0


def _cmd_compatible(args: argparse.Namespace) -> int:
    """``proto-language <kind> compatible <name>``."""
    report = get_compatibility(args.kind, args.name)
    if args.json:
        print(_dump_json(report))
        return 0
    print(f"## Compatible with {report.kind} `{report.key}`\n")
    sections = [
        ("Constraints", report.compatible_constraints, "proto-language constraint docs"),
        ("Generators", report.compatible_generators, "proto-language generator docs"),
        ("Optimizers", report.compatible_optimizers, "proto-language optimizer docs"),
    ]
    for title, items, inspect_cmd in sections:
        if not items:
            continue
        print(f"### {title} ({len(items)})")
        for k in items:
            print(f"  {k}")
        print(f"  ({inspect_cmd} <key>)")
        print()
    return 0


def _cmd_types(args: argparse.Namespace) -> int:
    """``proto-language types <name>`` or ``proto-language types`` to list."""
    if args.name is None:
        for n in _CORE_TYPES:
            print(n)
        return 0
    doc = get_core_type_doc(args.name)
    if args.json:
        print(_dump_json(doc))
        return 0
    print(f"## {doc.name}\n")
    if doc.docstring:
        print(doc.docstring)
        print()
    if doc.init_docstring:
        print("### __init__\n")
        print(doc.init_docstring)
        print()
    if doc.params:
        print("### Parameters\n")
        for p in doc.params:
            marker = "required" if p.required else f"default={p.default!r}"
            print(f"  {p.name:24s}  {p.type_str:30s}  ({marker})")
    return 0


# =============================================================================
# Argparse wiring
# =============================================================================


def _add_list_filters(parser: argparse.ArgumentParser, kind: ComponentKind | None) -> None:
    """Attach the common ``list`` filters; only the relevant ones per kind."""
    filt = parser.add_mutually_exclusive_group()
    filt.add_argument("--gpu", action="store_true", help="Only components that require a GPU.")
    filt.add_argument("--cpu", action="store_true", help="Only components that do not require a GPU.")
    parser.add_argument("--category", help="Filter by category string.")
    if kind == "constraint":
        parser.add_argument(
            "--mode",
            choices=["discrete", "gradient", "dual"],
            help="Filter constraints by mode.",
        )
    if kind == "generator":
        parser.add_argument(
            "--input-type",
            dest="input_type",
            choices=["prompt", "starting_sequence", "structure", "logits"],
            help="Filter generators by input type.",
        )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")


def _add_kind_subparsers(sub: argparse._SubParsersAction[argparse.ArgumentParser], kind: ComponentKind) -> None:
    """Wire ``list``/``categories``/``docs``/``config``/``schema``/``compatible`` under one kind."""
    kp = sub.add_parser(kind, help=f"{kind.capitalize()} discovery and docs.")
    kverbs = kp.add_subparsers(dest="verb", required=True)

    p_list = kverbs.add_parser("list", help=f"List registered {kind}s.")
    _add_list_filters(p_list, kind)
    p_list.set_defaults(func=_cmd_list, kind=kind)

    p_cats = kverbs.add_parser("categories", help=f"Categories present in the {kind} registry.")
    p_cats.add_argument("--json", action="store_true")
    p_cats.set_defaults(func=_cmd_categories, kind=kind)

    p_docs = kverbs.add_parser("docs", help=f"Full docs for one {kind} (docstring + config).")
    p_docs.add_argument("name", help=f"{kind.capitalize()} identifier (registry key or class/function name).")
    p_docs.add_argument("--json", action="store_true")
    p_docs.add_argument(
        "--with-tool",
        action="append",
        metavar="KEY",
        help="Inline full `proto-tools docs <KEY>` for the given tool. Repeatable.",
    )
    tools_group = p_docs.add_mutually_exclusive_group()
    tools_group.add_argument(
        "--with-tools-full",
        action="store_true",
        help="Inline full `proto-tools docs` for every key in tools_called.",
    )
    tools_group.add_argument(
        "--no-tools",
        action="store_true",
        help="Suppress the default per-tool summary section.",
    )
    p_docs.set_defaults(func=_cmd_docs, kind=kind)

    p_cfg = kverbs.add_parser("config", help=f"Config-model docs for one {kind}.")
    p_cfg.add_argument("name")
    p_cfg.add_argument("--json", action="store_true")
    p_cfg.set_defaults(func=_cmd_config, kind=kind)

    p_schema = kverbs.add_parser("schema", help=f"JSON Schema for one {kind}'s config model.")
    p_schema.add_argument("name")
    p_schema.set_defaults(func=_cmd_schema, kind=kind)

    p_compat = kverbs.add_parser(
        "compatible",
        help=f"Components pairable with this {kind} under the spec's compatibility rules.",
    )
    p_compat.add_argument("name")
    p_compat.add_argument("--json", action="store_true")
    p_compat.set_defaults(func=_cmd_compatible, kind=kind)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proto-language",
        description=(
            "Discover and inspect proto-language constraints, generators, optimizers, "
            "and core types. Verbs return text by default; pass --json for structured output."
        ),
    )
    sub = parser.add_subparsers(dest="kind", required=True)

    for kind in ("constraint", "generator", "optimizer"):
        _add_kind_subparsers(sub, kind)

    p_all = sub.add_parser("list", help="List components across all three registries.")
    _add_list_filters(p_all, kind=None)
    p_all.set_defaults(func=_cmd_list, kind="all")

    p_types = sub.add_parser(
        "types",
        help="Docs for a core type (Sequence | Segment | Construct | Program). Omit name to list.",
    )
    p_types.add_argument("name", nargs="?", choices=sorted(_CORE_TYPES), help="Core type name.")
    p_types.add_argument("--json", action="store_true")
    p_types.set_defaults(func=_cmd_types)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        # Unknown registry keys and core-type names all surface as ValueError.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
