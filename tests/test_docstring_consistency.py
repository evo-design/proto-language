"""tests/test_docstring_consistency.py

Tests that docstrings across the proto-language codebase follow the agreed style.
"""
from __future__ import annotations

import ast
import re
from fnmatch import fnmatch
from pathlib import Path

import pytest
from docstring_parser import DocstringStyle
from docstring_parser import parse as parse_docstring

_REPO_ROOT = Path(__file__).resolve().parents[1]

_ALL_DIRS = ["proto_language", "tests"]
_SOURCE_DIRS = ["proto_language"]

_EXCLUDE_PATTERNS = [
    # Git submodule (has its own tests)
    "proto-tools/*",
]

# Directory basenames to skip when walking the tree (venvs, caches, etc.)
_SKIP_DIRS = {".venv", ".venvs", "venv", "site-packages", "node_modules", "__pycache__"}


# ── Type normalization ──────────────────────────────────────────────────────

_TYPING_TO_BUILTIN = {
    "List": "list",
    "Dict": "dict",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Type": "type",
}


def _normalize_type(type_string: str) -> str:
    """Normalize a type string to canonical modern Python form."""
    if not type_string or not type_string.strip():
        return type_string

    t = type_string.strip()

    # Strip surrounding quotes from forward references: 'ClassName' -> ClassName
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t = t[1:-1]

    t = re.sub(r"\btyping\.", "", t)

    try:
        tree = ast.parse(t, mode="eval")
        new_tree = _normalize_ast(tree.body)
        return ast.unparse(new_tree)
    except (SyntaxError, ValueError):
        for old, new in _TYPING_TO_BUILTIN.items():
            t = re.sub(rf"\b{old}\b", new, t)
        return re.sub(r"\s+", " ", t).strip()


def _normalize_ast(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Name):
        if node.id in _TYPING_TO_BUILTIN:
            return ast.Name(id=_TYPING_TO_BUILTIN[node.id], ctx=ast.Load())
        return node

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "typing":
            return _normalize_ast(ast.Name(id=node.attr, ctx=ast.Load()))
        return node

    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "Optional":
            inner = _normalize_ast(node.slice)
            return ast.BinOp(left=inner, op=ast.BitOr(), right=ast.Constant(value=None))

        if isinstance(node.value, ast.Name) and node.value.id == "Union":
            if isinstance(node.slice, ast.Tuple):
                elements = [_normalize_ast(e) for e in node.slice.elts]
            else:
                elements = [_normalize_ast(node.slice)]
            return _make_bitor_chain(elements)

        new_value = _normalize_ast(node.value)
        if isinstance(node.slice, ast.Tuple):
            new_slice = ast.Tuple(
                elts=[_normalize_ast(e) for e in node.slice.elts], ctx=ast.Load()
            )
        else:
            new_slice = _normalize_ast(node.slice)
        return ast.Subscript(value=new_value, slice=new_slice, ctx=ast.Load())

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _normalize_ast(node.left)
        right = _normalize_ast(node.right)
        elements = _flatten_bitor(left) + _flatten_bitor(right)
        return _make_bitor_chain(elements)

    if isinstance(node, (ast.Constant, ast.List, ast.Tuple)):
        if isinstance(node, (ast.List, ast.Tuple)):
            return type(node)(
                elts=[_normalize_ast(e) for e in node.elts], ctx=ast.Load()
            )
        return node

    return node


def _flatten_bitor(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flatten_bitor(node.left) + _flatten_bitor(node.right)
    return [node]


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _make_bitor_chain(elements: list[ast.expr]) -> ast.expr:
    none_elts = [e for e in elements if _is_none(e)]
    non_none = [e for e in elements if not _is_none(e)]
    ordered = non_none + none_elts
    if len(ordered) == 1:
        return ordered[0]
    result = ordered[0]
    for elem in ordered[1:]:
        result = ast.BinOp(left=result, op=ast.BitOr(), right=elem)
    return result


# ── Continuation-line indentation checking ─────────────────────────────────

# Sections where entries follow the pattern: name (type): description
_NAMED_SECTIONS = {"Args", "Arguments", "Attributes"}

# Sections where entries follow the pattern: type: description
_TYPED_SECTIONS = {"Returns", "Yields"}

# Sections where entries follow the pattern: ExceptionType: description
_RAISES_SECTIONS = {"Raises"}

_ALL_SECTION_NAMES = _NAMED_SECTIONS | _TYPED_SECTIONS | _RAISES_SECTIONS

# Matches a section header like "    Args:" or "    Returns:"
_SECTION_HEADER_RE = re.compile(
    r"^(\s*)(" + "|".join(_ALL_SECTION_NAMES) + r")\s*:\s*$"
)

# Named entry: "name (type): desc" or "name: desc"
_NAMED_ENTRY_RE = re.compile(r"^\w[\w\d_]*\s*(?:\(.*?\))?\s*:")

# Typed entry (Returns/Yields): a type expression followed by ": desc"
# Must start with a word character (not punctuation like [ or {)
_TYPED_ENTRY_RE = re.compile(
    r"^\w[^\s:]*(?:\[.*?\])?(?:\s*\|\s*\w[^\s:]*(?:\[.*?\])?)*\s*:"
)

# Raises entry: "ExceptionName: desc"
_RAISES_ENTRY_RE = re.compile(r"^\w[\w\d_]*\s*:")


def _is_entry_line(stripped_line: str, section_name: str) -> bool:
    """Check whether a stripped line looks like the start of a new entry."""
    if section_name in _NAMED_SECTIONS:
        return bool(_NAMED_ENTRY_RE.match(stripped_line))
    if section_name in _TYPED_SECTIONS:
        return bool(_TYPED_ENTRY_RE.match(stripped_line))
    if section_name in _RAISES_SECTIONS:
        return bool(_RAISES_ENTRY_RE.match(stripped_line))
    return False


def _find_continuation_indent_violations(
    docstring: str,
) -> list[tuple[str, int, str]]:
    """Find continuation lines in docstring sections with incorrect indentation.

    Args:
        docstring (str): The raw docstring text.

    Returns:
        list[tuple[str, int, str]]: List of (section_name, line_number, line_text)
            for each violation found. Line numbers are 1-indexed within the
            docstring.
    """
    lines = docstring.split("\n")
    violations: list[tuple[str, int, str]] = []
    i = 0

    while i < len(lines):
        header_match = _SECTION_HEADER_RE.match(lines[i])
        if not header_match:
            i += 1
            continue

        section_name = header_match.group(2)
        section_indent = len(header_match.group(1))
        i += 1

        # Find the entry indent from first non-blank content line
        entry_indent = None
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            entry_indent = len(lines[i]) - len(lines[i].lstrip())
            break

        if entry_indent is None:
            continue

        # Process lines within this section
        in_entry = False
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Blank line, skip
            if not stripped:
                i += 1
                continue

            current_indent = len(line) - len(line.lstrip())

            # Left the section: dedented to or past section header level,
            # or hit a new section header
            if current_indent <= section_indent:
                break
            if _SECTION_HEADER_RE.match(line):
                break

            if current_indent == entry_indent:
                if _is_entry_line(stripped, section_name):
                    in_entry = True
                elif in_entry:
                    # Continuation at entry indent: violation
                    violations.append((section_name, i + 1, stripped))
            # Lines indented more than entry_indent are fine (proper continuation)

            i += 1

    return violations


# ── Returns type extraction (workaround for docstring_parser union bug) ────

_RETURNS_TYPE_RE = re.compile(
    r"^\s*"
    r"("
    r"[^\s:]+(?:\[.*?\])?"
    r"(?:\s*\|\s*[^\s:]+(?:\[.*?\])?)*"
    r")"
    r"\s*:\s*"
    r"(.+)",
    re.DOTALL,
)


def _extract_returns_type(docstring: str) -> str | None:
    """Extract return type from Returns: section via regex fallback."""
    match = re.search(
        r"(?:^|\n)\s*Returns:\s*\n(.*?)(?=\n\s*(?:Raises|Examples?|Note|Yields|$)|\Z)",
        docstring,
        re.DOTALL,
    )
    if not match:
        return None
    body = match.group(1)
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        type_match = _RETURNS_TYPE_RE.match(stripped)
        if type_match:
            return type_match.group(1).strip()
        return None
    return None


# ── Collection helpers ──────────────────────────────────────────────────────


def _matches_exclude(path: str) -> bool:
    return any(fnmatch(path, pat) for pat in _EXCLUDE_PATTERNS)


def _collect_py_files(directories: list[str]) -> list[tuple[Path, str]]:
    """Collect all .py files (excluding __init__.py) from the given directories."""
    results: list[tuple[Path, str]] = []
    for directory in directories:
        dir_path = _REPO_ROOT / directory
        if not dir_path.is_dir():
            continue
        for py_file in sorted(dir_path.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            if _SKIP_DIRS & set(py_file.parts):
                continue
            rel_path = str(py_file.relative_to(_REPO_ROOT))
            if _matches_exclude(rel_path):
                continue
            results.append((py_file, rel_path))
    return sorted(results, key=lambda x: x[1])


def _extract_function_annotations(node) -> dict[str, str]:
    annotations = {}
    all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
    for arg in all_args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is not None:
            try:
                annotations[arg.arg] = ast.unparse(arg.annotation)
            except Exception:
                pass
    if node.args.vararg and node.args.vararg.annotation:
        try:
            annotations[f"*{node.args.vararg.arg}"] = ast.unparse(node.args.vararg.annotation)
        except Exception:
            pass
    if node.args.kwarg and node.args.kwarg.annotation:
        try:
            annotations[f"**{node.args.kwarg.arg}"] = ast.unparse(node.args.kwarg.annotation)
        except Exception:
            pass
    return annotations


def _extract_class_annotations(node) -> dict[str, str]:
    annotations = {}
    for item in node.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        name = item.target.id
        if name.startswith("_") or name.isupper():
            continue
        try:
            type_str = ast.unparse(item.annotation)
        except Exception:
            continue
        if "ClassVar" in type_str:
            continue
        annotations[name] = type_str
    return annotations


def _collect_docstrings_with_annotations(
    directories: list[str],
) -> list[tuple[str, str, str, dict[str, str], str | None, str]]:
    """Collect multi-line docstrings with type annotations from AST."""
    results = []
    class_annotations: dict[str, dict[str, str]] = {}
    all_trees: list[tuple[str, ast.Module]] = []

    for directory in directories:
        dir_path = _REPO_ROOT / directory
        if not dir_path.is_dir():
            continue
        for py_file in sorted(dir_path.rglob("*.py")):
            if _SKIP_DIRS & set(py_file.parts):
                continue
            rel_path = str(py_file.relative_to(_REPO_ROOT))
            if _matches_exclude(rel_path):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue
            all_trees.append((rel_path, tree))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    attrs = _extract_class_annotations(node)
                    if attrs:
                        class_annotations[node.name] = attrs

    for rel_path, tree in all_trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            docstring = ast.get_docstring(node)
            if docstring is None or "\n\n" not in docstring.strip():
                continue

            qualified_name = _get_qualified_name(node, tree)

            if isinstance(node, ast.ClassDef):
                own = _extract_class_annotations(node)
                resolved = {}
                for base in node.bases:
                    base_name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
                    if base_name and base_name in class_annotations:
                        resolved.update(class_annotations[base_name])
                resolved.update(own)
                results.append((rel_path, qualified_name, docstring, resolved, None, "class", own))
            else:
                annotations = _extract_function_annotations(node)
                return_type = None
                if node.returns is not None:
                    try:
                        return_type = ast.unparse(node.returns)
                    except Exception:
                        pass
                results.append((rel_path, qualified_name, docstring, annotations, return_type, "function", annotations))

    return sorted(results, key=lambda x: (x[0], x[1]))


def _get_qualified_name(node: ast.AST, tree: ast.Module) -> str:
    name = getattr(node, "name", "?")
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.ClassDef):
            continue
        for child in ast.iter_child_nodes(parent):
            if child is node:
                return f"{parent.name}.{name}"
    return name


# ── Parametrize data ────────────────────────────────────────────────────────

_PY_FILES = _collect_py_files(_ALL_DIRS)
_DOCSTRINGS_WITH_ANNOTATIONS = _collect_docstrings_with_annotations(_SOURCE_DIRS)


# ── Module docstring tests ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "py_file, rel_path",
    _PY_FILES,
    ids=[rel for _, rel in _PY_FILES],
)
def test_module_docstring_exists(py_file: Path, rel_path: str):
    """Every .py file must have a module-level docstring."""
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        pytest.skip(f"Could not parse {rel_path}")

    docstring = ast.get_docstring(tree)
    assert docstring is not None and docstring.strip(), (
        f"{rel_path} is missing a module-level docstring."
    )


@pytest.mark.parametrize(
    "py_file, rel_path",
    _PY_FILES,
    ids=[rel for _, rel in _PY_FILES],
)
def test_module_docstring_path_header(py_file: Path, rel_path: str):
    """The first line of the module docstring must be the relative path from repo root."""
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        pytest.skip(f"Could not parse {rel_path}")

    docstring = ast.get_docstring(tree)
    if not docstring:
        pytest.skip(f"{rel_path} has no docstring (caught by test_module_docstring_exists)")

    first_line = docstring.strip().split("\n")[0].strip()
    assert first_line == rel_path, (
        f"{rel_path}: module docstring first line should be '{rel_path}', "
        f"got '{first_line}'"
    )


@pytest.mark.parametrize(
    "py_file, rel_path",
    _PY_FILES,
    ids=[rel for _, rel in _PY_FILES],
)
def test_module_docstring_format(py_file: Path, rel_path: str):
    """If the module docstring is multi-line, the second line must be blank."""
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        pytest.skip(f"Could not parse {rel_path}")

    docstring = ast.get_docstring(tree)
    if not docstring:
        pytest.skip(f"{rel_path} has no docstring")

    lines = docstring.split("\n")
    if len(lines) <= 1:
        return

    assert lines[1].strip() == "", (
        f"{rel_path}: multi-line module docstring must have a blank line after "
        f"the path header. Got: '{lines[1].strip()}'"
    )


# ── Docstring type matching tests ──────────────────────────────────────────


@pytest.mark.parametrize(
    "file_path, name, docstring, annotations, return_type, node_kind, own_annotations",
    _DOCSTRINGS_WITH_ANNOTATIONS,
    ids=[f"{fp}::{n}" for fp, n, *_ in _DOCSTRINGS_WITH_ANNOTATIONS],
)
def test_docstring_types_match_signatures(
    file_path: str,
    name: str,
    docstring: str,
    annotations: dict[str, str],
    return_type: str | None,
    node_kind: str,
    own_annotations: dict[str, str],
):
    """Docstring Args/Attributes types must be present and match signatures."""
    try:
        parsed = parse_docstring(docstring, style=DocstringStyle.GOOGLE)
    except Exception:
        pytest.skip(f"Could not parse docstring for {file_path}::{name}")
        return

    violations = []

    for param in parsed.params:
        if param.arg_name.startswith("*"):
            continue

        sig_type = annotations.get(param.arg_name)
        if sig_type is None:
            continue

        norm_sig = _normalize_type(sig_type)

        if param.type_name is None:
            violations.append(
                f"  {param.arg_name}: missing type in docstring "
                f"(should be '{norm_sig}')"
            )
            continue

        norm_doc = _normalize_type(param.type_name)

        if norm_doc != norm_sig:
            violations.append(
                f"  {param.arg_name}: docstring type '{param.type_name}' "
                f"!= signature type '{sig_type}' "
                f"(normalized: '{norm_doc}' vs '{norm_sig}')"
            )

    assert not violations, (
        f"{file_path}::{name} has docstring param type mismatches:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "file_path, name, docstring, annotations, return_type, node_kind, own_annotations",
    _DOCSTRINGS_WITH_ANNOTATIONS,
    ids=[f"{fp}::{n}" for fp, n, *_ in _DOCSTRINGS_WITH_ANNOTATIONS],
)
def test_docstring_documents_all_params(
    file_path: str,
    name: str,
    docstring: str,
    annotations: dict[str, str],
    return_type: str | None,
    node_kind: str,
    own_annotations: dict[str, str],
):
    """Every own (non-inherited) annotated parameter must be documented."""
    try:
        parsed = parse_docstring(docstring, style=DocstringStyle.GOOGLE)
    except Exception:
        pytest.skip(f"Could not parse docstring for {file_path}::{name}")
        return

    documented = {p.arg_name for p in parsed.params}

    missing = []
    for param_name in own_annotations:
        if param_name.startswith("*"):
            continue
        if param_name not in documented:
            missing.append(param_name)

    assert not missing, (
        f"{file_path}::{name} is missing documentation for annotated "
        f"params: {missing}"
    )


@pytest.mark.parametrize(
    "file_path, name, docstring, annotations, return_type, node_kind, own_annotations",
    [item for item in _DOCSTRINGS_WITH_ANNOTATIONS if item[5] == "function" and item[4] is not None],
    ids=[
        f"{fp}::{n}"
        for fp, n, _, _, rt, nk, _ in _DOCSTRINGS_WITH_ANNOTATIONS
        if nk == "function" and rt is not None
    ],
)
def test_docstring_return_type_matches_signature(
    file_path: str,
    name: str,
    docstring: str,
    annotations: dict[str, str],
    return_type: str | None,
    node_kind: str,
    own_annotations: dict[str, str],
):
    """Docstring Returns: type must match the return annotation."""
    if return_type is None:
        pytest.skip("No return annotation")
        return

    norm_sig = _normalize_type(return_type)
    if norm_sig == "None":
        pytest.skip("Returns None")
        return

    try:
        parsed = parse_docstring(docstring, style=DocstringStyle.GOOGLE)
    except Exception:
        pytest.skip(f"Could not parse docstring for {file_path}::{name}")
        return

    if not parsed.returns:
        pytest.skip("No Returns: section in docstring")
        return

    doc_return_type = parsed.returns.type_name

    # Fallback: docstring_parser can't handle union types with spaces
    if doc_return_type is None:
        doc_return_type = _extract_returns_type(docstring)

    if doc_return_type is None:
        pytest.fail(
            f"{file_path}::{name}: Returns: section missing type "
            f"(should be '{norm_sig}')"
        )
        return

    norm_doc = _normalize_type(doc_return_type)

    assert norm_doc == norm_sig, (
        f"{file_path}::{name}: return type mismatch: "
        f"docstring '{doc_return_type}' != signature '{return_type}' "
        f"(normalized: '{norm_doc}' vs '{norm_sig}')"
    )


# ── Continuation indentation tests ───────────────────────────────────────────


@pytest.mark.parametrize(
    "file_path, name, docstring, annotations, return_type, node_kind, own_annotations",
    _DOCSTRINGS_WITH_ANNOTATIONS,
    ids=[f"{fp}::{n}" for fp, n, *_ in _DOCSTRINGS_WITH_ANNOTATIONS],
)
def test_docstring_continuation_indentation(
    file_path: str,
    name: str,
    docstring: str,
    annotations: dict[str, str],
    return_type: str | None,
    node_kind: str,
    own_annotations: dict[str, str],
):
    """Continuation lines in docstring sections must be indented past the entry line."""
    violations = _find_continuation_indent_violations(docstring)
    if not violations:
        return

    details = "\n".join(
        f"  {section} line {lineno}: {text}"
        for section, lineno, text in violations
    )
    pytest.fail(
        f"{file_path}::{name} has continuation lines at entry indent "
        f"(should be indented further):\n{details}"
    )
