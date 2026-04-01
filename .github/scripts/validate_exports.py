#!/usr/bin/env python3
"""
Validate export chain consistency across proto-language and proto-tools.

AST-based, zero dependencies (stdlib only). Parses __init__.py files without
executing them.

Validation is driven by a domain config file (export_config.json). Each domain
defines an independent validation scope with its own root directory, checks, and
registry decorators. This avoids hard-coding package structure in Python; adding
or restructuring packages only requires editing the JSON config.

Two checks available per domain:
  1. all_consistency: Every item in __all__ must be actually imported or defined
     in that module (catches stale entries).
  2. registry_exports: Every @tool/@constraint/@generator/@optimizer decorated
     function must be exported by its immediate parent __init__.py.

Safety mitigations:
  - A configured domain root that doesn't exist is an ERROR (not a silent skip),
    catching stale config after package restructuring.
  - A domain with registry_decorators that finds zero decorated functions emits
    a WARNING, catching renamed or removed decorators.

Usage:
    python .github/scripts/validate_exports.py                # All domains
    python .github/scripts/validate_exports.py --domain Tools  # Single domain
    python .github/scripts/validate_exports.py --verbose       # Show all checks

Exit codes: 0 = pass, 1 = errors found.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# AST Helpers
# ---------------------------------------------------------------------------


def parse_init(init_path: Path) -> Optional[ast.Module]:
    """Parse an __init__.py file into an AST, returning None on failure."""
    try:
        source = init_path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(init_path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        print(f"WARNING: Could not parse {init_path}: {exc}", file=sys.stderr)
        return None


def extract_all_list(tree: ast.Module) -> Optional[List[str]]:
    """Extract __all__ from an AST module, merging base assignment and += augmentations."""
    result: Optional[List[str]] = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    result = _extract_string_list(node.value)
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                augmented = _extract_string_list(node.value)
                if augmented is not None:
                    if result is None:
                        result = augmented
                    else:
                        result.extend(augmented)
    return result


def _extract_string_list(node: ast.expr) -> Optional[List[str]]:
    """Extract a list of string constants from a List or Tuple AST node."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    names = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names.append(elt.value)
        else:
            warnings.warn(
                f"Non-string element in __all__ will be skipped: {ast.dump(elt)}",
                stacklevel=2,
            )
    return names


def extract_imports(tree: ast.Module) -> Tuple[Set[str], bool]:
    """
    Extract all imported names from an __init__.py AST.

    Returns:
        (set of imported symbol names, whether wildcard import is used)
    """
    names: Set[str] = set()
    has_wildcard = False

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.asname if alias.asname else alias.name.split(".")[-1]
                names.add(imported)

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    has_wildcard = True
                else:
                    imported = alias.asname if alias.asname else alias.name
                    names.add(imported)

    return names, has_wildcard


def extract_definitions(tree: ast.Module) -> Set[str]:
    """Extract top-level class and function definitions from an AST module."""
    defs: Set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defs.add(target.id)
    return defs


def extract_decorated_names(
    py_path: Path, decorator_names: Set[str]
) -> Dict[str, str]:
    """
    Find all functions/classes decorated with any of the given decorator names
    in a .py file. Returns a dict of {function_name: decorator_name}.
    """
    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_path))
    except (SyntaxError, UnicodeDecodeError):
        return {}

    decorated: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                dec_name = _get_decorator_name(dec)
                if dec_name in decorator_names:
                    decorated[node.name] = dec_name
    return decorated


def _get_decorator_name(node: ast.expr) -> Optional[str]:
    """Extract the name of a decorator from its AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _get_decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# ---------------------------------------------------------------------------
# Chain Discovery
# ---------------------------------------------------------------------------


def discover_init_chain(root_dir: Path) -> List[Path]:
    """
    Discover all __init__.py files under root_dir, sorted by depth (deepest first).
    """
    inits = sorted(root_dir.rglob("__init__.py"), key=lambda p: len(p.parts), reverse=True)
    return inits


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError:
    """A single validation error."""

    def __init__(self, symbol: str, message: str, file: Path):
        self.symbol = symbol
        self.message = message
        self.file = file

    def __str__(self) -> str:
        rel = self.file
        try:
            rel = self.file.relative_to(REPO_ROOT)
        except ValueError:
            try:
                rel = self.file.relative_to(REPO_ROOT.parent)
            except ValueError:
                pass
        return f"{self.symbol}: {self.message} ({rel})"


def load_exceptions(config: dict) -> Set[str]:
    """Extract exception symbols from the config's exceptions section."""
    exc_section = config.get("exceptions", {})
    if isinstance(exc_section, list):
        return set(exc_section)
    if isinstance(exc_section, dict):
        all_exceptions: Set[str] = set()
        for key, val in exc_section.items():
            if key.startswith("_"):
                continue
            if isinstance(val, list):
                all_exceptions.update(val)
            elif isinstance(val, str):
                all_exceptions.add(val)
        return all_exceptions
    return set()


def load_config(config_path: Path) -> dict:
    """Load and validate the domain config file."""
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"ERROR: Could not parse config: {exc}", file=sys.stderr)
        sys.exit(1)
    if "domains" not in data or not isinstance(data["domains"], list):
        print("ERROR: Config must have a 'domains' list", file=sys.stderr)
        sys.exit(1)
    return data


def resolve_domain_root(domain: dict, repo_root: Path) -> Optional[Path]:
    """
    Resolve the root directory for a domain.

    Checks the primary 'root' relative to repo_root first, then falls back
    to 'root_search' paths (for tools submodule / sibling repo).
    """
    primary = repo_root / domain["root"]
    if primary.is_dir():
        return primary

    for alt in domain.get("root_search", []):
        proposal = repo_root / alt
        if proposal.is_dir():
            return proposal

    return None


def validate_all_consistency(
    init_path: Path, tree: ast.Module, all_list: List[str]
) -> List[ValidationError]:
    """
    Check that every entry in __all__ is actually imported or defined in this module.
    Catches stale __all__ entries that reference removed symbols.
    """
    imported, _ = extract_imports(tree)
    defined = extract_definitions(tree)
    available = imported | defined
    # Remove __all__ itself from available (it's an assignment, not a real export)
    available.discard("__all__")

    errors = []
    for name in all_list:
        if name not in available:
            errors.append(
                ValidationError(
                    symbol=name,
                    message="listed in __all__ but not imported or defined",
                    file=init_path,
                )
            )
    return errors


def validate_registry_exports(
    parsed: Dict[Path, Tuple[ast.Module, Optional[List[str]]]],
    decorated_by_file: Dict[Path, Dict[str, str]],
    exceptions: Set[str],
) -> List[ValidationError]:
    """
    Check that every @decorator-registered function in the package is exported
    by its immediate parent __init__.py's __all__.
    """
    errors = []
    for py_file, decorated in sorted(decorated_by_file.items()):
        # Look up the immediate parent __init__.py's __all__
        parent_init = py_file.parent / "__init__.py"
        if parent_init in parsed:
            _, parent_all = parsed[parent_init]
            parent_exported = set(parent_all) if parent_all is not None else set()
        else:
            parent_exported = set()

        for name, dec_name in decorated.items():
            if name in exceptions:
                continue
            if name not in parent_exported:
                errors.append(
                    ValidationError(
                        symbol=name,
                        message=(
                            f"decorated with @{dec_name} "
                            f"but not in __all__ of {_rel(parent_init)}"
                        ),
                        file=py_file,
                    )
                )
    return errors


def validate_package_root_exports(
    package_root: Path,
    decorated_by_file: Dict[Path, Dict[str, str]],
    exceptions: Set[str],
) -> List[ValidationError]:
    """
    Check that every @decorator-registered function in the domain is exported
    by the package_root's __init__.py __all__.

    This catches symbols that are correctly exported by their immediate parent
    (e.g., constraint/__init__.py) but missing from the top-level package
    (e.g., proto_language/__init__.py).
    """
    errors = []
    pkg_root_init = package_root / "__init__.py"

    if not pkg_root_init.exists():
        root_all: Set[str] = set()
    else:
        tree = parse_init(pkg_root_init)
        if tree is None:
            return errors
        all_list = extract_all_list(tree)
        root_all = set(all_list) if all_list is not None else set()

    for py_file, decorated in sorted(decorated_by_file.items()):
        for name, dec_name in decorated.items():
            if name in exceptions:
                continue
            if name not in root_all:
                errors.append(
                    ValidationError(
                        symbol=name,
                        message=(
                            f"decorated with @{dec_name} "
                            f"but not in __all__ of {_rel(pkg_root_init)}"
                        ),
                        file=py_file,
                    )
                )
    return errors


def _rel(path: Path) -> str:
    """Make a path relative to REPO_ROOT for display."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT.parent))
        except ValueError:
            return str(path)


# ---------------------------------------------------------------------------
# Domain Validation (high-level)
# ---------------------------------------------------------------------------


def validate_domain(
    domain: dict,
    repo_root: Path,
    exceptions: Set[str],
    verbose: bool = False,
) -> Tuple[List[ValidationError], List[str]]:
    """
    Validate a single domain from the config.

    Returns:
        (list of errors, list of warnings)
    """
    name = domain["name"]
    checks = set(domain.get("checks", []))
    decorator_names = set(domain.get("registry_decorators", []))
    depth = domain.get("depth")

    domain_errors: List[ValidationError] = []
    domain_warnings: List[str] = []

    # Resolve root directory
    root_dir = resolve_domain_root(domain, repo_root)
    if root_dir is None:
        domain_errors.append(
            ValidationError(
                symbol=name,
                message=(
                    f"domain root not found: {domain['root']} "
                    f"(config may be stale after package restructuring)"
                ),
                file=repo_root / domain["root"] / "__init__.py",
            )
        )
        return domain_errors, domain_warnings

    if verbose:
        print(f"\n  [{name}] root: {_rel(root_dir)}")

    # Discover and parse __init__.py files
    if depth == 0:
        # Only check the root __init__.py itself
        root_init = root_dir / "__init__.py"
        if not root_init.exists():
            if verbose:
                print(f"    No __init__.py at {_rel(root_dir)}")
            return domain_errors, domain_warnings
        tree = parse_init(root_init)
        if tree is None:
            return domain_errors, domain_warnings
        all_list = extract_all_list(tree)
        parsed = {root_init: (tree, all_list)}
    else:
        init_files = discover_init_chain(root_dir)
        if not init_files:
            if verbose:
                print(f"    No __init__.py files found")
            return domain_errors, domain_warnings
        parsed: Dict[Path, Tuple[ast.Module, Optional[List[str]]]] = {}
        for init in init_files:
            tree = parse_init(init)
            if tree is None:
                continue
            all_list = extract_all_list(tree)
            parsed[init] = (tree, all_list)

    # Check 1: __all__ consistency
    if "all_consistency" in checks:
        if verbose:
            print(f"    Checking __all__ consistency...")
        for init, (tree, all_list) in parsed.items():
            if all_list is None:
                continue
            errors = validate_all_consistency(init, tree, all_list)
            if verbose and not errors:
                print(f"      OK: {_rel(init)} ({len(all_list)} exports)")
            domain_errors.extend(errors)

    # Scan decorated names once (shared by checks 2 + 3 + zero-count warning)
    decorated_by_file: Dict[Path, Dict[str, str]] = {}
    if "registry_exports" in checks and decorator_names:
        for py_file in sorted(root_dir.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            decorated = extract_decorated_names(py_file, decorator_names)
            if decorated:
                decorated_by_file[py_file] = decorated

    # Check 2: Registry decorator exports
    if "registry_exports" in checks and decorator_names:
        if verbose:
            print(f"    Checking registry decorators: {decorator_names}")
        errors = validate_registry_exports(
            parsed, decorated_by_file, exceptions
        )
        domain_errors.extend(errors)

        # Safety: warn if zero decorated functions found (possible stale config)
        if not decorated_by_file:
            msg = (
                f"[{name}] WARNING: found 0 functions decorated with "
                f"{decorator_names} under {_rel(root_dir)}. "
                f"If decorators were renamed, update export_config.json."
            )
            domain_warnings.append(msg)

    # Check 3: Package root exports (full-chain check)
    package_root_path = domain.get("package_root")
    if "registry_exports" in checks and decorator_names and package_root_path:
        pkg_root_dir = repo_root / package_root_path
        if verbose:
            print(f"    Checking package root exports: {_rel(pkg_root_dir)}")
        errors = validate_package_root_exports(
            pkg_root_dir, decorated_by_file, exceptions
        )
        domain_errors.extend(errors)

    return domain_errors, domain_warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate export chain consistency (config-driven)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / ".github" / "scripts" / "export_config.json",
        help="Path to the domain config file.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Validate only the named domain (e.g. 'Tools', 'Constraints').",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all checks, not just errors."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exceptions = load_exceptions(config)

    if args.verbose and exceptions:
        print(f"Loaded {len(exceptions)} exception(s)")

    all_errors: List[ValidationError] = []
    all_warnings: List[str] = []

    for domain in config["domains"]:
        # Filter to single domain if requested
        if args.domain and domain["name"] != args.domain:
            continue

        errors, domain_warnings = validate_domain(
            domain, REPO_ROOT, exceptions, args.verbose
        )
        all_errors.extend(errors)
        all_warnings.extend(domain_warnings)

    # Print warnings
    for w in all_warnings:
        print(w, file=sys.stderr)

    # Report
    if all_errors:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(
            f"EXPORT CHAIN ERRORS: {len(all_errors)} issue(s) found",
            file=sys.stderr,
        )
        print(f"{'=' * 60}", file=sys.stderr)
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        print(file=sys.stderr)
        return 1

    if args.verbose:
        print(f"\n{'=' * 60}")
        print("All export chains valid!")
        print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
