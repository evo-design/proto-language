"""tests/utils_tests/test_validate_exports.py

Uses synthetic temp files to test AST parsing and validation logic
without depending on the real codebase."""
from __future__ import annotations

import json

# Import the validator functions directly
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / ".github" / "scripts"))
from validate_exports import (
    extract_all_list,
    extract_decorated_names,
    extract_definitions,
    extract_imports,
    load_config,
    load_exceptions,
    parse_init,
    resolve_domain_root,
    validate_all_consistency,
    validate_domain,
    validate_package_root_exports,
    validate_registry_exports,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_package(tmp_path):
    """Create a synthetic package structure for testing."""
    # Level 1: tools/causal_models/evo2/__init__.py
    (tmp_path / "tools" / "causal_models" / "evo2").mkdir(parents=True)
    # Level 2: tools/causal_models/__init__.py
    # Level 3: tools/__init__.py
    return tmp_path


def write_init(path: Path, content: str) -> Path:
    """Write content to an __init__.py, creating dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    init = path / "__init__.py" if path.is_dir() else path
    init.write_text(textwrap.dedent(content))
    return init


# =============================================================================
# AST Parsing Tests
# =============================================================================


class TestExtractAllList:
    def test_simple_all(self, tmp_path):
        init = write_init(tmp_path, '''
            __all__ = ["Foo", "Bar", "baz"]
        ''')
        tree = parse_init(init)
        result = extract_all_list(tree)
        assert result == ["Foo", "Bar", "baz"]

    def test_no_all(self, tmp_path):
        init = write_init(tmp_path, '''
            from .foo import Foo
        ''')
        tree = parse_init(init)
        result = extract_all_list(tree)
        assert result is None

    def test_tuple_all(self, tmp_path):
        init = write_init(tmp_path, '''
            __all__ = ("Foo", "Bar")
        ''')
        tree = parse_init(init)
        result = extract_all_list(tree)
        assert result == ["Foo", "Bar"]

    def test_empty_all(self, tmp_path):
        init = write_init(tmp_path, '''
            __all__ = []
        ''')
        tree = parse_init(init)
        result = extract_all_list(tree)
        assert result == []

    def test_augmented_assignment_merged(self, tmp_path):
        """__all__ = [...] followed by __all__ += [...] should merge both."""
        init = write_init(tmp_path, '''
            __all__ = ["A", "B"]
            __all__ += ["C", "D"]
        ''')
        tree = parse_init(init)
        result = extract_all_list(tree)
        assert result == ["A", "B", "C", "D"]

    def test_augmented_only(self, tmp_path):
        """__all__ += [...] without a base assignment should still return the list."""
        init = write_init(tmp_path, '''
            __all__ += ["X"]
        ''')
        tree = parse_init(init)
        result = extract_all_list(tree)
        assert result == ["X"]

    def test_non_string_element_warns(self, tmp_path):
        """Non-string elements in __all__ should produce a warning."""
        init = write_init(tmp_path, '''
            SOME_VAR = "dynamic"
            __all__ = ["Foo", SOME_VAR]
        ''')
        tree = parse_init(init)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_all_list(tree)
            assert result == ["Foo"]
            assert len(w) == 1
            assert "Non-string element" in str(w[0].message)


class TestExtractImports:
    def test_relative_imports(self, tmp_path):
        init = write_init(tmp_path, '''
            from .foo import Foo, Bar
            from .baz import baz_func
        ''')
        tree = parse_init(init)
        names, has_wildcard = extract_imports(tree)
        assert names == {"Foo", "Bar", "baz_func"}
        assert not has_wildcard

    def test_wildcard_import(self, tmp_path):
        init = write_init(tmp_path, '''
            from .tools import *
        ''')
        tree = parse_init(init)
        names, has_wildcard = extract_imports(tree)
        assert has_wildcard

    def test_aliased_import(self, tmp_path):
        init = write_init(tmp_path, '''
            from .foo import Bar as Baz
        ''')
        tree = parse_init(init)
        names, _ = extract_imports(tree)
        assert "Baz" in names
        assert "Bar" not in names

    def test_absolute_import(self, tmp_path):
        init = write_init(tmp_path, '''
            import os
            from os.path import join
        ''')
        tree = parse_init(init)
        names, _ = extract_imports(tree)
        assert "os" in names
        assert "join" in names


class TestExtractDefinitions:
    def test_class_and_function(self, tmp_path):
        init = write_init(tmp_path, '''
            class FooConfig:
                pass

            def bar_func():
                pass

            CONSTANT = 42
        ''')
        tree = parse_init(init)
        defs = extract_definitions(tree)
        assert "FooConfig" in defs
        assert "bar_func" in defs
        assert "CONSTANT" in defs


class TestExtractDecoratedNames:
    def test_tool_decorator(self, tmp_path):
        py_file = tmp_path / "my_tool.py"
        py_file.write_text(textwrap.dedent('''
            @tool(key="my-tool", label="My Tool")
            def run_my_tool(inputs, config):
                pass
        '''))
        result = extract_decorated_names(py_file, {"tool"})
        assert result == {"run_my_tool": "tool"}

    def test_constraint_decorator(self, tmp_path):
        py_file = tmp_path / "my_constraint.py"
        py_file.write_text(textwrap.dedent('''
            @constraint(key="gc-content", label="GC Content", config=GCContentConfig)
            def gc_content_constraint(input_sequences, config):
                pass
        '''))
        result = extract_decorated_names(py_file, {"constraint"})
        assert result == {"gc_content_constraint": "constraint"}

    def test_no_match(self, tmp_path):
        py_file = tmp_path / "plain.py"
        py_file.write_text(textwrap.dedent('''
            @pytest.mark.slow
            def test_something():
                pass
        '''))
        result = extract_decorated_names(py_file, {"tool", "constraint"})
        assert result == {}

    def test_multiple_decorators(self, tmp_path):
        py_file = tmp_path / "multi.py"
        py_file.write_text(textwrap.dedent('''
            @tool(key="tool-a")
            def run_tool_a():
                pass

            @tool(key="tool-b")
            def run_tool_b():
                pass

            def helper():
                pass
        '''))
        result = extract_decorated_names(py_file, {"tool"})
        assert result == {"run_tool_a": "tool", "run_tool_b": "tool"}


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateAllConsistency:
    def test_consistent(self, tmp_path):
        init = write_init(tmp_path, '''
            from .foo import Foo
            from .bar import Bar

            __all__ = ["Foo", "Bar"]
        ''')
        tree = parse_init(init)
        all_list = extract_all_list(tree)
        errors = validate_all_consistency(init, tree, all_list)
        assert len(errors) == 0

    def test_stale_entry(self, tmp_path):
        init = write_init(tmp_path, '''
            from .foo import Foo

            __all__ = ["Foo", "Bar"]
        ''')
        tree = parse_init(init)
        all_list = extract_all_list(tree)
        errors = validate_all_consistency(init, tree, all_list)
        assert len(errors) == 1
        assert errors[0].symbol == "Bar"
        assert "not imported or defined" in errors[0].message

    def test_defined_not_imported(self, tmp_path):
        init = write_init(tmp_path, '''
            class MyClass:
                pass

            __all__ = ["MyClass"]
        ''')
        tree = parse_init(init)
        all_list = extract_all_list(tree)
        errors = validate_all_consistency(init, tree, all_list)
        assert len(errors) == 0


class TestValidateRegistryExports:
    def _build_parsed(self, pkg: Path):
        """Helper to build parsed dict for a package dir."""
        parsed = {}
        for init in pkg.rglob("__init__.py"):
            tree = parse_init(init)
            if tree:
                all_list = extract_all_list(tree)
                parsed[init] = (tree, all_list)
        return parsed

    def _scan_decorated(self, pkg: Path, decorator_names):
        """Helper to scan decorated names once for a package dir."""
        result = {}
        for py_file in sorted(pkg.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            decorated = extract_decorated_names(py_file, decorator_names)
            if decorated:
                result[py_file] = decorated
        return result

    def test_decorated_and_exported(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()

        write_init(pkg, '''
            from .my_tool import run_my_tool

            __all__ = ["run_my_tool"]
        ''')
        (pkg / "my_tool.py").write_text(textwrap.dedent('''
            @tool(key="my-tool")
            def run_my_tool():
                pass
        '''))

        parsed = self._build_parsed(pkg)
        decorated_by_file = self._scan_decorated(pkg, {"tool"})
        errors = validate_registry_exports(parsed, decorated_by_file, set())
        assert len(errors) == 0

    def test_decorated_but_not_exported(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()

        write_init(pkg, '''
            __all__ = []
        ''')
        (pkg / "my_tool.py").write_text(textwrap.dedent('''
            @tool(key="my-tool")
            def run_my_tool():
                pass
        '''))

        parsed = self._build_parsed(pkg)
        decorated_by_file = self._scan_decorated(pkg, {"tool"})
        errors = validate_registry_exports(parsed, decorated_by_file, set())
        assert len(errors) == 1
        assert errors[0].symbol == "run_my_tool"

    def test_same_name_in_unrelated_package_still_flagged(self, tmp_path):
        """A same-named symbol in an unrelated sub-package should not satisfy the check."""
        root = tmp_path / "root"

        # Sub-package A: has @tool run_foo but does NOT export it
        pkg_a = root / "pkg_a"
        pkg_a.mkdir(parents=True)
        write_init(pkg_a, '''
            __all__ = []
        ''')
        (pkg_a / "my_tool.py").write_text(textwrap.dedent('''
            @tool(key="foo")
            def run_foo():
                pass
        '''))

        # Sub-package B: exports run_foo (same name, different package)
        pkg_b = root / "pkg_b"
        pkg_b.mkdir(parents=True)
        write_init(pkg_b, '''
            from .other import run_foo

            __all__ = ["run_foo"]
        ''')

        # Root __init__.py
        write_init(root, '''
            __all__ = []
        ''')

        parsed = self._build_parsed(root)
        decorated_by_file = self._scan_decorated(root, {"tool"})
        errors = validate_registry_exports(parsed, decorated_by_file, set())
        # pkg_a/my_tool.py has @tool run_foo but pkg_a/__init__.py has __all__ = []
        assert len(errors) == 1
        assert errors[0].symbol == "run_foo"


# =============================================================================
# Config Loading
# =============================================================================


class TestLoadConfig:
    def test_valid_config(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps({
            "domains": [
                {"name": "Test", "root": "pkg", "checks": ["all_consistency"]}
            ],
            "exceptions": {"group": ["Foo"]},
        }))
        result = load_config(f)
        assert len(result["domains"]) == 1
        assert result["domains"][0]["name"] == "Test"

    def test_missing_config_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(tmp_path / "nonexistent.json")

    def test_invalid_json_exits(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        with pytest.raises(SystemExit):
            load_config(f)

    def test_missing_domains_key_exits(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"exceptions": {}}))
        with pytest.raises(SystemExit):
            load_config(f)


class TestLoadExceptions:
    def test_empty_config(self):
        result = load_exceptions({})
        assert result == set()

    def test_flat_list(self):
        result = load_exceptions({"exceptions": ["Foo", "Bar"]})
        assert result == {"Foo", "Bar"}

    def test_grouped_dict(self):
        result = load_exceptions({"exceptions": {
            "internal": ["Foo", "Bar"],
            "constants": ["BAZ"],
        }})
        assert result == {"Foo", "Bar", "BAZ"}

    def test_ignores_comment_keys(self):
        result = load_exceptions({"exceptions": {
            "_comment": "This is a comment",
            "_note": "Another underscore key",
            "real": ["Foo"],
        }})
        assert result == {"Foo"}
        assert "This is a comment" not in result
        assert "_note" not in result


class TestResolveDomainRoot:
    def test_primary_root(self, tmp_path):
        pkg = tmp_path / "my_pkg"
        pkg.mkdir()
        domain = {"name": "Test", "root": "my_pkg"}
        result = resolve_domain_root(domain, tmp_path)
        assert result == pkg

    def test_fallback_to_root_search(self, tmp_path):
        alt = tmp_path / "alt" / "my_pkg"
        alt.mkdir(parents=True)
        domain = {
            "name": "Test",
            "root": "my_pkg",
            "root_search": ["alt/my_pkg"],
        }
        result = resolve_domain_root(domain, tmp_path)
        assert result == alt

    def test_returns_none_if_not_found(self, tmp_path):
        domain = {"name": "Test", "root": "nonexistent"}
        result = resolve_domain_root(domain, tmp_path)
        assert result is None


# =============================================================================
# Domain Validation (Integration)
# =============================================================================


class TestValidateDomain:
    def test_all_consistency_domain(self, tmp_path):
        """A domain with all_consistency check catches stale __all__ entries."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        write_init(pkg, '''
            from .foo import Foo

            __all__ = ["Foo", "Stale"]
        ''')
        domain = {
            "name": "Test",
            "root": "pkg",
            "checks": ["all_consistency"],
        }
        errors, warnings = validate_domain(domain, tmp_path, set())
        assert len(errors) == 1
        assert errors[0].symbol == "Stale"

    def test_registry_exports_domain(self, tmp_path):
        """A domain with registry_exports check catches unexported decorated functions."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        write_init(pkg, '''
            __all__ = []
        ''')
        (pkg / "my_tool.py").write_text(textwrap.dedent('''
            @tool(key="my-tool")
            def run_my_tool():
                pass
        '''))
        domain = {
            "name": "Test",
            "root": "pkg",
            "registry_decorators": ["tool"],
            "checks": ["all_consistency", "registry_exports"],
        }
        errors, warnings = validate_domain(domain, tmp_path, set())
        assert len(errors) == 1
        assert errors[0].symbol == "run_my_tool"

    def test_clean_domain_no_errors(self, tmp_path):
        """A fully consistent domain produces no errors."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        write_init(pkg, '''
            from .my_tool import run_my_tool

            __all__ = ["run_my_tool"]
        ''')
        (pkg / "my_tool.py").write_text(textwrap.dedent('''
            @tool(key="my-tool")
            def run_my_tool():
                pass
        '''))
        domain = {
            "name": "Test",
            "root": "pkg",
            "registry_decorators": ["tool"],
            "checks": ["all_consistency", "registry_exports"],
        }
        errors, warnings = validate_domain(domain, tmp_path, set())
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_missing_root_is_error(self, tmp_path):
        """A domain whose root doesn't exist produces an error, not a silent skip."""
        domain = {
            "name": "Ghost",
            "root": "nonexistent_pkg",
            "checks": ["all_consistency"],
        }
        errors, warnings = validate_domain(domain, tmp_path, set())
        assert len(errors) == 1
        assert "domain root not found" in errors[0].message

    def test_zero_decorated_warning(self, tmp_path):
        """A domain with registry_decorators that finds zero matches emits a warning."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        write_init(pkg, '''
            __all__ = []
        ''')
        # A .py file with no matching decorators
        (pkg / "plain.py").write_text(textwrap.dedent('''
            def helper():
                pass
        '''))
        domain = {
            "name": "Test",
            "root": "pkg",
            "registry_decorators": ["tool"],
            "checks": ["registry_exports"],
        }
        errors, warnings = validate_domain(domain, tmp_path, set())
        assert len(errors) == 0
        assert len(warnings) == 1
        assert "found 0 functions" in warnings[0]

    def test_depth_zero_only_checks_root_init(self, tmp_path):
        """depth=0 only checks the root __init__.py, not children."""
        pkg = tmp_path / "pkg"
        sub = pkg / "sub"
        sub.mkdir(parents=True)

        # Root init is clean
        write_init(pkg, '''
            from .sub import Foo

            __all__ = ["Foo"]
        ''')
        # Sub init has a stale entry
        write_init(sub, '''
            __all__ = ["Foo", "Stale"]
        ''')

        domain = {
            "name": "Root Only",
            "root": "pkg",
            "checks": ["all_consistency"],
            "depth": 0,
        }
        errors, _ = validate_domain(domain, tmp_path, set())
        # Should only check root — root is clean, so no errors
        assert len(errors) == 0

    def test_exceptions_respected(self, tmp_path):
        """Exception symbols are skipped in registry export checks."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        write_init(pkg, '''
            __all__ = []
        ''')
        (pkg / "my_tool.py").write_text(textwrap.dedent('''
            @tool(key="my-tool")
            def run_my_tool():
                pass
        '''))
        domain = {
            "name": "Test",
            "root": "pkg",
            "registry_decorators": ["tool"],
            "checks": ["registry_exports"],
        }
        # run_my_tool is in exceptions, so it should not be flagged
        errors, _ = validate_domain(domain, tmp_path, {"run_my_tool"})
        assert len(errors) == 0

    def test_root_search_fallback(self, tmp_path):
        """Domain with root_search falls back to alternative paths."""
        alt = tmp_path / "alt_location" / "pkg"
        alt.mkdir(parents=True)
        write_init(alt, '''
            from .foo import Foo

            __all__ = ["Foo"]
        ''')
        domain = {
            "name": "Test",
            "root": "primary_pkg",
            "root_search": ["alt_location/pkg"],
            "checks": ["all_consistency"],
        }
        errors, _ = validate_domain(domain, tmp_path, set())
        assert len(errors) == 0

    def test_package_root_catches_missing_exports(self, tmp_path):
        """validate_domain with package_root catches symbols missing from package root."""
        pkg = tmp_path / "bio_prog" / "language" / "constraint"
        pkg.mkdir(parents=True)

        # Domain __init__.py exports the symbol (immediate parent is fine)
        write_init(pkg, '''
            from .gc import gc_content_constraint

            __all__ = ["gc_content_constraint"]
        ''')
        (pkg / "gc.py").write_text(textwrap.dedent('''
            @constraint(key="gc-content")
            def gc_content_constraint():
                pass
        '''))

        # Package root __init__.py does NOT export it
        pkg_root = tmp_path / "bio_prog"
        write_init(pkg_root, '''
            __all__ = ["SomeOtherThing"]
        ''')

        domain = {
            "name": "Constraints",
            "root": "bio_prog/language/constraint",
            "package_root": "bio_prog",
            "registry_decorators": ["constraint"],
            "checks": ["registry_exports"],
        }
        errors, _ = validate_domain(domain, tmp_path, set())
        # Should catch missing from package root
        pkg_root_errors = [e for e in errors if "bio_prog" in e.message and "__all__" in e.message]
        assert len(pkg_root_errors) == 1
        assert pkg_root_errors[0].symbol == "gc_content_constraint"

    def test_no_package_root_skips_chain_check(self, tmp_path):
        """validate_domain without package_root skips the full-chain check."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        write_init(pkg, '''
            from .my_tool import run_my_tool

            __all__ = ["run_my_tool"]
        ''')
        (pkg / "my_tool.py").write_text(textwrap.dedent('''
            @tool(key="my-tool")
            def run_my_tool():
                pass
        '''))
        domain = {
            "name": "Tools",
            "root": "pkg",
            "registry_decorators": ["tool"],
            "checks": ["registry_exports"],
        }
        # No package_root → only immediate parent check (which passes)
        errors, _ = validate_domain(domain, tmp_path, set())
        assert len(errors) == 0


# =============================================================================
# Package Root Export Tests
# =============================================================================


class TestValidatePackageRootExports:
    def _scan_decorated(self, pkg: Path, decorator_names):
        """Helper to scan decorated names once for a package dir."""
        result = {}
        for py_file in sorted(pkg.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            decorated = extract_decorated_names(py_file, decorator_names)
            if decorated:
                result[py_file] = decorated
        return result

    def test_symbol_in_both_domain_and_package_root(self, tmp_path):
        """Decorated symbol present in package root __all__ → no error."""
        domain_pkg = tmp_path / "domain"
        domain_pkg.mkdir()
        (domain_pkg / "my_constraint.py").write_text(textwrap.dedent('''
            @constraint(key="gc")
            def gc_constraint():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        write_init(pkg_root, '''
            __all__ = ["gc_constraint"]
        ''')

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, set())
        assert len(errors) == 0

    def test_symbol_missing_from_package_root(self, tmp_path):
        """Decorated symbol NOT in package root __all__ → error."""
        domain_pkg = tmp_path / "domain"
        domain_pkg.mkdir()
        (domain_pkg / "my_constraint.py").write_text(textwrap.dedent('''
            @constraint(key="gc")
            def gc_constraint():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        write_init(pkg_root, '''
            __all__ = ["something_else"]
        ''')

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, set())
        assert len(errors) == 1
        assert errors[0].symbol == "gc_constraint"

    def test_multiple_missing_symbols(self, tmp_path):
        """Multiple decorated symbols missing → multiple errors."""
        domain_pkg = tmp_path / "domain"
        domain_pkg.mkdir()
        (domain_pkg / "a.py").write_text(textwrap.dedent('''
            @constraint(key="a")
            def constraint_a():
                pass
        '''))
        (domain_pkg / "b.py").write_text(textwrap.dedent('''
            @constraint(key="b")
            def constraint_b():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        write_init(pkg_root, '''
            __all__ = []
        ''')

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, set())
        assert len(errors) == 2
        symbols = {e.symbol for e in errors}
        assert symbols == {"constraint_a", "constraint_b"}

    def test_exception_symbol_not_flagged(self, tmp_path):
        """Exception symbol missing from package root → no error."""
        domain_pkg = tmp_path / "domain"
        domain_pkg.mkdir()
        (domain_pkg / "my_constraint.py").write_text(textwrap.dedent('''
            @constraint(key="gc")
            def gc_constraint():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        write_init(pkg_root, '''
            __all__ = []
        ''')

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, {"gc_constraint"})
        assert len(errors) == 0

    def test_package_root_no_all(self, tmp_path):
        """Package root __init__.py with no __all__ → symbols flagged."""
        domain_pkg = tmp_path / "domain"
        domain_pkg.mkdir()
        (domain_pkg / "my_constraint.py").write_text(textwrap.dedent('''
            @constraint(key="gc")
            def gc_constraint():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        write_init(pkg_root, '''
            from .sub import something
        ''')

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, set())
        assert len(errors) == 1
        assert errors[0].symbol == "gc_constraint"

    def test_package_root_init_missing(self, tmp_path):
        """Package root dir exists but has no __init__.py → symbols flagged."""
        domain_pkg = tmp_path / "domain"
        domain_pkg.mkdir()
        (domain_pkg / "my_constraint.py").write_text(textwrap.dedent('''
            @constraint(key="gc")
            def gc_constraint():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        # No __init__.py

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, set())
        assert len(errors) == 1
        assert errors[0].symbol == "gc_constraint"

    def test_no_decorated_symbols(self, tmp_path):
        """No decorated symbols in domain → no errors."""
        domain_pkg = tmp_path / "domain"
        domain_pkg.mkdir()
        (domain_pkg / "helper.py").write_text(textwrap.dedent('''
            def plain_function():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        write_init(pkg_root, '''
            __all__ = []
        ''')

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, set())
        assert len(errors) == 0

    def test_subdirectory_decorated_symbols(self, tmp_path):
        """Decorated symbols in subdirectories are also checked."""
        domain_pkg = tmp_path / "domain"
        sub = domain_pkg / "sub"
        sub.mkdir(parents=True)
        write_init(sub, '''
            __all__ = ["nested_constraint"]
        ''')
        (sub / "nested.py").write_text(textwrap.dedent('''
            @constraint(key="nested")
            def nested_constraint():
                pass
        '''))

        pkg_root = tmp_path / "pkg_root"
        pkg_root.mkdir()
        write_init(pkg_root, '''
            __all__ = []
        ''')

        decorated_by_file = self._scan_decorated(domain_pkg, {"constraint"})
        errors = validate_package_root_exports(pkg_root, decorated_by_file, set())
        assert len(errors) == 1
        assert errors[0].symbol == "nested_constraint"
