#!/usr/bin/env python3
"""
Documentation generator for Proto Language.

This script auto-generates MDX documentation from:
1. Python docstrings (constraints, generators, optimizers)
2. Tool README.md files

Run from repository root:
    python scripts/generate_docs.py

Dependencies:
    pip install docstring-parser
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import docstring_parser, provide helpful error if missing
try:
    from docstring_parser import parse as parse_docstring, DocstringStyle, Docstring
    from docstring_parser.common import ParseError
except ImportError:
    print("Error: docstring-parser not installed.")
    print("Install with: pip install docstring-parser")
    sys.exit(1)


def safe_parse_docstring(text: str) -> Docstring:
    """Parse docstring safely, returning empty docstring on error."""
    if not text:
        return Docstring()
    try:
        return parse_docstring(text, style=DocstringStyle.GOOGLE)
    except ParseError:
        # Fall back to trying other styles or return basic docstring
        try:
            return parse_docstring(text, style=DocstringStyle.NUMPYDOC)
        except ParseError:
            # Create a basic docstring with just the text as description
            doc = Docstring()
            doc.short_description = extract_first_paragraph(text)
            doc.long_description = text
            return doc

from pydantic import BaseModel


# =============================================================================
# Configuration
# =============================================================================

DOCS_DIR = PROJECT_ROOT / "docs"
LANGUAGE_DIR = DOCS_DIR / "language"
TOOLS_DIR = DOCS_DIR / "tools"

# Tool categories and their source directories
TOOL_CATEGORIES = {
    "structure-prediction": "proto_language/tools/structure_prediction",
    "language-models": "proto_language/tools/language_models",
    "sequence-scoring": "proto_language/tools/sequence_scoring",
    "inverse-folding": "proto_language/tools/inverse_folding",
    "gene-annotation": "proto_language/tools/gene_annotation",
    "orf-prediction": "proto_language/tools/orf_prediction",
    "rna-splicing": "proto_language/tools/rna_splicing",
}


# =============================================================================
# Helpers
# =============================================================================

def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    return text.lower().replace("_", "-").replace(" ", "-")


def escape_mdx(text: str) -> str:
    """Escape characters that have special meaning in MDX.
    
    In MDX, `<` followed by a letter or number is interpreted as a JSX tag.
    We need to escape `<` when followed by digits (e.g., `<70` becomes `\\<70`).
    """
    if not text:
        return text
    # Escape < followed by a digit (e.g., <70, <0.5)
    return re.sub(r'<(\d)', r'\\<\1', text)


def extract_first_paragraph(text: str) -> str:
    """Extract first paragraph from text."""
    if not text:
        return ""
    paragraphs = text.strip().split("\n\n")
    return paragraphs[0].replace("\n", " ").strip()


def get_config_schema(config_class: Type[BaseModel]) -> Dict[str, Any]:
    """Get JSON schema from Pydantic config class."""
    return config_class.model_json_schema()


def parse_config_params(config_class: Type[BaseModel]) -> List[Dict[str, Any]]:
    """
    Parse parameters from a Pydantic config class.
    
    Returns list of dicts with keys: name, type, required, default, description, advanced, hidden
    """
    schema = get_config_schema(config_class)
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    
    params = []
    for name, prop in properties.items():
        param = {
            "name": name,
            "type": prop.get("type", "any"),
            "required": name in required,
            "default": prop.get("default"),
            "description": prop.get("description", ""),
            "title": prop.get("title", name),
            "advanced": prop.get("advanced", False),
            "hidden": prop.get("hidden", False),
        }
        
        # Handle anyOf types (e.g., Optional[str])
        if "anyOf" in prop:
            types = [t.get("type", "null") for t in prop["anyOf"]]
            param["type"] = " | ".join(t for t in types if t != "null")
        
        # Handle enum types
        if "enum" in prop:
            param["type"] = "enum"
            param["enum_values"] = prop["enum"]
        
        # Handle array types
        if param["type"] == "array" and "items" in prop:
            item_type = prop["items"].get("type", "any")
            param["type"] = f"List[{item_type}]"
        
        params.append(param)
    
    return params


def format_param_field(param: Dict[str, Any]) -> str:
    """Format a parameter as ParamField component."""
    required_str = "required" if param["required"] else ""
    default_str = f' default="{param["default"]}"' if param.get("default") is not None else ""
    
    field = f'<ParamField path="{param["name"]}" type="{param["type"]}" {required_str}{default_str}>\n'
    field += f'  {param["description"]}\n'
    
    if param.get("enum_values"):
        field += f'  \n  Options: `{"`, `".join(str(v) for v in param["enum_values"])}`\n'
    
    field += '</ParamField>\n'
    return field


def format_params_table(params: List[Dict[str, Any]]) -> str:
    """Format parameters as a markdown table."""
    if not params:
        return "_No parameters_"
    
    # Filter out hidden params
    visible_params = [p for p in params if not p.get("hidden")]
    if not visible_params:
        return "_No parameters_"
    
    lines = ["| Parameter | Type | Required | Default | Description |"]
    lines.append("|-----------|------|----------|---------|-------------|")
    
    for p in visible_params:
        required = "Yes" if p["required"] else "No"
        default = f'`{p["default"]}`' if p.get("default") is not None else "-"
        desc = p["description"][:80] + "..." if len(p["description"]) > 80 else p["description"]
        desc = escape_mdx(desc)  # Escape MDX special characters
        lines.append(f'| `{p["name"]}` | `{p["type"]}` | {required} | {default} | {desc} |')
    
    return "\n".join(lines)


# =============================================================================
# Constraint Documentation Generator
# =============================================================================

def generate_constraint_docs() -> List[str]:
    """Generate MDX documentation for all registered constraints."""
    # Import constraint modules to trigger registration
    from proto_language.language.constraint import (
        constraint_registry,
        protein_quality,
        protein_structure,
        rna_splicing,
        sequence_annotation,
        sequence_composition,
    )
    from proto_language.language.constraint.constraint_registry import ConstraintRegistry
    
    output_dir = LANGUAGE_DIR / "constraints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_pages = []
    
    for spec in ConstraintRegistry.list_all():
        # Parse docstrings
        func_doc = safe_parse_docstring(spec.function.__doc__ or "")
        config_doc = safe_parse_docstring(spec.config_model.__doc__ or "")
        
        # Get parameters from config class
        params = parse_config_params(spec.config_model)
        
        # Build MDX content (escape special characters)
        short_desc = escape_mdx(func_doc.short_description or spec.description)
        long_desc = escape_mdx(func_doc.long_description or "")
        
        mdx = f"""---
title: "{spec.label}"
description: "{spec.description}"
---

# {spec.label}

{short_desc}

{long_desc}

## Parameters

{format_params_table(params)}

"""
        
        # Add usage example
        if func_doc.examples:
            mdx += "## Usage\n\n```python\n"
            for example in func_doc.examples:
                mdx += f"{example.description}\n"
            mdx += "```\n\n"
        else:
            # Generate a default example
            mdx += f"""## Usage

```python
from proto_language.language.core import Constraint
from proto_language.language.constraint import {spec.function.__name__}, {spec.config_model.__name__}

# Create constraint
constraint = Constraint(
    inputs=[segment],
    function={spec.function.__name__},
    function_config={spec.config_model.__name__}(
        # Configure parameters here
    ),
)

# Evaluate
scores = constraint.evaluate()
```

"""
        
        # Add returns section
        if func_doc.returns:
            returns_desc = escape_mdx(func_doc.returns.description)
            mdx += f"## Returns\n\n{returns_desc}\n\n"
        
        # Add notes if present
        notes = [m for m in func_doc.meta if m.args == ["notes"] or (hasattr(m, "key") and m.key == "Notes")]
        if notes:
            mdx += "## Notes\n\n"
            for note in notes:
                mdx += f"{note.description}\n\n"
        
        # Add metadata section
        mdx += f"""## Metadata

| Property | Value |
|----------|-------|
| Key | `{spec.key}` |
| Category | `{spec.category or "general"}` |
| Batched | `{spec.batched}` |
| Concatenate | `{spec.concatenate}` |
| GPU Required | `{spec.gpu_required}` |
| Tools Called | {", ".join(f"`{t}`" for t in spec.tools_called) or "None"} |
"""
        
        # Write file
        filename = f"{spec.key}.mdx"
        output_path = output_dir / filename
        output_path.write_text(mdx)
        
        page_path = f"language/constraints/{spec.key}"
        generated_pages.append(page_path)
        print(f"  Generated: {output_path.relative_to(PROJECT_ROOT)}")
    
    return generated_pages


# =============================================================================
# Generator Documentation Generator
# =============================================================================

def generate_generator_docs() -> List[str]:
    """Generate MDX documentation for all registered generators."""
    # Import generator modules to trigger registration
    from proto_language.language.generator import (
        generator_registry,
        esm2_generator,
        esm3_generator,
        evo2_generator,
        progen2_generator,
        proteinmpnn_generator,
        uniform_mutation_generator,
    )
    from proto_language.language.generator.generator_registry import GeneratorRegistry
    
    output_dir = LANGUAGE_DIR / "generators"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_pages = []
    
    for spec in GeneratorRegistry.list_all():
        # Parse docstrings
        class_doc = safe_parse_docstring(spec.generator_class.__doc__ or "")
        config_doc = safe_parse_docstring(spec.config_model.__doc__ or "")
        
        # Get parameters from config class
        params = parse_config_params(spec.config_model)
        
        # Build MDX content (escape special characters)
        short_desc = escape_mdx(class_doc.short_description or spec.description)
        long_desc = escape_mdx(class_doc.long_description or "")
        
        mdx = f"""---
title: "{spec.label}"
description: "{spec.description}"
---

# {spec.label}

{short_desc}

{long_desc}

## Parameters

{format_params_table(params)}

"""
        
        # Add usage example
        if class_doc.examples:
            mdx += "## Usage\n\n```python\n"
            for example in class_doc.examples:
                mdx += f"{example.description}\n"
            mdx += "```\n\n"
        else:
            # Generate a default example
            class_name = spec.generator_class.__name__
            config_name = spec.config_model.__name__
            mdx += f"""## Usage

```python
from proto_language.language.generator import {class_name}, {config_name}
from proto_language.language.core import Segment

# Create config
config = {config_name}(
    # Configure parameters here
)

# Create generator
generator = {class_name}(config)

# Assign to segment and sample
segment = Segment(length=100, sequence_type="protein")
generator.assign(segment)
generator.sample()
```

"""
        
        # Add metadata section
        seq_types = ", ".join(f"`{t}`" for t in spec.supported_sequence_types) if spec.supported_sequence_types else "All"
        mdx += f"""## Metadata

| Property | Value |
|----------|-------|
| Key | `{spec.key}` |
| Category | `{spec.category}` |
| Requires GPU | `{spec.requires_gpu}` |
| Supported Sequence Types | {seq_types} |
| Tools Called | {", ".join(f"`{t}`" for t in spec.tools_called) or "None"} |
"""
        
        # Write file
        filename = f"{spec.key}.mdx"
        output_path = output_dir / filename
        output_path.write_text(mdx)
        
        page_path = f"language/generators/{spec.key}"
        generated_pages.append(page_path)
        print(f"  Generated: {output_path.relative_to(PROJECT_ROOT)}")
    
    return generated_pages


# =============================================================================
# Optimizer Documentation Generator
# =============================================================================

def generate_optimizer_docs() -> List[str]:
    """Generate MDX documentation for all registered optimizers."""
    # Import optimizer modules to trigger registration
    from proto_language.language.optimizer import (
        optimizer_registry,
        beam_search_optimizer,
        mcmc_optimizer,
        topk_optimizer,
    )
    from proto_language.language.optimizer.optimizer_registry import OptimizerRegistry
    
    output_dir = LANGUAGE_DIR / "optimizers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_pages = []
    
    for spec in OptimizerRegistry.list_all():
        # Parse docstrings
        class_doc = safe_parse_docstring(spec.optimizer_class.__doc__ or "")
        config_doc = safe_parse_docstring(spec.config_model.__doc__ or "")
        
        # Get parameters from config class
        params = parse_config_params(spec.config_model)
        
        # Build MDX content (escape special characters)
        short_desc = escape_mdx(class_doc.short_description or spec.description)
        long_desc = escape_mdx(class_doc.long_description or "")
        
        mdx = f"""---
title: "{spec.label}"
description: "{spec.description}"
---

# {spec.label}

{short_desc}

{long_desc}

## Parameters

{format_params_table(params)}

"""
        
        # Add usage example
        if class_doc.examples:
            mdx += "## Usage\n\n```python\n"
            for example in class_doc.examples:
                mdx += f"{example.description}\n"
            mdx += "```\n\n"
        else:
            # Generate a default example
            class_name = spec.optimizer_class.__name__
            config_name = spec.config_model.__name__
            mdx += f"""## Usage

```python
from proto_language.language.optimizer import {class_name}, {config_name}
from proto_language.language.core import Construct, Constraint, Program

# Create config
config = {config_name}(
    # Configure parameters here
)

# Create optimizer
optimizer = {class_name}(
    constructs=[construct],
    generators=[generator],
    constraints=[constraint],
    config=config,
)

# Run optimization
program = Program(optimizers=[optimizer])
program.run()

# Get results
results = program.constructs[0].joined_sequences
```

"""
        
        # Add metadata section
        mdx += f"""## Metadata

| Property | Value |
|----------|-------|
| Key | `{spec.key}` |
"""
        
        # Write file
        filename = f"{spec.key}.mdx"
        output_path = output_dir / filename
        output_path.write_text(mdx)
        
        page_path = f"language/optimizers/{spec.key}"
        generated_pages.append(page_path)
        print(f"  Generated: {output_path.relative_to(PROJECT_ROOT)}")
    
    return generated_pages


# =============================================================================
# Tool README Documentation Generator
# =============================================================================

def find_tool_readmes(category_dir: Path) -> List[Dict[str, Any]]:
    """Find all README.md files in a tool category directory."""
    tools = []
    
    if not category_dir.exists():
        return tools
    
    for item in category_dir.iterdir():
        if item.is_dir():
            readme_path = item / "README.md"
            if readme_path.exists():
                tools.append({
                    "name": item.name,
                    "readme_path": readme_path,
                })
    
    return tools


def extract_readme_title(content: str) -> str:
    """Extract title from README content (first H1)."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1) if match else "Untitled"


def extract_readme_description(content: str) -> str:
    """Extract description from README (overview section or first clean paragraph)."""
    # First try to get Overview section
    overview_match = re.search(r'## Overview\s*\n+(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if overview_match:
        overview_text = overview_match.group(1).strip()
        # Get first sentence or line
        first_line = overview_text.split('\n')[0].strip()
        # Clean markdown
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', first_line)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
        if clean:
            return clean[:200]
    
    # Fallback: first paragraph after title
    content_no_title = re.sub(r'^#\s+.+\n', '', content, count=1)
    paragraphs = content_no_title.strip().split("\n\n")
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith("#") and not p.startswith("```") and not p.startswith("**") and not p.startswith("-"):
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', p)
            clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
            clean = re.sub(r'`([^`]+)`', r'\1', clean)
            clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
            clean = clean.replace("\n", " ")
            return clean[:200]
    
    return ""


def convert_readme_to_mdx(readme_path: Path, output_path: Path) -> None:
    """Convert a README.md to MDX format."""
    content = readme_path.read_text()
    
    # Extract metadata
    title = extract_readme_title(content)
    description = extract_readme_description(content)
    
    # Clean description for frontmatter (remove quotes, truncate)
    clean_desc = description.replace('"', "'").replace('\n', ' ')[:100]
    if len(description) > 100:
        clean_desc += "..."
    
    # Remove existing "Last updated" line (we'll add it automatically)
    content = re.sub(r'\n*-\s*Last updated:.*$', '', content, flags=re.MULTILINE)
    
    # Escape MDX special characters in content
    content = escape_mdx(content)
    
    # Add current date as "Last updated" footer
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Add frontmatter and updated footer
    mdx_content = f"""---
title: "{title}"
description: "{clean_desc}"
---

{content.rstrip()}

---
**Maintenance notes:**
- Last updated: {today}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mdx_content)


def generate_tool_docs() -> Dict[str, List[str]]:
    """Generate MDX documentation from tool READMEs."""
    generated_pages = {}
    
    for category_slug, source_dir in TOOL_CATEGORIES.items():
        category_path = PROJECT_ROOT / source_dir
        output_dir = TOOLS_DIR / category_slug
        
        tools = find_tool_readmes(category_path)
        if not tools:
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        category_pages = []
        
        for tool in tools:
            tool_slug = slugify(tool["name"])
            output_path = output_dir / f"{tool_slug}.mdx"
            
            convert_readme_to_mdx(tool["readme_path"], output_path)
            
            page_path = f"tools/{category_slug}/{tool_slug}"
            category_pages.append(page_path)
            print(f"  Generated: {output_path.relative_to(PROJECT_ROOT)}")
        
        generated_pages[category_slug] = category_pages
    
    return generated_pages


# =============================================================================
# Navigation Generator
# =============================================================================

def generate_navigation_snippet(
    constraint_pages: List[str],
    generator_pages: List[str],
    optimizer_pages: List[str],
    tool_pages: Dict[str, List[str]],
) -> str:
    """Generate a navigation JSON snippet for docs.json."""
    
    # Format constraint pages
    constraint_items = ",\n              ".join(f'"{p}"' for p in sorted(constraint_pages))
    generator_items = ",\n              ".join(f'"{p}"' for p in sorted(generator_pages))
    optimizer_items = ",\n              ".join(f'"{p}"' for p in sorted(optimizer_pages))
    
    # Format tool pages by category
    tool_groups = []
    category_labels = {
        "structure-prediction": "Structure Prediction",
        "language-models": "Language Models", 
        "sequence-scoring": "Sequence Scoring",
        "inverse-folding": "Inverse Folding",
        "gene-annotation": "Gene Annotation",
        "orf-prediction": "ORF Prediction",
        "rna-splicing": "RNA Splicing",
    }
    
    for category, pages in tool_pages.items():
        if pages:
            pages_str = ",\n              ".join(f'"{p}"' for p in sorted(pages))
            label = category_labels.get(category, category.replace("-", " ").title())
            tool_groups.append(f'''          {{
            "group": "{label}",
            "pages": [
              {pages_str}
            ]
          }}''')
    
    tool_groups_str = ",\n".join(tool_groups)
    
    snippet = f'''
// Add this to your docs.json "navigation.tabs" array:

{{
  "tab": "Language Reference",
  "groups": [
    {{
      "group": "Constraints",
      "pages": [
        {constraint_items}
      ]
    }},
    {{
      "group": "Generators", 
      "pages": [
        {generator_items}
      ]
    }},
    {{
      "group": "Optimizers",
      "pages": [
        {optimizer_items}
      ]
    }}
  ]
}},
{{
  "tab": "Tools",
  "groups": [
{tool_groups_str}
  ]
}}
'''
    
    return snippet


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for documentation generation."""
    print("=" * 60)
    print("Proto Language Documentation Generator")
    print("=" * 60)
    
    # Ensure docs directories exist
    LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] Generating constraint documentation...")
    constraint_pages = generate_constraint_docs()
    print(f"  Total: {len(constraint_pages)} constraints")
    
    print("\n[2/4] Generating generator documentation...")
    generator_pages = generate_generator_docs()
    print(f"  Total: {len(generator_pages)} generators")
    
    print("\n[3/4] Generating optimizer documentation...")
    optimizer_pages = generate_optimizer_docs()
    print(f"  Total: {len(optimizer_pages)} optimizers")
    
    print("\n[4/4] Generating tool documentation from READMEs...")
    tool_pages = generate_tool_docs()
    total_tools = sum(len(pages) for pages in tool_pages.values())
    print(f"  Total: {total_tools} tools")
    
    # Generate navigation snippet
    nav_snippet = generate_navigation_snippet(
        constraint_pages, generator_pages, optimizer_pages, tool_pages
    )
    
    nav_file = DOCS_DIR / "_generated_navigation.json"
    nav_file.write_text(nav_snippet)
    print(f"\n[INFO] Navigation snippet written to: {nav_file.relative_to(PROJECT_ROOT)}")
    print("       Copy the relevant sections to docs.json")
    
    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"  - Constraints: {len(constraint_pages)}")
    print(f"  - Generators:  {len(generator_pages)}")
    print(f"  - Optimizers:  {len(optimizer_pages)}")
    print(f"  - Tools:       {total_tools}")
    print("=" * 60)


if __name__ == "__main__":
    main()
