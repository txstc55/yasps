#!/usr/bin/env python3
"""Fast, dependency-free consistency checks for the Markdown documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path


DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parent
REFERENCE = (
    DOCS / "reference" / "core.md",
    DOCS / "reference" / "advanced.md",
    DOCS / "reference" / "internal.md",
)


def front_matter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    values: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> int:
    errors: list[str] = []
    markdown_files = sorted(DOCS.rglob("*.md"))
    pages: dict[str, Path] = {"/": DOCS / "index.md"}

    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        metadata = front_matter(text)
        relative = path.relative_to(DOCS)
        if not metadata:
            errors.append(f"{relative}: missing YAML front matter")
            continue
        if not metadata.get("title"):
            errors.append(f"{relative}: missing title")
        if path.name != "index.md" and not metadata.get("permalink"):
            errors.append(f"{relative}: missing permalink")
        if metadata.get("permalink"):
            permalink = metadata["permalink"]
            if permalink in pages:
                errors.append(
                    f"{relative}: duplicate permalink {permalink} "
                    f"(already used by {pages[permalink].relative_to(DOCS)})"
                )
            pages[permalink] = path
        if text.count("```") % 2:
            errors.append(f"{relative}: unbalanced fenced code block")

    sources = markdown_files + [DOCS / "_layouts" / "default.html"]
    relative_url_pattern = re.compile(
        r"""\{\{\s*['"](?P<url>/[^'"]*)['"]\s*\|\s*relative_url\s*\}\}"""
    )
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for match in relative_url_pattern.finditer(text):
            url = match.group("url")
            if url.startswith("/assets/"):
                asset = DOCS / url.removeprefix("/")
                if not asset.is_file():
                    errors.append(
                        f"{path.relative_to(DOCS)}: missing asset {url}"
                    )
            elif url not in pages:
                errors.append(
                    f"{path.relative_to(DOCS)}: unresolved page {url}"
                )

    init_text = (REPO / "yasps" / "yasps" / "__init__.py").read_text(
        encoding="utf-8"
    )
    explicit_exports = re.findall(
        r"^from\s+\.[A-Za-z_][A-Za-z0-9_.]*\s+import\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*$",
        init_text,
        flags=re.MULTILINE,
    )
    wildcard_modules = re.findall(
        r"^from\s+\.([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+\*\s*$",
        init_text,
        flags=re.MULTILINE,
    )
    reference_text = "\n".join(
        path.read_text(encoding="utf-8") for path in REFERENCE
    )
    for name in explicit_exports + wildcard_modules:
        if not re.search(rf"\b{re.escape(name)}\b", reference_text):
            errors.append(f"reference: package export {name!r} is undocumented")

    github_example_pattern = re.compile(
        r"https://github\.com/txstc55/yasps/tree/main/examples/"
        r"([A-Za-z0-9_-]+)"
    )
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for directory in github_example_pattern.findall(text):
            if not (REPO / "examples" / directory).is_dir():
                errors.append(
                    f"{path.relative_to(DOCS)}: example directory "
                    f"{directory!r} does not exist"
                )

    if errors:
        print("Documentation checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"Documentation checks passed: {len(markdown_files)} Markdown pages, "
        f"{len(pages)} routes, {len(explicit_exports)} explicit exports."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
