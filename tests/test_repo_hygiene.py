"""Checks on the repository as a public artefact: licences, packaging metadata and documentation.

Each test is small and self-contained. Shared helpers sit at the top of the module.
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _readme_section(title: str) -> str:
    """Body of the `## <title>` section of README.md, up to the next level-2 heading (or the end of the file)."""
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = lines.index(f"## {title}") + 1
    body = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body)


def test_license_files_exist():
    entries = _pyproject()["project"]["license-files"]
    assert "src/s1d1_bench/vendor/litellm_heuristic/LICENSE" in entries
    missing = [entry for entry in entries if not (REPO_ROOT / entry).is_file()]
    assert missing == []


def test_readme_names_the_mit_copy():
    section = _readme_section("Licences")
    assert "MIT" in section
    assert "src/s1d1_bench/vendor/litellm_heuristic/" in section
