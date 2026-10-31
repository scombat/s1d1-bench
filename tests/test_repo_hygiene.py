"""Checks on the repository as a public artefact: licences, packaging metadata and documentation.

Each test is small and self-contained. Shared helpers sit at the top of the module.
"""

import re
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


CITING_FILES = ("README.md", "judges/README.md", "report/TEMPLATE.md", "report/base-v1/index.md", "data/README.md")


def test_every_cited_doc_is_present_or_explained():
    """Each `docs/NN` cited by a public document is a copied `docs/NN-*.md` or is named in `docs/README.md`."""
    present = {path.name[:2] for path in (REPO_ROOT / "docs").glob("[0-9][0-9]-*.md")}
    index = REPO_ROOT / "docs" / "README.md"
    named = set(re.findall(r"docs/(\d{2})", index.read_text(encoding="utf-8"))) if index.is_file() else set()
    unexplained = {}
    for name in CITING_FILES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        missing = sorted(set(re.findall(r"docs/(\d{2})", text)) - present - named)
        if missing:
            unexplained[name] = missing
    assert unexplained == {}
