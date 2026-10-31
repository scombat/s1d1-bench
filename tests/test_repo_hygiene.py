"""Checks on the repository as a public artefact: licences, packaging metadata and documentation.

Each test is small and self-contained. Shared helpers sit at the top of the module.
"""

import csv
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


REPORT_DIR = REPO_ROOT / "report" / "base-v1"


def _squash(text: str) -> str:
    """Quote markers dropped from the start of each line, then every run of whitespace turned into one space."""
    return " ".join(re.sub(r"^>[ \t]?", "", text, flags=re.MULTILINE).split())


def _report_sentence() -> str:
    """The block quote under the `La phrase de docs/32 § 5` heading of the report index (a link in it is tolerated)."""
    lines = (REPORT_DIR / "index.md").read_text(encoding="utf-8").splitlines()
    heading = next(i for i, line in enumerate(lines) if re.match(r"## .*La phrase de .*§ 5", line))
    quote = []
    for line in lines[heading + 1 :]:
        if line.startswith(">"):
            quote.append(line)
        elif quote:
            break
    assert quote, "no block quote under the docs/32 § 5 heading"
    return _squash("\n".join(quote))


def _csv_rows(name: str, key: str) -> dict[str, dict[str, str]]:
    with (REPORT_DIR / "tables" / name).open(encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _french(value: str) -> str:
    """A CSV float as the report index writes it: three decimals, decimal comma, true minus sign."""
    return f"{float(value):.3f}".replace(".", ",").replace("-", "\N{MINUS SIGN}")


def test_readme_results_match_the_report():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert _report_sentence() in _squash(readme)

    results = _readme_section("Résultats (`base-v1`, 2026-10-30)")
    candidates = _csv_rows("agreement_all.csv", "name")
    shown = ("s1d1-full-int8", "s1d1-small-int8", "length", "laya-english-tier", "laya-multilingual-tier")
    for name in (*shown, "litellm-heuristic"):
        row = candidates[name]
        assert f"{_french(row['qwk'])} [{_french(row['qwk_lo'])} ; {_french(row['qwk_hi'])}]" in results, name
        assert f"{_french(row['exact'])} [{_french(row['exact_lo'])} ; {_french(row['exact_hi'])}]" in results, name
    mean = _csv_rows("ceiling.csv", "judge")["mean"]
    assert f"{_french(mean['qwk'])} [{_french(mean['qwk_lo'])} ; {_french(mean['qwk_hi'])}]" in results
    assert f"{_french(mean['exact'])} [{_french(mean['exact_lo'])} ; {_french(mean['exact_hi'])}]" in results


def test_readme_results_show_both_figures_and_link_the_report():
    results = _readme_section("Résultats (`base-v1`, 2026-10-30)")
    for figure in ("qwk-bars.svg", "qwk-by-language.svg"):
        path = f"report/base-v1/figures/{figure}"
        assert f"]({path})" in results, figure
        assert (REPO_ROOT / path).is_file(), figure
    assert "](report/base-v1/index.md)" in results
