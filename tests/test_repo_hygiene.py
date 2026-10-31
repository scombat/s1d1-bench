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


def _github_slug(title: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped (but - and _), spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def test_readme_internal_anchors_resolve():
    """Each `](#anchor)` of the README is the slug of one of its headings (outside the fenced code blocks)."""
    anchors, in_fence = set(), False
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(r"#{1,6} ", line):
            anchors.add(_github_slug(line.lstrip("#").strip()))
    links = re.findall(r"\]\(#([^)]+)\)", text)
    assert links
    assert [link for link in links if link not in anchors] == []


def test_license_files_exist():
    entries = _pyproject()["project"]["license-files"]
    assert "src/s1d1_bench/vendor/litellm_heuristic/LICENSE" in entries
    missing = [entry for entry in entries if not (REPO_ROOT / entry).is_file()]
    assert missing == []


def test_readme_names_the_mit_copy():
    section = _readme_section("Licences")
    assert "MIT" in section
    assert "src/s1d1_bench/vendor/litellm_heuristic/" in section


CITING_FILES = (
    "README.md",
    "judges/README.md",
    "judges/MODELS.md",
    "report/TEMPLATE.md",
    "report/base-v1/index.md",
    "data/README.md",
)


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


RESULTS_TITLE = "Results (`base-v1`, 2026-10-30)"


def _csv_rows(name: str, key: str) -> dict[str, dict[str, str]]:
    with (REPORT_DIR / "tables" / name).open(encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _english(value: str) -> str:
    """A CSV float as the README writes it: three decimals, decimal point, true minus sign."""
    return f"{float(value):.3f}".replace("-", "\N{MINUS SIGN}")


def _estimate(row: dict[str, str], metric: str) -> str:
    """`0.807 [0.734, 0.852]` from the columns `<metric>`, `<metric>_lo`, `<metric>_hi` of a CSV row."""
    return f"{_english(row[metric])} [{_english(row[metric + '_lo'])}, {_english(row[metric + '_hi'])}]"


def _numbers(text: str, decimal: str) -> list[str]:
    """The numbers of a sentence, written with `decimal` as the decimal mark, as plain `0.807` strings."""
    return [token.replace(decimal, ".") for token in re.findall(rf"\d+(?:{re.escape(decimal)}\d+)?", text)]


def _readme_quote(section: str) -> str:
    """The first block quote of a README section, squashed."""
    quote = []
    for line in section.splitlines():
        if line.startswith(">"):
            quote.append(line)
        elif quote:
            break
    return _squash("\n".join(quote))


def test_readme_results_quote_the_4b_model_without_gemma():
    """The 4B model is of the family of a main judge: the README gives it, and s1d1-full-int8, against the median of the
    two other judges (`truth_all_nogemma`), as `agreement_nogemma.csv` has them."""
    results = " ".join(_readme_section(RESULTS_TITLE).split())
    rows = _csv_rows("agreement_nogemma.csv", "name")
    for name in ("llm4b-gemma4-e4b-off", "s1d1-full-int8"):
        row = rows[name]
        qwk = f"{_english(row['qwk'])} [{_english(row['qwk_lo'])}, {_english(row['qwk_hi'])}]"
        assert qwk in results, name
    assert "(`truth_all_nogemma`)" in results and "agreement_nogemma.csv" in results


def test_readme_results_match_the_report():
    """The README quotes the sentence of docs/32 § 5 in English: same numbers, in the same order, as the report's."""
    results = _readme_section(RESULTS_TITLE)
    sentence = _readme_quote(results)
    assert sentence and "2 σ rule" in sentence
    assert _numbers(sentence, ".") == _numbers(_report_sentence().replace("−", ""), ",")
    assert sentence.count("\N{MINUS SIGN}") == _report_sentence().count("\N{MINUS SIGN}")

    candidates = _csv_rows("agreement_all.csv", "name")
    shown = (
        "s1d1-full-int8", "s1d1-small-int8", "length", "laya-english-tier", "laya-multilingual-tier",
        "routellm-bert", "llm4b-gemma4-e4b-off", "nvidia-ptcc-eng",
    )  # fmt: skip
    for name in (*shown, "litellm-heuristic"):
        row = candidates[name]
        assert _estimate(row, "qwk") in results, name
        assert _estimate(row, "exact") in results, name
    mean = _csv_rows("ceiling.csv", "judge")["mean"]
    assert _estimate(mean, "qwk") in results
    assert _estimate(mean, "exact") in results


README_FIGURES = ("qwk-bars.svg", "quality-vs-latency.svg", "qwk-by-language.svg")


def test_readme_results_show_the_three_figures_with_a_caption_and_link_the_report():
    results = _readme_section(RESULTS_TITLE)
    places = []
    for figure in README_FIGURES:
        path = f"report/base-v1/figures/{figure}"
        assert f"]({path})" in results, figure
        assert (REPO_ROOT / path).is_file(), figure
        places.append(results.index(f"]({path})"))
    assert places == sorted(places)  # in this order
    # each figure is followed by its caption, a paragraph before the next figure
    blocks = re.split(r"!\[[^\]]*\]\(report/base-v1/figures/[^)]+\)", results)[1:]
    for figure, block in zip(README_FIGURES, blocks, strict=True):
        assert block.strip() and not block.strip().startswith("|"), figure
    assert "](report/base-v1/index.md)" in results
    table = [line for line in results.splitlines() if line.startswith("| ") and not line.startswith("| candidate")]
    assert len(table) == 10  # the seven lines of the first lot, RouteLLM bert and the 4B model of lot B, NVIDIA


def _span(values: list[float], unit) -> str:
    """'1.6 to 9.1 ms': the two ends, the unit once when they share it."""
    low, high = unit(min(values)), unit(max(values))
    number, _, word = low.rpartition(" ")
    return f"{number if word == high.rpartition(' ')[2] else low} to {high}"


def _ms(value: float) -> str:
    """A latency as the README writes it: ms under a second, s above, one decimal."""
    if value < 1000:
        return f"{value:.1f} ms"
    return f"{value / 1000:.1f} s"


def _mb(value: float) -> str:
    """A weight as the README writes it: MB without decimal under 1,000 MB, GB with one decimal above."""
    if value < 1000:
        return f"{value:.0f} MB"
    return f"{value / 1000:.1f} GB"


def test_readme_results_latency_and_weights_match_the_profile():
    results = _readme_section(RESULTS_TITLE)
    flat = " ".join(results.split())  # the README wraps its lines: look for phrases, not for line breaks
    profile = list(_csv_rows("profile.csv", "name").values())
    s1d1_cpu = [r for r in profile if r["name"].startswith("s1d1-") and r["device"] == "cpu"]
    s1d1 = [r for r in profile if r["name"].startswith("s1d1-")]
    judges = [r for r in profile if r["role"] == "judge-in-truth"]
    assert len(s1d1_cpu) == 4 and len(s1d1) == 5 and len(judges) == 3
    assert _span([float(r["latency_p50_ms"]) for r in s1d1_cpu], _ms) in results
    assert _span([float(r["latency_p50_ms"]) for r in judges], _ms) in results
    assert _span([float(r["weights_mb"]) for r in s1d1], _mb) in results
    assert _span([float(r["weights_mb"]) for r in judges], _mb) in results
    # the judge-level quality is said of s1d1-full-int8 alone, with its own latency and weight
    full = next(r for r in profile if r["name"] == "s1d1-full-int8")
    assert "`s1d1-full-int8`, whose CI contains the ceiling mean" in flat
    assert full["device"] == "cpu"
    assert f"answers in {_ms(float(full['latency_p50_ms']))} per prompt on the Mac's CPU" in flat
    assert f"`s1d1-full-int8` weighs {_mb(float(full['weights_mb']))}" in flat
    assert "At a QWK of the order of a judge's" not in flat
    # the tail of s1d1-full-int8, without a cause, and the judges' latency with and without reasoning
    p95 = _ms(float(full["latency_p95_ms"]))
    tail = f"At the 95th percentile, `s1d1-full-int8` takes {p95} (the cause of this tail is not"
    assert tail in flat
    off = [r for r in profile if r["name"].startswith("base-v1-") and r["name"].endswith("-off")]
    assert {r["name"] for r in off} == {"base-v1-gemma4-off", "base-v1-qwen3-off"}
    assert "and then reasoning (reasoning capped at 512 tokens)" in flat
    no_reasoning = _span([float(r["latency_p50_ms"]) for r in off], _ms)
    assert f"without reasoning, gemma4 and qwen3 take {no_reasoning}." in flat
    report_section = _readme_section("The report")
    assert "at rest" not in report_section and "idle" not in report_section  # no load of the machine is recorded


def test_readme_results_by_language_match_the_report():
    results = _readme_section(RESULTS_TITLE)
    with (REPORT_DIR / "tables" / "by_language.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for name in ("s1d1-full-int8", "litellm-heuristic"):
        qwk = {r["lang"]: float(r["qwk_all"]) for r in rows if r["name"] == name}
        others = [v for lang, v in qwk.items() if lang != "eng"]
        assert f"{_english(str(qwk['eng']))} in English" in results, name
        assert f"{_english(str(min(others)))} to {_english(str(max(others)))}" in results, name


WITNESSES = {"random", "length", "constant-low"}


def test_profile_figures_of_the_report_leave_the_witnesses_out_and_say_it():
    """Review focus 4, on the committed report: no witness point, LiteLLM (no weights) out of the footprint figure."""
    figures = REPORT_DIR / "figures"
    for name in ("quality-vs-latency", "quality-vs-footprint"):
        with (figures / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
            names = {row["name"] for row in csv.DictReader(handle)}
        assert names and not names & WITNESSES, name
        assert "s1d1-full-int8" in names and "base-v1-gemma4-cap512" in names, name
        assert not {"base-v1-gemma4-off", "base-v1-qwen3-off"} & names, name  # the main judges only
    with (figures / "quality-vs-footprint.csv").open(encoding="utf-8", newline="") as handle:
        assert "litellm-heuristic" not in {row["name"] for row in csv.DictReader(handle)}
    index = (REPORT_DIR / "index.md").read_text(encoding="utf-8")
    assert "ne sont pas des modèles" in index and "`litellm-heuristic` : pas de fichier de poids (0 Mo)" in index


# the files whose strings check, on purpose, that a local path is refused or kept out of meta.json
LOCAL_PATH_EXCEPTIONS = {
    "tests/test_check.py",
    "tests/test_judge_runner.py",
    "tests/test_translate.py",
    "tests/test_classify.py",
    "tests/test_laya_backend.py",
    "tests/test_routellm_backend.py",
    "tests/test_nvidia_backend.py",
}


def test_no_local_paths():
    """No `/Users/` nor `~/work/` in the code, the tests and the public texts, but in the named test files above."""
    files = [
        *(REPO_ROOT / "src").rglob("*.py"),
        *(REPO_ROOT / "tests").rglob("*.py"),
        *(REPO_ROOT / "docs").glob("*.md"),
        REPO_ROOT / "README.md",
        REPO_ROOT / "judges" / "README.md",
        REPO_ROOT / "report" / "TEMPLATE.md",
        REPO_ROOT / "data" / "README.md",
    ]
    vendor = REPO_ROOT / "src" / "s1d1_bench" / "vendor"
    found = []
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in LOCAL_PATH_EXCEPTIONS or path.is_relative_to(vendor) or path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "~/work/" in text:
            found.append(relative)
    assert found == []
