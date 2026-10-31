"""The figure style of `bench report` (docs/36 § 2): English only, no hatch nor marker shape for identity, the legend
one entry per family (and per reference line), every figure with its CSV."""

import re

import pytest
from test_report import fake_report_inputs, small  # noqa: F401 (the fixture `small`)

ACCENTED = re.compile("[À-ÖØ-öø-ÿĀ-ſ]")  # every Latin letter with a diacritic (Latin-1 and Extended-A)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """`bench report` on the fake set of test_report.py: the output folder."""
    pytest.importorskip("matplotlib")
    from s1d1_bench.cli import main

    tmp_path = tmp_path_factory.mktemp("figures")
    args, _ = fake_report_inputs(tmp_path)
    out = tmp_path / "out"
    assert main([*args, "--out", str(out)]) == 0
    return out


def test_no_french_and_no_hatch_in_svgs(generated):
    svgs = sorted((generated / "figures").glob("*.svg"))
    assert svgs
    for svg in svgs:
        text = svg.read_text(encoding="utf-8")
        assert not ACCENTED.findall(text), (svg.name, sorted(set(ACCENTED.findall(text))))
        assert "<pattern" not in text, svg.name  # a hatch is drawn as an SVG pattern


def test_no_french_and_no_accent_in_figure_csvs(generated):
    for path in sorted((generated / "figures").glob("*.csv")):
        text = path.read_text(encoding="utf-8")
        assert not ACCENTED.findall(text), (path.name, sorted(set(ACCENTED.findall(text))))


def test_each_figure_has_csv(generated):
    from s1d1_bench.report_figures import NAMES

    folder = generated / "figures"
    assert {p.stem for p in folder.glob("*.svg")} == set(NAMES)
    assert {p.stem for p in folder.glob("*.csv")} == set(NAMES)


def _drawn(computed):
    """Every figure of `_DRAW` drawn in the style, as (name, matplotlib figure)."""
    import matplotlib
    import matplotlib.colors
    import matplotlib.figure
    import matplotlib.lines
    import matplotlib.patches
    import matplotlib.ticker

    from s1d1_bench.report_figures import _DRAW, style

    out = []
    with style():
        for name, draw in _DRAW.items():
            fig, _, _, _ = draw(computed, 0.005, matplotlib)
            if fig is not None:
                out.append((name, fig))
    return out


def _legend_labels(fig) -> list[str]:
    legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes)]
    return [t.get_text() for legend in legends if legend is not None for t in legend.get_texts()]


def test_legend_entries_are_families(small):  # noqa: F811
    pytest.importorskip("matplotlib")
    from s1d1_bench.report_figures import FAMILIES, OTHERS

    computed, _ = small
    candidates = {r["name"] for r in computed["agreement_all"].to_pylist()}
    families = {label for _, label, _ in FAMILIES} | {OTHERS[0]}
    drawn = _drawn(computed)
    assert {name for name, _ in drawn} >= {"qwk-bars", "qwk-by-language", "needs-tools-pr"}
    for name, fig in drawn:
        labels = _legend_labels(fig)
        assert len(labels) == len(set(labels)), (name, labels)  # at most one entry per family and reference line
        assert not candidates & set(labels), (name, labels)  # never one entry per variant
        assert len(fig.legends) <= 1 and all(ax.get_legend() is None for ax in fig.axes), name  # below, one legend
        # every entry is a family or one of the named reference lines of that figure, nothing else
        others = [text for text in labels if text not in families]
        assert all(any(re.fullmatch(p, text) for p in REFERENCES.get(name, ())) for text in others), (name, others)
    # the exact legend of each figure on the fake set (s1d1-full-int8, laya-rank, notask, the witnesses; no `length`
    # floor line, no positive needs_tools id)
    families_here = ["s1d1", "Laya", OTHERS[0]]
    assert {name: _legend_labels(fig) for name, fig in drawn} == {
        "qwk-bars": [
            *families_here,
            "ceiling: mean of the main judges",
            "ceiling: each main judge against the other two (8 votes)",
            "floor: random, min-max of its draws",
            "± 2σ around s1d1-full-int8 (σ = 0.005)",
        ],
        "qwk-by-language": families_here,
        "confusion": [],
        "reliability-s1d1": ["s1d1", "perfect calibration (score = anchor)"],
        "judge-spread-by-tier": [],
        "language-disagreements": [],
        "needs-tools-pr": ["s1d1", "share of positives (0/8 ids)"],
        "quality-vs-latency": [*families_here, "main judges: each against the other two (8 votes)"],
        "quality-vs-footprint": [*families_here, "main judges: each against the other two (8 votes)"],
        "tier-distribution": ["trivial", "low", "medium", "high", "expert"],
    }
    # two variants of a family in the panels: still one entry for the family, no entry for the other panels
    import pyarrow as pa

    curve = computed["needs_tools_pr"].to_pylist()
    two = dict(computed, needs_tools_pr=pa.Table.from_pylist(curve + [r | {"name": "s1d1-tiny-int8"} for r in curve]))
    assert _legend_labels(dict(_drawn(two))["needs-tools-pr"]) == ["s1d1", "share of positives (0/8 ids)"]


# the reference lines (and bands) each figure may name in its legend, beside the families
REFERENCES = {
    "qwk-bars": (
        r"ceiling: mean of the main judges",
        r"ceiling: each main judge against the other two \(8 votes\)",
        r"floor: (length|constant-low)( and (length|constant-low))?",
        r"floor: random, min-max of its draws",
        r"± 2σ around s1d1-full-int8 \(σ = [0-9.]+\)",
    ),
    "reliability-s1d1": (r"perfect calibration \(score = anchor\)",),
    "needs-tools-pr": (r"share of positives \(\d+/\d+ ids\)",),
    "quality-vs-latency": (r"main judges: each against the other two \(8 votes\)",),
    "quality-vs-footprint": (r"main judges: each against the other two \(8 votes\)",),
    "tier-distribution": (r"trivial|low|medium|high|expert",),  # the five segments of every bar: the tiers
}


def test_one_marker_shape_and_no_hatch_in_any_figure(small):  # noqa: F811
    pytest.importorskip("matplotlib")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    computed, _ = small
    for name, fig in _drawn(computed):
        artists = [a for ax in fig.axes for a in ax.get_children()]
        artists += [h for legend in fig.legends for h in legend.legend_handles]
        markers = {a.get_marker() for a in artists if isinstance(a, Line2D)} - {"None", "none", "", " ", None}
        assert markers <= {"o"}, (name, markers)
        assert not any(isinstance(a, Patch) and a.get_hatch() for a in artists), name


def test_shade_and_family_colours():
    from s1d1_bench.report_figures import colour, family, shade

    assert family("s1d1-full-int8") == ("s1d1", "#2a78d6")
    assert family("laya-english-tier") == ("Laya", "#eb6834")
    assert family("litellm-heuristic") == ("LiteLLM", "#1baf7a")
    assert family("routellm-bert") == ("RouteLLM", "#7e57c2")
    assert family("llm4b-gemma4-e4b-off") == ("LLM 4B", "#b8860b")
    assert family("length") == ("witnesses and others", "#7a7975")
    assert shade("#2a78d6", 0) == "#2a78d6"
    assert shade("#000000", 0.5) == "#808080" and shade("#ffffff", -0.5) == "#808080"
    assert colour("s1d1-full-int8") == "#2a78d6"
    assert colour("s1d1-full-fp16") == shade("#2a78d6", -0.25)
    assert colour("s1d1-compact-int8") == shade("#2a78d6", 0.25)
    assert colour("s1d1-small-int8") == shade("#2a78d6", 0.45)
    assert colour("s1d1-tiny-int8") == shade("#2a78d6", 0.6)
    assert colour("laya-multilingual-tier") == shade("#eb6834", 0.4)
    assert colour("laya-english-tier") == "#eb6834"


def test_reading_directions_of_the_two_measure_figures(generated):
    for name in ("quality-vs-latency", "quality-vs-footprint"):
        text = (generated / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        assert "better ↖" in text and "higher is better →" in text and "← lower is better" in text, name


def test_witnesses_never_get_a_point_in_the_profile_figures(small):  # noqa: F811
    """Review focus 4: the witnesses are not models; even profiled by mistake, they get no point, and they are not
    said missing either (the caption under the figure says they are left out)."""
    pytest.importorskip("matplotlib")
    import matplotlib
    import matplotlib.figure
    import matplotlib.lines
    import matplotlib.patches
    import matplotlib.ticker
    import pyarrow as pa

    from s1d1_bench.report_figures import _DRAW, WITNESSES, style

    computed, _ = small
    assert set(WITNESSES) == {"random", "length", "constant-low"}
    agreement = computed["agreement_all"].to_pylist()
    s1d1 = next(r for r in agreement if r["name"] == "s1d1-full-int8")
    profile = computed["profile"].to_pylist()
    profiled = next(r for r in profile if r["name"] == "s1d1-full-int8")
    hacked = dict(
        computed,
        agreement_all=pa.Table.from_pylist(agreement + [s1d1 | {"name": w} for w in WITNESSES]),
        profile=pa.Table.from_pylist(profile + [profiled | {"name": w} for w in WITNESSES]),
    )
    with style():
        for name in ("quality-vs-latency", "quality-vs-footprint"):
            fig, _, rows, notes = _DRAW[name](hacked, 0.005, matplotlib)
            assert fig is not None and rows, name
            assert not {r["name"] for r in rows} & set(WITNESSES), name
            assert not [n for n in notes if any(w in n for w in WITNESSES)], (name, notes)
            labels = [t.get_text() for ax in fig.axes for t in ax.texts]
            assert not [t for t in labels if any(t.startswith(w + " ") for w in WITNESSES)], name


def test_a_model_without_weights_is_left_out_of_the_footprint_figure_and_said(small):  # noqa: F811
    pytest.importorskip("matplotlib")
    import matplotlib
    import matplotlib.figure
    import matplotlib.lines
    import matplotlib.patches
    import matplotlib.ticker
    import pyarrow as pa

    from s1d1_bench.report_figures import _DRAW, style

    computed, _ = small
    profile = [r | {"weights_mb": 0.0} if r["name"] == "notask" else r for r in computed["profile"].to_pylist()]
    with style():
        _, _, rows, notes = _DRAW["quality-vs-footprint"](dict(computed, profile=pa.Table.from_pylist(profile)),
                                                           0.005, matplotlib)  # fmt: skip
        assert "notask" not in {r["name"] for r in rows}
        assert any("`notask`" in n and "0 Mo" in n for n in notes), notes
        _, _, rows, _ = _DRAW["quality-vs-latency"](dict(computed, profile=pa.Table.from_pylist(profile)), 0.005,
                                                     matplotlib)  # fmt: skip
        assert "notask" in {r["name"] for r in rows}  # its latency is still measured


def test_point_labels_name_the_device_and_judges_by_short_name(small):  # noqa: F811
    pytest.importorskip("matplotlib")
    import matplotlib
    import matplotlib.figure
    import matplotlib.lines
    import matplotlib.patches
    import matplotlib.ticker

    from s1d1_bench.report_figures import _DRAW, style

    computed, _ = small
    with style():
        fig, _, rows, _ = _DRAW["quality-vs-latency"](computed, 0.005, matplotlib)
    texts = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert "s1d1-full-int8 · CPU" in texts and "j-a · GPU (MLX)" in texts
    judges = [r for r in rows if r["kind"].startswith("main judge")]
    ceiling = {r["judge"]: r for r in computed["ceiling"].to_pylist()}
    assert {r["name"] for r in judges} == {"j-a", "j-b", "j-c"}  # the main judges only, not the -off ones
    for r in judges:  # the judge's QWK is its ceiling value: against the other two main judges, as in qwk-bars
        assert (r["qwk"], r["lo"], r["hi"]) == (ceiling[r["name"]]["qwk"], ceiling[r["name"]]["qwk_lo"],
                                                ceiling[r["name"]]["qwk_hi"])  # fmt: skip
    agreement = {r["name"]: r for r in computed["agreement_all"].to_pylist()}
    s1d1 = next(r for r in rows if r["name"] == "s1d1-full-int8")
    assert s1d1["qwk"] == agreement["s1d1-full-int8"]["qwk"] and s1d1["device"] == "cpu"
