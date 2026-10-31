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
    bars = dict(drawn)["qwk-bars"]
    assert families & set(_legend_labels(bars)) == {"s1d1", "Laya", OTHERS[0]}  # the families of the fake set


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
