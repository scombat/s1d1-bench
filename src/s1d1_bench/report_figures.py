"""`bench report`, the figures: one SVG and one CSV of the same name per figure, in `<out>/figures/`, drawn from the
tables of `report.tables` (nothing is recomputed here; each CSV holds the values the figure plots).

matplotlib (the `report` dependency group) is imported inside `figures` only, so that the other commands, and
`bench --help`, never need it; the figures are made with `matplotlib.figure.Figure` (no pyplot, no GUI backend). The
SVG is byte-identical from one run to the next: a fixed `svg.hashsalt`, no date in the metadata, text kept as text
(`svg.fonttype: none`).

The style (docs/36 § 2 of the model repository) is one for every figure, and the helpers below (`style`, `title`,
`better`, `legend_below`, `label`, `colour`) carry it:

- English only in a figure (the French captions live under it, in `index.md`): no accented letter in an SVG.
- Title in bold at the top left, a grey subtitle under it with the unit and the number of ids; nothing else inside
  the frame but the marks, their value labels and their point labels.
- Each measured axis says, in grey at its end, "higher is better →" or "← lower is better".
- Identity by colour and written label, never by shape: one hue per family (s1d1 blue, Laya orange, LiteLLM green,
  witnesses grey, judges and ceiling black or dark grey); within a family, the variants differ by lightness steps of
  the same hue and by their label written beside the mark. No hatch, one marker shape (`o`), one line style per role
  of line (a reference line has its own style, a candidate's line is always solid).
- The legend sits below the frame, one entry per family and one per reference line, never one per variant; a figure
  of a single series has none.

A candidate or role a figure needs that is absent from the tables loses its panel (or its line), and a French line
saying so is returned for the index; a figure left with nothing to draw is not written (and a stale file of that name
is removed). Never an error.
"""

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from s1d1_bench.report import CONSTANT, FLOOR, LITELLM, RANDOM, REFERENCE, write_csv
from s1d1_bench.rubric import LEVELS

# the palette: text, secondary text, grid, paper; the family hues; the judges and the ceiling
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#7a7975"
JUDGE, JUDGE_SOFT = "#0b0b0b", "#52514e"
FAMILIES = (("s1d1-", "s1d1", BLUE), ("laya-", "Laya", ORANGE), ("litellm-", "LiteLLM", GREEN))
OTHERS = ("witnesses and others", GREY)
# within a family, the lightness step of a variant (shade: < 0 darker, > 0 lighter), by a word of its name
VARIANTS = {
    "s1d1-": (("fp16", -0.25), ("compact", 0.25), ("small", 0.45), ("tiny", 0.6)),
    "laya-": (("multilingual", 0.4),),
}
CONFUSION = (REFERENCE, "laya-multilingual-tier", LITELLM, FLOOR)  # the four panels of `confusion`
HEAT = (0.9, 0.0, -0.5)  # the single-hue heat maps: blue lightened (value 0), blue, blue darkened (the top)
RC = {
    "svg.hashsalt": "s1d1-bench",  # the ids of clip paths and glyphs, otherwise random
    "svg.fonttype": "none",
    "font.family": "sans-serif",  # DejaVu Sans first, then the fallbacks of matplotlib, in the SVG
    "font.size": 9,
    "text.color": INK,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "axes.unicode_minus": True,
}
NAMES = (
    "qwk-bars",
    "qwk-by-language",
    "confusion",
    "reliability-s1d1",
    "judge-spread-by-tier",
    "language-disagreements",
    "needs-tools-pr",
)

# The CSV beside each figure is in English like the figure: the header of each plotted column below. The numbers are
# those of the tables, untouched: only the header row differs from the internal column names.
HEADERS = {
    "kind": "kind", "name": "name", "family": "family", "value": "value", "lo": "lower bound", "hi": "upper bound",
    "draws_used": "draws used", "lang": "language", "n_ids": "ids", "qwk": "QWK", "candidate": "candidate",
    "truth_tier": "truth_all tier", "pred_tier": "predicted tier", "row_share": "share of the row",
    "bin_low": "bin: lower bound", "bin_high": "bin: upper bound", "mean_score": "mean score",
    "mean_anchor": "mean anchor", "tier": "tier", "spread": "spread (steps)", "n_ids_tier": "ids of the tier",
    "mean_spread_tier": "mean spread of the tier (steps)", "n_ids_null_spread_tier": "ids of the tier without spread",
    "group": "group", "n_ids_disagreement": "ids in disagreement", "share_of_ids": "share of ids",
    "n_keys": "base prompts", "n_keys_disagreement": "base prompts in disagreement", "threshold": "threshold",
    "precision": "precision", "recall": "recall",
}  # fmt: skip


class Figure(NamedTuple):
    """A figure of the report: its SVG and CSV (None when it was not drawn) and the French lines of what it left out."""

    name: str
    svg: Path | None
    csv: Path | None
    notes: tuple[str, ...]


# --- the style ------------------------------------------------------------------------------------------------------


def style():
    """The common style `RC` as a context manager (`with style(): ...`); the global rcParams are left alone."""
    import matplotlib

    return matplotlib.rc_context(RC)


def family(name: str) -> tuple[str, str]:
    """(family label, colour) of a candidate, by the prefix of its name."""
    for prefix, label, hue in FAMILIES:
        if name.startswith(prefix):
            return label, hue
    return OTHERS


def shade(colour: str, amount: float) -> str:
    """`colour` (#rrggbb) mixed with black (amount < 0) or white (amount > 0): the same hue, another lightness."""
    rgb = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
    target = 255 if amount > 0 else 0
    mixed = [round(c + (target - c) * abs(amount)) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def _step(name: str) -> float:
    """The lightness step of a candidate within its family (0 for the first variant and for every other name)."""
    for prefix, steps in VARIANTS.items():
        if name.startswith(prefix):
            return next((amount for word, amount in steps if word in name), 0.0)
    return 0.0


def colour(name: str) -> str:
    """The colour of a candidate: its family hue at the lightness of its variant."""
    return shade(family(name)[1], _step(name))


def title(ax, head: str, sub: str) -> None:
    """The bold title and the grey subtitle above the top-left corner of `ax` (for a figure of panels, the top-left
    panel; call it after the panel titles, so that it rises above them). Figure texts anchored to the axes, so that
    they follow the layout; `bbox_inches="tight"` keeps them in the saved file."""
    from matplotlib.transforms import offset_copy

    fig = ax.figure
    panels = any(a.get_title(loc) for a in fig.axes for loc in ("left", "center", "right"))
    base = 6 + (17 if panels else 0)  # points above the frame, past the panel titles
    lines = sub.count("\n") + 1
    fig.text(
        0,
        1,
        sub,
        transform=offset_copy(ax.transAxes, fig, y=base, units="points"),
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
        linespacing=1.3,
    )
    fig.text(
        0,
        1,
        head,
        transform=offset_copy(ax.transAxes, fig, y=base + 13 * lines + 2, units="points"),
        ha="left",
        va="bottom",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
    )


def better(ax, axis: str, higher: bool = True) -> None:
    """The reading direction of a measured axis, in grey at its better end: "higher is better →" at the right (top)
    end, "← lower is better" at the left (bottom) end. The axis label moves to the other end, on the same line."""
    text = "higher is better →" if higher else "← lower is better"
    grey = {"color": MUTED, "fontsize": 7.5, "annotation_clip": False}
    if axis == "x":
        ax.xaxis.label.set(x=0 if higher else 1, horizontalalignment="left" if higher else "right")
        ax.annotate(
            text,
            xy=(1 if higher else 0, 0.5),
            xycoords=(ax.transAxes, ax.xaxis.label),
            ha="right" if higher else "left",
            va="center",
            **grey,
        )
    elif axis == "y":
        ax.yaxis.label.set(y=0 if higher else 1, horizontalalignment="left" if higher else "right")
        ax.annotate(
            text,
            xy=(0.5, 1 if higher else 0),
            xycoords=(ax.yaxis.label, ax.transAxes),
            rotation=90,
            ha="center",
            va="top" if higher else "bottom",
            **grey,
        )
    else:
        raise ValueError(f"axis must be 'x' or 'y', not {axis!r}")


def legend_below(fig, handles: Sequence, ncol: int | None = None):
    """The one legend of the figure, below the frame, left-aligned with the first panel; none when `handles` is
    empty. `handles`: one per family (`family_handles`) and one per reference line."""
    if not handles:
        return None
    from matplotlib.transforms import blended_transform_factory

    anchor = blended_transform_factory(fig.axes[0].transAxes, fig.transFigure)
    return fig.legend(
        handles=list(handles),
        loc="upper left",
        bbox_to_anchor=(0, 0),
        bbox_transform=anchor,
        ncol=ncol or min(len(handles), 3),
        frameon=False,
        borderaxespad=0,
        handlelength=1.8,
        columnspacing=1.8,
        handletextpad=0.6,
    )


def family_handles(names: Sequence[str], mpl, line: bool = False) -> list:
    """One legend handle per family present among `names`, in the order of `FAMILIES` (the others last): a patch of
    the family hue, or a solid line for a figure of lines."""
    present = {family(name) for name in names}
    ordered = [(label, hue) for _, label, hue in FAMILIES if (label, hue) in present]
    ordered += [OTHERS] if OTHERS in present else []
    if line:
        return [mpl.lines.Line2D([], [], color=hue, linewidth=2, label=label) for label, hue in ordered]
    return [mpl.patches.Patch(facecolor=hue, edgecolor="none", label=label) for label, hue in ordered]


def label(ax, x: float, y: float, text: str, dx: float = 4, dy: float = 0, color: str = INK, **kw):
    """A value or point label beside (x, y) in data coordinates, `dx`, `dy` points away (left-aligned when dx >= 0),
    on a patch of paper so that a reference line behind it does not cross the text."""
    return ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.5,
        color=color,
        annotation_clip=False,
        bbox={"boxstyle": "square,pad=0.15", "facecolor": PAPER, "edgecolor": "none"},
        **kw,
    )


def number(value, digits: int = 2) -> str:
    """A number in English for a figure: `.` as decimal point, a true minus sign, "n/a" for nan."""
    if _nan(value):
        return "n/a"
    return f"{value:.{digits}f}".replace("-", "−")


def _tick(value: float, _position=None) -> str:
    """A tick label: at most two decimals, no trailing zero, a true minus sign."""
    text = f"{value:.2f}".rstrip("0").rstrip(".") if value else "0"
    return text.replace("-", "−")


def _axes(ax, x: bool = True, y: bool = True, grid: str | None = None) -> None:
    """Tick labels of the measured axes (`_tick`), and the thin light grid along `grid` ("x", "y" or "both")."""
    from matplotlib.ticker import FuncFormatter

    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(_tick))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(_tick))
    if grid:
        ax.grid(True, axis=grid)
    for axis, measured in (("x", x), ("y", y)):
        if measured:
            ax.tick_params(axis=axis, length=3, width=0.8)


def _heat(mpl):
    """The colour map of the heat maps: one hue (blue), light to dark."""
    return mpl.colors.LinearSegmentedColormap.from_list("s1d1-heat", [shade(BLUE, amount) for amount in HEAT])


def _ink(rgba) -> str:
    """Black or white text, whichever reads on a cell of that RGBA colour (relative luminance)."""
    r, g, b = rgba[:3]
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


# --- the data ---------------------------------------------------------------------------------------------------------


def _family_order(names: Sequence[str], value: dict[str, float] | None = None) -> list[str]:
    """The candidates grouped by family (s1d1, Laya, LiteLLM, the others); within a family by decreasing `value`
    (nan last), table order when no value is given or on a tie."""
    rank = {label: n for n, (_, label, _) in enumerate(FAMILIES)}

    def key(name):
        v = (value or {}).get(name)
        return rank.get(family(name)[0], len(FAMILIES)), (math.inf if _nan(v) else -v)

    return sorted(names, key=key)


def _variant_order(names: Sequence[str]) -> list[str]:
    """The variants of a family from the darkest to the lightest (fp16, full, compact, small, tiny), families first."""
    rank = {label: n for n, (_, label, _) in enumerate(FAMILIES)}
    return sorted(names, key=lambda name: (rank.get(family(name)[0], len(FAMILIES)), _step(name)))


def _positions(names: Sequence[str], gap: float = 0.6) -> list[float]:
    """The y of each bar from the top (0) down, with a gap between two families."""
    out, y = [], 0.0
    for n, name in enumerate(names):
        if n and family(name) != family(names[n - 1]):
            y += gap
        out.append(y)
        y += 1
    return out


def _nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _rows(table: pa.Table, **where) -> list[dict]:
    return [r for r in table.to_pylist() if all(r.get(k) == v for k, v in where.items())]


def _count(text: str | None) -> int | None:
    """The leading integer of a line of `inputs_summary` ("360 ids" -> 360), None when there is none."""
    found = re.match(r"\s*(\d+)", text or "")
    return int(found.group(1)) if found else None


def _units(computed: dict[str, pa.Table]) -> tuple[str, str, str]:
    """("360 ids", "90 base prompts", "1,000 bootstrap draws") from `inputs_summary`, "?" for a missing count."""
    summary = {r["item"]: r["value"] for r in computed["inputs_summary"].to_pylist()}
    ids, base, draws = (_count(summary.get(item)) for item in ("rows", "base keys", "bootstrap"))
    say = lambda n, unit: f"{n:,} {unit}" if n is not None else f"? {unit}"  # noqa: E731
    return say(ids, "ids"), say(base, "base prompts"), say(draws, "bootstrap draws")


def _errors(rows: list[dict], value: str, lo: str, hi: str) -> list[list[float]]:
    """The asymmetric error bars of matplotlib (distance below, distance above), 0 where the interval is missing."""
    below, above = [], []
    for r in rows:
        v, low, high = r[value], r[lo], r[hi]
        ok = not (_nan(v) or _nan(low) or _nan(high))
        below.append(v - low if ok else 0.0)
        above.append(high - v if ok else 0.0)
    return [below, above]


def _hbars(ax, names: Sequence[str], rows: dict[str, dict], value: str, lo: str, hi: str, digits: int = 3) -> None:
    """Thin horizontal bars, one per name from the top, in the candidate's colour, the 95 % CI as a thin black
    whisker and the value written past the end of the whisker; the names as tick labels beside the bars."""
    ys = _positions(names)
    for y, name in zip(ys, names, strict=True):
        r = rows[name]
        if _nan(r[value]):
            label(ax, 0, y, "n/a", color=MUTED)
            continue
        ax.barh(y, r[value], height=0.62, color=colour(name), linewidth=0, zorder=2)
        ax.errorbar(
            r[value], y, xerr=_errors([r], value, lo, hi), fmt="none", ecolor=INK, capsize=0, elinewidth=1, zorder=3
        )  # no cap: one marker shape in a figure
        end = max(v for v in (r[value], r[lo], r[hi], 0.0) if not _nan(v))
        label(ax, end, y, number(r[value], digits), zorder=4)
    ax.set_yticks(ys, list(names))
    ax.set_ylim(ys[-1] + 0.7, -0.7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _xlim_qwk(ax, values: Sequence[float]) -> None:
    """The QWK axis: from below the lowest bound (at least −0.2) to 1."""
    finite = [v for v in values if not _nan(v)]
    ax.set_xlim(min([-0.2, *finite]) - 0.02, 1.0)


# --- the figures: each returns (matplotlib figure or None, CSV columns, CSV rows, notes) ------------------------------


def _qwk_bars(computed: dict[str, pa.Table], sigma: float, mpl):
    n_ids, n_base, n_draws = _units(computed)
    notes: list[str] = []
    candidates = _rows(computed["agreement_all"], role="candidate")
    by_name = {r["name"]: r for r in candidates}
    table_order = _family_order([r["name"] for r in candidates])
    if not table_order:
        return None, [], [], ["aucun candidat : figure non tracée"]
    columns = ["kind", "name", "family", "value", "lo", "hi", "draws_used"]
    data = [
        {
            "kind": "candidate",
            "name": name,
            "family": family(name)[0],
            "value": by_name[name]["qwk"],
            "lo": by_name[name]["qwk_lo"],
            "hi": by_name[name]["qwk_hi"],
            "draws_used": by_name[name]["qwk_draws_used"],
        }
        for name in table_order
    ]
    order = _family_order(table_order, {name: by_name[name]["qwk"] for name in table_order})
    fig = mpl.figure.Figure(figsize=(8.0, 0.27 * len(order) + 1.9))
    ax = fig.add_subplot()
    _hbars(ax, order, by_name, "qwk", "qwk_lo", "qwk_hi")
    handles = family_handles(order, mpl)
    ceiling = computed["ceiling"].to_pylist()
    mean = [r for r in ceiling if r["judge"] == "mean"]
    if mean:
        ax.axvline(mean[0]["qwk"], color=JUDGE, linewidth=2, zorder=1.6)  # behind the bars
        handles.append(mpl.lines.Line2D([], [], color=JUDGE, linewidth=2, label="ceiling: mean of the main judges"))
    judges = [r for r in ceiling if r["judge"] != "mean"]
    for r in judges:
        ax.axvline(r["qwk"], color=JUDGE_SOFT, linewidth=1, zorder=1.5)
        data.append(
            {
                "kind": "ceiling (judge)",
                "name": r["judge"],
                "family": "judges",
                "value": r["qwk"],
                "lo": r["qwk_lo"],
                "hi": r["qwk_hi"],
                "draws_used": r["qwk_draws_used"],
            }
        )
    if judges:
        handles.append(
            mpl.lines.Line2D(
                [], [], color=JUDGE_SOFT, linewidth=1, label="ceiling: each main judge against the other two (8 votes)"
            )
        )
    if mean:
        data.append(
            {
                "kind": "ceiling (mean)",
                "name": "mean",
                "family": "judges",
                "value": mean[0]["qwk"],
                "lo": mean[0]["qwk_lo"],
                "hi": mean[0]["qwk_hi"],
                "draws_used": mean[0]["qwk_draws_used"],
            }
        )
    else:
        notes.append("plafond absent (pas de juge principal) : lignes du plafond non tracées")
    floor = computed["floor"].to_pylist()
    floors = []
    for name in (FLOOR, CONSTANT):
        found = [r for r in floor if r["name"] == name and r["draw"] is None]
        if found:
            ax.axvline(found[0]["qwk"], color=GREY, linestyle=(0, (4, 2)), linewidth=2, zorder=1.5)
            floors.append(name)
            data.append(
                {
                    "kind": "floor",
                    "name": name,
                    "family": family(name)[0],
                    "value": found[0]["qwk"],
                    "lo": found[0]["qwk_lo"],
                    "hi": found[0]["qwk_hi"],
                    "draws_used": found[0]["qwk_draws_used"],
                }
            )
        else:
            notes.append(f"`{name}` absent : sa ligne de plancher n'est pas tracée")
    if floors:
        handles.append(
            mpl.lines.Line2D(
                [], [], color=GREY, linestyle=(0, (4, 2)), linewidth=2, label="floor: " + " and ".join(floors)
            )
        )
    band = {r["kind"]: r["qwk"] for r in floor if r["name"] == RANDOM and r["kind"].startswith("random ")}
    if {"random min", "random max", "random mean"} <= set(band):
        ax.axvspan(band["random min"], band["random max"], color=GREY, alpha=0.22, linewidth=0, zorder=0)
        handles.append(
            mpl.patches.Patch(
                facecolor=GREY, alpha=0.22, edgecolor="none", label=f"floor: {RANDOM}, min-max of its draws"
            )
        )
        data.append(
            {
                "kind": "random band (min-max of the draws, value = mean)",
                "name": RANDOM,
                "family": family(RANDOM)[0],
                "value": band["random mean"],
                "lo": band["random min"],
                "hi": band["random max"],
                "draws_used": None,
            }
        )
    else:
        notes.append(f"tirages de `{RANDOM}` absents : bande du plancher non tracée")
    if REFERENCE in by_name and not _nan(by_name[REFERENCE]["qwk"]):
        ref = by_name[REFERENCE]["qwk"]
        ax.axvspan(ref - 2 * sigma, ref + 2 * sigma, color=BLUE, alpha=0.25, linewidth=0, zorder=0)
        handles.append(
            mpl.patches.Patch(
                facecolor=BLUE, alpha=0.25, edgecolor="none", label=f"± 2σ around {REFERENCE} (σ = {sigma:g})"
            )
        )
        data.append(
            {
                "kind": "± 2σ band",
                "name": REFERENCE,
                "family": family(REFERENCE)[0],
                "value": ref,
                "lo": ref - 2 * sigma,
                "hi": ref + 2 * sigma,
                "draws_used": None,
            }
        )
    else:
        notes.append(f"`{REFERENCE}` absent : bande ± 2 σ non tracée")
    _xlim_qwk(ax, [r["lo"] for r in data] + [r["value"] for r in data])
    ax.set_xlabel("QWK against truth_all")
    _axes(ax, y=False, grid="x")
    better(ax, "x", higher=True)
    title(
        ax,
        "Agreement with the median of the main judges",
        f"QWK against truth_all, {n_ids} ({n_base}); whiskers: 95% CI, {n_draws} of the base prompts",
    )
    fig.tight_layout()
    legend_below(fig, handles, ncol=2)
    return fig, columns, data, notes


def _qwk_by_language(computed: dict[str, pa.Table], sigma: float, mpl):
    _, n_base, n_draws = _units(computed)
    rows = _rows(computed["by_language"], role="candidate")
    table_order = _family_order(list(dict.fromkeys(r["name"] for r in rows)))
    if not table_order:
        return None, [], [], ["aucun candidat : figure non tracée"]
    langs = sorted({r["lang"] for r in rows})
    cell = {(r["name"], r["lang"]): r for r in rows}
    per_lang = max((r["n_ids_all"] for r in rows), default=0)
    columns = ["name", "family", "lang", "n_ids", "qwk", "lo", "hi", "draws_used"]
    data = [
        {
            "name": name,
            "family": family(name)[0],
            "lang": lang,
            "n_ids": cell[name, lang]["n_ids_all"],
            "qwk": cell[name, lang]["qwk_all"],
            "lo": cell[name, lang]["qwk_all_lo"],
            "hi": cell[name, lang]["qwk_all_hi"],
            "draws_used": cell[name, lang]["qwk_all_draws_used"],
        }
        for lang in langs
        for name in table_order
        if (name, lang) in cell
    ]
    # one row order in every panel (the candidates' overall QWK), so that a candidate is read across the languages
    overall = {r["name"]: r["qwk"] for r in _rows(computed["agreement_all"], role="candidate")}
    order = _family_order(table_order, overall)
    ncols = 2 if len(langs) > 1 else 1
    nrows = math.ceil(len(langs) / ncols)
    fig = mpl.figure.Figure(figsize=(8.6 if ncols == 2 else 5.0, nrows * (0.24 * len(order) + 1.0) + 0.8))
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True).ravel()
    for ax, lang in zip(axes, langs, strict=False):
        here = [name for name in order if (name, lang) in cell]
        _hbars(ax, here, {name: cell[name, lang] for name in here}, "qwk_all", "qwk_all_lo", "qwk_all_hi", 2)
        ax.set_title(f"{lang} ({max(cell[name, lang]['n_ids_all'] for name in here)} ids)", fontweight="bold")
        _axes(ax, y=False, grid="x")
    for ax in axes[len(langs) :]:
        ax.set_visible(False)
    _xlim_qwk(axes[0], [r[k] for r in data for k in ("qwk", "lo")])
    for ax in axes[(nrows - 1) * ncols : len(langs)]:
        ax.set_xlabel("QWK against truth_all")
        better(ax, "x", higher=True)
    title(
        axes[0],
        "Agreement with the median of the main judges, by language",
        f"QWK against truth_all, {per_lang} ids per language; whiskers: 95% CI, {n_draws} of the {n_base}",
    )
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    legend_below(fig, family_handles(order, mpl), ncol=4)
    return fig, columns, data, []


def _confusion(computed: dict[str, pa.Table], sigma: float, mpl):
    table = computed["confusion"]
    present = [name for name in CONFUSION if _rows(table, candidate=name)]
    notes = [f"`{name}` absent : pas de panneau de confusion" for name in CONFUSION if name not in present]
    if not present:
        return None, [], [], notes
    columns = ["candidate", "truth_tier", "pred_tier", "n_ids", "row_share"]
    data = []
    cmap = _heat(mpl)
    fig = mpl.figure.Figure(figsize=(2.55 * len(present) + 1.0, 3.5))
    axes = fig.subplots(1, len(present), squeeze=False)[0]
    first_totals = None
    for k, (ax, name) in enumerate(zip(axes, present, strict=True)):
        cells = {(r["truth_tier"], r["pred_tier"]): r for r in _rows(table, candidate=name)}
        data += [cells[t, p] | {"candidate": name} for t in LEVELS for p in LEVELS]
        share = [[cells[t, p]["row_share"] for p in LEVELS] for t in LEVELS]
        totals = {t: sum(cells[t, p]["n_ids"] for p in LEVELS) for t in LEVELS}
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i, t in enumerate(LEVELS):
            for j in range(len(LEVELS)):
                value = share[i][j]
                text = "–" if not totals[t] else number(value, 2)
                ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=_ink(cmap(value)))
        ax.set_xticks(range(len(LEVELS)), LEVELS, rotation=45, ha="right", rotation_mode="anchor")
        if k == 0 or totals != first_totals:
            ax.set_yticks(range(len(LEVELS)), [f"{t} ({totals[t]})" for t in LEVELS])
        else:
            ax.set_yticks(range(len(LEVELS)), [])
        first_totals = first_totals or totals
        ax.set_xlabel("predicted tier")
        ax.set_title(f"{name} ({sum(totals.values())} ids)", fontsize=8.5, fontweight="bold")
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("truth_all tier (ids)")
    title(
        axes[0],
        "Confusion matrices against truth_all",
        "each row: the ids of a truth_all tier by predicted tier, as shares of the row (rows sum to 1)",
    )
    fig.tight_layout(w_pad=1.2)
    return fig, columns, data, notes


def _reliability(computed: dict[str, pa.Table], sigma: float, mpl):
    bins = [r for r in computed["calibration_bins"].to_pylist() if not _nan(r["mean_score"])]
    names = _variant_order(list(dict.fromkeys(r["candidate"] for r in bins)))
    if not names:
        return None, [], [], ["aucun candidat s1d1 à score v7 : figure non tracée"]
    n_ids, _, _ = _units(computed)
    columns = ["candidate", "bin_low", "bin_high", "n_ids", "mean_score", "mean_anchor"]
    fig = mpl.figure.Figure(figsize=(2.15 * len(names) + 0.9, 3.4))
    axes = fig.subplots(1, len(names), squeeze=False, sharex=True, sharey=True)[0]
    for ax, name in zip(axes, names, strict=True):
        ax.plot([0, 1], [0, 1], color=GREY, linestyle=(0, (4, 2)), linewidth=1.2, zorder=1)
        found = [r for r in bins if r["candidate"] == name]
        ax.plot(
            [r["mean_score"] for r in found],
            [r["mean_anchor"] for r in found],
            color=colour(name),
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=3,
        )
        ax.set_title(name, fontsize=8.5, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("mean score of the bin")
        _axes(ax, grid="both")
    axes[0].set_ylabel("mean truth_all anchor of the bin")
    title(
        axes[0],
        "Reliability of the s1d1 score against the truth_all tier anchor",
        f"v7 scale, 0-1; 10 bins of score, {n_ids} per candidate; a point on the diagonal is calibrated",
    )
    fig.tight_layout(w_pad=0.8)
    handles = family_handles(names, mpl, line=True)
    handles.append(
        mpl.lines.Line2D(
            [], [], color=GREY, linestyle=(0, (4, 2)), linewidth=1.2, label="perfect calibration (score = anchor)"
        )
    )
    legend_below(fig, handles)
    return fig, columns, bins, []


def _judge_spread(computed: dict[str, pa.Table], sigma: float, mpl):
    rows = computed["judge_spread_by_tier"].to_pylist()
    if not rows:
        return None, [], [], ["pas d'écart entre juges : figure non tracée"]
    n_ids, _, _ = _units(computed)
    columns = ["tier", "spread", "n_ids", "n_ids_tier", "mean_spread_tier", "n_ids_null_spread_tier"]
    spreads = sorted({r["spread"] for r in rows})
    counts = [[next(r["n_ids"] for r in rows if r["tier"] == t and r["spread"] == s) for s in spreads] for t in LEVELS]
    tier_rows = {t: next(r for r in rows if r["tier"] == t) for t in LEVELS}
    fig = mpl.figure.Figure(figsize=(6.6, 3.4))
    ax = fig.add_subplot()
    top = max((c for line in counts for c in line), default=0) or 1
    cmap = _heat(mpl)
    ax.imshow(counts, cmap=cmap, vmin=0, vmax=top, aspect="auto")
    for i, line in enumerate(counts):
        for j, value in enumerate(line):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=_ink(cmap(value / top)))
    labels = [
        f"{t} ({tier_rows[t]['n_ids_tier']} ids, mean {number(tier_rows[t]['mean_spread_tier'], 2)})" for t in LEVELS
    ]
    ax.set_yticks(range(len(LEVELS)), labels)
    ax.set_xticks(range(len(spreads)), [str(s) for s in spreads])
    ax.set_xlabel("spread between the main judges on the id (max - min of their tiers, in steps)")
    ax.set_ylabel("truth_all tier")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    title(ax, "Spread between the main judges, by truth_all tier", f"number of ids in each cell; {n_ids} in all")
    fig.tight_layout()
    return fig, columns, rows, []


def _language_disagreements(computed: dict[str, pa.Table], sigma: float, mpl):
    rows = computed["truth_lang_vs_all"].to_pylist()
    groups = [("lang", "by language"), ("source", "by source")]
    present = [(g, head) for g, head in groups if any(r["group"] == g for r in rows)]
    if not present:
        return None, [], [], ["pas de désaccord de langue mesuré : figure non tracée"]
    columns = ["group", "value", "n_ids", "n_ids_disagreement", "share_of_ids", "n_keys", "n_keys_disagreement"]
    data = [r | {"share_of_ids": r["n_ids_disagreement"] / r["n_ids"] if r["n_ids"] else math.nan} for r in rows]
    overall = next((r for r in data if r["group"] == "all"), None)
    longest = max(sum(r["group"] == g for r in data) for g, _ in present)
    fig = mpl.figure.Figure(figsize=(4.2 * len(present), 0.3 * longest + 1.9))
    axes = fig.subplots(1, len(present), squeeze=False, sharex=True)[0]
    top = max((r["share_of_ids"] for r in data if not _nan(r["share_of_ids"])), default=0)
    for ax, (group, head) in zip(axes, present, strict=True):
        found = [r for r in data if r["group"] == group]
        ys = list(range(len(found)))
        for y, r in zip(ys, found, strict=True):
            if _nan(r["share_of_ids"]):
                label(ax, 0, y, "n/a", color=MUTED)
                continue
            ax.barh(y, r["share_of_ids"], height=0.62, color=JUDGE_SOFT, linewidth=0, zorder=2)
            text = f"{number(r['share_of_ids'], 3)} ({r['n_ids_disagreement']}/{r['n_ids']})"
            label(ax, r["share_of_ids"], y, text)
        ax.set_yticks(ys, [r["value"] for r in found])
        ax.set_ylim(len(found) - 0.4, -0.6)
        ax.set_title(head, fontsize=8.5, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("share of ids where truth_lang ≠ truth_all")
        _axes(ax, y=False, grid="x")
        better(ax, "x", higher=False)
    axes[0].set_xlim(0, max(0.05, top * 1.45))
    axes[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    sub = "share of the ids of each group (label: share, ids in disagreement / ids)"
    if overall:
        sub += (
            f"; in all {overall['n_ids_disagreement']}/{overall['n_ids']} ids, "
            f"{overall['n_keys_disagreement']}/{overall['n_keys']} base prompts"
        )
    title(axes[0], "Disagreements between the truth by language and truth_all", sub)
    fig.tight_layout(w_pad=2.5)
    return fig, columns, data, []


def _needs_tools_pr(computed: dict[str, pa.Table], sigma: float, mpl):
    curve = computed["needs_tools_pr"].to_pylist()
    names = _variant_order(list(dict.fromkeys(r["name"] for r in curve)))
    if not names:
        return None, [], [], ["aucun candidat avec une probabilité de needs_tools : figure non tracée"]
    columns = ["name", "threshold", "precision", "recall"]
    needs = {r["name"]: r for r in computed["needs_tools"].to_pylist()}
    fig = mpl.figure.Figure(figsize=(2.15 * len(names) + 0.9, 3.5))
    axes = fig.subplots(1, len(names), squeeze=False, sharex=True, sharey=True)[0]
    first = needs.get(names[0])
    prevalence = first["n_ids_positive"] / first["n_ids"] if first and first["n_ids"] else None
    by_name = {name: [r for r in curve if r["name"] == name] for name in names}  # by decreasing threshold
    for ax, name in zip(axes, names, strict=True):
        for other in names:  # the other candidates, light, behind: the panel's curve read against them
            if other != name:
                found = by_name[other]
                ax.step(
                    [r["recall"] for r in found],
                    [r["precision"] for r in found],
                    where="post",
                    color=GRID,
                    linewidth=1,
                    zorder=1,
                )
        found = by_name[name]
        ax.step(
            [r["recall"] for r in found],
            [r["precision"] for r in found],
            where="post",
            color=colour(name),
            linewidth=2,
            zorder=3,
        )
        if prevalence is not None:
            ax.axhline(prevalence, color=JUDGE_SOFT, linestyle=(0, (1, 1.5)), linewidth=1.2, zorder=2)
        ax.set_title(name, fontsize=8.5, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("recall")
        better(ax, "x", higher=True)
        _axes(ax, grid="both")
    axes[0].set_ylabel("precision")
    better(axes[0], "y", higher=True)
    sub = "precision and recall as the probability threshold moves"
    handles = family_handles(names, mpl, line=True)
    if prevalence is not None:
        sub += f"; {first['n_ids_positive']} positive ids out of {first['n_ids']}: few positives"
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=JUDGE_SOFT,
                linestyle=(0, (1, 1.5)),
                linewidth=1.2,
                label=f"share of positives ({first['n_ids_positive']}/{first['n_ids']} ids)",
            )
        )
    if len(names) > 1:
        handles.append(mpl.lines.Line2D([], [], color=GRID, linewidth=1, label="the other candidates, for comparison"))
    title(axes[0], "needs_tools: precision against recall", sub)
    fig.tight_layout(w_pad=0.8)
    legend_below(fig, handles)
    return fig, columns, curve, []


_DRAW = {
    "qwk-bars": _qwk_bars,
    "qwk-by-language": _qwk_by_language,
    "confusion": _confusion,
    "reliability-s1d1": _reliability,
    "judge-spread-by-tier": _judge_spread,
    "language-disagreements": _language_disagreements,
    "needs-tools-pr": _needs_tools_pr,
}


def figures(computed: dict[str, pa.Table], out_dir: Path | str, sigma: float = 0.005) -> list[Figure]:
    """Draw every figure of `NAMES` from the tables of `report.tables` into `out_dir/figures/<name>.svg`, with the
    plotted values in `<name>.csv` beside it; return one `Figure` per name, in that order (svg and csv None for a
    figure left with nothing to draw, its stale files removed)."""
    import matplotlib
    import matplotlib.colors
    import matplotlib.figure
    import matplotlib.lines
    import matplotlib.patches
    import matplotlib.ticker

    folder = Path(out_dir) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    drawn = []
    with style():
        for name in NAMES:
            svg, csv = folder / f"{name}.svg", folder / f"{name}.csv"
            fig, columns, rows, notes = _DRAW[name](computed, sigma, matplotlib)
            if fig is None:
                svg.unlink(missing_ok=True)
                csv.unlink(missing_ok=True)
                drawn.append(Figure(name, None, None, tuple(notes)))
                continue
            fig.savefig(svg, format="svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.15)
            write_csv(csv, [HEADERS[c] for c in columns], [{HEADERS[c]: row.get(c) for c in columns} for row in rows])
            drawn.append(Figure(name, svg, csv, tuple(notes)))
    return drawn
