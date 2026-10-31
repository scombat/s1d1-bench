"""`bench report`, the figures: one SVG and one CSV of the same name per figure, in `<out>/figures/`, drawn from the
tables of `report.tables` (nothing is recomputed here; each CSV holds the values the figure plots).

matplotlib (the `report` dependency group) is imported inside `figures` only, so that the other commands, and
`bench --help`, never need it; the figures are made with `matplotlib.figure.Figure` (no pyplot, no GUI backend). The
SVG is byte-identical from one run to the next: a fixed `svg.hashsalt`, no date in the metadata, text kept as text
(`svg.fonttype: none`).

Colours: the Okabe-Ito palette (colour-blind safe), one hue per candidate family and the same in every figure: s1d1
blue, Laya orange, LiteLLM bluish green; the witnesses and any other candidate grey, the judges (ceiling, spread,
language disagreements) black or grey: three hues and the neutrals. Within a family the candidates differ by hatch,
marker or line style, never by a new hue.

A candidate or role a figure needs that is absent from the tables loses its panel (or its line), and a French line
saying so is returned for the index; a figure left with nothing to draw is not written (and a stale file of that name
is removed). Never an error.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa

from s1d1_bench.report import CONSTANT, FLOOR, LITELLM, RANDOM, REFERENCE, french, write_csv
from s1d1_bench.rubric import LEVELS

BLUE, ORANGE, GREEN, GREY, LIGHT, BLACK = "#0072B2", "#E69F00", "#009E73", "#999999", "#DDDDDD", "#000000"
FAMILIES = (("s1d1-", "s1d1", BLUE), ("laya-", "Laya", ORANGE), ("litellm-", "LiteLLM", GREEN))
OTHERS = ("témoins et autres", GREY)
CONFUSION = (REFERENCE, "laya-multilingual-tier", LITELLM, FLOOR)  # the four panels of `confusion`
HATCHES = ("", "//", "\\\\", "xx", "--", "++")  # per language, or per candidate within a family
MARKERS = ("o", "s", "^", "D", "v", "P")
LINES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (8, 2)))  # solid, dashed, dotted, dash-dot, dash-dot-dot
SHADES = (0.0, -0.35, 0.4, -0.6, 0.6, -0.2)  # within a family: the same hue, darker (< 0) or lighter (> 0)
RC = {
    "svg.hashsalt": "s1d1-bench",  # the ids of clip paths and glyphs, otherwise random
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
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


class Figure(NamedTuple):
    """A figure of the report: its SVG and CSV (None when it was not drawn) and the French lines of what it left out."""

    name: str
    svg: Path | None
    csv: Path | None
    notes: tuple[str, ...]


def family(name: str) -> tuple[str, str]:
    """(family label, colour) of a candidate, by the prefix of its name."""
    for prefix, label, colour in FAMILIES:
        if name.startswith(prefix):
            return label, colour
    return OTHERS


def _family_order(names: Sequence[str]) -> list[str]:
    """The candidates grouped by family (s1d1, Laya, LiteLLM, the others), table order within a family."""
    rank = {label: n for n, (_, label, _) in enumerate(FAMILIES)}
    return sorted(names, key=lambda name: rank.get(family(name)[0], len(FAMILIES)))


def _shade(colour: str, amount: float) -> str:
    """`colour` (#RRGGBB) mixed with black (amount < 0) or white (amount > 0): the same hue, another lightness."""
    rgb = [int(colour[i : i + 2], 16) for i in (1, 3, 5)]
    target = 255 if amount > 0 else 0
    mixed = [round(c + (target - c) * abs(amount)) for c in rgb]
    return "#" + "".join(f"{c:02X}" for c in mixed)


def _series(name: str, n: int) -> dict:
    """The line style of the n-th series of a figure with several lines of one family: its hue at another lightness,
    its own dash pattern and marker."""
    return {
        "color": _shade(family(name)[1], SHADES[n % len(SHADES)]),
        "linestyle": LINES[n % len(LINES)],
        "marker": MARKERS[n % len(MARKERS)],
    }


def _rank_in_family(names: Sequence[str]) -> dict[str, int]:
    """The position of each candidate within its family (its hatch, marker and line style)."""
    seen: dict[str, int] = {}
    out = {}
    for name in names:
        label = family(name)[0]
        out[name] = seen.get(label, 0)
        seen[label] = out[name] + 1
    return out


def _nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _rows(table: pa.Table, **where) -> list[dict]:
    return [r for r in table.to_pylist() if all(r.get(k) == v for k, v in where.items())]


def _units(computed: dict[str, pa.Table]) -> tuple[str, str]:
    """("360 ids", "90 prompts de base") as `inputs_summary` says them."""
    summary = {r["item"]: r["value"] for r in computed["inputs_summary"].to_pylist()}
    return summary.get("rows", "? ids"), summary.get("base keys", "? prompts de base")


def _tick(value: float, _position=None) -> str:
    """A tick label in French: decimal comma, true minus sign."""
    text = f"{value:.2f}".rstrip("0").rstrip(".") if value else "0"
    return text.replace(".", ",").replace("-", "−")


def _french_axes(ax, x: bool = True, y: bool = True) -> None:
    from matplotlib.ticker import FuncFormatter

    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(_tick))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(_tick))


def _ink(colour) -> str:
    """Black or white text, whichever reads on a cell of that RGBA colour (relative luminance)."""
    r, g, b = colour[:3]
    return BLACK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "white"


def _errors(rows: list[dict], value: str, lo: str, hi: str) -> list[list[float]]:
    """The asymmetric error bars of matplotlib (distance below, distance above), 0 where the interval is missing."""
    below, above = [], []
    for r in rows:
        v, low, high = r[value], r[lo], r[hi]
        ok = not (_nan(v) or _nan(low) or _nan(high))
        below.append(v - low if ok else 0.0)
        above.append(high - v if ok else 0.0)
    return [below, above]


# --- the figures: each returns (matplotlib figure or None, CSV columns, CSV rows, notes) --------------------------


def _qwk_bars(computed: dict[str, pa.Table], sigma: float, mpl):
    n_ids, n_base = _units(computed)
    notes: list[str] = []
    candidates = _rows(computed["agreement_all"], role="candidate")
    by_name = {r["name"]: r for r in candidates}
    order = _family_order([r["name"] for r in candidates])
    if not order:
        return None, [], [], ["aucun candidat : figure non tracée"]
    columns = ["kind", "name", "family", "value", "lo", "hi", "draws_used"]
    data = [
        {
            "kind": "candidat",
            "name": name,
            "family": family(name)[0],
            "value": by_name[name]["qwk"],
            "lo": by_name[name]["qwk_lo"],
            "hi": by_name[name]["qwk_hi"],
            "draws_used": by_name[name]["qwk_draws_used"],
        }
        for name in order
    ]
    fig = mpl.figure.Figure(figsize=(8.5, 0.4 * len(order) + 1.6))
    ax = fig.add_subplot()
    ranks = _rank_in_family(order)
    y = list(range(len(order)))[::-1]
    for pos, row in zip(y, data, strict=True):
        if _nan(row["value"]):
            continue
        _, colour = family(row["name"])
        ax.barh(pos, row["value"], color=colour, hatch=HATCHES[ranks[row["name"]] % len(HATCHES)], edgecolor="white")
        error = _errors([row], "value", "lo", "hi")
        ax.errorbar(row["value"], pos, xerr=error, fmt="none", ecolor=BLACK, capsize=3, linewidth=1)
        # the point estimate as a marker: a bar of length 0 (constant-low's QWK of 0) stays visible
        ax.plot(row["value"], pos, marker="D", markersize=5, color=colour, markeredgecolor=BLACK, zorder=3)
    ax.set_yticks(y, [f"{name} ({french(by_name[name]['qwk'], 3)})" for name in order])
    ceiling = computed["ceiling"].to_pylist()
    judges = [r for r in ceiling if r["judge"] != "mean"]
    for n, r in enumerate(judges):
        label = "plafond : chaque juge principal contre les deux autres (8 votes)" if n == 0 else None
        ax.axvline(r["qwk"], color=BLACK, linestyle=":", linewidth=1, label=label)
        data.append(
            {
                "kind": "plafond (juge)",
                "name": r["judge"],
                "family": "juges",
                "value": r["qwk"],
                "lo": r["qwk_lo"],
                "hi": r["qwk_hi"],
                "draws_used": r["qwk_draws_used"],
            }
        )
    mean = [r for r in ceiling if r["judge"] == "mean"]
    if mean:
        ax.axvline(mean[0]["qwk"], color=BLACK, linewidth=2, label="plafond : moyenne des trois")
        data.append(
            {
                "kind": "plafond (moyenne)",
                "name": "mean",
                "family": "juges",
                "value": mean[0]["qwk"],
                "lo": mean[0]["qwk_lo"],
                "hi": mean[0]["qwk_hi"],
                "draws_used": mean[0]["qwk_draws_used"],
            }
        )
    else:
        notes.append("plafond absent (pas de juge principal) : lignes du plafond non tracées")
    floor = computed["floor"].to_pylist()
    for name, style, shade in ((FLOOR, "--", GREY), (CONSTANT, "-.", "#555555")):
        found = [r for r in floor if r["name"] == name and r["draw"] is None]
        if found:
            ax.axvline(found[0]["qwk"], color=shade, linestyle=style, linewidth=1.5, label=f"plancher : {name}",
                       zorder=2.5)  # fmt: skip
            data.append(
                {
                    "kind": "plancher",
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
    band = {r["kind"]: r["qwk"] for r in floor if r["name"] == RANDOM and r["kind"].startswith("random ")}
    if {"random min", "random max", "random mean"} <= set(band):
        ax.axvspan(
            band["random min"],
            band["random max"],
            color=LIGHT,
            zorder=0,
            label=f"plancher : {RANDOM}, min–max des tirages",
        )
        data.append(
            {
                "kind": "bande random (min–max des tirages, valeur = moyenne)",
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
        ax.axvspan(
            ref - 2 * sigma,
            ref + 2 * sigma,
            color=BLUE,
            alpha=0.3,
            zorder=0,
            label=f"± 2 σ autour de {REFERENCE} (σ = {french(sigma, 3)}, étroite par nature)",
        )
        data.append(
            {
                "kind": "bande ± 2 σ",
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
    lows = [r["lo"] for r in data if not _nan(r["lo"])] + [r["value"] for r in data if not _nan(r["value"])]
    ax.set_xlim(min([-0.2, *lows]) - 0.02, 1.0)
    ax.set_xlabel(f"QWK contre truth_all ({n_ids} ; IC 95 % : 1 000 tirages des {n_base})")
    ax.set_title("Accord de chaque candidat avec la médiane des juges principaux")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, fontsize=7, frameon=False)
    _french_axes(ax, y=False)
    fig.tight_layout()
    return fig, columns, data, notes


def _qwk_by_language(computed: dict[str, pa.Table], sigma: float, mpl):
    _, n_base = _units(computed)
    rows = _rows(computed["by_language"], role="candidate")
    order = _family_order(list(dict.fromkeys(r["name"] for r in rows)))
    if not order:
        return None, [], [], ["aucun candidat : figure non tracée"]
    langs = sorted({r["lang"] for r in rows})
    cell = {(r["name"], r["lang"]): r for r in rows}
    per_lang = max((r["n_ids_all"] for r in rows), default=0)
    columns = ["name", "family", "lang", "n_ids", "qwk", "lo", "hi", "draws_used"]
    data = []
    fig = mpl.figure.Figure(figsize=(max(7.0, 0.8 * len(order) + 2), 4.8))
    ax = fig.add_subplot()
    width = 0.8 / len(langs)
    for k, lang in enumerate(langs):
        found = [cell[name, lang] for name in order if (name, lang) in cell]
        xs = [order.index(r["name"]) - 0.4 + width * (k + 0.5) for r in found]
        for x, r in zip(xs, found, strict=True):
            data.append(
                {
                    "name": r["name"],
                    "family": family(r["name"])[0],
                    "lang": lang,
                    "n_ids": r["n_ids_all"],
                    "qwk": r["qwk_all"],
                    "lo": r["qwk_all_lo"],
                    "hi": r["qwk_all_hi"],
                    "draws_used": r["qwk_all_draws_used"],
                }
            )
            if _nan(r["qwk_all"]):
                continue
            ax.bar(
                x,
                r["qwk_all"],
                width,
                color=family(r["name"])[1],
                hatch=HATCHES[k % len(HATCHES)],
                edgecolor="white",
                linewidth=0,
            )
            ax.errorbar(
                x,
                r["qwk_all"],
                yerr=_errors([r], "qwk_all", "qwk_all_lo", "qwk_all_hi"),
                fmt="none",
                ecolor=BLACK,
                capsize=1.5,
                linewidth=0.8,
            )
    ax.axhline(0, color=BLACK, linewidth=0.6)
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    handles = [
        mpl.patches.Patch(facecolor=GREY, edgecolor="white", hatch=HATCHES[k % len(HATCHES)], label=lang)
        for k, lang in enumerate(langs)
    ]
    ax.legend(handles=handles, title="langue (hachure)", fontsize=7, title_fontsize=7, frameon=False, ncol=len(langs))
    ax.set_ylabel(f"QWK contre truth_all ({per_lang} ids par langue)")
    ax.set_title(f"QWK par langue, IC 95 % : 1 000 tirages des {n_base}")
    _french_axes(ax, x=False)
    fig.tight_layout()
    return fig, columns, data, []


def _confusion(computed: dict[str, pa.Table], sigma: float, mpl):
    table = computed["confusion"]
    present = [name for name in CONFUSION if _rows(table, candidate=name)]
    notes = [f"`{name}` absent : pas de panneau de confusion" for name in CONFUSION if name not in present]
    if not present:
        return None, [], [], notes
    columns = ["candidate", "truth_tier", "pred_tier", "n_ids", "row_share"]
    data = []
    fig = mpl.figure.Figure(figsize=(3.3 * len(present), 3.9))
    axes = fig.subplots(1, len(present), squeeze=False)[0]
    for ax, name in zip(axes, present, strict=True):
        cells = {(r["truth_tier"], r["pred_tier"]): r for r in _rows(table, candidate=name)}
        data += [cells[t, p] | {"candidate": name} for t in LEVELS for p in LEVELS]
        share = [[cells[t, p]["row_share"] for p in LEVELS] for t in LEVELS]
        totals = {t: sum(cells[t, p]["n_ids"] for p in LEVELS) for t in LEVELS}
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name, ["#FFFFFF", family(name)[1]])
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i, t in enumerate(LEVELS):
            for j in range(len(LEVELS)):
                value = share[i][j]
                text = "–" if not totals[t] else french(value, 2)
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=_ink(cmap(value)))
        ax.set_xticks(range(len(LEVELS)), LEVELS, rotation=45, ha="right")
        ax.set_yticks(range(len(LEVELS)), [f"{t} ({totals[t]})" for t in LEVELS])
        ax.set_xlabel("palier prédit")
        ax.set_title(f"{name} ({sum(totals.values())} ids)", fontsize=8)
        ax.spines[:].set_visible(True)
    axes[0].set_ylabel("palier de truth_all (ids)")
    fig.suptitle("Matrices de confusion, normalisées par ligne : part des ids du palier de la vérité (0–1)")
    fig.tight_layout()
    return fig, columns, data, notes


def _reliability(computed: dict[str, pa.Table], sigma: float, mpl):
    bins = [r for r in computed["calibration_bins"].to_pylist() if not _nan(r["mean_score"])]
    names = list(dict.fromkeys(r["candidate"] for r in bins))
    if not names:
        return None, [], [], ["aucun candidat s1d1 à score v7 : figure non tracée"]
    columns = ["candidate", "bin_low", "bin_high", "n_ids", "mean_score", "mean_anchor"]
    fig = mpl.figure.Figure(figsize=(5.6, 5.2))
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.8, label="score = ancre (calibration parfaite)")
    for n, name in enumerate(names):
        found = [r for r in bins if r["candidate"] == name]
        ax.plot(
            [r["mean_score"] for r in found],
            [r["mean_anchor"] for r in found],
            **_series(name, n),
            linewidth=1.2,
            markersize=4,
            label=name,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("score, moyenne de la case (échelle v7, 0–1 ; 10 cases de score)")
    ax.set_ylabel("ancre du palier de truth_all, moyenne de la case (0–1)")
    ax.set_title("Fiabilité du score de s1d1 contre l'ancre du palier de la vérité")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _french_axes(ax)
    fig.tight_layout()
    return fig, columns, bins, []


def _judge_spread(computed: dict[str, pa.Table], sigma: float, mpl):
    rows = computed["judge_spread_by_tier"].to_pylist()
    if not rows:
        return None, [], [], ["pas d'écart entre juges : figure non tracée"]
    columns = ["tier", "spread", "n_ids", "n_ids_tier", "mean_spread_tier", "n_ids_null_spread_tier"]
    spreads = sorted({r["spread"] for r in rows})
    counts = [[next(r["n_ids"] for r in rows if r["tier"] == t and r["spread"] == s) for s in spreads] for t in LEVELS]
    tier_rows = {t: next(r for r in rows if r["tier"] == t) for t in LEVELS}
    fig = mpl.figure.Figure(figsize=(6.4, 3.8))
    ax = fig.add_subplot()
    top = max((c for line in counts for c in line), default=0) or 1
    cmap = mpl.colors.LinearSegmentedColormap.from_list("judges", ["#FFFFFF", "#444444"])
    ax.imshow(counts, cmap=cmap, vmin=0, vmax=top, aspect="auto")
    for i, line in enumerate(counts):
        for j, value in enumerate(line):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=_ink(cmap(value / top)))
    labels = [
        f"{t} ({tier_rows[t]['n_ids_tier']} ids, moyenne {french(tier_rows[t]['mean_spread_tier'], 2)})" for t in LEVELS
    ]
    ax.set_yticks(range(len(LEVELS)), labels)
    ax.set_xticks(range(len(spreads)), [str(s) for s in spreads])
    ax.set_xlabel("écart entre les juges principaux sur l'id (max − min des paliers, en crans)")
    ax.set_ylabel("palier de truth_all")
    ax.set_title("Écart entre juges par palier de la vérité (nombre d'ids)")
    ax.spines[:].set_visible(True)
    fig.tight_layout()
    return fig, columns, rows, []


def _language_disagreements(computed: dict[str, pa.Table], sigma: float, mpl):
    rows = computed["truth_lang_vs_all"].to_pylist()
    groups = [("lang", "par langue"), ("source", "par source")]
    present = [(g, title) for g, title in groups if any(r["group"] == g for r in rows)]
    if not present:
        return None, [], [], ["pas de désaccord de langue mesuré : figure non tracée"]
    columns = ["group", "value", "n_ids", "n_ids_disagreement", "share_of_ids", "n_keys", "n_keys_disagreement"]
    data = [r | {"share_of_ids": r["n_ids_disagreement"] / r["n_ids"] if r["n_ids"] else math.nan} for r in rows]
    overall = next((r for r in data if r["group"] == "all"), None)
    fig = mpl.figure.Figure(figsize=(8.0, 3.8))
    axes = fig.subplots(1, len(present), squeeze=False, sharey=True)[0]
    for ax, (group, title) in zip(axes, present, strict=True):
        found = [r for r in data if r["group"] == group]
        xs = range(len(found))
        ax.bar(xs, [r["share_of_ids"] for r in found], color=GREY, edgecolor=BLACK, linewidth=0.6)
        for x, r in zip(xs, found, strict=True):
            ax.text(
                x, r["share_of_ids"], f"{r['n_ids_disagreement']}/{r['n_ids']}", ha="center", va="bottom", fontsize=7
            )
        ax.set_xticks(list(xs), [r["value"] for r in found], rotation=20 if group == "source" else 0)
        ax.set_title(title, fontsize=8)
        _french_axes(ax, x=False)
    axes[0].set_ylabel("part des ids où truth_lang ≠ truth_all (0–1)")
    head = "Désaccords de langue (étiquette : ids en désaccord / ids)"
    if overall:
        head += (
            f" ; en tout {overall['n_ids_disagreement']}/{overall['n_ids']} ids, "
            f"{overall['n_keys_disagreement']}/{overall['n_keys']} prompts de base"
        )
    fig.suptitle(head, fontsize=8)
    fig.tight_layout()
    return fig, columns, data, []


def _needs_tools_pr(computed: dict[str, pa.Table], sigma: float, mpl):
    curve = computed["needs_tools_pr"].to_pylist()
    names = list(dict.fromkeys(r["name"] for r in curve))
    if not names:
        return None, [], [], ["aucun candidat avec une probabilité de needs_tools : figure non tracée"]
    columns = ["name", "threshold", "precision", "recall"]
    fig = mpl.figure.Figure(figsize=(5.6, 4.8))
    ax = fig.add_subplot()
    needs = {r["name"]: r for r in computed["needs_tools"].to_pylist()}
    for n, name in enumerate(names):
        found = [r for r in curve if r["name"] == name]  # by decreasing threshold, as metrics.pr_curve gives them
        ax.step(
            [r["recall"] for r in found],
            [r["precision"] for r in found],
            where="post",
            **_series(name, n),
            markevery=max(1, len(found) // 10),  # a count, not a fraction: a curve may be all nan
            markersize=4,
            linewidth=1.2,
            label=name,
        )
    title = "needs_tools : précision et rappel selon le seuil de probabilité"
    first = needs.get(names[0])
    if first and first["n_ids"]:
        prevalence = first["n_ids_positive"] / first["n_ids"]
        ax.axhline(
            prevalence,
            color=GREY,
            linestyle=":",
            linewidth=1,
            label=f"part des positifs ({first['n_ids_positive']}/{first['n_ids']} ids)",
        )
        title += f"\n{first['n_ids_positive']} ids positifs sur {first['n_ids']} : peu de positifs"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("rappel (0–1)")
    ax.set_ylabel("précision (0–1)")
    ax.set_title(title, fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7, frameon=False)
    _french_axes(ax)
    fig.tight_layout()
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
    import matplotlib.patches

    folder = Path(out_dir) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    drawn = []
    with matplotlib.rc_context(RC):
        for name in NAMES:
            svg, csv = folder / f"{name}.svg", folder / f"{name}.csv"
            fig, columns, rows, notes = _DRAW[name](computed, sigma, matplotlib)
            if fig is None:
                svg.unlink(missing_ok=True)
                csv.unlink(missing_ok=True)
                drawn.append(Figure(name, None, None, tuple(notes)))
                continue
            fig.savefig(svg, format="svg", metadata={"Date": None}, bbox_inches="tight")
            write_csv(csv, columns, rows)
            drawn.append(Figure(name, svg, csv, tuple(notes)))
    return drawn
