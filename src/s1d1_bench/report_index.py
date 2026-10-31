"""`bench report`, the index: `<out>/index.md`, the template `report/TEMPLATE.md` with its named holes filled from the
tables of `report.tables` and the figures of `report_figures.figures`.

The template holds the French prose (the caveats of docs/32 § 2, the reading rules, one sentence of reading per table
and figure); this module fills its holes `{name}` mechanically, in neutral numeric wording. A hole without a value,
or a value without a hole, is a ValueError: nothing is left unfilled and nothing computed is dropped silently. The
index never holds the words of a superiority claim (`FORBIDDEN`): `write_index` refuses to write one.

The two clauses of the sentence of docs/32 § 5 are chosen by rules written here and quoted in the template:
`position_vs_ceiling` compares the mean of the ceiling with the 95 % interval of the headline QWK, `vs_floor` reads the
2 sigma rule of the paired difference headline - `length` (table `paired`). Prediction 1 and prediction 2 of docs/34
§ 7.4 come first, a prediction that does not hold first of all, in bold, with its numbers.
"""

import re
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa

from s1d1_bench.report import (
    CALIBRATED,
    FLOOR,
    HEADLINE,
    LITELLM,
    MACRO_F1_NOTE,
    PASSES,
    RAW,
    REFERENCE,
    V7,
    french,
    markdown_table,
)
from s1d1_bench.report_figures import Figure
from s1d1_bench.rubric import LEVELS

STATUS = {"json_error": "refus (json_error)", "truncated": "réponse coupée (truncated)"}  # abstentions, in prose
FORBIDDEN = ("meilleur que", "mieux que")  # docs/34 decision 7: no such sentence, the 2 sigma rule or not
LAYA = "laya-"
HOLE = re.compile(r"\{([a-z][a-z0-9_]*)\}")

LABELS = {
    "name": "nom", "role": "rôle", "n_ids": "ids", "qwk": "QWK", "exact": "exact", "within_one": "± 1",
    "alpha": "alpha ordinal", "tiers_never_predicted": "paliers jamais prédits", "judge": "juge", "note": "note",
    "kind": "type", "draw": "tirage", "share_low_of_ids": "part de low (ids)", "judge_a": "juge A",
    "judge_b": "juge B", "role_a": "rôle A", "role_b": "rôle B", "n_ids_left_out": "ids écartés",
    "thinking_pair": "même modèle (réflexion)", "lang": "langue", "n_ids_all": "ids", "qwk_all": "QWK (truth_all)",
    "exact_all": "exact (truth_all)", "qwk_lang": "QWK (truth_lang)", "exact_lang": "exact (truth_lang)",
    "n_keys": "prompts de base", "n_keys_dropped": "prompts de base écartés",
    "lang_change_rate": "part des prompts de base qui changent de palier", "lang_range_mean": "écart moyen (crans)",
    "group": "groupe", "value": "valeur", "n_ids_disagreement": "ids en désaccord",
    "n_keys_disagreement": "prompts de base en désaccord", "source": "source", "origin": "origine", "tier": "palier",
    "mean_signed_error": "erreur signée moyenne (crans)", "accuracy": "exactitude", "macro_f1": "F1 macro",
    "n_ids_positive": "ids positifs", "f1": "F1", "mcc": "MCC", "auroc": "AUROC", "score_column": "colonne lue",
    "scale": "échelle", "spearman": "Spearman", "auroc_high": "AUROC (high ou expert)", "mae": "MAE", "ece": "ECE",
    "candidate": "candidat", "difference": f"écart apparié de QWK ({REFERENCE} − candidat)",
    "rule_2sigma": "règle des 2 σ", "prediction": "prédiction", "verified": "vérifiée", "model": "modèle",
    "revision": "révision", "date": "date", "n_ids_abstained": "ids en abstention", "n_ids_null_vote": "ids à vote nul",
    "null_votes_ids": "ids des votes nuls", "item": "élément", "why": "pourquoi",
}  # fmt: skip
VALUES = {
    "role": {"candidate": "candidat", "judge-in-truth": "juge dans la vérité",
             "judge-outside-truth": "juge hors de la vérité"},
    "kind": {"informed witness": "témoin informé", "constant witness": "témoin constant", "random draw": "tirage",
             "random min": "minimum des tirages", "random max": "maximum des tirages",
             "random mean": "moyenne des tirages", "judge": "juge", "classifier": "classifieur"},
    "scale": {V7: "score v7", RAW: "brut (hors échelle v7 : rangs seulement)"},
    "prediction": {"prediction_1": "prédiction 1", "prediction_2": "prédiction 2"},
    "group": {"all": "tout", "lang": "langue", "source": "source"},
    "item": {"prompts": "prompts", "prompts_sha256": "sha256 des prompts", "meta": "méta", "date of the run":
             "date du run", "rows": "lignes", "base keys": "prompts de base", "languages": "langues",
             "main judges": "juges principaux", "gemma": "juge Gemma (sorti de truth_all_nogemma)",
             "undecidable": "indécidables", "abstentions": "abstentions", "bootstrap": "bootstrap",
             "random draws": "tirages de random", "sigma": "σ"},
}  # fmt: skip
VALUES["role_a"] = VALUES["role_b"] = VALUES["role"]
VALUES["judge"] = {"mean": "moyenne"}
VALUES["lang"] = {"all": "toutes"}
VALUES["value"] = {"all": "tous les ids"}
VALUES["revision"] = {"n/a": "—"}  # one convention for "nothing here": the dash
COLUMNS = {
    "agreement_all": ["name", "role", "n_ids", "qwk", "exact", "within_one", "alpha", "tiers_never_predicted"],
    "agreement_nogemma": ["name", "role", "n_ids", "qwk", "exact", "within_one", "alpha", "tiers_never_predicted"],
    "ceiling": ["judge", "n_ids", "qwk", "exact", "note"],
    "floor": ["name", "kind", "draw", "n_ids", "qwk", "exact", "within_one"],
    "exact_vs_constant_low": ["name", "n_ids", "exact", "share_low_of_ids"],
    "judge_pairs": ["judge_a", "role_a", "judge_b", "role_b", "n_ids", "n_ids_left_out", "qwk", "exact",
                    "thinking_pair"],
    "by_language": ["name", "role", "lang", "n_ids_all", "qwk_all", "exact_all", "qwk_lang", "exact_lang"],
    "lang_change": ["name", "role", "n_keys", "n_keys_dropped", "lang_change_rate", "lang_range_mean"],
    "truth_lang_vs_all": ["group", "value", "n_ids", "n_ids_disagreement", "n_keys", "n_keys_disagreement"],
    "by_source": ["name", "role", "source", "n_ids", "exact", "qwk"],
    "by_origin": ["name", "role", "origin", "n_ids", "exact", "qwk"],
    "by_tier": ["name", "role", "tier", "n_ids", "exact", "mean_signed_error"],
    "task": ["name", "role", "n_ids", "accuracy", "macro_f1"],
    "needs_tools": ["name", "role", "n_ids", "n_ids_positive", "f1", "mcc", "accuracy", "auroc"],
    "rank_metrics": ["name", "score_column", "scale", "n_ids", "spearman", "auroc_high", "note"],
    "calibration": ["name", "n_ids", "mae", "spearman", "ece"],
    "paired": ["candidate", "difference", "rule_2sigma", "why"],
    "predictions": ["prediction", "name", "lang", "qwk", "verified"],
    "inputs": ["name", "kind", "role", "model", "revision", "date", "n_ids_abstained", "n_ids_null_vote",
               "null_votes_ids"],
    "inputs_summary": ["item", "value"],
}  # fmt: skip
SPLIT = ("agreement_all", "agreement_nogemma", "by_language", "lang_change", "by_source", "by_origin", "by_tier",
         "task", "needs_tools")  # fmt: skip  # candidates and judges in two tables: the judges are the instrument
SKIPPED = {
    "headline": "chiffre principal",
    "floor": "plancher",
    "exact next to constant-low": "exactitude à côté de constant-low",
    "reference of the paired differences": "référence des écarts appariés",
    "prediction 1": "prédiction 1",
    "prediction 2": "prédiction 2",
    "no headline number for it": "pas de chiffre principal pour lui",
    "no floor row for it": "pas de ligne de plancher",
    "no row in exact_vs_constant_low": "pas de ligne dans exact_vs_constant_low",
    "no paired difference": "pas d'écart apparié",
    "prediction 1 not evaluated": "prédiction 1 non évaluée",
    "prediction 2 not evaluated": "prédiction 2 non évaluée",
}
CAPTIONS = {
    "qwk-bars": "QWK contre truth_all ({n_ids}), IC 95 % sur 1 000 tirages des {n_base} ; plafond, plancher, ± 2 σ",
    "qwk-by-language": "QWK contre truth_all par langue (ids de la langue), IC 95 % sur 1 000 tirages des {n_base}",
    "confusion": "Matrices de confusion normalisées par ligne : part des ids de chaque palier de truth_all (0–1)",
    "reliability-s1d1": "Score de s1d1 (échelle v7, 0–1) contre l'ancre du palier de truth_all (0–1), 10 cases",
    "judge-spread-by-tier": "Écart entre juges principaux (crans) par palier de truth_all, en nombre d'ids",
    "language-disagreements": "Part des ids où truth_lang ≠ truth_all (0–1), par langue et par source, avec les "
    "nombres d'ids",
    "needs-tools-pr": "Précision et rappel de needs_tools (0–1) selon le seuil de probabilité, sur {n_ids}",
}


def fill(template: str, values: dict[str, str]) -> str:
    """`template` with each hole `{name}` replaced by `values[name]`; a ValueError naming them when a hole has no
    value or a value has no hole."""
    holes = set(HOLE.findall(template))
    missing, unused = sorted(holes - set(values)), sorted(set(values) - holes)
    if missing:
        raise ValueError(f"the template has holes without a value: {', '.join(missing)}")
    if unused:
        raise ValueError(f"values without a hole in the template: {', '.join(unused)}")
    return HOLE.sub(lambda m: values[m.group(1)], template)


def _code(name: str) -> str:
    return f"`{name}`"


def _estimate(row: dict | None, metric: str = "qwk") -> str:
    """`0,807 [0,734 ; 0,852]`, the value alone when it has no interval, a dash when it is undefined."""
    if row is None:
        return "—"
    text = french(row[metric])
    lo, hi = row.get(f"{metric}_lo"), row.get(f"{metric}_hi")
    if text != "—" and french(lo) != "—" and french(hi) != "—":
        text += f" [{french(lo)} ; {french(hi)}]"
    return text


def _ids(n: int) -> str:
    """`n` ids in French: 0 and 1 take the singular."""
    return f"{n} id" if n <= 1 else f"{n} ids"


def _singular(text: str) -> str:
    """'0 ids', '1 ids' (the counts of `inputs_summary`) in French: '0 id', '1 id'."""
    return re.sub(r"\b([01]) ids\b", r"\1 id", text)


def _summary_value(item: str, value: str) -> str:
    """A value of `inputs_summary` as the index writes it: σ with a decimal comma, '1 000 tirages', '1 id'."""
    if item == "sigma":
        return french(float(value))
    if item == "bootstrap":
        value = re.sub(r"\b(\d{1,3})(\d{3}) tirages", r"\1 \2 tirages", value)
    return _singular(value)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}".replace(".", ",") + " %"


def _by(table: pa.Table, key: str) -> dict:
    return {r[key]: r for r in table.to_pylist()} if key in table.column_names else {}


def _summary(computed: dict[str, pa.Table]) -> dict[str, str]:
    return {r["item"]: r["value"] for r in computed["inputs_summary"].to_pylist()}


def _render(table: pa.Table, name: str, keep=None, drop: Sequence[str] = ()) -> str:
    """One table of the report as French Markdown: the columns of `COLUMNS[name]` (minus `drop`), the rows where
    `keep(row)` holds, the values of `VALUES` in French; "_(aucune ligne)_" for none."""
    rows = [r for r in table.to_pylist() if keep is None or keep(r)]
    if not rows:
        return "_(aucune ligne)_"
    if name == "paired":  # its interval columns are lo, hi: named as markdown_table pairs them; and why it fails
        rows = [r | {"difference_lo": r["lo"], "difference_hi": r["hi"], "why": _why(r)} for r in rows]
    if name == "inputs_summary":
        rows = [r | {"value": _summary_value(r["item"], r["value"])} for r in rows]
    for row in rows:
        for column, mapping in VALUES.items():
            if column in row and row[column] in mapping:
                row[column] = mapping[row[column]]
    present = set(rows[0])
    columns = [c for c in COLUMNS[name] if c in present and c not in drop]
    return markdown_table(pa.Table.from_pylist(rows), columns=columns, labels=LABELS)


def _why(row: dict) -> str:
    """Why a paired difference passes the 2 sigma rule or not (the verdict is the table's, `report.two_sigma_rule`)."""
    lo, hi = row["lo"], row["hi"]
    if row["rule_2sigma"] == PASSES:
        return "l'IC exclut zéro et se tient au-delà de 2 σ"
    if french(lo) == "—" or french(hi) == "—":
        return "pas d'intervalle"
    if lo <= 0 <= hi:
        return "l'IC contient zéro"
    return "l'IC exclut zéro, mais sa borne la plus proche de zéro n'est pas au-delà de 2 σ"


def _tables(computed: dict[str, pa.Table]) -> dict[str, str]:
    out = {}
    for name in COLUMNS:
        table = computed[name]
        if name in SPLIT:
            out[f"table_{name}_candidates"] = _render(table, name, lambda r: r["role"] == "candidate", drop=["role"])
            out[f"table_{name}_judges"] = _render(table, name, lambda r: r["role"] != "candidate")
        else:
            out[f"table_{name}"] = _render(table, name)
    return out


def _figures(figures: Sequence[Figure], n_ids: str, n_base: str) -> dict[str, str]:
    out = {}
    for figure in figures:
        caption = CAPTIONS.get(figure.name, figure.name).format(n_ids=n_ids, n_base=n_base)
        hole = "figure_" + figure.name.replace("-", "_")
        if figure.svg is None:
            notes = " ; ".join(figure.notes) or "rien à tracer"
            out[hole] = f"_Figure `{figure.name}` non tracée : {notes}._"
            continue
        text = (
            f"![{caption}](figures/{figure.svg.name})\n\n"
            f"_{caption}._ Données : [`figures/{figure.csv.name}`](figures/{figure.csv.name})."
        )
        if figure.notes:
            text += " Absent de la figure : " + " ; ".join(figure.notes) + "."
        out[hole] = text
    return out


def _and(items: Sequence[str]) -> str:
    """'a', 'a et b', 'a, b et c'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " et " + items[-1]


def _overlap(a: dict, b: dict) -> bool | None:
    """Whether the 95 % intervals of two rows (`qwk_lo`, `qwk_hi`) overlap; None when one is missing."""
    bounds = (a.get("qwk_lo"), a.get("qwk_hi"), b.get("qwk_lo"), b.get("qwk_hi"))
    if any(french(v) == "—" for v in bounds):
        return None
    return a["qwk_lo"] <= b["qwk_hi"] and b["qwk_lo"] <= a["qwk_hi"]


def _overlap_words(pairs: Sequence[tuple[str, dict, str, dict]]) -> str:
    """'les IC de a et de b se recouvrent ; ...' for each pair (name, row, name, row)."""
    words = []
    for left, a, right, b in pairs:
        found = _overlap(a, b)
        verb = "se recouvrent" if found else "ne se recouvrent pas" if found is False else "n'ont pas tous deux d'IC"
        words.append(f"les IC de {left} et de {right} {verb}")
    return " ; ".join(words)


def _predictions_first(computed: dict[str, pa.Table], n_ids: str, n_base: str) -> str:
    """Prediction 2 and prediction 1 of docs/34 § 7.4: those that do not hold first (in bold, with the numbers), then
    those not evaluated, then those that hold; one line saying so when both hold. Each says that it is read on the
    point values and whether the intervals involved overlap."""
    rows = computed["predictions"].to_pylist()
    failed, absent, held = [], [], []
    p2 = {r["name"]: r for r in rows if r["prediction"] == "prediction_2"}
    floor, litellm, reference = _code(FLOOR), _code(LITELLM), _code(REFERENCE)
    if p2:
        said = (
            f"Écrite avant les chiffres (docs/34 § 7.4), elle disait : {litellm} devait se tenir entre {floor} et "
            f"{reference}. QWK contre `truth_all`, {n_ids}, IC 95 % sur 1 000 tirages des {n_base}."
        )
        point = (
            "La vérification porte sur les valeurs ponctuelles ; "
            + _overlap_words(
                [(litellm, p2[LITELLM], floor, p2[FLOOR]), (litellm, p2[LITELLM], reference, p2[REFERENCE])]
            )
            + "."
        )
        numbers = {name: f"{_code(name)} ({_estimate(row)})" for name, row in p2.items()}
        if next(iter(p2.values()))["verified"]:
            held.append(
                f"La prédiction 2 est vérifiée : {numbers[LITELLM]} se tient entre {numbers[FLOOR]} et "
                f"{numbers[REFERENCE]}. {said} {point}"
            )
        else:
            low, mid, top = (p2[n]["qwk"] for n in (FLOOR, LITELLM, REFERENCE))
            if mid < low:
                where = f"est au-dessous de {numbers[FLOOR]}"
            elif mid == low:
                where = f"est au niveau de {numbers[FLOOR]}"
            elif mid > top:
                where = f"est au-dessus de {numbers[REFERENCE]}"
            elif mid == top:
                where = f"est au niveau de {numbers[REFERENCE]}"
            else:
                where = f"ne se tient pas entre {numbers[FLOOR]} et {numbers[REFERENCE]}"
            failed.append(
                f"**La prédiction 2 n'est pas vérifiée : {numbers[LITELLM]} {where}, au lieu de se tenir entre "
                f"{floor} et {numbers[REFERENCE] if numbers[REFERENCE] not in where else reference}.** {said} {point}"
            )
    else:
        absent.append("La prédiction 2 n'a pas été évaluée : un de ses candidats est absent (voir « Ce qui manque »).")
    p1 = {r["lang"]: r for r in rows if r["prediction"] == "prediction_1"}
    if p1:
        per_lang = computed["by_language"].to_pylist()
        n_lang = next((r["n_ids_all"] for r in per_lang if r["name"] == LITELLM), "?")
        listed = " ; ".join(f"{lang} {_estimate(r)}" for lang, r in p1.items())
        said = (
            f"Écrite avant les chiffres (docs/34 § 7.4), elle disait : les listes de mots-clés de {litellm} sont "
            f"anglaises, son QWK devait tomber en français, en russe et en chinois. QWK contre `truth_all`, {n_lang} "
            f"ids par langue, IC 95 % sur 1 000 tirages des {n_base}."
        )
        eng = p1.get("eng")
        others = [lang for lang in p1 if lang != "eng"]
        overlapping = [lang for lang in others if eng and _overlap(p1[lang], eng)]
        apart = [lang for lang in others if eng and _overlap(p1[lang], eng) is False]
        parts = []
        if overlapping:
            verb = "recouvre" if len(overlapping) == 1 else "recouvrent"
            subject = "l'IC" if len(overlapping) == 1 else "les IC"
            parts.append(f"{subject} de {_and(overlapping)} {verb} celui de eng")
        if apart:
            verb = "ne recouvre pas" if len(apart) == 1 else "ne recouvrent pas"
            subject = "l'IC" if len(apart) == 1 else "les IC"
            parts.append(f"{subject} de {_and(apart)} {verb} celui de eng")
        point = "La vérification porte sur les valeurs ponctuelles seulement"
        point += (" : " + " ; ".join(parts) if parts else "") + " (aucun écart apparié entre langues n'est calculé)."
        if next(iter(p1.values()))["verified"]:
            held.append(f"La prédiction 1 est vérifiée : QWK de {litellm} par langue : {listed}. {point} {said}")
        else:
            above = [lang for lang in others if not (eng and p1[lang]["qwk"] < eng["qwk"])]
            if not eng:
                why = "il n'y a pas de QWK en eng"
            elif len(above) == 1:
                why = f"le QWK en {above[0]} n'est pas au-dessous de celui en eng"
            else:
                why = f"les QWK en {_and(above)} ne sont pas au-dessous de celui en eng"
            failed.append(
                f"**La prédiction 1 n'est pas vérifiée : {why} (QWK de {litellm} par langue : {listed}).** {point} "
                f"{said}"
            )
    else:
        absent.append(f"La prédiction 1 n'a pas été évaluée : {litellm} est absent (voir « Ce qui manque »).")
    lines = failed + absent + held
    if not failed and not absent:
        lines.insert(0, "Les deux prédictions se vérifient.")
    return "\n\n".join(lines)


def _sentence(computed: dict[str, pa.Table], run: dict) -> dict[str, str]:
    """The holes of the sentence of docs/32 § 5 and the rules of its two clauses."""
    summary = _summary(computed)
    agreement = {r["name"]: r for r in computed["agreement_all"].to_pylist() if r["role"] == "candidate"}
    ceiling = _by(computed["ceiling"], "judge")
    paired = _by(computed["paired"], "candidate")
    headline = HEADLINE[0]

    def qwk_of(name: str) -> str:
        return _estimate(agreement[name]) if name in agreement else "absent de ce run"

    values = {
        "n_ids": summary.get("rows", "?").split()[0],
        "n_base": summary.get("base keys", "?").split()[0],
        "n_langs": str(len(summary.get("languages", "").split(", "))),
        "k_judges": str(len(run["main"])),
        "headline_name": _code(headline),
        "headline_qwk": french(agreement[headline]["qwk"]) if headline in agreement else "—",
        "headline_ci": _estimate(agreement[headline]).partition(" ")[2] if headline in agreement else "(absent)",
        "ceiling_mean": _estimate(ceiling.get("mean")),
        "ceiling_values": ", ".join(f"{_code(j)} {french(r['qwk'])}" for j, r in ceiling.items() if j != "mean")
        or "aucun juge principal",
        "laya_english": qwk_of("laya-english-tier"),
        "laya_multilingual": qwk_of("laya-multilingual-tier"),
        "litellm": qwk_of(LITELLM),
        "length": qwk_of(FLOOR),
    }
    others = [f"{_code(name)} {qwk_of(name)}" for name in HEADLINE[1:]]
    values["headline_others"] = (
        "L'autre chiffre principal (même vérité, même unité) : " + " ; ".join(others) + "." if others else ""
    )
    head, mean = agreement.get(headline), ceiling.get("mean")
    if head is None or mean is None or french(head["qwk_lo"]) == "—":
        values["position_vs_ceiling"] = "ne se place pas par rapport au plafond (chiffre principal ou plafond absent)"
    elif head["qwk_lo"] <= mean["qwk"] <= head["qwk_hi"]:
        values["position_vs_ceiling"] = "contient la moyenne du plafond"
    elif head["qwk_hi"] < mean["qwk"]:
        values["position_vs_ceiling"] = "est au-dessous de la moyenne du plafond"
    else:
        values["position_vs_ceiling"] = "est au-dessus de la moyenne du plafond"
    pair = paired.get(FLOOR)
    if pair is None or headline != REFERENCE:
        values["vs_floor"] = f"sans comparaison au plancher {_code(FLOOR)} (candidat absent)"
    else:
        verb = "se distingue" if pair["rule_2sigma"] == PASSES else "ne se distingue pas"
        gap = _estimate(pair | {"qwk": pair["difference"], "qwk_lo": pair["lo"], "qwk_hi": pair["hi"]})
        values["vs_floor"] = (
            f"{verb} du plancher {_code(FLOOR)} selon la règle des 2 σ (écart apparié {_code(headline)} − "
            f"{_code(FLOOR)} : {gap})"
        )
    return values


def _reading(computed: dict[str, pa.Table], run: dict, sigma: float, classifiers_root: Path | None) -> dict[str, str]:
    """The holes of the reading rules, the folders and the sentences beside the tables."""
    summary = _summary(computed)
    candidates = [e["name"] for e in run["classifiers"]]
    main = list(run["main"])
    others = [e["name"] for e in run["judges"] if e["name"] not in main]
    never = [
        f"{_code(r['name'])} : {r['tiers_never_predicted']}"
        for r in computed["agreement_all"].to_pylist()
        if r["role"] == "candidate" and r["tiers_never_predicted"]
    ]
    abstained = {j: [a["id"] for a in v] for j, v in run.get("abstentions", {}).items()}
    listed = [
        f"{_code(j)} sur " + ", ".join(f"{_code(a['id'])} : {STATUS.get(a.get('status'), a.get('status'))}" for a in v)
        for j, v in run.get("abstentions", {}).items()
        if v
    ]
    nulls = []
    for row in computed["inputs"].to_pylist():
        if row["kind"] == "judge" and row["n_ids_null_vote"]:
            ids = [i for i in row["null_votes_ids"].split(", ") if i]
            marked = [_code(i) + (" (abstention)" if i in abstained.get(row["name"], []) else "") for i in ids]
            nulls.append(f"{_code(row['name'])} {_ids(row['n_ids_null_vote'])} : {', '.join(marked)}")
    spread = computed["judge_spread_by_tier"].to_pylist()
    per_tier = {
        level: next((r["n_ids_tier"] + r["n_ids_null_spread_tier"] for r in spread if r["tier"] == level), 0)
        for level in LEVELS
    }
    laya = [_code(n) for n in candidates if n.startswith(LAYA)]
    extra = []
    if classifiers_root is not None and Path(classifiers_root).is_dir():
        extra = sorted(p.name for p in Path(classifiers_root).iterdir() if p.is_dir() and p.name not in candidates)
    low = _by(computed["exact_vs_constant_low"], "name").get(LITELLM)
    needs = computed["needs_tools"].to_pylist()
    first_needs = needs[0] if needs else None
    task = [_code(r["name"]) for r in computed["task"].to_pylist() if r["role"] == "candidate"]
    no_task = [_code(n) for n in candidates if _code(n) not in task]
    rank = computed["rank_metrics"].to_pylist()
    calibrated = [_code(r["name"]) for r in computed["calibration"].to_pylist()]
    return {
        "main_judges": ", ".join(_code(j) for j in main),
        "other_judges": ", ".join(_code(j) for j in others) or "aucun",
        "gemma": ", ".join(_code(j) for j in run.get("gemma", [])) or "aucun juge",
        "tiers_never_predicted": " ; ".join(never) or "aucun",
        "undecidable": _singular(summary.get("undecidable", "?")),
        "abstentions": _ids(sum(len(v) for v in abstained.values())) + (" : " + " ; ".join(listed) if listed else ""),
        "null_votes": " ; ".join(nulls) or "aucun",
        "tier_counts": ", ".join(f"{level} {n}" for level, n in per_tier.items()) + " ids",
        "laya_in_report": ", ".join(laya) or "aucun",
        "folders_not_in_report": ", ".join(_code(n) for n in extra) or "aucun",
        "litellm_next_to_constant": (
            f"{_percent(low['share_low_of_ids'])} des paliers de {_code(LITELLM)} sont `low` ({low['n_ids']} ids) : "
            f"son exactitude se lit à côté de celle de `constant-low`, qui dit `low` partout."
            if low
            else f"{_code(LITELLM)} est absent de ce run : pas de comparaison avec `constant-low`."
        ),
        "task_candidates": ", ".join(task) or "aucun",
        "no_task_candidates": ", ".join(no_task) or "aucun",
        "macro_f1_note": MACRO_F1_NOTE.replace("macro-F1", "F1 macro") + ".",
        "needs_tools_note": (
            f"Peu de positifs : {first_needs['n_ids_positive']} ids positifs sur {first_needs['n_ids']} dans "
            "`truth_all`, l'IC le dit."
            if first_needs
            else "Aucune ligne needs_tools."
        ),
        "rank_candidates": ", ".join(_code(r["name"]) for r in rank if r["scale"] == RAW) or "aucun",
        "calibrated_candidates": ", ".join(calibrated) or f"aucun (pas de candidat `{CALIBRATED}*` avec un score)",
        "reference": _code(REFERENCE),
        "sigma": french(sigma),
        "two_sigma": french(2 * sigma),
    }


def _missing(computed: dict[str, pa.Table], figures: Sequence[Figure]) -> str:
    lines = [
        f"- {SKIPPED.get(r['role'], r['role'])}, {_code(r['name'])} : {SKIPPED.get(r['consequence'], r['consequence'])}"
        for r in computed["skipped"].to_pylist()
    ]
    for figure in figures:
        lines += [f"- figure {_code(figure.name)} : {note}" for note in figure.notes]
    return "\n".join(lines) or "Aucun : tous les rôles du rapport ont leur candidat, toutes les figures sont complètes."


def write_index(
    out_dir: Path | str,
    tables: dict[str, pa.Table],
    figures: Sequence[Figure],
    run: dict,
    template: Path | str,
    classifiers_root: Path | str | None = None,
) -> Path:
    """Fill `template` (a path) from the tables of `report.tables`, the figures and the run record, and write
    `out_dir/index.md`. ValueError when a hole of the template has no value (or a value no hole), or when the text
    would hold a word of `FORBIDDEN`. `classifiers_root`, when given, names its folders that the run did not load."""
    summary = _summary(tables)
    sigma = float(summary.get("sigma", "nan"))
    n_ids, n_base = summary.get("rows", "? ids"), summary.get("base keys", "? prompts de base")
    values = {
        "set_name": Path(str(run.get("prompts", "?"))).stem,
        "run_date": str(run.get("date", "?")),
        "inputs_line": (
            f"prompts `{run.get('prompts')}` (sha256 `{run.get('prompts_sha256')}`), méta `{run.get('meta')}`, "
            f"{len(run['judges'])} juges, {len(run['classifiers'])} classifieurs ; {n_ids}, {n_base}"
        ),
        "predictions_first": _predictions_first(tables, n_ids, n_base),
        "missing": _missing(tables, figures),
        "raw_data": ", ".join(f"[`{name}.csv`](tables/{name}.csv)" for name in tables),
    }
    values |= _sentence(tables, run)
    values |= _reading(tables, run, sigma, Path(classifiers_root) if classifiers_root is not None else None)
    values |= _tables(tables)
    values |= _figures(figures, n_ids, n_base)
    text = fill(Path(template).read_text(encoding="utf-8"), values)
    lowered = text.lower()
    found = [words for words in FORBIDDEN if words in lowered]
    if found:
        raise ValueError(f"the index would say {found[0]!r}: no superiority sentence in the report")
    path = Path(out_dir) / "index.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
