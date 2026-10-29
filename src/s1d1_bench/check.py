"""`bench check`: flags on the translations of a set, one per broken rule, for a human to read; never a correction.

`structure` compares each translation with its English source (fences, URLs, numbers, quoted strings, lines, length,
leaked prompt tags), `labels` compares the labels a judge gave both, `review` gathers the flags into a sheet.
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from s1d1_bench.export import read_translations
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import read_labels

LENGTH_BOUNDS = {"fra": (0.7, 1.6), "rus": (0.7, 1.6), "zho": (0.25, 0.9)}  # characters, translation / English
LINES_TOLERANCE = 0.2
STRUCTURE_RULES = ["fences", "urls", "numbers", "quotes", "lines", "length", "tag", "envelope"]
LABEL_RULES = ["task", "needs_tools"]  # the tier (`complexity`) is never compared
CHECKS = ["structure", "labels", "request"]
# every rule, in the order flags are sorted and listed after their id
RULES = [*STRUCTURE_RULES, *LABEL_RULES, "label_missing", "request"]
FLAG_KEYS = ["id", "base_key", "lang", "check", "rule", "detail", "lifted"]
DETAIL_SHOWN = 10  # items of a set difference written in a detail

_URL = re.compile(r"https?://\S+")
_NUMBER = re.compile(r"\d+")
_QUOTED = re.compile(r'"([^"\n]*)"')
_TAG = re.compile(r"</?prompt")


def _fences(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("```"))


def _numbers(text: str) -> Counter:
    # NFKC: the full-width digits a Chinese text may use count as the ASCII ones
    return Counter(unicodedata.normalize("NFKC", m) for m in _NUMBER.findall(text))


def _envelope_lines(text: str) -> int:
    """Lines that could pass for an envelope line (same rule as `export.check_envelope_safe`)."""
    return sum(
        1 for raw in text.splitlines() if (line := raw.strip().lower()) == "</prompt>" or line.startswith("<prompt")
    )


def _difference(eng: Counter, tr: Counter) -> str:
    """`eng=[...] tr=[...]`: what each side has that the other lacks (as JSON lists, at most DETAIL_SHOWN each)."""

    def shown(items: Counter) -> str:
        values = sorted(items.elements())
        more = f" (+{len(values) - DETAIL_SHOWN})" if len(values) > DETAIL_SHOWN else ""
        return json.dumps(values[:DETAIL_SHOWN], ensure_ascii=False) + more

    return f"eng={shown(eng - tr)} tr={shown(tr - eng)}"


def structural_faults(english: str, translation: str, lang: str) -> list[dict]:
    """One `{"rule", "detail"}` per broken rule, in the order of STRUCTURE_RULES; `detail` says both values."""
    if lang not in LENGTH_BOUNDS:
        raise ValueError(f"no length bounds for {lang!r} (known: {', '.join(LENGTH_BOUNDS)})")
    faults = []

    def fault(rule: str, detail: str) -> None:
        faults.append({"rule": rule, "detail": detail})

    eng, tr = _fences(english), _fences(translation)
    if eng != tr:
        fault("fences", f"eng={eng} tr={tr}")
    eng_urls, tr_urls = set(_URL.findall(english)), set(_URL.findall(translation))
    if eng_urls != tr_urls:
        fault("urls", _difference(Counter(eng_urls), Counter(tr_urls)))
    eng_numbers, tr_numbers = _numbers(english), _numbers(translation)
    if eng_numbers != tr_numbers:
        fault("numbers", _difference(eng_numbers, tr_numbers))
    eng_quoted = Counter(q for line in english.splitlines() for q in _QUOTED.findall(line))
    tr_quoted = Counter(q for line in translation.splitlines() for q in _QUOTED.findall(line))
    if eng_quoted != tr_quoted:
        fault("quotes", _difference(eng_quoted, tr_quoted))
    eng, tr = len(english.splitlines()), len(translation.splitlines())
    if abs(tr - eng) > max(1, LINES_TOLERANCE * eng):  # one line of slack, so a 1-line prompt may become 2
        fault("lines", f"eng={eng} tr={tr}")
    eng, tr = len(english), len(translation)
    low, high = LENGTH_BOUNDS[lang]
    if eng and not low <= tr / eng <= high:
        fault("length", f"eng={eng} tr={tr} (ratio {tr / eng:.2f}, bounds {low}-{high})")
    eng, tr = len(_TAG.findall(english.lower())), len(_TAG.findall(translation.lower()))
    if tr > eng:  # a tag the English already holds is the prompt's own
        fault("tag", f"eng={eng} tr={tr}")
    eng, tr = _envelope_lines(english), _envelope_lines(translation)
    if tr > eng:
        fault("envelope", f"eng={eng} tr={tr}")
    return faults


def _show(value) -> str:
    return json.dumps(value) if isinstance(value, bool) else str(value)


def label_faults(english: dict, label: dict, witness: dict | None) -> list[dict]:
    """One `{"rule", "detail", "witness", "lifted"}` per label of LABEL_RULES that differs from the English one.

    `witness` is the label of the witness translation: equal to ours, the change comes from the judge, not from our
    translation, and the flag is lifted (`agrees_with_translation`); equal to the English one, it stays
    (`agrees_with_english`); equal to neither, it stays too (`agrees_with_neither`)."""
    faults = []
    for rule in LABEL_RULES:
        if label[rule] == english[rule]:
            continue
        if witness is None:
            agrees = None
        elif witness[rule] == label[rule]:
            agrees = "agrees_with_translation"
        elif witness[rule] == english[rule]:
            agrees = "agrees_with_english"
        else:
            agrees = "agrees_with_neither"
        detail = f"eng={_show(english[rule])} tr={_show(label[rule])}"
        lifted = agrees == "agrees_with_translation"
        faults.append({"rule": rule, "detail": detail, "witness": agrees, "lifted": lifted})
    return faults


def _sorted_flags(flags: list[dict]) -> list[dict]:
    return sorted(flags, key=lambda f: (f["id"], RULES.index(f["rule"])))


def _fenced(text: str) -> str:
    """`text` in a Markdown code block whose fence is longer than any run of backticks inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def review_sheet(
    flags: list[dict],
    english: dict[str, str],
    translations: dict[str, str],
    witness: dict[str, str],
    lang: str | None = None,
) -> str:
    """The Markdown sheet a human fills in: a header with the counts, then one section per flagged id (sorted), its
    rules (`[levé]` when lifted), the English, the translation, the witness when there is one, and the two lines to
    fill, `avis :` and `décision :`. The three text maps are keyed by base key; `lang` defaults to that of the flags."""
    flags = _sorted_flags(flags)
    if lang is None:
        lang = flags[0]["lang"] if flags else "?"
    by_id: dict[str, list[dict]] = {}
    for f in flags:
        by_id.setdefault(f["id"], []).append(f)
    lifted = sum(f["lifted"] for f in flags)
    lines = [
        f"# Relecture des traductions : {lang}",
        "",
        f"Langue : {lang}. {len(by_id)} ids signalés, {len(flags)} signalements, dont {lifted} levé(s) "
        "(le témoin reçoit du juge la même étiquette que notre traduction).",
        "",
        "Pour chaque id : `avis :`, puis `décision :` rien, correction ou veto.",
    ]
    for flag_id, group in by_id.items():
        key = group[0]["base_key"]
        lines += ["", f"## {flag_id}", ""]
        lines += [f"- {f['check']} / {f['rule']} : {f['detail']}" + (" [levé]" if f["lifted"] else "") for f in group]
        lines += ["", "Anglais :", "", _fenced(english[key]), "", "Traduction :", "", _fenced(translations[key])]
        if key in witness:
            lines += ["", "Témoin :", "", _fenced(witness[key])]
        lines += ["", "avis :", "", "décision :"]
    return "\n".join(lines) + "\n"


def _flag(row: dict, lang: str, check: str, rule: str, detail: str, lifted: bool = False) -> dict:
    return {
        "id": row["id"],
        "base_key": row["base_key"],
        "lang": lang,
        "check": check,
        "rule": rule,
        "detail": detail,
        "lifted": lifted,
    }


def _english_texts(path: Path) -> dict[str, str]:
    """base key -> English text, from a prompts file whose ids end with `-eng`."""
    texts = {}
    for row in read_jsonl(path):
        if not row["id"].endswith("-eng"):
            raise ValueError(f"{path}: id {row['id']} is not <base_key>-eng")
        texts[row["id"].removesuffix("-eng")] = row["prompt"]
    return texts


def _structure_flags(rows: list[dict], english: dict[str, str], lang: str) -> list[dict]:
    flags = []
    for row in rows:
        if row["base_key"] not in english:
            raise ValueError(f"{row['id']}: no English prompt {row['base_key']}-eng")
        for f in structural_faults(english[row["base_key"]], row["prompt"], lang):
            flags.append(_flag(row, lang, "structure", f["rule"], f["detail"]))
    return flags


def _label_flags(rows: list[dict], lang: str, eng_dir: Path, tr_dir: Path, witness_dir: Path | None) -> list[dict]:
    """The label flags of each translated id. An id the English or the translation judge did not label (its labels
    file holds only the ids labelled ok) gives a `label_missing` flag naming the folder, not a comparison."""
    eng_labels, tr_labels = read_labels(eng_dir / "labels.jsonl"), read_labels(tr_dir / "labels.jsonl")
    witness_labels = read_labels(witness_dir / "labels.jsonl") if witness_dir else {}
    flags = []
    for row in rows:
        eng_id = f"{row['base_key']}-eng"
        sides = [(eng_id, eng_dir, eng_labels), (row["id"], tr_dir, tr_labels)]
        missing = [f"{i} in {folder}" for i, folder, labels in sides if i not in labels]
        if missing:
            flags.append(_flag(row, lang, "labels", "label_missing", "no label for " + ", ".join(missing)))
            continue
        witness = witness_labels.get(row["id"])
        for f in label_faults(eng_labels[eng_id], tr_labels[row["id"]], witness):
            detail = f["detail"] + (f" witness={_show(witness[f['rule']])}" if witness else "")
            flags.append(_flag(row, lang, "labels", f["rule"], detail, f["lifted"]))
    return flags


def _request_flags(path: Path, rows: list[dict], lang: str) -> list[dict]:
    """A `same_request: false` line of `<lang>.request.jsonl` is a `request` flag whose detail is its reason."""
    by_id = {row["id"]: row for row in rows}
    flags = []
    for line in read_jsonl(path):
        if line["same_request"] is False:
            if line["id"] not in by_id:
                raise ValueError(f"{path}: {line['id']} is not a translation of the set")
            flags.append(_flag(by_id[line["id"]], lang, "request", "request", line["reason"]))
    return flags


def _print_counts(lang: str, kind: str, flags: list[dict], total: int) -> None:
    ids = len({f["id"] for f in flags})
    print(f"{lang} {kind}: {len(flags)} flag(s) on {ids} of {total} id(s)")
    for rule in RULES:
        hits = [f for f in flags if f["rule"] == rule]
        if hits:
            lifted = sum(f["lifted"] for f in hits)
            print(f"  {rule}: {len(hits)}" + (f" ({lifted} lifted)" if lifted else ""))


def run_check(args) -> int:
    """`bench check <kind>`: `structure` and `labels` rewrite `<out>/<set>/<lang>.<kind>.jsonl` (derivations of the
    files they read); `review` writes `<lang>.review.md` from the check files present, and never replaces a sheet
    whose text differs from the new one (it may have been filled in)."""

    def fail(message: str, code: int = 1) -> int:
        print(f"bench check {args.kind}: {message}", file=sys.stderr)
        return code

    if args.kind == "request":
        return fail("not implemented yet", 2)
    lang = args.lang
    out = Path(args.out) / args.set
    # the model's text, not the corrected one: the checks are on what the translator produced
    rows = sorted(read_translations([Path(args.translations)], [lang], corrections=False)[lang], key=lambda r: r["id"])

    if args.kind == "structure":
        flags = _structure_flags(rows, _english_texts(Path(args.english)), lang)
    elif args.kind == "labels":
        if not args.labels_english or not args.labels:
            return fail("needs --labels-english and --labels (judge folders holding labels.jsonl)")
        witness_dir = Path(args.labels_witness) if args.labels_witness else None
        flags = _label_flags(rows, lang, Path(args.labels_english), Path(args.labels), witness_dir)
    if args.kind in ("structure", "labels"):
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / f"{lang}.{args.kind}.jsonl", _sorted_flags(flags))
        _print_counts(lang, args.kind, flags, len(rows))
        return 0

    found = [out / f"{lang}.{check}.jsonl" for check in CHECKS if (out / f"{lang}.{check}.jsonl").exists()]
    if not found:
        return fail(f"no {lang}.structure.jsonl, {lang}.labels.jsonl or {lang}.request.jsonl in {out}")
    flags = []
    for path in found:
        if path.name.endswith(".request.jsonl"):
            flags += _request_flags(path, rows, lang)
        else:
            flags += read_jsonl(path)
    english = _english_texts(Path(args.english))
    translations = {row["base_key"]: row["prompt"] for row in rows}
    witness = {}
    if args.witness:
        witness_rows = read_translations([Path(args.witness)], [lang], corrections=False)[lang]
        witness = {row["base_key"]: row["prompt"] for row in witness_rows}
    sheet = review_sheet(flags, english, translations, witness, lang)
    path = out / f"{lang}.review.md"
    if path.exists() and path.read_text(encoding="utf-8") != sheet:
        return fail(f"{path} exists and differs from the new sheet (filled in, or new flags); move it away first")
    path.write_text(sheet, encoding="utf-8", newline="\n")
    _print_counts(lang, args.kind, flags, len(rows))
    return 0
