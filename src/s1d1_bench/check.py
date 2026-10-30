"""`bench check`: flags on the translations of a set, one per broken rule, for a human to read; never a correction.

`structure` compares each translation with its English source (fences, URLs, numbers, quoted strings, lines, length,
leaked prompt tags), `labels` compares the labels a judge gave both, `request` asks a local model whether the
translation is the same request as the English, `review` gathers the flags into a sheet.
"""

import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from s1d1_bench.export import read_translations
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import read_labels
from s1d1_bench.judge_runner import GenerateFn, GenerateResult, local_path_keys, resume_conflicts
from s1d1_bench.runfile import find_json_objects, read_done, repair_log, strip_reasoning, write_atomic
from s1d1_bench.translate import LANGUAGES

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


# --- request: does the translation ask for the same thing as the English? A local model answers, one pair per call

REQUEST_INSTRUCTION = (
    "Is the translation between the second pair of tags the same request as the English message between the first "
    "pair? Return the JSON object now."
)
REQUEST_RETRY_SUFFIX = "\nOutput only the JSON object."
REQUEST_KEYS = ["same_request", "reason"]
REQUEST_CONSIGNE = "consigne/check-same-request-v1.md"  # relative to the working directory, as the CLI defaults
RAW_MAX_CHARS = 2000
# <lang>.request.meta.json keys a resumed run must share with the run it resumes
REQUEST_RESUME_KEYS = [
    "model",
    "revision",
    "thinking",
    "consigne_sha256",
    "english_prompts_sha256",
    "translations_sha256",
    "max_tokens",
]
_COMMIT = re.compile(r"[0-9a-f]{40}")


def build_request_messages(consigne: str, english: str, translation: str, lang: str, retry: bool = False) -> list[dict]:
    """System message = the consigne, byte-exact; user message = the English and the translation as data, each
    between its own <prompt lang="..."> tags. `retry` appends a nudge, so that a greedy second try is not the same."""
    user = (
        '<prompt lang="eng">\n' + english + '\n</prompt>\n\n<prompt lang="' + lang + '">\n' + translation
        + "\n</prompt>\n\n" + REQUEST_INSTRUCTION
    )
    if retry:
        user += REQUEST_RETRY_SUFFIX
    return [{"role": "system", "content": consigne}, {"role": "user", "content": user}]


def parse_request(text: str) -> dict | None:
    """The verdict in an answer, or None. Reasoning blocks are dropped; a code fence around the object is harmless
    (the object is found inside it). The answer must hold exactly one JSON object, with exactly the keys
    `same_request` (a bool: "true" and 1 do not pass) and `reason` (a string)."""
    found = find_json_objects(strip_reasoning(text))
    if len(found) != 1 or sorted(found[0]) != sorted(REQUEST_KEYS):
        return None
    obj = found[0]
    if type(obj["same_request"]) is not bool or not isinstance(obj["reason"], str):
        return None
    return {key: obj[key] for key in REQUEST_KEYS}


def _request_one(consigne: str, pair: dict, lang: str, generate: GenerateFn) -> tuple[str, dict | None, dict]:
    """(status, verdict or None, log fields) for one pair: a second generation only after an answer that holds no
    verdict, none after a cut one (`finish_reason == "length"`) or an exception."""
    totals = {"retries": 0, "wall_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    status, found, raw = "json_error", None, ""
    for retry in (False, True):
        started = time.perf_counter()
        try:
            res = generate(build_request_messages(consigne, pair["english"], pair["translation"], lang, retry=retry))
        except Exception as exc:  # the run goes on: this pair is logged as an error
            totals["wall_s"] = round(totals["wall_s"] + (time.perf_counter() - started), 3)
            totals["retries"] = int(retry)
            return "error", None, {**totals, "raw": repr(exc)[:RAW_MAX_CHARS]}
        totals["wall_s"] += res.wall_s
        totals["prompt_tokens"] += res.prompt_tokens
        totals["completion_tokens"] += res.completion_tokens
        totals["retries"] = int(retry)
        raw = res.text
        found = parse_request(res.text)
        if found is not None:
            status = "ok"
            break
        if res.finish_reason == "length":
            status = "truncated"
            break
    totals["wall_s"] = round(totals["wall_s"], 3)
    return status, found, totals if status == "ok" else {**totals, "raw": raw[:RAW_MAX_CHARS]}


def _check_pairs(pairs: list, lang: str) -> None:
    """Raise ValueError unless every pair is {"id": "<base_key>-<lang>", "english": str, "translation": str} with a
    distinct id."""
    seen: set[str] = set()
    for number, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            raise ValueError(f"pair {number}: not a JSON object")
        for key in ("id", "english", "translation"):
            if not isinstance(pair.get(key), str):
                raise ValueError(f"pair {number}: '{key}' must be a string, got {pair.get(key)!r}")
        if not pair["id"].endswith(f"-{lang}") or pair["id"] == f"-{lang}":
            raise ValueError(f"pair {number}: id {pair['id']!r} is not <base_key>-{lang}")
        if pair["id"] in seen:
            raise ValueError(f"duplicate pair id {pair['id']!r} (pair {number})")
        seen.add(pair["id"])


def run_request(
    pairs: list[dict],
    generate: GenerateFn,
    out_dir: Path,
    lang: str,
    resume: bool = False,
    *,
    consigne: str | None = None,
) -> dict[str, int]:
    """Ask `generate` whether each translation is the same request as its English (`pairs`: {"id", "english",
    "translation"}): `out_dir/<lang>.request.jsonl` gets {"id", "same_request", "reason"} per verdict (input order at
    the end), `out_dir/<lang>.request.log.jsonl` a line per pair tried. `consigne` is the system message (default:
    the file consigne/check-same-request-v1.md). Returns the count of each status that occurred in this call.

    A model's file is immutable: without `resume`, existing files raise FileExistsError. With `resume`, ids already
    judged are skipped and the new lines are appended.
    """
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language {lang!r}: one of {', '.join(LANGUAGES)}")
    _check_pairs(pairs, lang)
    if consigne is None:
        consigne = Path(REQUEST_CONSIGNE).read_bytes().decode("utf-8")
    ids = [p["id"] for p in pairs]
    out_dir = Path(out_dir)
    out_path, log_path = out_dir / f"{lang}.request.jsonl", out_dir / f"{lang}.request.log.jsonl"
    if not resume:
        existing = [p.name for p in (out_path, log_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"{out_dir}: {', '.join(existing)} already exist; a model's verdicts are immutable. "
                "Use a new folder, or resume the run."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    done = read_done(out_path) if resume else {}
    foreign = [i for i in done if i not in set(ids)]
    if foreign:  # the wrong translations: refuse before touching anything, never drop verdicts silently
        raise ValueError(f"{out_path}: id(s) with a verdict not in the pairs: {', '.join(map(repr, foreign))}")
    if resume:
        repair_log(log_path)
    if done or out_path.exists():
        write_atomic(out_path, list(done.values()))  # drops a cut-short last line, restores the newline
    counts: dict[str, int] = {}
    with (
        open(out_path, "a", encoding="utf-8", newline="\n") as out_f,
        open(log_path, "a", encoding="utf-8", newline="\n") as log_f,
    ):
        for pair in pairs:
            if pair["id"] in done:
                continue
            status, found, fields = _request_one(consigne, pair, lang, generate)
            counts[status] = counts.get(status, 0) + 1
            log_f.write(json.dumps({"id": pair["id"], "status": status, **fields}, ensure_ascii=False) + "\n")
            log_f.flush()  # log first: a crash before the verdict line only costs a regeneration on resume
            if found is not None:
                row = {"id": pair["id"], **found}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done[pair["id"]] = row
    write_atomic(out_path, [done[i] for i in ids if i in done])
    return counts


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


def _last_statuses(log_path: Path) -> dict[str, str]:
    """id -> the status of its last line in a run log (a resumed id is logged again); a line that does not parse
    (a write cut short) is skipped, so the review still runs on a stopped run."""
    statuses: dict[str, str] = {}
    if not log_path.is_file():
        return statuses
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            statuses[row["id"]] = str(row.get("status"))
    return statuses


def _request_flags(path: Path, rows: list[dict], lang: str) -> list[dict]:
    """A `same_request: false` line of `<lang>.request.jsonl` is a `request` flag whose detail is its reason. A
    translation without a verdict line is a `request` flag too, naming its last status in `<lang>.request.log.jsonl`
    (`not run` when the log has none): a translation the model could not judge is never silent."""
    by_id = {row["id"]: row for row in rows}
    flags, judged = [], set()
    for line in read_jsonl(path):
        if line["id"] not in by_id:
            raise ValueError(f"{path}: {line['id']} is not a translation of the set")
        judged.add(line["id"])
        if line["same_request"] is False:
            flags.append(_flag(by_id[line["id"]], lang, "request", "request", line["reason"]))
    statuses = _last_statuses(path.with_name(f"{lang}.request.log.jsonl"))
    for row in rows:
        if row["id"] not in judged:
            status = statuses.get(row["id"])
            detail = f"no verdict ({status} in the log)" if status else "no verdict (not run)"
            flags.append(_flag(row, lang, "request", "request", detail))
    return flags


def _print_counts(lang: str, kind: str, flags: list[dict], total: int) -> None:
    ids = len({f["id"] for f in flags})
    print(f"{lang} {kind}: {len(flags)} flag(s) on {ids} of {total} id(s)")
    for rule in RULES:
        hits = [f for f in flags if f["rule"] == rule]
        if hits:
            lifted = sum(f["lifted"] for f in hits)
            print(f"  {rule}: {len(hits)}" + (f" ({lifted} lifted)" if lifted else ""))


def _run_request_cmd(args, fail) -> int:
    """`bench check request`: ask a local MLX model, pair by pair, whether the translation (the model's text, no
    correction applied) is the same request as its English, into `<out>/<set>/` (`<lang>.request.meta.json` first,
    then `<lang>.request.jsonl` and `<lang>.request.log.jsonl`); `--resume` goes on with a run that has the same
    model, revision, thinking, consigne, English prompts, translations and max_tokens."""
    import datetime
    import hashlib
    import statistics

    from s1d1_bench import mlx_backend  # imported here: importing check never imports mlx

    missing = [name for name in ("model", "author", "thinking") if not getattr(args, name, None)]
    if missing:
        return fail("needs " + ", ".join(f"--{name}" for name in missing) + " (the model that answers, who runs it)")
    lang = args.lang
    out = Path(args.out) / args.set
    meta_path = out / f"{lang}.request.meta.json"
    files = (meta_path, out / f"{lang}.request.jsonl", out / f"{lang}.request.log.jsonl")
    existing = [p.name for p in files if p.exists()]
    if existing and not args.resume:
        return fail(f"{out}: {', '.join(existing)} already exist; a model's verdicts are immutable. "
                    "Use a new --out, or --resume to go on with that run.")
    consigne_path = Path(args.consigne or REQUEST_CONSIGNE)
    try:  # before the model loads and before anything is written
        consigne_bytes = consigne_path.read_bytes()
        consigne = consigne_bytes.decode("utf-8")
        english_sha256 = hashlib.sha256(Path(args.english).read_bytes()).hexdigest()
        translations_sha256 = hashlib.sha256((Path(args.translations) / f"{lang}.jsonl").read_bytes()).hexdigest()
        english = _english_texts(Path(args.english))
        # the model's text, not the corrected one, in the order of its file
        rows = read_translations([Path(args.translations)], [lang], corrections=False)[lang]
        absent = [f"{row['base_key']}-eng" for row in rows if row["base_key"] not in english]
        if absent:
            raise ValueError(f"{args.english}: no English prompt {', '.join(absent)}")
        pairs = [{"id": row["id"], "english": english[row["base_key"]], "translation": row["prompt"]} for row in rows]
        _check_pairs(pairs, lang)
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    consigne_sha256 = hashlib.sha256(consigne_bytes).hexdigest()
    old = json.loads(meta_path.read_text(encoding="utf-8")) if args.resume and meta_path.is_file() else None
    if old is not None:  # what can be checked before loading the model (describe_model reads no weights)
        try:
            _, _, described = mlx_backend.describe_model(args.model, args.thinking)
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        early = {"model": described["model"], "revision": described["revision"], "thinking": args.thinking,
                 "consigne_sha256": consigne_sha256, "english_prompts_sha256": english_sha256,
                 "translations_sha256": translations_sha256, "max_tokens": args.max_tokens}
        conflicts = resume_conflicts(old, early, REQUEST_RESUME_KEYS)
        if conflicts:
            return fail(f"{meta_path}: cannot resume a run with other settings: {'; '.join(conflicts)}")

    started = time.perf_counter()
    try:
        generate, backend = mlx_backend.make_mlx_generate(args.model, args.thinking, max_tokens=args.max_tokens)
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as exc:
        return fail(str(exc))
    print(f"model loaded in {time.perf_counter() - started:.1f} s")
    replayable = bool(_COMMIT.fullmatch(backend["revision"]))
    notes = [backend["behaviour"]]
    if not replayable:
        notes.append(f"replayable false: revision {backend['revision']} (no Hub commit recorded for the weights).")
    if args.notes:
        notes.append(args.notes)
    meta = {
        "lang": lang,
        "model": backend["model"],
        "provider": "local",
        "revision": backend["revision"],
        "date": datetime.date.today().isoformat(),
        "temperature": backend["temperature"],
        "thinking": backend["thinking"],
        "consigne": consigne_path.name.removesuffix(".md"),
        "consigne_sha256": consigne_sha256,
        "english_prompts_sha256": english_sha256,
        "translations_sha256": translations_sha256,
        "author": args.author,
        "replayable": replayable,
        "notes": " ".join(notes),
        "quantization": backend["quantization"],
        "mlx_lm_version": backend["mlx_lm_version"],
        "max_tokens": args.max_tokens,
    }
    leaks = local_path_keys(meta)
    if leaks:
        return fail(f"{meta_path.name} would hold a local path in {', '.join(leaks)}: it is published, remove it")
    if old is not None:  # a backstop: the loaded weights must match what describe_model announced
        conflicts = resume_conflicts(old, meta, REQUEST_RESUME_KEYS)
        if conflicts:
            return fail(f"{meta_path}: cannot resume a run with other settings: {'; '.join(conflicts)}")
    else:  # written before the first generation, so that a resumed run keeps it
        out.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    rates: list[float] = []

    def timed(messages: list[dict]) -> GenerateResult:
        res = generate(messages)
        if res.wall_s > 0:
            rates.append(res.completion_tokens / res.wall_s)
        return res

    try:
        counts = run_request(pairs, timed, out, lang, resume=args.resume, consigne=consigne)
    except (FileExistsError, ValueError) as exc:
        return fail(str(exc))
    print(", ".join(f"{status} {n}" for status, n in sorted(counts.items())) or "nothing left to check")
    if rates:
        median = statistics.median(rates)
        print(f"median {median:.1f} tokens/s (completion tokens / wall time, {len(rates)} generations)")
    return 0


def run_check(args) -> int:
    """`bench check <kind>`: `structure` and `labels` rewrite `<out>/<set>/<lang>.<kind>.jsonl` (derivations of the
    files they read); `request` runs a local model into `<lang>.request.jsonl` (immutable, resumable); `review`
    writes `<lang>.review.md` from the check files present, and never replaces a sheet whose text differs from the
    new one (it may have been filled in)."""

    def fail(message: str, code: int = 1) -> int:
        print(f"bench check {args.kind}: {message}", file=sys.stderr)
        return code

    if args.kind == "request":
        return _run_request_cmd(args, fail)
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
