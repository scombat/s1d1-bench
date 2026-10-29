"""`bench translate`: translate an English prompts file into one language with a local model, one prompt per call,
the consigne as system message and the prompt as data between <prompt> tags. Writes `<lang>.jsonl` (translations
only, prompt order at the end) and `<lang>.log.jsonl` (every prompt tried) line by line, resumes like `bench judge`."""

import json
import re
import time
from pathlib import Path

from s1d1_bench.judge_runner import GenerateFn, GenerateResult, local_path_keys, resume_conflicts
from s1d1_bench.runfile import read_done, repair_log, strip_reasoning, write_atomic

LANGUAGES = {"fra": "French", "rus": "Russian", "zho": "Simplified Chinese"}
INSTRUCTION = "Translate the prompt between the <prompt> tags into {language}. Return the translation only."
RETRY_SUFFIX = "\nOutput the translation only, nothing else."
RAW_MAX_CHARS = 2000
SOURCE_SUFFIX = "-eng"
DEFAULT_CONSIGNE = "consigne/translate-v1-{lang}.md"  # relative to the working directory, as the CLI defaults

# <lang>.meta.json keys a resumed run must share with the run it resumes
RESUME_KEYS = ["model", "revision", "thinking", "consigne_sha256", "source_prompts_sha256", "max_tokens"]
_COMMIT = re.compile(r"[0-9a-f]{40}")


def read_consigne(path: Path | str) -> str:
    """The consigne file, byte for byte (no newline translation): it is the system message."""
    return Path(path).read_bytes().decode("utf-8")


def build_messages(consigne: str, prompt: str, lang: str, retry: bool = False) -> list[dict]:
    """System message = the consigne, byte-exact; user message = the prompt as data between <prompt> tags."""
    user = "<prompt>\n" + prompt + "\n</prompt>\n\n" + INSTRUCTION.format(language=LANGUAGES[lang])
    if retry:
        user += RETRY_SUFFIX
    return [{"role": "system", "content": consigne}, {"role": "user", "content": user}]


def parse_translation(text: str) -> tuple[str, str | None]:
    """("ok", translation) once the reasoning is dropped (code fences stay); ("empty", None) if nothing is left;
    ("envelope", None) if a line could pass for an envelope line (`</prompt>`, or one starting with `<prompt`), the
    rule of `export.check_envelope_safe`: such a text is never written."""
    text = strip_reasoning(text).strip()
    if not text:
        return "empty", None
    if any((line := raw.strip().lower()) == "</prompt>" or line.startswith("<prompt") for raw in text.splitlines()):
        return "envelope", None
    return "ok", text


def _translate_one(consigne: str, prompt: str, lang: str, generate: GenerateFn) -> tuple[str, str | None, dict]:
    """(status, translation or None, log fields) for one prompt: a second generation only after an empty or an
    envelope answer, none after a cut one (`finish_reason == "length"`) or an exception."""
    totals = {"retries": 0, "wall_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    status, translation, raw = "empty", None, ""
    for retry in (False, True):
        started = time.perf_counter()
        try:
            res = generate(build_messages(consigne, prompt, lang, retry=retry))
        except Exception as exc:  # the run goes on: this prompt is logged as an error
            totals["wall_s"] = round(totals["wall_s"] + (time.perf_counter() - started), 3)
            totals["retries"] = int(retry)
            return "error", None, {**totals, "raw": repr(exc)[:RAW_MAX_CHARS]}
        totals["wall_s"] += res.wall_s
        totals["prompt_tokens"] += res.prompt_tokens
        totals["completion_tokens"] += res.completion_tokens
        totals["retries"] = int(retry)
        raw = res.text
        if res.finish_reason == "length":  # a cut translation reads as a whole one: never kept, never retried
            status, translation = "truncated", None
            break
        status, translation = parse_translation(res.text)
        if status == "ok":
            break
    totals["wall_s"] = round(totals["wall_s"], 3)
    return status, translation, totals if status == "ok" else {**totals, "raw": raw[:RAW_MAX_CHARS]}


def check_sources(prompts: list) -> None:
    """Raise ValueError unless every row is {"id": "<base_key>-eng", "prompt": str} with a distinct id."""
    seen: set[str] = set()
    for number, row in enumerate(prompts, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"prompt {number}: not a JSON object")
        for key in ("id", "prompt"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"prompt {number}: '{key}' must be a string, got {row.get(key)!r}")
        if not row["id"].endswith(SOURCE_SUFFIX) or row["id"] == SOURCE_SUFFIX:
            raise ValueError(f"prompt {number}: id {row['id']!r} does not end with {SOURCE_SUFFIX} (not English)")
        if row["id"] in seen:
            raise ValueError(f"duplicate prompt id {row['id']!r} (prompt {number})")
        seen.add(row["id"])


def run_translate(
    prompts: list[dict],
    generate: GenerateFn,
    out_dir: Path,
    lang: str,
    resume: bool = False,
    *,
    consigne: str | None = None,
) -> dict[str, int]:
    """Translate `prompts` ({"id": "<base_key>-eng", "prompt"}) into `lang`: `out_dir/<lang>.jsonl` gets
    {"id": "<base_key>-<lang>", "base_key", "prompt"} for each ok translation (prompt order at the end),
    `out_dir/<lang>.log.jsonl` a line per prompt tried. `consigne` is the system message (default: the file
    consigne/translate-v1-<lang>.md). Returns the count of each status that occurred in this call.

    A model's file is immutable: without `resume`, existing files raise FileExistsError. With `resume`, ids already
    translated are skipped and the new lines are appended.
    """
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language {lang!r}: one of {', '.join(LANGUAGES)}")
    check_sources(prompts)
    if consigne is None:
        consigne = read_consigne(DEFAULT_CONSIGNE.format(lang=lang))
    base_keys = [p["id"][: -len(SOURCE_SUFFIX)] for p in prompts]
    ids = [f"{base_key}-{lang}" for base_key in base_keys]
    out_dir = Path(out_dir)
    out_path, log_path = out_dir / f"{lang}.jsonl", out_dir / f"{lang}.log.jsonl"
    if not resume:
        existing = [p.name for p in (out_path, log_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"{out_dir}: {', '.join(existing)} already exist; a model's translations are immutable. "
                "Use a new folder, or resume the run."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    done = read_done(out_path) if resume else {}
    foreign = [i for i in done if i not in set(ids)]
    if foreign:  # the wrong prompts file: refuse before touching anything, never drop translations silently
        raise ValueError(f"{out_path}: translated id(s) not in the prompts: {', '.join(map(repr, foreign))}")
    if resume:
        repair_log(log_path)
    if done or out_path.exists():
        write_atomic(out_path, list(done.values()))  # drops a cut-short last line, restores the newline
    counts: dict[str, int] = {}
    with (
        open(out_path, "a", encoding="utf-8", newline="\n") as out_f,
        open(log_path, "a", encoding="utf-8", newline="\n") as log_f,
    ):
        for item, base_key, out_id in zip(prompts, base_keys, ids, strict=True):
            if out_id in done:
                continue
            status, translation, fields = _translate_one(consigne, item["prompt"], lang, generate)
            counts[status] = counts.get(status, 0) + 1
            log_f.write(json.dumps({"id": out_id, "status": status, **fields}, ensure_ascii=False) + "\n")
            log_f.flush()  # log first: a crash before the translation line only costs a regeneration on resume
            if translation is not None:
                row = {"id": out_id, "base_key": base_key, "prompt": translation}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done[out_id] = row
    write_atomic(out_path, [done[i] for i in ids if i in done])
    return counts


def run_translate_cmd(args) -> int:
    """`bench translate`: translate an English prompts file into `--lang` with a local MLX model into `--out`
    (`<lang>.meta.json` first, then `<lang>.jsonl` and `<lang>.log.jsonl`); `--resume` goes on with a run that has
    the same model, revision, thinking, consigne, prompts and max_tokens."""
    import datetime
    import hashlib
    import statistics
    import sys

    from s1d1_bench import mlx_backend  # imported here: importing translate never imports mlx
    from s1d1_bench.jsonl import read_jsonl

    def fail(message: str) -> int:
        print(f"bench translate: {message}", file=sys.stderr)
        return 1

    lang = args.lang
    out = Path(args.out)
    meta_path = out / f"{lang}.meta.json"
    existing = [p.name for p in (meta_path, out / f"{lang}.jsonl", out / f"{lang}.log.jsonl") if p.exists()]
    if existing and not args.resume:
        return fail(f"{out}: {', '.join(existing)} already exist; a model's translations are immutable. "
                    "Use a new --out, or --resume to go on with that run.")
    consigne_path = Path(args.consigne or DEFAULT_CONSIGNE.format(lang=lang))
    try:
        consigne_bytes = consigne_path.read_bytes()
        consigne = consigne_bytes.decode("utf-8")
        digest = hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest()
        prompts = read_jsonl(args.prompts)
        check_sources(prompts)  # before the model loads and before anything is written
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
                 "consigne_sha256": consigne_sha256, "source_prompts_sha256": digest, "max_tokens": args.max_tokens}
        conflicts = resume_conflicts(old, early, RESUME_KEYS)
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
        "language": LANGUAGES[lang],
        "model": backend["model"],
        "provider": "local",
        "revision": backend["revision"],
        "date": datetime.date.today().isoformat(),
        "temperature": backend["temperature"],
        "thinking": backend["thinking"],
        "consigne": consigne_path.name.removesuffix(".md"),
        "consigne_sha256": consigne_sha256,
        "source_prompts_sha256": digest,
        "author": args.author,
        "replayable": replayable,
        "translator": f"{backend['model']}@{backend['revision']}",
        "notes": " ".join(notes),
        "quantization": backend["quantization"],
        "mlx_lm_version": backend["mlx_lm_version"],
        "max_tokens": args.max_tokens,
    }
    leaks = local_path_keys(meta)
    if leaks:
        return fail(f"{meta_path.name} would hold a local path in {', '.join(leaks)}: it is published, remove it")
    if old is not None:  # a backstop: the loaded weights must match what describe_model announced
        conflicts = resume_conflicts(old, meta, RESUME_KEYS)
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
        counts = run_translate(prompts, timed, out, lang, resume=args.resume, consigne=consigne)
    except (FileExistsError, ValueError) as exc:
        return fail(str(exc))
    print(", ".join(f"{status} {n}" for status, n in sorted(counts.items())) or "nothing left to translate")
    if rates:
        median = statistics.median(rates)
        print(f"median {median:.1f} tokens/s (completion tokens / wall time, {len(rates)} generations)")
    return 0
