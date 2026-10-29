import datetime
import hashlib
import json
from pathlib import Path

import pytest

from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_runner import GenerateResult
from s1d1_bench.translate import INSTRUCTION, build_messages, parse_translation, run_translate

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_3 = [
    {"id": "k1-eng", "prompt": "Hello"},
    {"id": "k2-eng", "prompt": "Write a function that reverses a string."},
    {"id": "k3-eng", "prompt": "Design a migration plan."},
]


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch):
    """The default consigne, consigne/translate-v1-<lang>.md, is relative to the working directory, as in the CLI."""
    monkeypatch.chdir(REPO_ROOT)


def result(text, finish_reason="stop", prompt_tokens=10, completion_tokens=5, wall_s=0.5):
    return GenerateResult(text, prompt_tokens, completion_tokens, finish_reason, wall_s)


def fake_generate(answers):
    """A GenerateFn that pops `answers` (a str, a GenerateResult or an exception), counts and records its calls."""
    queue = list(answers)

    def generate(messages):
        generate.calls += 1
        generate.messages.append(messages)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, GenerateResult) else result(answer)

    generate.calls = 0
    generate.messages = []
    return generate


def consigne_text(lang):
    return (REPO_ROOT / f"consigne/translate-v1-{lang}.md").read_bytes().decode("utf-8")


# --- build_messages and parse_translation


def test_messages_put_the_consigne_in_system_and_the_prompt_as_data():
    m = build_messages("CONSIGNE", "Hello", "fra")
    assert m[0] == {"role": "system", "content": "CONSIGNE"}
    assert m[1]["content"] == "<prompt>\nHello\n</prompt>\n\n" + INSTRUCTION.format(language="French")


def test_retry_message_appends_the_suffix_and_keeps_the_system_message():
    first, retry = build_messages("C", "Hello", "zho"), build_messages("C", "Hello", "zho", retry=True)
    assert retry[0] == first[0] and len(retry) == 2
    assert retry[1]["content"] == first[1]["content"] + "\nOutput the translation only, nothing else."
    assert "Simplified Chinese" in first[1]["content"]


def test_parse_keeps_fences_and_drops_a_prefilled_think_close():
    assert parse_translation("</think>\n\n```cpp\nint a;\n```") == ("ok", "```cpp\nint a;\n```")


def test_parse_drops_a_closed_think_block():
    assert parse_translation("<think>hmm</think>\nBonjour") == ("ok", "Bonjour")


@pytest.mark.parametrize("text,status", [("", "empty"), ("<think>…", "empty"), ("ok\n</prompt>", "envelope"),
                                         ("<prompt id='x'>\ny", "envelope")])
def test_parse_statuses(text, status):
    assert parse_translation(text) == (status, None)


@pytest.mark.parametrize("text,status", [("  \n ", "empty"), ("x\n  </PROMPT>  ", "envelope"),
                                         ("  <Prompt>\nx", "envelope")])
def test_parse_statuses_ignore_case_and_blanks(text, status):
    assert parse_translation(text) == (status, None)


def test_parse_keeps_a_prompt_tag_inside_a_line():
    assert parse_translation("Use the <prompt> tag.") == ("ok", "Use the <prompt> tag.")


# --- run_translate


def test_run_translate_writes_ids_in_the_target_language(tmp_path):
    counts = run_translate([{"id": "k1-eng", "prompt": "Hi"}], fake_generate(["Salut"]), tmp_path, "fra")
    assert counts == {"ok": 1}
    assert read_jsonl(tmp_path / "fra.jsonl") == [{"id": "k1-fra", "base_key": "k1", "prompt": "Salut"}]


def test_run_translate_sends_the_consigne_of_the_language_byte_exact(tmp_path):
    generate = fake_generate(["Привет"])
    run_translate([{"id": "k1-eng", "prompt": "Hi"}], generate, tmp_path, "rus")
    assert generate.messages == [build_messages(consigne_text("rus"), "Hi", "rus")]
    custom = fake_generate(["Привет"])
    run_translate([{"id": "k1-eng", "prompt": "Hi"}], custom, tmp_path / "c", "rus", consigne="MINE")
    assert custom.messages[0][0] == {"role": "system", "content": "MINE"}
    with pytest.raises(TypeError):  # keyword-only: never taken for `resume`
        run_translate([{"id": "k1-eng", "prompt": "Hi"}], custom, tmp_path / "d", "rus", False, "MINE")


def test_run_translate_refuses_an_id_that_is_not_english(tmp_path):
    with pytest.raises(ValueError, match="-eng"):
        run_translate([{"id": "k1-fra", "prompt": "Hi"}], fake_generate(["x"]), tmp_path, "rus")


def test_run_translate_refuses_an_unknown_language_or_duplicate_ids(tmp_path):
    generate = fake_generate(["x", "x"])
    with pytest.raises(ValueError, match="deu"):
        run_translate(PROMPTS_3[:1], generate, tmp_path, "deu")
    with pytest.raises(ValueError, match="k1-eng"):
        run_translate([PROMPTS_3[0], PROMPTS_3[0]], generate, tmp_path, "fra")
    assert generate.calls == 0 and not (tmp_path / "fra.jsonl").exists()


def test_code_fences_are_written_as_the_model_wrote_them(tmp_path):
    answer = "Voici :\n\n```cpp\nint a; // un entier\n```"
    run_translate(PROMPTS_3[:1], fake_generate([answer]), tmp_path, "fra")
    assert read_jsonl(tmp_path / "fra.jsonl")[0]["prompt"] == answer


def test_log_lines_carry_the_documented_fields(tmp_path):
    run_translate(PROMPTS_3[:2], fake_generate(["Bonjour", "", ""]), tmp_path, "fra")
    ok, empty = read_jsonl(tmp_path / "fra.log.jsonl")
    assert ok == {"id": "k1-fra", "status": "ok", "retries": 0, "wall_s": 0.5, "prompt_tokens": 10,
                  "completion_tokens": 5}
    assert empty == {"id": "k2-fra", "status": "empty", "retries": 1, "wall_s": 1.0, "prompt_tokens": 20,
                     "completion_tokens": 10, "raw": ""}
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.jsonl")] == ["k1-fra"]


def test_envelope_answer_is_retried_once_then_logged_never_written(tmp_path):
    generate = fake_generate(["Bonjour\n</prompt>", "<prompt>\nBonjour"])
    counts = run_translate(PROMPTS_3[:1], generate, tmp_path, "fra")
    assert counts == {"envelope": 1} and generate.calls == 2
    first, retry = generate.messages
    assert retry == build_messages(consigne_text("fra"), "Hello", "fra", retry=True) and first != retry
    (line,) = read_jsonl(tmp_path / "fra.log.jsonl")
    assert line["status"] == "envelope" and line["retries"] == 1 and line["raw"] == "<prompt>\nBonjour"
    assert read_jsonl(tmp_path / "fra.jsonl") == []


@pytest.mark.parametrize("first", ["", "<think>still", "x\n</prompt>"])
def test_an_empty_or_envelope_answer_whose_retry_is_good_is_ok(tmp_path, first):
    generate = fake_generate([first, "Bonjour"])
    assert run_translate(PROMPTS_3[:1], generate, tmp_path, "fra") == {"ok": 1}
    (line,) = read_jsonl(tmp_path / "fra.log.jsonl")
    assert line["retries"] == 1 and "raw" not in line
    assert read_jsonl(tmp_path / "fra.jsonl")[0]["prompt"] == "Bonjour"


def test_truncated_is_not_retried(tmp_path):
    generate = fake_generate([result("Bonjour, voici la tra", finish_reason="length")])
    counts = run_translate(PROMPTS_3[:1], generate, tmp_path, "fra")
    assert counts == {"truncated": 1} and generate.calls == 1  # a cut translation is never written, even if it reads
    (line,) = read_jsonl(tmp_path / "fra.log.jsonl")
    assert line["status"] == "truncated" and line["retries"] == 0 and line["raw"] == "Bonjour, voici la tra"
    assert read_jsonl(tmp_path / "fra.jsonl") == []


def test_a_retry_cut_by_length_is_truncated(tmp_path):
    generate = fake_generate(["", result("cut", finish_reason="length")])
    assert run_translate(PROMPTS_3[:1], generate, tmp_path, "fra") == {"truncated": 1}
    assert read_jsonl(tmp_path / "fra.log.jsonl")[0]["retries"] == 1


def test_generate_raising_is_status_error_without_retry_and_the_run_continues(tmp_path):
    generate = fake_generate([RuntimeError("boom"), "Bonjour"])
    assert run_translate(PROMPTS_3[:2], generate, tmp_path, "fra") == {"error": 1, "ok": 1}
    first, second = read_jsonl(tmp_path / "fra.log.jsonl")
    assert first["status"] == "error" and first["retries"] == 0 and "boom" in first["raw"]
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.jsonl")] == ["k2-fra"]


def test_raw_is_truncated_to_2000_chars(tmp_path):
    run_translate(PROMPTS_3[:1], fake_generate([result("x" * 5000, finish_reason="length")]), tmp_path, "fra")
    assert read_jsonl(tmp_path / "fra.log.jsonl")[0]["raw"] == "x" * 2000


# --- immutability and resume


@pytest.mark.parametrize("name", ["fra.jsonl", "fra.log.jsonl"])
def test_existing_file_without_resume_raises(tmp_path, name):
    (tmp_path / name).write_text("", encoding="utf-8")
    generate = fake_generate(["Bonjour"])
    with pytest.raises(FileExistsError, match=name):
        run_translate(PROMPTS_3[:1], generate, tmp_path, "fra")
    assert generate.calls == 0 and (tmp_path / name).read_text(encoding="utf-8") == ""


def test_another_language_in_the_same_folder_is_not_an_existing_file(tmp_path):
    run_translate(PROMPTS_3[:1], fake_generate(["Bonjour"]), tmp_path, "fra")
    assert run_translate(PROMPTS_3[:1], fake_generate(["Привет"]), tmp_path, "rus") == {"ok": 1}


def test_resume_skips_done_ids_and_keeps_prompt_order(tmp_path):
    run_translate([PROMPTS_3[2]], fake_generate(["T3"]), tmp_path, "fra")  # k3 done first
    generate = fake_generate(["T1", "", ""])  # k1 ok, k2 empty twice
    assert run_translate(PROMPTS_3, generate, tmp_path, "fra", resume=True) == {"ok": 1, "empty": 1}
    assert generate.calls == 3
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.jsonl")] == ["k1-fra", "k3-fra"]
    generate = fake_generate(["T2"])  # only the failed id is tried again
    assert run_translate(PROMPTS_3, generate, tmp_path, "fra", resume=True) == {"ok": 1} and generate.calls == 1
    assert [r["prompt"] for r in read_jsonl(tmp_path / "fra.jsonl")] == ["T1", "T2", "T3"]
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.log.jsonl")] == ["k3-fra", "k1-fra", "k2-fra", "k2-fra"]


def test_resume_drops_cut_short_last_lines(tmp_path):
    run_translate(PROMPTS_3[:1], fake_generate(["T1"]), tmp_path, "fra")
    for name in ("fra.jsonl", "fra.log.jsonl"):
        with open(tmp_path / name, "a", encoding="utf-8") as f:
            f.write('{"id": "k2-fra", "ba')  # killed mid-write
    generate = fake_generate(["T2", "T3"])
    assert run_translate(PROMPTS_3, generate, tmp_path, "fra", resume=True) == {"ok": 2} and generate.calls == 2
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.jsonl")] == ["k1-fra", "k2-fra", "k3-fra"]
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.log.jsonl")] == ["k1-fra", "k2-fra", "k3-fra"]


def test_resume_with_foreign_ids_raises_and_touches_nothing(tmp_path):
    run_translate(PROMPTS_3, fake_generate(["T1", "T2", "T3"]), tmp_path, "fra")
    before = {n: (tmp_path / n).read_bytes() for n in ("fra.jsonl", "fra.log.jsonl")}
    generate = fake_generate(["x"])
    with pytest.raises(ValueError, match="k2-fra.*k3-fra"):
        run_translate([PROMPTS_3[0], {"id": "k4-eng", "prompt": "y"}], generate, tmp_path, "fra", resume=True)
    assert generate.calls == 0 and {n: (tmp_path / n).read_bytes() for n in before} == before


def test_resume_on_an_empty_folder_just_runs(tmp_path):
    assert run_translate(PROMPTS_3[:1], fake_generate(["T1"]), tmp_path / "new", "fra", resume=True) == {"ok": 1}


# --- bench translate (run_translate_cmd), with the mlx backend replaced by a fake

BACKEND_META = {
    "model": "org/model-8bit",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "quantization": "8bit-gs64",
    "mlx_lm_version": "0.32.2",
    "temperature": 0,
    "thinking": "off",
    "behaviour": "fake: no reasoning.",
}


def prompts_file(tmp_path, prompts=PROMPTS_3):
    path = tmp_path / "set.jsonl"
    write_jsonl(path, prompts)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def fake_backend(monkeypatch, answers, meta=None, described=None):
    """Replace make_mlx_generate and describe_model (`described`: what describe_model says, default `meta`); return
    the fake GenerateFn (its .calls, .messages) and the recorded arguments (empty while the model was never loaded)."""
    from s1d1_bench import mlx_backend

    generate = fake_generate(answers)
    seen = {}

    def make(model_id, thinking, max_tokens=1024):
        seen.update(model_id=model_id, thinking=thinking, max_tokens=max_tokens)
        return generate, dict(meta or BACKEND_META, model=model_id, thinking=thinking)

    def describe(model_id, thinking):
        fields = dict(described or meta or BACKEND_META, model=model_id, thinking=thinking)
        fields.pop("mlx_lm_version")  # known only once mlx_lm is imported
        return Path("unused"), None, fields

    monkeypatch.setattr(mlx_backend, "make_mlx_generate", make)
    monkeypatch.setattr(mlx_backend, "describe_model", describe)
    return generate, seen


def translate_cmd(*argv):
    from s1d1_bench.cli import build_parser

    args = build_parser().parse_args(["translate", *map(str, argv)])
    return args.handler(args)


def test_cmd_writes_the_documented_meta(tmp_path, monkeypatch):
    prompts, digest = prompts_file(tmp_path)
    generate, seen = fake_backend(monkeypatch, ["T1", "T2", "T3"])
    out = tmp_path / "translations"
    code = translate_cmd("--model", "org/model-8bit", "--prompts", prompts, "--lang", "zho", "--out", out,
                         "--author", "steve", "--thinking", "off", "--max-tokens", 900)
    assert code == 0
    assert seen == {"model_id": "org/model-8bit", "thinking": "off", "max_tokens": 900}
    assert generate.messages[0][0]["content"] == consigne_text("zho")
    meta = json.loads((out / "zho.meta.json").read_text(encoding="utf-8"))
    assert meta == {
        "lang": "zho",
        "language": "Simplified Chinese",
        "model": "org/model-8bit",
        "provider": "local",
        "revision": BACKEND_META["revision"],
        "date": datetime.date.today().isoformat(),
        "temperature": 0,
        "thinking": "off",
        "consigne": "translate-v1-zho",
        "consigne_sha256": hashlib.sha256((REPO_ROOT / "consigne/translate-v1-zho.md").read_bytes()).hexdigest(),
        "source_prompts_sha256": digest,
        "author": "steve",
        "replayable": True,
        "translator": f"org/model-8bit@{BACKEND_META['revision']}",
        "notes": "fake: no reasoning.",
        "quantization": "8bit-gs64",
        "mlx_lm_version": "0.32.2",
        "max_tokens": 900,
    }
    assert list(meta) == list(json.loads((out / "zho.meta.json").read_text(encoding="utf-8")))
    assert [r["id"] for r in read_jsonl(out / "zho.jsonl")] == ["k1-zho", "k2-zho", "k3-zho"]


def test_cmd_writes_meta_before_generating_and_refuses_another_consigne_on_resume(tmp_path, monkeypatch, capsys):
    prompts, _ = prompts_file(tmp_path)
    out = tmp_path / "t"
    seen_meta = []

    def answer(messages):
        seen_meta.append((out / "fra.meta.json").is_file())
        return result("" if len(seen_meta) > 1 else "T1")  # k1 ok, k2 empty twice, then the run is "killed"

    from s1d1_bench import mlx_backend

    monkeypatch.setattr(mlx_backend, "make_mlx_generate", lambda m, t, max_tokens=1024: (answer, dict(BACKEND_META)))
    base = ["--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", out, "--author", "a",
            "--thinking", "off"]
    assert translate_cmd(*base) == 0
    assert seen_meta and all(seen_meta)
    meta_before = (out / "fra.meta.json").read_bytes()
    other = tmp_path / "translate-v2-fra.md"
    other.write_text(consigne_text("fra") + "\nOne more rule.\n", encoding="utf-8")
    generate, seen = fake_backend(monkeypatch, ["T2", "T3"])
    code = translate_cmd(*base, "--consigne", other, "--resume")
    err = capsys.readouterr().err
    assert code == 1 and generate.calls == 0 and seen == {}  # refused before the model loads
    assert err.startswith("bench translate:") and "consigne_sha256" in err and "fra.meta.json" in err
    assert (out / "fra.meta.json").read_bytes() == meta_before


def test_cmd_resume_with_the_same_settings_goes_on_and_keeps_the_meta(tmp_path, monkeypatch):
    prompts, _ = prompts_file(tmp_path)
    out = tmp_path / "t"
    base = ["--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", out, "--author", "a",
            "--thinking", "off"]
    fake_backend(monkeypatch, ["T1", "", "", "", ""])
    translate_cmd(*base)
    meta_before = (out / "fra.meta.json").read_bytes()
    generate, _ = fake_backend(monkeypatch, ["T2", "T3"])
    assert translate_cmd(*base, "--resume") == 0 and generate.calls == 2
    assert (out / "fra.meta.json").read_bytes() == meta_before
    assert [r["prompt"] for r in read_jsonl(out / "fra.jsonl")] == ["T1", "T2", "T3"]


OTHER_REVISION = {**BACKEND_META, "revision": "f" * 40}


@pytest.mark.parametrize(
    ("argv", "meta", "described", "key", "loaded"),
    [
        (["--model", "org/m", "--thinking", "cap:512"], None, None, "thinking", False),
        (["--model", "org/m", "--thinking", "off", "--max-tokens", "512"], None, None, "max_tokens", False),
        (["--model", "org/m", "--thinking", "off", "--prompts", "{other}"], None, None, "source_prompts_sha256", False),
        (["--model", "org/other", "--thinking", "off"], None, None, "model", False),
        (["--model", "org/m", "--thinking", "off"], OTHER_REVISION, None, "revision", False),
        # describe_model announced the recorded revision, the loaded weights carry another: the backstop refuses
        (["--model", "org/m", "--thinking", "off"], OTHER_REVISION, BACKEND_META, "revision", True),
    ],
)
def test_cmd_resume_refuses_other_settings(tmp_path, monkeypatch, capsys, argv, meta, described, key, loaded):
    prompts, _ = prompts_file(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text(prompts.read_text(encoding="utf-8") + "\n", encoding="utf-8")  # same prompts, other bytes
    out = tmp_path / "t"
    fake_backend(monkeypatch, ["T1", "", "", "", ""])
    translate_cmd("--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", out, "--author", "a",
                  "--thinking", "off")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    generate, seen = fake_backend(monkeypatch, ["T2", "T3"], meta=meta, described=described)
    argv = [a.format(other=other) for a in argv]
    if "--prompts" not in argv:
        argv += ["--prompts", str(prompts)]
    code = translate_cmd(*argv, "--lang", "fra", "--out", out, "--author", "a", "--resume")
    err = capsys.readouterr().err
    assert code == 1 and generate.calls == 0 and bool(seen) is loaded  # what describe_model knows: checked unloaded
    assert key in err and "fra.meta.json" in err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_cmd_resume_of_a_model_describe_model_cannot_find_is_one_line_unloaded(tmp_path, monkeypatch, capsys):
    from s1d1_bench import mlx_backend

    prompts, _ = prompts_file(tmp_path)
    out = tmp_path / "t"
    base = ["--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", out, "--author", "a",
            "--thinking", "off"]
    fake_backend(monkeypatch, ["T1", "", "", "", ""])
    translate_cmd(*base)
    generate, seen = fake_backend(monkeypatch, ["T2", "T3"])

    def missing(model_id, thinking):
        raise FileNotFoundError(f"{model_id}: not found")

    monkeypatch.setattr(mlx_backend, "describe_model", missing)
    assert translate_cmd(*base, "--resume") == 1 and seen == {} and generate.calls == 0
    assert capsys.readouterr().err == "bench translate: org/m: not found\n"


def test_cmd_refuses_existing_files_without_resume(tmp_path, monkeypatch, capsys):
    prompts, _ = prompts_file(tmp_path)
    generate, seen = fake_backend(monkeypatch, ["T1"] * 3)
    out = tmp_path / "t"
    out.mkdir()
    (out / "fra.meta.json").write_text("{}")
    code = translate_cmd("--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", out, "--author", "a",
                         "--thinking", "off")
    assert code == 1 and generate.calls == 0 and seen == {}
    assert (out / "fra.meta.json").read_text() == "{}" and "--resume" in capsys.readouterr().err


@pytest.mark.parametrize("notes", ["see /Users/steve/mlx-models/x", "from ~/mlx-models/x", "in {home}/x"])
def test_cmd_refuses_a_local_path_in_notes(tmp_path, monkeypatch, capsys, notes):
    prompts, _ = prompts_file(tmp_path)
    generate, _ = fake_backend(monkeypatch, ["T1"] * 3)
    out = tmp_path / "t"
    code = translate_cmd("--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", out, "--author", "a",
                         "--thinking", "off", "--notes", notes.format(home=Path.home()))
    assert code == 1 and generate.calls == 0 and not out.exists()
    assert "local path" in capsys.readouterr().err


def test_cmd_refuses_prompts_that_are_not_english_before_loading(tmp_path, monkeypatch, capsys):
    prompts, _ = prompts_file(tmp_path, [{"id": "k1-fra", "prompt": "Salut"}])
    generate, seen = fake_backend(monkeypatch, ["x"])
    code = translate_cmd("--model", "org/m", "--prompts", prompts, "--lang", "rus", "--out", tmp_path / "t",
                         "--author", "a", "--thinking", "off")
    err = capsys.readouterr().err
    assert code == 1 and seen == {} and not (tmp_path / "t").exists()
    assert err.startswith("bench translate:") and "-eng" in err and "Traceback" not in err


def test_cmd_unknown_revision_is_not_replayable_and_says_why(tmp_path, monkeypatch):
    prompts, _ = prompts_file(tmp_path)
    fake_backend(monkeypatch, ["T1"] * 3, meta={**BACKEND_META, "revision": "unknown"})
    out = tmp_path / "t"
    translate_cmd("--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", out, "--author", "a",
                  "--thinking", "off", "--notes", "mine")
    meta = json.loads((out / "fra.meta.json").read_text(encoding="utf-8"))
    assert meta["replayable"] is False and meta["translator"] == "org/m@unknown"
    assert meta["notes"].startswith("fake: no reasoning.") and "revision unknown" in meta["notes"]
    assert meta["notes"].endswith("mine")


def test_cmd_prints_counts_and_median_tokens_per_second(tmp_path, monkeypatch, capsys):
    prompts, _ = prompts_file(tmp_path)
    fake_backend(monkeypatch, ["T1", "", "", result("T3", completion_tokens=30, wall_s=1.0)])
    translate_cmd("--model", "org/m", "--prompts", prompts, "--lang", "fra", "--out", tmp_path / "t",
                  "--author", "a", "--thinking", "off")
    out = capsys.readouterr().out
    assert "ok 2" in out and "empty 1" in out
    assert "median 10.0 tokens/s" in out  # per generation: 10, 10, 10, 30


def test_cli_lists_translate(capsys):
    from s1d1_bench.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    assert "translate" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["translate", "--help"])
    out = capsys.readouterr().out
    assert "--lang" in out and "fra,rus,zho" in out and "consigne/translate-v1-<lang>.md" in out
    assert "4096" in out
