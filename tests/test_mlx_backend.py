import json
import os
import subprocess
import sys

import pytest

from s1d1_bench import mlx_backend
from s1d1_bench.judge_runner import build_messages, parse_label
from s1d1_bench.mlx_backend import (
    FAMILIES,
    Family,
    describe_model,
    detect_family,
    generate_capped,
    make_mlx_generate,
    parse_thinking,
    quantization_label,
    read_revision,
    resolve_model,
)

SHA = "773a7da77e569019bb0fd17a554b263738d669a3"
GOOD = '{"complexity": "low", "task": "chat", "needs_tools": false}'

# --- thinking


def test_parse_thinking_off_and_cap():
    assert parse_thinking("off") is None
    assert parse_thinking("cap:512") == 512
    assert parse_thinking("cap:1") == 1


@pytest.mark.parametrize("bad", ["", "on", "OFF", "cap", "cap:", "cap:0", "cap:-3", "cap:abc", "cap:1.5", "cap: 5"])
def test_parse_thinking_refuses_anything_else(bad):
    with pytest.raises(ValueError):
        parse_thinking(bad)


# --- families


def test_the_three_families():
    assert set(FAMILIES) == {"gemma4", "qwen3", "gpt_oss"}
    for family in FAMILIES.values():
        assert family.open_marker and family.close_marker and family.inject


def test_qwen_and_gemma_switch_thinking_with_the_template_kwarg():
    for name in ("qwen3", "gemma4"):
        family = FAMILIES[name]
        assert family.template_kwargs("off") == {"enable_thinking": False}
        assert family.off_prefill == "" and family.fixed_date is None
    assert FAMILIES["gemma4"].template_kwargs("cap:512") == {"enable_thinking": True}
    # Qwen3.8 defaults to reasoning effort xhigh: low, as for Qwen in docs/05 (low256)
    assert FAMILIES["qwen3"].template_kwargs("cap:512") == {"enable_thinking": True, "reasoning_effort": "low"}


def test_template_kwargs_are_a_copy():
    FAMILIES["qwen3"].template_kwargs("off")["enable_thinking"] = True
    assert FAMILIES["qwen3"].template_kwargs("off") == {"enable_thinking": False}


def test_qwen_markers():
    qwen = FAMILIES["qwen3"]
    assert (qwen.open_marker, qwen.close_marker) == ("<think>", "</think>")
    assert qwen.cap_applicable


def test_gemma_markers():
    gemma = FAMILIES["gemma4"]
    assert (gemma.open_marker, gemma.close_marker) == ("<|channel>thought", "<channel|>")
    assert gemma.cap_applicable


def test_gpt_oss_has_no_off_switch_in_its_template_so_off_prefills_an_empty_analysis():
    gpt = FAMILIES["gpt_oss"]
    assert gpt.template_kwargs("off") == {"reasoning_effort": "low"}
    assert gpt.template_kwargs("cap:512") == {"reasoning_effort": "low"}
    assert gpt.off_prefill == "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    assert gpt.open_marker == "<|channel|>analysis<|message|>" and gpt.close_marker == "<|end|>"
    assert gpt.inject == "<|end|><|start|>assistant<|channel|>final<|message|>"
    assert gpt.fixed_date is not None and gpt.cap_applicable


def test_the_off_prefill_of_gpt_oss_is_stripped_by_parse_label():
    assert parse_label(FAMILIES["gpt_oss"].off_prefill + GOOD) is not None


@pytest.mark.parametrize(
    ("config", "family"),
    [
        ({"model_type": "gemma4"}, "gemma4"),
        ({"model_type": "gemma4_text"}, "gemma4"),
        ({"model_type": "qwen3"}, "qwen3"),
        ({"model_type": "qwen3_5"}, "qwen3"),
        ({"model_type": "gpt_oss"}, "gpt_oss"),
        ({"architectures": ["GptOssForCausalLM"]}, "gpt_oss"),
    ],
)
def test_detect_family(config, family):
    assert detect_family(config) == family


def test_detect_family_refuses_an_unknown_type():
    with pytest.raises(ValueError, match="llama"):
        detect_family({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})


# --- cap logic over a fake token stream

VOCAB = {1: "<think>", 2: "\n", 3: "</think>", 4: "\n\n", 10: "hmm ", 20: GOOD}
OPEN, CLOSE, INJECT = (1,), (3,), (3, 4)


def fake_stream(*scripts):
    """A StreamFn replaying one script (list of token ids) per call; records each call and how far it was read."""

    def stream(prompt, max_tokens):
        script = list(scripts[len(stream.calls)])
        call = {"prompt": list(prompt), "max_tokens": max_tokens, "pulled": 0}
        stream.calls.append(call)
        for token in script[:max_tokens]:
            call["pulled"] += 1
            yield token

    stream.calls = []
    return stream


def text_of(tokens):
    return "".join(VOCAB[t] for t in tokens)


def capped(stream, cap, max_tokens=50, open_in_prompt=False, prompt=(7, 8)):
    return generate_capped(stream, list(prompt), max_tokens, cap, OPEN, CLOSE, INJECT, open_in_prompt)


def test_cap_reached_injects_the_closing_marker_and_restarts_for_the_answer():
    stream = fake_stream([1, 2] + [10] * 100, [20])
    out = capped(stream, cap=5)
    assert out.capped and out.finish_reason == "stop"
    assert out.tokens == [1, 2, 10, 10, 10, 10, 3, 4, 20]
    assert out.generated == 7  # 6 tokens of reasoning (newline included) + 1 of answer, the injected ones excluded
    assert stream.calls[0]["pulled"] == 6  # the first stream is not read past the cap
    assert stream.calls[1]["prompt"] == [7, 8, 1, 2, 10, 10, 10, 10, 3, 4]
    assert stream.calls[1]["max_tokens"] == 50 - 6
    assert parse_label(text_of(out.tokens)) is not None


def test_cap_not_reached_is_a_single_generation():
    stream = fake_stream([1, 2, 10, 10, 3, 4, 20])
    out = capped(stream, cap=5)
    assert not out.capped and out.finish_reason == "stop"
    assert out.tokens == [1, 2, 10, 10, 3, 4, 20] and out.generated == 7
    assert len(stream.calls) == 1
    assert parse_label(text_of(out.tokens)) is not None


def test_no_reasoning_at_all():
    stream = fake_stream([20])
    out = capped(stream, cap=5)
    assert (out.tokens, out.generated, out.capped, out.finish_reason) == ([20], 1, False, "stop")
    assert len(stream.calls) == 1


def test_no_cap_lets_the_reasoning_run():
    stream = fake_stream([1] + [10] * 30 + [3, 20])
    out = capped(stream, cap=None)
    assert not out.capped and out.generated == 33 and len(stream.calls) == 1


def test_reasoning_opened_by_the_chat_template_counts_from_the_first_token():
    stream = fake_stream([10] * 100, [20])
    out = capped(stream, cap=3, open_in_prompt=True)
    assert out.capped and out.tokens == [10, 10, 10, 3, 4, 20]
    assert stream.calls[1]["prompt"] == [7, 8, 10, 10, 10, 3, 4]


def test_max_tokens_reached_is_length():
    stream = fake_stream([20, 20, 20, 20, 20])
    out = capped(stream, cap=None, max_tokens=4)
    assert out.finish_reason == "length" and out.generated == 4


def test_cap_that_uses_the_whole_budget_does_not_restart():
    stream = fake_stream([1] + [10] * 100)
    out = capped(stream, cap=10, max_tokens=11)
    assert out.capped and out.finish_reason == "length" and out.generated == 11
    assert len(stream.calls) == 1


def test_answer_cut_by_max_tokens_after_the_cap_is_length():
    stream = fake_stream([1] + [10] * 100, [20] * 100)
    out = capped(stream, cap=4, max_tokens=8)
    assert out.capped and out.finish_reason == "length" and out.generated == 8
    assert stream.calls[1]["max_tokens"] == 3


def test_a_closing_marker_without_an_opening_one_is_not_reasoning():
    stream = fake_stream([3, 10, 10, 10, 10, 10, 20])
    out = capped(stream, cap=2)
    assert not out.capped and out.generated == 7


def test_multi_token_markers():
    stream = fake_stream([5, 6] + [10] * 20, [20])
    out = generate_capped(stream, [7], 40, 3, (5, 6), (8, 9), (8, 9), False)
    assert out.capped and out.tokens == [5, 6, 10, 10, 10, 8, 9, 20]


# --- local model directory, revision, quantization


def test_missing_model_dir_tells_the_download_command(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        resolve_model("mlx-community/some-model-8bit", root=tmp_path)
    message = str(info.value)
    target = tmp_path / "mlx-community--some-model-8bit"
    assert f"hf download mlx-community/some-model-8bit --local-dir {target}" in message
    assert "MODELS.md" in message


def test_hub_id_resolves_under_the_models_root(tmp_path):
    (tmp_path / "org--name").mkdir()
    assert resolve_model("org/name", root=tmp_path) == (tmp_path / "org--name", "org/name")


def test_a_directory_under_the_models_root_is_recorded_as_its_hub_id(tmp_path):
    (tmp_path / "org--name").mkdir()
    assert resolve_model(str(tmp_path / "org--name"), root=tmp_path) == (tmp_path / "org--name", "org/name")
    assert resolve_model(str(tmp_path / "org--name") + "/", root=tmp_path)[1] == "org/name"


def test_a_directory_elsewhere_is_refused(tmp_path):
    local = tmp_path / "converted"
    local.mkdir()
    with pytest.raises(ValueError, match="mlx-models"):
        resolve_model(str(local), root=tmp_path / "models")


@pytest.mark.parametrize("name", ["no-separator", "org--name/sub"])
def test_a_directory_under_the_root_without_an_org_name_layout_is_refused(tmp_path, name):
    (tmp_path / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="org--name"):
        resolve_model(str(tmp_path / name), root=tmp_path)


def test_revision_from_the_hub_download_metadata(tmp_path):
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text(f"{SHA}\nsome-etag\n1793226864.56\n")
    assert read_revision(tmp_path) == SHA


def test_revision_from_the_sidecar_file(tmp_path):
    (tmp_path / "REVISION").write_text(SHA + "\n")
    assert read_revision(tmp_path) == SHA


def test_revision_unknown(tmp_path):
    assert read_revision(tmp_path) == "unknown"


def test_quantization_labels():
    assert quantization_label({"quantization": {"group_size": 64, "bits": 8}}) == "8bit-gs64"
    assert quantization_label({"quantization": {"group_size": 64, "bits": 8, "mode": "affine"}}) == "8bit-gs64"
    mixed = {
        "quantization": {
            "group_size": 32,
            "bits": 4,
            "mode": "mxfp4",
            "model.embed_tokens": {"group_size": 64, "bits": 8, "mode": "affine"},
            "lm_head": {"group_size": 64, "bits": 8},
        }
    }
    assert quantization_label(mixed) == "mxfp4-gs32+8bit-gs64"
    assert quantization_label({}) == "none"


# --- make_mlx_generate refuses before loading anything


def _model_dir(tmp_path, config):
    d = tmp_path / "org--m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_describe_model_never_records_a_local_path(tmp_path, monkeypatch):
    d = _model_dir(tmp_path, {"model_type": "gpt_oss", "quantization": {"group_size": 64, "bits": 8}})
    (d / "REVISION").write_text(SHA + "\n")
    monkeypatch.setattr(mlx_backend, "MODELS_ROOT", tmp_path)
    for model_id in ("org/m", str(d)):
        model_dir, family, meta = describe_model(model_id, "cap:512")
        assert model_dir == d and family.name == "gpt_oss"
        assert meta == {
            "model": "org/m",
            "revision": SHA,
            "quantization": "8bit-gs64",
            "temperature": 0,
            "thinking": "cap:512",
            "behaviour": FAMILIES["gpt_oss"].behaviour("cap:512"),
        }
        assert str(tmp_path) not in json.dumps(meta)


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("thinking", ["off", "cap:512"])
def test_every_family_says_what_its_thinking_setting_does(family, thinking):
    text = FAMILIES[family].behaviour(thinking)
    assert text.startswith(family + ": ") and text.endswith(".")
    assert "/" not in text.replace("</", "").replace("<|", "").replace("|>", "")  # no path, markers allowed
    if thinking == "cap:512":
        assert "512" in text


def test_behaviour_states_the_effort_and_the_prefill():
    assert "reasoning_effort=low" in FAMILIES["qwen3"].behaviour("cap:512")
    assert "system turn" in FAMILIES["qwen3"].behaviour("cap:512")
    off = FAMILIES["gpt_oss"].behaviour("off")
    assert "reasoning_effort=low" in off and "empty analysis channel prefilled" in off and "2026-10-28" in off
    assert "enable_thinking=False" in FAMILIES["gemma4"].behaviour("off")
    assert FAMILIES["gpt_oss"].behaviour("cap:64") != FAMILIES["gpt_oss"].behaviour("cap:512")


def test_cap_refused_for_a_family_without_one(tmp_path, monkeypatch):
    no_cap = Family("nocap", ("nocap_type",), {}, {}, "", "<r>", "</r>", "</r>", False)
    monkeypatch.setitem(FAMILIES, "nocap", no_cap)
    _model_dir(tmp_path, {"model_type": "nocap_type"})
    monkeypatch.setattr(mlx_backend, "MODELS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="MODELS.md"):
        make_mlx_generate("org/m", thinking="cap:512")


def test_bad_thinking_refused_before_loading(tmp_path, monkeypatch):
    _model_dir(tmp_path, {"model_type": "qwen3"})
    monkeypatch.setattr(mlx_backend, "MODELS_ROOT", tmp_path)
    with pytest.raises(ValueError):
        make_mlx_generate("org/m", thinking="on")


def test_local_directory_outside_the_models_root_refused_before_loading(tmp_path, monkeypatch):
    d = _model_dir(tmp_path, {"model_type": "qwen3"})
    monkeypatch.setattr(mlx_backend, "MODELS_ROOT", tmp_path / "models")
    with pytest.raises(ValueError, match="mlx-models"):
        make_mlx_generate(str(d), thinking="off")


def test_unknown_family_refused_before_loading(tmp_path, monkeypatch):
    _model_dir(tmp_path, {"model_type": "llama"})
    monkeypatch.setattr(mlx_backend, "MODELS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="llama"):
        make_mlx_generate("org/m", thinking="off")


def test_importing_the_backend_does_not_import_mlx():
    code = "import sys, s1d1_bench.mlx_backend, s1d1_bench.judge_runner; assert 'mlx' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --- the real model (needs the weights under ~/mlx-models)

MESSAGES_ONE_PROMPT = build_messages("How many syllables are in the word smarter?")


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("S1D1_BENCH_MODEL"), reason="set S1D1_BENCH_MODEL to a model id or directory")
def test_mlx_generate_returns_parseable_json_at_temperature_zero_twice():
    gen, meta = make_mlx_generate(os.environ["S1D1_BENCH_MODEL"], thinking="off")
    a, b = gen(MESSAGES_ONE_PROMPT), gen(MESSAGES_ONE_PROMPT)
    assert parse_label(a.text) is not None and a.text == b.text and meta["temperature"] == 0
    assert a.finish_reason == "stop" and a.completion_tokens > 0 and a.prompt_tokens > 0 and a.wall_s > 0
