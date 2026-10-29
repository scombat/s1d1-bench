"""The `mlx` backend of `bench judge`: a local MLX model as a GenerateFn, greedy (temperature 0), reasoning off or
capped. `mlx_lm` is imported inside `make_mlx_generate` only, so this module imports without the `judge` group.

What each family does with its reasoning, and why, is in judges/MODELS.md.
"""

import importlib.metadata
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from s1d1_bench.judge_runner import GenerateFn, GenerateResult

MODELS_ROOT = Path.home() / "mlx-models"  # model weights live here, never in a repository
REVISION_FILE = "REVISION"  # sidecar written at download time when the Hub metadata is missing
_HUB_METADATA = Path(".cache") / "huggingface" / "download"
_SHA = re.compile(r"[0-9a-f]{40}")
_CAP = re.compile(r"cap:([1-9][0-9]*)")

# (prompt token ids, max new tokens) -> the generated token ids, end-of-sequence token excluded
StreamFn = Callable[[list[int], int], Iterable[int]]


@dataclass(frozen=True)
class Family:
    """How a model family reasons, and how `bench judge` switches that off or caps it."""

    name: str
    model_types: tuple[str, ...]  # config.json `model_type` values of the family
    off_kwargs: dict  # chat-template kwargs for thinking "off"
    cap_kwargs: dict  # chat-template kwargs for thinking "cap:<n>"
    off_prefill: str  # text appended after the generation prompt for "off" (when the template has no switch)
    open_marker: str  # opens the reasoning
    close_marker: str  # closes it, as the model writes it
    inject: str  # appended to the generated text when the cap is reached, before generating the answer
    cap_applicable: bool
    fixed_date: str | None = None  # the template prints today's date: pinned, or a run would not replay
    off_behaviour: str = ""  # what "off" does, for meta.json notes
    cap_behaviour: str = ""  # what "cap:{n}" does, for meta.json notes

    def template_kwargs(self, thinking: str) -> dict:
        return dict(self.off_kwargs if parse_thinking(thinking) is None else self.cap_kwargs)

    def behaviour(self, thinking: str) -> str:
        """One or two factual sentences on what `thinking` does for this family (written first in meta notes)."""
        cap = parse_thinking(thinking)
        text = self.off_behaviour if cap is None else self.cap_behaviour.format(n=cap)
        if self.fixed_date is not None:
            text += f" The date the chat template prints is pinned to {self.fixed_date}."
        return f"{self.name}: {text}"


FAMILIES: dict[str, Family] = {
    "gemma4": Family(
        name="gemma4",
        model_types=("gemma4", "gemma4_text"),
        off_kwargs={"enable_thinking": False},
        cap_kwargs={"enable_thinking": True},
        off_prefill="",
        open_marker="<|channel>thought",
        close_marker="<channel|>",
        inject="<channel|>",
        cap_applicable=True,
        off_behaviour="enable_thinking=False (the chat template prefills an empty thought channel): no reasoning.",
        cap_behaviour="enable_thinking=True; bench judge cuts the thought channel at {n} tokens (closes it and "
        "generates the answer after it), only if the reasoning would go past {n} tokens.",
    ),
    "qwen3": Family(
        name="qwen3",
        model_types=("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe"),
        off_kwargs={"enable_thinking": False},
        cap_kwargs={"enable_thinking": True, "reasoning_effort": "low"},  # the template's default effort is xhigh
        off_prefill="",
        open_marker="<think>",
        close_marker="</think>",
        inject="\n</think>\n\n",
        cap_applicable=True,
        off_behaviour="enable_thinking=False (the chat template prefills an empty think block): no reasoning.",
        cap_behaviour="enable_thinking=True with reasoning_effort=low (the chat template prepends a low-effort "
        "sentence to the system turn); bench judge cuts the think block at {n} tokens (closes it and generates the "
        "answer after it), only if the reasoning would go past {n} tokens.",
    ),
    "gpt_oss": Family(
        name="gpt_oss",
        model_types=("gpt_oss",),
        # the harmony template has no off switch: "off" prefills an empty analysis channel and opens the final one
        off_kwargs={"reasoning_effort": "low"},
        cap_kwargs={"reasoning_effort": "low"},
        off_prefill="<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
        open_marker="<|channel|>analysis<|message|>",
        close_marker="<|end|>",
        inject="<|end|><|start|>assistant<|channel|>final<|message|>",
        cap_applicable=True,
        fixed_date="2026-10-28",
        off_behaviour="reasoning_effort=low; empty analysis channel prefilled and final channel opened (the "
        "template has no off switch): no reasoning.",
        cap_behaviour="reasoning_effort=low; bench judge cuts the analysis channel at {n} tokens (closes it, opens "
        "the final channel and generates the answer), only if the analysis would go past {n} tokens.",
    ),
}


def parse_thinking(thinking: str) -> int | None:
    """`off` -> None, `cap:<n>` (n >= 1) -> n; anything else raises ValueError."""
    if thinking == "off":
        return None
    match = _CAP.fullmatch(thinking)
    if match is None:
        raise ValueError(f"thinking must be 'off' or 'cap:<n>' with n >= 1, got {thinking!r}")
    return int(match.group(1))


def _squash(name: str) -> str:
    return name.lower().replace("_", "")


def detect_family(config: dict) -> str:
    """The FAMILIES key of a model, from its config.json `model_type` (or `text_config.model_type`, or the
    architecture names). Raises ValueError for anything else."""
    types = [config.get("model_type"), (config.get("text_config") or {}).get("model_type")]
    for name, family in FAMILIES.items():
        if any(t in family.model_types for t in types if t):
            return name
    for arch in config.get("architectures") or []:
        stem = _squash(re.sub(r"For(CausalLM|ConditionalGeneration)$", "", arch))
        for name, family in FAMILIES.items():
            if stem in {_squash(t) for t in family.model_types}:
                return name
    raise ValueError(
        f"unknown model family (model_type {config.get('model_type')!r}, architectures "
        f"{config.get('architectures')!r}): bench judge knows {', '.join(FAMILIES)} (see judges/MODELS.md)"
    )


def resolve_model(model_id: str, root: Path | None = None) -> tuple[Path, str]:
    """(local directory, Hub repo id recorded in meta.json) of a model given as a Hub repo id (`org/name` ->
    `<root>/org--name`) or as such a directory of the models root. A directory anywhere else is refused (meta.json
    must never hold a local path, nor a model nobody can fetch). Never downloads: a missing model raises
    FileNotFoundError with the command to run."""
    root = MODELS_ROOT if root is None else Path(root)
    given = Path(model_id).expanduser()
    if given.is_dir():
        where = given.resolve()
        if where.parent != root.resolve():
            raise ValueError(
                f"{model_id}: a --model directory must be one of the models directory ~/mlx-models/ (here {root}), "
                "named org--name after its Hub repo; give the Hub repo id instead (see judges/MODELS.md)"
            )
        org, sep, name = where.name.partition("--")
        if not (sep and org and name):
            raise ValueError(f"{model_id}: a model directory must be named org--name after its Hub repo org/name")
        return given, f"{org}/{name}"
    if model_id.count("/") != 1 or model_id.startswith((".", "/", "~")):
        raise FileNotFoundError(f"{model_id}: neither a directory nor a Hub repo id (org/name)")
    target = root / model_id.replace("/", "--")
    if not target.is_dir():
        raise FileNotFoundError(
            f"{model_id}: not found at {target}. Download it first (public Hub, no token; see judges/MODELS.md):\n"
            f"  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --group pool hf download {model_id} --local-dir {target}"
        )
    return target, model_id


def read_revision(model_dir: Path | str) -> str:
    """The Hub commit of a downloaded model: from the metadata `hf download --local-dir` writes, else from a
    `REVISION` sidecar file, else "unknown"."""
    model_dir = Path(model_dir)
    meta_dir = model_dir / _HUB_METADATA
    candidates = [meta_dir / "config.json.metadata", *sorted(meta_dir.glob("*.metadata"))]
    for path in [*candidates, model_dir / REVISION_FILE]:
        if path.is_file():
            first = (path.read_text(encoding="utf-8", errors="replace").splitlines() or [""])[0].strip()
            if _SHA.fullmatch(first):
                return first
    return "unknown"


def _describe(q: dict) -> str:
    mode = q.get("mode", "affine")
    return f"{q['bits']}bit-gs{q['group_size']}" if mode == "affine" else f"{mode}-gs{q['group_size']}"


def quantization_label(config: dict) -> str:
    """`8bit-gs64` from config.json `quantization`; layers quantized otherwise are appended (`mxfp4-gs32+8bit-gs64`);
    "none" when the weights are not quantized."""
    q = config.get("quantization") or config.get("quantization_config")
    if not q or "bits" not in q:
        return "none"
    base = _describe(q)
    others = sorted({_describe(v) for v in q.values() if isinstance(v, dict) and "bits" in v} - {base})
    return "+".join([base, *others])


@dataclass(frozen=True)
class Capped:
    tokens: list[int]  # the assistant turn: generated tokens plus the injected closing marker, if any
    generated: int  # tokens the model produced (reasoning included, injected ones excluded)
    finish_reason: str  # "length" if max_tokens was reached, else "stop"
    capped: bool


def _ends_with(tokens: list[int], marker: tuple[int, ...]) -> bool:
    return bool(marker) and len(tokens) >= len(marker) and tuple(tokens[-len(marker) :]) == marker


def generate_capped(
    stream: StreamFn,
    prompt: list[int],
    max_tokens: int,
    cap: int | None,
    open_ids: tuple[int, ...],
    close_ids: tuple[int, ...],
    inject_ids: tuple[int, ...],
    open_in_prompt: bool,
) -> Capped:
    """Generate at most `max_tokens`; when the reasoning (after `open_ids`, or from the first token if the prompt
    opened it) reaches `cap` tokens without closing, stop, append `inject_ids` and generate the answer from there,
    within what is left of `max_tokens`. Only the first reasoning block counts."""
    tokens: list[int] = []
    state = "reasoning" if open_in_prompt else "before"
    count = 0
    capped = False
    for token in stream(prompt, max_tokens):
        tokens.append(token)
        if state == "before":
            if _ends_with(tokens, open_ids):
                state = "reasoning"
        elif state == "reasoning":
            if _ends_with(tokens, close_ids):
                state = "done"
            else:
                count += 1
                if cap is not None and count >= cap:
                    capped = True
                    break
    generated = len(tokens)
    if not capped:
        return Capped(tokens, generated, "length" if generated >= max_tokens else "stop", False)
    tokens.extend(inject_ids)
    remaining = max_tokens - generated
    if remaining <= 0:
        return Capped(tokens, generated, "length", True)
    answer = list(stream(prompt + tokens, remaining))
    tokens.extend(answer)
    return Capped(tokens, generated + len(answer), "length" if len(answer) >= remaining else "stop", True)


def _mlx_stream(model, tokenizer, sampler) -> StreamFn:
    from mlx_lm import stream_generate

    def stream(prompt: list[int], max_tokens: int):
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
            if response.finish_reason == "stop":  # the last response carries the end-of-sequence token
                return
            yield response.token

    return stream


def describe_model(model_id: str, thinking: str) -> tuple[Path, Family, dict]:
    """Everything known about a model before loading it: its directory, its family and the meta.json fields it
    fills (`model` is the Hub repo id, never a path; `behaviour` says what `thinking` does). Raises ValueError for a
    bad `thinking`, an unknown family or a cap the family cannot apply, FileNotFoundError for a missing model."""
    cap = parse_thinking(thinking)
    model_dir, hub_id = resolve_model(model_id)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    family = FAMILIES[detect_family(config)]
    if cap is not None and not family.cap_applicable:
        raise ValueError(f"{model_id}: no reasoning cap for the {family.name} family (see judges/MODELS.md)")
    meta = {
        "model": hub_id,
        "revision": read_revision(model_dir),
        "quantization": quantization_label(config),
        "temperature": 0,
        "thinking": thinking,
        "behaviour": family.behaviour(thinking),
    }
    return model_dir, family, meta


def make_mlx_generate(model_id: str, thinking: str, max_tokens: int = 1024) -> tuple[GenerateFn, dict]:
    """Load a local MLX model (see `resolve_model`) and return a GenerateFn and the meta.json fields it fills
    (`describe_model`, plus `mlx_lm_version`).

    The GenerateFn renders the chat template (system message = the rubric) with the family's kwargs for `thinking`,
    generates greedily and returns the whole assistant turn, reasoning included, special tokens kept, so that
    `parse_label` strips the reasoning itself.
    """
    model_dir, family, meta = describe_model(model_id, thinking)
    cap = parse_thinking(thinking)
    try:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("bench judge needs mlx-lm: `uv run --group judge bench judge ...`") from exc

    meta["mlx_lm_version"] = importlib.metadata.version("mlx-lm")
    model, tokenizer = load(str(model_dir))
    stream = _mlx_stream(model, tokenizer, make_sampler(temp=0.0))
    kwargs = family.template_kwargs(thinking)
    if family.fixed_date is not None:
        kwargs["strftime_now"] = lambda _format, date=family.fixed_date: date
    prefill = family.off_prefill if cap is None else ""

    def encode(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    open_ids, close_ids, inject_ids = encode(family.open_marker), encode(family.close_marker), encode(family.inject)

    def generate(messages: list[dict]) -> GenerateResult:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
        text += prefill
        prompt = list(encode(text))
        open_in_prompt = text.rfind(family.open_marker) > text.rfind(family.close_marker)
        started = time.perf_counter()
        out = generate_capped(stream, prompt, max_tokens, cap, open_ids, close_ids, inject_ids, open_in_prompt)
        wall_s = time.perf_counter() - started
        answer = tokenizer.decode(out.tokens, skip_special_tokens=False)
        return GenerateResult(answer, len(prompt), out.generated, out.finish_reason, round(wall_s, 3))

    return generate, meta
