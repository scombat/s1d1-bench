"""The routellm adapter of `bench classify`: the published routers of RouteLLM (lm-sys/RouteLLM, arXiv 2406.18665) run
locally, one prompt at a time, as the RouteLLM package runs them.

What a RouteLLM router predicts is not a complexity tier: it is the probability that the strong model (GPT-4) beats
the weak one (Mixtral 8x7B) on an English question, learned from Chatbot Arena battles. `classify` returns that
probability, `strong_win_rate`, as the router gives it, and `tier_of_win_rate` reads it onto the v7 tiers with five
equal bounds fixed in advance, with no calibration: trivial [0 ; 0.2), low [0.2 ; 0.4), medium [0.4 ; 0.6), high
[0.6 ; 0.8), expert [0.8 ; 1]. The prompts are given as they are, in the four languages; the routers are English, and
what they do on the other three is a result of the bench, not a defect to correct.

- `bert` (`routellm/bert_gpt4_augmented`, an XLM-RoBERTa-base with three labels, fine-tuned on the Arena battles
  augmented with GPT-4-judged data): `BERTRouter.calculate_strong_win_rate` of RouteLLM at commit `ROUTELLM_COMMIT`
  (`routellm/routers/routers.py`, read on 2026-10-31), reimplemented here so that the `routellm` package (it pulls
  `openai`, `litellm`, `datasets`...) is not a dependency. The same calls: `AutoModelForSequenceClassification` with
  `num_labels=3`, the tokenizer with `padding=True, truncation=True` (so at its `model_max_length`, 512), the model on
  the CPU in float32, softmax of the three logits (max subtracted), `1 - (p_1 + p_2)`: label 0 is "the strong model
  is needed", labels 1 and 2 are a tie and a win of the weak one.
- `mf` is not here: it embeds each prompt with OpenAI's `text-embedding-3-small`, a paid API call, which the bench
  does not make (`EXCLUDED`).
- Revision: the commit of the Hub repo run (`--revision`, else the head when the series starts), whose weights are
  downloaded (read only, no token, `optimizer.pt` and the training states left out) and loaded from the local folder.
"""

import math
import os
from collections.abc import Callable

from s1d1_bench.rubric import LEVELS
from s1d1_bench.s1d1_backend import hub_env, model_info

ROUTELLM_COMMIT = "0b64fdafe049e596a3f5657c219329f24af24198"  # lm-sys/RouteLLM, the commit the research harness pins
ROUTERS = {"bert": "routellm/bert_gpt4_augmented"}
EXCLUDED = {
    "mf": (
        "the matrix-factorisation router embeds each prompt with OpenAI's text-embedding-3-small, a paid API: "
        "the bench makes no paid call (it has no local variant in RouteLLM)"
    ),
}
BERT_FILES = ("config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json", "sentencepiece.bpe.model")
NUM_LABELS = 3
# lower bounds of low, medium, high, expert: a win rate on a bound is in the tier above it (0.2 is low, 0.8 is expert)
TIER_BOUNDS = (0.2, 0.4, 0.6, 0.8)


def tier_of_win_rate(win_rate: float) -> str:
    """The v7 tier of a strong win rate in [0, 1]: five equal bounds, the lower bound belonging to the tier above it
    (trivial below 0.2, low from 0.2, medium from 0.4, high from 0.6, expert from 0.8 up to and including 1.0).
    Raises ValueError outside [0, 1] or for NaN: nothing is clipped."""
    if not isinstance(win_rate, int | float) or isinstance(win_rate, bool) or not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"a strong win rate is a probability in [0, 1], not {win_rate!r}")
    return LEVELS[sum(win_rate >= bound for bound in TIER_BOUNDS)]


def strong_win_rate(logits) -> tuple[float, list[float]]:
    """(1 - (p_1 + p_2), [p_0, p_1, p_2]) from the three logits, as `BERTRouter.calculate_strong_win_rate` computes
    them (numpy float32, the maximum subtracted before the exponential). Values rounded to 6 decimals, which is what
    the folder holds: the tier is read from the rounded value."""
    import numpy as np

    logits = np.asarray(logits)
    exp_scores = np.exp(logits - np.max(logits))
    softmax_scores = exp_scores / np.sum(exp_scores)
    binary_prob = np.sum(softmax_scores[-2:])  # labels 1 and 2: a tie, the weak model wins
    return round(float(1 - binary_prob), 6), [round(float(p), 6) for p in softmax_scores]


def make_routellm_classifier(router: str, revision: str | None) -> tuple[Callable[[list[str]], list[dict]], dict]:
    """(classify, meta): `classify(prompts)` runs the router `router` on each prompt alone and returns, in order,
    dicts `strong_win_rate` (float in [0, 1]) and `probabilities` (the three softmax values). `meta` holds `model`,
    `revision` (the full commit), `device`, `precision`, `truncated` (prompts cut at the 512 tokens; 0 until
    `classify` has run) and `max_length`. Raises ValueError for an excluded or unknown router, ImportError when
    torch or transformers is missing."""
    if router in EXCLUDED:
        raise ValueError(f"the router {router!r} is excluded: {EXCLUDED[router]}")
    if router not in ROUTERS:
        raise ValueError(f"unknown RouteLLM router {router!r}: expected one of {', '.join(ROUTERS)}")
    hub_env()
    os.environ.pop("HF_TOKEN", None)  # public repo, read only
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo = ROUTERS[router]
    commit = model_info(repo, revision).sha
    from huggingface_hub import snapshot_download

    folder = snapshot_download(repo, revision=commit, allow_patterns=list(BERT_FILES), token=False)
    model = AutoModelForSequenceClassification.from_pretrained(folder, num_labels=NUM_LABELS)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(folder)
    meta = {
        "model": repo,
        "revision": commit,
        "device": "cpu",
        "precision": str(next(model.parameters()).dtype).removeprefix("torch."),
        "max_length": tokenizer.model_max_length,
        "truncated": 0,
    }

    def classify(prompts: list[str]) -> list[dict]:
        out = []
        truncated = 0
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            truncated += len(tokenizer(prompt)["input_ids"]) > tokenizer.model_max_length
            with torch.no_grad():
                logits = model(**inputs).logits.numpy()[0]
            rate, probabilities = strong_win_rate(logits)
            if not math.isfinite(rate):
                raise ValueError(f"the router gave a win rate of {rate} for a prompt: nothing is written")
            out.append({"strong_win_rate": rate, "probabilities": probabilities})
        meta["truncated"] = truncated
        return out

    return classify, meta
