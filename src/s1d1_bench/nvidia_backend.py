"""The nvidia adapter of `bench classify`: NVIDIA's `prompt-task-and-complexity-classifier` (NemoCurator,
DeBERTa-v3-base with eight classification heads) run locally, one prompt at a time, as its model card runs it.

What the model measures is not a v7 tier: it labels an English prompt with one of eleven task types and with six
complexity dimensions (creativity, reasoning, contextual knowledge, domain knowledge, constraints, number of few
shots), and `prompt_complexity_score` is their weighted sum, 0.35 creativity + 0.25 reasoning + 0.15 constraints +
0.15 domain knowledge + 0.05 contextual knowledge + 0.05 few shots, in [0 ; 1]. `classify` returns every head as the
model gives it; `tier_of_score` reads `prompt_complexity_score` onto the v7 tiers with the five equal bounds of the
RouteLLM adapter (`routellm_backend.TIER_BOUNDS`), fixed in advance, with no calibration: trivial [0 ; 0.2), low
[0.2 ; 0.4), medium [0.4 ; 0.6), high [0.6 ; 0.8), expert [0.8 ; 1].

The model is English only (its card says so, and its 4,024 training prompts are English): the bench gives it the English
ids of the set and nothing else (`COVERED_LANGUAGES`); the other ids are written as not predicted, never run.

The model is rebuilt from the code of its card (`CustomModel`, revision `CARD_REVISION`, read on 2026-10-31) without
`PyTorchModelHubMixin` and without downloading the backbone weights: the backbone is `AutoModel.from_config` of the
config of `microsoft/deberta-v3-base` (revision `BACKBONE_REVISION`, config file only, MIT), and the weights of the
card's repo are loaded over it with `strict=True`. The same calls as the card: mean pooling over the attention mask,
one linear head per target, the tokenizer with `padding="max_length", truncation=True, max_length=512`, softmax per
head, the weights of `weights_map` and the divisors of `divisor_map`, scores rounded to four decimals, the number of
few shots under 0.05 set to 0, the complexity score rounded to five decimals, the top two task types with the second
replaced by "NA" under a probability of 0.1. CPU, float32.

- The text given to the model is the prompt as it is. The card's code sample prefixes "Prompt: " to its text; its
  first table (the mystery prompt) is reproduced digit for digit only without the prefix, its code sample only with it.
  The bench does not add the prefix.
- Revision: the commit of the Hub repo run (`--revision`, else the head when the series starts), whose files are
  downloaded (read only, no token) and loaded from the local folder.
"""

import json
import math
import os
from collections.abc import Callable

from s1d1_bench.routellm_backend import TIER_BOUNDS
from s1d1_bench.rubric import LEVELS
from s1d1_bench.s1d1_backend import hub_env, model_info

REPO = "nvidia/prompt-task-and-complexity-classifier"
CARD_REVISION = "fea1121511eafabaf7dd6fc66863dcb04f74defb"  # the head of the repo when the card was read
BACKBONE_REPO = "microsoft/deberta-v3-base"
BACKBONE_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "spm.model",
)  # fmt: skip
COVERED_LANGUAGES = ("eng",)
MAX_LENGTH = 512
HIDDEN = 768
# target of each head, in the order of config["target_sizes"] (head_0 ... head_7); no_label_reason is the seventh
SCORED = (
    "creativity_scope", "reasoning", "contextual_knowledge", "number_of_few_shots", "domain_knowledge",
    "no_label_reason", "constraint_ct",
)  # fmt: skip
SCORE_WEIGHTS = {
    "creativity_scope": 0.35, "reasoning": 0.25, "constraint_ct": 0.15, "domain_knowledge": 0.15,
    "contextual_knowledge": 0.05, "number_of_few_shots": 0.05,
}  # fmt: skip


def tier_of_score(score: float) -> str:
    """The v7 tier of a complexity score in [0, 1]: five equal bounds, the lower bound belonging to the tier above it
    (trivial below 0.2, low from 0.2, medium from 0.4, high from 0.6, expert from 0.8 up to and including 1.0).
    Raises ValueError outside [0, 1] or for NaN: nothing is clipped."""
    if not isinstance(score, int | float) or isinstance(score, bool) or not 0.0 <= score <= 1.0:
        raise ValueError(f"a complexity score is a number in [0, 1], not {score!r}")
    return LEVELS[sum(score >= bound for bound in TIER_BOUNDS)]


def covered(language: str) -> bool:
    return language in COVERED_LANGUAGES


def heads_of_probabilities(probabilities: list, config: dict) -> dict:
    """The output of the card's `CustomModel.process_logits` for ONE prompt, from the softmax of each of the eight heads
    (`probabilities`: eight 1-D float32 numpy arrays, in the order of `config["target_sizes"]`). Keys: task_type_1,
    task_type_2, task_type_prob, creativity_scope, reasoning, contextual_knowledge, number_of_few_shots,
    domain_knowledge, no_label_reason, constraint_ct, prompt_complexity_score (floats rounded as the card rounds)."""
    import numpy as np

    order = np.argsort(-probabilities[0], kind="stable")[:2]  # torch.topk(k=2)
    top2 = [config["task_type_map"][str(int(i))] for i in order]
    top2_prob = [round(float(probabilities[0][i]), 3) for i in order]
    if top2_prob[1] < 0.1:
        top2[1] = "NA"
    out = {"task_type_1": top2[0], "task_type_2": top2[1], "task_type_prob": top2_prob[0]}
    for index, target in enumerate(SCORED, start=1):
        weights = np.array(config["weights_map"][target])
        score = round(float(np.sum(probabilities[index] * weights) / config["divisor_map"][target]), 4)
        if target == "number_of_few_shots" and score < 0.05:
            score = 0
        out[target] = score
    out["prompt_complexity_score"] = round(sum(w * out[target] for target, w in SCORE_WEIGHTS.items()), 5)
    return out


def make_nvidia_classifier(revision: str | None) -> tuple[Callable[[list[str]], list[dict]], dict]:
    """(classify, meta): `classify(prompts)` runs the model on each prompt alone and returns, in order, the dicts of
    `heads_of_probabilities`. `meta` holds `model`, `revision` (the full commit), `backbone`, `backbone_revision`,
    `device`, `precision`, `max_length` and `truncated` (prompts cut at 512 tokens; 0 until `classify` has run).
    Raises ImportError when torch or transformers is missing."""
    hub_env()
    os.environ.pop("HF_TOKEN", None)  # public repos, read only
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    commit = model_info(REPO, revision).sha
    folder = snapshot_download(REPO, revision=commit, allow_patterns=list(FILES), token=False)
    backbone_folder = snapshot_download(
        BACKBONE_REPO, revision=BACKBONE_REVISION, allow_patterns=["config.json"], token=False
    )
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    sizes = list(config["target_sizes"].values())

    class MeanPooling(torch.nn.Module):
        def forward(self, last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, 1)
            return summed / torch.clamp(mask.sum(1), min=1e-9)

    class MulticlassHead(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(backbone_folder))
            self.heads = [MulticlassHead(self.backbone.config.hidden_size, size) for size in sizes]
            for i, head in enumerate(self.heads):
                self.add_module(f"head_{i}", head)  # the names of the state dict: head_0.fc.weight ...
            self.pool = MeanPooling()

        def forward(self, batch):
            outputs = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pooled = self.pool(outputs.last_hidden_state, batch["attention_mask"])
            return [head(pooled) for head in self.heads]

    model = CustomModel()
    model.load_state_dict(load_file(os.path.join(folder, "model.safetensors")), strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(folder)
    meta = {
        "model": REPO,
        "revision": commit,
        "backbone": BACKBONE_REPO,
        "backbone_revision": BACKBONE_REVISION,
        "device": "cpu",
        "precision": str(next(model.parameters()).dtype).removeprefix("torch."),
        "max_length": MAX_LENGTH,
        "truncated": 0,
    }

    def classify(prompts: list[str]) -> list[dict]:
        out = []
        truncated = 0
        for prompt in prompts:
            encoded = tokenizer(
                [prompt], return_tensors="pt", add_special_tokens=True, max_length=MAX_LENGTH, padding="max_length",
                truncation=True,
            )  # fmt: skip
            truncated += len(tokenizer(prompt)["input_ids"]) > MAX_LENGTH
            with torch.no_grad():
                softmaxes = [torch.softmax(logits, dim=1)[0].numpy() for logits in model(encoded)]
            heads = heads_of_probabilities(softmaxes, config)
            if not math.isfinite(heads["prompt_complexity_score"]):
                raise ValueError("the model gave a complexity score that is not finite: nothing is written")
            out.append(heads)
        meta["truncated"] = truncated
        return out

    return classify, meta
