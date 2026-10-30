"""LiteLLM's heuristic complexity scorer (tag v1.106.6), vendored verbatim; see NOTICE.md and LICENSE.

`classify(prompt)` runs the scorer with its default configuration and no system prompt, the way the router calls it for
a bare user message. The tier is the value of LiteLLM's `ComplexityTier` ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING").
`cause` says whether the weighted score chose the tier ("heuristic_scorer") or the two-reasoning-markers override did
("reasoning_override").
"""

from typing import NamedTuple

from s1d1_bench.vendor.litellm_heuristic.scorer import HeuristicScorer

TAG = "v1.106.6"
COMMIT = "61c6ab78393ef45683dd7d963d2a315bd845e20b"


class Decision(NamedTuple):
    tier: str
    score: float
    signals: tuple[str, ...]
    cause: str


_SCORER = HeuristicScorer()


def classify(prompt: str) -> Decision:
    tier, score, signals, cause = _SCORER._score_and_classify(prompt)
    return Decision(tier.value, score, signals, cause)
