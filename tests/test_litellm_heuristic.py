"""The vendored LiteLLM heuristic scorer (v1.106.6): its defaults and the few behaviours the bench relies on."""

import json
from pathlib import Path

import pytest

from s1d1_bench.vendor.litellm_heuristic import COMMIT, TAG, Decision, classify
from s1d1_bench.vendor.litellm_heuristic import scorer as litellm_scorer

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "litellm_heuristic"
BASE_PROMPTS = ROOT / "data" / "prompts" / "base-v1.jsonl"


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_pin_is_the_documented_tag_and_commit():
    assert TAG == "v1.106.6"
    assert COMMIT == "61c6ab78393ef45683dd7d963d2a315bd845e20b"


def test_the_default_weights_boundaries_and_thresholds_are_litellms():
    assert litellm_scorer.DEFAULT_DIMENSION_WEIGHTS == {
        "tokenCount": 0.10,
        "codePresence": 0.30,
        "reasoningMarkers": 0.25,
        "technicalTerms": 0.25,
        "simpleIndicators": 0.05,
        "multiStepPatterns": 0.03,
        "questionComplexity": 0.02,
    }
    assert litellm_scorer.DEFAULT_TIER_BOUNDARIES == {
        "simple_medium": 0.15,
        "medium_complex": 0.35,
        "complex_reasoning": 0.60,
    }
    assert litellm_scorer.DEFAULT_TOKEN_THRESHOLDS == {"simple": 15, "complex": 400}
    scorer = litellm_scorer.HeuristicScorer()
    assert scorer.config.dimension_weights == litellm_scorer.DEFAULT_DIMENSION_WEIGHTS
    assert scorer.config.tier_boundaries == litellm_scorer.DEFAULT_TIER_BOUNDARIES
    assert scorer.config.token_thresholds == litellm_scorer.DEFAULT_TOKEN_THRESHOLDS
    assert scorer.config.reasoning_override_min_score is None


def test_an_empty_prompt_is_simple_with_a_negative_token_score():
    decision = classify("")
    assert decision == Decision("SIMPLE", -0.10, ("short (0 tokens)",), "heuristic_scorer")
    assert decision.tier == "SIMPLE"
    assert decision.score == -0.10
    assert decision.signals == ("short (0 tokens)",)
    assert decision.cause == "heuristic_scorer"


def test_thanks_is_simple_and_below_zero():
    decision = classify("thanks")
    assert decision.tier == "SIMPLE"
    assert decision.score == pytest.approx(-0.15)
    assert decision.signals == ("short (1 tokens)", "simple (thanks)")
    assert decision.cause == "heuristic_scorer"


def test_two_reasoning_markers_under_the_floor_stay_simple():
    # Two markers give reasoning 1.0 * 0.25, the short prompt -0.10 and the simple word "hi" -0.05: 0.10 < 0.15.
    decision = classify("hi, step by step and show your work")
    assert decision.signals == ("short (8 tokens)", "reasoning (step by step, show your work)", "simple (hi)")
    assert decision.score == pytest.approx(0.10)
    assert decision.tier == "SIMPLE"
    assert decision.cause == "heuristic_scorer"


def test_two_reasoning_markers_over_the_floor_override():
    # The same two markers with a code keyword (python): the score is 0.40, over the 0.15 floor, and the
    # override promotes the prompt straight to REASONING, whatever the tier boundaries say (0.40 alone is COMPLEX).
    prompt = "Think step by step and show your work on this latency problem:\n```python\nprint(1)\n```"
    decision = classify(prompt)
    assert decision.cause == "reasoning_override"
    assert decision.tier == "REASONING"
    assert any(signal.startswith("reasoning (") for signal in decision.signals)
    assert decision.score >= 0.15
    assert decision.score < 0.60  # the boundaries alone would not have said REASONING


def test_four_question_marks_and_a_multi_step_pattern():
    decision = classify("first read it then answer: a? b? c? d?")
    assert "multi-step" in decision.signals
    assert "4 questions" in decision.signals


def test_a_cjk_keyword_matches_as_a_substring_and_a_latin_one_needs_boundaries():
    # The defaults carry no CJK keyword, so the substring branch is checked on the method with an ad hoc one.
    scorer = litellm_scorer.HeuristicScorer()
    assert not any(litellm_scorer._CJK_CHARACTER.search(k) for k in scorer.code_keywords + scorer.simple_keywords)
    assert scorer._keyword_matches("我需要开发票", "发票")
    assert not scorer._keyword_matches("我需要api开发", "api")
    # A Latin keyword keeps its word boundaries: "api" is not inside "capital".
    assert scorer._keyword_matches("the api is down", "api")
    assert not scorer._keyword_matches("the capital is paris", "api")
    assert "code (" not in "".join(classify("the capital is paris").signals)


def test_the_vendored_scorer_reproduces_the_real_package():
    # v1.106.6.jsonl was made once by the real litellm package (make_fixture.py, see NOTICE.md) on base-v1 and the
    # eight edge prompts. Strict equality: the copy runs the same arithmetic in the same order, so even the float
    # score must be identical, not merely close.
    prompts = {row["id"]: row["prompt"] for row in _read_jsonl(BASE_PROMPTS) + _read_jsonl(FIXTURE_DIR / "edge.jsonl")}
    expected = _read_jsonl(FIXTURE_DIR / "v1.106.6.jsonl")
    assert len(expected) == 368
    assert [row["id"] for row in expected] == list(prompts)
    for row in expected:
        decision = classify(prompts[row["id"]])
        actual = {
            "tier": decision.tier,
            "score": decision.score,
            "signals": list(decision.signals),
            "cause": decision.cause,
        }
        for field, value in actual.items():
            assert value == row[field], f"{row['id']}: {field} differs: vendored {value!r}, litellm {row[field]!r}"
