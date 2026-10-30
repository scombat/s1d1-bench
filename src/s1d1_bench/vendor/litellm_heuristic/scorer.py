"""LiteLLM's heuristic complexity scorer, copied verbatim from tag v1.106.6 (see NOTICE.md and LICENSE).

The blocks below are copied line for line from `litellm/router_strategy/complexity_router/config.py` and
`complexity_router.py` at commit 61c6ab78393ef45683dd7d963d2a315bd845e20b. Each is introduced by a `# copied:` marker.
The few edits that make the code importable without the `litellm` package are listed in NOTICE.md. Do not touch the
copied text: the tests check that it still scores as the real package does.
"""

from __future__ import annotations

import re
from enum import Enum
from itertools import chain, islice
from typing import Final, Literal


# copied: config.py lines 47-54
class ComplexityTier(str, Enum):
    """Complexity tiers for routing decisions."""

    NON_REASONING = "NON_REASONING"
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"


# copied: config.py lines 288-392
# ─── Default Keyword Lists ───
# Note: Keywords should be full words/phrases to avoid substring false positives.
# The matching logic uses word boundary detection for single-word keywords.

DEFAULT_CODE_KEYWORDS: Final[list[str]] = [
    "function",
    "class",
    "def",
    "const",
    "let",
    "var",
    "import",
    "export",
    "return",
    "async",
    "await",
    "try",
    "catch",
    "exception",
    "error",
    "debug",
    "api",
    "endpoint",
    "request",
    "response",
    "database",
    "sql",
    "query",
    "schema",
    "algorithm",
    "implement",
    "refactor",
    "optimize",
    "python",
    "javascript",
    "typescript",
    "java",
    "rust",
    "golang",
    "react",
    "vue",
    "angular",
    "node",
    "docker",
    "kubernetes",
    "git",
    "commit",
    "merge",
    "branch",
    "pull request",
]

DEFAULT_REASONING_KEYWORDS: Final[list[str]] = [
    "step by step",
    "think through",
    "let's think",
    "reason through",
    "analyze this",
    "break down",
    "explain your reasoning",
    "show your work",
    "chain of thought",
    "think carefully",
    "consider all",
    "evaluate",
    "pros and cons",
    "compare and contrast",
    "weigh the options",
    "logical",
    "deduce",
    "infer",
    "conclude",
]

DEFAULT_TECHNICAL_KEYWORDS: Final[list[str]] = [
    "architecture",
    "distributed",
    "scalable",
    "microservice",
    "machine learning",
    "neural network",
    "deep learning",
    "encryption",
    "authentication",
    "authorization",
    "performance",
    "latency",
    "throughput",
    "benchmark",
    "concurrency",
    "parallel",
    "threading",
    "memory",
    "cpu",
    "gpu",
    "optimization",
    "protocol",
    "tcp",
    "http",
    "grpc",
    "websocket",
    "container",
    "orchestration",
    # Note: "async", "kubernetes", "docker" are in DEFAULT_CODE_KEYWORDS
]

# copied: config.py lines 419-447
DEFAULT_SIMPLE_KEYWORDS: Final[list[str]] = [
    "what is",
    "what's",
    "define",
    "definition of",
    "who is",
    "who was",
    "when did",
    "when was",
    "where is",
    "where was",
    "how many",
    "how much",
    "yes or no",
    "true or false",
    "simple",
    "brief",
    "short",
    "quick",
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "goodbye",
    "bye",
    "okay",
    # Note: "ok" removed due to false positives (matches "token", "book", etc.)
]


# copied: config.py lines 450-477
# ─── Default Dimension Weights ───

DEFAULT_DIMENSION_WEIGHTS: Final[dict[str, float]] = {
    "tokenCount": 0.10,  # Reduced - length is less important than content
    "codePresence": 0.30,  # High - code requests need capable models
    "reasoningMarkers": 0.25,  # High - explicit reasoning requests
    "technicalTerms": 0.25,  # High - technical content matters
    "simpleIndicators": 0.05,  # Low - don't over-penalize simple patterns
    "multiStepPatterns": 0.03,
    "questionComplexity": 0.02,
}


# ─── Default Tier Boundaries ───

DEFAULT_TIER_BOUNDARIES: Final[dict[str, float]] = {
    "simple_medium": 0.15,  # Lower threshold to catch more MEDIUM cases
    "medium_complex": 0.35,  # Lower threshold to catch technical COMPLEX cases
    "complex_reasoning": 0.60,  # Reasoning tier reserved for explicit reasoning markers
}


# ─── Default Token Thresholds ───

DEFAULT_TOKEN_THRESHOLDS: Final[dict[str, int]] = {
    "simple": 15,  # Only very short prompts (<15 tokens) are penalized
    "complex": 400,  # Long prompts (>400 tokens) get complexity boost
}


# copied: config.py line 865
CUSTOM_PATTERN_SCAN_CHARS: Final[int] = 2048


# copied: complexity_router.py lines 406-411
def _append_custom_keywords(base_keywords: list[str], custom_keywords: list[str] | None) -> list[str]:
    if not custom_keywords:
        return base_keywords
    base_lowered: Final = frozenset(keyword.lower() for keyword in base_keywords)
    deduped_custom = {keyword.lower(): keyword for keyword in custom_keywords if keyword.lower() not in base_lowered}
    return [*base_keywords, *deduped_custom.values()]


# copied: complexity_router.py line 491
_CJK_CHARACTER: Final = re.compile("[぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿ｦ-ﾝ\U00020000-\U0003ffff]")


# copied: complexity_router.py lines 1030-1038
class DimensionScore:
    """Represents a score for a single dimension with optional signal."""

    __slots__ = ("name", "score", "signal")

    def __init__(self, name: str, score: float, signal: str | None = None):
        self.name = name
        self.score = score
        self.signal = signal


class _DefaultConfig:
    """Stands for `ComplexityRouterConfig()`: the default value of each field the copied methods read."""

    def __init__(self) -> None:
        self.token_thresholds = DEFAULT_TOKEN_THRESHOLDS.copy()
        self.dimension_weights = DEFAULT_DIMENSION_WEIGHTS.copy()
        self.tier_boundaries = DEFAULT_TIER_BOUNDARIES.copy()
        self.reasoning_override_min_score = None
        self.code_keywords = None
        self.reasoning_keywords = None
        self.technical_keywords = None
        self.custom_technical_keywords = None
        self.simple_keywords = None


class HeuristicScorer:
    """`ComplexityRouter` reduced to its heuristic scoring methods, with the default configuration."""

    def __init__(self) -> None:
        self.config = _DefaultConfig()

        # copied: complexity_router.py lines 1428-1434
        self.code_keywords = self.config.code_keywords or DEFAULT_CODE_KEYWORDS
        self.reasoning_keywords = self.config.reasoning_keywords or DEFAULT_REASONING_KEYWORDS
        self.technical_keywords = _append_custom_keywords(
            self.config.technical_keywords or DEFAULT_TECHNICAL_KEYWORDS,
            self.config.custom_technical_keywords,
        )
        self.simple_keywords = self.config.simple_keywords or DEFAULT_SIMPLE_KEYWORDS
        self._custom_dimensions = ()

        # copied: complexity_router.py lines 1465-1470
        self._multi_step_patterns = [
            re.compile(r"first.*?then", re.IGNORECASE),
            re.compile(r"step\s*\d", re.IGNORECASE),
            re.compile(r"\d+\.\s"),
            re.compile(r"[a-z]\)\s", re.IGNORECASE),
        ]

    # copied: complexity_router.py lines 1606-1820
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        Uses a simple heuristic: ~4 characters per token on average.
        """
        return len(text) // 4

    def _score_token_count(self, estimated_tokens: int) -> DimensionScore:
        """Score based on token count."""
        thresholds: Final = self.config.token_thresholds
        simple_threshold: Final = thresholds.get("simple", 15)
        complex_threshold: Final = thresholds.get("complex", 400)

        if estimated_tokens < simple_threshold:
            return DimensionScore("tokenCount", -1.0, f"short ({estimated_tokens} tokens)")
        if estimated_tokens > complex_threshold:
            return DimensionScore("tokenCount", 1.0, f"long ({estimated_tokens} tokens)")
        return DimensionScore("tokenCount", 0, None)

    def _keyword_matches(self, text: str, keyword: str) -> bool:
        r"""
        Check if a keyword matches in text.

        Single-word keywords use regex word boundaries to avoid false positives, e.g. "api"
        must not match "capital" and "error" must not match "terrorism".

        Multi-word phrases and keywords containing CJK match as plain substrings. CJK is
        written without spaces and every CJK character is a regex word character, so `\b`
        never fires between two of them: `\b发票\b` misses "我需要开发票" entirely. The gate is
        on the keyword rather than the text, so a keyword with no CJK in it keeps word
        boundary matching no matter what script the prompt is written in.
        """
        kw_lower: Final = keyword.lower()

        if " " in kw_lower or _CJK_CHARACTER.search(kw_lower):
            return kw_lower in text

        pattern: Final = r"\b" + re.escape(kw_lower) + r"\b"
        return bool(re.search(pattern, text))

    def _score_keyword_match(
        self,
        text: str,
        keywords: list[str],
        name: str,
        signal_label: str,
        thresholds: tuple[int, int],  # (low, high)
        scores: tuple[float, float, float],  # (none, low, high)
    ) -> tuple[DimensionScore, int]:
        """Score based on keyword matches using word boundary matching.

        `text` is always the caller's own message (never the system prompt) -- see
        `_score_and_classify`. Signals are persisted to the request's spend log, which
        the caller can read, so every matched term named in the signal is one the
        caller supplied itself; there is nothing left to disclose that it couldn't
        already see.

        Returns:
            Tuple of (DimensionScore, match_count) so callers can reuse the count.
        """
        low_threshold, high_threshold = thresholds
        score_none, score_low, score_high = scores

        matches: Final = [kw for kw in keywords if self._keyword_matches(text, kw)]
        match_count: Final = len(matches)
        if match_count < low_threshold:
            return DimensionScore(name, score_none, None), match_count

        detail: Final = ", ".join(matches[:3])
        score: Final = score_high if match_count >= high_threshold else score_low
        return DimensionScore(name, score, f"{signal_label} ({detail})"), match_count

    def _count_custom_hits(self, matchers: _CustomDimensionMatchers, user_text: str, scanned: str) -> int:
        hits: Final = chain(
            (self._keyword_matches(user_text, keyword) for keyword in matchers.keywords),
            (pattern.search(scanned) is not None for pattern in matchers.patterns),
        )
        return sum(islice((1 for hit in hits if hit), matchers.saturation))

    def _score_custom_dimensions(self, prompt: str, user_text: str) -> tuple[tuple[DimensionScore, float], ...]:
        if not self._custom_dimensions:
            return ()
        scanned: Final = prompt[:CUSTOM_PATTERN_SCAN_CHARS]
        return tuple(
            (
                DimensionScore(
                    matchers.dimension.name, hits / matchers.saturation, f"custom ({matchers.dimension.name})"
                ),
                matchers.dimension.weight,
            )
            for matchers in self._custom_dimensions
            if (hits := self._count_custom_hits(matchers, user_text, scanned))
        )

    def _score_multi_step(self, text: str) -> DimensionScore:
        """Score based on multi-step patterns."""
        hits: Final = sum(1 for p in self._multi_step_patterns if p.search(text))
        if hits > 0:
            return DimensionScore("multiStepPatterns", 0.5, "multi-step")
        return DimensionScore("multiStepPatterns", 0, None)

    def _score_question_complexity(self, text: str) -> DimensionScore:
        """Score based on number of question marks."""
        count: Final = text.count("?")
        if count > 3:
            return DimensionScore("questionComplexity", 0.5, f"{count} questions")
        return DimensionScore("questionComplexity", 0, None)

    def classify(self, prompt: str, system_prompt: str | None = None) -> tuple[ComplexityTier, float, list[str]]:
        """Classify a prompt by complexity, discarding which rule decided the tier.

        Kept for callers that only need the tier and score; `_score_and_classify` is the
        single computation behind both, so the two can never disagree.
        """
        tier, score, signals, _cause = self._score_and_classify(prompt, system_prompt)
        return tier, score, list(signals)

    def _score_and_classify(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[ComplexityTier, float, tuple[str, ...], Literal["heuristic_scorer", "reasoning_override"]]:
        """
        Classify a prompt by complexity, reporting whether the score chose the tier.

        Args:
            prompt: The user's prompt/message.
            system_prompt: Optional system prompt for context.

        Returns:
            Tuple of (tier, score, signals) where:
            - tier: The ComplexityTier (SIMPLE, MEDIUM, COMPLEX, REASONING)
            - score: The raw weighted score
            - signals: List of triggered signals for debugging
        """
        # Score the caller's ask only. The system prompt is a per-session constant, so it
        # carries no information about how requests within a session differ, yet it
        # saturates the keyword thresholds (codePresence trips at 2 matches, which any
        # agent identity prompt clears on its first line) while spending 0.63 of the
        # dimension weight budget. That collapses the scorer's dynamic range and escalates
        # every request alike. reasoningMarkers was already scoped this way for the same
        # reason. Deployment-level model capability is expressed in tier config instead.
        user_text: Final = prompt.lower()

        # Estimate tokens
        estimated_tokens: Final = self._estimate_tokens(prompt)

        # Score all dimensions, capturing match counts where needed
        code_score, _ = self._score_keyword_match(
            user_text,
            self.code_keywords,
            "codePresence",
            "code",
            (1, 2),
            (0, 0.5, 1.0),
        )
        reasoning_score, reasoning_match_count = self._score_keyword_match(
            user_text,
            self.reasoning_keywords,
            "reasoningMarkers",
            "reasoning",
            (1, 2),
            (0, 0.7, 1.0),
        )
        technical_score, _ = self._score_keyword_match(
            user_text,
            self.technical_keywords,
            "technicalTerms",
            "technical",
            (2, 4),
            (0, 0.5, 1.0),
        )
        simple_score, _ = self._score_keyword_match(
            user_text,
            self.simple_keywords,
            "simpleIndicators",
            "simple",
            (1, 2),
            (0, -1.0, -1.0),
        )

        dimensions: Final[list[DimensionScore]] = [
            self._score_token_count(estimated_tokens),
            code_score,
            reasoning_score,
            technical_score,
            simple_score,
            self._score_multi_step(user_text),
            self._score_question_complexity(prompt),
        ]

        custom_dimensions: Final = self._score_custom_dimensions(prompt, user_text)
        signals: Final = [d.signal for d in (*dimensions, *(d for d, _ in custom_dimensions)) if d.signal is not None]

        weights: Final = self.config.dimension_weights
        weighted_score: Final = sum(d.score * weights.get(d.name, 0) for d in dimensions) + sum(
            dimension.score * weight for dimension, weight in custom_dimensions
        )

        boundaries: Final = self._effective_tier_boundaries()
        clears_override_floor: Final = weighted_score >= self._effective_reasoning_override_min_score()

        # Reuse match count from _score_keyword_match to avoid scanning twice
        if reasoning_match_count >= 2 and clears_override_floor:
            return ComplexityTier.REASONING, weighted_score, tuple(signals), "reasoning_override"

        # Map score to tier
        if weighted_score < boundaries["simple_medium"]:
            tier = ComplexityTier.SIMPLE
        elif weighted_score < boundaries["medium_complex"]:
            tier = ComplexityTier.MEDIUM
        elif weighted_score < boundaries["complex_reasoning"]:
            tier = ComplexityTier.COMPLEX
        else:
            tier = ComplexityTier.REASONING

        return tier, weighted_score, tuple(signals), "heuristic_scorer"

    # copied: complexity_router.py lines 1831-1854
    def _effective_reasoning_override_min_score(self) -> float:
        """The score a request must reach before the reasoning-marker override may promote it.

        Unset tracks the SIMPLE/MEDIUM boundary, so moving that boundary moves this floor with it
        and the override still cannot rescue a request the mapping would call SIMPLE. An explicit
        0 is a real floor, not an absent one, so the comparison is against None.
        """
        configured: Final = self.config.reasoning_override_min_score
        if configured is None:
            return self._effective_tier_boundaries()["simple_medium"]
        return configured

    def _effective_tier_boundaries(self) -> StandardLoggingRoutingDecisionTierBoundaries:
        """The tier boundaries in effect, with the documented defaults filled in.

        Shared by score-to-tier mapping and the per-request routing decision snapshot,
        so a logged decision always reflects the boundaries that actually applied.
        """
        boundaries: Final = self.config.tier_boundaries
        return dict(
            simple_medium=boundaries.get("simple_medium", 0.15),
            medium_complex=boundaries.get("medium_complex", 0.35),
            complex_reasoning=boundaries.get("complex_reasoning", 0.60),
        )
