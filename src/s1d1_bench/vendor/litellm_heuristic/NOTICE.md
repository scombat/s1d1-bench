# LiteLLM heuristic scorer, vendored

This folder holds the built-in heuristic complexity scorer of [LiteLLM](https://github.com/BerriAI/litellm), copied
verbatim so the bench runs exactly their code without installing the `litellm` package (which is not, and must not
become, a dependency of this repository).

- Upstream: <https://github.com/BerriAI/litellm>
- Tag: `v1.106.6`
- Commit: `61c6ab78393ef45683dd7d963d2a315bd845e20b`
- Source files, under `litellm/router_strategy/complexity_router/`:
  - `config.py`
  - `complexity_router.py`
- Licence: MIT, Copyright (c) 2023 Berri AI. The `LICENSE` file next to this notice is the one of the tag. Both files
  sit outside LiteLLM's `enterprise/` directory, so the MIT licence is the one that applies.

The copy is in `scorer.py`. Each copied block there is introduced by a `# copied: <file> lines X-Y` comment. Nothing
is renamed, reordered, reformatted or improved; ruff skips the whole `vendor/` folder (`extend-exclude` in
`pyproject.toml`) so it never rewrites it.

## Copied blocks

In the order they appear in `scorer.py`:

| Block | Lines |
| --- | --- |
| `ComplexityTier` | lines 47-54 of `config.py` |
| the "Default Keyword Lists" comment and `DEFAULT_CODE_KEYWORDS`, `DEFAULT_REASONING_KEYWORDS`, `DEFAULT_TECHNICAL_KEYWORDS` | lines 288-392 of `config.py` |
| `DEFAULT_SIMPLE_KEYWORDS` | lines 419-447 of `config.py` |
| `DEFAULT_DIMENSION_WEIGHTS`, `DEFAULT_TIER_BOUNDARIES`, `DEFAULT_TOKEN_THRESHOLDS` (with their section comments) | lines 450-477 of `config.py` |
| `CUSTOM_PATTERN_SCAN_CHARS` | line 865 of `config.py` |
| `_append_custom_keywords` | lines 406-411 of `complexity_router.py` |
| `_CJK_CHARACTER` | line 491 of `complexity_router.py` |
| `DimensionScore` | lines 1030-1038 of `complexity_router.py` |
| in `__init__`: `self.code_keywords`, `self.reasoning_keywords`, `self.technical_keywords`, `self.simple_keywords` | lines 1428-1434 of `complexity_router.py` |
| in `__init__`: the four `self._multi_step_patterns` regexes | lines 1465-1470 of `complexity_router.py` |
| methods `_estimate_tokens`, `_score_token_count`, `_keyword_matches`, `_score_keyword_match`, `_count_custom_hits`, `_score_custom_dimensions`, `_score_multi_step`, `_score_question_complexity`, `classify`, `_score_and_classify` | lines 1606-1820 of `complexity_router.py` |
| methods `_effective_reasoning_override_min_score`, `_effective_tier_boundaries` | lines 1831-1854 of `complexity_router.py` |

Not copied: everything else of the two files (the other tiers and classifiers, keyword tier rules, the LLM,
capability, OSS and adaptive classifiers, session affinity, logging, `DEFAULT_ESCALATION_KEYWORDS`,
`DEFAULT_TIER_MODELS`, `_is_near_tier_boundary`, `_build_routing_decision`, and the rest of `ComplexityRouter`).

## Changes to the copied code (exhaustive)

Only what is needed to import and run the copied code without `litellm`. Each change gives the original text.

1. **Imports.** `complexity_router.py` starts with `from __future__ import annotations` (line 17), kept as is, and
   imports `litellm.*`, `pydantic` and the sibling modules at lines 19-131, among them `import re` (23),
   `from itertools import accumulate, chain, islice, takewhile` (26), `from typing import TYPE_CHECKING, Any, Final,
   Literal, NamedTuple, cast` (29) and `from .config import (...)` (99), the last one standing for `ComplexityTier` and
   the `DEFAULT_*` constants. `config.py` imports `Enum` (line 13) and `Final`, `Literal` (line 15) among others. `scorer.py` replaces them by
   `import re`, `from enum import Enum`, `from itertools import chain, islice` and `from typing import Final, Literal`;
   the constants and the enumeration come from the copied blocks above. No other import is needed.
2. **The class.** Line 1320 of `complexity_router.py` reads `class ComplexityRouter(CustomLogger):`. It becomes
   `class HeuristicScorer:` (no base class, a short docstring of ours), because `CustomLogger` is a `litellm` class.
3. **`__init__`.** The original `__init__` (lines 1368-1530) takes `model_name`, `litellm_router_instance`,
   `complexity_router_config`, `default_model`, `derive_savings_baseline`, `jev_client` and builds the whole router.
   `HeuristicScorer.__init__(self)` takes no argument and keeps only the assignments of lines 1428-1434 and 1465-1470.
   Dropped with it: the docstring and the three assignments of lines 1389-1391 (`self.model_name = model_name`,
   `self.litellm_router_instance = litellm_router_instance`, `self._derive_savings_baseline =
   derive_savings_baseline`), the config parsing at lines 1393-1401 (`if complexity_router_config:` /
   `self.config = ComplexityRouterConfig.model_validate(complexity_router_config)` / `else:` / `self.config =
   ComplexityRouterConfig()` and the `default_model` override), lines 1403-1426 (JEV client, affinity hash,
   `classifier_fallback` check), lines 1444-1462 (escalation keywords, reminder markers, the semantic route layer, as
   far as the multi-step comment at 1463) and lines 1471-1530 (adaptive router, savings baseline, classifier prompts
   and response format, circuit breaker, tier success predictor, the debug log). The two comment lines 1463-1464
   above the regexes are not copied either.
4. **`self.config`.** In `ComplexityRouter`, `self.config` is a pydantic `ComplexityRouterConfig()` (line 1397,
   no override). In `HeuristicScorer.__init__` it is `self.config = _DefaultConfig()`, a plain object defined just above
   the class. Its attributes are the fields the copied code reads, each with the default of the same field in
   `config.py`, read there and not recalled:
   - `token_thresholds` = `DEFAULT_TOKEN_THRESHOLDS.copy()` (lines 1108-1111, `default_factory`)
   - `dimension_weights` = `DEFAULT_DIMENSION_WEIGHTS.copy()` (lines 1114-1117, `default_factory`)
   - `tier_boundaries` = `DEFAULT_TIER_BOUNDARIES.copy()` (lines 1089-1096, `default_factory`)
   - `reasoning_override_min_score` = `None` (lines 1098-1105, `default=None`)
   - `code_keywords`, `reasoning_keywords`, `technical_keywords`, `custom_technical_keywords`, `simple_keywords` = `None`
     (lines 1134-1158, `default=None` each)

   The only other field the copied code reads is `custom_dimensions` (its default is `()`, line 1119-1120), see the
   next point. `ComplexityRouterConfig` has validators and many more fields; none of them touches the scoring
   dimensions the above fields default to.
5. **Custom dimensions.** Lines 1435-1443 are
   ```
           self._custom_dimensions = tuple(
               _CustomDimensionMatchers(
                   dimension,
                   tuple(dict.fromkeys(keyword.lower() for keyword in dimension.keywords)),
                   tuple(re.compile(pattern, re.IGNORECASE) for pattern in dict.fromkeys(dimension.patterns)),
                   2 if dimension.scoring_mode == "match_count" else 1,
               )
               for dimension in self.config.custom_dimensions
           )
   ```
   With the default `custom_dimensions = ()` this is the empty tuple. `HeuristicScorer.__init__` writes
   `self._custom_dimensions = ()` instead, so `_CustomDimensionMatchers` and `CustomDimension` are not copied.
   `_count_custom_hits` and `_score_custom_dimensions` are copied unchanged and never find a custom dimension.
6. **Tier boundaries.** Line 1850 of `complexity_router.py`, in `_effective_tier_boundaries`:
   `        return StandardLoggingRoutingDecisionTierBoundaries(` becomes `        return dict(`. The three keyword
   arguments on the next lines (`simple_medium=`, `medium_complex=`, `complex_reasoning=`) are unchanged. The original
   is a `TypedDict` of `litellm.types.utils`, which builds a plain `dict` with the same keys.
7. **Annotations that name something not copied.** `_count_custom_hits` (`_CustomDimensionMatchers`) and
   `_effective_tier_boundaries` (`-> StandardLoggingRoutingDecisionTierBoundaries`) keep the annotation of the original.
   `from __future__ import annotations` makes annotations lazy, so the names are never looked up and the copied lines
   stay byte-identical. (The names stay undefined in `scorer.py`; this is why ruff skips the folder.)
8. **Additions of ours, not copied.** The module docstring of `scorer.py`, the `# copied: ...` marker comments, the
   blank lines between blocks, the `_DefaultConfig` class, the `HeuristicScorer` class line, its docstring and
   `__init__`'s `self.config = _DefaultConfig()` and `self._custom_dimensions = ()` lines. `__init__.py` (the `Decision`
   tuple, `classify`, `TAG`, `COMMIT`) is ours too and is not part of the copy.

Each copied range can be compared with the upstream text (`gh api -H "Accept: application/vnd.github.raw"
"repos/BerriAI/litellm/contents/litellm/router_strategy/complexity_router/<file>?ref=v1.106.6"`): apart from the
changes above, `scorer.py` contains it line for line.

## Identity proof

`tests/fixtures/litellm_heuristic/v1.106.6.jsonl` (368 rows: the 360 prompts of `data/prompts/base-v1.jsonl` and the
eight prompts of `tests/fixtures/litellm_heuristic/edge.jsonl`) holds what the real package returns for them. It was
made once, in a throwaway environment, by `tests/fixtures/litellm_heuristic/make_fixture.py`, which calls
`ComplexityRouter._score_and_classify(prompt, None)` of the real `litellm.router_strategy.complexity_router`:

```
uv run --no-project --with litellm==1.106.6 python tests/fixtures/litellm_heuristic/make_fixture.py \
    data/prompts/base-v1.jsonl tests/fixtures/litellm_heuristic/edge.jsonl \
    > tests/fixtures/litellm_heuristic/v1.106.6.jsonl
```

The script printed `litellm 1.106.6` on stderr (the version imported, from `importlib.metadata`). `litellm` is not in
`pyproject.toml` or `uv.lock`. `test_the_vendored_scorer_reproduces_the_real_package` checks that the vendored scorer
returns the same tier, score (`==`, not approximately), signals and cause on every row.
