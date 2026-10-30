"""Run the REAL litellm complexity scorer over prompt files and print one JSON line per prompt.

This script made `v1.106.6.jsonl`, the reference the vendored copy is tested against. It is run once, in a throwaway
environment where the real package is installed (litellm is never a dependency of this repository):

    uv run --no-project --with litellm==1.106.6 python tests/fixtures/litellm_heuristic/make_fixture.py \\
        data/prompts/base-v1.jsonl tests/fixtures/litellm_heuristic/edge.jsonl \\
        > tests/fixtures/litellm_heuristic/v1.106.6.jsonl

Each argument is a JSONL file of `{"id": ..., "prompt": ...}` rows. For each row the script calls
`ComplexityRouter._score_and_classify(prompt, None)` of `litellm.router_strategy.complexity_router`, built with its
default configuration, and prints `{"id", "tier", "score", "signals", "cause"}` (tier is the enum value, e.g.
"SIMPLE"). The imported litellm version is printed on stderr.

The constructor wants a `litellm_router_instance`; `_score_and_classify` never reads it (it only scores the text with
`self.config` and the keyword lists), so `None` is passed. Nothing from `s1d1_bench` is imported.
"""

import importlib.metadata
import json
import sys

from litellm.router_strategy.complexity_router import ComplexityRouter


def main(paths: list[str]) -> None:
    print(f"litellm {importlib.metadata.version('litellm')}", file=sys.stderr)
    router = ComplexityRouter(model_name="fixture", litellm_router_instance=None)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                tier, score, signals, cause = router._score_and_classify(row["prompt"], None)
                record = {"id": row["id"], "tier": tier.value, "score": score, "signals": list(signals), "cause": cause}
                print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
