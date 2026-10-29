# Trial of three local judges on 50 prompts

Run on 2026-10-29 on the bench Mac (M4 Max, 128 GB), `mlx-lm` 0.32.2, one model process at a time, greedy decoding
(temperature 0), `--max-tokens 1024`, rubric `v7-single`. The three models, their revisions and how `bench judge`
switches their reasoning off or caps it are in [MODELS.md](MODELS.md). Six runs were made (`off` and `cap512` for
`gemma4`, `qwen3` and `gptoss`); five folders are kept, `judges/trial-<family>-off` and `-cap512` for `gemma4` and
`qwen3`, and `judges/trial-gptoss-cap512`. The sixth, `trial-gptoss-off`, was removed (see below). Every kept folder
passes `bench validate`.

## The 50 prompts

`data/prompts/trial-50.jsonl` (sha256 `ce182edb6197a3df32c7ed9e49f329762b0a24743f149cff09f70da4542abb94`), exported
with `bench export --pool data/pool/pool.parquet --select data/prompts/trial-50.select.txt --langs eng --name trial-50
--seed 0` (no handwritten prompts). Composition: **17 Dolly, 17 OASST2, 16 Arena-Hard**, all English (the 17/17/16
split of the plan).

- This is the second draw. The first one (3 OASST2, 24 Dolly, 23 Arena-Hard) was made with an exclusion list that also
  held the `dataset/interim` keys, which removed almost every OASST2 prompt from the pool. The list now holds only
  `dataset/final/*` (112,947 keys), the pool keeps 6,733 OASST2 prompts, and the whole trial was rerun on the new draw.
  The figures below replace those of the first draw; none of them carries over.
- Draw rule: `random.Random(0)`; per source the candidate ids (`data/pool/candidates-eng.jsonl`, 300 per source) are
  sorted first, then `rng.sample` is called in the order Dolly (17), OASST2 (17), Arena-Hard (16) on the same
  generator. The source of an id comes from `data/pool/pool.parquet` (`prompt_key` to `source`).
- The 50 keys, one per line, are in `data/prompts/trial-50.select.txt`, so the draw can be checked without rerunning
  it. `bench export` shuffles them (seed 0) and writes the ids `<key>-eng`.

## Per judge folder

All figures come from the last line of each id in `log.jsonl`. Tokens/s is `completion_tokens / wall_s` per prompt,
median over the 50 prompts (the figure `bench judge` prints). `wall_s` covers prompt processing and decoding, not model
loading. The first attempt of every prompt is 1 857 to 2 997 tokens (the rubric is about 1 800 of them). Estimate for
400 prompts = mean `wall_s` x 400. A third of the trial mix is Arena-Hard (16 of 50), which is long, so the estimate
for the real 400-prompt set may differ.

| folder | ok | json_error | truncated | error | median tokens/s | median s/prompt | mean s/prompt | total wall (sum) | 400 prompts (est.) | median / max completion tokens | prompts above 512 completion tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trial-gptoss-cap512 | 50 | 0 | 0 | 0 | 43.7 | 1.8 | 1.87 | 93 s | 12.4 min | 77 / 221 | 0 |
| trial-qwen3-off | 50 | 0 | 0 | 0 | 2.2 | 9.3 | 9.51 | 476 s (7.9 min) | 1.06 h | 21 / 22 | 0 |
| trial-qwen3-cap512 | 50 | 0 | 0 | 0 | 10.3 | 21.5 | 22.48 | 1 124 s (18.7 min) | 2.50 h | 216 / 533 | 2 |
| trial-gemma4-off | 50 | 0 | 0 | 0 | 1.8 | 11.3 | 11.61 | 581 s (9.7 min) | 1.29 h | 21 / 21 | 0 |
| trial-gemma4-cap512 | 50 | 0 | 0 | 0 | 8.6 | 33.2 | 37.33 | 1 866 s (31.1 min) | 4.15 h | 307 / 535 | 8 |

**One unreadable answer, and the folder removed.** The gpt-oss run with thinking off had a `json_error` on one prompt
(1 of 50, 2 %, far under the 10 % bar): id `ec5d4c815aa41d94-eng`, "behaving like an expert, I want you to help me
design a radar that can detect a 155mm artillery shell ...". Both attempts returned
`{"complexity":"high","task":"design","needs_tools":false}`; `design` is not one of the 19 task names. A `--resume`
gave the same answer (greedy decoding). The judge was not replaced, no label was filled in, and the validator was not
changed; as the format requires a label for every prompt, the folder (49 labels) failed `bench validate`, and the user
decided to remove `judges/trial-gptoss-off` from the repository. The figures of that run are not in the tables below
(for the record: 49 ok, median 1.2 s per prompt, 13.6 tokens/s, 17 completion tokens, about 8.6 min per 400 prompts).
The five kept folders have no unreadable answer and no retry.

The sum of the five kept total wall times is 4 140 s (69.0 min); the chain of six runs took from 09:30 to about 10:41 including model
loading. Against the 10-prompt figures of MODELS.md, the median seconds per prompt are close: Gemma off 11.3 s against
11.5 s, Qwen off 9.3 s against 10.0 s, Qwen cap:512 21.5 s against 20.0 s, gpt-oss cap:512 1.8 s against 1.7 s, Gemma
cap:512 33.2 s against 31.6 s.

## Label distributions

Counts per complexity level and the number of prompts labelled `needs_tools: true`. This time Qwen and Gemma use `null`
once each (in both modes), on the same prompt.

| folder | null | trivial | low | medium | high | expert | needs_tools true |
|---|---|---|---|---|---|---|---|
| trial-gptoss-cap512 | 0 | 15 | 22 | 12 | 1 | 0 | 0 |
| trial-qwen3-off | 1 | 6 | 27 | 13 | 3 | 0 | 1 |
| trial-qwen3-cap512 | 1 | 7 | 23 | 15 | 4 | 0 | 1 |
| trial-gemma4-off | 1 | 6 | 30 | 11 | 2 | 0 | 0 |
| trial-gemma4-cap512 | 1 | 5 | 32 | 10 | 2 | 0 | 0 |

No judge used `expert` and `needs_tools` is true for 0 or 1 of the 50 prompts in every folder (the ten `expert`
handwritten prompts are not in this set). So `needs_tools` agreement (98 to 100 %) and the within-one-tier rate (96 to
100 %) say little here: almost all labels sit on `trivial`, `low` and `medium`.

## Agreement (`bench agree judges/trial-*`)

All pairs are over the same 50 ids. "tier exact" counts `null` = `null` as agreement; "tier ±1" is a distance of at most
one level.

| judge A | judge B | n | tier exact | tier ±1 | task | needs_tools |
|---|---|---|---|---|---|---|
| trial-gemma4-cap512 | trial-gemma4-off | 50 | 84.0% | 100.0% | 90.0% | 100.0% |
| trial-gemma4-cap512 | trial-gptoss-cap512 | 50 | 52.0% | 96.0% | 74.0% | 100.0% |
| trial-gemma4-cap512 | trial-qwen3-cap512 | 50 | 74.0% | 100.0% | 86.0% | 98.0% |
| trial-gemma4-cap512 | trial-qwen3-off | 50 | 84.0% | 100.0% | 88.0% | 98.0% |
| trial-gemma4-off | trial-gptoss-cap512 | 50 | 54.0% | 98.0% | 68.0% | 100.0% |
| trial-gemma4-off | trial-qwen3-cap512 | 50 | 78.0% | 100.0% | 84.0% | 98.0% |
| trial-gemma4-off | trial-qwen3-off | 50 | 88.0% | 100.0% | 86.0% | 98.0% |
| trial-gptoss-cap512 | trial-qwen3-cap512 | 50 | 56.0% | 98.0% | 76.0% | 98.0% |
| trial-gptoss-cap512 | trial-qwen3-off | 50 | 54.0% | 98.0% | 70.0% | 98.0% |
| trial-qwen3-cap512 | trial-qwen3-off | 50 | 90.0% | 100.0% | 90.0% | 100.0% |

Gemma off and Qwen off give the same tier on 44 of 50 prompts and the same task on 43 of 50. The three cap:512 runs
give the same tier on 21 of 50 prompts and the same task on 34 of 50.

## Conclusions

- Which judges hold up: Gemma and Qwen return readable JSON on 50 of 50 prompts in both modes; gpt-oss does on 99 of
  100 over its two runs (one out-of-enum task, deterministic, see above). Gemma and Qwen agree with each other on tier
  at 88.0 % (off) and 74.0 % (cap:512), while gpt-oss (cap:512) agrees with Qwen at 54.0 to 56.0 % and with Gemma at
  52.0 to 54.0 %; gpt-oss labels `trivial` more often (15 of 50 against 5 to 7) and `high` less often than the other
  two.
- Which one to replace, if any: by agreement, gpt-oss-20b is the judge furthest from the other two (its cross-judge
  pairs are at 52.0 to 56.0 % on tier and 68.0 to 76.0 % on task, the Gemma/Qwen pairs at 74.0 to 88.0 % on tier and
  84.0 to 88.0 % on task). It is also the one for which "off" is a prefilled empty reasoning channel at
  `reasoning_effort=low` rather than a template switch, and the only one with an unreadable answer. By cost, Gemma at
  cap:512 is 4.15 h per 400 prompts against 2.50 h for Qwen and 0.21 h for gpt-oss. This trial gives no ground truth,
  so it says who disagrees, not who is right; the decision is the user's.
- Does cap:512 change agreement against off: for Gemma and Qwen, a little. Same judge, off against cap:512, tier
  exact: Gemma 84.0 %, Qwen 90.0 %; the labels change tier on 8 of 50 (Gemma: 4 higher, 4 lower) and 5 of 50 prompts
  (Qwen: 4 higher, 1 lower). Between those two judges, the tier agreement of the cap:512 runs is lower than that of the
  off runs (74.0 % against 88.0 %). As MODELS.md says, for Qwen and gpt-oss `cap:512` means `reasoning_effort=low` and
  the cap rarely binds (2 of 50 Qwen prompts and 0 of 50 gpt-oss prompts went above 512 completion tokens); for Gemma
  it binds more often (8 of 50 prompts above 512 completion tokens). gpt-oss has no off-against-cap comparison any
  more, since its `off` folder was removed. 50 prompts and a single run per setting: a one-prompt change is 2 points.

## Things worth knowing

- The measured speeds are close to MODELS.md (10-prompt runs; see above).
- Qwen went above 512 completion tokens on 2 of 50 prompts in the cap:512 run, whereas the 10-prompt runs never did
  (max 285); the maximum here is 533.
- This trial used the prompts of `trial-50` only; the single-run-per-setting design means that differences of a few
  points between pairs are within what one prompt can move.
