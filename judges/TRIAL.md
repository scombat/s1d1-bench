# Trial of three local judges on 50 prompts

Run on 2026-10-29 on the bench Mac (M4 Max, 128 GB), `mlx-lm` 0.32.2, one model process at a time, greedy decoding
(temperature 0), `--max-tokens 1024`, rubric `v7-single`. The three models, their revisions and how `bench judge`
switches their reasoning off or caps it are in [MODELS.md](MODELS.md). The six folders are `judges/trial-<family>-off`
and `judges/trial-<family>-cap512` for `gemma4`, `qwen3` and `gptoss`. Every folder passes `bench validate`.

## The 50 prompts

`data/prompts/trial-50.jsonl` (sha256 `56d7b8083d59ccf63f520b5203cab119b9d7558794d4b742b2993302834a2bdd`), exported
with `bench export --pool data/pool/pool.parquet --select data/prompts/trial-50.select.txt --langs eng --name trial-50
--seed 0` (no handwritten prompts). Composition: **3 OASST2, 24 Dolly, 23 Arena-Hard**, all English.

- Why OASST2 is so small: `data/pool/candidates-eng.jsonl` holds 603 candidates, 300 Dolly, 300 Arena-Hard and only 3
  OASST2. The exclusion list (`data/exclusions/s1d1-seen-keys.txt`) covers almost every OASST2 prompt, because it
  includes the `dataset/interim` keys. This is a known property of the pool, not something this trial changed.
- Draw rule: all 3 OASST2 candidates, then 24 Dolly and 23 Arena-Hard drawn with `random.Random(0)`. Per source, the
  candidate ids are sorted first, then `rng.sample` is called (Dolly first, then Arena-Hard, on the same generator).
  The source of an id comes from `data/pool/pool.parquet` (`prompt_key` to `source`).
- The 50 keys, one per line, are in `data/prompts/trial-50.select.txt`, so the draw can be checked without rerunning
  it. `bench export` shuffles them (seed 0) and writes the ids `<key>-eng`.

## Per judge folder

All figures come from the last line of each id in `log.jsonl` (each folder has exactly 50 lines, no resume was
needed). Tokens/s is `completion_tokens / wall_s` per prompt, median over the 50 prompts (the figure `bench judge`
prints). `wall_s` covers prompt processing and decoding, not model loading. The prompt is 1 858 to 3 296 tokens (the
rubric is about 1 800 of them). Estimate for 400 prompts = mean `wall_s` x 400.

| folder | ok | json_error | truncated | error | median tokens/s | median s/prompt | mean s/prompt | total wall (sum) | 400 prompts (est.) | median / max completion tokens | prompts above 512 completion tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trial-gptoss-off | 50 | 0 | 0 | 0 | 13.2 | 1.3 | 1.34 | 67 s | 8.9 min | 17 / 18 | 0 |
| trial-gptoss-cap512 | 50 | 0 | 0 | 0 | 43.3 | 1.8 | 1.91 | 95 s | 12.7 min | 80 / 208 | 0 |
| trial-qwen3-off | 50 | 0 | 0 | 0 | 2.3 | 9.2 | 9.58 | 479 s (8.0 min) | 1.06 h | 21 / 22 | 0 |
| trial-qwen3-cap512 | 50 | 0 | 0 | 0 | 10.2 | 21.3 | 23.72 | 1 186 s (19.8 min) | 2.64 h | 225 / 533 | 2 |
| trial-gemma4-off | 50 | 0 | 0 | 0 | 1.8 | 11.5 | 11.97 | 599 s (10.0 min) | 1.33 h | 21 / 21 | 0 |
| trial-gemma4-cap512 | 50 | 0 | 0 | 0 | 8.9 | 35.7 | 38.63 | 1 932 s (32.2 min) | 4.29 h | 336 / 540 | 8 |

The share of `json_error` and `truncated` is 0 % in all six folders; no retry was needed (`retries` is 0 on every
line) and no row is missing from any `labels.jsonl`. The sum of the six total wall times is 4 358 s (72.6 min); the
chain ran from 00:32:55 to 01:46:14 including model loading. Against the 10-prompt figures of MODELS.md, the median
seconds per prompt are the same or higher here: Gemma off 11.5 s on both, Qwen off 9.2 s against 10.0 s, Qwen cap:512
21.3 s against 20.0 s, gpt-oss cap:512 1.8 s against 1.7 s, Gemma cap:512 35.7 s against 31.6 s (the largest gap,
+13 %).

## Label distributions

Counts per complexity level (`null` never occurred) and the number of prompts labelled `needs_tools: true`.

| folder | null | trivial | low | medium | high | expert | needs_tools true |
|---|---|---|---|---|---|---|---|
| trial-gptoss-off | 0 | 5 | 31 | 13 | 1 | 0 | 0 |
| trial-gptoss-cap512 | 0 | 6 | 34 | 9 | 1 | 0 | 0 |
| trial-qwen3-off | 0 | 5 | 25 | 17 | 3 | 0 | 2 |
| trial-qwen3-cap512 | 0 | 4 | 26 | 17 | 3 | 0 | 0 |
| trial-gemma4-off | 0 | 4 | 28 | 17 | 1 | 0 | 1 |
| trial-gemma4-cap512 | 0 | 2 | 30 | 15 | 3 | 0 | 1 |

No judge used `expert` and `needs_tools` is false for 47 to 50 of the 50 prompts in every folder (the ten `expert`
handwritten prompts are not in this set). So `needs_tools` agreement (96 to 100 %) and the within-one-tier rate
(98 to 100 %) say little here: almost all labels sit on `low` and `medium`.

## Agreement (`bench agree judges/trial-*`)

All pairs are over the same 50 ids. "tier exact" counts `null` = `null` as agreement; "tier ±1" is a distance of at
most one level.

| judge A | judge B | n | tier exact | tier ±1 | task | needs_tools |
|---|---|---|---|---|---|---|
| trial-gemma4-cap512 | trial-gemma4-off | 50 | 88.0% | 100.0% | 90.0% | 100.0% |
| trial-gemma4-cap512 | trial-gptoss-cap512 | 50 | 70.0% | 98.0% | 62.0% | 98.0% |
| trial-gemma4-cap512 | trial-gptoss-off | 50 | 78.0% | 100.0% | 72.0% | 98.0% |
| trial-gemma4-cap512 | trial-qwen3-cap512 | 50 | 80.0% | 100.0% | 80.0% | 98.0% |
| trial-gemma4-cap512 | trial-qwen3-off | 50 | 82.0% | 100.0% | 78.0% | 98.0% |
| trial-gemma4-off | trial-gptoss-cap512 | 50 | 72.0% | 100.0% | 60.0% | 98.0% |
| trial-gemma4-off | trial-gptoss-off | 50 | 78.0% | 100.0% | 66.0% | 98.0% |
| trial-gemma4-off | trial-qwen3-cap512 | 50 | 88.0% | 100.0% | 84.0% | 98.0% |
| trial-gemma4-off | trial-qwen3-off | 50 | 90.0% | 100.0% | 82.0% | 98.0% |
| trial-gptoss-cap512 | trial-gptoss-off | 50 | 82.0% | 100.0% | 86.0% | 100.0% |
| trial-gptoss-cap512 | trial-qwen3-cap512 | 50 | 64.0% | 100.0% | 72.0% | 100.0% |
| trial-gptoss-cap512 | trial-qwen3-off | 50 | 62.0% | 100.0% | 70.0% | 96.0% |
| trial-gptoss-off | trial-qwen3-cap512 | 50 | 70.0% | 100.0% | 78.0% | 100.0% |
| trial-gptoss-off | trial-qwen3-off | 50 | 68.0% | 100.0% | 76.0% | 96.0% |
| trial-qwen3-cap512 | trial-qwen3-off | 50 | 90.0% | 100.0% | 94.0% | 96.0% |

All three judges give the same tier on 34 of 50 prompts with thinking off and on 29 of 50 with cap:512; the same task
on 32 of 50 (off) and 29 of 50 (cap:512).

## Conclusions

- Which judges hold up: all three return readable JSON on 50 of 50 prompts in both modes (0 json_error, 0 truncated),
  and Gemma and Qwen agree with each other on tier at 90.0 % (off) and 80.0 % (cap:512, the two cap runs), while
  gpt-oss agrees with Qwen on tier at 68.0 % (off) and 64.0 % (cap:512) and with Gemma at 78.0 % (off) and 70.0 %
  (cap:512); gpt-oss labels `low` more often (31 to 34 of 50) and `high` less often (1 of 50) than the other two.
- Which one to replace, if any: by agreement, gpt-oss-20b is the judge furthest from the other two (its cross-judge
  pairs are at 62.0 to 78.0 % on tier and 60.0 to 78.0 % on task, the Gemma/Qwen pairs at 80.0 to 90.0 % on tier and
  78.0 to 84.0 % on task), and it is also the one for which "off" is a prefilled empty reasoning
  channel at `reasoning_effort=low` rather than a template switch; by cost, Gemma at cap:512 is 4.29 h per 400 prompts
  against 2.64 h for Qwen and 0.21 h for gpt-oss. This trial gives no ground truth, so it says who disagrees, not who is
  right; the decision is the user's.
- Does cap:512 change agreement against off: yes, a little, and not in a uniform direction. Same judge, off against
  cap:512, tier exact: Gemma 88.0 %, Qwen 90.0 %, gpt-oss 82.0 %; the labels change tier on 6 of 50 (Gemma: 5 higher,
  1 lower), 5 of 50 (Qwen: 3 higher, 2 lower) and 9 of 50 prompts (gpt-oss: 2 higher, 7 lower). Between judges, the
  tier agreement of the cap:512 runs is lower than that of the off runs for all three pairs (Gemma/Qwen 80.0 % against
  90.0 %, Gemma/gpt-oss 70.0 % against 78.0 %, gpt-oss/Qwen 64.0 % against 68.0 %). As MODELS.md says, for Qwen and
  gpt-oss `cap:512` means `reasoning_effort=low` and the cap rarely binds (2 of 50 Qwen prompts and 0 of 50 gpt-oss
  prompts went above 512 completion tokens); for Gemma it binds more often (8 of 50 prompts above 512 completion
  tokens). 50 prompts and a single run per setting: a one-prompt change is 2 points.

## Things worth knowing

- The measured speeds are close to MODELS.md (10-prompt runs; see above), and no run produced an unreadable answer, so nothing here was
  retried or repaired by hand.
- Qwen went above 512 completion tokens on 2 of 50 prompts in the cap:512 run, whereas the 10-prompt runs never did
  (max 285); the maximum here is 533.
- This trial used the prompts of `trial-50` only; the single-run-per-setting design means that differences of a few
  points between pairs are within what one prompt can move.
