# Trial of three local judges on 50 prompts

Run on 2026-10-29 on the bench Mac (M4 Max, 128 GB), `mlx-lm` 0.32.2, one model process at a time, greedy decoding
(temperature 0), `--max-tokens 1024`, rubric `v7-single`. The three models, their revisions and how `bench judge`
switches their reasoning off or caps it are in [MODELS.md](MODELS.md). The six folders are `judges/trial-<family>-off`
and `judges/trial-<family>-cap512` for `gemma4`, `qwen3` and `gptoss`. Five of the six folders pass `bench validate`;
`trial-gptoss-off` is missing one label (see below).

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
| trial-gptoss-off | 49 | 1 | 0 | 0 | 13.6 | 1.2 | 1.29 | 65 s | 8.6 min | 17 / 30 | 0 |
| trial-gptoss-cap512 | 50 | 0 | 0 | 0 | 43.7 | 1.8 | 1.87 | 93 s | 12.4 min | 77 / 221 | 0 |
| trial-qwen3-off | 50 | 0 | 0 | 0 | 2.2 | 9.3 | 9.51 | 476 s (7.9 min) | 1.06 h | 21 / 22 | 0 |
| trial-qwen3-cap512 | 50 | 0 | 0 | 0 | 10.3 | 21.5 | 22.48 | 1 124 s (18.7 min) | 2.50 h | 216 / 533 | 2 |
| trial-gemma4-off | 50 | 0 | 0 | 0 | 1.8 | 11.3 | 11.61 | 581 s (9.7 min) | 1.29 h | 21 / 21 | 0 |
| trial-gemma4-cap512 | 50 | 0 | 0 | 0 | 8.6 | 33.2 | 37.33 | 1 866 s (31.1 min) | 4.15 h | 307 / 535 | 8 |

**One unreadable answer.** `trial-gptoss-off` has `json_error` on one prompt (2 %, far under the 10 % bar): id
`ec5d4c815aa41d94-eng`, "behaving like an expert, I want you to help me design a radar that can detect a 155mm
artillery shell ...". Both attempts returned `{"complexity":"high","task":"design","needs_tools":false}`; `design` is
not one of the 19 task names. The run was resumed once with `--resume` and gave the same answer (greedy decoding), so
the log has two lines for that id, and `labels.jsonl` has 49 rows. As the format requires a label for every prompt,
`bench validate judges/trial-gptoss-off` reports `missing 1 id(s)` and the CI loop over `judges/*/` fails on that
folder. The judge was not replaced and the validator was not changed; this is left to the user (accept incomplete
folders with a recorded failure, or drop that folder). The `bench agree` rows that involve this folder use 49 prompts.
No label was filled in by hand. The other five folders have no unreadable answer and no retry.

The sum of the six total wall times is 4 205 s (70.1 min); the chain ran from 09:30 to about 10:41 including model
loading. Against the 10-prompt figures of MODELS.md, the median seconds per prompt are close: Gemma off 11.3 s against
11.5 s, Qwen off 9.3 s against 10.0 s, Qwen cap:512 21.5 s against 20.0 s, gpt-oss cap:512 1.8 s against 1.7 s, Gemma
cap:512 33.2 s against 31.6 s.

## Label distributions

Counts per complexity level and the number of prompts labelled `needs_tools: true`. This time Qwen and Gemma use `null`
once each (in both modes), on the same prompt.

| folder | null | trivial | low | medium | high | expert | needs_tools true |
|---|---|---|---|---|---|---|---|
| trial-gptoss-off | 0 | 8 | 32 | 9 | 0 | 0 | 0 |
| trial-gptoss-cap512 | 0 | 15 | 22 | 12 | 1 | 0 | 0 |
| trial-qwen3-off | 1 | 6 | 27 | 13 | 3 | 0 | 1 |
| trial-qwen3-cap512 | 1 | 7 | 23 | 15 | 4 | 0 | 1 |
| trial-gemma4-off | 1 | 6 | 30 | 11 | 2 | 0 | 0 |
| trial-gemma4-cap512 | 1 | 5 | 32 | 10 | 2 | 0 | 0 |

No judge used `expert` and `needs_tools` is true for 0 or 1 of the 50 prompts in every folder (the ten `expert`
handwritten prompts are not in this set). So `needs_tools` agreement (98 to 100 %) and the within-one-tier rate (96 to
100 %) say little here: almost all labels sit on `trivial`, `low` and `medium`.

## Agreement (`bench agree judges/trial-*`)

Pairs that do not involve `trial-gptoss-off` are over the same 50 ids; those that do are over 49. "tier exact" counts
`null` = `null` as agreement; "tier ±1" is a distance of at most one level.

| judge A | judge B | n | tier exact | tier ±1 | task | needs_tools |
|---|---|---|---|---|---|---|
| trial-gemma4-cap512 | trial-gemma4-off | 50 | 84.0% | 100.0% | 90.0% | 100.0% |
| trial-gemma4-cap512 | trial-gptoss-cap512 | 50 | 52.0% | 96.0% | 74.0% | 100.0% |
| trial-gemma4-cap512 | trial-gptoss-off | 49 | 65.3% | 98.0% | 73.5% | 100.0% |
| trial-gemma4-cap512 | trial-qwen3-cap512 | 50 | 74.0% | 100.0% | 86.0% | 98.0% |
| trial-gemma4-cap512 | trial-qwen3-off | 50 | 84.0% | 100.0% | 88.0% | 98.0% |
| trial-gemma4-off | trial-gptoss-cap512 | 50 | 54.0% | 98.0% | 68.0% | 100.0% |
| trial-gemma4-off | trial-gptoss-off | 49 | 65.3% | 98.0% | 67.3% | 100.0% |
| trial-gemma4-off | trial-qwen3-cap512 | 50 | 78.0% | 100.0% | 84.0% | 98.0% |
| trial-gemma4-off | trial-qwen3-off | 50 | 88.0% | 100.0% | 86.0% | 98.0% |
| trial-gptoss-cap512 | trial-gptoss-off | 49 | 59.2% | 100.0% | 79.6% | 100.0% |
| trial-gptoss-cap512 | trial-qwen3-cap512 | 50 | 56.0% | 98.0% | 76.0% | 98.0% |
| trial-gptoss-cap512 | trial-qwen3-off | 50 | 54.0% | 98.0% | 70.0% | 98.0% |
| trial-gptoss-off | trial-qwen3-cap512 | 49 | 59.2% | 98.0% | 75.5% | 98.0% |
| trial-gptoss-off | trial-qwen3-off | 49 | 65.3% | 98.0% | 69.4% | 98.0% |
| trial-qwen3-cap512 | trial-qwen3-off | 50 | 90.0% | 100.0% | 90.0% | 100.0% |

On the 49 prompts that all six folders label, the three judges give the same tier on 29 of 49 prompts with thinking off
and on 20 of 49 with cap:512; the same task on 30 of 49 (off) and 33 of 49 (cap:512).

## Conclusions

- Which judges hold up: Gemma and Qwen return readable JSON on 50 of 50 prompts in both modes; gpt-oss does on 99 of
  100 (one out-of-enum task, deterministic, see above). Gemma and Qwen agree with each other on tier at 88.0 % (off)
  and 74.0 % (cap:512, the two cap runs), while gpt-oss agrees with Qwen at 65.3 % (off) and 56.0 % (cap:512) and with
  Gemma at 65.3 % (off) and 52.0 % (cap:512); gpt-oss labels `trivial` and `low` more often and `high` less often than
  the other two.
- Which one to replace, if any: by agreement, gpt-oss-20b is again the judge furthest from the other two (its
  cross-judge pairs are at 52.0 to 65.3 % on tier and 67.3 to 76.0 % on task, the Gemma/Qwen pairs at 74.0 to 88.0 % on
  tier and 84.0 to 88.0 % on task), and it is also the one for which "off" is a prefilled empty reasoning channel at
  `reasoning_effort=low` rather than a template switch, and the only one with an unreadable answer. By cost, Gemma at
  cap:512 is 4.15 h per 400 prompts against 2.50 h for Qwen and 0.21 h for gpt-oss. This trial gives no ground truth,
  so it says who disagrees, not who is right; the decision is the user's.
- Does cap:512 change agreement against off: yes, and more than in the first draw. Same judge, off against cap:512,
  tier exact: Gemma 84.0 %, Qwen 90.0 %, gpt-oss 59.2 % (49 prompts); the labels change tier on 8 of 50 (Gemma: 4
  higher, 4 lower), 5 of 50 (Qwen: 4 higher, 1 lower) and 20 of 49 prompts (gpt-oss: 8 higher, 12 lower; it moves
  `trivial` from 8 to 15 prompts). Between judges, the tier agreement of the cap:512 runs is lower than that of the off
  runs for all three pairs (Gemma/Qwen 74.0 % against 88.0 %, Gemma/gpt-oss 52.0 % against 65.3 %, gpt-oss/Qwen
  56.0 % against 65.3 %). As MODELS.md says, for Qwen and gpt-oss `cap:512` means `reasoning_effort=low` and the cap
  rarely binds (2 of 50 Qwen prompts and 0 of 50 gpt-oss prompts went above 512 completion tokens), so for gpt-oss the
  difference between the two columns is the difference between "no reasoning" and "low reasoning", not a cap effect;
  for Gemma the cap binds more often (8 of 50 prompts above 512 completion tokens). 50 prompts and a single run per
  setting: a one-prompt change is 2 points.

## Things worth knowing

- The measured speeds are close to MODELS.md (10-prompt runs; see above).
- Qwen went above 512 completion tokens on 2 of 50 prompts in the cap:512 run, whereas the 10-prompt runs never did
  (max 285); the maximum here is 533.
- This trial used the prompts of `trial-50` only; the single-run-per-setting design means that differences of a few
  points between pairs are within what one prompt can move.
