# s1d1-bench

An open, replayable bench of how well a prompt classifier agrees with local LLM judges on the v7 rubric (complexity
tier, score, task, `needs_tools`). On `base-v1`, 90 base prompts in four languages (360 ids), `s1d1-full-int8` agrees
with the median of three judges at QWK 0.807 [0.734, 0.852]; the inter-judge ceiling is 0.774, the prompt-length
witness 0.543, Laya 0.405 (english) and 0.119 (multilingual), LiteLLM's heuristic 0.089. The judges are the instrument,
not competitors; no "better than" claim is made unless the paired difference passes the 2 σ rule. Licences: code
Apache-2.0 (one MIT copy), data CC-BY-4.0 or its source's.

The model can be tried in the browser, where it runs in the page, in the
[demo Space](https://huggingface.co/spaces/scombat/s1d1-router-demo).

Below are what the bench does not measure, the [results](#results-base-v1-2026-10-30), the commands and how to replay. The full report is [`report/base-v1/index.md`](report/base-v1/index.md): its first section is in English, the
rest in French. The decisions cited as `docs/NN` are in [`docs/`](docs/README.md); those decision documents are in
French.

## What the bench measures, and what it does not

- **Measured.** The agreement of a classifier's four outputs with the median of the judges: complexity tier, score, task
  and `needs_tools`, overall, and for the complexity tier also by language, by source, by origin and by truth tier. The
  random and prompt-length witnesses give the floor, each judge's agreement with the others gives the ceiling.
- **Not measured: the rubric.** Whether the rubric itself matches "which model is enough" belongs to another bench, the
  one on answers, which is not covered here.
- **Not measured: routing.** The mapping from score to model is the deployer's business.
- **Not measured: the real distribution.** The set is built by cells, `expert` will be the least populated; agentic and
  multi-turn prompts are absent, the set is made of first prompts.
- **Not measured: other languages.** Only the four chosen languages are covered.

### Where this bench comes from

Before this bench, s1d1 was run on RouteLLM's public evaluation (lm-sys,
[arXiv 2406.18665](https://arxiv.org/abs/2406.18665): MT-Bench, GSM8K, MMLU; GPT-4 against Mixtral 8x7B; APGR metric),
after reproducing exactly the nine published figures of their BERT router. Depending on the evaluation set, s1d1 scored
between 0.40 and 0.59 APGR. A trivial witness, the prompt length in characters, scored 0.56 to 0.60: 0.598 on GSM8K and
0.583 on MMLU, where RouteLLM's BERT router scores 0.531 and 0.572. On MT-Bench (72 questions), everything fell within
the noise, except the BERT trained on the GPT-4 against Mixtral preference, which is exactly what that evaluation
measures. It answers the question "on this question, does GPT-4 beat Mixtral?": a gap between two models on an English
pair, not the complexity of a prompt. A prompt that is hard for both models earns nothing there, whereas s1d1 predicts
the answering effort as judged by an LLM. Hence the present bench, one of its own: witnesses (chance, length, constant),
a ceiling made of the judges themselves, bootstrap intervals and four languages, that is `base-v1`.

## Results (`base-v1`, 2026-10-30)

The sentence of docs/32 § 5, filled with the figures of the `base-v1` report (360 ids, 90 base prompts):

> Over 360 ids, that is 90 base prompts in 4 languages, `s1d1-full-int8` agrees with the median of 3 judges at QWK 0.807
> [0.734, 0.852]; the inter-judge ceiling is 0.774 [0.718, 0.815] (each main judge against the other two, 8 votes and
> not 12: `base-v1-gemma4-cap512` 0.827, `base-v1-qwen3-cap512` 0.794, `base-v1-gptoss-cap512` 0.702); Laya english 0.405
> [0.314, 0.490], Laya multilingual 0.119 [−0.040, 0.264]; LiteLLM's heuristic 0.089 [0.026, 0.172]; the prompt length
> 0.543 [0.424, 0.647]. The CI of `s1d1-full-int8` contains the ceiling mean; `s1d1-full-int8` is distinguished from the
> `length` floor by the 2 σ rule (paired gap `s1d1-full-int8` − `length`: 0.264 [0.159, 0.383]).

![Agreement of each candidate with the median of the main judges](report/base-v1/figures/qwk-bars.svg)

`s1d1-full-int8` reaches the inter-judge ceiling: its CI, 0.807 [0.734, 0.852], contains the ceiling mean (0.774, black
line; dark grey lines: each main judge against the other two). The witnesses make the floor (dashed grey lines and grey
band); Laya (0.405 and 0.119) and LiteLLM (0.089) are below the prompt length (0.543), on point values. A position on
the figure is not a superiority sentence: that needs the 2 σ rule, the CI of the paired gap excluding zero and exceeding
2 σ (σ = 0.005, blue band around `s1d1-full-int8`).

![Agreement with the judges against latency per prompt](report/base-v1/figures/quality-vs-latency.svg)

`s1d1-full-int8`, whose CI contains the ceiling mean (a main judge is placed there at its QWK against the other two),
answers in 9.1 ms per prompt on the Mac's CPU (median latency, batch of 1); the four int8 variants of s1d1, each at its
own QWK, answer in 1.6 to 9.1 ms. At the 95th percentile, `s1d1-full-int8` takes 76.8 ms (the cause of this
tail is not measured). A main judge takes 1.8 to 42.8 s on the same Mac's GPU (MLX), reading instructions of about 1,900
tokens and then reasoning (reasoning capped at 512 tokens); without reasoning, gemma4 and qwen3 take
9.2 to 12.6 s. s1d1 fp16 and Laya run on the GPU (MPS), LiteLLM on the CPU. The
witnesses are not models and are not in the figure. On disk, `s1d1-full-int8` weighs 310 MB (the five
s1d1 variants: 25 to 615 MB), a main judge 12.1 to 33.8 GB (figure `quality-vs-footprint` of the report). Measurements
of `bench profile` on an M4 Max, see [The profile](#the-profile).

![QWK by language](report/base-v1/figures/qwk-by-language.svg)

s1d1 holds up in the four languages: `s1d1-full-int8` is at 0.840 in English and from 0.787 to 0.802 in the other
three. LiteLLM drops outside English, its keyword lists being English: 0.205 in English, 0.044 to 0.061
elsewhere, on point values and without the CIs separating.

QWK against `truth_all` (the median of the three main judges), 95% CI over 1,000 draws of the 90 base prompts;
"exact tier" is the share of ids whose tier is that of `truth_all`:

| candidate | QWK [95% CI] | exact tier [95% CI] |
|---|---|---|
| s1d1 full int8 | 0.807 [0.734, 0.852] | 0.656 [0.569, 0.733] |
| s1d1 small int8 | 0.772 [0.700, 0.822] | 0.650 [0.561, 0.731] |
| ceiling (mean of the three judges) | 0.774 [0.718, 0.815] | 0.600 [0.523, 0.671] |
| length (`length` witness) | 0.543 [0.424, 0.647] | 0.394 [0.300, 0.483] |
| Laya english (`laya-english-tier`) | 0.405 [0.314, 0.490] | 0.311 [0.247, 0.386] |
| Laya multilingual (`laya-multilingual-tier`) | 0.119 [−0.040, 0.264] | 0.278 [0.203, 0.353] |
| LiteLLM heuristic | 0.089 [0.026, 0.172] | 0.342 [0.250, 0.436] |

The judges are the measuring instrument, not competitors: no table ranks a candidate against a judge, except the
ceiling. No "better than" sentence without the 2 σ rule (the CI of the paired gap excludes zero and exceeds 2 σ,
σ = 0.005): the rule does not separate `s1d1-full-int8` from `s1d1-full-fp16`, `s1d1-small-int8` or
`s1d1-compact-int8` (only `tiny` is separated from it).

LiteLLM's heuristic is the default classifier of LiteLLM's Auto Router, and Laya the other option that LiteLLM
documents for a self-hosted classifier; the bench reads both without training or tuning (zero-shot) on
this rubric, LiteLLM's four tiers translated name by name (see
[The thirteen classifier folders](#the-thirteen-classifier-folders-eleven-in-the-results)). For `small`, the
bench does not separate `s1d1-small-int8` from `s1d1-full-int8`: the paired gap `s1d1-full-int8` − `s1d1-small-int8` is
0.035 [−0.004, 0.076] (`report/base-v1/tables/paired.csv`, the 2 σ rule does not pass). `s1d1-small-int8` weighs 56 MB
against 310 MB for `s1d1-full-int8` and answers in 5.8 ms per prompt against 9.1 ms (`report/base-v1/tables/profile.csv`,
median latency, batch of 1, CPU of an M4 Max). Everything else is in
[`report/base-v1/index.md`](report/base-v1/index.md), whose first section is in English and the rest in French; to
regenerate it, from the repository root:

```
uv run --group report bench report --results results/base-v1.parquet --meta data/prompts/base-v1.meta.parquet \
    --classifiers classifiers --out report/base-v1
```

## The commands

- `bench pool`: builds the prompt pool from public Hub datasets.
- `bench export`: exports the prompts to be judged.
- `bench judge`: has a local model (MLX) judge the prompts with the v7 rubric.
- `bench validate`: checks that a judge folder is complete and consistent with the prompts.
- `bench agree`: computes the pairwise agreement between judge folders (markdown table).
- `bench draw`: draws the base prompts by stratum (random draw with a fixed seed over a presort folder).
- `bench translate`: translates a prompts file into a language (fra, rus, zho) with a local model (MLX).
- `bench witness`: fetches the m-ArenaHard (Google) translations of a set's Arena-Hard prompts, as witnesses.
- `bench check`: flags the translations to proofread (`structure`, `labels`, `request`) and writes the review sheet
  (`review`).
- `bench classify`: has a classifier classify the prompts and writes a `classifiers/<name>` folder in the format of a
  judge, plus the score when it is on the v7 scale. Six adapters: three witnesses (`random`, `length`, `constant`) and
  three systems under test (`s1d1`, weights from the Hub; `laya`; `litellm-heuristic`, LiteLLM's rule-based scorer,
  with no model), the last one and `laya` keeping their raw decisions apart in `decisions.jsonl` (tier name by name,
  no `score` in `labels.jsonl`).
- `bench run`: gathers the judge and classifier folders of a set into one table, one row per id, with the truths of the
  main judges (`results/<set>.parquet` and `<set>.run.json`).
- `bench report`: writes the report of a set (`report/<set>/index.md`, twenty-seven CSV tables, ten SVG figures with
  their CSV) from `results/<set>.parquet` and, when it exists, `results/<set>.profile.json`; see
  [The report](#the-report).
- `bench profile`: measures, for each model of a run, the latency per prompt at batch 1 after warm-up (p50, p95, mean),
  the peak memory of the process and the size of the weight files (`results/<set>.profile.json`); see
  [The profile](#the-profile).

Installation: `uv sync --group pool --group dev` (on Apple silicon, to run a judge:
`uv sync --group pool --group judge --group dev`), then `uv run bench --help`.

## Reproducing the trial

The trial has three local models judge 50 prompts, each with reasoning turned off (`off`) and capped at 512 tokens
(`cap:512`), that is six runs; five `judges/trial-*` folders are kept (gpt-oss `off` could not label one prompt and was
removed). The results and the durations are in
[judges/TRIAL.md](judges/TRIAL.md): it is a trial of the tooling on 50 prompts, not a result of the bench. The three
models, their revisions and the way `bench judge` turns off or caps their reasoning are in
[judges/MODELS.md](judges/MODELS.md); the format of the judge folders is in [judges/README.md](judges/README.md).

All commands are run from the repository root. Steps 4 and 5 need an Apple silicon Mac and
`uv sync --group pool --group judge --group dev` (or `uv run --group judge`); only one model runs at a time.

1. **Exclusion list** (`data/exclusions/s1d1-seen-keys.txt`). It is produced by the private research repository
   s1d1-model-router, on its `seen-keys` branch (`pipeline/13_seen_keys.py`). The command is run in that research
   repository; `--out` is the path of the list in this repository (written here as a relative path, to adapt):

   ```
   uv run python -I pipeline/13_seen_keys.py --inputs 'dataset/final/*.parquet' \
       --out data/exclusions/s1d1-seen-keys.txt
   ```

   Without access to that repository, the file already committed is enough: [data/exclusions/README.md](data/exclusions/README.md)
   says what it contains (hash keys, no prompt text).

2. **The prompt pool.** Public Hub datasets, no token (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` is set by the command
   itself):

   ```
   uv run --group pool bench pool
   ```

   Output: `data/pool/pool.parquet`, `data/pool/pool.stats.json`, `data/pool/candidates-eng.jsonl`.

3. **The 50 prompts of the trial.** The draw (seed 0; 17 Dolly, 17 OASST2, then 16 Arena-Hard)
   writes the key list `data/prompts/trial-50.select.txt`, then `bench export` makes the prompts file:

   ```
   uv run python -I - <<'PY'
   import json
   import random

   import pyarrow.parquet as pq

   source_of = dict(zip(*pq.read_table("data/pool/pool.parquet", columns=["prompt_key", "source"]).to_pydict().values()))
   by_source: dict[str, list[str]] = {}
   for line in open("data/pool/candidates-eng.jsonl", encoding="utf-8"):
       key = json.loads(line)["id"]
       by_source.setdefault(source_of[key], []).append(key)
   rng = random.Random(0)
   quota = {"dolly": 17, "oasst2": 17, "arena_hard": 16}
   keys = []
   for source, n in quota.items():
       candidates = sorted(by_source[source])
       keys += rng.sample(candidates, n)
   assert len(keys) == len(set(keys)) == 50
   open("data/prompts/trial-50.select.txt", "w", encoding="utf-8").write("\n".join(keys) + "\n")
   PY

   uv run bench export --pool data/pool/pool.parquet --select data/prompts/trial-50.select.txt --langs eng \
       --name trial-50 --seed 0
   ```

4. **The models**, in `~/mlx-models/<org>--<name>` (never in a repository). `bench judge` prints the exact command
   if one is missing; the identifiers and revisions to use are in [judges/MODELS.md](judges/MODELS.md):

   ```
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --group pool hf download <org>/<name> --revision <revision> \
       --local-dir ~/mlx-models/<org>--<name>
   ```

   to be done for `mlx-community/gemma-4-31b-it-8bit`, `mlx-community/Qwen3.8-27B-8bit` and
   `mlx-community/gpt-oss-20b-MXFP4-Q8`.

5. **The judges**, one after the other (provider `local`, MLX, Apple silicon only). For each family,
   `--thinking off` then `--thinking cap:512`; `--author` takes your name:

   ```
   export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
   uv run --group judge bench judge --model mlx-community/gpt-oss-20b-MXFP4-Q8 --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-gptoss-cap512 --author <name> --thinking cap:512 --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/Qwen3.8-27B-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-qwen3-off --author <name> --thinking off --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/Qwen3.8-27B-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-qwen3-cap512 --author <name> --thinking cap:512 --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/gemma-4-31b-it-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-gemma4-off --author <name> --thinking off --max-tokens 1024
   uv run --group judge bench judge --model mlx-community/gemma-4-31b-it-8bit --prompts data/prompts/trial-50.jsonl \
       --out judges/trial-gemma4-cap512 --author <name> --thinking cap:512 --max-tokens 1024
   ```

   An interrupted run is resumed with `--resume`.

6. **Validate and compare**:

   ```
   for d in judges/*/; do uv run bench validate "$d"; done
   uv run bench agree judges/trial-*
   ```

## The draw of the base prompts

The base set is not chosen by hand (docs/32 § 3.1, amended on 2026-10-29). The 900 English candidates
(`data/prompts/presort-900`, export of `candidates-eng.jsonl`) are labelled by a presort judge that is not
Gemma 4 (`judges/presort-qwen3-off`, Qwen3.8 27B without reasoning; it is an ordinary judge folder, not a judge of the
bench). `bench draw` then draws 20 prompts per stratum `trivial`, `low`, `medium`, `high`: a shuffle of each stratum with
a fixed seed, then the true `needs_tools` in round-robin up to 20, then each task of the rubric up to 4
(from the stratum with the most room), then filling. A veto (`<id><TAB><reason>` per line, two rules only:
same template in the same stratum, or a translation that would change the task) removes an id from its order, the next
one of the stratum takes its place and the other draws do not move. What the candidates do not allow (a minimum of
`needs_tools` or of a task) is reported, not silently corrected.

```
uv run bench draw --presort judges/presort-qwen3-off --meta data/prompts/presort-900.meta.parquet \
    --out data/prompts/base-v1.select.txt --seed 0 --veto data/prompts/base-v1.veto.txt
```

`base-v1.select.txt` (the drawn keys), `base-v1.draw.json` (rule, seed, strata, shortfalls) and `base-v1.veto.txt`
(the removals and their reason) are published: the draw can be replayed. The `expert` cell is made of the ten prompts of
`data/handwritten/expert-v1.jsonl`, not of the pool.

## The translation

The 90 English base prompts (`data/prompts/base-v1-eng`, export of `base-v1.select.txt` and of the ten handwritten
prompts) are translated into French, Russian and simplified Chinese by the presort model, Qwen3.8 27B 8-bit without
reasoning, at temperature 0 ([docs/33](docs/33-translation-lot2.md)). `bench translate` translates a prompts file
into one language, one prompt per generation: the instructions (`consigne/translate-v1-<lang>.md`) are the system
message, the prompt is given between `<prompt>` tags as data.

```
uv run --group judge bench translate --model mlx-community/Qwen3.8-27B-8bit --prompts data/prompts/base-v1-eng.jsonl \
    --lang fra --consigne consigne/translate-v1-fra.md --out data/translations/base-v1 --author <name> --thinking off
```

Output in `--out`: `<lang>.jsonl` (the translations), `<lang>.meta.json` (model, revision, instructions and their
SHA-256, SHA-256 of the English file, settings), `<lang>.log.jsonl` (one status per prompt: `ok`, `empty`, `envelope`,
`truncated`, `error`). As with a judge, the files are never rewritten; `--resume` resumes an interrupted run with the
same settings. A translation that contains a `</prompt>` line or that starts with `<prompt` is a logged error,
never a translation; the one exception, an answer entirely wrapped in a `<prompt>…</prompt>` pair (the
model copies the tags), is stripped of that pair and the log says so (`unwrapped`).

Measured speed (M-series 128 GB, MLX, run of 90 prompts, loading included): 825 s in French, 825 s in Russian,
657 s in Chinese, that is a median of 9 to 11 tokens per second in decoding.

### The checks

A translation is not delivered without proofreading. `bench check` corrects nothing: it flags lines for a human to
read, and writes everything in `data/checks/<set>/` (`<lang>.structure.jsonl`, `<lang>.labels.jsonl`,
`<lang>.request.jsonl`, then `<lang>.review.md`, the sheet where each flagged id has an `avis :` (opinion) and a
`décision :` (decision)).

- `structure`: code blocks, URLs, numbers, straight quotes, number of lines, length (translation/English ratio,
  bounded per language), `<prompt>` tag. Reading rule: the `quotes` flags are almost all typographic quotes (« »,
  「」) in place of straight quotes, a limit of the rule and not an error; the `numbers` ones are "3:30" turned into
  "15h30", ordinals written out in words or thousands separators; the `length` ones are very short prompts.
- `labels`: the Qwen3 judge (the presort one) labels the English and the translation (`judges/check-qwen3-off-<lang>`);
  a `task` or a `needs_tools` that changes is flagged. For the 28 Arena-Hard prompts, the Google translation
  (m-ArenaHard) serves as a witness (`bench witness`, `judges/witness-qwen3-off-<lang>`): if the witness gets from the
  judge the same label as our translation, the flag is lifted. Judge noise dominates this check: almost all
  label differences are noise, except three mistranslations that Gemma 4 also spotted.
- `request`: Gemma 4 31B (a different model from the translator) reads the English and the translation and answers in
  JSON `{"same_request": bool, "reason": str}` (`consigne/check-same-request-v1.md`). A `false` is a signal, not a verdict.

```
uv run --group judge bench check request --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1 --model mlx-community/gemma-4-31b-it-8bit --author <name> --thinking off
uv run bench check structure --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1
uv run bench check labels --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1 --labels-english judges/check-qwen3-off-eng \
    --labels judges/check-qwen3-off-fra --labels-witness judges/witness-qwen3-off-fra
uv run bench check review --set base-v1 --lang fra --english data/prompts/base-v1-eng.jsonl \
    --translations data/translations/base-v1 --witness data/translations/base-v1/witness-arena
```

The Google witness is raw: HTML entities (`&#39;`) and spaces at the edge of a line stay in it, we do not clean it.

A human correction is an overlay: `data/translations/base-v1/<lang>.corrections.jsonl` (one line
`{"id", "prompt", "reason", "author"}` per corrected id). `bench export` applies it and marks `corrected` in
`meta.parquet`; the model's files (`<lang>.jsonl`) are never rewritten. A veto (a translation that would change the
task, two signals together) would remove the id from the draw and its replacements would go into an `-r1` wave
(`--translations` is repeated, once per wave); the proofreading of `base-v1` decided none.

The proofreading of `base-v1` (the three languages read against the English) retained 22 corrections: French 9,
Russian 6, Chinese 7. 11 change the meaning (the English typos "captiols" and "store" taken literally in the three
languages; "banks'" lost in French and in Chinese; "stochastic" added in Russian; "aloud" added and "keep score"
poorly rendered in French), 7 are wrong technical terms (key schedule, "responsable du traitement", "assignment" in
French; shellcode in Russian; litmus test, Proto-Indo-European and a journal title translated in Chinese), 4 are damaged
names or agreements (Tanooki, a grammatical agreement, "Шлой", Bourbon rendered as Bourgogne). No
veto. What this does not prove: the meaning check is a reading check over 90 prompts per language, done by a
single reviewer (Claude Fable 5.1), not by native speakers; neither stylistic quality nor register is measured.

### The paired set

`data/prompts/base-v1.jsonl` gathers the 90 prompts in the four languages (360 lines, 9 batches, ids `<key>-<lang>`),
with the corrections applied:

```
uv run bench export --select data/prompts/base-v1.select.txt --handwritten data/handwritten/expert-v1.jsonl \
    --translations data/translations/base-v1 --langs eng,fra,rus,zho --name base-v1
```

The `base-v1-<lang>.jsonl` files are those the verification judges saw, before corrections; they stay as they
are (their SHA-256 is the one `bench validate` asks of the `check-*` folders), and `bench export` refuses to rewrite
a prompts file that a judge folder pins by its SHA-256 if the content changes. `meta.parquet` carries `translator`
(`model@revision`, empty in English) and `corrected`.

## Bringing your own prompts

`base-v1` is the first set of a series: nothing obliges the following ones to reuse its prompts, its languages or its
sources. To propose a set `<set>`:

1. **A pool.** `bench pool` (`--exclusions` for the list of keys to leave out, `--candidates` and `--per-source` for the
   candidates) builds the pool; `bench export --select <keys> --name <set>` makes the prompts file
   `data/prompts/<set>.*` from it, with `--langs` and `--translations` for the languages other than English.
2. **A presort by a judge that is not a bench judge.** A model other than the three main judges labels the
   candidates (`bench judge`), in an ordinary judge folder, like `judges/presort-qwen3-off` for `base-v1`.
3. **A published draw.** `bench draw --presort … --meta … --out … --seed <seed> --veto <file>`; publish the drawn keys,
   the seed, the veto and its reasons: the draw can be replayed (see [The draw](#the-draw-of-the-base-prompts)).
4. **A proofread translation.** `bench translate` translates each language, `bench check` flags what to proofread, and a
   speaker of the language proofreads (see [The translation](#the-translation)).
5. **The bench judges.** `bench judge` has the three main judges judge the set, then `bench validate`,
   `bench agree`, `bench classify`, `bench run` and `bench report` produce the folders and the report of the set.

A PR adds: a prompts folder, its judge and classifier folders, and its report (`report/<set>/`, with
`results/<set>.parquet` that feeds it). It never touches an existing file: neither the judges, nor the classifiers,
nor the results, nor the prompts of a set already published. A set that wants to correct `base-v1` is another set.

To read before starting: the licences of the sources (the `licence` column of `pool.parquet` and of `<set>.meta.parquet`,
section [Licences](#licences)) and the exclusion list (`data/exclusions/`, which says what s1d1 has already seen: see
[Reproducing the trial](#reproducing-the-trial), step 1).

## The truths

`bench run` gathers the judge and classifier folders of a set into one table, one row per id; the truths come only from
the main judges (`--main`). The true tier is the ordinal median (trivial < low < medium < high < expert) of the
non-null votes, `truth_all` over the 12 votes of the four versions of a base prompt (3 judges × 4 languages),
`truth_lang` over the 3 votes of the id. Rule R5 (docs/34 § 0.3): an odd number of votes takes the middle one; an even
number takes the mean of the ranks of the two middle votes, rounded down. It is the lower median when the
two middle votes are equal or adjacent (6/6 on two adjacent tiers gives the lower), but low and high give
medium. The tier is null (`undecidable`) when the `null`s are a strict majority (7 out of 12; 6 out of 12 is not
one). `task` and `needs_tools` go to the majority vote, a tie broken by the first judge of `--main`.

Abstentions: a judge whose output gives no readable label (refusal, unreadable JSON, truncated answer:
`json_error` or `truncated`) has an `"abstained"` line in `labels.jsonl` (see [judges/README.md](judges/README.md));
an engine `error` is not one (it may be transient: the id stays without a line and `--resume` asks for it again).
For the truths, an abstention is a null vote on the three fields: outside the median, outside the majority and outside
`spread`, counted as a `null` towards `undecidable`. `bench run` writes it in the `<judge>.abstained` column and the list
in `run.json`; an id without any line is still refused, never filled. `bench agree` leaves out of the pair an id where
one of the two judges abstained (column `abstained`).

## The report's judges and the classifiers

The `base-v1` report rests on five series of judges and eleven classifiers, all on `data/prompts/base-v1.jsonl` (360
lines, SHA-256 `e6a6d4c12ef8b95caea942337a3a1e16ec1622cb4e75c8ee0b473ef609a32490`). The judges ran on 2026-10-30
on a Mac M4 Max 128 GB, with `mlx-lm` 0.32.2, one model at a time, temperature 0, `--max-tokens 1024`, the `v7-single`
instructions, revisions of [judges/MODELS.md](judges/MODELS.md).

### The five judges

| folder | setting | duration | role |
|---|---|---|---|
| `judges/base-v1-gemma4-cap512` | Gemma 4 31B, reasoning capped at 512 | 4 h 28 min (4.48 h) | main median |
| `judges/base-v1-qwen3-cap512` | Qwen3.8 27B, reasoning capped at 512 | 3 h 02 min (3.04 h) | main median |
| `judges/base-v1-gptoss-cap512` | gpt-oss-20b, reasoning capped at 512 | 11 min 36 s (0.19 h) | main median |
| `judges/base-v1-gemma4-off` | Gemma 4 31B, without reasoning | 1 h 18 min (1.30 h) | effect of reasoning, outside the median |
| `judges/base-v1-qwen3-off` | Qwen3.8 27B, without reasoning | 56 min 41 s (0.94 h) | effect of reasoning, outside the median |

The durations run from the start to the end of each series, that is 9 h 57 min 38 s in total (the plan foresaw 8.4 h);
the gpt-oss one does not count the 9 s of the resume that wrote the abstention. The first series (`qwen3-off`) ran alone on
the machine; the others ran while review subagents and light trials were also using it:
the durations therefore cannot be compared to the nearest second. The three `cap512` judges form the main median: Gemma 4
is the setting of the training judge, and [docs/32](docs/32-open-bench-kit.md) asks for at least three local judges,
gpt-oss being the third model that reads the JSON on the 50 prompts of the trial
([docs/34](docs/34-bench-run-report-lot3.md) decision 2). The two `off` folders measure the effect of reasoning on the
360 prompts; they appear in the judges table but not in the main median, because two settings of the same
model are not two independent judges.

**The abstention case comes first.** `judges/base-v1-gptoss-cap512` has 359 labels and one abstention: gpt-oss
refuses the id `3ce175ed3ef6b933-zho` (the Chinese prompt on the synthesis of hydrochloric acid) and returns unreadable JSON
(`json_error`). It is not an invented label: for this id the folder holds a line `"abstained": "json_error"`
with the three fields `null` (the line was written by a plain `--resume` with the same settings, after the rule on
abstentions), and `bench validate` prints `ok (1 abstention)`. What an abstention is and is worth for the truths is said in
[The truths](#the-truths) and in [judges/README.md](judges/README.md); this section does not return to it. In the agreements
below, the `abstained` column counts the ids left out of the pair.

### The thirteen classifier folders (eleven in the results)

| folder | what it is | revision | duration |
|---|---|---|---|
| `classifiers/random` | witness: uniform draw, seed 0, 10 draws (`labels.jsonl` = draw 0, `draws.jsonl` = the ten) | seed 0 | n/a |
| `classifiers/length` | witness: deciles of the length in characters, per language, mapping 2/3/3/1/1 | not applicable | n/a |
| `classifiers/constant-low` | witness: `low` everywhere, score 0.25 | not applicable | n/a |
| `classifiers/s1d1-full-fp16` | `scombat/s1d1-router`, PyTorch fp16, mps | Hub head `7fb9a4e` | 7.9 s |
| `classifiers/s1d1-full-int8` | `scombat/s1d1-router-6src-full-onnx-int8`, ONNX int8, CPU | card `068d0d2` | 35.6 s |
| `classifiers/s1d1-compact-int8` | `scombat/s1d1-router-6src-compact-onnx-int8`, ONNX int8, CPU | card `7e87870` | 28.5 s |
| `classifiers/s1d1-small-int8` | `scombat/s1d1-router-6src-small-onnx-int8`, ONNX int8, CPU | card `33c7f34` | 12.2 s |
| `classifiers/s1d1-tiny-int8` | `scombat/s1d1-router-6src-tiny-onnx-int8`, ONNX int8, CPU | Hub head `dac13b5` | 6.7 s |
| `classifiers/laya-english` | no longer counts (see below): Laya 0.4.15, english checkpoint, mps, nearest anchor | `convaiinnovations/laya@7b928d82…` | 13.1 s |
| `classifiers/laya-multilingual` | no longer counts (see below): Laya 0.4.15, multilingual checkpoint, mps, nearest anchor | `convaiinnovations/laya@7b928d82…` | 9.0 s |
| `classifiers/laya-english-tier` | Laya 0.4.15, english checkpoint, mps, most probable tier name by name | `convaiinnovations/laya@7b928d82…` | ≈ 20 s (not timed) |
| `classifiers/laya-multilingual-tier` | Laya 0.4.15, multilingual checkpoint, mps, most probable tier name by name | `convaiinnovations/laya@7b928d82…` | ≈ 20 s (not timed) |
| `classifiers/litellm-heuristic` | LiteLLM's built-in heuristic scorer (`classifier_type: heuristic`, default configuration), code copied as is at tag v1.106.6, no model | tag `v1.106.6`, commit `61c6ab78…` | under a second |

The classifiers' durations are those of classifying the 360 prompts, with the small Hub download, the weights being
already in cache. Three s1d1 revisions (`small`, `compact`, `full` int8) are pinned to the revision of the repository's card.
For `tiny` in int8 and for the fp16 model, the card gives no revision: the bench takes the Hub head at the time of
the run, after checking (read-only, no token) that the weight files have not changed since the last
commit that modified them; the check is written in the `notes` field of each folder's `meta.json`. For
fp16, `config.json` and `s1d1_router.py` changed after the weights were added: "unchanged" holds for the weight files, and the repository head
provides the `config.json` and the router used. Laya is pinned to `convaiinnovations/laya@7b928d82…` (the full hash is in `meta.json`).

**LiteLLM's scorer.** `classifiers/litellm-heuristic` has neither model nor network: the bench embeds the code of LiteLLM's
heuristic scorer, copied as is at tag `v1.106.6` (commit `61c6ab78393ef45683dd7d963d2a315bd845e20b`) into
`src/s1d1_bench/vendor/litellm_heuristic/`, without installing the `litellm` package. The `NOTICE.md` file next to the code says
where it comes from, which blocks are taken and how; the licence of this code is LiteLLM's, MIT (`LICENSE`, copied
with it). Each prompt is classified alone, with no system instructions, with LiteLLM's default configuration; the
tiers are translated name by name (`SIMPLE` to `low`, `MEDIUM` to `medium`, `COMPLEX` to `high`, `REASONING` to `expert`),
and `task` and `needs_tools` stay `null`. Its score is not on the v7 scale (it can be negative): `labels.jsonl` has
therefore no `score`, and the scorer's raw decision (tier, score, signals, cause) is in `decisions.jsonl`, next to
`labels.jsonl`. `bench run` reads it in the columns `litellm-heuristic.raw_score` and `litellm-heuristic.raw_tier`, and
marks `"raw_decisions": true` in the classifier's entry of `run.json`. The raw score is used only for the rank measures.

**Hub revisions for the int8 repositories.** For `small`, `compact` and `full`, the current Hub head is not the revision of
the card: the commits between them touch only metadata (`README.md`, `eval.json`, `footprint.json`,
`variant.json`, where only `sigma` and its date change); the `*.onnx` and `*.npy` files have the same LFS SHA-256 at the head
and at the revision of the card. The folders keep the revision of the card, the head and this check being
in their `notes`.

### Replaying

All commands are run from the repository root, the judge models being already in `~/mlx-models/` (see
step 4 of "Reproducing the trial"). An existing folder is never rewritten: to redo a series, write it out under
another name.

- **A judge** (Apple silicon, group `judge`):

  ```
  uv run --group judge bench judge --model mlx-community/gemma-4-31b-it-8bit --prompts data/prompts/base-v1.jsonl \
      --out judges/base-v1-gemma4-cap512 --author <name> --thinking cap:512 --max-tokens 1024
  ```

  To be repeated for `mlx-community/Qwen3.8-27B-8bit` (`qwen3`) and `mlx-community/gpt-oss-20b-MXFP4-Q8` (`gptoss`) in
  `cap:512`, and for Gemma and Qwen in `--thinking off` (`base-v1-gemma4-off`, `base-v1-qwen3-off`). The `meta.json` of
  each folder says the revision, the reasoning setting and the role of the folder.
- **The witnesses**, no particular group:

  ```
  uv run bench classify random --seed 0 --draws 10 --prompts data/prompts/base-v1.jsonl --out classifiers/random --author <name>
  uv run bench classify length --prompts data/prompts/base-v1.jsonl --out classifiers/length --author <name>
  uv run bench classify constant --level low --prompts data/prompts/base-v1.jsonl --out classifiers/constant-low --author <name>
  ```
- **s1d1 in ONNX int8** (CPU, group `s1d1`), with the revision of the card:

  ```
  uv run --group s1d1 bench classify s1d1 --model scombat/s1d1-router-6src-small-onnx-int8 --revision 33c7f34a30af04735390264824d6bfa8546abb7a \
      --backend onnx --prompts data/prompts/base-v1.jsonl --out classifiers/s1d1-small-int8 --author <name>
  ```

  `compact` (`7e87870132a19d12b32dbad1f8233801b8145b9f`) and `full` (`068d0d20d5901c2db6934c89fde48b87c973fed8`) in the same
  way; `tiny` (`dac13b5c0356864ea4501fa151d74dcd8b3ea22a`, the Hub head) too, with its identifier
  `scombat/s1d1-router-6src-tiny-onnx-int8`.
- **s1d1 in fp16** (PyTorch, groups `s1d1-torch` and `s1d1`):

  ```
  uv run --group s1d1 --group s1d1-torch bench classify s1d1 --model scombat/s1d1-router \
      --revision 7fb9a4e3d03e3d942a98648268ff76c6152c18d6 --backend torch --prompts data/prompts/base-v1.jsonl \
      --out classifiers/s1d1-full-fp16 --author <name>
  ```
- **Laya** (group `laya`, separate):

  ```
  uv run --group laya bench classify laya --laya-model english --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
      --prompts data/prompts/base-v1.jsonl --out classifiers/laya-english-tier --author <name>
  ```

  `--laya-model multilingual` and `--out classifiers/laya-multilingual-tier` for the other checkpoint.

  **Two Laya folders per checkpoint, and which ones count.** Laya answers LiteLLM's four-tier question (`SIMPLE`,
  `MEDIUM`, `COMPLEX`, `REASONING`), the same as LiteLLM's scorer below, and the two
  candidates are read with the same table, name by name: `SIMPLE` → low, `MEDIUM` → medium, `COMPLEX` → high, `REASONING` →
  expert, never trivial. The folders **`laya-english-tier` and `laya-multilingual-tier` are the ones that count**: the tier
  is the most probable of the four (tie: the lowest name), `labels.jsonl` has no `score`, and `decisions.jsonl` keeps
  per id Laya's tier, the expected level `Σ i·pᵢ` over 0..3 (written `score`, it serves only for the rank metrics) and the
  four probabilities as they are. The folders **`laya-english` and `laya-multilingual`** of the first PR converted
  the expected level divided by 3 into a tier by the nearest anchor (SIMPLE became trivial, MEDIUM low, COMPLEX high):
  a different table from that of LiteLLM's scorer. They stay in the repository, because a classifier folder is never
  modified, but `bench run` no longer loads them: they are neither in the results nor in the report's numbers,
  which only says why.
- **LiteLLM's scorer** (no model, no group, no network):

  ```
  uv run bench classify litellm-heuristic --prompts data/prompts/base-v1.jsonl --out classifiers/litellm-heuristic --author <name>
  ```
- **The table**:

  ```
  uv run bench run --prompts data/prompts/base-v1.jsonl --judges 'judges/base-v1-*' \
      --main base-v1-gemma4-cap512,base-v1-qwen3-cap512,base-v1-gptoss-cap512 \
      --classifiers 'classifiers/s1d1-*' classifiers/laya-english-tier classifiers/laya-multilingual-tier \
      classifiers/litellm-heuristic classifiers/random classifiers/length classifiers/constant-low \
      --out results/base-v1.parquet
  ```

  `--judges` and `--classifiers` take several values in a row (globs between quotes, which `bench run`
  expands itself), or the repeated option.

  The classifiers are listed one by one: `'classifiers/*'` would also load `laya-english` and `laya-multilingual`, which no longer
  count (see Laya above).
- **Validating** the deposited folders, as the CI does (`.github/workflows/ci.yml`):

  ```
  for d in judges/*/ classifiers/*/; do uv run bench validate "$d"; done
  ```

Warning: `uv sync --group <g>` makes the environment exact and removes the groups it does not name; syncing the
`judge` group then the `laya` group uninstalls the first. For one command, `uv run --group <g> ...`; for a
complete environment, `uv sync --all-groups` (which includes `mlx-lm`, hence Apple silicon only).

### The truth rules

The truths (median over all languages on 12 votes, median per language on 3 votes, rule R5, undecidables, majority vote
of `task` and `needs_tools`, abstentions) are those of [The truths](#the-truths); they are not repeated
here. The count of `bench run` on `base-v1`, with the three main judges: 1 abstention, no undecidable id;
33 ids out of 360 have a language disagreement (24 of the 90 base prompts), mean spread 0.675 (spread 0 for 140 ids, 1 for 197,
2 for 23); `truth_all.complexity` gives trivial 68, low 116, medium 116, high 52, expert 8, and `truth_all.needs_tools`
is true for 52 ids. A plain lower median, instead of rule R5, would change `truth_all` for 0 of the 90 base keys, and
`truth_all_nogemma` for 1.

### What `results/` is, and is not

`results/base-v1.parquet` (360 rows, 92 columns) and `results/base-v1.run.json` (the folders loaded, their revisions,
the abstentions, the rules) are **derived** from the `judges/` and `classifiers/` folders: they are regenerated at each
`bench run`; a classifier's columns are `<name>.complexity`, `<name>.task`, `<name>.needs_tools`, `<name>.score` when its
`labels.jsonl` has one and, when it has a `decisions.jsonl`, `<name>.raw_score` and `<name>.raw_tier` (`litellm-heuristic` has
these two and no `score`: five columns; `laya-english-tier` and `laya-multilingual-tier` too). It is not an immutable record, unlike the judge
and classifier folders, which are added to and not rewritten: what is replayed is the inputs. Nor is it
a report: no metric, no figure is in it, they are in `report/` (see [The report](#the-report)).

### The judges' counts

Distribution of each judge's tiers (`needs_tools`: number of ids labelled true; `null`: tiers left without
a label, the gpt-oss abstention and a `null` tier of Gemma `cap512`, output of the judge):

| judge | trivial | low | medium | high | expert | null | needs_tools true |
|---|---|---|---|---|---|---|---|
| base-v1-gemma4-cap512 | 51 | 135 | 105 | 58 | 10 | 1 | 61 |
| base-v1-gemma4-off | 59 | 130 | 105 | 43 | 23 | 0 | 57 |
| base-v1-gptoss-cap512 | 81 | 122 | 143 | 13 | 0 | 1 | 40 |
| base-v1-qwen3-cap512 | 85 | 79 | 106 | 71 | 19 | 0 | 59 |
| base-v1-qwen3-off | 70 | 85 | 92 | 91 | 22 | 0 | 69 |

Pairwise agreement (`bench agree judges/base-v1-*`):

| judge A | judge B | n | abstained | exact tier | tier ±1 | task | needs_tools |
|---|---|---|---|---|---|---|---|
| base-v1-gemma4-cap512 | base-v1-gemma4-off | 360 | 0 | 81.4% | 99.4% | 85.6% | 97.2% |
| base-v1-gemma4-cap512 | base-v1-gptoss-cap512 | 359 | 1 | 56.3% | 96.4% | 56.5% | 91.4% |
| base-v1-gemma4-cap512 | base-v1-qwen3-cap512 | 360 | 0 | 70.3% | 99.7% | 83.1% | 93.9% |
| base-v1-gemma4-cap512 | base-v1-qwen3-off | 360 | 0 | 65.8% | 99.7% | 80.3% | 94.4% |
| base-v1-gemma4-off | base-v1-gptoss-cap512 | 359 | 1 | 55.4% | 94.7% | 49.3% | 91.9% |
| base-v1-gemma4-off | base-v1-qwen3-cap512 | 360 | 0 | 72.2% | 99.7% | 80.3% | 95.6% |
| base-v1-gemma4-off | base-v1-qwen3-off | 360 | 0 | 67.8% | 99.7% | 83.3% | 95.0% |
| base-v1-gptoss-cap512 | base-v1-qwen3-cap512 | 359 | 1 | 47.1% | 93.9% | 59.6% | 91.9% |
| base-v1-gptoss-cap512 | base-v1-qwen3-off | 359 | 1 | 46.5% | 91.4% | 53.5% | 90.8% |
| base-v1-qwen3-cap512 | base-v1-qwen3-off | 360 | 0 | 77.8% | 100.0% | 84.7% | 95.6% |

These two tables are counts, given once here, with no interval or agreement metric; the QWKs, their confidence
intervals and what they say are in [The report](#the-report).

## Le rapport

`report/base-v1/` est écrit par `bench report` à partir de `results/base-v1.parquet` ; il est dérivé comme `results/`,
et se régénère (même entrée, mêmes octets : le rapport est déterministe, SVG compris). Le refaire :

```
uv run --group report bench report --results results/base-v1.parquet --meta data/prompts/base-v1.meta.parquet \
    --classifiers classifiers --out report/base-v1
```

`--classifiers` est la racine des dossiers : la bande du témoin aléatoire se lit dans `classifiers/random/draws.jsonl`
(dix tirages), `results/base-v1.run.json` ne donnant que les noms. Le profil de `bench profile` est lu à côté des
résultats (`results/base-v1.profile.json`) ou là où `--profile` le nomme ; sans profil, le tableau `profile` et les deux
figures de vitesse et d'empreinte manquent, et le rapport le dit. Une quinzaine de secondes. La commande refuse de
tourner (code 1, un message) si un dossier de classifieur nommé par `run.json` manque à la racine, si `run.json` n'a pas
l'une des clés `classifiers`, `judges`, `main` ou la table méta l'une des colonnes `source`, `origin`, si le gabarit
`report/TEMPLATE.md` est introuvable, si `--profile` nomme un fichier absent, ou si `matplotlib` (groupe `report`)
n'est pas installé.

**Ce qu'il contient.** `index.md` (lu de haut en bas : les deux prédictions de docs/34 § 7.4 d'abord, celle qui ne se
vérifie pas en tête ; la phrase de docs/32 § 5 avec ses chiffres ; les mises en garde de docs/32 § 2 ; l'accord de chaque
candidat et de chaque juge avec `truth_all` puis `truth_all_nogemma` ; le plafond, le plancher, les écarts appariés ; par
langue, par source, par origine, par palier ; `task` et `needs_tools` ; les rangs et la calibration du score ; ce qui
manque), `tables/` (vingt-sept CSV, dont `profile` et `tier_distribution`) et `figures/` (dix SVG, chacun avec le CSV
de ses données et une légende en français dessous dans `index.md` : les sept de l'accord, plus la qualité contre la
latence, contre le poids sur disque, et la distribution des paliers prédits). Le gabarit
`report/TEMPLATE.md` porte les mises en garde et la phrase à trous ; le code remplit les trous avec les chiffres tels
qu'ils sortent, et les deux clauses finales de la phrase suivent des règles écrites dans `src/s1d1_bench/report_index.py`
et citées dans le rapport. Les intervalles sont à 95 %, par rééchantillonnage des 90 prompts de base (1 000 tirages,
graine 0, les quatre versions suivent leur prompt de base).

**Ce qu'il dit.** Sur 360 ids (90 prompts de base en quatre langues), `s1d1-full-int8` s'accorde avec la médiane des trois
juges principaux à QWK 0,807 [0,734 ; 0,852] et `s1d1-full-fp16` à 0,814 [0,738 ; 0,865] ; le plafond (chaque juge
principal contre les deux autres, 8 votes) vaut 0,774 [0,718 ; 0,815] en moyenne ; le plancher informé, la longueur du
prompt, 0,543 [0,424 ; 0,647], et le hasard −0,028 à 0,049 sur dix tirages. Laya, lu nom à nom, est à 0,405 [0,314 ; 0,490]
(english) et 0,119 [−0,040 ; 0,264] (multilingual) ; l'heuristique de LiteLLM à 0,089 [0,026 ; 0,172]. **La prédiction 2 de
docs/34 § 7.4 (LiteLLM entre `length` et s1d1) ne se vérifie pas** : LiteLLM est au-dessous de `length`, et sa prédiction
de palier est `low` pour 93 % des ids, son exactitude (0,342) restant à côté de celle de `constant-low` (0,322). La
prédiction 1 (son QWK tombe en français, russe et chinois) se vérifie sur les valeurs ponctuelles, sans que les IC se
séparent. Chaque chiffre est dans un tableau de `report/base-v1/tables/`, cité dans `index.md`.

**Ce qu'il ne prouve pas.** Il ne sépare pas `s1d1-full-int8` de `s1d1-full-fp16`, de `small` ni de `compact` : les
écarts appariés de `fp16` et de `small` ont un IC qui contient zéro, celui de `compact` (0,048 [0,009 ; 0,089]) exclut
zéro mais sa borne basse n'est pas au-delà de 2 σ = 0,010 (σ = 0,005, [docs/18](docs/README.md), lecture stricte écrite
dans le rapport). Les trois juges principaux sont dans la vérité : leur accord avec elle compte leurs propres votes, et
le chiffre qui les compare est le plafond. Gemma 4 a produit les étiquettes d'entraînement de s1d1 (docs/32 § 4.4) et
est dans `truth_all` : le rapport donne l'accord contre la médiane sans Gemma (`truth_all_nogemma`, tableau « Sans Gemma
4 ») ; `s1d1-full-int8` y passe de 0,807 à 0,785 (`fp16` de 0,814 à 0,794) ; parmi les candidats qui n'ont pas vu Gemma,
`length` passe de 0,543 à 0,478 et `laya-english-tier` de 0,405 à 0,347, tandis que `litellm-heuristic` (0,089 → 0,106)
et `laya-multilingual-tier` (0,119 → 0,115), proches du plancher, ne baissent guère. Ces écarts se lisent sur les
valeurs ponctuelles, sans IC (aucun écart apparié entre vérités n'est calculé). Sans Gemma, la vérité est la médiane des
8 votes de deux juges, arrondie vers le bas : `expert` disparaît (8 ids → 0), `high` passe de 52 à 28 ids, 20 des 90
prompts de base changent de palier (18 vers le bas), et un candidat qui ne prédit jamais `expert`, comme s1d1, peut y
perdre moins. Les deux juges restants s'accordent avec Gemma (QWK 0,879 pour Qwen 3, 0,707 pour gpt-oss, tableau des
juges). Le tableau « Sans Gemma 4 » ne dit donc pas si s1d1 hérite des biais de Gemma. Le rapport ne mesure pas la
grille, le routage, la distribution réelle des prompts, les prompts agentiques ou multi-tours, ni les langues autres que
les quatre ; trois analyses de [docs/32](docs/32-open-bench-kit.md) ne sont pas produites dans ce lot (la part des
prompts notés 0,70 à 0,80 que les juges mettent en `high` ou au-dessus, les confusions entre tâches voisines, la matrice
de confusion normalisée de la tâche), et `expert` n'a que 8 ids sur 360 (2 prompts de base) : tout ce qui touche ce
palier est dit sur très peu de prompts. Les 360 ids ne sont pas 360 prompts indépendants : ce sont 90 prompts de base en
quatre versions, et les IC en tiennent compte.

### Le profil

`results/base-v1.profile.json` est écrit par `bench profile` à partir de `results/base-v1.run.json`. Il se refait sur
le Mac, avec les poids déjà présents en local : la commande pose `HF_HUB_OFFLINE=1`, rien n'est téléchargé, et un modèle
absent du cache échoue au lieu de se télécharger.

```
uv run --group s1d1 --group s1d1-torch --group laya bench profile --prompts data/prompts/base-v1.jsonl \
    --run results/base-v1.run.json --judges-dir judges --models-dir ~/mlx-models --out results/base-v1.profile.json
```

**Ce qu'il mesure.** Pour chaque classifieur du run dont le `meta.json` nomme un modèle (s1d1 ×5, Laya ×2,
l'heuristique de LiteLLM), la commande recharge le modèle par son adaptateur, aux mêmes `model`, `revision`, backend et
checkpoint (la ligne « Replay » de `meta.json`), dans un processus Python neuf ; elle le chauffe sur 10 appels, puis
chronomètre un appel par prompt du fichier, un prompt par appel (lot de 1, `time.perf_counter_ns`, 360 prompts) : p50,
p95 et moyenne en millisecondes. La mémoire de crête est le `ru_maxrss` de ce processus, lu après les appels
chronométrés ; un processus par modèle, pour que la crête de l'un ne soit pas celle d'un autre. Le poids est la taille
sur disque des fichiers de poids de la révision épinglée dans le cache local du Hub (le `.onnx` int8 des variantes int8,
le `.safetensors` fp16 de `s1d1-full-fp16`, le fichier de chaque tête pour Laya ; 0 pour LiteLLM), en Mo de 10⁶ octets.
Pour les juges, la latence vient de leur `log.jsonl` (`wall_s` des appels finis `ok`, un prompt par appel par
construction) et le poids des fichiers de `~/mlx-models/<org>--<nom>`. Les appareils sont ceux que les adaptateurs
rapportent : s1d1 int8 et LiteLLM sur CPU, s1d1 fp16 et Laya sur le GPU du Mac (MPS), les juges sur le GPU par MLX. Les
témoins ne sont pas des modèles : ils sont listés dans `meta.skipped`, comme le serait un modèle qui n'a pas pu se
charger (le fichier est alors écrit sans lui et la commande sort en code 1). `--only nom,...` restreint la mesure à des
noms du run, les autres allant dans `meta.skipped` ; un nom hors du run est refusé. `bench validate` ne lit pas ce
fichier.

**Ce qu'il ne dit pas.** La latence d'un juge comprend la lecture de la consigne v7 (environ 1 900 jetons) et la
génération de sa réponse, raisonnement compris : elle dit ce que coûte un juge employé comme routeur, pas la vitesse du
modèle sur le seul prompt. La mémoire de crête comprend l'interpréteur, les bibliothèques (onnxruntime, torch,
tokenizers) et le chargement du modèle : ce n'est pas la taille du modèle seul ; sur MPS, la mémoire tenue par le pilote
du GPU peut y manquer ; pour les juges, elle n'est pas mesurable après coup (`null`). Les latences bougent d'une
exécution à l'autre : le fichier est une mesure, pas un calcul, et ne se refait pas à l'octet près. Il a été mesuré une
fois, sur un M4 Max, en une minute (58 s pour les huit classifieurs et les cinq juges).

## Dépendances

| Dépendance | Groupe | Raison |
|---|---|---|
| `pyarrow` | principal | lecture et écriture des fichiers parquet du jeu de prompts |
| `datasets` | `pool` | lecture des jeux publics dont le jeu de prompts est tiré |
| `huggingface_hub` | `pool` | accès en lecture seule aux jeux publics du Hub, sans jeton |
| `mlx-lm` | `judge` | exécution locale des juges sur Apple silicon |
| `onnxruntime`, `tokenizers`, `numpy`, `huggingface_hub` | `s1d1` | classifieur s1d1 en ONNX int8 (CPU) |
| `torch`, `transformers` | `s1d1-torch` | classifieur s1d1 en PyTorch fp16 |
| `laya` (0.4.15) | `laya` | classifieur Laya |
| `matplotlib` (et `numpy`, qu'il apporte) | `report` | figures SVG de `bench report` et statistiques de `metrics.py` |
| aucune | aucun | classifieur `litellm-heuristic` : code de LiteLLM copié dans le dépôt, sans le paquet `litellm` |
| `pytest` | `dev` | tests, sans réseau et sans modèle |
| `ruff` | `dev` | lint et tri des imports |

## Licences

- Code : [Apache-2.0](LICENSE), **sauf** `src/s1d1_bench/vendor/litellm_heuristic/`, copie mot pour mot du scorer
  heuristique de LiteLLM au tag v1.106.6, sous licence MIT (licence et notice des changements dans le dossier).
- Prompts d'autres jeux publics : les prompts de `data/pool/*`, de `data/prompts/trial-50.*` et des lots sont repris
  tels quels de Dolly (CC-BY-SA-3.0), d'OASST2 et de m-ArenaHard (Apache-2.0) ; chacun garde sa licence, donnée ligne
  par ligne dans la colonne `licence` de `pool.parquet` et de `<jeu>.meta.parquet`. Le détail est dans
  [data/README.md](data/README.md).
- Arena-Hard : les prompts anglais d'Arena-Hard viennent d'`arena-hard-auto` v0.1 (lmarena, Apache-2.0, prompts
  d'utilisateurs de Chatbot Arena) et leurs traductions de `CohereLabs/m-ArenaHard` (Apache-2.0) ; les conditions de la
  collecte amont auprès des utilisateurs de Chatbot Arena ne sont pas vérifiables en aval.
- [CC-BY-4.0](data/LICENSE) (`data/LICENSE`) : les prompts écrits à la main et leurs traductions, les fichiers de juges
  (`judges/`), les fichiers de classifieurs (`classifiers/`), les résultats (`results/`), le rapport (`report/<jeu>/`),
  la rubrique `src/s1d1_bench/rubric/judge-rubric-v7.txt`, les consignes `consigne/`, les vérifications
  (`data/checks/`), les documents copiés dans `docs/` (voir [docs/README.md](docs/README.md)), la liste d'exclusion
  et les statistiques. Les traductions des prompts d'un jeu public sont des
  œuvres dérivées et gardent la licence de la ligne (colonne `licence` de `meta.parquet`).
