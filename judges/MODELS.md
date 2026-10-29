# Local judges on MLX: what was measured (2026-10-28/29)

The three local judges chosen by the user (docs/32 § 7, decision 5): **Gemma 4 31B-it, Qwen3.8 27B, gpt-oss-20b**.
All three load and run with the pinned `mlx-lm` (0.32.2, mlx 0.32.4) on the bench Mac (M4 Max, 128 GB), one at a
time. On the 10 test prompts, all three return JSON that `parse_label` reads, 10 out of 10, with thinking off and
with thinking capped at 512 tokens, and greedy decoding gives the same text twice.

These are 10-prompt measurements, not a bench run: the 50-prompt timing comes next.

## Summary

| | Gemma 4 31B-it | Qwen3.8 27B | gpt-oss-20b |
|---|---|---|---|
| repo (`--model`) | `mlx-community/gemma-4-31b-it-8bit` | `mlx-community/Qwen3.8-27B-8bit` | `mlx-community/gpt-oss-20b-MXFP4-Q8` |
| revision | `f5f3dc92ab4af76724c36c21eb6bedadb3a851be` | `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` | `773a7da77e569019bb0fd17a554b263738d669a3` |
| base model | `google/gemma-4-31B-it` | `Qwen/Qwen3.8-27B` | `openai/gpt-oss-20b` |
| quantization (meta) | `8bit-gs64` | `8bit-gs64` | `mxfp4-gs32+8bit-gs64` (see its section) |
| size on disk | 33.0 GB (31 GiB) | 28.8 GB (28 GiB) | 11.8 GB (11 GiB) |
| licence (model cards) | Apache-2.0 | Apache-2.0 | Apache-2.0, plus a usage policy |
| load time (3 loads each) | 3.2–7.2 s | 2.8–6.4 s | 1.4–3.2 s |
| peak memory | 35.3 GB (MLX), 37.6 GB footprint | 31.4 GB (MLX), 32.7 GB footprint | 13.0 GB (MLX), 13.6 GB footprint |
| prompt processing | ≈ 190–210 tokens/s | ≈ 230–250 tokens/s | ≈ 1 600–1 800 tokens/s |
| decoding | ≈ 13 tokens/s | ≈ 16–17 tokens/s | ≈ 120–128 tokens/s |
| **off**: median tokens/s (completion / wall) | 1.8 | 2.1 | 14.3 |
| **off**: median completion tokens · wall per prompt | 21 · 11.5 s | 21 · 10.0 s | 17 · 1.2 s |
| **off**: unreadable JSON (json_error + truncated) | 0 / 10 | 0 / 10 | 0 / 10 |
| **cap:512**: median tokens/s (completion / wall) | 8.6 | 9.8 | 43.2 |
| **cap:512**: median · max completion tokens · wall per prompt | 282 · 533 · 31.6 s | 201 · 285 · 20.0 s | 74 · 197 · 1.7 s |
| **cap:512**: unreadable JSON | 0 / 10 | 0 / 10 | 0 / 10 |
| same text twice at temperature 0 | yes | yes | yes |

"tokens/s" is the figure `bench judge` prints: the median over prompts of `completion_tokens / wall_s`, where
`wall_s` covers prompt processing and decoding (not loading). With thinking off the answer is about 20 tokens and the
~1 900-token prompt (the rubric is most of it) takes most of the time, so this figure is low for Qwen and Gemma
even though they decode at 13–17 tokens/s: about 9–10 s of their 10–11 s per prompt is spent processing the prompt.
Projection for 400 prompts, from the mean wall time per prompt: Gemma ≈ 1.3 h off and ≈ 3.8 h at cap:512; Qwen ≈ 1.1 h
off and ≈ 2.3 h at cap:512; gpt-oss ≈ 8 min off and ≈ 12 min at cap:512.

Trial on 50 prompts ([TRIAL.md](TRIAL.md)): one unreadable answer in 300 judge calls, for gpt-oss-20b with thinking
off (1 of 50, 2 %, under the 10 % bar). The model answered `"task":"design"` twice, which is not one of the 19 task
names, and greedy decoding gives the same answer on a resume. No label was filled in, and the incomplete folder
`judges/trial-gptoss-off` was removed from the repository by the user's decision; the other five trial folders are
complete.

"Peak memory" is `mx.get_peak_memory()` in the exploration script (MLX) and `peak memory footprint` from
`/usr/bin/time -l` around `bench judge` (whole process).

## How `bench judge` drives each family

`thinking` is `off` or `cap:<n>`. The backend returns the whole assistant turn, reasoning included and special
tokens kept, as `GenerateResult.text`; `parse_label` strips the reasoning (`judge_runner._strip_reasoning`, which now
also knows the Gemma 4 markers). `completion_tokens` counts every token the model generated, reasoning included (an
injected closing marker is not counted). `finish_reason` is `length` when `--max-tokens` (default 1024, reasoning
included) is reached.

**The cap.** The backend counts the tokens generated after the opening marker (or from the first token when the chat
template already opened the reasoning, as Qwen's does). If the reasoning reaches `n` tokens without closing, it stops,
appends the family's closing text to prompt + generated tokens, and generates the answer within what is left of
`--max-tokens`. The whole prompt is processed again for that second generation (no cache reuse), so a capped prompt
costs one more prompt-processing pass (≈ 10 s for Gemma and Qwen). Checked on the real models with a tiny cap
(`cap:16` / `cap:32`, 2 prompts each): the closing text is injected and the answer that follows parses, for all
three. This is the bench's own cap: the training judge's 512 cap was vLLM's `thinking_token_budget`, which may cut
and close slightly differently.

| family | `model_type` | off | cap:<n> | opens | closes (as written by the model) | appended at the cap |
|---|---|---|---|---|---|---|
| `gemma4` | `gemma4` | `enable_thinking=False` | `enable_thinking=True` | `<\|channel>thought` | `<channel\|>` | `<channel\|>` |
| `qwen3` | `qwen3_5` | `enable_thinking=False` | `enable_thinking=True, reasoning_effort="low"` | `<think>` (in the prompt) | `</think>` | `\n</think>\n\n` |
| `gpt_oss` | `gpt_oss` | `reasoning_effort="low"` + empty analysis prefilled | `reasoning_effort="low"` | `<\|channel\|>analysis<\|message\|>` | `<\|end\|>` | `<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>` |

A cap is applicable to all three. An unknown `model_type` is refused.

**What meta.json records.** `model` is always the Hub repo id: `--model` takes that id, or a directory of
`~/mlx-models/` named `org--name` (recorded as `org/name`); any other directory is refused, and `bench judge` refuses
a meta.json that would hold a local path. `replayable` is `true` only when the revision is a known Hub commit
(otherwise `false`, and `notes` says why). `notes` starts with one or two sentences on what the `thinking` setting
does for the family (the table above, plus the pinned date for gpt-oss), followed by the `--notes` text.
`max_tokens` is recorded too. A `--resume` is refused unless model, revision, thinking, prompts, quantization,
mlx-lm version and max tokens all match the meta.json of the run it resumes.

## Gemma 4 31B-it — runs

- Repo `mlx-community/gemma-4-31b-it-8bit` @ `f5f3dc92ab4af76724c36c21eb6bedadb3a851be`, converted from
  `google/gemma-4-31B-it` (the training judge, BF16 revision `842da3794eaa0b77d5f08bae87a17459d91ff475` in
  pipeline/03_label_job.py). 8-bit affine, group size 64. A ready-made 8-bit repo exists, so nothing was converted.
  The checkpoint is the multimodal one (`Gemma4ForConditionalGeneration`); mlx-lm loads the language model only.
- Licence: Apache-2.0 on both the Google card and the mlx-community card.
- Reasoning: the template puts `<|think|>` at the top of the system turn when `enable_thinking=True`; the model then
  writes `<|channel>thought\n…<channel|>` before the answer. With `enable_thinking=False` the template itself ends the
  prompt with an empty `<|channel>thought\n<channel|>`, and the model answers with the JSON alone.
- Uncapped thinking on the 10 prompts (exploration): median 287 completion tokens (reasoning + ~20-token answer), max
  739 ("Please repeat the string…"); so 512 does bind sometimes (in the cap:512 run, two prompts generated more than
  512 completion tokens, max 533).
- Example outputs:
  - off: `{"complexity": "trivial", "task": "information_seeking", "needs_tools": false}`
  - cap:512: `<|channel>thought\n*   Prompt: "How many syllables are in the word smarter?"\n    *   Complexity: … *   Needs Tools: \`false\`.<channel|>{"complexity": "low", "task": "information_seeking", "needs_tools": false}`
  - cap:32: `<|channel>thought\n*   Prompt: "Rank countries with highest population"\n    *   Task: The user wants a list of countries ordered by population. This is asking<channel|>{"complexity": "low", "task": "information_seeking", "needs_tools": false}`

## Qwen3.8 27B — runs

- Repo `mlx-community/Qwen3.8-27B-8bit` @ `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9`, from `Qwen/Qwen3.8-27B`.
  8-bit affine, group size 64. The checkpoint is the vision-language one (`Qwen3_5ForConditionalGeneration`, hybrid
  Gated DeltaNet + attention); mlx-lm loads the language model only (`qwen3_5` in mlx-lm 0.32.2).
- Licence: Apache-2.0 on both cards.
- Reasoning: `<think>…</think>`. With `enable_thinking=False` the template ends the prompt with
  `<think>\n\n</think>\n\n`; with `enable_thinking=True` it ends with `<think>\n` (so the generated text has only the
  closing tag, which `parse_label` already handled). **The template's default reasoning effort is `xhigh`** (it adds an
  instruction to think carefully at the top of the system message); `bench judge` passes `reasoning_effort="low"`
  for a cap, as docs/05 did for Qwen (`low256`). With `low`, the template prepends its own sentence to the system
  turn ("Reasoning effort is set to low. Keep your thinking brief…") before the rubric; the message handed to the
  template is still the rubric byte for byte. With low effort the uncapped runs were 137–285 completion tokens
  (reasoning + ~20-token answer): 512 never binds on these 10 prompts.
- Example outputs:
  - off: `{"complexity": "trivial", "task": "other", "needs_tools": false}`
  - cap:512: `The prompt asks to repeat a specific string "?????-?????-" back to the user. This is a trivial task … simple mechanical task of repeating a string.\n</think>\n\n{"complexity":"trivial","task":"other","needs_tools":false}`
  - cap:32: `The user is asking about how to generate power in a roundhouse kick in Muay Thai. This is a question about martial arts technique - specifically the biomechan\n</think>\n\n{"complexity":"low","task":"information_seeking","needs_tools":false}`

## gpt-oss-20b — runs, with two points for the user

- Repo `mlx-community/gpt-oss-20b-MXFP4-Q8` @ `773a7da77e569019bb0fd17a554b263738d669a3`, from `openai/gpt-oss-20b`.
- **Quantization: there is no plain 8-bit.** OpenAI released gpt-oss with its MoE experts already in MXFP4; this
  repo keeps the experts in MXFP4 (group 32) and puts attention, embeddings, router and output head in 8-bit affine
  (group 64), hence `mxfp4-gs32+8bit-gs64` in meta. It is the closest to "8 bits" on the Hub
  (mlx-community also has `gpt-oss-20b-MXFP4-Q4` and `gpt-oss-20b-mxfp4-bf16`).
- Licence: Apache-2.0 on both cards; the OpenAI repo also ships a `USAGE_POLICY` file.
- **Thinking "off" is not native.** The harmony template only has `reasoning_effort` (low / medium / high, default
  medium), no off switch. `bench judge` makes "off" by prefilling an empty analysis channel and opening the final one
  after the generation prompt (`<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>`),
  with `reasoning_effort="low"`; the model then writes the JSON alone (10/10). Whether that counts as "thinking off"
  for the bench is the user's call. For a cap, `reasoning_effort="low"`: uncapped runs on the 10 prompts were
  54–197 completion tokens (analysis + ~20-token answer) at low (90–521 at medium), so 512 does not bind at low.
- **The template prints today's date** in the system message (`Current date: …`), which would make a run depend on
  the day it was made. `bench judge` pins it to 2026-10-28 (`strftime_now` passed to the template). The template
  also adds its own system message (`You are ChatGPT, a large language model trained by OpenAI.`, knowledge cutoff,
  reasoning level); the rubric goes in the developer message, under `# Instructions`.
- Example outputs:
  - off: `{"complexity":"trivial","task":"information_seeking","needs_tools":true}`
  - cap:512: `<|channel|>analysis<|message|>We need to label. It's asking for ranking countries by population. That's information seeking. Needs up-to-date data. So needs_tools true. Complexity medium? …<|end|><|start|>assistant<|channel|>final<|message|>{"complexity":"medium","task":"information_seeking","needs_tools":true}`
  - cap:16: `<|channel|>analysis<|message|>We need to label. It's asking for ranking countries by population. That's information seeking<|end|><|start|>assistant<|channel|>final<|message|>{"complexity":"low","task":"information_seeking","needs_tools":true}`

## Other observations

- Gemma's and Qwen's templates `trim` the system message, so the rubric's trailing blank lines are not sent; the
  message passed to the template is still the rubric byte for byte.
- The prompt is ~1 860–1 900 tokens and almost all of it is the rubric, identical on every call. Reusing its
  processed state across prompts would cut the "off" wall time of Gemma and Qwen several times over; not done here.
- The labels differ between judges and between off and cap:512 on several of the 10 prompts (for example
  `needs_tools` on "Rank countries with highest population"), as expected from judges that are compared to each
  other.

## How this was measured (to reproduce)

Prompts: lines 0–4 (Dolly) and 300–304 (Arena-Hard) of `data/pool/candidates-eng.jsonl`, in that order, as a
10-line jsonl (sha256 `a33442a84f4886274e0ade779b4dc9be163c2dee3e648992f77ea0b859b7f3ca`).

```sh
# download (public Hub, no token), one at a time, pinned revision
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download mlx-community/gemma-4-31b-it-8bit \
  --revision f5f3dc92ab4af76724c36c21eb6bedadb3a851be --local-dir ~/mlx-models/mlx-community--gemma-4-31b-it-8bit
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download mlx-community/Qwen3.8-27B-8bit \
  --revision 815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9 --local-dir ~/mlx-models/mlx-community--Qwen3.8-27B-8bit
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download mlx-community/gpt-oss-20b-MXFP4-Q8 \
  --revision 773a7da77e569019bb0fd17a554b263738d669a3 --local-dir ~/mlx-models/mlx-community--gpt-oss-20b-MXFP4-Q8

# the measurement: one model process at a time, for thinking off and cap:512
uv run --group judge /usr/bin/time -l bench judge --model mlx-community/gemma-4-31b-it-8bit \
  --prompts prompts10.jsonl --out <scratch>/gemma-off --author steve --thinking off --max-tokens 1024
# … same for cap:512 and for the two other models; tokens/s, counts and load time are printed by bench judge,
# per-prompt figures are in <out>/log.jsonl

# greedy repeatability, per model
S1D1_BENCH_MODEL=mlx-community/gemma-4-31b-it-8bit uv run --group judge pytest -m slow -v
```

Before the backend existed, an exploration script (mlx-lm `load`, `apply_chat_template`, `stream_generate` with
`make_sampler(temp=0.0)`) ran the same 10 prompts per thinking variant, decoded with special tokens kept, and gave the
prompt-processing and decoding rates, uncapped reasoning lengths and the MLX peak memory above.
