# Licences of the files under `data/`

`data/LICENSE` (CC-BY-4.0) does **not** cover everything in this folder.

## Verbatim prompts of public datasets: each keeps its own licence

These files contain prompts copied verbatim from public datasets. Each prompt keeps the licence of the dataset it
comes from:

| dataset | licence |
|---|---|
| Databricks Dolly | CC-BY-SA-3.0 |
| OpenAssistant OASST2 | Apache-2.0 |
| m-ArenaHard | Apache-2.0 |

Concerned files:

- `data/pool/*` (`pool.parquet`, `candidates-eng.jsonl`, ...);
- `data/prompts/trial-50.*` (the prompts file, its `.sha256`, `.select.txt` and `.meta.parquet`) and the lots under
  `data/prompts/trial-50.lots/`;
- the translations of those prompts (`data/translations/<set>/<lang>.jsonl` and `<lang>.corrections.jsonl`, and the
  `base-v1*` prompts files): a translation is a derivative work and keeps the licence of the prompt it translates,
  given in its row (`licence`); the witness translations `data/translations/<set>/witness-arena/` are m-ArenaHard's
  own Google translations, Apache-2.0.

The licence of each prompt is in its row: column `licence` of `data/pool/pool.parquet` and of
`data/prompts/<set>.meta.parquet` (values `cc-by-sa-3.0` or `apache-2.0`). Reusing these prompts means following
the licence of that row, including attribution and, for CC-BY-SA-3.0, share-alike.

## Covered by `data/LICENSE` (CC-BY-4.0)

- the handwritten prompts (`data/handwritten/`) and their translations (the rows with `origin` handwritten);
- the checks of the translations (`data/checks/`: flags, Gemma 4 verdicts, review sheets);
- the judge files (`judges/`);
- the exclusion list (`data/exclusions/`, hashed keys only, no prompt text);
- the statistics (`data/pool/pool.stats.json`).
