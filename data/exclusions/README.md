# s1d1 seen prompt keys

`s1d1-seen-keys.txt` lists the prompts that the s1d1 model has seen (interim pool and the final
train/validation/test splits), so that a benchmark can exclude them. Generated on 2026-10-28.

- 2,049,748 lines, one key per line, sorted and unique.
- Each key is the first 16 hex characters of the SHA-256 of the NFC-normalized prompt
  (`prompt_key`, see `pipeline/01_interim.py` in s1d1-model-router). No prompt text is stored here.
- It is the union of `prompt_key` over `dataset/interim/*.parquet` and `dataset/final/*.parquet`.

Command, run from the s1d1-model-router repo (branch `seen-keys`, `pipeline/13_seen_keys.py`):

```
uv run python -I pipeline/13_seen_keys.py --inputs 'dataset/interim/*.parquet' 'dataset/final/*.parquet' \
    --out /Users/steve/work/s1d1-bench/data/exclusions/s1d1-seen-keys.txt
```
