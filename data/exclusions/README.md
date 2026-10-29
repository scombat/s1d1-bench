# s1d1 seen prompt keys

`s1d1-seen-keys.txt` lists the prompts that the s1d1 model has seen (the final
train/validation/test splits), so that a benchmark can exclude them. Generated on 2026-10-29.

- 112,947 lines, one key per line, sorted and unique.
- Each key is the first 16 hex characters of the SHA-256 of the NFC-normalized prompt
  (`prompt_key`, see `pipeline/01_interim.py` in s1d1-model-router). No prompt text is stored here.
- It is the union of `prompt_key` over `dataset/final/*.parquet` (train 98,033, validation 4,914, test 10,000). The
  larger `dataset/interim` pool is not excluded: its prompts were not all used for training, and excluding it
  removed almost every OASST2 prompt from the bench pool.

Command, run from the s1d1-model-router repo (branch `seen-keys`, `pipeline/13_seen_keys.py`):

```
uv run python -I pipeline/13_seen_keys.py --inputs 'dataset/final/*.parquet' \
    --out data/exclusions/s1d1-seen-keys.txt
```
