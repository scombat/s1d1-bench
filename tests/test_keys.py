import hashlib
import unicodedata
from pathlib import Path

from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.keys import load_exclusions, normalize, prompt_key

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEN_KEYS = REPO_ROOT / "data" / "exclusions" / "s1d1-seen-keys.txt"

# A short public prompt taken from the research test set (s1d1-model-router dataset/final/test.parquet),
# with its prompt_key as computed by the research pipeline.
KNOWN_SEEN_PROMPT = "Bonjour est-ce que tu parles français?"
KNOWN_SEEN_KEY = "664ef43d18278220"


def test_normalize_matches_research_pipeline():
    assert normalize("  é\r\nx ") == "é\nx"  # NFC, CRLF, strip


def test_key_is_16_hex_of_sha256():
    assert prompt_key("hello") == hashlib.sha256(b"hello").hexdigest()[:16]


def test_key_of_a_known_test_prompt_is_excluded():
    assert KNOWN_SEEN_KEY in load_exclusions(SEEN_KEYS)


def test_key_of_known_prompt_matches_research_key():
    assert prompt_key(normalize(KNOWN_SEEN_PROMPT)) == KNOWN_SEEN_KEY


def test_nfd_and_crlf_forms_get_the_research_key():
    nfd_crlf = unicodedata.normalize("NFD", KNOWN_SEEN_PROMPT).replace(" ", "\r\n", 1)
    assert nfd_crlf != KNOWN_SEEN_PROMPT
    assert prompt_key(normalize(nfd_crlf)) == prompt_key(normalize(KNOWN_SEEN_PROMPT.replace(" ", "\n", 1)))
    nfd = unicodedata.normalize("NFD", KNOWN_SEEN_PROMPT)
    assert nfd != KNOWN_SEEN_PROMPT
    assert prompt_key(normalize(f"  {nfd}\r\n")) == KNOWN_SEEN_KEY
    assert prompt_key(normalize("é\r\nx")) == prompt_key(normalize("é\nx"))
    assert prompt_key(normalize("é\nx")) == prompt_key(normalize("é\nx"))


def test_load_exclusions_strips_and_skips_blank_lines(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("aaaa\n\n  bbbb  \r\n\ncccc", encoding="utf-8")
    assert load_exclusions(path) == frozenset({"aaaa", "bbbb", "cccc"})


def test_jsonl_round_trip_with_non_ascii(tmp_path):
    rows = [{"id": "a", "text": "é 日本語 ✓\nsecond line"}, {"id": "b", "n": 2}]
    path = tmp_path / "out.jsonl"
    assert write_jsonl(path, iter(rows)) == 2
    raw = path.read_bytes().decode("utf-8")
    assert "日本語" in raw  # ensure_ascii=False
    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 2
    assert read_jsonl(path) == rows
    assert read_jsonl(str(path)) == rows  # str accepted


def test_write_jsonl_accepts_str_path_and_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    assert write_jsonl(str(path), []) == 0
    assert read_jsonl(path) == []


def test_read_jsonl_skips_only_empty_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_jsonl_keeps_unicode_line_separators_inside_strings(tmp_path):
    rows = [{"t": "a b\u0085c\x0bd\re"}, {"t": "z"}]
    path = tmp_path / "sep.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
