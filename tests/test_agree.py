import pytest

from s1d1_bench.agree import pairwise_agreement
from s1d1_bench.cli import main
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.judge_file import LABEL_KEYS


def L(level, task="chat", needs_tools=False):
    return {"complexity": level, "task": task, "needs_tools": needs_tools}


def test_pairwise_agreement_counts_exact_and_within_one():
    r = pairwise_agreement(
        {
            "a": {"1": L("low"), "2": L("high"), "3": L(None)},
            "b": {"1": L("low"), "2": L("medium"), "3": L(None)},
        }
    )
    assert r[("a", "b")] == {"n": 3, "tier_exact": 2 / 3, "tier_within_1": 1.0, "task": 1.0, "needs_tools": 1.0}


def test_only_common_ids_are_compared():
    r = pairwise_agreement(
        {
            "a": {"1": L("low"), "2": L("high"), "only-a": L("expert")},
            "b": {"1": L("low"), "2": L("trivial"), "only-b": L("trivial")},
        }
    )
    assert r[("a", "b")]["n"] == 2
    assert r[("a", "b")]["tier_exact"] == 0.5
    assert r[("a", "b")]["tier_within_1"] == 0.5


def test_null_against_a_level_is_a_disagreement_even_within_one():
    r = pairwise_agreement({"a": {"1": L(None), "2": L(None)}, "b": {"1": L("trivial"), "2": L(None)}})
    assert r[("a", "b")]["tier_exact"] == 0.5
    assert r[("a", "b")]["tier_within_1"] == 0.5


def test_task_and_needs_tools_are_compared_exactly():
    r = pairwise_agreement(
        {
            "a": {"1": L("low", "chat", False), "2": L("low", "coding", True)},
            "b": {"1": L("low", "chat", True), "2": L("low", "writing", True)},
        }
    )
    assert r[("a", "b")]["task"] == 0.5
    assert r[("a", "b")]["needs_tools"] == 0.5


def test_within_one_is_an_index_distance_of_at_most_one():
    r = pairwise_agreement(
        {
            "a": {"1": L("trivial"), "2": L("trivial"), "3": L("expert")},
            "b": {"1": L("low"), "2": L("medium"), "3": L("high")},
        }
    )
    assert r[("a", "b")]["tier_exact"] == 0.0
    assert r[("a", "b")]["tier_within_1"] == pytest.approx(2 / 3)


def test_no_common_id_gives_none_rates():
    r = pairwise_agreement({"a": {"1": L("low")}, "b": {"2": L("low")}})
    assert r[("a", "b")] == {"n": 0, "tier_exact": None, "tier_within_1": None, "task": None, "needs_tools": None}


def test_each_unordered_pair_once_in_iteration_order():
    r = pairwise_agreement({"c": {"1": L("low")}, "a": {"1": L("low")}, "b": {"1": L("low")}})
    assert list(r) == [("c", "a"), ("c", "b"), ("a", "b")]


def _judge_dir(root, name, rows):
    folder = root / name
    folder.mkdir()
    write_jsonl(folder / "labels.jsonl", [dict(zip(LABEL_KEYS, row, strict=True)) for row in rows])
    return folder


def test_cli_agree_prints_a_markdown_table(tmp_path, capsys):
    a = _judge_dir(tmp_path, "trial-a", [("1", "low", "chat", False), ("2", "high", "chat", False)])
    b = _judge_dir(tmp_path, "trial-b", [("1", "low", "chat", False), ("2", "medium", "coding", False)])
    assert main(["agree", str(a), str(b)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "| judge A | judge B | n | tier exact | tier ±1 | task | needs_tools |"
    assert lines[1].startswith("|---") or lines[1].startswith("| ---")
    assert lines[2] == "| trial-a | trial-b | 2 | 50.0% | 100.0% | 50.0% | 100.0% |"


def test_cli_agree_shows_a_dash_when_nothing_is_common(tmp_path, capsys):
    a = _judge_dir(tmp_path, "a", [("1", "low", "chat", False)])
    b = _judge_dir(tmp_path, "b", [("2", "low", "chat", False)])
    assert main(["agree", str(a), str(b)]) == 0
    assert "| a | b | 0 | - | - | - | - |" in capsys.readouterr().out


def test_cli_agree_needs_two_directories(tmp_path, capsys):
    a = _judge_dir(tmp_path, "a", [("1", "low", "chat", False)])
    assert main(["agree", str(a)]) == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "at least two" in err


def test_cli_agree_names_a_directory_without_labels(tmp_path, capsys):
    a = _judge_dir(tmp_path, "a", [("1", "low", "chat", False)])
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["agree", str(a), str(empty)]) == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "empty" in err and "labels.jsonl" in err
