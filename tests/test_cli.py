import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from s1d1_bench.cli import build_parser, main
from s1d1_bench.rubric import LEVELS, TASKS, load_rubric

# sha256 of docs/judge-rubric-v7.txt of the research notebook, the source of the packaged copy
RUBRIC_SHA256 = "fd9f87bd0b390d94c675ddc2807affd9d1323dae0e752b830fe0e5bbb2d5ff79"


def test_cli_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for name in ("pool", "export", "judge", "validate", "agree"):
        assert name in out


def test_rubric_is_byte_identical_and_lists_19_tasks():
    text = load_rubric()
    assert hashlib.sha256(text.encode()).hexdigest() == RUBRIC_SHA256
    assert len(TASKS) == 19 and "information_seeking" in TASKS and "coding_debugging" in TASKS


def test_tasks_are_unique_and_in_file_order():
    assert len(set(TASKS)) == len(TASKS)
    assert TASKS[0] == "information_seeking" and TASKS[-1] == "other"
    positions = [load_rubric().index(f"\n- {name}: ") for name in TASKS]
    assert positions == sorted(positions)


def test_levels():
    assert LEVELS == ["trivial", "low", "medium", "high", "expert"]


def test_report_help_lists_its_options(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "report" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["report", "--help"])
    out = capsys.readouterr().out
    for option in ("--results", "--meta", "--out", "--sigma", "--classifiers", "--template"):
        assert option in out
    args = build_parser().parse_args(["report", "--results", "r.parquet", "--meta", "m.parquet", "--out", "o"])
    assert (args.sigma, args.classifiers, args.template) == (0.005, Path("classifiers"), Path("report/TEMPLATE.md"))


def test_the_command_line_and_the_report_modules_load_without_matplotlib_nor_numpy():
    # the command line: neither (bench --help and the other commands); the report modules: matplotlib only on drawing
    code = "import sys, s1d1_bench.cli as c; c.build_parser(); assert not {'numpy', 'matplotlib'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
    code = "import sys, s1d1_bench.report, s1d1_bench.report_figures, s1d1_bench.report_index; "
    code += "assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
