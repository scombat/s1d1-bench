"""The judge rubric (v7) shipped with the package, and the label vocabulary read from it."""

import re
from importlib import resources

LEVELS = ["trivial", "low", "medium", "high", "expert"]  # `null` (cannot be judged) is handled by callers

_RUBRIC_PATH = ("rubric", "judge-rubric-v7.txt")


def load_rubric() -> str:
    """The rubric text, byte-exact (no newline translation), whether run from a checkout or a wheel."""
    data = resources.files("s1d1_bench").joinpath(*_RUBRIC_PATH).read_bytes()
    return data.decode("utf-8")


def _task_names(rubric: str) -> list[str]:
    # only the first block of the `## task` section (up to the first blank line): the tie-breaking block that
    # follows repeats some of these names
    match = re.search(r"^## task\b[^\n]*\n(.*?)(?:\n\s*\n|\Z)", rubric, flags=re.DOTALL | re.MULTILINE)
    if match is None:
        raise ValueError("the rubric has no `## task` section")
    return re.findall(r"^- (\w+):", match.group(1), flags=re.MULTILINE)


TASKS: list[str] = _task_names(load_rubric())
