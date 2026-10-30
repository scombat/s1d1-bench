"""The `bench` command line: one subcommand per step of the kit."""

import argparse
import sys
from importlib import import_module
from pathlib import Path

# subcommand -> (module of s1d1_bench, function taking the parsed arguments), imported lazily so that `--help`
# and the other subcommands never pay for (or break on) the heavy dependencies of a module
COMMANDS: dict[str, tuple[str, str]] = {
    "pool": ("pool", "run_pool"),
    "export": ("export", "run_export"),
    "judge": ("judge_runner", "run_judge_cmd"),
    "validate": ("judge_file", "run_validate"),
    "agree": ("agree", "run_agree"),
    "draw": ("draw", "run_draw"),
    "translate": ("translate", "run_translate_cmd"),
    "witness": ("witness", "run_witness"),
    "check": ("check", "run_check"),
    "classify": ("classify", "run_classify_cmd"),
}

_HELP = {
    "pool": "build the prompt pool",
    "export": "export the prompts to judge",
    "judge": "run a local judge over the prompts",
    "validate": "check a judge or classifier directory",
    "agree": "measure how often judges agree, pair by pair",
    "draw": "draw the base prompts, stratified on a pre-sort judge file",
    "translate": "translate a prompts file into one language",
    "witness": "fetch the m-ArenaHard translations of the Arena-Hard prompts of a set",
    "check": "flag the translations of a set for a human review (structure, labels, same request, review sheet)",
    "classify": "write a classifier folder: a witness (random, length, constant) or a system (s1d1, laya)",
}


def _handler(module: str, function: str):
    def run(args: argparse.Namespace) -> int:
        full_name = f"s1d1_bench.{module}"
        try:
            target = getattr(import_module(full_name), function)
        except ModuleNotFoundError as exc:
            if exc.name != full_name:
                raise
            print(f"bench {args.command}: not implemented yet", file=sys.stderr)
            return 2
        return target(args) or 0

    return run


def _pool_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--out", type=Path, default=Path("data/pool/pool.parquet"), help="pool file (parquet)")
    sub.add_argument(
        "--exclusions", type=Path, default=Path("data/exclusions/s1d1-seen-keys.txt"), help="prompt keys to leave out"
    )
    sub.add_argument(
        "--candidates", type=Path, default=Path("data/pool/candidates-eng.jsonl"), help="English candidates (jsonl)"
    )
    sub.add_argument("--per-source", type=int, default=300, help="English candidates drawn from each source")
    sub.add_argument("--seed", type=int, default=0, help="seed of the candidate draw")


def _export_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--pool", type=Path, default=Path("data/pool/pool.parquet"), help="pool file (parquet)")
    sub.add_argument("--select", type=Path, required=True, help="selected prompt keys, one per line")
    sub.add_argument("--handwritten", type=Path, default=None, help="handwritten prompts (jsonl)")
    sub.add_argument("--langs", default="eng", help="comma-separated language codes (default: eng)")
    sub.add_argument(
        "--translations",
        type=Path,
        action="append",
        default=None,
        help="folder of <lang>.jsonl translations and their corrections; repeat it for each wave (needed beyond eng)",
    )
    sub.add_argument("--name", required=True, help="name of the set: files are data/prompts/<name>.*")
    sub.add_argument("--seed", type=int, default=0, help="seed of the shuffle")
    sub.add_argument("--root", type=Path, default=Path("."), help=argparse.SUPPRESS)  # for tests and smoke runs


def _judge_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--model", required=True, help="Hub repo id (weights in ~/mlx-models/<org>--<name>) or a directory"
    )
    sub.add_argument("--prompts", type=Path, required=True, help="prompts to judge (jsonl of id, prompt)")
    sub.add_argument("--out", type=Path, required=True, help="judge directory, judges/<name>")
    sub.add_argument("--author", required=True, help="who contributes the judge file")
    sub.add_argument("--thinking", required=True, help="off, or cap:<n> tokens of reasoning (see judges/MODELS.md)")
    sub.add_argument("--max-tokens", type=int, default=1024, help="tokens generated per call, reasoning included")
    sub.add_argument("--resume", action="store_true", help="go on with an interrupted run in --out")
    sub.add_argument("--judge", default=None, help="name of the judge in meta.json (default: the --out folder name)")
    sub.add_argument("--notes", default="", help="free text for meta.json")


def _validate_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("judge_dir", type=Path, help="judge or classifier directory, judges/<name> or classifiers/<name>")
    sub.add_argument("--prompts", type=Path, default=Path("data/prompts"), help="directory of the prompts files")


def _agree_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("judge_dirs", nargs="*", type=Path, help="judge directories (at least two), e.g. judges/trial-*")


def _draw_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--presort", type=Path, required=True, help="pre-sort judge directory, judges/<name>")
    sub.add_argument("--meta", type=Path, required=True, help="meta.parquet of the pre-sorted set (id -> base_key)")
    sub.add_argument("--out", type=Path, required=True, help="selected base keys, one per line (+ a .draw.json record)")
    sub.add_argument("--veto", type=Path, default=None, help="vetoed ids, '<id><TAB><reason>' per line")
    sub.add_argument("--seed", type=int, default=0, help="seed of the per-stratum shuffle")
    sub.add_argument("--per-stratum", type=int, default=20, help="ids drawn per stratum (trivial, low, medium, high)")
    sub.add_argument("--tools-min", type=int, default=20, help="needs_tools true ids wanted in all")
    sub.add_argument("--task-min", type=int, default=4, help="ids wanted per task of the rubric")


def _translate_arguments(sub: argparse.ArgumentParser) -> None:
    from s1d1_bench.translate import LANGUAGES  # light: translate imports mlx only when it runs

    sub.add_argument(
        "--model", required=True, help="Hub repo id (weights in ~/mlx-models/<org>--<name>) or a directory"
    )
    sub.add_argument("--prompts", type=Path, required=True, help="English prompts (jsonl of id, prompt; ids end -eng)")
    sub.add_argument("--lang", required=True, choices=list(LANGUAGES), help="target language (ISO 639-3)")
    sub.add_argument(
        "--consigne", type=Path, default=None, help="system message (default: consigne/translate-v1-<lang>.md)"
    )
    sub.add_argument("--out", type=Path, required=True, help="translations directory, data/translations/<set>")
    sub.add_argument("--author", required=True, help="who contributes the translations")
    sub.add_argument("--thinking", required=True, help="off, or cap:<n> tokens of reasoning (see judges/MODELS.md)")
    sub.add_argument(
        "--max-tokens", type=int, default=4096, help="tokens generated per call, reasoning included (default: 4096)"
    )
    sub.add_argument("--resume", action="store_true", help="go on with an interrupted run in --out")
    sub.add_argument("--notes", default="", help="free text for <lang>.meta.json")


def _witness_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--meta", type=Path, required=True, help="meta.parquet of the prompts set (source, source_id)")
    sub.add_argument("--langs", default="fra,rus,zho", help="comma-separated language codes (default: fra,rus,zho)")
    sub.add_argument("--out", type=Path, required=True, help="witness directory, data/translations/<set>/witness-arena")
    sub.add_argument("--cache", type=Path, default=Path("data/pool/cache"), help="Hub download cache")


def _check_arguments(sub: argparse.ArgumentParser) -> None:
    from s1d1_bench.translate import LANGUAGES  # light, as for `translate`; check itself pulls pyarrow

    sub.add_argument(
        "kind",
        choices=["structure", "labels", "request", "review"],
        help="structure and labels write <lang>.<kind>.jsonl; review gathers them into <lang>.review.md",
    )
    sub.add_argument("--set", required=True, help="name of the set: files are <out>/<set>/<lang>.*")
    sub.add_argument("--lang", required=True, choices=list(LANGUAGES), help="language checked (ISO 639-3)")
    sub.add_argument("--english", type=Path, required=True, help="English prompts (jsonl of id, prompt; ids end -eng)")
    sub.add_argument("--translations", type=Path, required=True, help="folder of <lang>.jsonl translations")
    sub.add_argument("--witness", type=Path, default=None, help="folder of <lang>.jsonl witness translations (review)")
    sub.add_argument("--labels-english", type=Path, default=None, help="judge folder of the English prompts (labels)")
    sub.add_argument("--labels", type=Path, default=None, help="judge folder of the translated prompts (labels)")
    sub.add_argument(
        "--labels-witness", type=Path, default=None, help="judge folder of the witness translations (labels)"
    )
    sub.add_argument("--out", type=Path, default=Path("data/checks"), help="checks directory (default: data/checks)")
    # request only: the local model that answers "same request?" (optional here, so the other kinds need none)
    sub.add_argument(
        "--model", default=None, help="request: Hub repo id (weights in ~/mlx-models/<org>--<name>) or a directory"
    )
    sub.add_argument("--author", default=None, help="request: who runs the check")
    sub.add_argument("--thinking", default=None, help="request: off, or cap:<n> tokens of reasoning")
    sub.add_argument(
        "--max-tokens", type=int, default=512, help="request: tokens generated per call, reasoning included (512)"
    )
    sub.add_argument(
        "--consigne",
        type=Path,
        default=Path("consigne/check-same-request-v1.md"),
        help="request: system message (default: consigne/check-same-request-v1.md)",
    )
    sub.add_argument("--resume", action="store_true", help="request: go on with an interrupted run")
    sub.add_argument("--notes", default="", help="request: free text for <lang>.request.meta.json")


def _classify_arguments(sub: argparse.ArgumentParser) -> None:
    from s1d1_bench.rubric import LEVELS  # light, as above

    sub.add_argument(
        "adapter",
        choices=["random", "length", "constant", "s1d1", "laya"],
        help="random, length, constant: the witnesses; s1d1, laya: the systems under test",
    )
    sub.add_argument("--prompts", type=Path, required=True, help="prompts to classify (jsonl of id, prompt)")
    sub.add_argument(
        "--out", type=Path, required=True, help="classifier directory, classifiers/<name> (must not exist)"
    )
    sub.add_argument("--author", required=True, help="who contributes the classifier file")
    sub.add_argument("--notes", default="", help="free text for meta.json")
    sub.add_argument("--seed", type=int, default=0, help="random: seed of the draws")
    sub.add_argument("--draws", type=int, default=10, help="random: number of draws (labels.jsonl is draw 0)")
    sub.add_argument("--level", choices=LEVELS, default="low", help="constant: the tier given to every prompt")
    sub.add_argument("--model", default=None, help="s1d1: Hub repo id of the weights")
    sub.add_argument("--revision", default=None, help="s1d1: Hub revision of the weights (default: the current one)")
    sub.add_argument("--backend", default="auto", help="s1d1: torch, onnx or auto")
    sub.add_argument(
        "--laya-model", choices=["english", "multilingual"], default=None, help="laya: the checkpoint to question"
    )


# subcommand -> function adding its arguments to its parser
_ARGUMENTS = {
    "pool": _pool_arguments,
    "export": _export_arguments,
    "judge": _judge_arguments,
    "validate": _validate_arguments,
    "agree": _agree_arguments,
    "draw": _draw_arguments,
    "translate": _translate_arguments,
    "witness": _witness_arguments,
    "check": _check_arguments,
    "classify": _classify_arguments,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="s1d1 open bench kit")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{" + ",".join(COMMANDS) + "}")
    for name, (module, function) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=_HELP[name], description=_HELP[name])
        _ARGUMENTS.get(name, lambda _sub: None)(sub)
        sub.set_defaults(handler=_handler(module, function))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
