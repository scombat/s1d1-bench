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
}

_HELP = {
    "pool": "build the prompt pool",
    "export": "export the prompts to judge",
    "judge": "run a local judge over the prompts",
    "validate": "check a judge directory",
    "agree": "measure how often judges agree, pair by pair",
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
    sub.add_argument("judge_dir", type=Path, help="judge directory, judges/<name>")
    sub.add_argument("--prompts", type=Path, default=Path("data/prompts"), help="directory of the prompts files")


def _agree_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("judge_dirs", nargs="*", type=Path, help="judge directories (at least two), e.g. judges/trial-*")


# subcommand -> function adding its arguments to its parser
_ARGUMENTS = {
    "pool": _pool_arguments,
    "export": _export_arguments,
    "judge": _judge_arguments,
    "validate": _validate_arguments,
    "agree": _agree_arguments,
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
