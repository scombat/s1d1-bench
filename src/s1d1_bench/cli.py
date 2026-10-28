"""The `bench` command line: one subcommand per step of the kit."""

import argparse
import sys
from importlib import import_module

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
    "agree": "measure the agreement of a classifier with the judges",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="s1d1 open bench kit")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{" + ",".join(COMMANDS) + "}")
    for name, (module, function) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=_HELP[name], description=_HELP[name])
        sub.set_defaults(handler=_handler(module, function))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
