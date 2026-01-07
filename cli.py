from __future__ import annotations

import argparse
from typing import Optional

from nl2spec.pipeline.runner import run_pipeline
from nl2spec.pipeline_types import PipelineFlags
from nl2spec.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nl2spec",
        description="Pipeline for NL-to-Runtime Specification generation and IR-based evaluation."

    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    run = sub.add_parser("run", help="Run pipeline stages.")
    run.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    run.add_argument("-g", "--generate", action="store_true")
    run.add_argument("-l", "--llm", action="store_true")
    run.add_argument("-c", "--compare", action="store_true")
    run.add_argument("--csv", action="store_true")
    run.add_argument("--stats", action="store_true")
    run.add_argument("--all", action="store_true")
    run.add_argument("--log-level", default="INFO")

    # test
    tst = sub.add_parser("test", help="Run tests.")
    tst.add_argument("-g", "--generate", action="store_true")
    tst.add_argument("-c", "--compare", action="store_true")
    tst.add_argument("--all", action="store_true")
    tst.add_argument("--log-level", default="INFO")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    if args.cmd == "run":
        flags = PipelineFlags(
            generate=args.generate or args.all,
            llm=args.llm or args.all,
            compare=args.compare or args.all,
            csv=args.csv or args.all,
            stats=args.stats or args.all,
        )

        if not any([flags.generate, flags.llm, flags.compare, flags.csv, flags.stats]):
            flags = PipelineFlags(
                generate=True,
                llm=True,
                compare=True,
                csv=True,
                stats=True,
            )

        run_pipeline(config_path=args.config, flags=flags)
        return 0

    if args.cmd == "test":
        flags = PipelineFlags(
            test=True,
            generate=args.generate or args.all,
            compare=args.compare or args.all,
        )
        run_pipeline(config_path="config.yaml", flags=flags)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
