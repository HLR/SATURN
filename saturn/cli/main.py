"""Async SATURN benchmark runner — entry point.

Parses args (with the per-dataset defaults), performs the process-level side effects
(faulthandler, .env, vendored tool path), then dispatches to either the
multi-process driver or the single-process async orchestrator.
"""

from __future__ import annotations

import asyncio
import sys
from typing import List
from saturn.log import configure, progress


def main(argv: List[str] | None = None) -> int:
    configure()
    import faulthandler as _fh
    import signal as _sig
    # `kill -USR1 <pid>` dumps every thread's Python stack to stderr, which shows
    # where a CPU-bound program in an execute thread is spending its time.
    _fh.register(_sig.SIGUSR1, all_threads=True, chain=False)

    from dotenv import load_dotenv
    load_dotenv()

    import os
    from saturn.settings import env, unknown_env
    sys.path.append(os.path.join(env("SATURN_TOOLS_DIR"), "Orient-Anything-V2"))
    for name in unknown_env():
        progress(f"WARNING: {name} is set but is not a SATURN setting (configs/settings.json); it has no effect.")

    from saturn.cli.args import parse_args
    from saturn.pipeline.multiproc import _dispatch_multiproc
    from saturn.pipeline.runner import run_benchmark

    args = parse_args(argv)
    if args.num_workers > 1:
        try:
            return _dispatch_multiproc(args, argv)
        except KeyboardInterrupt:
            progress("[multiproc] Interrupted.")
            return 130
    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        progress("[async] Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
