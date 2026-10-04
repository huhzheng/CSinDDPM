"""Record wall time and process memory for an explicit argv command."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("provide an explicit command after --")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    process = subprocess.Popen(command)
    peak_rss = 0
    while process.poll() is None:
        try:
            root = psutil.Process(process.pid)
            rss = root.memory_info().rss + sum(
                child.memory_info().rss for child in root.children(recursive=True)
            )
            peak_rss = max(peak_rss, rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.1)
    payload = {
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_process_tree_rss_mb": peak_rss / (1024 ** 2),
        "return_code": process.returncode,
        "gpu_peak_memory_mb": "PENDING_INSTRUMENTED_TRAIN_OR_SAMPLE_LOG",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
