"""Command line entry point: `python -m evals.run`.

Examples:
    python -m evals.run --list
    python -m evals.run --exclude network --reps 3
    python -m evals.run --fixtures read_pdf,image_colours --reps 5 --concurrency 3
    python -m evals.run --report-only .evals/2026-09-22T1930/results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from evals import assets, report
from evals.fixtures import Fixture, select
from evals.runner import append_jsonl, completed_keys, run_fixture

RESULTS_ROOT = Path(".evals")


def _parse_list(value: str | None) -> list[str] | None:
    return [part.strip() for part in value.split(",") if part.strip()] if value else None


async def _run_all(
    fixtures: list[Fixture],
    reps: int,
    concurrency: int,
    run_dir: Path,
    results_path: Path,
) -> list[dict]:
    done = completed_keys(results_path)
    if done:
        print(f"resuming: {len(done)} run(s) already recorded")

    jobs = [
        (fixture, rep)
        for fixture in fixtures
        for rep in range(reps)
        if f"{fixture.name}#{rep}" not in done
    ]
    if not jobs:
        print("nothing left to run")
        return report.load(results_path) if results_path.exists() else []

    gate = asyncio.Semaphore(concurrency)
    # One writer at a time, so a JSONL line is never interleaved with another.
    write_lock = asyncio.Lock()
    finished = 0
    total = len(jobs)

    async def one(fixture: Fixture, rep: int) -> None:
        nonlocal finished
        async with gate:
            result = await run_fixture(fixture, rep, run_dir)
        async with write_lock:
            append_jsonl(results_path, result)
            finished += 1
            mark = "PASS" if result.passed else "FAIL"
            failed = [c["name"] for c in result.checks if c["passed"] is False]
            reason = f"  ({', '.join(failed[:2])})" if failed else ""
            if not result.ok:
                reason = f"  ({result.termination}: {result.error})"
            print(
                f"[{finished:>3}/{total}] {mark}  {fixture.name}#{rep}  "
                f"{result.latency.get('total_s', 0):.1f}s  "
                f"{result.model_calls} model calls{reason}",
                flush=True,
            )

    await asyncio.gather(*(one(f, r) for f, r in jobs))
    return report.load(results_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument("--fixtures", help="comma-separated fixture names")
    parser.add_argument("--tags", help="only fixtures with any of these tags")
    parser.add_argument("--exclude", help="skip fixtures with any of these tags")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per fixture (default 3)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="runs in flight at once. >1 is much faster against a batching server, "
        "but makes wall-clock latency an upper bound rather than a measurement",
    )
    parser.add_argument("--out", help="results directory (default .evals/<timestamp>)")
    parser.add_argument("--list", action="store_true", help="list fixtures and exit")
    parser.add_argument("--report-only", help="rebuild the report from an existing results.jsonl")
    args = parser.parse_args()

    load_dotenv()

    if args.report_only:
        results_path = Path(args.report_only)
        rows = report.load(results_path)
        meta = {
            "model": os.environ.get("MODEL_NAME"),
            "base_url": os.environ.get("MODEL_BASE_URL"),
            "started_at": "(rebuilt)",
            "reps": max((r["rep"] for r in rows), default=-1) + 1,
            "wall_s": 0,
            "concurrency": 1,
        }
        path = results_path.parent / "report.md"
        path.write_text(report.build(rows, meta), encoding="utf-8")
        print(f"wrote {path}")
        return

    fixtures = select(
        names=_parse_list(args.fixtures),
        tags=_parse_list(args.tags),
        exclude=_parse_list(args.exclude),
    )

    if args.list:
        for fixture in fixtures:
            tags = ",".join(fixture.tags)
            print(f"{fixture.name:26} [{tags}]")
            print(f"{'':26} {fixture.what}")
        return

    if not fixtures:
        raise SystemExit("no fixtures selected")

    assets.ensure_assets()
    stamp = time.strftime("%Y-%m-%dT%H%M%S")
    run_dir = Path(args.out) if args.out else RESULTS_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    print(
        f"model: {os.environ.get('MODEL_NAME')} at "
        f"{os.environ.get('MODEL_BASE_URL') or 'default endpoint'}"
    )
    print(f"{len(fixtures)} fixtures x {args.reps} reps -> {run_dir}")

    started = time.perf_counter()
    rows = asyncio.run(
        _run_all(fixtures, args.reps, args.concurrency, run_dir, results_path)
    )
    wall = time.perf_counter() - started

    meta = {
        "model": os.environ.get("MODEL_NAME"),
        "base_url": os.environ.get("MODEL_BASE_URL"),
        "started_at": stamp,
        "reps": args.reps,
        "wall_s": wall,
        "concurrency": args.concurrency,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report_path = run_dir / "report.md"
    report_path.write_text(report.build(rows, meta), encoding="utf-8")

    passed = sum(1 for r in rows if r["passed"])
    print(f"\n{passed}/{len(rows)} passed in {wall:.0f}s")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
