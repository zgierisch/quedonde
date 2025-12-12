#!/usr/bin/env python3
"""Automate the structural benchmark workflow."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_FILE_COUNT = 10_000
DEFAULT_SAMPLE_INDEX = 123
DEFAULT_INDEX_TARGET = 3.0
DEFAULT_QUERY_TARGET = 0.1


@dataclass
class StructuralMeasurement:
    command: str
    symbol: str
    warmup_seconds: float
    measured_seconds: float


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def default_structural_targets(sample_index: int) -> List[Tuple[str, str]]:
    return [
        ("find", f"fn_{sample_index}"),
        ("callers", f"fn_{sample_index}"),
        ("deps", f"call_{sample_index}"),
    ]


def parse_structural_targets(values: Iterable[str]) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(
                f"Invalid structural target '{value}'. Use the format command:symbol."
            )
        command, symbol = value.split(":", 1)
        command = command.strip()
        symbol = symbol.strip()
        if not command or not symbol:
            raise argparse.ArgumentTypeError(
                f"Invalid structural target '{value}'. Use the format command:symbol."
            )
        targets.append((command, symbol))
    return targets


def timed_run(command: Sequence[str], cwd: Path) -> float:
    start = time.perf_counter()
    subprocess.run(command, cwd=str(cwd), check=True)
    return time.perf_counter() - start


def remove_repo(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)


def reset_repo_state(repo_root: Path) -> None:
    for artifact in (".code_index.sqlite", ".code_index.cache"):
        target = repo_root / artifact
        if target.exists():
            target.unlink()


def sync_migrations(repo_root: Path, source_dir: Path) -> None:
    if not source_dir.exists():
        return
    dest_dir = repo_root / "migrations"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)


def format_relative(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def write_results(path: Path, workspace_root: Path, *,
                  timestamp: datetime,
                  file_count: int,
                  repo_root: Path,
                  index_times: List[float],
                  index_target: float,
                  structural: List[StructuralMeasurement],
                  query_target: float,
                  generator_seconds: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index_fastest = min(index_times) if index_times else float("inf")
    index_status = "PASS" if index_fastest <= index_target else "FAIL"
    query_slowest = max((m.measured_seconds for m in structural), default=float("inf"))
    query_status = "PASS" if structural and query_slowest <= query_target else "FAIL"

    lines: List[str] = []
    if timestamp.tzinfo is None:
        rendered_timestamp = f"{timestamp.isoformat()}Z"
    else:
        rendered_timestamp = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    lines.append("# Benchmark Results")
    lines.append("")
    lines.append(f"Generated: {rendered_timestamp}")
    lines.append(f"Repository: {format_relative(repo_root, workspace_root)}")
    lines.append(f"Files: {file_count}")
    if generator_seconds is not None:
        lines.append(f"Generation time: {generator_seconds:.2f}s")
    lines.append(f"Index target (< {index_target:.2f}s): {index_status} (best={index_fastest:.2f}s)")
    if structural:
        lines.append(
            f"Structural target (< {query_target:.3f}s): {query_status} (slowest={query_slowest:.3f}s)"
        )
    lines.append("")
    lines.append("## Indexing runs")
    if index_times:
        lines.append("Run | Seconds")
        lines.append("--- | ---")
        for idx, elapsed in enumerate(index_times, start=1):
            lines.append(f"{idx} | {elapsed:.3f}")
    else:
        lines.append("No indexing runs recorded.")
    lines.append("")
    lines.append("## Structural commands")
    if structural:
        lines.append("Command | Symbol | Warmup (s) | Measured (s)")
        lines.append("--- | --- | --- | ---")
        for measurement in structural:
            lines.append(
                f"{measurement.command} | {measurement.symbol} | "
                f"{measurement.warmup_seconds:.3f} | {measurement.measured_seconds:.3f}"
            )
    else:
        lines.append("No structural commands configured.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run the structural performance benchmarks.")
    parser.add_argument("--files", type=int, default=DEFAULT_FILE_COUNT, help="Number of files to generate.")
    parser.add_argument(
        "--repo",
        type=str,
        default="build/benchmark_repo",
        help="Destination for the synthetic repository.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="documentation/reports/benchmark_results.md",
        help="Where to write the Markdown summary.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="scripts/generate_benchmark_repo.py",
        help="Path to the repo generator script.",
    )
    parser.add_argument(
        "--migrations",
        type=str,
        default="migrations",
        help="Path to the migrations directory that should be copied into the synthetic repo.",
    )
    parser.add_argument(
        "--quedonde",
        type=str,
        default="quedonde.py",
        help="Path to the quedonde CLI entry point.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run child processes.",
    )
    parser.add_argument(
        "--reuse-repo",
        action="store_true",
        help="Reuse an existing synthetic repository instead of regenerating it.",
    )
    parser.add_argument(
        "--struct",
        action="append",
        default=[],
        help="Structural command in the form command:symbol. Repeat for multiple commands.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help="Index used when building the default structural targets.",
    )
    parser.add_argument(
        "--index-runs",
        type=int,
        default=2,
        help="How many times to run the index command.",
    )
    parser.add_argument(
        "--index-target",
        type=float,
        default=DEFAULT_INDEX_TARGET,
        help="Target time (seconds) for indexing.",
    )
    parser.add_argument(
        "--query-target",
        type=float,
        default=DEFAULT_QUERY_TARGET,
        help="Target time (seconds) for warmed structural queries.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the warmup pass before measuring structural commands.",
    )

    args = parser.parse_args()

    repo_root = resolve_path(args.repo, workspace_root)
    results_path = resolve_path(args.results, workspace_root)
    generator_path = resolve_path(args.generator, workspace_root)
    migrations_path = resolve_path(args.migrations, workspace_root)
    quedonde_path = resolve_path(args.quedonde, workspace_root)
    python_exec = args.python

    if args.files <= 0:
        parser.error("--files must be greater than zero")
    use_default_targets = not args.struct
    if args.sample_index >= args.files and use_default_targets:
        parser.error("--sample-index must be less than --files")

    structural_targets = (
        parse_structural_targets(args.struct)
        if args.struct
        else default_structural_targets(args.sample_index)
    )

    repo_root.parent.mkdir(parents=True, exist_ok=True)

    if repo_root.exists() and not args.reuse_repo:
        print(f"[bench] removing existing repository at {repo_root}")
        remove_repo(repo_root)

    generator_elapsed: float | None = None
    if not repo_root.exists():
        print(
            f"[bench] generating {args.files} files into {repo_root} using {generator_path}"  # noqa: E501
        )
        generator_elapsed = timed_run(
            [python_exec, str(generator_path), str(repo_root), "--files", str(args.files)],
            cwd=workspace_root,
        )
        print(f"[bench] generation completed in {generator_elapsed:.2f}s")
    else:
        print(f"[bench] reusing synthetic repository at {repo_root}")

    if migrations_path.exists():
        print(f"[bench] syncing migrations from {migrations_path} -> {repo_root / 'migrations'}")
        sync_migrations(repo_root, migrations_path)
    else:
        print(f"[bench] migrations directory {migrations_path} not found; skipping sync")

    reset_repo_state(repo_root)

    if migrations_path.exists():
        print("[bench] running migrations")
        timed_run([python_exec, str(quedonde_path), "migrate"], cwd=repo_root)

    index_times: List[float] = []
    for run in range(1, args.index_runs + 1):
        print(f"[bench] indexing run {run}/{args.index_runs}")
        elapsed = timed_run([python_exec, str(quedonde_path), "index"], cwd=repo_root)
        index_times.append(elapsed)
        print(f"[bench] run {run} finished in {elapsed:.3f}s")

    structural_measurements: List[StructuralMeasurement] = []
    for command, symbol in structural_targets:
        cmd = [python_exec, str(quedonde_path), command, symbol, "--json"]
        if not args.skip_warmup:
            print(f"[bench] warming up {command} {symbol}")
            warmup_elapsed = timed_run(cmd, cwd=repo_root)
        else:
            warmup_elapsed = 0.0
        print(f"[bench] measuring {command} {symbol}")
        measured_elapsed = timed_run(cmd, cwd=repo_root)
        structural_measurements.append(
            StructuralMeasurement(command, symbol, warmup_elapsed, measured_elapsed)
        )
        print(
            f"[bench] {command} {symbol} warmup={warmup_elapsed:.3f}s measured={measured_elapsed:.3f}s"
        )

    timestamp = datetime.now(timezone.utc)
    write_results(
        results_path,
        workspace_root,
        timestamp=timestamp,
        file_count=args.files,
        repo_root=repo_root,
        index_times=index_times,
        index_target=args.index_target,
        structural=structural_measurements,
        query_target=args.query_target,
        generator_seconds=generator_elapsed,
    )
    print(f"[bench] wrote results to {results_path}")


if __name__ == "__main__":
    main()
