#!/usr/bin/env python3
"""Batch helper for quedonde session checkpoints.

Provide a JSON config describing multiple session dumps and this helper will
invoke `python quedonde.py session dump ...` for each entry. See the README for
an example config structure.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REQUIRED_FIELDS = ("session_id", "subsystems", "files", "symbols")
DEFAULT_CONFIDENCE = "medium"
DEFAULT_OUTPUT_DIR = "checkpoints"


def _coerce_list(value: Any, field: str) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    raise ValueError(f"config field '{field}' must be a non-empty list or string")


def _validate_session(session: Dict[str, Any], index: int) -> None:
    for field in REQUIRED_FIELDS:
        if field not in session:
            raise ValueError(f"session[{index}] missing required field '{field}'")
    for list_field in ("subsystems", "files", "symbols"):
        values = _coerce_list(session[list_field], list_field)
        if not values:
            raise ValueError(f"session[{index}] field '{list_field}' cannot be empty")
        session[list_field] = values
    decisions = session.get("decisions")
    if decisions is not None:
        session["decisions"] = _coerce_list(decisions, "decisions")
    questions = session.get("questions")
    if questions is not None:
        session["questions"] = _coerce_list(questions, "questions")


def _resolve_output_path(
    session: Dict[str, Any],
    *,
    config_dir: Path,
    default_dir: Path,
) -> Path:
    raw_output = session.get("output")
    if raw_output:
        candidate = Path(str(raw_output))
        if not candidate.is_absolute():
            candidate = (config_dir / candidate).resolve()
        return candidate
    target = default_dir / f"{session['session_id']}.json"
    return target.resolve()


def _build_command(
    session: Dict[str, Any],
    *,
    python_exe: str,
    quedonde_path: Path,
    output_path: Path,
    deps_limit: int,
    default_confidence: str,
    fallback_append: bool,
    fallback_force: bool,
) -> List[str]:
    confidence = session.get("confidence", default_confidence) or DEFAULT_CONFIDENCE
    cmd: List[str] = [
        python_exe,
        str(quedonde_path),
        "session",
        "dump",
        "--session-id",
        session["session_id"],
        "--confidence",
        confidence,
        "--output",
        str(output_path),
        "--deps-limit",
        str(session.get("deps_limit", deps_limit)),
    ]
    for subsystem in session["subsystems"]:
        cmd.extend(["--subsystem", subsystem])
    for file_path in session["files"]:
        cmd.extend(["--file", file_path])
    for symbol in session["symbols"]:
        cmd.extend(["--symbol", symbol])
    for decision in session.get("decisions", []):
        cmd.extend(["--decision", decision])
    for question in session.get("questions", []):
        cmd.extend(["--question", question])
    append_flag = session.get("append")
    force_flag = session.get("force")
    if append_flag or (append_flag is None and fallback_append):
        cmd.append("--append")
    if force_flag or (force_flag is None and fallback_force):
        cmd.append("--force")
    return cmd


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    sessions = data.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("config must include a non-empty 'sessions' array")
    return data


def _print_command(label: str, cmd: Sequence[str]) -> None:
    rendered = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"[checkpoint] {label}: {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch quedonde session checkpoint emitter")
    parser.add_argument("--config", default="session_checkpoints.json", help="JSON config describing sessions")
    parser.add_argument("--output-dir", help="Override default output directory from config")
    parser.add_argument("--python", dest="python_exe", default=sys.executable, help="Python interpreter to run quedonde.py")
    parser.add_argument(
        "--quedonde",
        default=str(Path(__file__).resolve().parents[1] / "quedonde.py"),
        help="Path to quedonde.py",
    )
    parser.add_argument("--deps-limit", type=int, default=25, help="Default dependency cap for --deps-limit")
    parser.add_argument("--append", action="store_true", help="Append by default when a session omits --append")
    parser.add_argument("--force", action="store_true", help="Force overwrite when a session omits --force")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed command")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"[checkpoint] config file {config_path} not found", file=sys.stderr)
        return 1

    try:
        config = _load_config(config_path)
    except ValueError as exc:
        print(f"[checkpoint] {exc}", file=sys.stderr)
        return 1

    config_dir = config_path.parent
    default_dir = args.output_dir or config.get("output_dir") or DEFAULT_OUTPUT_DIR
    default_dir_path = Path(default_dir)
    if not default_dir_path.is_absolute():
        default_dir_path = (config_dir / default_dir_path).resolve()
    default_confidence = config.get("default_confidence", DEFAULT_CONFIDENCE)

    quedonde_path = Path(args.quedonde).expanduser().resolve()
    if not quedonde_path.exists():
        print(f"[checkpoint] quedonde.py not found at {quedonde_path}", file=sys.stderr)
        return 1

    sessions = config["sessions"]
    overall_rc = 0
    for index, session in enumerate(sessions):
        try:
            if not isinstance(session, dict):
                raise ValueError(f"session[{index}] must be an object")
            _validate_session(session, index)
            output_path = _resolve_output_path(session, config_dir=config_dir, default_dir=default_dir_path)
            cmd = _build_command(
                session,
                python_exe=args.python_exe,
                quedonde_path=quedonde_path,
                output_path=output_path,
                deps_limit=args.deps_limit,
                default_confidence=default_confidence,
                fallback_append=args.append,
                fallback_force=args.force,
            )
        except ValueError as exc:
            print(f"[checkpoint] {exc}", file=sys.stderr)
            return 1

        _print_command(session["session_id"], cmd)
        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            overall_rc = result.returncode
            print(
                f"[checkpoint] command for session '{session['session_id']}' failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            if args.fail_fast:
                break

    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
