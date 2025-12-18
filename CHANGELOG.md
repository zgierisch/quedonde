# Changelog

## [Unreleased]

### Added
- _Nothing yet_

### Fixed
- _Nothing yet_

## [2.0.1] - 2025-12-18

### Added
- Session checkpoint workflow documentation: README now details the CLI flow, guardrails, automation hooks, and links to a new LLM playbook (`documentation/guides/session_checkpoint_llm_usage.md`).
- Batch tooling for checkpoints: `scripts/session_checkpoint.py` fans out multiple `session dump` runs from a JSON plan, and `documentation/fixtures/checkpoints/sample_sessions.json` demonstrates the format.
- Containment-aware `quedonde context` command with Level 0-3 expansion, path/kind filters, JSON formatting, and file-summary output, plus docs covering usage.
- Context history plumbing: every context invocation is recorded in `.quedonde_context_history.json`, and `session resume` now hydrates symbol entries with the last known level/status bookmark.
- Test coverage for the new context CLI (containment ambiguity, Level 3 summaries) and session-state bookmarks.
- Session-state bootstrapper: `quedonde.py` now generates `session_state.py` on demand when the helper file is missing, preserving the single-file drop experience for standalone installs.

### Fixed
- Restored the `session` CLI pipeline (dump/resume, revalidation, and schema helpers) after a regression, including automated stress testing via `documentation/fixtures/checkpoints/stress_sessions.json`.
- Session-state schema compliance: `session_state.py` now enforces `additionalProperties: false`, the five-path limit, the minimum session ID length, and no longer serializes `context` blocks so emitted checkpoints validate cleanly against `session_state.schema.json`. Context history attachments also respect pre-existing bookmarks instead of overwriting them.
- Eliminated Pyright/Pylance false-positives by wrapping all `_ensure` truthiness checks in `bool(...)`, and mirrored the fix inside the embedded template so regenerated helpers stay healthy.

## [2.0.0] - 2025-12-12

### Added
- Structural CLI commands (`find`, `callers`, `deps`, `explain`) with deterministic SQLite-backed symbol + edge extraction.
- Natural-language `ask` dispatcher plus Python helpers (`find_symbol`, `get_callers`, `get_dependencies`, `explain_symbol`) for programmatic use.
- Benchmark harness tooling (`scripts/generate_benchmark_repo.py`, `scripts/run_benchmarks.py`) that records Markdown reports locally (see README for usage).

### Performance
- Warm indexing pass on the 10k-file synthetic repository completes in 0.772 s (target < 3.0 s).
- Warm structural queries stay under 0.100 s (`find fn_123` 0.071 s, `callers fn_123` 0.067 s, `deps call_123` 0.067 s).
- Run `python scripts/run_benchmarks.py` to reproduce the measurement log locally (the script writes to `documentation/reports/benchmark_results.md`, which remains untracked).

## [2025-11-13]

### Added
- `--lines` flag to include matching line numbers for content searches.
- `--title` flag to filter results by path segments while keeping content searches intact.
- Context collection now reuses a single pass to supply both snippets and line numbers, enabling richer `--context` output.
- README guidance covering the interaction between `--paths`, `--title`, and other flags.

### Fixed
- Normalize content queries containing punctuation so SQLite FTS no longer throws syntax errors.
- Handle trailing parentheses in content queries without triggering FTS syntax errors.
