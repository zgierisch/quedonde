# Changelog

## [Unreleased]

_No changes yet._

## [2.0.0] - 2025-12-12

### Added
- Structural CLI commands (`find`, `callers`, `deps`, `explain`) with deterministic SQLite-backed symbol + edge extraction.
- Natural-language `ask` dispatcher plus Python helpers (`find_symbol`, `get_callers`, `get_dependencies`, `explain_symbol`) for programmatic use.
- Benchmark harness tooling (`scripts/generate_benchmark_repo.py`, `scripts/run_benchmarks.py`) and accompanying report at [documentation/reports/benchmark_results.md](documentation/reports/benchmark_results.md).

### Performance
- Warm indexing pass on the 10k-file synthetic repository completes in 0.772 s (target < 3.0 s).
- Warm structural queries stay under 0.100 s (`find fn_123` 0.071 s, `callers fn_123` 0.067 s, `deps call_123` 0.067 s).
- See [documentation/reports/benchmark_results.md](documentation/reports/benchmark_results.md) for the full measurement log produced on 2025-12-12.

## [2025-11-13]

### Added
- `--lines` flag to include matching line numbers for content searches.
- `--title` flag to filter results by path segments while keeping content searches intact.
- Context collection now reuses a single pass to supply both snippets and line numbers, enabling richer `--context` output.
- README guidance covering the interaction between `--paths`, `--title`, and other flags.

### Fixed
- Normalize content queries containing punctuation so SQLite FTS no longer throws syntax errors.
- Handle trailing parentheses in content queries without triggering FTS syntax errors.
