# whereis 2.0.0 Release Notes

_Date: 2025-12-12_

## Overview
The 2.0.0 release transforms `whereis` from a pure text-search helper into a fully structural code assistant. A new extractor pipeline now maintains symbol, edge, and annotation tables, and the CLI/Python APIs surface deterministic answers for `find`, `callers`, `deps`, and `explain`. The `ask` command routes natural language prompts through the same structural data, giving large language models trustworthy provenance for every response.

## Highlights
- **Deterministic structural CLI** – `python quedonde.py find|callers|deps|explain <symbol>` returns JSON records with file, line, and relation metadata so downstream tooling can trace every hop.
- **Python API coverage** – Helper functions (`find_symbol`, `get_callers`, `get_dependencies`, `explain_symbol`) provide typed dictionaries for embedding structural lookups inside notebooks or automation scripts.
- **Natural-language dispatcher** – `python quedonde.py ask "who calls foo"` classifies intent, runs the appropriate structural command, and prints a human-readable narrative alongside the raw data.
- **Benchmark tooling** – `scripts/generate_benchmark_repo.py` synthesizes large repositories, while `scripts/run_benchmarks.py` automates repo creation, migrations, indexing, warm-query measurements, and reporting to `documentation/reports/benchmark_results.md`.

## Performance Targets
Using the automated harness against a synthetic repository containing 10,000 Python files:

| Scenario | Result | Target |
| --- | --- | --- |
| Initial indexing pass | 52.374 s | Informational (cold cache) |
| Warm indexing pass | 0.772 s | < 3.0 s |
| `find fn_123` (warmed) | 0.071 s | < 0.100 s |
| `callers fn_123` (warmed) | 0.067 s | < 0.100 s |
| `deps call_123` (warmed) | 0.067 s | < 0.100 s |

Raw measurements live in [documentation/reports/benchmark_results.md](benchmark_results.md), which the harness overwrites on every run. Targets are therefore tracked with reproducible scripts plus concrete evidence for QA sign-off.

## Upgrade Notes
1. Run `python quedonde.py migrate` to ensure the structural tables and triggers exist in `.code_index.sqlite`.
2. Re-index your project with `python quedonde.py index` to populate symbols, edges, and annotations.
3. Use `python scripts/run_benchmarks.py --files 10000` (or adjust the parameters) when you need to re-validate performance before shipping downstream changes.

## Verification Artifacts
- Benchmark evidence: [documentation/reports/benchmark_results.md](benchmark_results.md)
- Release checklist & plan: [planning/2-0_update/plan_2-0_update.md](../../planning/2-0_update/plan_2-0_update.md)
- End-to-end harnesses: `scripts/run_benchmarks.py`, `scripts/generate_benchmark_repo.py`
