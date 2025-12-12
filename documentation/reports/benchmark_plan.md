# Benchmark Plan

Targets:
- Index 10,000 files in under 3 seconds.
- Structural query (`find`/`callers`/`deps`) executes under 100 ms on warmed cache.

Procedure:
0. Preferred: run `python scripts/run_benchmarks.py` to orchestrate all steps below and capture a Markdown report automatically.
1. Use `scripts/generate_benchmark_repo.py` to synthesize a repository with 10k Python files (`python scripts/generate_benchmark_repo.py bench_repo --files 10000`).
2. Run `python quedonde.py index bench_repo` twice and record the fastest run.
3. Execute representative structural commands against `bench_repo` (e.g., `python quedonde.py find symbol_123`).
4. Capture timings in `documentation/reports/benchmark_results.md` once targets are hit.
