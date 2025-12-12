# Benchmark Results

Generated: 2025-12-12T17:36:43.603972Z
Repository: build/benchmark_repo
Files: 10000
Generation time: 3.29s
Index target (< 3.00s): PASS (best=0.77s)
Structural target (< 0.100s): PASS (slowest=0.071s)

## Indexing runs
Run | Seconds
--- | ---
1 | 52.374
2 | 0.772

## Structural commands
Command | Symbol | Warmup (s) | Measured (s)
--- | --- | --- | ---
find | fn_123 | 0.071 | 0.071
callers | fn_123 | 0.070 | 0.067
deps | call_123 | 0.068 | 0.067
