# Structural Extraction Benchmark – 2025-12-12

Command: `python quedonde.py benchmark_structural --json`

Summary:
- structural files scanned: 13 (processed 13, skipped 0)
- average extraction cost: 0.101 ms per file
- slowest file: `quedonde.py` at 0.86 ms (1,720 LOC, 40 symbols)
- fastest file: `documentation/fixtures/structural/notes_sample.md` at 0.007 ms (6 LOC, 3 symbols)
- safeguard threshold: files exceeding 5,000 LOC are skipped (none encountered in this run)

| Path | Language | LOC | Symbols | Duration (ms) | Skipped |
| --- | --- | ---: | ---: | ---: | --- |
| `quedonde.py` | python | 1,720 | 40 | 0.86 | no |
| `documentation/specs/speculative/quedonde_2-0_structural_code_intelligence_spec.md` | markdown | 306 | 33 | 0.142 | no |
| `documentation/fixtures/structural/python_sample.py` | python | 15 | 5 | 0.025 | no |
| `documentation/fixtures/structural/ts_sample.ts` | typescript | 10 | 3 | 0.014 | no |
| `documentation/fixtures/structural/config_sample.yaml` | yaml | 5 | 4 | 0.011 | no |

Full JSON output is stored in the command log above and can be regenerated at any time with the same invocation.
