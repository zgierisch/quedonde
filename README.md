# whereis 2.0.0

Single-file, dependency-free code search with deterministic structural answers. `whereis` keeps an SQLite FTS5 index plus symbol/edge tables so you can mix fuzzy filename/content lookups with queries such as “who calls `update_structural_data`?”—all without leaving the terminal.

## Feature Highlights
- 🔍 Fast filename/content search with FTS5, fuzzy matching, context lines, JSON/paths-only output, and cache-aware reruns.
- 🧠 Structural extraction populates `symbols`, `edges`, and `annotations`, powering `find`, `callers`, `deps`, `explain`, and natural-language `ask` commands.
- 🐍 Python helpers (`search_repo`, `find_symbol`, `get_callers`, `get_dependencies`, `explain_symbol`) expose the same data for scripts or notebooks.
- ⚙️ Benchmark harness (`scripts/run_benchmarks.py`) synthesizes a 10k-file repo and records <3 s indexing / <100 ms query timings for release audits.

## Requirements
- Python 3.9+
- A writable working directory (the CLI drops `.code_index.sqlite` + `.code_index.cache` next to your project)

## Installation
Pick whichever option fits your workflow:
- **Standalone download** – copy `quedonde.py` into your repository root (or clone this repo and symlink the script). The CLI stores its SQLite/cache files next to wherever the script lives.
- **Virtual environment** – create a venv in the repo (`python -m venv .venv && .\.venv\Scripts\activate`) and run `python quedonde.py ...` so dependencies stay isolated.
- **Editable install (optional)** – if you prefer `pip install -e .`, add this repo to your workspace and reference `quedonde` as a module (`python -m quedonde ...`). The script is self-contained, so no extra dependencies are required.

After installation, run `python quedonde.py --help` to verify the CLI is accessible from your current directory.

## Quick Start
1. **Install / copy** `quedonde.py` into your project root.
2. **Run migrations once** to create structural tables:
	```powershell
	python quedonde.py migrate
	```
3. **Index the repo** (run this whenever files change):
	```powershell
	python quedonde.py index
	```
4. **Search immediately**:
	```powershell
	python quedonde.py "http client"
	```

The CLI prints matches to `stdout` and status messages (e.g., `[done] 0.18s`) to `stderr`, so piping results stays clean.

## Searching the Index
```powershell
# Content search (default) with JSON output
python quedonde.py --json "retry logic"

# Filename-only
python quedonde.py --name "settings.toml"

# Fuzzy search plus 2 lines of surrounding context
python quedonde.py --fuzzy --context 2 "setup logging"

# Require "docs" somewhere in the path and show 3 lines of context
python quedonde.py --title docs --context 3 "authentication"

# Emit paths only (one per line) so they can be piped elsewhere
python quedonde.py --paths --title data/export.csv "customer"
```

> Queries containing SQLite FTS operators (`OR`, `NEAR`, wildcards, quoted literals with punctuation) are treated as raw expressions. When you use raw syntax, qualify the column you intend to search (e.g., `content:"Mountain Orientation:"`). Plain phrases automatically search both path and content fields.

### Common Flags
- `index` – refresh the database after edits (also removes deleted files and runs `VACUUM`).
- `--json` / `--paths` – choose structured JSON or newline-delimited paths (mutually exclusive).
- `--name`, `--content`, `--fuzzy` – toggle which fields participate and whether fuzzy matching applies.
- `--context N` – include `N` lines of surrounding text (ignored for `--paths`).
- `--lines` – attach line numbers to snippets (disabled when combined with `--paths`, `--fuzzy`, or raw FTS operators).
- `--title TEXT` – only include results where `TEXT` appears anywhere in the path (repeat to AND terms together).

### Piping Matches
```powershell
# Delete *.tmp files after confirming the list
python quedonde.py --paths "*.tmp" | ForEach-Object { Remove-Item $_ }

# Open every TODO hit in VS Code
python quedonde.py --paths "TODO" | ForEach-Object { code $_ }
```

## Structural Commands
Structural data becomes available after you run `python quedonde.py migrate` followed by `python quedonde.py index`. Once populated:

```powershell
# Direct symbol queries
python quedonde.py find --context 2 update_structural_data
python quedonde.py callers --json update_structural_data
python quedonde.py deps --limit 25 classify_intent
python quedonde.py explain structural_ready

# Intent-driven natural language
python quedonde.py ask "who calls update_structural_data"
python quedonde.py ask --json "explain structural_ready"
```

What each command returns:
- `find` – definition spans inside each file (optionally `--paths` or `--context`).
- `callers` – inbound `calls` edges with file hints.
- `deps` – outbound relations (`calls`, `imports`, `includes`, `references`).
- `explain` – bundles definitions, callers, dependencies, and annotations.
- `ask` – classifies a natural-language question, invokes the correct handler, and prints a narrative plus structured payload.

If the tables go stale (or you cloned a repo without the DB), rerun `index`. The CLI warns on `stderr` and returns an empty list until fresh data exists.

### How intent detection works
- The `ask` command does **not** run an LLM. Instead, it lower-cases the query, strips helper words (`what`, `is`, `does`, etc.), then matches deterministic keyword patterns hard-coded in `quedonde.py` (see `_INTENT_RULES`).
- Phrases like “where is … defined” map to the `find` handler, “who calls …” maps to `callers`, “depends on …” maps to `deps`, and verbs such as “explain”/“tell me about” route to the `explain` aggregator. The first matching rule wins, so unambiguous wording helps.
- The final meaningful token becomes the candidate symbol; you can force disambiguation with the `symbol:<name>` prefix if needed.
- To extend the heuristics, update `_INTENT_RULES` and quickly validate changes by running a few `python quedonde.py ask --json ...` commands (or wire up your own tests) before committing the new patterns.

## Python API
All CLI functionality is backed by importable helpers:

```python
from quedonde import (
	 search_repo,
	 find_symbol,
	 get_callers,
	 get_dependencies,
	 explain_symbol,
)

# Plain-text / fuzzy search
for hit in search_repo("retry logic", content=True, context=2):
	 print(hit["path"], hit["snippet"])

# Structural lookups
symbol_defs = find_symbol("update_structural_data", limit=5)
callers = get_callers("update_structural_data")
deps = get_dependencies("update_structural_data")
details = explain_symbol("update_structural_data")
```

Each helper returns JSON-compatible dictionaries that mirror the CLI output, making it easy to embed results in notebooks, CI checks, or bots.

## Benchmarking & Performance Evidence
- `scripts/generate_benchmark_repo.py <dest> --files 10000` – create a deterministic synthetic repo.
- `python scripts/run_benchmarks.py` – automate repo generation, migrations, multiple `index` runs, warmed structural queries, and Markdown reporting to `documentation/reports/benchmark_results.md` (create that folder locally; it stays untracked).

The reference run used for the 2.0.0 release produced the following metrics (see the report for the full table):

| Scenario | Result | Target |
| --- | --- | --- |
| Warm indexing pass | 0.772 s | < 3.0 s |
| `find fn_123` (warmed) | 0.071 s | < 0.100 s |
| `callers fn_123` (warmed) | 0.067 s | < 0.100 s |
| `deps call_123` (warmed) | 0.067 s | < 0.100 s |

Use the harness whenever you need reproducible, evidence-backed timings before shipping changes.

## Tips & Troubleshooting
- Re-run `python quedonde.py index` after modifying or deleting files; stale entries are purged automatically.
- `.code_index.cache` stores CLI outputs keyed by query/flags—delete it (or run `index`) to invalidate caches.
- Status lines land on `stderr`; redirect `stdout` freely when piping into other tools.
- Structural commands rely on migrations. If you see `[struct] structural tables unavailable`, run `python quedonde.py migrate` followed by `python quedonde.py index`.
- For CI, call `python quedonde.py diagnose --json` to verify migrations and table counts before running structural tests.

Happy searching! If you run into edge cases or need additional examples, open an issue and share the command output so we can reproduce it.
