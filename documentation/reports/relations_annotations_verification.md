# Relations & Annotations Verification – 2025-12-12

Commands:
1. `python quedonde.py index`
2. `python quedonde.py diagnose --json`
3. `python - <<'PY' ...` querying `edges` and `annotations`.

Excerpt of `edges` (filtered to `python_sample.py`):
```json
[
  {"relation": "imports", "dst_symbol": "json"},
  {"relation": "imports", "dst_symbol": "pathlib"},
  {"relation": "calls", "src_symbol": "Controller", "dst_symbol": "helper"},
  {"relation": "owns", "src_symbol": "Controller", "dst_symbol": "run"},
  {"relation": "references", "src_symbol": "Controller", "dst_symbol": "state"}
]
```

Excerpt of `annotations`:
```json
[
  {"path": "./documentation/fixtures/structural/python_sample.py", "symbol": "Controller", "tag": "legacy", "line": 4},
  {"path": "./documentation/fixtures/structural/python_sample.py", "symbol": "helper", "tag": "bridge", "line": 17},
  {"path": "./documentation/fixtures/structural/notes_sample.md", "symbol": "Subsystems", "tag": "orchestrator", "line": 3}
]
```

`python quedonde.py diagnose --json` now reports `edges: 350` and `annotations: 3`, confirming the extractor pipeline populates both tables.
