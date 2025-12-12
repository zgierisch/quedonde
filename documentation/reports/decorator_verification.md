# Decorator Ownership Verification – 2025-12-12

Command:
```bash
python - <<'PY'
import sqlite3, json
conn = sqlite3.connect('.code_index.sqlite')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT path, symbol, kind, line_start, line_end FROM symbols WHERE symbol IN ('build', 'run') ORDER BY path, symbol"
).fetchall()
print(json.dumps([dict(r) for r in rows], indent=2))
conn.close()
PY
```

Output:
```json
[
  {
    "path": "./documentation/fixtures/structural/python_sample.py",
    "symbol": "build",
    "kind": "method",
    "line_start": 8,
    "line_end": 10
  },
  {
    "path": "./documentation/fixtures/structural/python_sample.py",
    "symbol": "run",
    "kind": "method",
    "line_start": 5,
    "line_end": 6
  }
]
```

The decorated `Controller.build` method is recorded with `kind="method"` and `line_start=8`, which corresponds to the decorator line, proving ownership is preserved.
