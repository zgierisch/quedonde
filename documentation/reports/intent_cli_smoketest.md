# Intent CLI Smoke Test

Date: 2025-12-12

## Callers intent

Command:

```
python quedonde.py ask --json "who calls update_structural_data"
```

Output:

```
{
  "intent": "callers",
  "symbol": "update_structural_data",
  "raw": "who calls update_structural_data",
  "results": [
    {
      "src_path": ".\\quedonde.py",
      "src_symbol": "index_repo",
      "relation": "calls"
    },
    {
      "src_path": ".\\quedonde.py",
      "src_symbol": "update_structural_data",
      "relation": "calls"
    }
  ]
}
```

## Explain intent

Command:

```
python quedonde.py ask --json "explain update_structural_data"
```

Output (truncated for brevity):

```
{
  "intent": "explain",
  "symbol": "update_structural_data",
  "raw": "explain update_structural_data",
  "details": {
    "symbol": "update_structural_data",
    "definitions": [
      {
        "path": ".\\quedonde.py",
        "symbol": "update_structural_data",
        "kind": "function",
        "line_start": 943,
        "line_end": 1009
      }
    ],
    "callers": [ ... ],
    "dependencies": [ ... ],
    "annotations": []
  }
}
```
