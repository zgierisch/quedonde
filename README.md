# quedonde

Lightweight, dependency-free code search helper built around SQLite FTS5. Index a project once, then query filenames or file contents with optional fuzzy matching, context lines, JSON output, or plain path lists that are easy to pipe into other tools.

## Requirements
- Python 3.9 or newer
- Run the script from a writable directory (creates `.code_index.sqlite` and `.code_index.cache` next to the script)

## Indexing a project
```powershell
cd C:\path\to\project
python C:\path\to\quedonde\quedonde.py index
```
The `index` command crawls the current directory (and below), inserts/updates files in the FTS index, removes stale entries, and compacts the database.

## Searching
```powershell
# Search file contents (default) and pretty-print JSON
python quedonde.py "http client"

# Filename search only
python quedonde.py --name "settings.json"

# Fuzzy match with 2 lines of context
python quedonde.py --fuzzy --context 2 "setup logging"
```

### CLI flags
- `index` — rebuild the index after changes.
- `--json` — emit indented JSON (default output when `--paths` is not set).
- `--paths` — print only matching file paths (one per line) so you can pipe into other commands.
- `--name` / `--content` — limit the search scope.
- `--fuzzy` — fuzzy-match file names and content.
- `--context N` — include `N` lines of surrounding context in the text output (ignored for `--paths`).
- `--lines` — show matching line numbers for content searches (disabled with `--paths`, `--fuzzy`, or advanced FTS syntax).
- Multi-word queries (without special operators) are treated as exact phrases; include FTS operators (`OR`, `NEAR`, wildcards, etc.) if you want advanced matching.

### Piping results
Because search output goes to `stdout` and status messages go to `stderr`, you can safely pipe the matches:
```powershell
# Remove every file whose path matches the query (review the list first!)
python quedonde.py --paths "*.tmp" | ForEach-Object { Remove-Item $_ }

# Open all matches in VS Code
python quedonde.py --paths "TODO" | ForEach-Object { code $_ }
```

## Python API
```python
from quedonde import search_repo

results = search_repo("search term", content=True, context=2)
for match in results:
	print(match["path"], match["snippet"])
```

## Tips
- Re-run `python quedonde.py index` whenever files are added, removed, or edited.
- Deleted files are detected automatically and purged from the index.
- The cache (`.code_index.cache`) stores raw CLI output keyed by the query/flags combo.
- Status lines (e.g., `[done] 0.03s`, `[cache] hit`) are printed to `stderr`; redirect or pipe `stdout` when chaining commands.
