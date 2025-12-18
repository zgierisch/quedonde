# Structural Code Intelligence — Implementation Note

**Version:** whereis 2.0.0  
**Status:** Shipped (quedonde CLI + API)

This note documents the production state of structural code intelligence in `quedonde.py`. It supersedes the speculative roadmap in `documentation/specs/speculative/quedonde_2-0_structural_code_intelligence_spec.md` and reflects what is live today.

---

## 0. Executive Snapshot

`quedonde` now pairs its SQLite FTS5 index with deterministic structural tables (`symbols`, `edges`, `annotations`). Refactor-heavy workflows gain:
- Containment-aware lookups (`find`, `callers`, `deps`, `explain`, `context`).
- Deterministic CLI/Python parity without external dependencies.
- Provenance tagging and session checkpoint hooks for LLM-assisted investigations.

See [README.md](../README.md#L1-L225) for user-facing guidance.

---

## 1. Design Constraints (Still Enforced)

| Constraint | Enforcement in code |
| --- | --- |
| No full parsers | Regex + line-based scans per extension inside `quedonde.py`. |
| Incremental indexing | `meta` table mtime diff drives re-index in `index` command. |
| Failure-tolerant intelligence | Missing spans/ambiguous parents emit warnings; CLI falls back to lexical search only when requested. |
| Explainability | Every structural response cites symbol names, kinds, paths, and line spans; JSON payloads include `reason` fields. |

---

## 2. Storage Model

Existing `meta`/`files` tables remain untouched. Added tables:

```sql
symbols(
  path TEXT NOT NULL,
  symbol TEXT NOT NULL,
  kind TEXT NOT NULL,
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  PRIMARY KEY(path, symbol, kind)
);

edges(
  src_path TEXT NOT NULL,
  src_symbol TEXT NOT NULL,
  relation TEXT NOT NULL,
  dst_path TEXT,
  dst_symbol TEXT
);

annotations(
  path TEXT NOT NULL,
  symbol TEXT,
  tag TEXT NOT NULL,
  line INTEGER
);
```

Indexes ensure `symbol` lookups and `path`-scoped queries stay <100 ms even on the synthetic 10k-file benchmark ([documentation/reports/benchmark_results.md](../documentation/reports/benchmark_results.md)).

---

## 3. Extraction Pipeline

1. Determine language class by file extension.  
2. Run ordered regex passes (Python `def`/`class`, C/C++ `class|struct` + brace open, JS `function|class|export`, JSON/YAML top-level keys, Markdown headings).  
3. Record spans + kinds inside `symbols`.  
4. Within each file, emit `edges` for `imports`, `includes`, call signatures, and ownership (`owns`).  
5. Parse inline annotations using the `@quedonde:<tag>` syntax.

No cross-file context is required; extraction remains pure and incremental.

---

## 4. Relationship Semantics

| Relation | Source evidence | Notes |
| --- | --- | --- |
| `imports` | `import`, `from`, `require` statements | Multi-symbol imports split into separate edges. |
| `includes` | `#include` | File-scoped; `dst_path` resolves via include string. |
| `calls` | `symbolName(` tokens inside owning span | Prefers local symbols; falls back to global symbol table when unique. |
| `references` | Qualified identifiers (`Foo.Bar`) | Captures non-call usages. |
| `owns` | Span containment (class → method, module → function) | Drives containment-aware context. |

Edges never infer cross-file links without textual evidence; unresolved targets keep `dst_path` NULL for traceability.

---

## 5. Provenance Tags

Annotations honor the shipped tag set: `legacy`, `bridge`, `orchestrator`, `deprecated`, `temporary`. Tags surface in:
- `explain` output (annotation list per symbol).
- `context` Level 2+ summaries.
- Session checkpoints (`session dump` embeds tag metadata alongside decisions/questions).

---

## 6. Query Engine & Intent Routing

`quedonde.py` maintains deterministic rules that map natural-language queries to handlers (see `_INTENT_RULES`). Examples:
- “where is X defined” → `find`
- “who calls X” → `callers`
- “what depends on X” → `deps`
- “why does X exist” / “explain X” → `explain`

If no rule matches, the CLI treats the input as a raw FTS search and labels structural results as unavailable.

Execution pipeline:
1. Parse CLI args / intent rule.  
2. Hit minimal table set (`symbols`, `edges`, `annotations`, `files`).  
3. Format both human output and JSON payloads with the same data.  
4. Emit `[warn]` lines when data is missing or ambiguous.

---

## 7. CLI & API Surface

### CLI

```powershell
python quedonde.py index
python quedonde.py find <symbol> [--context N] [--json]
python quedonde.py callers <symbol>
python quedonde.py deps <symbol>
python quedonde.py explain <symbol>
python quedonde.py context <symbol> [--level 0-3]
python quedonde.py ask "who calls ..."
```

Common flags: `--json`, `--paths`, `--context`, `--limit`, `--path`, `--kind`. Each command shares the same SQLite connection and respects offline, single-file constraints.

### Python API

```python
from quedonde import (
    search_repo,
    find_symbol,
    get_callers,
    get_dependencies,
    explain_symbol,
)
```

All helpers return dicts identical to CLI JSON output. See [README.md](../README.md#L105-L149) for examples.

---

## 8. Indexing & Cache Behavior

- Structural tables regenerate only when `meta.mtime` detects a newer file.
- The `index` command prunes deleted paths, rebuilds affected symbol/edge/annotation rows, and runs `VACUUM` to keep the DB tight.
- Cache invalidation and timing evidence are documented in [documentation/reports/benchmark_results.md](../documentation/reports/benchmark_results.md).

---

## 9. Performance & Evidence

| Operation | Target | Latest benchmark (10k synthetic repo) |
| --- | --- | --- |
| Index | < 3 s | 0.772 s |
| `find fn_123` | < 50 ms | 0.071 s |
| `callers fn_123` | < 100 ms | 0.067 s |
| `deps call_123` | < 100 ms | 0.067 s |

Bench harness lives in `scripts/run_benchmarks.py` and publishes Markdown to `documentation/reports/benchmark_results.md`.

---

## 10. Deferred Work

- Semantic ranking remains optional; no embeddings ship today.
- Additional language heuristics (Rust, Go) can slot into the same regex pipeline.
- Future iterations may record dependency confidence scores, but current release sticks to binary evidence.

---

## 11. References

- Implementation: `quedonde.py` (tables, indexer, CLI handlers, intent rules).  
- Structural context documentation: [documentation/specs/structural_context_expansion.md](structural_context_expansion.md).  
- Session state specification: [documentation/specs/session_state_checkpointing.md](documentation/specs/session_state_checkpointing.md).  
- Tests: `tests/test_indexing.py`, `tests/test_structural_cli.py`, `tests/test_context_cli.py`, `tests/test_session_state_context.py`.  
- Historical design: [documentation/specs/speculative/quedonde_2-0_structural_code_intelligence_spec.md](speculative/quedonde_2-0_structural_code_intelligence_spec.md).
