# Structural Context Expansion — Implementation Note

**Version:** whereis 2.0.0  
**Status:** Shipped (quedonde CLI)

This document captures the production behavior of structural context expansion as implemented in `quedonde.py`. It supersedes the speculative draft in `documentation/specs/speculative/quedonde_structural_context_expansion.md`.

---

## 1. Feature Summary

Structural context expansion turns symbol-centric queries into containment-aware context blocks. Instead of spraying line-based snippets, the CLI walks up the structural hierarchy—symbol → owning construct → parent outline → file summary—emitting complete logical units at each step.

Key guarantees:
- Expansion is **opt-in** via `quedonde context`.
- Each level returns **bounded, deterministic** output (hard line caps, no horizontal fan-out).
- Ambiguous containment emits warnings and stops expanding until the index is repaired.

See [README.md](../../README.md#L77-L134) for the CLI quick-start.

---

## 2. Command Interface

```powershell
python quedonde.py context <symbol> [--level 0-3] [--path <glob>] [--kind <type>] [--json]
```

- Defaults to `--level 1`.
- `--path` filters candidate definitions by glob; repeat to AND multiple patterns.
- `--kind` limits structural kinds (function, class, method, etc.).
- `--json` switches from human-readable blocks to structured payloads for automation.
- Output always records a bookmark in `.quedonde_context_history.json`; session checkpoints automatically stitch the latest history into `session resume` output.

---

## 3. Context Levels

| Level | Scope | What ships today |
| --- | --- | --- |
| 0 | Symbol definition | Name, kind, signature, file, line span, plus any annotations. No body. |
| 1 (default) | Immediate container | Full function/method/class body that owns the symbol, with guard clauses and docstrings. |
| 2 | Parent outline | Header-only view of the parent symbol (class/namespace/module) and its siblings plus role classification. |
| 3 | File summary | File role, imports, primary symbols, fan-in/out metrics, annotations, truncated at 800 lines. No raw code bodies. |

Each level can be expanded via `--level <N>` without retyping the previous command; repeated invocations keep stacking history.

---

## 4. Data Sources & Resolution Rules

- `symbols` table: definition spans, kinds, annotations, file metadata.
- `edges` table: containment links resolved by span inclusion (owner span must strictly contain child span).
- `annotations`: role tags (orchestrator, transformer, bridge, legacy, etc.) surface in Level 2+ summaries.
- Containment conflicts (two parents with identical spans) raise an explicit warning and return no context.

Implementation pragmatics:
- Level 1 returns the entire owning construct but caps at 400 emitted lines to prevent runaway bodies.
- Level 3 summarization stops at 800 lines and always leads with docstring/import summary before listing top-level symbols sorted by structural importance.
- All blocks include a `status` field (`ok`, `ambiguous`, `missing`) in JSON output so callers can detect partial results.

---

## 5. Session-State Integration

- Every `quedonde context` call appends `{symbol, level, filters, timestamp}` to `.quedonde_context_history.json`.
- `session dump` persists structural context references alongside decisions/questions.
- `session resume` reads the history file and attaches the most recent bookmark per symbol so operators know how far investigation already progressed.
- Guardrails in [README.md](../../README.md#L175-L225) cover provenance, drift detection, and schema alignment for checkpoint consumers.

---

## 6. Safeguards

| Guardrail | Behavior |
| --- | --- |
| Line caps | Level 1 ≤ 400 lines, Level 3 ≤ 800 lines. Truncation emits `[...]` markers and a warning. |
| Explicit invocation | The CLI never auto-expands context; users must run `quedonde context`. |
| Ambiguity handling | Multiple candidate parents trigger an error message; no guesswork. |
| Horizontal fan-out | The resolver only walks *up* containment; no sibling or dependency expansion occurs. |
| Schema drift | Structural mismatches cause `session resume` to warn until `python quedonde.py index` refreshes the DB. |

---

## 7. Example Output

```
SYMBOL
- name: update_structural_data
- kind: function
- file: quedonde.py:542-618

CONTAINER (Level 1)
- function: update_structural_data
- role: transformer
- fan-in: 5
- fan-out: 3
- body:
    ... full function snippet ...

PARENT OUTLINE (Level 2)
- class StructuralIndexer
    • refresh()
    • update_structural_data()
    • emit_metrics()

FILE SUMMARY (Level 3)
- path: quedonde.py
- role: CLI entrypoint
- fan-in: 22 | fan-out: 17
- key imports: sqlite3, json, argparse
- primary symbols: StructuralIndexer, resolve_symbol_context, session_dump
```

JSON mode mirrors the same structure with `symbol`, `container`, `parent_outline`, and `file_summary` objects plus metadata fields (`status`, `line_cap_hit`, etc.).

---

## 8. Future Enhancements

- Richer Level 3 narratives (e.g., paragraphs summarizing module responsibilities) once additional telemetry lands.
- Optional context deduplication across multiple symbols when batching session dumps.
- Configurable bookmark retention window for `.quedonde_context_history.json`.

---

## 9. References

- Implementation: `quedonde.py` (`resolve_symbol_context`, CLI parser branch `context`).
- Tests: `tests/test_context_cli.py`, `tests/test_session_state_context.py`.
- User docs: [README.md](../../README.md#L77-L225).
- Historical spec: [documentation/specs/speculative/quedonde_structural_context_expansion.md](speculative/quedonde_structural_context_expansion.md).
