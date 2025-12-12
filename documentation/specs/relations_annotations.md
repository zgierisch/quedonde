# Relation & Annotation Extraction Heuristics

## Relations

`quedonde.py` emits the following structural relations while indexing:

| Relation | Trigger | Notes |
| --- | --- | --- |
| `imports` | `import` / `from` statements in Python, ES module imports, or `require()` calls | Recorded with `src_symbol="__file__"` and `dst_symbol` equal to the imported module specifier. |
| `includes` | `#include` directives in C/C++ | `dst_symbol` stores the header literal. |
| `calls` | Regex match on `identifier(` after keyword filtering | Scoped to the containing function/method; `dst_path` is populated when the callee symbol is defined in the same file. |
| `references` | `self.foo`, `this->foo`, or `this.foo` attribute access | Indicates member usage within the current scope. |
| `owns` | Nested definitions (`class` → `def` in Python, `class`/`struct` braces in C/C++) | Captures containment edges for later traversal. |

## Annotations

Annotations are discovered by scanning comments for `@quedonde:<tag>` where `<tag>` is one of `legacy`, `bridge`, `orchestrator`, `deprecated`, or `temporary`. Each tag is associated with the next symbol block in the file (if any) and persisted in the `annotations` table with the source line number.

The artifacts that exercise these heuristics live under `documentation/fixtures/structural/` and are re-indexed as part of the standard `python quedonde.py index` workflow.
