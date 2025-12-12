# Intent Query Fixtures

| Query | Expected intent | Expected symbol | Notes |
| --- | --- | --- | --- |
| `where is update_structural_data defined` | `find_symbol` | `update_structural_data` | Baseline lookup phrasing. |
| `find symbol: structural_available` | `find_symbol` | `structural_available` | Explicit prefix should short-circuit other rules. |
| `who calls update_structural_data` | `callers` | `update_structural_data` | Matches `who calls` phrase. |
| `call tree for update_structural_data` | `callers` | `update_structural_data` | Additional callers synonym. |
| `incoming references to classify_intent` | `callers` | `classify_intent` | "incoming" keyword forces caller intent. |
| `what does update_structural_data depend on` | `deps` | `update_structural_data` | Dependency phrasing with "depend on". |
| `what imports does repository_manager use` | `deps` | `repository_manager` | Covers "imports" synonym and noun placement. |
| `show downstream deps for handle_structural_intent` | `deps` | `handle_structural_intent` | Downstream/fan-out phrasing. |
| `explain update_structural_data` | `explain` | `update_structural_data` | Straightforward explain request. |
| `tell me about structural_available` | `explain` | `structural_available` | Should map to explain via "tell me about". |
| `annotate callers for handle_structural_intent` | `explain` | `handle_structural_intent` | "annotate" triggers explain, despite extra word. |
| `update_structural_data usage` | `deps` | `update_structural_data` | Symbol-only query with trailing noun should fall back to dependency intent. |
