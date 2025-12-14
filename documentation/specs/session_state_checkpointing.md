# Session State Checkpointing Specification

**Version:** 1.0 (2025-12-14)

## Overview
Session state checkpointing captures derived structural reasoning so long-running LLM workflows can pause and resume without re-deriving context. This document defines the canonical schema, validation rules, and guardrails for files named `quedonde_session_state.json` (or similarly scoped variants).

## Schema Summary
| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | ✓ | Always `"1.0"` for this release. Enables future migrations. |
| `session_id` | string | ✓ | Unique, human-readable identifier (`subsystem_YYYYMMDD`). |
| `timestamp` | RFC3339 string | ✓ | UTC emission time. |
| `structural_version` | string | ✓ | Value from `get_structural_version()` so drift can be detected. |
| `scope.subsystems` | array[string] | ✓ | High-level areas covered; each entry ≤ 64 chars. |
| `scope.files` | array[string] | ✓ | Relative paths examined; at least one entry. |
| `symbols` | object | ✓ | Map of symbol name → summary (kind, role, evidence). |
| `symbols.*.kind` | string | ✓ | e.g., `class`, `function`. |
| `symbols.*.role` | string | ✓ | Architectural role (`orchestrator`, `bridge`, etc.). |
| `symbols.*.source_paths` | array[string] | ✓ | Distinct definition paths for the symbol (≤5 entries, each ≤256 chars). |
| `symbols.*.evidence.fan_in` | integer | ✓ | Non-negative count derived from quedonde edges. |
| `symbols.*.evidence.fan_out` | integer | ✓ | Non-negative count derived from quedonde edges. |
| `symbols.*.evidence.annotations` | array[string] | optional | Matches estructural annotations when present. |
| `dependencies` | array[object] | optional | Each record captures `src`, `relation`, `dst` (strings). |
| `decisions` | array[object] | optional | `decision` + `reason` strings (≤ 240 chars). |
| `open_questions` | array[string] | optional | Outstanding questions (≤ 200 chars). |
| `confidence` | string enum | ✓ | One of `low`, `medium`, `high`. |

See `documentation/specs/session_state.schema.json` for the formal JSON Schema.

## Validation Rules
1. **Derived-only content:** No field may include raw code snippets longer than 120 characters or containing newline sequences. Validators will raise `ValueError` if violated.
2. **Non-empty collections:** `scope.subsystems`, `scope.files`, `symbols` must all contain at least one entry.
3. **Evidence completeness:** Every symbol summary must include both `fan_in` and `fan_out` counts; annotations may be empty but always provided as a list.
4. **Source path provenance:** Every symbol summary records at least one `source_paths` entry so revalidation can confirm the definition still exists. Paths are capped at five entries, each ≤ 256 characters.
4. **Dependency relations:** `relation` must be one of `calls`, `imports`, `includes`, `references`, `annotates`.
5. **ISO timestamps:** `timestamp` must parse via `datetime.fromisoformat(...replace("Z", "+00:00"))` and represent UTC.
6. **Structural version:** Checkpoint consumers compare `structural_version` against the current database token and warn when mismatched; this comparison is mandatory in the CLI unless `--skip-verify` is provided.

## File Naming & Storage
- Default filename: `quedonde_session_state.json` in the workspace root.
- Alternate per-session filenames are allowed (`quedonde_session_state_<session_id>.json`).
- Checkpoints stay local; they are not committed to the public repository.

## CLI/API Integration Targets
- `python quedonde.py session dump --session-id climate_migration --scope-subsystem Climate --scope-file src/climate.py --out checkpoints/climate.json`
- `python quedonde.py session resume --input checkpoints/climate.json`
- During `session resume`, quedonde re-queries the SQLite tables to verify symbol existence, definition paths, and fan-in/out counts. Warnings are emitted to `stderr` when drift is detected. Pass `--skip-verify` to bypass this behavior (not recommended outside of CI mocks).
- Python helpers: `from session_state import SessionState, load_session_state, validate_session_state_dict`.

## References
- Design note: `documentation/specs/speculative/quedonde_session_state_checkpointing.md`
- Implementation plan: `planning/session_state_checkpointing/plan_session_state_checkpointing.md`
- Schema artifact: `documentation/specs/session_state.schema.json`
