## Quedonde 2.0 Upgrade Plan

This plan translates the `quedonde_2-0_structural_code_intelligence_spec.md` requirements into concrete, verifiable work packages. Every section lists objective deliverables plus explicit verification steps so progress can be audited.

### 1. Governance & Paper Trail
- **Single source**: This document is the canonical plan. No change may be implemented unless the corresponding task entry exists here.
- **Mandatory paper trail procedure**:
	1. Before starting work on an item, add a checkbox entry for it (see per-phase checklists) and reference the plan section in the commit message (e.g., `P2.2 regex extractor`).
	2. After completing work, fill in the "Evidence" placeholder with commit SHA(s), test IDs, or log locations, then tick the checkbox.
	3. Append a dated note to the change log describing what changed, why, and where it lives (file paths + commit IDs).
	4. If scope changes, update the relevant phase text _first_ and capture that change in the log before writing code.
- **Evidence links**: Each checklist item contains an "Evidence" placeholder. A box can only be marked complete once the evidence reference is supplied.

### 2. Phase Breakdown

#### Phase 0 – Repository Instrumentation (Target: Day 0–1)
- Tasks:
	- [ ] P0-T1 Capture current schema (`.schema`) and table counts.
	- [ ] P0-T2 Add lightweight migration harness (SQL scripts callable from `quedonde.py`).
	- [ ] P0-T3 Implement automated sanity check (`python quedonde.py diagnose`).
- Deliverables: SQL snapshot, `diagnose` command, documentation update.
- Verification: `diagnose` exits 0 and reports table counts; migration harness runs without modifying data.
- Checklist:
	- [x] **P0.1** Schema snapshot committed to `documentation/schema/` (Evidence: `documentation/schema/2025-12-12_schema.sql`, `documentation/schema/2025-12-12_table_counts.json`)
	- [x] **P0.2** Migration harness callable from CLI & module (Evidence: `quedonde.py` `run_migrations()` + `python quedonde.py migrate --dry-run`)
	- [x] **P0.3** `diagnose` command exercised on sample repo (Evidence: `python quedonde.py diagnose --json` output capturing table counts)

#### Phase 1 – Storage Layer Extension (Day 1–3)
- Tasks:
	- [ ] P1-T1 Introduce `symbols`, `edges`, `annotations` tables exactly as specified.
	- [ ] P1-T2 Write schema migration guarding against multiple runs.
	- [ ] P1-T3 Extend cache invalidation logic to observe non-FTS tables.
- Deliverables: schema migration script; updated cache routines; unit test covering repeated migration run.
- Verification: Run migration on clean DB (tables created) and on already-migrated DB (no-op). Confirm cache clears when structural tables mutate.
- Checklist:
	- [x] **P1.1** `symbols/edges/annotations` tables created via migration (Evidence: `migrations/001_structural_tables.sql`, `python quedonde.py migrate`)
	- [x] **P1.2** Second migration run is a no-op (Evidence: `python quedonde.py migrate --dry-run` after applying 001)
	- [x] **P1.3** Cache invalidates when structural tables change (Evidence: `quedonde.py` structural-versioned cache + `migrations/002_structural_state_triggers.sql`; verified via `python quedonde.py quedonde` before/after inserting demo symbol)

#### Phase 2 – Symbol Extraction Pipeline (Day 3–7)
- Tasks:
	- [ ] P2-T1 Build extension-based language classifier.
	- [ ] P2-T2 Implement regex extractors per language group (Python, C/C++, JS/TS, JSON/YAML, Markdown).
	- [ ] P2-T3 Persist symbol spans with kind/line numbers; resolve decorator ownership for Python.
	- [ ] P2-T4 Benchmark extraction cost per file and add safeguards for >5k LOC files.
- Deliverables: extractor module plus regression tests using fixture corpora.
- Verification: Run `diagnose --symbols` to report symbol count; sample manual spot-check across each language class.
- Checklist:
	- [x] **P2.1** Extension classifier table documented (Evidence: `documentation/specs/extension_classifier.md`)
	- [x] **P2.2** Regex extractor fixtures committed (Evidence: `documentation/fixtures/structural/*` + structural extractor implementation in `quedonde.py`)
	- [x] **P2.3** Decorator ownership handled for Python (Evidence: `quedonde.py` `_extract_python_symbols` + fixture `documentation/fixtures/structural/python_sample.py`; verification captured in `documentation/reports/decorator_verification.md`)
	- [x] **P2.4** Benchmark report for >5k LOC files (Evidence: `MAX_STRUCTURAL_LINES` guard + CLI `python quedonde.py benchmark_structural --json`; summary captured in `documentation/reports/extraction_benchmark.md`)

#### Phase 3 – Relationship & Annotation Capture (Day 7–10)
- Tasks:
	- [ ] P3-T1 Implement intra-file relation detection (`imports`, `includes`, `calls`, `references`, `owns`).
	- [ ] P3-T2 Parse `@quedonde:<tag>` annotations.
	- [ ] P3-T3 Ensure extraction pipeline writes both edges and annotations in the same transaction as symbols.
- Deliverables: relation extractor, annotation parser, tests for each relation/tag.
- Verification: Run targeted fixtures verifying expected edge counts; annotations query returns correct tags.
- Checklist:
	- [x] **P3.1** Relation fixtures cover all relation types (Evidence: `documentation/fixtures/structural/*` + `documentation/specs/relations_annotations.md`; verified via `python quedonde.py index` + `diagnose --json` edge counts)
	- [x] **P3.2** Annotation parser detects every supported tag (Evidence: `documentation/fixtures/structural/python_sample.py`, `documentation/fixtures/structural/notes_sample.md`, and `documentation/reports/relations_annotations_verification.md`)
	- [x] **P3.3** Single transaction writes symbols + edges + annotations (Evidence: `quedonde.py` `update_structural_data()` inserts into all tables per file; confirmed through `python quedonde.py index` and trigger-driven `diagnose --json` output)

#### Phase 4 – Query Engine Expansion (Day 10–13)
- Tasks:
	- [ ] P4-T1 Add intent classifier mapping natural-language prompts to structural queries.
	- [ ] P4-T2 Implement handlers: `find_symbol`, `callers`, `deps`, `explain`.
	- [ ] P4-T3 Ensure fallbacks label responses when structural context missing.
- Deliverables: query dispatcher, new Python API functions, CLI plumbing stubs.
- Verification: Automated tests covering each intent; manual CLI dry-run per command.
- Checklist:
	- [x] **P4.1** Intent classifier rules documented (Evidence: `documentation/specs/intent_classifier.md`, `documentation/fixtures/intent/queries.md`)
	- [x] **P4.2** Python APIs (`find_symbol`, `callers`, `deps`, `explain`) return structured data (Evidence: `quedonde.py` structural query helpers + `handle_structural_intent()` dispatch)
	- [x] **P4.3** CLI smoke tests recorded for each new command (Evidence: `documentation/reports/intent_cli_smoketest.md`, sample `python quedonde.py ask --json "who calls update_structural_data"`)

#### Phase 5 – CLI & API Surface (Day 13–15)
Status: _Completed (2025-12-12)_ — CLI + API surfaces, docs, and tests finalized.
- Tasks:
	- [x] P5-T1 Introduce new CLI commands per spec while keeping existing search flags.
	- [x] P5-T2 Document new options in README and `--help` output (_README coverage already queued to expand alongside implementation_).
	- [x] P5-T3 Provide Python API wrappers returning structured dicts.
- Deliverables: CLI help text, README section, docstrings.
- Verification: `quedonde.py --help` lists new commands; README walkthrough demonstrates structural queries.
- Checklist:
	- [x] **P5.1** CLI help output captured in docs (Evidence: `README.md` structural command section + `quedonde.py` feature list)
	- [x] **P5.2** README structural search tutorial merged (Evidence: `README.md` "Deterministic structural commands" walkthrough)
	- [x] **P5.3** Python API docstrings audited (Evidence: `quedonde.py` public helpers `find_symbol`/`get_callers`/`get_dependencies`/`explain_symbol`)

#### Phase 6 – Quality, Performance, Release (Day 15–18)
Status: _Completed (2025-12-12)_ — performance benchmarks locked and release v2.0.0 documented/tagged.
- Tasks:
	- [x] P6-T1 Expand automated tests (unit + golden data) to cover structural features.
	- [x] P6-T2 Benchmark indexing 10k files and structural queries; optimise until targets met (<3s indexing, <100ms queries).
	- [x] P6-T3 Finalise changelog entry, bump version to 2.0.0, and tag release.
- Deliverables: benchmark report, test suite updates, release notes.
- Verification: CI pipeline captures runtime metrics; `CHANGELOG.md` records release summary; git tag `v2.0.0` created.
- Checklist:
	- [x] **P6.1** Structural feature tests land in CI (Evidence: `tests/test_indexing.py` + fixture repo `documentation/fixtures/indexing_repo/`)
	- [x] **P6.2** Benchmark sheet shows <3s indexing / <100ms queries (Evidence: `documentation/reports/benchmark_results.md`)
	- [x] **P6.3** Release notes + tag `v2.0.0` published (Evidence: `documentation/reports/release_notes_v2.0.0.md`)

 _2025-12-12_: Captured CLI help transcript (`documentation/reports/cli_help.md`) and added structural snapshot tests (`tests/test_structural_cli.py` + fixture `documentation/fixtures/structural_cli/sample.py`) to guard the new commands.
 _2025-12-12_: Phase 6 kicked off with indexing regression coverage (`tests/test_indexing.py`) using fixture repo `documentation/fixtures/indexing_repo/` to ensure structural tables populate during CI.
- **Developer ergonomics**: Provide sample scripts showing how to consume structural data (e.g., dependency graph export).
- Checklist:
	- [ ] **X.1** Determinism audit report attached (Evidence: __________)
	- [ ] **X.2** Feature flag fallback documented + tested (Evidence: __________)
	- [ ] **X.3** Sample consumption scripts published (Evidence: __________)

### 4. Acceptance Checklist
- [ ] **A.1** Migration can be run idempotently (Evidence: __________)
- [ ] **A.2** Extraction pipeline populates `symbols`, `edges`, `annotations` across all supported languages (Evidence: __________)
- [ ] **A.3** New CLI commands return results with traceable provenance (Evidence: __________)
- [x] **A.4** Performance targets met on reference repository (Evidence: `documentation/reports/benchmark_results.md`)
- [x] **A.5** Documentation (README + CHANGELOG + `documentation/reports/release_notes_v2.0.0.md`) reflects 2.0 capabilities

### 5. Change Log (update upon every change)
- _2025-12-11_: Initial 2.0 upgrade plan drafted.
- _2025-12-11_: Added explicit paper-trail procedure plus evidence-based checklists for every phase/cross-cutting item/acceptance gate.
- _2025-12-11_: Converted per-phase task lists into checkable checkbox items (IDs P*-T*).
- _2025-12-12_: Completed Phase 0 instrumentation: schema snapshot archived, migration harness added to `quedonde.py`, and `diagnose` CLI command implemented/tested.
- _2025-12-12_: Phase 1 storage extension kicked off with `migrations/001_structural_tables.sql`; applied via `python quedonde.py migrate` and validated idempotency with `--dry-run`.
- _2025-12-12_: Finished Phase 1 cache invalidation work by introducing structural-version tracking in `quedonde.py`, auto-updating triggers via `migrations/002_structural_state_triggers.sql`, and demonstrating cache busts on structural mutations.
- _2025-12-12_: Phase 2 started by implementing the extension classifier (`documentation/specs/extension_classifier.md`), committing extractor fixtures under `documentation/fixtures/structural/`, and wiring the regex-based pipeline so `python quedonde.py index` now populates `symbols` (verified via `diagnose --json`).
- _2025-12-12_: Resolved Python decorator ownership, added the 5k LOC safeguard, and introduced `benchmark_structural` reporting (see `quedonde.py`, `documentation/fixtures/structural/python_sample.py`, and `documentation/reports/extraction_benchmark.md`).
- _2025-12-12_: Completed Phase 3 by adding relation/annotation extractors (imports/includes/calls/references/owns + `@quedonde:<tag>` parser), documenting heuristics, and demonstrating populated `edges`/`annotations` tables via `diagnose --json` and the verification report.
- _2025-12-12_: Began Phase 4 by documenting the intent classifier/ask workflow, adding structured handlers in `quedonde.py`, introducing intent fixtures, and capturing CLI smoke test output under `documentation/reports/intent_cli_smoketest.md`.
- _2025-12-12_: Improved the `ask` CLI formatting, documented the workflow in `README.md`, and added intent classifier regression tests backed by `documentation/fixtures/intent/queries.md` + `tests/test_intent_classifier.py`.
- _2025-12-12_: Phase 5 officially in progress; README/CLI help alignment and automated test discovery work queued so new commands can ship with documentation and `python -m unittest` coverage.
- _2025-12-12_: Wired Phase 5 CLI surface (`find`, `callers`, `deps`, `explain`) in `quedonde.py`, expanded README/help text, added Python API wrappers, and enabled default unittest discovery via `test_all.py` + `tests/__init__.py`.
- _2025-12-12_: Captured CLI help transcript (`documentation/reports/cli_help.md`) and added structural snapshot tests (`tests/test_structural_cli.py` + fixture `documentation/fixtures/structural_cli/sample.py`) to guard the new commands.
- _2025-12-12_: Phase 6 kicked off with indexing regression coverage (`tests/test_indexing.py`) using fixture repo `documentation/fixtures/indexing_repo/` to ensure structural tables populate during CI.
- _2025-12-12_: Benchmarks recorded (<3s / <100ms) and release v2.0.0 documented (see `documentation/reports/benchmark_results.md`, `documentation/reports/release_notes_v2.0.0.md`, `CHANGELOG.md`).
