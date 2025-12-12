# quedonde 2.0 — Structural Code Intelligence Specification

**Document type:** Technical design + implementation roadmap
**Target:** Single-file Python tool, offline, deterministic
**Audience:** Systems engineers, tooling authors, refactor-heavy codebases

---

## 0. Executive Summary

`quedonde` currently provides fast, incremental **lexical search** over a repository using SQLite FTS5. This document specifies an extension to **structural code intelligence** while preserving the following invariants:

* Single-file distribution (`quedonde.py`)
* Offline operation
* Zero non-stdlib runtime dependencies
* Deterministic indexing and query behavior
* CLI and Python API parity

The goal is to support *refactor reasoning*, *dependency discovery*, and *architectural intent inspection* without introducing ASTs, language servers, or opaque ML inference.

---

## 1. Design Constraints (Hard)

1. **No full parsers or compilers**

   * Heuristic extraction only
   * Regex + line-based scanning

2. **Incremental indexing only**

   * File modification time remains the sole invalidation signal

3. **Failure-tolerant intelligence**

   * False negatives acceptable
   * False positives minimized
   * Tool must degrade gracefully to lexical search

4. **Explainability over cleverness**

   * Every result must have a traceable reason (symbol, edge, tag)

---

## 2. Storage Model Extensions

### 2.1 Existing Tables (Unchanged)

```sql
meta(path TEXT PRIMARY KEY, mtime REAL)
files(path TEXT, content TEXT) -- FTS5
```

### 2.2 New Table: symbols

Stores *definitions* extracted from files.

```sql
symbols(
  path TEXT NOT NULL,
  symbol TEXT NOT NULL,
  kind TEXT NOT NULL,          -- class | function | method | namespace | key | heading
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  PRIMARY KEY(path, symbol, kind)
)
```

**Indexing:**

* `(symbol)`
* `(path)`

---

### 2.3 New Table: edges

Stores *relationships* between symbols.

```sql
edges(
  src_path TEXT NOT NULL,
  src_symbol TEXT NOT NULL,
  relation TEXT NOT NULL,      -- calls | imports | includes | references | owns
  dst_path TEXT,
  dst_symbol TEXT
)
```

**Notes:**

* `dst_path` may be NULL if unresolved
* Edges are directional

---

### 2.4 New Table: annotations

Stores explicit architectural intent.

```sql
annotations(
  path TEXT NOT NULL,
  symbol TEXT,
  tag TEXT NOT NULL,           -- legacy | bridge | orchestrator | deprecated | temporary
  line INTEGER
)
```

Tags are discovered via comments or docstrings.

---

## 3. Symbol Extraction Specification

### 3.1 Extraction Pipeline

For each indexed file:

1. Load file contents (existing behavior)
2. Run language classifier by extension
3. Apply ordered regex passes
4. Record symbol spans

Extraction must be **pure** (no cross-file context).

---

### 3.2 Language-Specific Heuristics

#### Python (`.py`)

* `^class\s+(\w+)`
* `^def\s+(\w+)`
* Decorators associated with following def/class

#### C / C++ (`.c`, `.cpp`, `.h`)

* `^\s*(class|struct)\s+(\w+)`
* Function signatures ending with `{`
* `namespace (\w+)`

#### JavaScript / TypeScript

* `function (\w+)`
* `class (\w+)`
* `export (class|function)`

#### JSON / YAML

* Top-level keys only

#### Markdown

* Headings mapped as symbols (`heading` kind)

---

## 4. Relationship (Edge) Extraction

Edges are collected **within the same file** only.

### 4.1 Supported Relations

| Relation   | Trigger                     |
| ---------- | --------------------------- |
| imports    | `import`, `from`, `require` |
| includes   | `#include`                  |
| calls      | `symbolName(`               |
| references | qualified name usage        |
| owns       | class → method containment  |

### 4.2 Edge Resolution Rules

* Prefer local symbols
* Fallback to filename-based inference
* Never invent cross-file edges without textual evidence

---

## 5. Provenance Annotation System

### 5.1 Syntax

Annotations are parsed from comments:

```text
@quedonde:legacy
@quedonde:bridge
@quedonde:orchestrator
@quedonde:deprecated
@quedonde:temporary
```

### 5.2 Semantics

| Tag          | Meaning                               |
| ------------ | ------------------------------------- |
| legacy       | Old system retained for compatibility |
| bridge       | Transitional glue between systems     |
| orchestrator | High-level control surface            |
| deprecated   | Scheduled for removal                 |
| temporary    | Known short-lived code                |

---

## 6. Query Engine Architecture

### 6.1 Query Intent Classification

Queries are classified via deterministic rules:

* `where is X defined` → symbols
* `who calls X` → edges (reverse)
* `what depends on X` → edges
* `why does X exist` → annotations
* fallback → FTS

### 6.2 Execution Model

1. Parse query
2. Determine intent
3. Execute minimal table set
4. Merge + label results

No query may silently fallback without marking uncertainty.

---

## 7. CLI Surface (Proposed)

```bash
quedonde index
quedonde search <text>
quedonde find <symbol>
quedonde callers <symbol>
quedonde deps <symbol>
quedonde explain <symbol>
```

All commands support:

* `--json`
* `--paths`
* `--context N`

---

## 8. Python API Surface

```python
find_symbol(name)
get_callers(name)
get_dependencies(name)
explain_symbol(name)
```

Returns structured dictionaries; no printing.

---

## 9. Indexing & Cache Behavior

* Structural tables updated only when file mtime changes
* Cache invalidated when symbol or edge tables mutate
* VACUUM and FTS optimize unchanged

---

## 10. Performance Targets

| Operation        | Target  |
| ---------------- | ------- |
| Index 10k files  | < 3s    |
| Symbol query     | < 50ms  |
| Dependency query | < 100ms |

---

## 11. Deferred / Optional Phase

### Semantic Ranking Layer (Explicitly Optional)

* Embeddings only for comments/docstrings
* Bound to symbol IDs
* Used for ranking only, never generation

---

## 12. Invariants & Anti-Goals

* No language servers
* No AST frameworks
* No network access
* No hallucinated relationships
* No non-deterministic ranking

---

## 13. Design Philosophy

> Structural truth beats semantic cleverness.

`quedonde` should answer *how the code is shaped*, not *what it feels like*.
