#!/usr/bin/env python3

"""

Ultimate single-file code search helper for LLMs.



Features:

- Incremental FTS5 index (SQLite)

- Filename + content search

- Fuzzy search (--fuzzy)

- Context lines (--context N)

- JSON output (--json)

- Lightweight cache for repeated queries

- Direct Python API: search_repo(query, ...)

- Structural commands: find/callers/deps/explain plus natural-language `ask`

- Fully offline, no server, no dependencies

"""

__version__ = "2.0.0"



import os, sys, sqlite3, difflib, hashlib, pickle, json, time, re, bisect, fnmatch
from pathlib import Path

from typing import List, Dict, Optional, Tuple, Callable, Set

_SESSION_STATE_TEMPLATE = r'''"""Session state checkpoint schema helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Set

SCHEMA_VERSION = "1.0"
ALLOWED_RELATIONS = {"calls", "imports", "includes", "references", "annotates"}
CONFIDENCE_LEVELS = {"low", "medium", "high"}
MAX_TEXT_FIELD_LENGTH = 240
MAX_QUESTION_LENGTH = 200
MAX_SYMBOL_NAME = 128
CONTEXT_LEVELS = {0, 1, 2, 3}
CONTEXT_HISTORY_FILE = Path(".quedonde_context_history.json")
MAX_SOURCE_PATHS = 5
MIN_SESSION_ID_LENGTH = 3


class SessionStateValidationError(ValueError):
    """Raised when a checkpoint payload violates the schema."""


@dataclass
class Evidence:
    fan_in: int
    fan_out: int
    annotations: List[str] = field(default_factory=list)

    def validate(self) -> None:
        _ensure(self.fan_in >= 0, "fan_in must be non-negative")
        _ensure(self.fan_out >= 0, "fan_out must be non-negative")
        for value in self.annotations:
            _ensure(bool(value) and len(value) <= 64, "annotation entries must be <= 64 chars")


@dataclass
class SymbolSummary:
    kind: str
    role: str
    evidence: Evidence
    source_paths: List[str] = field(default_factory=list)
    context: Optional["ContextBookmark"] = None

    def validate(self, symbol_name: str) -> None:
        _ensure(bool(symbol_name) and len(symbol_name) <= MAX_SYMBOL_NAME, "invalid symbol name")
        _ensure(bool(self.kind) and len(self.kind) <= 64, "invalid symbol kind")
        _ensure(bool(self.role) and len(self.role) <= 64, "invalid role")
        _ensure(bool(self.source_paths), "each symbol summary requires at least one source path")
        _ensure(len(self.source_paths) <= MAX_SOURCE_PATHS, "source_paths must contain at most five entries")
        for path in self.source_paths:
            _ensure(bool(path) and len(path) <= 256, "source paths must be <= 256 chars")
        self.evidence.validate()
        if self.context:
            self.context.validate()


@dataclass
class ContextBookmark:
    level: int
    status: str
    timestamp: str

    def validate(self) -> None:
        _ensure(self.level in CONTEXT_LEVELS, "context level must be 0-3")
        _ensure(_valid_text(self.status, 64), "context status invalid")
        _ensure(_is_iso_timestamp(self.timestamp), "context timestamp invalid")


@dataclass
class DependencyRecord:
    src: str
    relation: str
    dst: str

    def validate(self) -> None:
        for value in (self.src, self.dst):
            _ensure(bool(value) and len(value) <= MAX_SYMBOL_NAME, "dependency endpoints required")
        _ensure(self.relation in ALLOWED_RELATIONS, f"relation must be one of {sorted(ALLOWED_RELATIONS)}")


@dataclass
class DecisionRecord:
    decision: str
    reason: str

    def validate(self) -> None:
        _ensure(_valid_text(self.decision), "decision text invalid")
        _ensure(_valid_text(self.reason), "reason text invalid")


def _valid_text(value: str, limit: int = MAX_TEXT_FIELD_LENGTH, *, min_length: int = 1) -> bool:
    return bool(value) and min_length <= len(value) <= limit and "\n" not in value and "\r" not in value


@dataclass
class SessionScope:
    subsystems: List[str]
    files: List[str]

    def validate(self) -> None:
        _ensure(bool(self.subsystems), "scope.subsystems must not be empty")
        _ensure(bool(self.files), "scope.files must not be empty")
        for entry in self.subsystems:
            _ensure(bool(entry) and len(entry) <= 64, "subsystem entries must be <= 64 chars")
        for entry in self.files:
            _ensure(bool(entry) and len(entry) <= 256, "file paths must be <= 256 chars")


@dataclass
class SessionState:
    session_id: str
    timestamp: str
    structural_version: str
    scope: SessionScope
    symbols: Dict[str, SymbolSummary]
    confidence: str
    schema_version: str = SCHEMA_VERSION
    dependencies: List[DependencyRecord] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)

    def validate(self) -> None:
        _ensure(self.schema_version == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
        _ensure(
            _valid_text(self.session_id, 128, min_length=MIN_SESSION_ID_LENGTH),
            f"session_id must be at least {MIN_SESSION_ID_LENGTH} chars",
        )
        _ensure(self.confidence in CONFIDENCE_LEVELS, "confidence must be low/medium/high")
        _ensure(bool(self.structural_version), "structural_version required")
        _ensure(_is_iso_timestamp(self.timestamp), "timestamp must be RFC3339/ISO")
        self.scope.validate()
        _ensure(bool(self.symbols), "symbols map must not be empty")
        for name, summary in self.symbols.items():
            summary.validate(name)
        for dep in self.dependencies:
            dep.validate()
        for decision in self.decisions:
            decision.validate()
        for question in self.open_questions:
            _ensure(_valid_text(question, MAX_QUESTION_LENGTH), "open question text invalid")


def load_session_state(path: Path) -> SessionState:
    data = json.loads(path.read_text(encoding="utf-8"))
    state = session_state_from_dict(data)
    attach_context_history(state)
    return state


def session_state_from_dict(payload: Mapping[str, Any]) -> SessionState:
    payload = _ensure_mapping(payload, "session state payload")
    _ensure_allowed_keys(
        payload,
        {
            "schema_version",
            "session_id",
            "timestamp",
            "structural_version",
            "scope",
            "symbols",
            "dependencies",
            "decisions",
            "open_questions",
            "confidence",
        },
        "session state payload",
    )

    scope_dict = _ensure_mapping(payload.get("scope") or {}, "scope")
    _ensure_allowed_keys(scope_dict, {"subsystems", "files"}, "scope")
    scope = SessionScope(
        subsystems=list(scope_dict.get("subsystems") or []),
        files=list(scope_dict.get("files") or []),
    )
    symbols_payload = payload.get("symbols") or {}
    _ensure(isinstance(symbols_payload, Mapping), "symbols must be an object keyed by symbol name")
    symbols: Dict[str, SymbolSummary] = {}
    for name, summary in symbols_payload.items():
        summary_map = _ensure_mapping(summary, f"symbol '{name}'")
        _ensure_allowed_keys(summary_map, {"kind", "role", "evidence", "source_paths"}, f"symbol '{name}'")
        evidence_payload = _ensure_mapping(summary_map.get("evidence") or {}, f"symbol '{name}'.evidence")
        _ensure_allowed_keys(evidence_payload, {"fan_in", "fan_out", "annotations"}, f"symbol '{name}'.evidence")
        symbols[name] = SymbolSummary(
            kind=summary_map.get("kind", ""),
            role=summary_map.get("role", ""),
            evidence=Evidence(
                fan_in=int(evidence_payload.get("fan_in", 0)),
                fan_out=int(evidence_payload.get("fan_out", 0)),
                annotations=list(evidence_payload.get("annotations") or []),
            ),
            source_paths=list(summary_map.get("source_paths") or []),
        )
    dependencies_payload = payload.get("dependencies") or []
    _ensure(isinstance(dependencies_payload, list), "dependencies must be a list")
    dependencies = [
        _dependency_from_mapping(item)
        for item in dependencies_payload
    ]
    decisions_payload = payload.get("decisions") or []
    _ensure(isinstance(decisions_payload, list), "decisions must be a list")
    decisions = [
        _decision_from_mapping(item)
        for item in decisions_payload
    ]
    open_questions = list(payload.get("open_questions") or [])
    state = SessionState(
        session_id=str(payload.get("session_id", "")),
        timestamp=str(payload.get("timestamp", "")),
        structural_version=str(payload.get("structural_version", "")),
        scope=scope,
        symbols=symbols,
        confidence=str(payload.get("confidence", "")),
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        dependencies=dependencies,
        decisions=decisions,
        open_questions=open_questions,
    )
    state.validate()
    return state


def session_state_to_dict(state: SessionState) -> Dict[str, Any]:
    state.validate()
    return {
        "schema_version": state.schema_version,
        "session_id": state.session_id,
        "timestamp": state.timestamp,
        "structural_version": state.structural_version,
        "scope": {
            "subsystems": list(state.scope.subsystems),
            "files": list(state.scope.files),
        },
        "symbols": {
            name: {
                "kind": summary.kind,
                "role": summary.role,
                "evidence": {
                    "fan_in": summary.evidence.fan_in,
                    "fan_out": summary.evidence.fan_out,
                    "annotations": list(summary.evidence.annotations),
                },
                "source_paths": list(summary.source_paths),
            }
            for name, summary in state.symbols.items()
        },
        "dependencies": [
            {"src": dep.src, "relation": dep.relation, "dst": dep.dst}
            for dep in state.dependencies
        ],
        "decisions": [
            {"decision": dec.decision, "reason": dec.reason}
            for dec in state.decisions
        ],
        "open_questions": list(state.open_questions),
        "confidence": state.confidence,
    }


def dump_session_state(state: SessionState, path: Path) -> None:
    payload = session_state_to_dict(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _is_iso_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise SessionStateValidationError(message)


def _ensure_mapping(value: Any, location: str) -> Mapping[str, Any]:
    _ensure(isinstance(value, Mapping), f"{location} must be an object")
    return value


def _ensure_allowed_keys(payload: Mapping[str, Any], allowed: Set[str], location: str) -> None:
    extras = set(payload.keys()) - allowed
    _ensure(not extras, f"{location} contains unsupported fields: {', '.join(sorted(extras))}")


def _dependency_from_mapping(payload: Mapping[str, Any]) -> DependencyRecord:
    mapping = _ensure_mapping(payload, "dependency entry")
    _ensure_allowed_keys(mapping, {"src", "relation", "dst"}, "dependency entry")
    record = DependencyRecord(
        src=str(mapping.get("src", "")),
        relation=str(mapping.get("relation", "")),
        dst=str(mapping.get("dst", "")),
    )
    record.validate()
    return record


def _decision_from_mapping(payload: Mapping[str, Any]) -> DecisionRecord:
    mapping = _ensure_mapping(payload, "decision entry")
    _ensure_allowed_keys(mapping, {"decision", "reason"}, "decision entry")
    record = DecisionRecord(
        decision=str(mapping.get("decision", "")),
        reason=str(mapping.get("reason", "")),
    )
    record.validate()
    return record


def new_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_context_history(storage_path: Optional[Path] = None) -> Dict[str, Any]:
    target = storage_path or CONTEXT_HISTORY_FILE
    try:
        if not target.exists():
            return {}
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}


def record_context_event(
    symbol: str,
    level: int,
    status: str,
    *,
    storage_path: Optional[Path] = None,
) -> None:
    target = storage_path or CONTEXT_HISTORY_FILE
    try:
        history = load_context_history(target)
        history[str(symbol)] = {
            "symbol": symbol,
            "level": int(level),
            "status": status,
            "timestamp": new_timestamp(),
        }
        target.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _context_bookmark_from_mapping(entry: Mapping[str, Any]) -> Optional[ContextBookmark]:
    try:
        bookmark = ContextBookmark(
            level=int(entry.get("level", 0)),
            status=str(entry.get("status", "")) or "unknown",
            timestamp=str(entry.get("timestamp", new_timestamp())),
        )
        bookmark.validate()
        return bookmark
    except Exception:
        return None


def attach_context_history(
    state: SessionState,
    history: Optional[Mapping[str, Any]] = None,
    *,
    storage_path: Optional[Path] = None,
) -> None:
    payload = history if history is not None else load_context_history(storage_path)
    if not payload:
        return
    for name, summary in state.symbols.items():
        if summary.context:
            continue
        entry = payload.get(name)
        if isinstance(entry, Mapping):
            bookmark = _context_bookmark_from_mapping(entry)
            if bookmark:
                summary.context = bookmark
'''


def _bootstrap_session_state_module() -> None:
    module_path = Path(__file__).with_name("session_state.py")
    if module_path.exists():
        return
    try:
        module_path.write_text(_SESSION_STATE_TEMPLATE, encoding="utf-8")
    except Exception:
        pass


_bootstrap_session_state_module()

try:
    from session_state import record_context_event
except Exception:
    def record_context_event(*_args, **_kwargs):
        return None



# -----------------------------

# Config

# -----------------------------



DB = ".code_index.sqlite"

CACHE = ".code_index.cache"

IGNORES = {'.git', '__pycache__', 'node_modules', '.venv', 'dist', 'build'}

EXTS = (

    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".md", ".json",

    ".yaml", ".yml", ".cpp", ".h", ".java", ".go", ".rs", ".sh", ".txt",

    ".ps1", ".psm1", ".cmd", ".bat"

)

MIGRATIONS_DIR = "migrations"
STRUCTURAL_VERSION_KEY = "structural_version"
CACHE_META_KEY = "__meta__"
STRUCTURAL_TABLES = ("symbols", "edges", "annotations")
STRUCTURAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS symbols (
    path TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT,
    line_start INTEGER,
    line_end INTEGER
);
CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);

CREATE TABLE IF NOT EXISTS edges (
    src_path TEXT NOT NULL,
    src_symbol TEXT NOT NULL,
    relation TEXT NOT NULL,
    dst_path TEXT,
    dst_symbol TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_src_symbol ON edges(src_symbol);
CREATE INDEX IF NOT EXISTS idx_edges_dst_symbol ON edges(dst_symbol);
CREATE INDEX IF NOT EXISTS idx_edges_src_path ON edges(src_path);

CREATE TABLE IF NOT EXISTS annotations (
    path TEXT NOT NULL,
    symbol TEXT,
    tag TEXT NOT NULL,
    line INTEGER
);
CREATE INDEX IF NOT EXISTS idx_annotations_symbol ON annotations(symbol);
CREATE INDEX IF NOT EXISTS idx_annotations_path ON annotations(path);
"""
MAX_STRUCTURAL_LINES = 5000
CONTEXT_LINE_CAP = 800
CONTEXT_ALLOWED_LEVELS = {0, 1, 2, 3}
CONTEXT_DEFAULT_LEVEL = 1
FILE_DEP_SUMMARY_LIMIT = 25
ROLE_PRIORITY = ("orchestrator", "bridge", "legacy", "deprecated", "temporary")
FILE_SYMBOL = "__file__"
ANNOTATION_TAGS = {"legacy", "bridge", "orchestrator", "deprecated", "temporary"}
ANNOTATION_RE = re.compile(r"@quedonde:(legacy|bridge|orchestrator|deprecated|temporary)", re.IGNORECASE)
CALL_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")
SELF_ATTR_RE = re.compile(r"\bself\.([A-Za-z_][\w]*)")
THIS_ATTR_RE = re.compile(r"\bthis(?:->|\.)([A-Za-z_][\w]*)")
CALL_KEYWORDS = {
    "if",
    "for",
    "while",
    "return",
    "class",
    "def",
    "with",
    "switch",
    "case",
    "sizeof",
    "delete",
    "new",
    "catch",
    "try",
}
INTENT_STOP_WORDS = {
    "what",
    "is",
    "who",
    "calls",
    "callers",
    "depends",
    "depend",
    "on",
    "the",
    "a",
    "an",
    "explain",
    "find",
    "symbol",
    "show",
    "me",
    "locate",
    "where",
    "does",
    "use",
    "uses",
    "of",
    "for",
    "function",
    "list",
    "all",
    "defined",
    "definition",
    "definitions",
    "usage",
    "incoming",
    "reference",
    "references",
    "caller",
    "callers",
    "call",
    "invokes",
    "invoke",
    "context",
    "deps",
}

LANGUAGE_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "python": (".py",),
    "cpp": (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"),
    "javascript": (".js", ".jsx"),
    "typescript": (".ts", ".tsx"),
    "json": (".json",),
    "yaml": (".yml", ".yaml"),
    "markdown": (".md",),
    "powershell": (".ps1", ".psm1"),
    "batch": (".cmd", ".bat"),
}
LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ext: language
    for language, extensions in LANGUAGE_EXTENSIONS.items()
    for ext in extensions
}

SymbolRecord = Tuple[str, str, int, int]
SymbolExtractor = Callable[[str], List[SymbolRecord]]

_STRUCTURAL_READY: Optional[bool] = None



# -----------------------------

# Database & indexing

# -----------------------------



def connect_db():

    conn = sqlite3.connect(DB)

    conn.row_factory = sqlite3.Row

    conn.execute("CREATE TABLE IF NOT EXISTS meta(path TEXT PRIMARY KEY, mtime REAL)")

    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS files USING fts5(path, content)")

    conn.execute("CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)")

    ensure_structural_tables(conn)

    return conn


# -----------------------------
# Migrations
# -----------------------------


def ensure_migration_tracking(conn: sqlite3.Connection) -> None:

    conn.execute(

        """

        CREATE TABLE IF NOT EXISTS schema_migrations (

            name TEXT PRIMARY KEY,

            applied_at REAL

        )

        """

    )


def discover_migration_scripts(directory: str = MIGRATIONS_DIR) -> List[Path]:

    base = Path(directory)

    if not base.exists():

        return []

    scripts = [p for p in base.iterdir() if p.is_file() and p.suffix == ".sql"]

    return sorted(scripts, key=lambda p: p.name)


def get_applied_migrations(conn: sqlite3.Connection) -> Dict[str, float]:

    ensure_migration_tracking(conn)

    rows = conn.execute("SELECT name, applied_at FROM schema_migrations").fetchall()

    return {row[0]: row[1] for row in rows}


def run_migrations(

    conn: sqlite3.Connection,

    *,

    directory: str = MIGRATIONS_DIR,

    dry_run: bool = False,

) -> List[str]:

    scripts = discover_migration_scripts(directory)

    applied = get_applied_migrations(conn)

    pending = [path for path in scripts if path.name not in applied]

    if dry_run:

        return [p.name for p in pending]

    for script_path in pending:

        with open(script_path, "r", encoding="utf-8") as script_file:

            sql = script_file.read()

        conn.executescript(sql)

        conn.execute(

            "INSERT INTO schema_migrations(name, applied_at) VALUES(?, ?)",

            (script_path.name, time.time()),

        )

        conn.commit()

    if pending:

        mark_structural_change(conn)

        conn.commit()

        global _STRUCTURAL_READY

        _STRUCTURAL_READY = None

    return [p.name for p in pending]


# -----------------------------
# Structural state tracking
# -----------------------------


def get_structural_version(conn: sqlite3.Connection) -> str:

    row = conn.execute(

        "SELECT value FROM state WHERE key=?",

        (STRUCTURAL_VERSION_KEY,)

    ).fetchone()

    if row and row[0]:

        return row[0]

    return "0"


def mark_structural_change(conn: sqlite3.Connection) -> str:

    version = f"{time.time():.6f}"

    conn.execute(

        "INSERT OR REPLACE INTO state(key, value) VALUES(?, ?)",

        (STRUCTURAL_VERSION_KEY, version)

    )

    return version


# -----------------------------
# Structural extraction
# -----------------------------


def structural_ready(conn: sqlite3.Connection) -> bool:

    global _STRUCTURAL_READY

    if _STRUCTURAL_READY is None:

        try:

            names = [

                conn.execute(

                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",

                    (table,),

                ).fetchone()

                for table in STRUCTURAL_TABLES

            ]

            _STRUCTURAL_READY = all(names)

        except sqlite3.Error:

            _STRUCTURAL_READY = False

    return bool(_STRUCTURAL_READY)


def ensure_structural_tables(conn: sqlite3.Connection) -> None:

    global _STRUCTURAL_READY

    if structural_ready(conn):

        return

    try:

        conn.executescript(STRUCTURAL_SCHEMA_SQL)

        mark_structural_change(conn)

        conn.commit()

        _STRUCTURAL_READY = None

        print("[struct] initialized structural tables")

    except sqlite3.Error as exc:

        print(f"[struct] failed to initialize structural tables: {exc}", file=sys.stderr)


def classify_language(path: str) -> Optional[str]:

    _, ext = os.path.splitext(path)

    return LANGUAGE_BY_EXTENSION.get(ext.lower())


def _line_offsets(text: str) -> List[int]:

    offsets = [0]

    total = 0

    for line in text.splitlines(True):

        total += len(line)

        offsets.append(total)

    return offsets


def _line_from_offset(offsets: List[int], position: int) -> int:

    index = bisect.bisect_right(offsets, position)

    return max(1, index)


def _regex_symbol_scan(text: str, patterns: List[Tuple[str, re.Pattern]]) -> List[SymbolRecord]:

    offsets = _line_offsets(text)

    records: List[SymbolRecord] = []

    for kind, regex in patterns:

        for match in regex.finditer(text):

            symbol = match.group(1)

            line = _line_from_offset(offsets, match.start(1))

            records.append((symbol, kind, line, line))

    return records


PY_CLASS_DECL = re.compile(r"^class\s+([A-Za-z_][\w]*)")
PY_DEF_DECL = re.compile(r"^(?:async\s+)?def\s+([A-Za-z_][\w]*)")


def _estimate_block_end(lines: List[str], start_index: int) -> int:

    base_indent = len(lines[start_index - 1]) - len(lines[start_index - 1].lstrip(" \t"))

    end_line = start_index

    for idx in range(start_index, len(lines)):

        candidate = lines[idx]

        stripped = candidate.strip()

        if not stripped:

            continue

        indent = len(candidate) - len(candidate.lstrip(" \t"))

        if indent <= base_indent and idx + 1 > start_index:

            break

        end_line = idx + 1

    return max(end_line, start_index)


def _extract_python_symbols(text: str) -> List[SymbolRecord]:

    lines = text.splitlines()

    records: List[SymbolRecord] = []

    class_stack: List[Tuple[int, str]] = []

    decorator_start: Optional[int] = None

    decorator_depth = 0

    saw_decorator = False

    def reset_decorator() -> None:

        nonlocal decorator_start, decorator_depth, saw_decorator

        decorator_start = None

        decorator_depth = 0

        saw_decorator = False

    for line_number, raw in enumerate(lines, 1):

        stripped = raw.strip()

        if not stripped:

            if decorator_depth <= 0:

                reset_decorator()

            continue

        if stripped.startswith("#"):

            continue

        if stripped.startswith("@") or (saw_decorator and decorator_depth > 0):

            if decorator_start is None:

                decorator_start = line_number

            saw_decorator = True

            decorator_depth += stripped.count("(") - stripped.count(")")

            if decorator_depth < 0:

                decorator_depth = 0

            continue

        indent = len(raw) - len(raw.lstrip(" \t"))

        while class_stack and indent <= class_stack[-1][0]:

            class_stack.pop()

        class_match = PY_CLASS_DECL.match(stripped)

        if class_match:

            name = class_match.group(1)

            start_line = decorator_start or line_number

            end = _estimate_block_end(lines, line_number)

            records.append((name, "class", start_line, end))

            class_stack.append((indent, name))

            reset_decorator()

            continue

        def_match = PY_DEF_DECL.match(stripped)

        if def_match:

            name = def_match.group(1)

            kind = "method" if class_stack and indent > class_stack[-1][0] else "function"

            start_line = decorator_start or line_number

            end = _estimate_block_end(lines, line_number)

            records.append((name, kind, start_line, end))

            reset_decorator()

            continue

        reset_decorator()

    return records


CPP_PATTERNS: List[Tuple[str, re.Pattern]] = [

    ("class", re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_][\w]*)", re.MULTILINE)),

    ("namespace", re.compile(r"^\s*namespace\s+([A-Za-z_][\w]*)", re.MULTILINE)),

    (

        "function",

        re.compile(

            r"^\s*(?:[A-Za-z_][\w:<>,\s\*&\[\]]+)?\s+([A-Za-z_][\w:]*)\s*\([^;{}]*\)\s*\{",

            re.MULTILINE,

        ),

    ),

]


JS_TS_PATTERNS: List[Tuple[str, re.Pattern]] = [

    ("class", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][\w]*)", re.MULTILINE)),

    ("function", re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][\w]*)", re.MULTILINE)),

    (

        "function",

        re.compile(r"^\s*const\s+([A-Za-z_][\w]*)\s*=\s*\([^)]*\)\s*=>", re.MULTILINE),

    ),

]


POWERSHELL_PATTERNS: List[Tuple[str, re.Pattern]] = [

    (

        "function",

        re.compile(r"^\s*function\s+([A-Za-z_][\w\-]*)\s*(?:\(|\{)", re.IGNORECASE | re.MULTILINE),

    ),

    ("class", re.compile(r"^\s*class\s+([A-Za-z_][\w]*)", re.IGNORECASE | re.MULTILINE)),

    (

        "filter",

        re.compile(r"^\s*filter\s+([A-Za-z_][\w\-]*)\s*(?:\(|\{)", re.IGNORECASE | re.MULTILINE),

    ),

]


BATCH_LABEL_RE = re.compile(r"^\s*:([A-Za-z0-9_\.]+)", re.MULTILINE)


JSON_YAML_KEY_RE = re.compile(r'"?([A-Za-z0-9_\-\. ]+)"?\s*:')
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _extract_cpp_symbols(text: str) -> List[SymbolRecord]:

    return _regex_symbol_scan(text, CPP_PATTERNS)


def _extract_js_symbols(text: str) -> List[SymbolRecord]:

    return _regex_symbol_scan(text, JS_TS_PATTERNS)


def _extract_powershell_symbols(text: str) -> List[SymbolRecord]:

    return _regex_symbol_scan(text, POWERSHELL_PATTERNS)


def _extract_batch_symbols(text: str) -> List[SymbolRecord]:

    records: List[SymbolRecord] = []

    for line_number, raw in enumerate(text.splitlines(), 1):

        stripped = raw.strip()

        if not stripped or stripped.startswith("::"):

            continue

        match = BATCH_LABEL_RE.match(stripped)

        if match:

            label = match.group(1)

            if label:

                records.append((label, "label", line_number, line_number))

    return records


def _extract_top_level_keys(text: str) -> List[SymbolRecord]:

    records: List[SymbolRecord] = []

    for line_number, raw in enumerate(text.splitlines(), 1):

        if not raw.strip() or raw.strip().startswith(('#', '//')):

            continue

        indent = len(raw) - len(raw.lstrip(" \t"))

        if indent > 2:

            continue

        match = JSON_YAML_KEY_RE.search(raw)

        if match:

            key = match.group(1).strip('"')

            if key:

                records.append((key, "key", line_number, line_number))

    return records


def _extract_markdown_symbols(text: str) -> List[SymbolRecord]:

    offsets = _line_offsets(text)

    records: List[SymbolRecord] = []

    for match in MD_HEADING_RE.finditer(text):

        heading = match.group(2).strip()

        if heading:

            line = _line_from_offset(offsets, match.start(2))

            records.append((heading, "heading", line, line))

    return records


SYMBOL_EXTRACTORS: Dict[str, SymbolExtractor] = {

    "python": _extract_python_symbols,

    "cpp": _extract_cpp_symbols,

    "javascript": _extract_js_symbols,

    "typescript": _extract_js_symbols,

    "json": _extract_top_level_keys,

    "yaml": _extract_top_level_keys,

    "markdown": _extract_markdown_symbols,

    "powershell": _extract_powershell_symbols,

    "batch": _extract_batch_symbols,

}


def _build_symbol_lookup(records: List[SymbolRecord]) -> List[SymbolRecord]:

    return sorted(records, key=lambda item: (item[2], item[3], item[0]))


def _symbol_for_line(records: List[SymbolRecord], line_number: int) -> str:

    for symbol, _kind, start, end in records:

        if start <= line_number <= end:

            return symbol

    return FILE_SYMBOL


def _next_symbol(records: List[SymbolRecord], line_number: int) -> Optional[str]:

    for symbol, _kind, start, _end in records:

        if start >= line_number:

            return symbol

    return None


def extract_structural_symbols(language: str, text: str) -> List[SymbolRecord]:

    extractor = SYMBOL_EXTRACTORS.get(language)

    if not extractor:

        return []

    return extractor(text)


PY_SIMPLE_IMPORT = re.compile(r"^import\s+([\w\.]+)")
PY_FROM_IMPORT = re.compile(r"^from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)")
PY_CLASS_DECL_LINE = re.compile(r"^class\s+([A-Za-z_][\w]*)")
PY_DEF_DECL_LINE = re.compile(r"^(?:async\s+)?def\s+([A-Za-z_][\w]*)")
CPP_INCLUDE_RE = re.compile(r"#include\s+[<\"]([^>\"]+)[>\"]")
JS_IMPORT_FROM_RE = re.compile(r"^import\s+.*from\s+['\"](.+?)['\"]")
JS_REQUIRE_RE = re.compile(r"require\(['\"](.+?)['\"]\)")


def _add_edge(
    edges: Set[Tuple[str, str, str, Optional[str], Optional[str]]],
    path: str,
    src_symbol: str,
    relation: str,
    dst_symbol: Optional[str] = None,
    dst_path: Optional[str] = None,
) -> None:

    edges.add((path, src_symbol or FILE_SYMBOL, relation, dst_path, dst_symbol))


def _extract_python_relations(
    path: str,
    text: str,
    records: List[SymbolRecord],
    lookup: List[SymbolRecord],
) -> List[Tuple[str, str, str, Optional[str], Optional[str]]]:

    edges: Set[Tuple[str, str, str, Optional[str], Optional[str]]] = set()

    local_symbols = {symbol for symbol, _kind, *_ in records}

    lines = text.splitlines()

    class_stack: List[Tuple[int, str]] = []

    for line_number, raw in enumerate(lines, 1):

        stripped = raw.strip()

        if not stripped:

            continue

        indent = len(raw) - len(raw.lstrip(" \t"))

        while class_stack and indent <= class_stack[-1][0]:

            class_stack.pop()

        match = PY_SIMPLE_IMPORT.match(stripped)

        if match:

            _add_edge(edges, path, FILE_SYMBOL, "imports", match.group(1))

            continue

        match = PY_FROM_IMPORT.match(stripped)

        if match:

            _add_edge(edges, path, FILE_SYMBOL, "imports", match.group(1))

            continue

        class_match = PY_CLASS_DECL_LINE.match(stripped)

        if class_match:

            class_stack.append((indent, class_match.group(1)))

        def_match = PY_DEF_DECL_LINE.match(stripped)

        if def_match and class_stack and indent > class_stack[-1][0]:

            _add_edge(edges, path, class_stack[-1][1], "owns", def_match.group(1))

        current_symbol = _symbol_for_line(lookup, line_number)

        for call_match in CALL_IDENTIFIER_RE.finditer(stripped):

            name = call_match.group(1)

            if name in CALL_KEYWORDS:

                continue

            dst_path = path if name in local_symbols else None

            _add_edge(edges, path, current_symbol, "calls", name, dst_path)

        for ref_match in SELF_ATTR_RE.finditer(stripped):

            _add_edge(edges, path, current_symbol, "references", ref_match.group(1), path)

    return list(edges)


def _extract_js_relations(
    path: str,
    text: str,
    records: List[SymbolRecord],
    lookup: List[SymbolRecord],
) -> List[Tuple[str, str, str, Optional[str], Optional[str]]]:

    edges: Set[Tuple[str, str, str, Optional[str], Optional[str]]] = set()

    local_symbols = {symbol for symbol, _kind, *_ in records}

    for line_number, raw in enumerate(text.splitlines(), 1):

        stripped = raw.strip()

        if not stripped:

            continue

        match = JS_IMPORT_FROM_RE.match(stripped)

        if match:

            _add_edge(edges, path, FILE_SYMBOL, "imports", match.group(1))

        for require_match in JS_REQUIRE_RE.finditer(stripped):

            _add_edge(edges, path, FILE_SYMBOL, "imports", require_match.group(1))

        current_symbol = _symbol_for_line(lookup, line_number)

        for call_match in CALL_IDENTIFIER_RE.finditer(stripped):

            name = call_match.group(1)

            if name in CALL_KEYWORDS:

                continue

            dst_path = path if name in local_symbols else None

            _add_edge(edges, path, current_symbol, "calls", name, dst_path)

        for ref_match in THIS_ATTR_RE.finditer(stripped):

            _add_edge(edges, path, current_symbol, "references", ref_match.group(1), path)

    return list(edges)


def _extract_cpp_relations(
    path: str,
    text: str,
    records: List[SymbolRecord],
    lookup: List[SymbolRecord],
) -> List[Tuple[str, str, str, Optional[str], Optional[str]]]:

    edges: Set[Tuple[str, str, str, Optional[str], Optional[str]]] = set()

    local_symbols = {symbol for symbol, _kind, *_ in records}

    brace_level = 0

    class_stack: List[Tuple[int, str]] = []

    for line_number, raw in enumerate(text.splitlines(), 1):

        stripped = raw.strip()

        if not stripped:

            continue

        include_match = CPP_INCLUDE_RE.search(stripped)

        if include_match:

            _add_edge(edges, path, FILE_SYMBOL, "includes", include_match.group(1))

        class_match = re.match(r"^(class|struct)\s+([A-Za-z_][\w]*)", stripped)

        if class_match:

            class_stack.append((brace_level, class_match.group(2)))

        if "{" in stripped:

            brace_level += stripped.count("{")

        if "}" in stripped:

            brace_level -= stripped.count("}")

            brace_level = max(brace_level, 0)

            while class_stack and brace_level <= class_stack[-1][0]:

                class_stack.pop()

        current_symbol = _symbol_for_line(lookup, line_number)

        for call_match in CALL_IDENTIFIER_RE.finditer(stripped):

            name = call_match.group(1)

            if name in CALL_KEYWORDS:

                continue

            dst_path = path if name in local_symbols else None

            _add_edge(edges, path, current_symbol, "calls", name, dst_path)

        for ref_match in THIS_ATTR_RE.finditer(stripped):

            _add_edge(edges, path, current_symbol, "references", ref_match.group(1), path)

        func_match = re.match(

            r"^[A-Za-z_][\w:<>,\s\*&\[\]]+\s+([A-Za-z_][\w:]*)\s*\([^;]*\)\s*\{",

            stripped,

        )

        if func_match and class_stack:

            _add_edge(edges, path, class_stack[-1][1], "owns", func_match.group(1))

    return list(edges)


def extract_relations(
    language: str,
    path: str,
    text: str,
    records: List[SymbolRecord],
) -> List[Tuple[str, str, str, Optional[str], Optional[str]]]:

    lookup = _build_symbol_lookup(records)

    if language == "python":

        return _extract_python_relations(path, text, records, lookup)

    if language in {"javascript", "typescript"}:

        return _extract_js_relations(path, text, records, lookup)

    if language == "cpp":

        return _extract_cpp_relations(path, text, records, lookup)

    if language in {"json", "yaml", "markdown"}:

        return []

    return []


def extract_annotations(
    path: str,
    text: str,
    lookup: List[SymbolRecord],
) -> List[Tuple[str, Optional[str], str, int]]:

    annotations: List[Tuple[str, Optional[str], str, int]] = []

    lines = text.splitlines()

    for line_number, raw in enumerate(lines, 1):

        for tag_match in ANNOTATION_RE.finditer(raw):

            tag = tag_match.group(1).lower()

            if tag not in ANNOTATION_TAGS:

                continue

            symbol = _next_symbol(lookup, line_number)

            annotations.append((path, symbol, tag, line_number))

    return annotations


# -----------------------------
# Structural context expansion
# -----------------------------


def _fetch_symbol_rows(conn: sqlite3.Connection, symbol: str) -> List[Dict[str, object]]:
    rows = conn.execute(
        "SELECT path, symbol, kind, line_start, line_end FROM symbols WHERE symbol=?",
        (symbol,),
    ).fetchall()
    normalized: List[Dict[str, object]] = []
    seen: Set[Tuple[str, str, str, int, int]] = set()
    for row in rows:
        path = row["path"]
        name = row["symbol"]
        kind = row["kind"] or ""
        line_start = row["line_start"]
        line_end = row["line_end"]
        if not path or line_start is None or line_end is None:
            continue
        start = int(line_start)
        end = int(line_end)
        if end < start:
            start, end = end, start
        key = (path, name, kind, start, end)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "name": name,
                "kind": kind,
                "path": path,
                "line_start": start,
                "line_end": end,
            }
        )
    return normalized


def _filter_symbol_rows(
    rows: List[Dict[str, object]],
    path_filter: Optional[str],
    kind_filter: Optional[str],
) -> List[Dict[str, object]]:
    filtered = rows
    if path_filter:
        filtered = [row for row in filtered if fnmatch.fnmatch(row["path"], path_filter)]
    if kind_filter:
        filtered = [row for row in filtered if row.get("kind") == kind_filter]
    return filtered


def _classify_role(tags: List[str]) -> str:
    for candidate in ROLE_PRIORITY:
        if candidate in tags:
            return candidate
    return "unknown"


def _collect_annotations(
    conn: sqlite3.Connection,
    path: str,
    symbol: Optional[str] = None,
) -> List[str]:
    if symbol:
        rows = conn.execute(
            "SELECT DISTINCT tag FROM annotations WHERE path=? AND (symbol=? OR symbol IS NULL) ORDER BY tag",
            (path, symbol),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT tag FROM annotations WHERE path=? AND symbol IS NULL ORDER BY tag",
            (path,),
        ).fetchall()
    return [row["tag"] for row in rows if row["tag"]]


def _build_symbol_section(row: Dict[str, object], annotations: Optional[List[str]] = None) -> Dict[str, object]:
    start = int(row["line_start"])
    end = int(row["line_end"])
    span = max(0, end - start + 1)
    tags = annotations or []
    section = {
        "name": row["name"],
        "kind": row.get("kind"),
        "path": row["path"],
        "line_start": start,
        "line_end": end,
        "span": span,
        "role": _classify_role(tags),
    }
    if tags:
        section["annotations"] = tags
    return section


def _infer_parent_from_ranges(
    conn: sqlite3.Connection,
    definition: Dict[str, object],
) -> Tuple[Optional[Dict[str, object]], bool]:
    rows = conn.execute(
        "SELECT symbol, kind, line_start, line_end FROM symbols WHERE path=?",
        (definition["path"],),
    ).fetchall()
    best: Optional[Dict[str, object]] = None
    best_span: Optional[int] = None
    duplicate_span = False
    child_start = definition["line_start"]
    child_end = definition["line_end"]
    for row in rows:
        name = row["symbol"]
        kind = row["kind"] or ""
        start = row["line_start"]
        end = row["line_end"]
        if start is None or end is None:
            continue
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        if start_i > child_start or end_i < child_end:
            continue
        if (
            name == definition["name"]
            and start_i == child_start
            and end_i == child_end
        ):
            continue
        span = end_i - start_i
        if best_span is None or span < best_span:
            best_span = span
            best = {
                "name": name,
                "kind": kind,
                "path": definition["path"],
                "line_start": start_i,
                "line_end": end_i,
            }
            duplicate_span = False
        elif span == best_span and best is not None:
            duplicate_span = True
    return best, duplicate_span


def _resolve_parent_container(
    conn: sqlite3.Connection,
    definition: Dict[str, object],
) -> Dict[str, object]:
    info: Dict[str, object] = {
        "status": "missing",
        "container": None,
        "warnings": [],
        "ownership_inferred": False,
    }
    owner_rows = conn.execute(
        "SELECT src_symbol FROM edges WHERE relation='owns' AND dst_symbol=? AND src_path=?",
        (definition["name"], definition["path"]),
    ).fetchall()
    owner_names = sorted({row["src_symbol"] for row in owner_rows if row["src_symbol"]})
    if owner_names:
        if len(owner_names) > 1:
            info["status"] = "ambiguous"
            info["diagnostics"] = {"owners": owner_names}
            return info
        parent_rows = [row for row in _fetch_symbol_rows(conn, owner_names[0]) if row["path"] == definition["path"]]
        if not parent_rows:
            info["status"] = "partial"
            info["warnings"].append("parent definition missing; falling back to child span")
            info["container"] = {
                "name": owner_names[0],
                "kind": "unknown",
                "path": definition["path"],
                "line_start": definition["line_start"],
                "line_end": definition["line_end"],
            }
            return info
        if len(parent_rows) > 1:
            info["status"] = "ambiguous"
            info["diagnostics"] = {"owners": owner_names}
            return info
        info["status"] = "ok"
        info["container"] = parent_rows[0]
        return info

    inferred, duplicate_span = _infer_parent_from_ranges(conn, definition)
    if inferred:
        info["status"] = "ok"
        info["ownership_inferred"] = True
        if duplicate_span:
            info["warnings"].append("multiple enclosing spans detected; chose smallest envelope")
        info["container"] = inferred
        return info

    return info


def _read_symbol_body(
    section: Dict[str, object],
    *,
    allow_truncate: bool,
) -> Tuple[Optional[str], bool, Dict[str, object]]:
    path = section["path"]
    start = int(section["line_start"])
    end = int(section["line_end"])
    total_lines = max(0, end - start + 1)
    try:
        with open(path, "r", errors="ignore") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return None, False, {"status": "io_error", "message": str(exc)}

    if not lines:
        return "", False, {"status": "ok", "line_count": 0, "total_lines": 0}

    max_index = len(lines)
    start_idx = max(1, start)
    end_idx = min(max_index, end)
    if end_idx < start_idx:
        end_idx = start_idx
    snippet = lines[start_idx - 1 : end_idx]
    truncated = False
    if len(snippet) > CONTEXT_LINE_CAP:
        if not allow_truncate:
            return None, False, {
                "status": "truncation_required",
                "line_count": len(snippet),
                "cap": CONTEXT_LINE_CAP,
            }
        truncated = True
        snippet = snippet[:CONTEXT_LINE_CAP]
    text = "".join(snippet).rstrip("\n")
    return text, truncated, {
        "status": "ok",
        "line_count": len(snippet),
        "total_lines": total_lines,
        "cap": CONTEXT_LINE_CAP,
    }


def _collect_parent_outline(
    conn: sqlite3.Connection,
    container: Dict[str, object],
    *,
    limit: int = 64,
) -> List[Dict[str, object]]:
    rows = conn.execute(
        "SELECT dst_symbol FROM edges WHERE relation='owns' AND src_symbol=? AND src_path=? ORDER BY dst_symbol LIMIT ?",
        (container["name"], container["path"], limit),
    ).fetchall()
    outline: List[Dict[str, object]] = []
    seen: Set[str] = set()
    for row in rows:
        child = row["dst_symbol"]
        if not child or child in seen:
            continue
        seen.add(child)
        child_rows = [entry for entry in _fetch_symbol_rows(conn, child) if entry["path"] == container["path"]]
        if child_rows:
            outline.append(_build_symbol_section(child_rows[0]))
        else:
            outline.append({"name": child, "kind": "unknown"})
    return outline


def _summarize_file_context(conn: sqlite3.Connection, path: str) -> Dict[str, object]:
    fan_in = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE dst_path=?",
        (path,),
    ).fetchone()[0]
    fan_out = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src_path=?",
        (path,),
    ).fetchone()[0]
    deps = conn.execute(
        """
        SELECT DISTINCT dst_path
        FROM edges
        WHERE src_path=? AND dst_path IS NOT NULL AND dst_path != ?
        ORDER BY dst_path
        LIMIT ?
        """,
        (path, path, FILE_DEP_SUMMARY_LIMIT),
    ).fetchall()
    dependencies = [row["dst_path"] for row in deps if row["dst_path"]]
    annotations = _collect_annotations(conn, path, symbol=None)
    return {
        "path": path,
        "fan_in": fan_in,
        "fan_out": fan_out,
        "dependencies": dependencies,
        "annotations": annotations,
        "role": _classify_role(annotations),
    }


def resolve_symbol_context(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    level: int = CONTEXT_DEFAULT_LEVEL,
    path_filter: Optional[str] = None,
    kind_filter: Optional[str] = None,
    allow_truncate: bool = False,
) -> Dict[str, object]:
    requested_level = level if level in CONTEXT_ALLOWED_LEVELS else CONTEXT_DEFAULT_LEVEL
    result: Dict[str, object] = {
        "symbol": symbol,
        "level": requested_level,
        "status": "ok",
        "sections": {},
    }
    diagnostics: Dict[str, object] = {}
    warnings: List[str] = []

    rows = _fetch_symbol_rows(conn, symbol)
    if not rows:
        result["status"] = "not_found"
        diagnostics["reason"] = "symbol not indexed in structural tables"
        result["diagnostics"] = diagnostics
        return result

    filtered = _filter_symbol_rows(rows, path_filter, kind_filter)
    if filtered:
        rows = filtered
    elif path_filter or kind_filter:
        result["status"] = "not_found"
        diagnostics["reason"] = "no definitions matched provided filters"
        diagnostics["candidates"] = rows
        result["diagnostics"] = diagnostics
        return result

    if len(rows) > 1:
        result["status"] = "ambiguous"
        diagnostics["reason"] = "multiple definitions detected"
        diagnostics["candidates"] = rows
        result["diagnostics"] = diagnostics
        return result

    definition = rows[0]
    symbol_annotations = _collect_annotations(conn, definition["path"], definition["name"])
    symbol_section = _build_symbol_section(definition, annotations=symbol_annotations)
    result["sections"]["symbol"] = symbol_section

    parent_info = _resolve_parent_container(conn, definition)
    if parent_info.get("status") == "ambiguous":
        result["status"] = "ambiguous"
        diagnostics["reason"] = "ambiguous parent containment"
        diagnostics.update(parent_info.get("diagnostics", {}))
        result["diagnostics"] = diagnostics
        return result

    if parent_info.get("warnings"):
        warnings.extend(parent_info["warnings"])

    container = parent_info.get("container")
    if container:
        container_annotations = _collect_annotations(conn, container["path"], container["name"])
        container_section = _build_symbol_section(container, annotations=container_annotations)
        container_section["ownership_inferred"] = parent_info.get("ownership_inferred", False)
        result["sections"]["container"] = container_section

    if requested_level >= 1:
        context_text, truncated, context_meta = _read_symbol_body(symbol_section, allow_truncate=allow_truncate)
        if context_meta.get("status") != "ok":
            result["status"] = context_meta.get("status", "error")
            diagnostics["context_block"] = context_meta
            if diagnostics:
                result["diagnostics"] = diagnostics
            if warnings:
                result["warnings"] = warnings
            return result
        result["sections"]["context_block"] = {
            "text": context_text or "",
            "truncated": truncated,
            "line_count": context_meta.get("line_count", 0),
            "total_lines": context_meta.get("total_lines", context_meta.get("line_count", 0)),
        }
        if truncated:
            warnings.append(
                f"context truncated to {CONTEXT_LINE_CAP} line(s); original span {context_meta.get('total_lines')}"
            )

    if requested_level >= 2:
        if container:
            outline = _collect_parent_outline(conn, container)
            if outline:
                result["sections"]["parent_outline"] = outline
            else:
                warnings.append("parent outline unavailable for requested level")
        else:
            warnings.append("no parent container available for level 2 outline")

    if requested_level >= 3:
        result["sections"]["file_summary"] = _summarize_file_context(conn, definition["path"])

    if diagnostics:
        result["diagnostics"] = diagnostics
    if warnings:
        result["warnings"] = warnings

    return result


def clear_structural_records(conn: sqlite3.Connection, path: str) -> None:

    if not structural_ready(conn):

        return

    conn.execute("DELETE FROM symbols WHERE path=?", (path,))

    conn.execute("DELETE FROM edges WHERE src_path=? OR dst_path=?", (path, path))

    conn.execute("DELETE FROM annotations WHERE path=?", (path,))


def update_structural_data(conn: sqlite3.Connection, path: str, text: str) -> None:

    if not structural_ready(conn):

        return

    clear_structural_records(conn, path)

    language = classify_language(path)

    if not language:

        return

    line_count = text.count("\n") + 1

    if line_count > MAX_STRUCTURAL_LINES:

        print(

            f"[struct] skip {path} - {line_count} LOC exceeds {MAX_STRUCTURAL_LINES}",

            file=sys.stderr,

        )

        return

    records = extract_structural_symbols(language, text)

    if not records:

        return

    lookup = _build_symbol_lookup(records)

    conn.executemany(

        "INSERT INTO symbols(path, symbol, kind, line_start, line_end) VALUES(?, ?, ?, ?, ?)",

        [(path, symbol, kind, line_start, line_end) for symbol, kind, line_start, line_end in records],

    )

    edges = extract_relations(language, path, text, records)

    if edges:

        conn.executemany(

            "INSERT INTO edges(src_path, src_symbol, relation, dst_path, dst_symbol) VALUES(?, ?, ?, ?, ?)",

            edges,

        )

    annotations = extract_annotations(path, text, lookup)

    if annotations:

        conn.executemany(

            "INSERT INTO annotations(path, symbol, tag, line) VALUES(?, ?, ?, ?)",

            annotations,

        )


def benchmark_structural(root: str = ".", *, json_mode: bool = False):

    measurements: List[Dict[str, object]] = []

    processed = 0

    skipped = 0

    total_ms = 0.0

    slowest: Optional[Dict[str, object]] = None

    for path in file_iter(root):

        language = classify_language(path)

        if not language:

            continue

        try:

            with open(path, "r", errors="ignore") as handle:

                text = handle.read()

        except Exception as exc:

            print(f"[benchmark] skip {path}: {exc}", file=sys.stderr)

            continue

        line_count = text.count("\n") + 1

        skip = line_count > MAX_STRUCTURAL_LINES

        start = time.perf_counter()

        symbol_count = 0

        if not skip:

            symbol_count = len(extract_structural_symbols(language, text))

        duration_ms = (time.perf_counter() - start) * 1000.0

        measurements.append(

            {

                "path": path,

                "language": language,

                "lines": line_count,

                "symbols": symbol_count,

                "duration_ms": round(duration_ms, 3),

                "skipped": skip,

            }

        )

        if skip:

            skipped += 1

            continue

        processed += 1

        total_ms += duration_ms

        if slowest is None or duration_ms > slowest["duration_ms"]:  # type: ignore[index]

            slowest = {

                "path": path,

                "duration_ms": round(duration_ms, 3),

            }

    avg_ms = round(total_ms / processed, 3) if processed else 0.0

    summary: Dict[str, object] = {

        "structural_files": processed + skipped,

        "processed": processed,

        "skipped": skipped,

        "avg_ms": avg_ms,

        "slowest_path": slowest["path"] if slowest else None,

        "slowest_ms": slowest["duration_ms"] if slowest else 0.0,

    }

    if json_mode:

        print(json.dumps({"summary": summary, "files": measurements}, indent=2))

    else:

        print(

            f"[benchmark] structural files: {summary['structural_files']} (processed {processed}, skipped {skipped})"

        )

        print(

            f"[benchmark] avg {avg_ms}ms, slowest {summary['slowest_ms']}ms at {summary['slowest_path']}"

        )

    return summary, measurements


# -----------------------------
# Structural queries
# -----------------------------


def structural_available(conn: sqlite3.Connection) -> bool:

    if structural_ready(conn):

        return True

    print("[struct] structural tables unavailable; run migrations and index", file=sys.stderr)

    return False


def find_symbol_records(

    conn: sqlite3.Connection,

    query: str,

    *,

    limit: int = 50,

) -> List[Dict[str, object]]:

    if not structural_available(conn):

        return []

    like = f"%{query}%"

    rows = conn.execute(

        """

        SELECT path, symbol, kind, line_start, line_end

        FROM symbols

        WHERE symbol LIKE ?

        ORDER BY symbol, path

        LIMIT ?

        """,

        (like, limit),

    ).fetchall()

    return [dict(row) for row in rows]


def callers_for_symbol(

    conn: sqlite3.Connection,

    symbol: str,

    *,

    limit: int = 50,

) -> List[Dict[str, object]]:

    if not structural_available(conn):

        return []

    rows = conn.execute(

        """

        SELECT src_path, src_symbol, relation

        FROM edges

        WHERE relation='calls' AND dst_symbol=?

        ORDER BY src_symbol, src_path

        LIMIT ?

        """,

        (symbol, limit),

    ).fetchall()

    return [dict(row) for row in rows]


def dependencies_for_symbol(

    conn: sqlite3.Connection,

    symbol: str,

    *,

    limit: int = 50,

) -> List[Dict[str, object]]:

    if not structural_available(conn):

        return []

    rows = conn.execute(

        """

        SELECT dst_symbol, dst_path, relation

        FROM edges

        WHERE src_symbol=?

            AND relation IN ('calls', 'imports', 'includes', 'references')

        ORDER BY relation, dst_symbol

        LIMIT ?

        """,

        (symbol, limit),

    ).fetchall()

    return [dict(row) for row in rows]


def explain_symbol_details(

    conn: sqlite3.Connection,

    symbol: str,

    *,

    limit: int = 50,

) -> Dict[str, object]:

    if not structural_available(conn):

        return {}

    defs = conn.execute(

        "SELECT path, symbol, kind, line_start, line_end FROM symbols WHERE symbol=? ORDER BY path LIMIT ?",

        (symbol, limit),

    ).fetchall()

    callers = callers_for_symbol(conn, symbol, limit=limit)

    deps = dependencies_for_symbol(conn, symbol, limit=limit)

    annotations = conn.execute(

        "SELECT path, symbol, tag, line FROM annotations WHERE symbol=? OR symbol IS NULL ORDER BY path, line",

        (symbol,),

    ).fetchall()

    return {

        "symbol": symbol,

        "definitions": [dict(row) for row in defs],

        "callers": callers,

        "dependencies": deps,

        "annotations": [dict(row) for row in annotations],

    }


def _extract_symbol_from_query(query: str) -> Optional[str]:

    tokens = re.findall(r"[A-Za-z_][\w\.]+", query)

    for token in reversed(tokens):

        if token.lower() not in INTENT_STOP_WORDS:

            return token

    return tokens[-1] if tokens else None


CALLER_KEYWORDS = (

    "who calls",

    "callers of",

    "callers",

    "call tree",

    "incoming references",

    "incoming reference",

    "incoming call",

    "incoming calls",

    "referenced by",

    "who invokes",

    "trace to",

)


DEPS_KEYWORDS = (

    "depends on",

    "depend on",

    "dependencies",

    "what uses",

    "uses of",

    "imports for",

    "what imports",

    "what imports does",

    "downstream",

    "fan-out",

    "fanout",

    "fan out",

    "deps",

    "outgoing",

    "usage",

)


EXPLAIN_KEYWORDS = (

    "explain",

    "why does",

    "annotations for",

    "describe",

    "tell me about",

    "give context",

    "annotate",

    "explain why",

)


FIND_KEYWORDS = (

    "where is",

    "find symbol",

    "definition of",

    "symbol:",

    "locate",

    "find ",

)


def _contains_phrase(text: str, phrases: Tuple[str, ...]) -> bool:

    return any(phrase in text for phrase in phrases)


def classify_intent(query: str) -> Dict[str, Optional[str]]:

    lowered = query.strip().lower()

    explain_hit = _contains_phrase(lowered, EXPLAIN_KEYWORDS) or lowered.startswith("what is")

    deps_combo = (

        ("what does" in lowered or "what imports does" in lowered)

        and (" use" in lowered or " uses" in lowered)

    )

    if _contains_phrase(lowered, FIND_KEYWORDS):

        intent = "find_symbol"

    elif explain_hit:

        intent = "explain"

    elif _contains_phrase(lowered, CALLER_KEYWORDS):

        intent = "callers"

    elif _contains_phrase(lowered, DEPS_KEYWORDS) or deps_combo:

        intent = "deps"

    else:

        intent = "find_symbol"

    symbol = _extract_symbol_from_query(query)

    if intent == "find_symbol" and not symbol:

        stripped = query.strip()

        symbol = stripped or None

    return {"intent": intent, "symbol": symbol, "raw": query}


def handle_structural_intent(

    conn: sqlite3.Connection,

    query: str,

    *,

    limit: int = 50,

) -> Dict[str, object]:

    info = classify_intent(query)

    intent = info["intent"] or "find_symbol"

    symbol = info.get("symbol")

    response: Dict[str, object] = {"intent": intent, "symbol": symbol, "raw": query}

    if intent == "find_symbol":

        needle = (symbol or query).strip()

        if not needle:

            response["results"] = []

            return response

        response["results"] = find_symbol_records(conn, needle, limit=limit)

        return response

    if not symbol:

        print("[struct] unable to detect symbol in query", file=sys.stderr)

        response["results"] = []

        return response

    if intent == "callers":

        response["results"] = callers_for_symbol(conn, symbol, limit=limit)

    elif intent == "deps":

        response["results"] = dependencies_for_symbol(conn, symbol, limit=limit)

    elif intent == "explain":

        response["details"] = explain_symbol_details(conn, symbol, limit=limit)

    else:

        response["results"] = find_symbol_records(conn, symbol, limit=limit)

    return response


ASK_DISPLAY_LIMIT = 10


def _format_path_segment(path: Optional[str], start: Optional[int], end: Optional[int]) -> Optional[str]:

    if not path:

        return None

    if start and end and end != start:

        return f"{path}:{start}-{end}"

    if start:

        return f"{path}:{start}"

    return path


def _definition_snippet(

    path: Optional[str],

    line_start: Optional[int],

    line_end: Optional[int],

    context: int,

) -> Optional[str]:

    if context <= 0 or not path or line_start is None:

        return None

    try:

        with open(path, "r", errors="ignore") as handle:

            lines = handle.readlines()

    except OSError:

        return None

    total = len(lines)

    start = max(1, int(line_start) - context)

    effective_end = line_end if isinstance(line_end, int) else line_start

    if effective_end is None:

        effective_end = line_start

    end = min(total, int(effective_end) + context)

    snippet = "".join(lines[start - 1 : end])

    return snippet.rstrip("\n") if snippet else None


def _print_find_results(results: List[Dict[str, object]], limit: int = ASK_DISPLAY_LIMIT) -> None:

    if not results:

        print("  (no matching symbols)")

        return

    for entry in results[:limit]:

        path = entry.get("path") if isinstance(entry, dict) else None

        line_start = entry.get("line_start") if isinstance(entry, dict) else None

        line_end = entry.get("line_end") if isinstance(entry, dict) else None

        symbol = entry.get("symbol") if isinstance(entry, dict) else None

        kind = entry.get("kind") if isinstance(entry, dict) else None

        location = _format_path_segment(path, line_start, line_end)

        parts = [p for p in (location, symbol, f"[{kind}]" if kind else None) if p]

        print("  - " + " ".join(parts))

    if len(results) > limit:

        print(f"  ... {len(results) - limit} more symbol(s)")


def _print_caller_results(symbol: Optional[str], results: List[Dict[str, object]], limit: int = ASK_DISPLAY_LIMIT) -> None:

    if not results:

        print("  (no callers)")

        return

    for entry in results[:limit]:

        caller = entry.get("src_symbol") if isinstance(entry, dict) else None

        path = entry.get("src_path") if isinstance(entry, dict) else None

        relation = entry.get("relation") if isinstance(entry, dict) else "calls"

        line = f"  - {caller or '(unknown)'} --{relation}--> {symbol or '(target)'}"

        if path:

            line += f" ({path})"

        print(line)

    if len(results) > limit:

        print(f"  ... {len(results) - limit} more caller(s)")


def _print_dependency_results(symbol: Optional[str], results: List[Dict[str, object]], limit: int = ASK_DISPLAY_LIMIT) -> None:

    if not results:

        print("  (no dependencies)")

        return

    for entry in results[:limit]:

        target = entry.get("dst_symbol") if isinstance(entry, dict) else None

        path = entry.get("dst_path") if isinstance(entry, dict) else None

        relation = entry.get("relation") if isinstance(entry, dict) else "relation"

        line = f"  - {symbol or '(source)'} --{relation}--> {target or '(unknown)'}"

        if path:

            line += f" ({path})"

        print(line)

    if len(results) > limit:

        print(f"  ... {len(results) - limit} more dependency record(s)")


def _print_explain_details(

    details: Dict[str, object],

    limit: int = ASK_DISPLAY_LIMIT,

    context: int = 0,

) -> None:

    definitions = details.get("definitions") if isinstance(details, dict) else None

    print("  definitions:")

    if isinstance(definitions, list):

        for entry in definitions[:limit]:

            path = entry.get("path") if isinstance(entry, dict) else None

            symbol = entry.get("symbol") if isinstance(entry, dict) else None

            kind = entry.get("kind") if isinstance(entry, dict) else None

            line_start = entry.get("line_start") if isinstance(entry, dict) else None

            line_end = entry.get("line_end") if isinstance(entry, dict) else None

            location = _format_path_segment(path, line_start, line_end)

            parts = [p for p in (location, symbol, f"[{kind}]" if kind else None) if p]

            print("    - " + " ".join(parts))

            if context > 0:

                snippet = _definition_snippet(path, line_start, line_end, context)

                if snippet:

                    for line in snippet.splitlines():

                        print("      " + line.rstrip("\n"))

        if len(definitions) > limit:

            print(f"    ... {len(definitions) - limit} more definition(s)")

    else:

        print("    (none)")

    callers = details.get("callers") if isinstance(details, dict) else None

    deps = details.get("dependencies") if isinstance(details, dict) else None

    annotations = details.get("annotations") if isinstance(details, dict) else None

    print(f"  callers: {len(callers or [])}")

    print(f"  dependencies: {len(deps or [])}")

    print(f"  annotations: {len(annotations or [])}")


def render_structural_cli_response(response: Dict[str, object]) -> None:

    intent = response.get("intent")

    symbol = response.get("symbol")

    print(f"[ask] intent={intent} symbol={symbol}")

    results = response.get("results")

    if isinstance(results, list):

        if intent == "find_symbol":

            _print_find_results(results)

        elif intent == "callers":

            _print_caller_results(symbol, results)

        elif intent == "deps":

            _print_dependency_results(symbol, results)

        else:

            _print_find_results(results)

    elif results is not None:

        print("  (unrecognized results payload)")

    details = response.get("details")

    if isinstance(details, dict):

        _print_explain_details(details)


def find_symbol(symbol: str, *, limit: int = 50) -> List[Dict[str, object]]:

    conn = connect_db()

    try:

        return find_symbol_records(conn, symbol, limit=limit)

    finally:

        conn.close()


def get_callers(symbol: str, *, limit: int = 50) -> List[Dict[str, object]]:

    conn = connect_db()

    try:

        return callers_for_symbol(conn, symbol, limit=limit)

    finally:

        conn.close()


def get_dependencies(symbol: str, *, limit: int = 50) -> List[Dict[str, object]]:

    conn = connect_db()

    try:

        return dependencies_for_symbol(conn, symbol, limit=limit)

    finally:

        conn.close()


def explain_symbol(symbol: str, *, limit: int = 50) -> Dict[str, object]:

    conn = connect_db()

    try:

        return explain_symbol_details(conn, symbol, limit=limit)

    finally:

        conn.close()


def _collect_paths_from_results(

    entries: List[Dict[str, object]], keys: Tuple[str, ...]

) -> List[str]:

    ordered: List[str] = []

    seen: Set[str] = set()

    for entry in entries:

        if not isinstance(entry, dict):

            continue

        for key in keys:

            value = entry.get(key)

            if isinstance(value, str) and value and value not in seen:

                seen.add(value)

                ordered.append(value)

    return ordered


def _print_find_cli(symbol: str, rows: List[Dict[str, object]], context: int) -> None:

    if not rows:

        print(f"[find] no matches for '{symbol}'")

        return

    print(f"[find] {len(rows)} match(es) for '{symbol}'")

    for entry in rows:

        path = entry.get("path") if isinstance(entry, dict) else None

        entry_symbol = entry.get("symbol") if isinstance(entry, dict) else None

        kind = entry.get("kind") if isinstance(entry, dict) else None

        line_start = entry.get("line_start") if isinstance(entry, dict) else None

        line_end = entry.get("line_end") if isinstance(entry, dict) else None

        location = _format_path_segment(path, line_start, line_end)

        parts = [p for p in (location, entry_symbol, f"[{kind}]" if kind else None) if p]

        print("  - " + " ".join(parts))

        if context > 0:

            snippet = _definition_snippet(path, line_start, line_end, context)

            if snippet:

                for line in snippet.splitlines():

                    print("    " + line.rstrip("\n"))


def _print_callers_cli(symbol: str, rows: List[Dict[str, object]]) -> None:

    if not rows:

        print(f"[callers] no callers found for '{symbol}'")

        return

    print(f"[callers] {len(rows)} caller(s) for '{symbol}'")

    for entry in rows:

        caller = entry.get("src_symbol") if isinstance(entry, dict) else None

        path = entry.get("src_path") if isinstance(entry, dict) else None

        relation = entry.get("relation") if isinstance(entry, dict) else "calls"

        line = f"  - {caller or '(unknown)'} --{relation}--> {symbol}"

        if path:

            line += f" ({path})"

        print(line)


def _print_dependencies_cli(symbol: str, rows: List[Dict[str, object]]) -> None:

    if not rows:

        print(f"[deps] no dependencies recorded for '{symbol}'")

        return

    print(f"[deps] {len(rows)} relation(s) for '{symbol}'")

    for entry in rows:

        target = entry.get("dst_symbol") if isinstance(entry, dict) else None

        path = entry.get("dst_path") if isinstance(entry, dict) else None

        relation = entry.get("relation") if isinstance(entry, dict) else "relation"

        line = f"  - {symbol} --{relation}--> {target or '(unknown)'}"

        if path:

            line += f" ({path})"

        print(line)


def _print_explain_cli(symbol: str, details: Dict[str, object], context: int) -> None:

    if not details:

        print(f"[explain] no structural record for '{symbol}'")

        return

    print(f"[explain] {symbol}")

    _print_explain_details(details, context=context)


def _parse_structural_cli_args(

    args: List[str],

) -> Tuple[Optional[str], bool, bool, int, int, Optional[str]]:

    json_flag = False

    paths_only = False

    context = 0

    limit = 50

    symbol_parts: List[str] = []

    i = 0

    while i < len(args):

        arg = args[i]

        if arg == "--json":

            json_flag = True

        elif arg == "--paths":

            paths_only = True

        elif arg == "--context":

            if i + 1 >= len(args):

                return None, False, False, 0, 50, "Invalid --context usage"

            try:

                context = max(0, int(args[i + 1]))

            except ValueError:

                return None, False, False, 0, 50, "Invalid --context value"

            i += 1

        elif arg == "--limit":

            if i + 1 >= len(args):

                return None, False, False, 0, 50, "Invalid --limit usage"

            try:

                limit = max(1, int(args[i + 1]))

            except ValueError:

                return None, False, False, 0, 50, "Invalid --limit value"

            i += 1

        else:

            symbol_parts.append(arg)

        i += 1

    if paths_only and json_flag:

        print("[warn] --json ignored when --paths is set", file=sys.stderr)

        json_flag = False

    symbol = " ".join(symbol_parts).strip()

    return symbol, json_flag, paths_only, context, limit, None


def _parse_context_cli_args(
    args: List[str],
) -> Tuple[Optional[str], Dict[str, object], Optional[str]]:
    options: Dict[str, object] = {
        "json": False,
        "level": CONTEXT_DEFAULT_LEVEL,
        "path_filter": None,
        "kind_filter": None,
        "allow_truncate": False,
    }
    symbol_parts: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            options["json"] = True
        elif arg == "--level":
            if i + 1 >= len(args):
                return None, options, "Invalid --level usage"
            try:
                level_value = int(args[i + 1])
            except ValueError:
                return None, options, "Invalid --level value"
            options["level"] = level_value if level_value in CONTEXT_ALLOWED_LEVELS else CONTEXT_DEFAULT_LEVEL
            i += 1
        elif arg == "--path":
            if i + 1 >= len(args):
                return None, options, "Invalid --path usage"
            options["path_filter"] = args[i + 1]
            i += 1
        elif arg == "--kind":
            if i + 1 >= len(args):
                return None, options, "Invalid --kind usage"
            options["kind_filter"] = args[i + 1]
            i += 1
        elif arg == "--allow-truncate":
            options["allow_truncate"] = True
        else:
            symbol_parts.append(arg)
        i += 1

    symbol = " ".join(symbol_parts).strip()
    return symbol, options, None


def _handle_context_cli(conn: sqlite3.Connection, argv: List[str]) -> None:
    symbol, options, error = _parse_context_cli_args(argv)
    if error:
        print(error, file=sys.stderr)
        return
    if not symbol:
        print(
            "Usage: python quedonde.py context [--json] [--level N] [--path GLOB] [--kind KIND] [--allow-truncate] <symbol>",
            file=sys.stderr,
        )
        return

    response = resolve_symbol_context(
        conn,
        symbol,
        level=int(options["level"]),
        path_filter=options.get("path_filter"),
        kind_filter=options.get("kind_filter"),
        allow_truncate=bool(options.get("allow_truncate")),
    )

    try:
        record_context_event(symbol, response.get("level", options["level"]), response.get("status", "error"))
    except Exception:
        pass

    if options.get("json"):
        print(json.dumps(response, indent=2))
        return

    render_context_cli(response)


def _format_line_range(section: Dict[str, object]) -> Optional[str]:
    start = section.get("line_start")
    end = section.get("line_end")
    if start is None:
        return None
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def _print_context_metadata(title: str, section: Dict[str, object]) -> None:
    print(title)
    print(f"- name: {section.get('name')}")
    if section.get("kind"):
        print(f"- kind: {section.get('kind')}")
    if section.get("role"):
        print(f"- role: {section.get('role')}")
    if section.get("path"):
        print(f"- path: {section.get('path')}")
    line_segment = _format_line_range(section)
    if line_segment:
        print(f"- lines: {line_segment}")
    if section.get("span"):
        print(f"- span: {section.get('span')} line(s)")
    if section.get("ownership_inferred"):
        print("- ownership_inferred: True")
    annotations = section.get("annotations")
    if isinstance(annotations, list) and annotations:
        print(f"- annotations: {', '.join(annotations)}")


def _print_context_block(block: Dict[str, object]) -> None:
    print("CONTEXT BLOCK")
    if block.get("truncated"):
        print("- truncated: True")
    print("- lines: {}".format(block.get("line_count", 0)))
    print("```")
    print(block.get("text", ""))
    print("```")


def _print_file_summary(summary: Dict[str, object]) -> None:
    print("FILE SUMMARY")
    print(f"- path: {summary.get('path')}")
    print(f"- role: {summary.get('role')}")
    print(f"- fan_in: {summary.get('fan_in')}")
    print(f"- fan_out: {summary.get('fan_out')}")
    dependencies = summary.get("dependencies") or []
    if dependencies:
        print(f"- dependencies ({len(dependencies)}):")
        for dep in dependencies:
            print(f"  - {dep}")
    annotations = summary.get("annotations") or []
    if annotations:
        print(f"- annotations: {', '.join(annotations)}")


def render_context_cli(payload: Dict[str, object]) -> None:
    print(
        f"[context] symbol={payload.get('symbol')} level={payload.get('level')} status={payload.get('status')}"
    )
    sections = payload.get("sections") if isinstance(payload, dict) else None
    if isinstance(sections, dict):
        symbol_section = sections.get("symbol")
        if isinstance(symbol_section, dict):
            _print_context_metadata("SYMBOL", symbol_section)
        container_section = sections.get("container")
        if isinstance(container_section, dict):
            _print_context_metadata("CONTAINER", container_section)
        context_block = sections.get("context_block")
        if isinstance(context_block, dict):
            _print_context_block(context_block)
        parent_outline = sections.get("parent_outline")
        if isinstance(parent_outline, list):
            print("PARENT OUTLINE")
            for entry in parent_outline:
                name = entry.get("name") if isinstance(entry, dict) else entry
                kind = entry.get("kind") if isinstance(entry, dict) else None
                if kind:
                    print(f"- {name} [{kind}]")
                else:
                    print(f"- {name}")
        file_summary = sections.get("file_summary")
        if isinstance(file_summary, dict):
            _print_file_summary(file_summary)

    warnings = payload.get("warnings") if isinstance(payload, dict) else None
    if isinstance(warnings, list) and warnings:
        print("WARNINGS")
        for note in warnings:
            print(f"- {note}")

    diagnostics = payload.get("diagnostics") if isinstance(payload, dict) else None
    if diagnostics:
        print("DIAGNOSTICS")
        print(json.dumps(diagnostics, indent=2))

# -----------------------------
# Diagnostics
# -----------------------------


def get_table_counts(conn: sqlite3.Connection) -> Dict[str, object]:

    rows = conn.execute(

        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"

    ).fetchall()

    counts: Dict[str, object] = {}

    for (table_name,) in rows:

        try:

            counts[table_name] = conn.execute(

                f"SELECT COUNT(*) FROM '{table_name}'"

            ).fetchone()[0]

        except sqlite3.DatabaseError as exc:

            counts[table_name] = f"error: {exc}"

    return counts


def diagnose(conn: sqlite3.Connection, *, json_mode: bool = False) -> Dict[str, object]:

    ensure_migration_tracking(conn)

    result = {

        "database": os.path.abspath(DB),

        "cache_present": os.path.exists(CACHE),

        "structural_version": get_structural_version(conn),

        "tables": get_table_counts(conn),

        "pending_migrations": run_migrations(conn, dry_run=True),

    }

    if json_mode:

        print(json.dumps(result, indent=2))

    else:

        print(f"database: {result['database']}")

        print(f"cache_present: {result['cache_present']}")

        print(f"structural_version: {result['structural_version']}")

        if result["pending_migrations"]:

            print("pending_migrations:")

            for name in result["pending_migrations"]:

                print(f"  - {name}")

        else:

            print("pending_migrations: []")

        print("table_counts:")

        for table, count in result["tables"].items():

            print(f"  - {table}: {count}")

    return result



def file_iter(root="."):

    for d, dirs, files in os.walk(root):

        dirs[:] = [dd for dd in dirs if dd not in IGNORES]

        for f in files:

            if f.endswith(EXTS):

                yield os.path.join(d, f)



def index_repo(conn, root="."):

    cur = conn.cursor()

    print("[index] scanning repository...")



    added = updated = removed = 0

    seen_paths = set()



    for path in file_iter(root):

        seen_paths.add(path)

        try:

            mtime = os.path.getmtime(path)

            row = cur.execute("SELECT mtime FROM meta WHERE path=?", (path,)).fetchone()

            if row and abs(row[0] - mtime) < 1e-6:

                continue

            with open(path, "r", errors="ignore") as fh:

                text = fh.read()

            if row:

                cur.execute("DELETE FROM files WHERE path=?", (path,))

                updated += 1

            else:

                added += 1

            cur.execute("INSERT INTO files(path, content) VALUES(?, ?)", (path, text))

            cur.execute("REPLACE INTO meta(path, mtime) VALUES(?, ?)", (path, mtime))

            update_structural_data(conn, path, text)

        except Exception as e:

            print(f"[skip] {path}: {e}")



    # Remove DB entries for files that disappeared since the last run.

    for (indexed_path,) in cur.execute("SELECT path FROM meta"):

        if indexed_path not in seen_paths and not os.path.exists(indexed_path):

            cur.execute("DELETE FROM files WHERE path=?", (indexed_path,))

            cur.execute("DELETE FROM meta WHERE path=?", (indexed_path,))

            removed += 1

            clear_structural_records(conn, indexed_path)



    conn.commit()

    print(f"[done] added {added}, updated {updated}, removed {removed}")



    if added or updated or removed:

        try:

            cur.execute("INSERT INTO files(files) VALUES('optimize')")

            conn.commit()

            print("[index] optimized FTS index")

        except sqlite3.Error as e:

            print(f"[warn] optimize failed: {e}")

        try:

            conn.execute("VACUUM")

            print("[index] vacuumed database")

        except sqlite3.Error as e:

            print(f"[warn] vacuum failed: {e}")



    clear_cache()



# -----------------------------


# Cache

# -----------------------------



def clear_cache() -> None:

    if os.path.exists(CACHE):

        os.remove(CACHE)

        print("[cache] cleared")


def cache_key(pattern, mode, fuzzy, context, json_mode, paths_only, lines_flag, title_filter):

    data = f"{pattern}:{mode}:{fuzzy}:{context}:{json_mode}:{paths_only}:{lines_flag}:{title_filter or ''}"

    return hashlib.md5(data.encode()).hexdigest()



def load_cache(expected_structural_version: Optional[str] = None):

    if not os.path.exists(CACHE):

        return {}

    try:

        with open(CACHE, "rb") as fh:

            cache_obj = pickle.load(fh)

        if expected_structural_version is None:

            return cache_obj

        meta = cache_obj.get(CACHE_META_KEY) if isinstance(cache_obj, dict) else None

        if not isinstance(meta, dict):

            return {}

        if meta.get(STRUCTURAL_VERSION_KEY) != expected_structural_version:

            return {}

        return cache_obj

    except Exception:

        return {}



def save_cache(cache, structural_version: Optional[str] = None):

    try:

        if structural_version is not None:

            meta = cache.get(CACHE_META_KEY) if isinstance(cache.get(CACHE_META_KEY), dict) else {}

            meta[STRUCTURAL_VERSION_KEY] = structural_version

            cache[CACHE_META_KEY] = meta

        with open(CACHE, "wb") as fh:

            pickle.dump(cache, fh)

    except Exception:

        pass



# -----------------------------

# Search helpers

# -----------------------------



def read_context_lines(path: str, snippet: str, context: int) -> str:

    if context <= 0:

        return snippet

    try:

        with open(path, "r", errors="ignore") as f:

            lines = f.readlines()

        for i, line in enumerate(lines):

            if snippet.strip()[:30] in line:

                start = max(0, i - context)

                end = min(len(lines), i + context + 1)

                return "".join(lines[start:end])

    except Exception:

        pass

    return snippet


def _needs_raw_fts(query: str) -> bool:

    upper = query.upper()

    if any(token in upper for token in (" OR ", " AND ", " NOT ", " NEAR ", " WITHIN ")):

        return True

    if any(ch in query for ch in ('"', '*', '~', '^')):

        return True

    if ':' in query:

        return True

    return False


def _normalize_phrase(query: str) -> str:

    tokens = re.findall(r"\w+", query)

    if not tokens:

        return ""

    if len(tokens) == 1:

        return tokens[0]

    return '"' + " ".join(tokens) + '"'


def build_line_regex(query: str):

    tokens = re.findall(r"\w+", query)

    if not tokens:

        return None

    pattern = r"\\b" + r"\\W+".join(re.escape(tok) for tok in tokens) + r"\\b"

    try:

        return re.compile(pattern, re.IGNORECASE)

    except re.error:

        return None


def collect_line_info(path: str, regex, fallback: Optional[str], context: int) -> Tuple[List[int], str]:

    numbers: List[int] = []

    if regex is None and not fallback:

        return numbers, ""

    fallback_lower = fallback.lower() if fallback else None

    try:

        with open(path, "r", errors="ignore") as fh:

            lines = fh.readlines()

    except Exception:

        return numbers, ""

    seen = set()

    for idx, line in enumerate(lines, start=1):

        matched = False

        if regex and regex.search(line):

            matched = True

        elif fallback_lower and fallback_lower in line.lower():

            matched = True

        if matched and idx not in seen:

            seen.add(idx)

            numbers.append(idx)

    if not numbers:

        return numbers, ""

    max_line = len(lines)

    intervals: List[List[int]] = []

    context = max(context, 0)

    for n in numbers:

        start = max(1, n - context)

        end = min(max_line, n + context)

        if intervals and start <= intervals[-1][1] + 1:

            intervals[-1][1] = max(intervals[-1][1], end)

        else:

            intervals.append([start, end])

    blocks = []

    for start, end in intervals:

        blocks.append("".join(lines[start - 1:end]))

    snippet = "\n...\n".join(blocks).rstrip("\n")

    return numbers, snippet


def build_match_query(query: str, search_content: bool, search_name: bool) -> str:

    stripped = query.strip()

    if not stripped:

        return stripped

    if _needs_raw_fts(stripped):

        return stripped

    term = _normalize_phrase(stripped)

    if not term:

        return stripped

    if search_name and not search_content:

        return f'path:{term}'

    if search_content and not search_name:

        return f'content:{term}'

    return f'(path:{term} OR content:{term})'



def fuzzy_score(a: str, b: str) -> float:

    return difflib.SequenceMatcher(None, a, b).ratio()



def search_repo(

    query: str,

    content: bool = True,

    name: bool = False,

    fuzzy: bool = False,

    context: int = 0,

    json_mode: bool = True,

    collect_lines: bool = False,

    limit: int = 200,

    title_filters: Optional[List[str]] = None

) -> List[Dict]:

    """

    LLM-friendly search API.



    Returns list of {"path": path, "snippet": snippet}

    """

    conn = connect_db()

    cur = conn.cursor()

    results = []

    title_terms = [t.lower() for t in title_filters if t] if title_filters else []

    requires_raw = _needs_raw_fts(query)

    simple_content_search = content and not name and not fuzzy and not requires_raw

    collect_line_numbers = collect_lines and simple_content_search

    collect_context = context > 0 and simple_content_search

    line_regex = build_line_regex(query) if (collect_line_numbers or collect_context) else None

    fallback_substring: Optional[str] = query.strip() if (collect_line_numbers or collect_context) else None

    if fallback_substring == "":

        fallback_substring = None


    if fuzzy:

        for path, content_text in cur.execute("SELECT path, content FROM files"):

            score_path = fuzzy_score(path.lower(), query.lower())

            score_content = fuzzy_score(content_text.lower(), query.lower())

            score = max(score_path, score_content)

            if score > 0.4:

                snippet = content_text[:200].replace("\n", " ") + "..."

                results.append((score, path, snippet))

        results.sort(reverse=True)

        rows = [(p, s) for _, p, s in results[:limit]]

    else:

        mode_query = build_match_query(query, content, name)

        rows = cur.execute(

            "SELECT path, snippet(files, -1, '', '', '...', 1) FROM files WHERE files MATCH ? LIMIT ?",

            (mode_query, limit)

        ).fetchall()



    output = []

    for path, snippet in rows:

        if title_terms:

            path_lower = path.lower()

            if not all(term in path_lower for term in title_terms):

                continue

        line_numbers: List[int] = []

        collected_snippet = ""

        if (collect_line_numbers or collect_context) and (line_regex or fallback_substring):

            line_numbers, collected_snippet = collect_line_info(

                path,

                line_regex,

                fallback_substring,

                max(context, 0)

            )

        snippet_with_context = snippet

        if collect_context:

            if collected_snippet:

                snippet_with_context = collected_snippet

            else:

                snippet_with_context = read_context_lines(path, snippet, context)

        elif context > 0:

            snippet_with_context = read_context_lines(path, snippet, context)

        elif collected_snippet:

            snippet_with_context = collected_snippet

        record = {"path": path, "snippet": snippet_with_context}

        if collect_line_numbers:

            record["lines"] = line_numbers

        output.append(record)

    return output



# -----------------------------

# CLI

# -----------------------------



def main():

    if len(sys.argv) < 2:

        print(__doc__)

        return



    conn = connect_db()



    if sys.argv[1] == "index":

        index_repo(conn)

        return

    if sys.argv[1] == "migrate":

        dry_run = "--dry-run" in sys.argv[2:]

        pending = run_migrations(conn, dry_run=dry_run)

        if dry_run:

            if pending:

                print("\n".join(pending))

            else:

                print("[migrate] no pending migrations")

            return

        if pending:

            print(f"[migrate] applied {len(pending)} migration(s): {', '.join(pending)}")

        else:

            print("[migrate] no pending migrations")

        return

    if sys.argv[1] == "diagnose":

        json_flag = "--json" in sys.argv[2:]

        diagnose(conn, json_mode=json_flag)

        return

    if sys.argv[1] == "benchmark_structural":

        json_flag = "--json" in sys.argv[2:]

        positional = [arg for arg in sys.argv[2:] if not arg.startswith("--")]

        root = positional[0] if positional else "."

        benchmark_structural(root=root, json_mode=json_flag)

        return

    if sys.argv[1] == "ask":

        json_flag = "--json" in sys.argv[2:]

        question_parts = [arg for arg in sys.argv[2:] if not arg.startswith("--")]

        if not question_parts:

            print("Usage: python quedonde.py ask [--json] <question>", file=sys.stderr)

            return

        question = " ".join(question_parts)

        response = handle_structural_intent(conn, question)

        if json_flag:

            print(json.dumps(response, indent=2))

            return

        render_structural_cli_response(response)

        return

    if sys.argv[1] == "context":

        _handle_context_cli(conn, sys.argv[2:])

        return

    if sys.argv[1] in {"find", "callers", "deps", "explain"}:

        command = sys.argv[1]

        symbol, json_flag, paths_only, context, limit, error = _parse_structural_cli_args(sys.argv[2:])

        if error:

            print(error, file=sys.stderr)

            return

        if not symbol:

            print(

                f"Usage: python quedonde.py {command} [--json] [--paths] [--context N] [--limit N] <symbol>",

                file=sys.stderr,

            )

            return

        if command in {"callers", "deps"} and context > 0:

            print("[warn] --context applies only to find/explain commands", file=sys.stderr)

            context = 0

        if command == "find":

            rows = find_symbol_records(conn, symbol, limit=limit)

            if json_flag:

                print(json.dumps(rows, indent=2))

                return

            if paths_only:

                for path in _collect_paths_from_results(rows, ("path",)):

                    print(path)

                return

            _print_find_cli(symbol, rows, context)

            return

        if command == "callers":

            rows = callers_for_symbol(conn, symbol, limit=limit)

            if json_flag:

                print(json.dumps(rows, indent=2))

                return

            if paths_only:

                for path in _collect_paths_from_results(rows, ("src_path",)):

                    print(path)

                return

            _print_callers_cli(symbol, rows)

            return

        if command == "deps":

            rows = dependencies_for_symbol(conn, symbol, limit=limit)

            if json_flag:

                print(json.dumps(rows, indent=2))

                return

            if paths_only:

                for path in _collect_paths_from_results(rows, ("dst_path",)):

                    print(path)

                return

            _print_dependencies_cli(symbol, rows)

            return

        if command == "explain":

            details = explain_symbol_details(conn, symbol, limit=limit)

            if json_flag:

                print(json.dumps(details, indent=2))

                return

            if paths_only:

                defs = details.get("definitions") if isinstance(details, dict) else []

                for path in _collect_paths_from_results(defs or [], ("path",)):

                    print(path)

                return

            _print_explain_cli(symbol, details, context)

            return

    structural_token = get_structural_version(conn)



    json_mode = "--json" in sys.argv

    fuzzy = "--fuzzy" in sys.argv

    paths_only = "--paths" in sys.argv

    show_lines = "--lines" in sys.argv

    if paths_only:

        json_mode = False

        if show_lines:

            print("[warn] --lines ignored when --paths is set", file=sys.stderr)

            show_lines = False

    if show_lines and fuzzy:

        print("[warn] --lines is not supported with --fuzzy", file=sys.stderr)

        show_lines = False



    mode = "both"

    if "--name" in sys.argv:

        mode = "name"

    elif "--content" in sys.argv:

        mode = "content"

    if show_lines and mode == "name":

        print("[warn] --lines only applies to content searches", file=sys.stderr)

        show_lines = False



    context = 0

    title_filters: List[str] = []

    args = sys.argv[1:]

    pattern_parts: List[str] = []

    i = 0

    while i < len(args):

        arg = args[i]

        if arg == "--context":

            if i + 1 >= len(args):

                print("Invalid --context usage", file=sys.stderr)

                return

            try:

                context = int(args[i + 1])

            except Exception:

                print("Invalid --context usage", file=sys.stderr)

                return

            i += 2

            continue

        if arg == "--title":

            if i + 1 >= len(args):

                print("Invalid --title usage", file=sys.stderr)

                return

            title_filters.append(args[i + 1])

            i += 2

            continue

        if arg in {"--json", "--name", "--content", "--fuzzy", "--paths", "--lines"}:

            i += 1

            continue

        if arg.startswith("--"):

            i += 1

            continue

        pattern_parts.append(arg)

        i += 1



    if not pattern_parts and not title_filters:

        print("Usage: python quedonde.py [--json|--paths] [--name|--content|--fuzzy] [--context N] [--title TEXT] <pattern>", file=sys.stderr)

        return



    pattern = " ".join(pattern_parts)

    title_filter_key = "|".join(title_filters)

    if show_lines and _needs_raw_fts(pattern):

        print("[warn] --lines ignored for advanced FTS queries", file=sys.stderr)

        show_lines = False



    cache = load_cache(structural_token)

    key = cache_key(pattern, mode, fuzzy, context, json_mode, paths_only, show_lines, title_filter_key)

    if key in cache:

        cached_output = cache[key]

        if cached_output:

            print(cached_output)

        print("[cache] hit", file=sys.stderr)

        return



    start = time.time()

    results = search_repo(

        pattern,

        content=(mode != "name"),

        name=(mode == "name"),

        fuzzy=fuzzy,

        context=context,

        json_mode=json_mode,

        collect_lines=show_lines,

        title_filters=title_filters

    )

    elapsed = time.time() - start



    if paths_only:

        output = "\n".join([r["path"] for r in results])

    elif json_mode:

        output = json.dumps(results, indent=2)

    else:
        lines_formatted = []

        for r in results:

            suffix = ""

            if show_lines:

                line_numbers = r.get("lines") or []

                if line_numbers:

                    suffix = ":" + ",".join(str(num) for num in line_numbers)

            lines_formatted.append(f"{r['path']}{suffix}:\n{r['snippet']}")

        output = "\n".join(lines_formatted)



    if output:

        print(output)



    print(f"[done] {elapsed:.2f}s", file=sys.stderr)



    cache[key] = output

    save_cache(cache, structural_token)



if __name__ == "__main__":

    main()
