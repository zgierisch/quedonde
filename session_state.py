"""Session state checkpoint schema helpers."""

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
