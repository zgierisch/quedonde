"""Session state checkpoint schema helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

SCHEMA_VERSION = "1.0"
ALLOWED_RELATIONS = {"calls", "imports", "includes", "references", "annotates"}
CONFIDENCE_LEVELS = {"low", "medium", "high"}
MAX_TEXT_FIELD_LENGTH = 240
MAX_QUESTION_LENGTH = 200
MAX_SYMBOL_NAME = 128


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
            _ensure(value and len(value) <= 64, "annotation entries must be <= 64 chars")


@dataclass
class SymbolSummary:
    kind: str
    role: str
    evidence: Evidence
    source_paths: List[str] = field(default_factory=list)

    def validate(self, symbol_name: str) -> None:
        _ensure(symbol_name and len(symbol_name) <= MAX_SYMBOL_NAME, "invalid symbol name")
        _ensure(self.kind and len(self.kind) <= 64, "invalid symbol kind")
        _ensure(self.role and len(self.role) <= 64, "invalid role")
        _ensure(bool(self.source_paths), "each symbol summary requires at least one source path")
        for path in self.source_paths:
            _ensure(path and len(path) <= 256, "source paths must be <= 256 chars")
        self.evidence.validate()


@dataclass
class DependencyRecord:
    src: str
    relation: str
    dst: str

    def validate(self) -> None:
        for value in (self.src, self.dst):
            _ensure(value and len(value) <= MAX_SYMBOL_NAME, "dependency endpoints required")
        _ensure(self.relation in ALLOWED_RELATIONS, f"relation must be one of {sorted(ALLOWED_RELATIONS)}")


@dataclass
class DecisionRecord:
    decision: str
    reason: str

    def validate(self) -> None:
        _ensure(_valid_text(self.decision), "decision text invalid")
        _ensure(_valid_text(self.reason), "reason text invalid")


def _valid_text(value: str, limit: int = MAX_TEXT_FIELD_LENGTH) -> bool:
    return bool(value) and len(value) <= limit and "\n" not in value and "\r" not in value


@dataclass
class SessionScope:
    subsystems: List[str]
    files: List[str]

    def validate(self) -> None:
        _ensure(bool(self.subsystems), "scope.subsystems must not be empty")
        _ensure(bool(self.files), "scope.files must not be empty")
        for entry in self.subsystems:
            _ensure(entry and len(entry) <= 64, "subsystem entries must be <= 64 chars")
        for entry in self.files:
            _ensure(entry and len(entry) <= 256, "file paths must be <= 256 chars")


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
        _ensure(_valid_text(self.session_id, 128), "session_id invalid")
        _ensure(self.confidence in CONFIDENCE_LEVELS, "confidence must be low/medium/high")
        _ensure(self.structural_version, "structural_version required")
        _ensure(_is_iso_timestamp(self.timestamp), "timestamp must be RFC3339/ISO")
        self.scope.validate()
        _ensure(self.symbols, "symbols map must not be empty")
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
    return session_state_from_dict(data)


def session_state_from_dict(payload: Mapping[str, Any]) -> SessionState:
    _ensure(isinstance(payload, Mapping), "session state payload must be an object")
    scope_dict = payload.get("scope") or {}
    scope = SessionScope(
        subsystems=list(scope_dict.get("subsystems") or []),
        files=list(scope_dict.get("files") or []),
    )
    symbols: Dict[str, SymbolSummary] = {}
    for name, summary in (payload.get("symbols") or {}).items():
        evidence = summary.get("evidence") or {}
        symbols[name] = SymbolSummary(
            kind=summary.get("kind", ""),
            role=summary.get("role", ""),
            evidence=Evidence(
                fan_in=int(evidence.get("fan_in", 0)),
                fan_out=int(evidence.get("fan_out", 0)),
                annotations=list(evidence.get("annotations") or []),
            ),
            source_paths=list(summary.get("source_paths") or []),
        )
    dependencies = [
        DependencyRecord(
            src=item.get("src", ""),
            relation=item.get("relation", ""),
            dst=item.get("dst", ""),
        )
        for item in (payload.get("dependencies") or [])
    ]
    decisions = [
        DecisionRecord(decision=item.get("decision", ""), reason=item.get("reason", ""))
        for item in (payload.get("decisions") or [])
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


def new_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
*** End of File***