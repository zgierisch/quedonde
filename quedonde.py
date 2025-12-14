            _add_edge(edges, path, FILE_SYMBOL, "imports", match.group(1))

        for require_match in JS_REQUIRE_RE.finditer(stripped):

            _add_edge(edges, path, FILE_SYMBOL, "imports", require_match.group(1))

        current_symbol = _symbol_for_line(lookup, line_number)

        for call_match in CALL_IDENTIFIER_RE.finditer(stripped):
    for entry in results[:limit]:

        path = entry.get("path") if isinstance(entry, dict) else None

        line_start = entry.get("line_start") if isinstance(entry, dict) else None

        line_end = entry.get("line_end") if isinstance(entry, dict) else None
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


    # -----------------------------
    # Session state helpers
    # -----------------------------


    def _build_session_symbol_summary(
        conn: sqlite3.Connection,
        symbol: str,
        *,
        dependency_limit: int = 25,
    ) -> Tuple[SymbolSummary, List[DependencyRecord]]:
        row = conn.execute(
            "SELECT kind FROM symbols WHERE symbol=? ORDER BY path LIMIT 1",
            (symbol,),
        ).fetchone()
        if not row:
            raise SessionStateValidationError(f"symbol '{symbol}' not found in structural tables")

        path_rows = conn.execute(
            "SELECT DISTINCT path FROM symbols WHERE symbol=? AND path IS NOT NULL ORDER BY path LIMIT 5",
            (symbol,),
        ).fetchall()
        source_paths = [entry[0] for entry in path_rows if entry[0]]
        if not source_paths:
            raise SessionStateValidationError(f"symbol '{symbol}' is missing definition paths")

        annotations = [
            entry["tag"]
            for entry in conn.execute(
                "SELECT DISTINCT tag FROM annotations WHERE symbol=? ORDER BY tag",
                (symbol,),
            ).fetchall()
        ]
        role = annotations[0] if annotations else "unspecified"
        fan_in = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE dst_symbol=?",
            (symbol,),
        ).fetchone()[0]
        fan_out = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_symbol=?",
            (symbol,),
        ).fetchone()[0]
        summary = SymbolSummary(
            kind=row["kind"],
            role=role,
            evidence=Evidence(fan_in=int(fan_in), fan_out=int(fan_out), annotations=annotations),
            source_paths=source_paths,
        )

        dep_rows = conn.execute(
            """
            SELECT relation, dst_symbol, dst_path
            FROM edges
            WHERE src_symbol=?
                AND relation IN ('calls', 'imports', 'includes', 'references')
            ORDER BY relation, dst_symbol, dst_path
            LIMIT ?
            """,
            (symbol, dependency_limit),
        ).fetchall()
        seen: Set[Tuple[str, str, str]] = set()
        dependencies: List[DependencyRecord] = []
        for dep in dep_rows:
            dst = dep["dst_symbol"] or dep["dst_path"]
            if not dst:
                continue
            key = (symbol, dep["relation"], dst)
            if key in seen:
                continue
            seen.add(key)
            dependencies.append(DependencyRecord(src=symbol, relation=dep["relation"], dst=dst))

        summary.validate(symbol)
        for dep in dependencies:
            dep.validate()
        return summary, dependencies


    def _parse_decision_entry(text: str) -> DecisionRecord:
        if "::" not in text:
            raise SessionStateValidationError("decisions must use 'decision::reason' format")
        decision, reason = text.split("::", 1)
        record = DecisionRecord(decision=decision.strip(), reason=reason.strip())
        record.validate()
        return record


    def _write_session_output(
        payload: Dict[str, object],
        *,
        path: Path,
        append: bool,
        force: bool,
    ) -> None:
        state_json = json.dumps(payload, indent=2) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if append and path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                existing.append(payload)
            else:
                existing = [existing, payload]
            path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            print(f"[session] appended checkpoint to {path}")
            return

        if path.exists() and not force:
            raise SessionStateValidationError(
                f"Refusing to overwrite existing checkpoint {path}; use --force or --append"
            )
        path.write_text(state_json, encoding="utf-8")
        print(f"[session] wrote checkpoint to {path}")


    def _revalidate_session_state(
        conn: sqlite3.Connection,
        state: SessionState,
        *,
        verify_paths: bool = True,
    ) -> List[str]:
        warnings: List[str] = []
        if not structural_ready(conn):
            return [
                "structural tables unavailable; run 'python quedonde.py migrate' + 'index' before verifying",
            ]

        current_version = get_structural_version(conn)
        if current_version != state.structural_version:
            warnings.append(
                f"structural version drift: checkpoint={state.structural_version} current={current_version}"
            )

        for symbol_name, summary in state.symbols.items():
            row = conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE symbol=?",
                (symbol_name,),
            ).fetchone()
            if not row or int(row[0]) == 0:
                warnings.append(f"symbol '{symbol_name}' missing from structural tables")
                continue

            fan_in = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE dst_symbol=?",
                (symbol_name,),
            ).fetchone()[0]
            fan_out = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE src_symbol=?",
                (symbol_name,),
            ).fetchone()[0]
            if int(fan_in) != summary.evidence.fan_in:
                warnings.append(
                    f"symbol '{symbol_name}' fan_in mismatch (checkpoint={summary.evidence.fan_in}, current={fan_in})"
                )
            if int(fan_out) != summary.evidence.fan_out:
                warnings.append(
                    f"symbol '{symbol_name}' fan_out mismatch (checkpoint={summary.evidence.fan_out}, current={fan_out})"
                )

            if verify_paths and summary.source_paths:
                missing_paths = [
                    path
                    for path in summary.source_paths
                    if not conn.execute(
                        "SELECT 1 FROM symbols WHERE symbol=? AND path=? LIMIT 1",
                        (symbol_name, path),
                    ).fetchone()
                ]
                if missing_paths:
                    warnings.append(
                        f"symbol '{symbol_name}' missing previously recorded path(s): {', '.join(missing_paths)}"
                    )

        return warnings


    def _print_session_state_summary(state: SessionState) -> None:
        print(f"Session: {state.session_id} (confidence={state.confidence})")
        print(f"Timestamp: {state.timestamp}")
        print(f"Structural version: {state.structural_version}")
        print("Subsystems: " + ", ".join(state.scope.subsystems))
        print("Files: " + ", ".join(state.scope.files))
        print("Symbols:")
        for name, summary in state.symbols.items():
            evidence = summary.evidence
            annotations = ", ".join(summary.evidence.annotations) if summary.evidence.annotations else "none"
            print(
                f"  - {name} ({summary.kind}, role={summary.role}) fan_in={evidence.fan_in} "
                f"fan_out={evidence.fan_out} annotations={annotations}"
            )
        if state.dependencies:
            print("Dependencies:")
            for dep in state.dependencies:
                print(f"  - {dep.src} --{dep.relation}--> {dep.dst}")
        if state.decisions:
            print("Decisions:")
            for dec in state.decisions:
                print(f"  - {dec.decision} (reason: {dec.reason})")
        if state.open_questions:
            print("Open questions:")
            for question in state.open_questions:
                print(f"  - {question}")


    def session_dump_cli(conn: sqlite3.Connection, argv: List[str]) -> None:
        parser = argparse.ArgumentParser(prog="python quedonde.py session dump")
        parser.add_argument("--session-id", required=True)
        parser.add_argument("--subsystem", dest="subsystems", action="append", required=True, help="Scope subsystem (repeat)")
        parser.add_argument("--file", dest="files", action="append", required=True, help="Scope file path (repeat)")
        parser.add_argument("--symbol", dest="symbols", action="append", required=True, help="Symbol to summarize (repeat)")
        parser.add_argument("--confidence", choices=sorted(CONFIDENCE_LEVELS), default="medium")
        parser.add_argument("--decision", dest="decisions", action="append", help="Decision entry formatted as 'decision::reason'")
        parser.add_argument("--question", dest="questions", action="append", help="Open question (repeat)")
        parser.add_argument("--output", default="quedonde_session_state.json", help="Destination file")
        parser.add_argument("--append", action="store_true", help="Append to an existing JSON array of checkpoints")
        parser.add_argument("--force", action="store_true", help="Overwrite existing file when not using --append")
        parser.add_argument("--deps-limit", type=int, default=25, help="Max dependency edges per symbol")
        args = parser.parse_args(argv)

        if not structural_available(conn):
            return

        try:
            scope = SessionScope(subsystems=args.subsystems, files=args.files)
            scope.validate()
            symbols: Dict[str, SymbolSummary] = {}
            dependencies: List[DependencyRecord] = []
            for symbol in args.symbols:
                summary, deps = _build_session_symbol_summary(
                    conn,
                    symbol,
                    dependency_limit=max(1, args.deps_limit),
                )
                symbols[symbol] = summary
                dependencies.extend(deps)

            decisions = [
                _parse_decision_entry(text)
                for text in (args.decisions or [])
            ]
            open_questions = list(args.questions or [])

            state = SessionState(
                session_id=args.session_id,
                timestamp=new_timestamp(),
                structural_version=get_structural_version(conn),
                scope=scope,
                symbols=symbols,
                confidence=args.confidence,
                dependencies=dependencies,
                decisions=decisions,
                open_questions=open_questions,
            )
            payload = session_state_to_dict(state)
            _write_session_output(
                payload,
                path=Path(args.output),
                append=args.append,
                force=args.force,
            )
        except SessionStateValidationError as exc:
            print(f"[session] {exc}", file=sys.stderr)


    def _load_session_payload(path: Path, session_id: Optional[str]) -> Dict[str, object]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not session_id:
                raise SessionStateValidationError("--session-id is required when loading from an array")
            for item in data:
                if item.get("session_id") == session_id:
                    return item
            raise SessionStateValidationError(f"session_id '{session_id}' not found in {path}")
        if session_id and data.get("session_id") != session_id:
            raise SessionStateValidationError(
                f"Checkpoint session_id {data.get('session_id')} does not match requested {session_id}"
            )
        return data


    def session_resume_cli(conn: sqlite3.Connection, argv: List[str]) -> None:
        parser = argparse.ArgumentParser(prog="python quedonde.py session resume")
        parser.add_argument("--input", default="quedonde_session_state.json", help="Checkpoint file to read")
        parser.add_argument("--session-id", help="Specific session_id inside the file")
        parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of summary")
        parser.add_argument(
            "--skip-verify",
            action="store_true",
            help="Skip structural drift checks when resuming",
        )
        args = parser.parse_args(argv)

        path = Path(args.input)
        if not path.exists():
            print(f"[session] checkpoint file {path} not found", file=sys.stderr)
            return

        try:
            payload = _load_session_payload(path, args.session_id)
            state = session_state_from_dict(payload)
        except SessionStateValidationError as exc:
            print(f"[session] {exc}", file=sys.stderr)
            return
        except json.JSONDecodeError as exc:
            print(f"[session] failed to parse {path}: {exc}", file=sys.stderr)
            return

        if not args.skip_verify:
            warnings = _revalidate_session_state(conn, state)
            if warnings:
                print("[session] revalidation warnings detected:", file=sys.stderr)
                for message in warnings:
                    print(f"  - {message}", file=sys.stderr)
            else:
                print("[session] checkpoint matches current structural state", file=sys.stderr)

        if args.json:
            print(json.dumps(session_state_to_dict(state), indent=2))
        else:
            _print_session_state_summary(state)


    def session_cli(conn: sqlite3.Connection, argv: List[str]) -> None:
        if not argv:
            print("Usage: python quedonde.py session <dump|resume> [options]", file=sys.stderr)
            return
        subcommand = argv[0]
        if subcommand == "dump":
            session_dump_cli(conn, argv[1:])
        elif subcommand == "resume":
            session_resume_cli(conn, argv[1:])
        else:
            print(f"[session] unknown subcommand '{subcommand}'", file=sys.stderr)

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
# -----------------------------
# Session state helpers
# -----------------------------


def _build_session_symbol_summary(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    dependency_limit: int = 25,
) -> Tuple[SymbolSummary, List[DependencyRecord]]:
    row = conn.execute(
        "SELECT kind FROM symbols WHERE symbol=? ORDER BY path LIMIT 1",
        (symbol,),
    ).fetchone()
    if not row:
        raise SessionStateValidationError(f"symbol '{symbol}' not found in structural tables")

    path_rows = conn.execute(
        "SELECT DISTINCT path FROM symbols WHERE symbol=? AND path IS NOT NULL ORDER BY path LIMIT 5",
        (symbol,),
    ).fetchall()
    source_paths = [entry[0] for entry in path_rows if entry[0]]
    if not source_paths:
        raise SessionStateValidationError(f"symbol '{symbol}' is missing definition paths")

    annotations = [
        entry["tag"]
        for entry in conn.execute(
            "SELECT DISTINCT tag FROM annotations WHERE symbol=? ORDER BY tag",
            (symbol,),
        ).fetchall()
    ]
    role = annotations[0] if annotations else "unspecified"
    fan_in = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE dst_symbol=?",
        (symbol,),
    ).fetchone()[0]
    fan_out = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src_symbol=?",
        (symbol,),
    ).fetchone()[0]
    summary = SymbolSummary(
        kind=row["kind"],
        role=role,
        evidence=Evidence(fan_in=int(fan_in), fan_out=int(fan_out), annotations=annotations),
        source_paths=source_paths,
    )

    dep_rows = conn.execute(
        """
        SELECT relation, dst_symbol, dst_path
        FROM edges
        WHERE src_symbol=?
            AND relation IN ('calls', 'imports', 'includes', 'references')
        ORDER BY relation, dst_symbol, dst_path
        LIMIT ?
        """,
        (symbol, dependency_limit),
    ).fetchall()
    seen: Set[Tuple[str, str, str]] = set()
    dependencies: List[DependencyRecord] = []
    for dep in dep_rows:
        dst = dep["dst_symbol"] or dep["dst_path"]
        if not dst:
            continue
        key = (symbol, dep["relation"], dst)
        if key in seen:
            continue
        seen.add(key)
        dependencies.append(DependencyRecord(src=symbol, relation=dep["relation"], dst=dst))

    summary.validate(symbol)
    for dep in dependencies:
        dep.validate()
    return summary, dependencies


def _parse_decision_entry(text: str) -> DecisionRecord:
    if "::" not in text:
        raise SessionStateValidationError("decisions must use 'decision::reason' format")
    decision, reason = text.split("::", 1)
    record = DecisionRecord(decision=decision.strip(), reason=reason.strip())
    record.validate()
    return record


def _write_session_output(
    payload: Dict[str, object],
    *,
    path: Path,
    append: bool,
    force: bool,
) -> None:
    state_json = json.dumps(payload, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            existing.append(payload)
        else:
            existing = [existing, payload]
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(f"[session] appended checkpoint to {path}")
        return

    if path.exists() and not force:
        raise SessionStateValidationError(
            f"Refusing to overwrite existing checkpoint {path}; use --force or --append"
        )
    path.write_text(state_json, encoding="utf-8")
    print(f"[session] wrote checkpoint to {path}")


def _revalidate_session_state(
    conn: sqlite3.Connection,
    state: SessionState,
    *,
    verify_paths: bool = True,
) -> List[str]:
    warnings: List[str] = []
    if not structural_ready(conn):
        return [
            "structural tables unavailable; run 'python quedonde.py migrate' + 'index' before verifying",
        ]

    current_version = get_structural_version(conn)
    if current_version != state.structural_version:
        warnings.append(
            f"structural version drift: checkpoint={state.structural_version} current={current_version}"
        )

    for symbol_name, summary in state.symbols.items():
        row = conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE symbol=?",
            (symbol_name,),
        ).fetchone()
        if not row or int(row[0]) == 0:
            warnings.append(f"symbol '{symbol_name}' missing from structural tables")
            continue

        fan_in = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE dst_symbol=?",
            (symbol_name,),
        ).fetchone()[0]
        fan_out = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_symbol=?",
            (symbol_name,),
        ).fetchone()[0]
        if int(fan_in) != summary.evidence.fan_in:
            warnings.append(
                f"symbol '{symbol_name}' fan_in mismatch (checkpoint={summary.evidence.fan_in}, current={fan_in})"
            )
        if int(fan_out) != summary.evidence.fan_out:
            warnings.append(
                f"symbol '{symbol_name}' fan_out mismatch (checkpoint={summary.evidence.fan_out}, current={fan_out})"
            )

        if verify_paths and summary.source_paths:
            missing_paths = [
                path
                for path in summary.source_paths
                if not conn.execute(
                    "SELECT 1 FROM symbols WHERE symbol=? AND path=? LIMIT 1",
                    (symbol_name, path),
                ).fetchone()
            ]
            if missing_paths:
                warnings.append(
                    f"symbol '{symbol_name}' missing previously recorded path(s): {', '.join(missing_paths)}"
                )

    return warnings


def _print_session_state_summary(state: SessionState) -> None:
    print(f"Session: {state.session_id} (confidence={state.confidence})")
    print(f"Timestamp: {state.timestamp}")
    print(f"Structural version: {state.structural_version}")
    print("Subsystems: " + ", ".join(state.scope.subsystems))
    print("Files: " + ", ".join(state.scope.files))
    print("Symbols:")
    for name, summary in state.symbols.items():
        evidence = summary.evidence
        annotations = ", ".join(summary.evidence.annotations) if summary.evidence.annotations else "none"
        print(
            f"  - {name} ({summary.kind}, role={summary.role}) fan_in={evidence.fan_in} fan_out={evidence.fan_out} "
            f"annotations={annotations}"
        )
    if state.dependencies:
        print("Dependencies:")
        for dep in state.dependencies:
            print(f"  - {dep.src} --{dep.relation}--> {dep.dst}")
    if state.decisions:
        print("Decisions:")
        for dec in state.decisions:
            print(f"  - {dec.decision} (reason: {dec.reason})")
    if state.open_questions:
        print("Open questions:")
        for question in state.open_questions:
            print(f"  - {question}")


def session_dump_cli(conn: sqlite3.Connection, argv: List[str]) -> None:
    parser = argparse.ArgumentParser(prog="python quedonde.py session dump")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--subsystem", dest="subsystems", action="append", required=True, help="Scope subsystem (repeat)")
    parser.add_argument("--file", dest="files", action="append", required=True, help="Scope file path (repeat)")
    parser.add_argument("--symbol", dest="symbols", action="append", required=True, help="Symbol to summarize (repeat)")
    parser.add_argument("--confidence", choices=sorted(CONFIDENCE_LEVELS), default="medium")

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

    if sys.argv[1] == "session":

        session_cli(conn, sys.argv[2:])

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
