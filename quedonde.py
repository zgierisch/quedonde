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

- Fully offline, no server, no dependencies

"""



import os, sys, sqlite3, difflib, hashlib, pickle, json, time, re

from typing import List, Dict, Optional



# -----------------------------

# Config

# -----------------------------



DB = ".code_index.sqlite"

CACHE = ".code_index.cache"

IGNORES = {'.git', '__pycache__', 'node_modules', '.venv', 'dist', 'build'}

EXTS = (

    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".md", ".json",

    ".yaml", ".yml", ".cpp", ".h", ".java", ".go", ".rs", ".sh", ".txt"

)



# -----------------------------

# Database & indexing

# -----------------------------



def connect_db():

    conn = sqlite3.connect(DB)

    conn.execute("CREATE TABLE IF NOT EXISTS meta(path TEXT PRIMARY KEY, mtime REAL)")

    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS files USING fts5(path, content)")

    return conn



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

        except Exception as e:

            print(f"[skip] {path}: {e}")



    # Remove DB entries for files that disappeared since the last run.

    for (indexed_path,) in cur.execute("SELECT path FROM meta"):

        if indexed_path not in seen_paths and not os.path.exists(indexed_path):

            cur.execute("DELETE FROM files WHERE path=?", (indexed_path,))

            cur.execute("DELETE FROM meta WHERE path=?", (indexed_path,))

            removed += 1



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



    if os.path.exists(CACHE):

        os.remove(CACHE)

        print("[cache] cleared")



# -----------------------------

# Cache

# -----------------------------



def cache_key(pattern, mode, fuzzy, context, json_mode, paths_only, lines_flag):

    data = f"{pattern}:{mode}:{fuzzy}:{context}:{json_mode}:{paths_only}:{lines_flag}"

    return hashlib.md5(data.encode()).hexdigest()



def load_cache():

    if not os.path.exists(CACHE):

        return {}

    try:

        with open(CACHE, "rb") as fh:

            return pickle.load(fh)

    except Exception:

        return {}



def save_cache(cache):

    try:

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

    if any(ch in query for ch in ('"', '*', '(', ')', '~', '^')):

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


def find_line_numbers(path: str, regex, fallback: Optional[str]) -> List[int]:

    numbers: List[int] = []

    if regex is None and not fallback:

        return numbers

    seen = set()

    fallback_lower = fallback.lower() if fallback else None

    try:

        with open(path, "r", errors="ignore") as fh:

            for idx, line in enumerate(fh, start=1):

                matched = False

                if regex and regex.search(line):

                    matched = True

                elif fallback_lower and fallback_lower in line.lower():

                    matched = True

                if matched and idx not in seen:

                    seen.add(idx)

                    numbers.append(idx)

    except Exception:

        pass

    return numbers


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

    limit: int = 200

) -> List[Dict]:

    """

    LLM-friendly search API.



    Returns list of {"path": path, "snippet": snippet}

    """

    conn = connect_db()

    cur = conn.cursor()

    results = []

    collect_line_numbers = collect_lines and content and not name and not fuzzy

    if collect_line_numbers and _needs_raw_fts(query):

        collect_line_numbers = False

    line_regex = build_line_regex(query) if collect_line_numbers else None

    fallback_substring: Optional[str] = query.strip() if collect_line_numbers else None

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

        snippet_with_context = read_context_lines(path, snippet, context)

        record = {"path": path, "snippet": snippet_with_context}

        if collect_line_numbers:

            record["lines"] = find_line_numbers(path, line_regex, fallback_substring)

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

    if "--context" in sys.argv:

        try:

            idx = sys.argv.index("--context")

            context = int(sys.argv[idx + 1])

        except Exception:

            print("Invalid --context usage", file=sys.stderr)

            return



    args = sys.argv[1:]

    pattern_parts = []

    skip_next = False

    for arg in args:

        if skip_next:

            skip_next = False

            continue

        if arg == "--context":

            skip_next = True

            continue

        if arg in {"--json", "--name", "--content", "--fuzzy", "--paths", "--lines"}:

            continue

        if arg.startswith("--"):

            continue

        pattern_parts.append(arg)



    if not pattern_parts:

        print("Usage: python quedonde.py [--json|--paths] [--name|--content|--fuzzy] [--context N] <pattern>", file=sys.stderr)

        return



    pattern = " ".join(pattern_parts)

    if show_lines and _needs_raw_fts(pattern):

        print("[warn] --lines ignored for advanced FTS queries", file=sys.stderr)

        show_lines = False



    cache = load_cache()

    key = cache_key(pattern, mode, fuzzy, context, json_mode, paths_only, show_lines)

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

        collect_lines=show_lines

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

    save_cache(cache)



if __name__ == "__main__":

    main()
