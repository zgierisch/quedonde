"""Snapshot-style tests for structural CLI helpers."""

from __future__ import annotations

import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import quedonde

FIXTURE_PATH = Path("documentation/fixtures/structural_cli/sample.py")


class StructuralCLISnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._original_db = quedonde.DB
        self._original_structural_flag = quedonde._STRUCTURAL_READY  # type: ignore[attr-defined]
        quedonde.DB = os.path.join(self._tmpdir.name, ".code_index.sqlite")
        quedonde._STRUCTURAL_READY = None  # type: ignore[attr-defined]
        conn = quedonde.connect_db()
        self._create_structural_tables(conn)
        self._seed_structural_data(conn)
        conn.close()
        quedonde._STRUCTURAL_READY = None  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        quedonde.DB = self._original_db
        quedonde._STRUCTURAL_READY = self._original_structural_flag  # type: ignore[attr-defined]
        self._tmpdir.cleanup()

    def _create_structural_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE symbols (
                path TEXT NOT NULL,
                symbol TEXT NOT NULL,
                kind TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                PRIMARY KEY(path, symbol, kind)
            );
            CREATE TABLE edges (
                src_path TEXT NOT NULL,
                src_symbol TEXT NOT NULL,
                relation TEXT NOT NULL,
                dst_path TEXT,
                dst_symbol TEXT
            );
            CREATE TABLE annotations (
                path TEXT NOT NULL,
                symbol TEXT,
                tag TEXT NOT NULL,
                line INTEGER
            );
            INSERT OR REPLACE INTO state(key, value) VALUES('structural_version', 'test');
            """
        )
        conn.commit()

    def _seed_structural_data(self, conn: sqlite3.Connection) -> None:
        fixture = FIXTURE_PATH.as_posix()
        conn.executemany(
            "INSERT INTO symbols(path, symbol, kind, line_start, line_end) VALUES(?, ?, ?, ?, ?)",
            [
                (fixture, "alpha", "function", 1, 3),
                (fixture, "beta", "function", 5, 6),
                (fixture, "gamma", "function", 8, 10),
            ],
        )
        conn.executemany(
            "INSERT INTO edges(src_path, src_symbol, relation, dst_path, dst_symbol) VALUES(?, ?, ?, ?, ?)",
            [
                (fixture, "beta", "calls", fixture, "alpha"),
                (fixture, "gamma", "calls", fixture, "beta"),
            ],
        )
        conn.execute(
            "INSERT INTO annotations(path, symbol, tag, line) VALUES(?, ?, ?, ?)",
            (fixture, "alpha", "legacy", 2),
        )
        conn.commit()

    def _capture(self, func, *args, **kwargs) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            func(*args, **kwargs)
        return buffer.getvalue().strip()

    def test_find_command_snapshot(self) -> None:
        rows = quedonde.find_symbol("alpha", limit=5)
        output = self._capture(quedonde._print_find_cli, "alpha", rows, 1)  # type: ignore[attr-defined]
        self.assertEqual(
            output,
            "\n".join(
                [
                    "[find] 1 match(es) for 'alpha'",
                    "  - documentation/fixtures/structural_cli/sample.py:1-3 alpha [function]",
                    "    def alpha():",
                    "        \"\"\"Alpha function returns a constant.\"\"\"",
                    "        return \"alpha\"",
                ]
            ),
        )

    def test_callers_command_snapshot(self) -> None:
        rows = quedonde.get_callers("alpha", limit=5)
        output = self._capture(quedonde._print_callers_cli, "alpha", rows)  # type: ignore[attr-defined]
        self.assertEqual(
            output,
            """[callers] 1 caller(s) for 'alpha'\n  - beta --calls--> alpha (documentation/fixtures/structural_cli/sample.py)""",
        )

    def test_deps_command_snapshot(self) -> None:
        rows = quedonde.get_dependencies("beta", limit=5)
        output = self._capture(quedonde._print_dependencies_cli, "beta", rows)  # type: ignore[attr-defined]
        self.assertEqual(
            output,
            """[deps] 1 relation(s) for 'beta'\n  - beta --calls--> alpha (documentation/fixtures/structural_cli/sample.py)""",
        )

    def test_explain_command_snapshot(self) -> None:
        details = quedonde.explain_symbol("alpha", limit=5)
        output = self._capture(quedonde._print_explain_cli, "alpha", details, 1)  # type: ignore[attr-defined]
        self.assertEqual(
            output,
            "\n".join(
                [
                    "[explain] alpha",
                    "  definitions:",
                    "    - documentation/fixtures/structural_cli/sample.py:1-3 alpha [function]",
                    "      def alpha():",
                    "          \"\"\"Alpha function returns a constant.\"\"\"",
                    "          return \"alpha\"",
                    "  callers: 1",
                    "  dependencies: 0",
                    "  annotations: 1",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
