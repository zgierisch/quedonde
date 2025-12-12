"""Regression tests for the indexing pipeline using a tiny fixture repo."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import quedonde

FIXTURE_REPO = Path("documentation/fixtures/indexing_repo")


class IndexingRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._tmpdir.name) / "repo"
        shutil.copytree(FIXTURE_REPO, self.repo_root)

        self._old_db = quedonde.DB
        self._old_cache = quedonde.CACHE
        self._old_struct_ready = quedonde._STRUCTURAL_READY  # type: ignore[attr-defined]

        quedonde.DB = os.path.join(self._tmpdir.name, ".code_index.sqlite")
        quedonde.CACHE = os.path.join(self._tmpdir.name, ".code_index.cache")
        quedonde._STRUCTURAL_READY = None  # type: ignore[attr-defined]

        self.conn = quedonde.connect_db()
        self._create_structural_tables(self.conn)
        quedonde._STRUCTURAL_READY = None  # type: ignore[attr-defined]

    def tearDown(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        quedonde.DB = self._old_db
        quedonde.CACHE = self._old_cache
        quedonde._STRUCTURAL_READY = self._old_struct_ready  # type: ignore[attr-defined]
        self._tmpdir.cleanup()

    def _create_structural_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                path TEXT NOT NULL,
                symbol TEXT NOT NULL,
                kind TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                PRIMARY KEY(path, symbol, kind)
            );
            CREATE TABLE IF NOT EXISTS edges (
                src_path TEXT NOT NULL,
                src_symbol TEXT NOT NULL,
                relation TEXT NOT NULL,
                dst_path TEXT,
                dst_symbol TEXT
            );
            CREATE TABLE IF NOT EXISTS annotations (
                path TEXT NOT NULL,
                symbol TEXT,
                tag TEXT NOT NULL,
                line INTEGER
            );
            INSERT OR REPLACE INTO state(key, value) VALUES('structural_version', 'test');
            """
        )
        conn.commit()

    def test_index_repo_populates_structural_tables(self) -> None:
        quedonde.index_repo(self.conn, root=str(self.repo_root))

        file_count = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        symbol_count = self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        edge_count = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        annotation_count = self.conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]

        self.assertEqual(file_count, 2)
        self.assertEqual(symbol_count, 3)
        self.assertGreaterEqual(edge_count, 1)
        self.assertEqual(annotation_count, 1)

        orchestrator_row = self.conn.execute(
            "SELECT symbol FROM symbols WHERE symbol='orchestrator'"
        ).fetchone()
        self.assertIsNotNone(orchestrator_row)

        annotation = self.conn.execute(
            "SELECT tag FROM annotations WHERE symbol='orchestrator'"
        ).fetchone()
        self.assertEqual(annotation[0], "orchestrator")


if __name__ == "__main__":
    unittest.main()
