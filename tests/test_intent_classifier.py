"""Intent classifier regression tests based on markdown fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import unittest

import quedonde

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "documentation" / "fixtures" / "intent" / "queries.md"


def load_fixtures() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not FIXTURE_PATH.exists():
        return rows
    for raw_line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("| ---"):
            continue
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        header = cells[0].lower()
        if header == "query":
            continue
        rows.append(
            {
                "query": cells[0],
                "intent": cells[1],
                "symbol": cells[2],
            }
        )
    return rows


class IntentClassifierTests(unittest.TestCase):
    def test_queries_match_expected_intents(self) -> None:
        fixtures = load_fixtures()
        self.assertGreater(len(fixtures), 0, "intent fixtures file is empty")
        for fixture in fixtures:
            with self.subTest(query=fixture["query"]):
                result = quedonde.classify_intent(fixture["query"])
                self.assertEqual(
                    fixture["intent"],
                    result.get("intent"),
                    msg=f"intent mismatch for query '{fixture['query']}'",
                )
                expected_symbol = fixture["symbol"]
                if expected_symbol:
                    self.assertEqual(
                        expected_symbol,
                        result.get("symbol"),
                        msg=f"symbol mismatch for query '{fixture['query']}'",
                    )


if __name__ == "__main__":
    unittest.main()
