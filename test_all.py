"""Aggregate unittest discovery so `python -m unittest` works out of the box."""

from __future__ import annotations

import unittest

from tests import test_intent_classifier


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_intent_classifier))
    return suite


if __name__ == "__main__":
    unittest.main()
