"""Utility script to synthesize a repository for structural benchmarks."""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

TEMPLATE = """def fn_{idx}():
    value = {value}
    return value


def call_{idx}(arg):
    for _ in range({loops}):
        value = fn_{idx}()
    return value + arg

"""


def make_random_string(length: int) -> str:
    rng = random.Random(length)
    return "".join(rng.choice(string.ascii_letters) for _ in range(length))


def generate_repo(root: Path, file_count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    src = root / "src"
    src.mkdir(exist_ok=True)
    for idx in range(file_count):
        body = TEMPLATE.format(
            idx=idx,
            value=repr(make_random_string(20)),
            loops=(idx % 5) + 1,
        )
        path = src / f"module_{idx:04d}.py"
        path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Destination directory")
    parser.add_argument("--files", type=int, default=2000, help="How many files to generate")
    args = parser.parse_args()

    generate_repo(Path(args.root), args.files)


if __name__ == "__main__":
    main()
