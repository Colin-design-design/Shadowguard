"""Command-line interface.

Usage:
    shadowguard "text to scan"
    shadowguard --file prompt.txt
    shadowguard --json "text"
    cat prompts.txt | shadowguard -
"""
from __future__ import annotations

import argparse
import json
import sys

from .guard import ShadowGuard


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="shadowguard",
        description="Phonetic-shadow prompt-injection detector",
    )
    parser.add_argument("text", nargs="?", help="text to scan ('-' for stdin)")
    parser.add_argument("--file", "-f", help="scan contents of a file")
    parser.add_argument("--json", action="store_true", help="output JSON report")
    parser.add_argument("--fast", action="store_true",
                        help="seed lexicon only (fast startup)")
    parser.add_argument("--fuzzy", action="store_true",
                        help="enable Phase-2 fuzzy phoneme matching "
                             "(higher recall, slower, more false positives)")
    parser.add_argument("--limit", type=int, default=20000,
                        help="procedural lexicon word cap (default 20000)")
    args = parser.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text == "-":
        text = sys.stdin.read()
    elif args.text:
        text = args.text
    else:
        parser.error("provide text, --file, or pipe with '-'")

    guard = ShadowGuard(use_procedural=not args.fast, procedural_limit=args.limit,
                        use_fuzzy=args.fuzzy)
    result = guard.scan(text)

    if args.json:
        print(result.to_json())
    else:
        print(result.summary())
        # exit code 1 when risky — scriptable gate for CI / agent loops
        return 0 if result.risk_level in ("clean", "low") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
