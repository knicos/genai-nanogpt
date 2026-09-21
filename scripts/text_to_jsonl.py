#!/usr/bin/env python3
"""Convert plain text to TS-compatible conversation JSONL.

Input format:
- Blank line separates conversations (each becomes one JSONL record).
- Each non-blank input line is a speaker turn and becomes an object
  with `role: "text"` and `content: <line>`.

Output per line:
[
  {"role": "text", "content": "..."},
  ...
]
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterator, List


def _iter_conversation_blocks(path: Path) -> Iterator[List[str]]:
    """Yield conversation blocks as lists of non-empty lines.

    Consecutive blank lines are treated as a single separator. Leading/trailing
    whitespace on each line is stripped; empty lines are ignored.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        block: List[str] = []
        for raw in f:
            line = raw.rstrip("\n\r")
            if line.strip() == "":
                if block:
                    yield block
                    block = []
                continue
            block.append(line)

        if block:
            yield block


def _block_to_conversation(block: List[str]) -> List[dict[str, str]]:
    return [{"role": "text", "content": line} for line in block]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert plain text conversations to JSONL where each line is a conversation array."
    )
    parser.add_argument("input", help="Input text file path")
    parser.add_argument(
        "-o", "--output", default=None, help="Output JSONL file path (default: <input_stem>.jsonl)"
    )
    parser.add_argument(
        "--zip", action="store_true", help="Also create a ZIP archive containing the JSONL output"
    )
    parser.add_argument(
        "--zip-output", default=None, help="ZIP output path (default: <output>.zip)"
    )
    parser.add_argument(
        "--zip-entry-name",
        default=None,
        help="File name to use inside ZIP (default: JSONL output file name)",
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output) if args.output else input_path.with_suffix(".jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for block in _iter_conversation_blocks(input_path):
            conversation = _block_to_conversation(block)
            out.write(json.dumps(conversation, ensure_ascii=False))
            out.write("\n")
            written += 1

    print(f"Wrote {written} JSONL records to: {output_path}")

    if args.zip:
        zip_path = Path(args.zip_output) if args.zip_output else output_path.with_suffix(output_path.suffix + ".zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        entry_name = args.zip_entry_name if args.zip_entry_name else output_path.name
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_path, arcname=entry_name)
        print(f"Wrote ZIP archive: {zip_path} (entry: {entry_name})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
