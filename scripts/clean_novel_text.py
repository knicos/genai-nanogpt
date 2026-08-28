#!/usr/bin/env python3
"""
clean_novel_text.py

Cleans plain-text novel files that have been manually word-wrapped to a
fixed column width, producing text suitable for LLM training.

What it does:
  - Un-wraps hard-wrapped lines within a paragraph into a single line
  - Preserves paragraph breaks (one or more blank lines -> one blank line)
  - Strips leading/trailing whitespace (spaces/tabs) from every line,
    removing indentation
  - Collapses any accidental runs of multiple spaces left over from joining

Usage:
  Single file:
      python clean_novel_text.py input.txt
      python clean_novel_text.py input.txt output.txt

  Directory of .txt files:
      python clean_novel_text.py novels/ cleaned_novels/
      python clean_novel_text.py novels/          # writes alongside originals
"""

import argparse
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Un-wrap word-wrapped lines while preserving paragraph breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # A paragraph break is one or more blank (or whitespace-only) lines.
    paragraphs = re.split(r"\n[ \t]*\n+", text)

    cleaned_paragraphs = []
    for para in paragraphs:
        lines = para.split("\n")
        # Strip indentation and trailing whitespace from each wrapped line
        lines = [line.strip() for line in lines]
        # Drop any stray empty lines left inside the block
        lines = [line for line in lines if line]
        if not lines:
            continue
        # Join the wrapped lines back into one paragraph line
        joined = " ".join(lines)
        # Collapse any double spaces introduced by the join
        joined = re.sub(r" {2,}", " ", joined)
        cleaned_paragraphs.append(joined)

    return "\n\n".join(cleaned_paragraphs) + "\n"


def process_file(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Clean word-wrapped novel text files for LLM training."
    )
    parser.add_argument("input", help="Input .txt file or directory of .txt files")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file or directory (optional; defaults to *_cleaned)",
    )
    parser.add_argument(
        "--suffix",
        default="_cleaned",
        help="Suffix added to output filenames when no explicit output is given "
        "(default: _cleaned)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path
        txt_files = sorted(input_path.glob("*.txt"))
        if not txt_files:
            raise SystemExit(f"No .txt files found in {input_path}")
        for f in txt_files:
            out = output_dir / f"{f.stem}{args.suffix}{f.suffix}"
            process_file(f, out)
            print(f"Cleaned: {f} -> {out}")
    else:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(
                f"{input_path.stem}{args.suffix}{input_path.suffix}"
            )
        process_file(input_path, output_path)
        print(f"Cleaned: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()