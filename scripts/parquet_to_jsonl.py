#!/usr/bin/env python3
"""Convert parquet datasets to TS-compatible conversation JSONL.

Output format per line:
[
  {"role": "text", "content": "..."}
]

If a conversation column is provided (or detected), and its value contains a
valid conversation array, that array is emitted directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def _load_parquet_batches(input_path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "pyarrow is required for parquet conversion. Install with: pip install pyarrow"
        ) from exc

    parquet_file = pq.ParquetFile(str(input_path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        table = batch.to_pydict()
        if not table:
            continue
        columns = list(table.keys())
        row_count = len(table[columns[0]]) if columns else 0
        rows: list[dict[str, Any]] = []
        for i in range(row_count):
            row = {col: table[col][i] for col in columns}
            rows.append(row)
        if rows:
            yield rows


def _is_valid_turn(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str)


def _is_conversation(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(_is_valid_turn(turn) for turn in value)


def _parse_json_if_string(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        if stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _row_to_conversation(
    row: dict[str, Any],
    text_column: str | None,
    conversation_column: str | None,
    prompt_column: str | None,
    response_column: str | None,
    include_columns: list[str] | None,
) -> list[dict[str, str]]:
    # 1) Explicit conversation column takes precedence when valid.
    if conversation_column and conversation_column in row:
        candidate = _parse_json_if_string(row.get(conversation_column))
        if _is_conversation(candidate):
            return [
                {"role": str(turn["role"]), "content": str(turn["content"])}
                for turn in candidate
            ]

    # 2) Explicit prompt/response columns map to user/assistant turns.
    if (
        prompt_column
        and response_column
        and prompt_column in row
        and response_column in row
    ):
        return [
            {"role": "user", "content": _stringify(row.get(prompt_column))},
            {"role": "assistant", "content": _stringify(row.get(response_column))},
        ]

    # 3) Auto-detect a conversation shape in common columns.
    for key in ("conversation", "messages"):
        if key in row:
            candidate = _parse_json_if_string(row.get(key))
            if _is_conversation(candidate):
                return [
                    {"role": str(turn["role"]), "content": str(turn["content"])}
                    for turn in candidate
                ]

    # 4) Text-column mapping for non-conversational records.
    chosen_text_column: str | None = None
    if text_column and text_column in row:
        chosen_text_column = text_column
    elif "text" in row:
        chosen_text_column = "text"

    if chosen_text_column is not None:
        content = _stringify(row.get(chosen_text_column))
        return [{"role": "text", "content": content}]

    # 5) Fallback: serialize selected columns or the whole row as text payload.
    payload: Any
    if include_columns:
        payload = {col: row.get(col) for col in include_columns}
    else:
        payload = row

    return [{"role": "text", "content": json.dumps(payload, ensure_ascii=False)}]


def _iter_jsonl_lines(
    rows: Iterable[dict[str, Any]],
    text_column: str | None,
    conversation_column: str | None,
    prompt_column: str | None,
    response_column: str | None,
    include_columns: list[str] | None,
) -> Iterator[str]:
    for row in rows:
        conversation = _row_to_conversation(
            row=row,
            text_column=text_column,
            conversation_column=conversation_column,
            prompt_column=prompt_column,
            response_column=response_column,
            include_columns=include_columns,
        )
        yield json.dumps(conversation, ensure_ascii=False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert parquet files to JSONL where each line is a conversation array."
    )
    parser.add_argument("input", help="Input parquet file path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL file path (default: <input_stem>.jsonl next to input)",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column to use as non-conversational text content (default: text)",
    )
    parser.add_argument(
        "--conversation-column",
        default=None,
        help="Column that contains conversation arrays (object/array or JSON string)",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help="Column to use as user prompt content (requires --response-column)",
    )
    parser.add_argument(
        "--response-column",
        default=None,
        help="Column to use as assistant response content (requires --prompt-column)",
    )
    parser.add_argument(
        "--include-columns",
        nargs="*",
        default=None,
        help="When no text column is found, include only these columns in fallback JSON payload",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows to process per parquet batch (default: 5000)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a ZIP archive containing the JSONL output",
    )
    parser.add_argument(
        "--zip-output",
        default=None,
        help="ZIP output path (default: <output>.zip)",
    )
    parser.add_argument(
        "--zip-entry-name",
        default=None,
        help="File name to use inside ZIP (default: JSONL output file name)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if bool(args.prompt_column) != bool(args.response_column):
        print(
            "Both --prompt-column and --response-column must be provided together.",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_file():
        print(f"Input parquet file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output) if args.output else input_path.with_suffix(".jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for batch in _load_parquet_batches(input_path, max(1, args.batch_size)):
            for line in _iter_jsonl_lines(
                batch,
                text_column=args.text_column,
                conversation_column=args.conversation_column,
                prompt_column=args.prompt_column,
                response_column=args.response_column,
                include_columns=args.include_columns,
            ):
                out.write(line)
                out.write("\n")
                rows_written += 1

    print(f"Wrote {rows_written} JSONL records to: {output_path}")

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
