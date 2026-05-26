#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import random
from contextlib import nullcontext
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from py_nanogpt.io import load_model_zip, save_model_zip
from py_nanogpt.model import NanoGPTV2, NanoGPTV2Config
from py_nanogpt.tokenizer import BPETokenizer, CharTokenizer, SPECIALS, load_tokenizer, parse_tokens


Conversation = List[Dict[str, str]]


def _set_max_csv_field_size() -> None:
    # Python's csv parser has a default field size cap; increase it for long training records.
    max_size = sys.maxsize
    while max_size > 0:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


def _log(message: str, quiet: bool = False) -> None:
    if quiet:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


class LRScheduler:
    def __init__(
        self,
        learning_rate: float,
        warmup_steps: int,
        decay_epochs: int,
        min_learning_rate: float,
        epoch_steps: int,
    ) -> None:
        self.step = 0
        self.start_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.warmup_steps = max(0, warmup_steps)
        self.decay_epochs = max(1, decay_epochs)
        self.min_learning_rate = min_learning_rate
        self.epoch_steps = max(1, epoch_steps)

    def next_lr(self) -> float:
        step = self.step

        if self.warmup_steps > 0 and step < self.warmup_steps:
            warmup_factor = float(step + 1) / float(self.warmup_steps)
            self.learning_rate = self.start_learning_rate * warmup_factor
            self.step += 1
            return self.learning_rate

        decay_steps = self.epoch_steps * self.decay_epochs
        if step >= decay_steps or decay_steps <= self.warmup_steps:
            self.learning_rate = self.min_learning_rate
            self.step += 1
            return self.learning_rate

        decay_ratio = float(step - self.warmup_steps) / float(decay_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        self.learning_rate = self.min_learning_rate + coeff * (self.start_learning_rate - self.min_learning_rate)
        self.step += 1
        return self.learning_rate


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_grad_scaler(enable_cuda_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enable_cuda_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=enable_cuda_amp)
    return torch.cuda.amp.GradScaler(enabled=enable_cuda_amp)


def _autocast_context(device: torch.device, enable_cuda_amp: bool, autocast_dtype: torch.dtype | None):
    if not enable_cuda_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True)
    return torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True)


def cmd_load(args: argparse.Namespace) -> None:
    model, tokenizer, config, meta = load_model_zip(args.model)
    print(f"Loaded: {args.model}")
    print(f"Model type: {config.get('model_type')}")
    print(f"Vocab size: {config.get('vocab_size')}")
    print(f"Hidden size: {config.get('hidden_size')}")
    print(f"Layers: {config.get('num_hidden_layers')}")
    print(f"Heads: {config.get('num_attention_heads')}")
    print(f"Block size: {config.get('block_size')}")
    print(f"Tokenizer: {type(tokenizer).__name__}")
    print(f"Weight tensors loaded: {len(model.to_weight_dict())}")
    print(f"Meta version: {meta.get('version')}")
    print(f"Meta application: {meta.get('application')}")


def cmd_save(args: argparse.Namespace) -> None:
    model, tokenizer, _, meta = load_model_zip(args.model)
    save_model_zip(args.output, model, tokenizer, meta=meta, name=args.name)
    print(f"Saved: {args.output}")


def _build_batch(tokens: List[int] | torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    token_tensor = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.long)

    max_start = int(token_tensor.numel()) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for this block size")

    starts = torch.randint(0, max_start, (batch_size,), device=token_tensor.device)
    offsets = torch.arange(block_size, device=token_tensor.device)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)

    x = token_tensor[idx]
    y = token_tensor[idx + 1]

    if x.device != device:
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)
    return x, y


def _collect_files(
    text_files: Sequence[str] | None,
    text_dirs: Sequence[str] | None,
    globs: Sequence[str] | None,
) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()

    for f in text_files or []:
        p = Path(f)
        if p.is_file() and p not in seen:
            files.append(p)
            seen.add(p)

    supported_exts = {".txt", ".csv", ".json", ".jsonl"}

    for d in text_dirs or []:
        base = Path(d)
        if not base.exists() or not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in supported_exts and p not in seen:
                files.append(p)
                seen.add(p)

    for pattern in globs or []:
        for p in Path(".").glob(pattern):
            if p.is_file() and p not in seen:
                files.append(p)
                seen.add(p)

    return files


def _check_for_text_column(header: List[str], name: str) -> int:
    lower = name.lower()
    for i, col in enumerate(header):
        if str(col).lower() == lower:
            return i
    return 0


def _check_first_row_is_header(row: List[str]) -> bool:
    return all(len(str(cell)) < 64 for cell in row)


def _is_conversation_obj(obj: Any) -> bool:
    if not isinstance(obj, list) or len(obj) == 0:
        return False
    first = obj[0]
    return isinstance(first, dict) and isinstance(first.get("role"), str) and isinstance(first.get("content"), str)


def _as_text_conversation(obj: Any) -> Conversation:
    if isinstance(obj, str):
        return [{"role": "text", "content": obj}]
    if isinstance(obj, dict) and "text" in obj:
        return [{"role": "text", "content": str(obj["text"])}]
    return [{"role": "text", "content": json.dumps(obj, ensure_ascii=False)}]


def _load_file_conversations(path: Path, args: argparse.Namespace) -> List[Conversation]:
    ext = path.suffix.lower()

    if ext == ".txt":
        return [[{"role": "text", "content": path.read_text(encoding="utf-8")}]]

    if ext == ".csv":
        _set_max_csv_field_size()
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        rows = [r for r in rows if len(r) > 0]
        if not rows:
            return []

        column = _check_for_text_column(rows[0], args.csv_column or "text")
        has_header = args.csv_has_header if args.csv_has_header is not None else _check_first_row_is_header(rows[0])
        filtered = rows[1:] if has_header else rows
        out: List[Conversation] = []
        for row in filtered:
            value = row[column] if column < len(row) else ""
            out.append([{"role": "text", "content": value}])
        return out

    if ext == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}")
        return [_as_text_conversation(item) for item in payload]

    if ext == ".jsonl":
        out: List[Conversation] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if _is_conversation_obj(obj):
                    conv = []
                    for frag in obj:
                        role = str(frag.get("role", "text"))
                        content = str(frag.get("content", ""))
                        conv.append({"role": role, "content": content})
                    out.append(conv)
                else:
                    out.append(_as_text_conversation(obj))
            except Exception:
                out.append([{"role": "text", "content": raw}])
        return out

    raise ValueError(f"Unsupported file type: {path}")


def _load_training_conversations(args: argparse.Namespace) -> List[Conversation]:
    files = _collect_files(args.text_files, args.text_dirs, args.globs)
    explicit = [Path(args.text_file)] if args.text_file else []
    sources = explicit + files

    if len(sources) == 0:
        raise ValueError("No training data found. Use --text-file and/or --text-files/--text-dirs/--globs")

    _log(f"Loading training data from {len(sources)} source(s)", args.quiet)

    all_conversations: List[Conversation] = []
    for idx, p in enumerate(sources, start=1):
        convs = _load_file_conversations(p, args)
        all_conversations.extend(convs)
        _log(
            f"Loaded [{idx}/{len(sources)}] {p} ({len(convs):,} conversation records)",
            args.quiet,
        )

    _log(f"Total conversation records: {len(all_conversations):,}", args.quiet)
    return all_conversations


def _conversations_to_corpus_text(conversations: List[Conversation]) -> str:
    chunks: List[str] = []
    for conv in conversations:
        for frag in conv:
            chunks.append(frag.get("content", ""))
    return "\n\n".join(chunks)


def _special_token_index(tokenizer: CharTokenizer | BPETokenizer, token: str) -> int | None:
    if isinstance(tokenizer, CharTokenizer):
        return tokenizer.special_tokens.get(token)
    if isinstance(tokenizer, BPETokenizer):
        return tokenizer.vocab_index.get(token)
    return None


def _encode_conversation(tokenizer: CharTokenizer | BPETokenizer, conversation: Conversation) -> List[int]:
    bos = int(getattr(tokenizer, "bos_token", 0))
    eos = int(getattr(tokenizer, "eos_token", 0))
    tokens: List[int] = [bos]

    role_to_start = {
        "user": _special_token_index(tokenizer, "<|user_start|>"),
        "assistant": _special_token_index(tokenizer, "<|assistant_start|>"),
        "system": _special_token_index(tokenizer, "<|system_start|>"),
    }
    role_to_end = {
        "user": _special_token_index(tokenizer, "<|user_end|>"),
        "assistant": _special_token_index(tokenizer, "<|assistant_end|>"),
        "system": _special_token_index(tokenizer, "<|system_end|>"),
    }

    for frag in conversation:
        role = str(frag.get("role", "text"))
        content = str(frag.get("content", ""))

        if role in role_to_start and role_to_start[role] is not None and role_to_end[role] is not None:
            tokens.append(int(role_to_start[role]))
            tokens.extend(tokenizer.encode(content))
            tokens.append(int(role_to_end[role]))
        else:
            # text or unknown role: plain content tokens, matching TS behavior for role='text'.
            tokens.extend(tokenizer.encode(content))

    tokens.append(eos)
    return tokens


def _load_vocab_payload(path: Path) -> object:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _load_tokenizer_file(path: Path) -> CharTokenizer | BPETokenizer:
    spec = json.loads(path.read_text(encoding="utf-8"))
    tok = load_tokenizer(spec)
    if isinstance(tok, (CharTokenizer, BPETokenizer)):
        return tok
    raise ValueError(f"Unsupported tokenizer file type in {path}")


def _save_token_ids(path: Path, token_ids: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "genai_nanogpt_token_ids_v1",
        "dtype": "int32",
        "num_tokens": len(token_ids),
        "tokens": torch.tensor(token_ids, dtype=torch.int32),
    }
    torch.save(payload, path)


def _load_token_ids(path: Path) -> List[int]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "tokens" in payload:
        tokens = payload["tokens"]
    else:
        tokens = payload

    if isinstance(tokens, torch.Tensor):
        tokens = tokens.to(dtype=torch.int64, device="cpu").tolist()
    if not isinstance(tokens, list):
        raise ValueError(f"Unsupported token ids payload in {path}")
    return [int(t) for t in tokens]


def _save_training_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}") if path.suffix else Path(str(path) + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_training_state(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported resume payload in {path}")
    return payload


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def _values_match(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-12
        except Exception:
            return False
    return a == b


def _collect_resume_mismatches(saved: Any, current: Any, prefix: str = "") -> List[str]:
    mismatches: List[str] = []
    if isinstance(saved, dict) and isinstance(current, dict):
        keys = sorted(set(saved.keys()) | set(current.keys()))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved:
                mismatches.append(f"{path}: missing in resume sidecar")
                continue
            if key not in current:
                mismatches.append(f"{path}: unknown in current run")
                continue
            mismatches.extend(_collect_resume_mismatches(saved[key], current[key], path))
        return mismatches

    if isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            mismatches.append(f"{prefix}: list length mismatch ({len(saved)} != {len(current)})")
            return mismatches
        for i, (sv, cv) in enumerate(zip(saved, current)):
            item_path = f"{prefix}[{i}]"
            mismatches.extend(_collect_resume_mismatches(sv, cv, item_path))
        return mismatches

    if not _values_match(saved, current):
        mismatches.append(f"{prefix}: resume={saved!r}, current={current!r}")
    return mismatches


def _build_resume_fingerprint(
    args: argparse.Namespace,
    model: NanoGPTV2,
    tokenizer: CharTokenizer | BPETokenizer,
    *,
    effective_block_size: int,
    epoch_steps: int,
    chosen_optimizer_impl: str,
    selected_mp_dtype: str,
    enable_cuda_amp: bool,
) -> Dict[str, Any]:
    return {
        "model_config": model.config.to_transformers_config(),
        "tokenizer": {
            "vocab_size": len(tokenizer.get_vocab()),
            "type": "char" if isinstance(tokenizer, CharTokenizer) else "bpe",
        },
        "training": {
            "effective_block_size": int(effective_block_size),
            "batch_size": int(args.batch_size),
            "validation_split": float(args.validation_split),
            "context_scaling": float(args.context_scaling),
            "optimizer_impl": str(chosen_optimizer_impl),
            "attention_backend": str(args.attention_backend),
            "mixed_precision": bool(enable_cuda_amp),
            "mixed_precision_dtype": str(selected_mp_dtype),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "trainable_weights": sorted(str(p) for p in (args.trainable_weights or [])),
            "learning_rate": float(args.learning_rate),
            "min_learning_rate": float(args.min_learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "decay_epochs": int(args.decay_epochs),
            "epoch_steps": int(epoch_steps),
            "beta1": float(args.beta1),
            "beta2": float(args.beta2),
            "epsilon": float(args.epsilon),
            "weight_decay": float(args.weight_decay),
            "label_smoothing": float(args.label_smoothing),
            "dropout": float(args.dropout),
            "layer_drop": float(args.layer_drop),
            "clip_norm": float(args.clip_norm) if args.clip_norm is not None else None,
        },
    }


def _extract_vocab_from_payload(payload: object) -> List[str]:
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        if "vocab" in payload and isinstance(payload["vocab"], list):
            return [str(x) for x in payload["vocab"]]
    raise ValueError("Tokenizer vocab payload must be a JSON list or object with a vocab field")


def _extract_merges_from_payload(payload: object) -> List[List[str]]:
    if isinstance(payload, dict) and isinstance(payload.get("merges"), list):
        return [list(m) for m in payload["merges"] if isinstance(m, (list, tuple)) and len(m) == 2]
    if isinstance(payload, list):
        return [list(m) for m in payload if isinstance(m, (list, tuple)) and len(m) == 2]
    return []


def _build_char_vocab_from_text(text: str, vocab_size: int) -> List[str]:
    if vocab_size < len(SPECIALS):
        raise ValueError(f"vocab-size must be >= {len(SPECIALS)} for char tokenizer")

    counts = Counter(text)
    # Most common first, stable fallback by char code for deterministic ordering.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    base = list(SPECIALS)
    capacity = vocab_size - len(base)
    for ch, _ in ranked:
        if ch in base:
            continue
        base.append(ch)
        if len(base) >= len(SPECIALS) + capacity:
            break

    # Fill any remaining slots with empty tokens.
    while len(base) < vocab_size:
        base.append("")

    return base


def _build_bpe_from_text(
    text: str,
    vocab_size: int,
    quiet: bool = False,
    progress_interval: int = 100,
) -> tuple[List[str], List[List[str]], Dict[str, List[str]]]:
    if vocab_size < len(SPECIALS):
        raise ValueError(f"vocab-size must be >= {len(SPECIALS)} for bpe tokenizer")

    _log("BPE build: parsing pretokens from text", quiet)
    pretokens = parse_tokens(text)
    if not pretokens:
        # Empty corpus fallback.
        _log("BPE build: empty corpus, using special tokens only", quiet)
        return list(SPECIALS), [], {}

    _log(f"BPE build: parsed {len(pretokens):,} pretokens", quiet)
    unique_pretokens = list(dict.fromkeys(pretokens))
    _log(f"BPE build: {len(unique_pretokens):,} unique pretokens", quiet)

    vocab: List[str] = []
    vocab_seen: set[str] = set()

    def add_vocab(tok: str) -> None:
        if tok not in vocab_seen:
            vocab_seen.add(tok)
            vocab.append(tok)

    for s in SPECIALS:
        add_vocab(s)

    token_sequences: List[List[str]] = []
    total_unique = len(unique_pretokens)
    for idx, token in enumerate(unique_pretokens, start=1):
        chars = list(token)
        token_sequences.append(chars)
        for ch in chars:
            add_vocab(ch)
        if idx % max(1, progress_interval * 10) == 0 or idx == total_unique:
            pct = (idx / total_unique) * 100.0
            _log(
                f"BPE build: initialized base token sequences {idx:,}/{total_unique:,} ({pct:.1f}%)",
                quiet,
            )

    # If character inventory already exceeds target, keep most frequent chars (TS behavior fallback).
    if len(vocab) >= vocab_size:
        char_counts = Counter(ch for token in pretokens for ch in token)
        chars_by_freq = [ch for ch, _ in sorted(char_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        out = list(SPECIALS)
        for ch in chars_by_freq:
            if ch in out:
                continue
            out.append(ch)
            if len(out) >= vocab_size:
                break
        _log(
            f"BPE build: character inventory exceeded target, using top {len(out):,} tokens without merges",
            quiet,
        )
        return out, [], {}

    merges: List[List[str]] = []

    # TS-style incremental pair statistics.
    pair_counts: Dict[tuple[str, str], int] = {}
    pair_instances: Dict[tuple[str, str], set[int]] = {}

    def _add_pair(pair: tuple[str, str], instance_idx: int, delta: int) -> None:
        old = pair_counts.get(pair, 0)
        new = old + delta
        if new <= 0:
            pair_counts.pop(pair, None)
            pair_instances.pop(pair, None)
            return

        pair_counts[pair] = new
        instances = pair_instances.get(pair)
        if instances is None:
            instances = set()
            pair_instances[pair] = instances
        if delta > 0:
            instances.add(instance_idx)
        else:
            instances.discard(instance_idx)

    for j, seq in enumerate(token_sequences):
        for i in range(len(seq) - 1):
            _add_pair((seq[i], seq[i + 1]), j, 1)

    def _best_pair() -> tuple[str, str] | None:
        best = None
        best_count = 0
        for pair, cnt in pair_counts.items():
            if cnt > best_count:
                best = pair
                best_count = cnt
        return best

    while len(vocab) < vocab_size and len(merges) < vocab_size:
        pair = _best_pair()
        if pair is None:
            break

        a, b = pair
        merged_token = a + b
        merges.append([a, b])
        add_vocab(merged_token)

        instances = list(pair_instances.get(pair, set()))
        for idx in instances:
            seq = token_sequences[idx]
            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(merged_token)

                    if i > 0:
                        _add_pair((seq[i - 1], a), idx, -1)
                        _add_pair((seq[i - 1], merged_token), idx, 1)

                    i += 2

                    if i < len(seq):
                        _add_pair((b, seq[i]), idx, -1)
                        _add_pair((merged_token, seq[i]), idx, 1)
                else:
                    new_seq.append(seq[i])
                    i += 1

            token_sequences[idx] = new_seq

        # Remove merged pair from candidate set, same as TS.
        pair_counts.pop(pair, None)
        pair_instances.pop(pair, None)

        merge_count = len(merges)
        if merge_count % max(1, progress_interval) == 0 or len(vocab) >= vocab_size:
            pct = (len(vocab) / max(1, vocab_size)) * 100.0
            _log(
                f"BPE build: merges={merge_count:,}, vocab={len(vocab):,}/{vocab_size:,} ({pct:.1f}%)",
                quiet,
            )

    _log(f"BPE build complete: vocab={len(vocab):,}, merges={len(merges):,}", quiet)
    learned_map: Dict[str, List[str]] = {
        unique_pretokens[i]: token_sequences[i] for i in range(len(unique_pretokens))
    }
    return vocab, merges, learned_map


def _encode_conversations_with_progress(
    tokenizer: CharTokenizer | BPETokenizer,
    conversations: List[Conversation],
    quiet: bool,
    progress_interval: int,
) -> List[int]:
    _log("Tokenizing conversation records", quiet)
    total = len(conversations)
    token_ids: List[int] = []
    if total == 0:
        _log("Tokenization complete: no conversation records found", quiet)
        return token_ids

    for i, conv in enumerate(conversations, start=1):
        token_ids.extend(_encode_conversation(tokenizer, conv))
        if i % max(1, progress_interval) == 0 or i == total:
            pct = (i / total) * 100.0
            _log(
                f"Tokenizing records: {i:,}/{total:,} ({pct:.1f}%), ids={len(token_ids):,}",
                quiet,
            )

    _log(f"Tokenization complete: produced {len(token_ids):,} token ids", quiet)
    return token_ids


def _create_model_and_tokenizer_from_scratch(args: argparse.Namespace, text: str):
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be > 0")
    if args.n_layer <= 0 or args.n_head <= 0 or args.n_embed <= 0 or args.block_size <= 0:
        raise ValueError("--n-layer, --n-head, --n-embed, and --block-size must be > 0")
    if args.n_embed % args.n_head != 0:
        raise ValueError("--n-embed must be divisible by --n-head")

    tokenizer_type = args.tokenizer_type.lower()
    if tokenizer_type not in {"char", "bpe"}:
        raise ValueError("--tokenizer-type must be char or bpe")

    vocab_payload = None
    merges_payload = None
    if args.tokenizer_vocab_file:
        vocab_payload = _load_vocab_payload(Path(args.tokenizer_vocab_file))
    if args.tokenizer_merges_file:
        merges_payload = _load_vocab_payload(Path(args.tokenizer_merges_file))

    if tokenizer_type == "char":
        if vocab_payload is not None:
            vocab = _extract_vocab_from_payload(vocab_payload)
            _log(f"Using provided char vocab with {len(vocab):,} tokens", args.quiet)
        else:
            _log("Building char vocab from training text", args.quiet)
            vocab = _build_char_vocab_from_text(text, args.vocab_size)
            _log(f"Built char vocab with {len(vocab):,} tokens", args.quiet)
        tokenizer = CharTokenizer(vocab)
        vocab_size = len(vocab)
    else:
        if vocab_payload is None:
            _log("No BPE vocab file provided; building BPE vocab+merges from training text", args.quiet)
            vocab, merges, learned_map = _build_bpe_from_text(
                text,
                args.vocab_size,
                quiet=args.quiet,
                progress_interval=args.progress_interval,
            )
        else:
            vocab = _extract_vocab_from_payload(vocab_payload)
            merges = _extract_merges_from_payload(vocab_payload)
            learned_map = {}
            _log(
                f"Using provided BPE vocab ({len(vocab):,}) and merges ({len(merges):,})",
                args.quiet,
            )
        if merges_payload is not None:
            merges = _extract_merges_from_payload(merges_payload)
            _log(f"Using external merges file with {len(merges):,} merges", args.quiet)
        tokenizer = BPETokenizer(vocab, merges)
        if learned_map:
            tokenizer.pretoken_map.update(learned_map)
        vocab_size = len(vocab)

    config = NanoGPTV2Config(
        model_type="GenAI_NanoGPT_v2",
        vocab_size=vocab_size,
        hidden_size=args.n_embed,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        block_size=args.block_size,
        mlpFactor=args.mlp_factor,
        windowSize=args.window_size,
    )
    model = NanoGPTV2(config)
    meta = {
        "version": 2,
        "application": "@genai-fi/nanogpt",
        "phase": "untrained",
    }
    return model, tokenizer, meta


def _build_tokenizer_from_args(args: argparse.Namespace, text: str) -> CharTokenizer | BPETokenizer:
    tokenizer_type = args.tokenizer_type.lower()
    if tokenizer_type not in {"char", "bpe"}:
        raise ValueError("--tokenizer-type must be char or bpe")

    vocab_payload = None
    merges_payload = None
    if args.tokenizer_vocab_file:
        vocab_payload = _load_vocab_payload(Path(args.tokenizer_vocab_file))
    if args.tokenizer_merges_file:
        merges_payload = _load_vocab_payload(Path(args.tokenizer_merges_file))

    if tokenizer_type == "char":
        if vocab_payload is not None:
            vocab = _extract_vocab_from_payload(vocab_payload)
            _log(f"Using provided char vocab with {len(vocab):,} tokens", args.quiet)
        else:
            _log("Building char vocab from training text", args.quiet)
            vocab = _build_char_vocab_from_text(text, args.vocab_size)
            _log(f"Built char vocab with {len(vocab):,} tokens", args.quiet)
        return CharTokenizer(vocab)

    if vocab_payload is None:
        _log("No BPE vocab file provided; building BPE vocab+merges from training text", args.quiet)
        vocab, merges, learned_map = _build_bpe_from_text(
            text,
            args.vocab_size,
            quiet=args.quiet,
            progress_interval=args.progress_interval,
        )
    else:
        vocab = _extract_vocab_from_payload(vocab_payload)
        merges = _extract_merges_from_payload(vocab_payload)
        learned_map = {}
        _log(
            f"Using provided BPE vocab ({len(vocab):,}) and merges ({len(merges):,})",
            args.quiet,
        )
    if merges_payload is not None:
        merges = _extract_merges_from_payload(merges_payload)
        _log(f"Using external merges file with {len(merges):,} merges", args.quiet)
    tok = BPETokenizer(vocab, merges)
    if learned_map:
        tok.pretoken_map.update(learned_map)
    return tok


def cmd_prepare_data(args: argparse.Namespace) -> None:
    conversations = _load_training_conversations(args)
    text = _conversations_to_corpus_text(conversations)

    if args.model:
        _log(f"Loading tokenizer from base model: {args.model}", args.quiet)
        _, tokenizer, _, _ = load_model_zip(args.model, require_weights=False, strict_weights=False)
        _log(f"Using tokenizer from base model with vocab size {len(tokenizer.get_vocab()):,}", args.quiet)
    else:
        tokenizer = _build_tokenizer_from_args(args, text)

    token_ids = _encode_conversations_with_progress(
        tokenizer,
        conversations,
        quiet=args.quiet,
        progress_interval=args.progress_interval,
    )

    tok_type = "char" if isinstance(tokenizer, CharTokenizer) else "bpe"
    tokeniser_payload = {
        "type": tok_type,
        "vocab": tokenizer.get_vocab(),
        "merges": tokenizer.get_merges(),
    }

    tokenizer_path = Path(args.output_tokenizer_file) if args.output_tokenizer_file else None
    tokens_path = Path(args.output_token_ids_file)

    if tokenizer_path is not None:
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text(json.dumps(tokeniser_payload, indent=4, ensure_ascii=False), encoding="utf-8")
    elif not args.model:
        raise ValueError("--output-tokenizer-file is required unless --model is provided")

    _save_token_ids(tokens_path, token_ids)

    if tokenizer_path is not None:
        _log(
            f"Prepared data saved: tokenizer={tokenizer_path} tokens={tokens_path} num_tokens={len(token_ids):,}",
            args.quiet,
        )
    else:
        _log(
            f"Prepared data saved: tokens={tokens_path} num_tokens={len(token_ids):,} (tokenizer reused from base model)",
            args.quiet,
        )


def _split_train_validation(
    tokens: List[int],
    block_size: int,
    validation_split: float,
    forced_val_pages: Sequence[int] | None = None,
) -> tuple[List[int], List[int], Dict[str, Any]]:
    if validation_split <= 0.0:
        return tokens, [], {
            "validation_split": float(validation_split),
            "page_size": int(block_size * 8),
            "total_pages": 0,
            "val_pages": [],
            "split_rng_state_before": None,
        }

    page_size = block_size * 8
    total_pages = max(1, len(tokens) // max(page_size, 1))
    val_pages = max(1, int(total_pages * validation_split))

    split_rng_state_before = torch.get_rng_state().cpu()
    if forced_val_pages is not None:
        val_set = {int(v) for v in forced_val_pages if 0 <= int(v) < total_pages}
        if len(val_set) == 0:
            perm = torch.randperm(total_pages).tolist()
            val_set = set(perm[:val_pages])
        split_rng_state_before = None
    else:
        perm = torch.randperm(total_pages).tolist()
        val_set = set(perm[:val_pages])

    train_tokens: List[int] = []
    val_tokens: List[int] = []
    for i, t in enumerate(tokens):
        page = i // page_size
        if page in val_set:
            val_tokens.append(t)
        else:
            train_tokens.append(t)

    split_info = {
        "validation_split": float(validation_split),
        "page_size": int(page_size),
        "total_pages": int(total_pages),
        "val_pages": sorted(int(v) for v in val_set),
        "split_rng_state_before": split_rng_state_before,
    }
    return train_tokens, val_tokens, split_info


def _set_trainable_weights(model: torch.nn.Module, patterns: Sequence[str] | None) -> None:
    if not patterns:
        for p in model.parameters():
            p.requires_grad = True
        return

    for name, p in model.named_parameters():
        p.requires_grad = any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _evaluate(
    model: torch.nn.Module,
    tokens: List[int] | torch.Tensor,
    block_size: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> Dict[str, float]:
    token_count = int(tokens.numel()) if isinstance(tokens, torch.Tensor) else len(tokens)
    if token_count < block_size + 2:
        return {"loss": float("nan"), "accuracy": float("nan"), "perplexity": float("nan")}

    model.eval()
    losses: List[float] = []
    accuracies: List[float] = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = _build_batch(tokens, block_size, batch_size, device)
            logits = model(x, training=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean()
            losses.append(float(loss.item()))
            accuracies.append(float(acc.item()))

    mean_loss = float(sum(losses) / max(len(losses), 1))
    mean_acc = float(sum(accuracies) / max(len(accuracies), 1))
    ppl = float(math.exp(min(20.0, mean_loss)))
    return {"loss": mean_loss, "accuracy": mean_acc, "perplexity": ppl}


def _start_stdin_command_listener(control: Dict[str, bool], quiet: bool) -> threading.Thread | None:
    if not sys.stdin or not sys.stdin.isatty():
        return None

    def _worker() -> None:
        while not control.get("stop_requested", False):
            try:
                line = sys.stdin.readline()
            except Exception:
                return

            if not line:
                return

            cmd = line.strip().lower()
            if cmd in {"s", "save"}:
                control["save_requested"] = True
                _log("Manual save requested from stdin command", quiet)
            elif cmd in {"q", "quit", "exit", "stop"}:
                control["save_requested"] = True
                control["stop_requested"] = True
                _log("Stop requested from stdin command; saving after current step", quiet)

    thread = threading.Thread(target=_worker, name="nanogpt-stdin-commands", daemon=True)
    thread.start()
    return thread


def cmd_train(args: argparse.Namespace) -> None:
    device = pick_device(args.cpu)
    _log(f"Using device: {device}", args.quiet)
    if args.cpu:
        _log("CPU mode forced via config/CLI (--cpu)", args.quiet)
    elif device.type != "cuda":
        _log("CUDA not available to torch (torch.cuda.is_available() == False); running on CPU", args.quiet)
    else:
        try:
            _log(f"CUDA device: {torch.cuda.get_device_name(0)}", args.quiet)
        except Exception:
            _log("CUDA device available", args.quiet)

    if device.type == "cuda":
        if args.enable_tf32:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            _log("TF32 enabled for CUDA matmuls", args.quiet)
        else:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False
            _log("TF32 disabled", args.quiet)
    text = None
    conversations: List[Conversation] | None = None
    tokenizer_for_data: CharTokenizer | BPETokenizer | None = None
    if args.token_ids_file:
        _log(f"Using pre-tokenized ids from {args.token_ids_file}", args.quiet)
        tokens = _load_token_ids(Path(args.token_ids_file))
        _log(f"Loaded {len(tokens):,} token ids", args.quiet)
        if args.tokenizer_file:
            tokenizer_for_data = _load_tokenizer_file(Path(args.tokenizer_file))
            _log(f"Loaded tokenizer from {args.tokenizer_file}", args.quiet)
    else:
        conversations = _load_training_conversations(args)
        text = _conversations_to_corpus_text(conversations)

    if args.model:
        _log(f"Loading model from {args.model}", args.quiet)
        model, tokenizer, _, meta = load_model_zip(args.model, require_weights=False, strict_weights=False)
        _log("Loaded model and tokenizer from zip", args.quiet)
    elif args.init_from_scratch:
        if text is None:
            raise ValueError("--init-from-scratch requires text inputs for tokenizer/model initialization")
        model, tokenizer, meta = _create_model_and_tokenizer_from_scratch(args, text)
        print(
            f"Initialized new model: layers={model.config.num_hidden_layers} heads={model.config.num_attention_heads} "
            f"embed={model.config.hidden_size} block={model.config.block_size} vocab={model.config.vocab_size}"
        )
    else:
        raise ValueError("Provide --model or set --init-from-scratch")

    model.to(device)
    model.set_attention_backend(args.attention_backend)
    _log(f"Attention backend: {args.attention_backend}", args.quiet)

    effective_layer_drop = float(args.layer_drop)
    train_model: torch.nn.Module = model
    if args.compile_model and device.type != "cuda":
        _log("Disabling torch.compile on CPU (expected to be slower for this workload)", args.quiet)
    elif args.compile_model:
        if effective_layer_drop > 0.0:
            _log(
                "Disabling layer_drop while torch.compile is enabled (avoids Dynamo tracing issues with stochastic control flow)",
                args.quiet,
            )
            effective_layer_drop = 0.0
        if hasattr(torch, "compile"):
            try:
                train_model = torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=False)
                _log(f"Enabled torch.compile (mode={args.compile_mode})", args.quiet)
            except Exception as exc:
                _log(f"torch.compile unavailable/failing, falling back to eager mode: {exc}", args.quiet)
                train_model = model
        else:
            _log("torch.compile not available in this torch build; using eager mode", args.quiet)

    train_model.train()

    if args.sft_mode != "full":
        print("warning: --sft-mode is TS compatibility only and is ignored in Python pretraining")
    if args.masked_loss:
        print("warning: --masked-loss is TS compatibility only and is ignored in Python pretraining")
    if args.lora_name or args.lora_config:
        print("warning: --lora-name/--lora-config are TS compatibility only and are ignored in Python pretraining")

    if args.token_ids_file:
        if tokenizer_for_data is not None and len(tokenizer_for_data.get_vocab()) != model.config.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({len(tokenizer_for_data.get_vocab())}) does not match model vocab size ({model.config.vocab_size})"
            )
    else:
        assert text is not None
        assert conversations is not None
        tokens = _encode_conversations_with_progress(
            tokenizer,
            conversations,
            quiet=args.quiet,
            progress_interval=args.progress_interval,
        )
    effective_block_size = max(2, int(model.config.block_size * max(0.05, min(1.0, args.context_scaling))))
    _log(f"Effective block size after context scaling: {effective_block_size}", args.quiet)
    if len(tokens) < effective_block_size + 2:
        raise ValueError("Training text is too short for the model block size")

    resume_path = Path(args.resume_state_file) if args.resume_state_file else None
    resume_payload: Dict[str, Any] | None = None
    resume_training: Dict[str, Any] | None = None
    forced_val_pages: Sequence[int] | None = None
    if resume_path is not None and resume_path.is_file():
        _log(f"Resume state found: {resume_path}", args.quiet)
        resume_payload = _load_training_state(resume_path)
        training_obj = resume_payload.get("training")
        if isinstance(training_obj, dict):
            resume_training = training_obj
            split_obj = training_obj.get("data_split")
            if isinstance(split_obj, dict) and isinstance(split_obj.get("val_pages"), list):
                forced_val_pages = [int(v) for v in split_obj["val_pages"]]
        expected_model = resume_payload.get("model_output")
        if isinstance(expected_model, str) and expected_model and expected_model != str(args.output):
            _log(
                f"warning: resume sidecar model_output={expected_model} differs from --output={args.output}",
                args.quiet,
            )
    elif resume_path is not None:
        _log(f"Resume state path not found, starting fresh: {resume_path}", args.quiet)

    train_tokens, val_tokens, split_info = _split_train_validation(
        tokens,
        effective_block_size,
        args.validation_split,
        forced_val_pages=forced_val_pages,
    )
    train_tokens_tensor = torch.tensor(train_tokens, dtype=torch.long, device=device)
    val_tokens_tensor = (
        torch.tensor(val_tokens, dtype=torch.long, device=device)
        if val_tokens
        else torch.empty(0, dtype=torch.long, device=device)
    )
    train_token_count = int(train_tokens_tensor.numel())
    val_token_count = int(val_tokens_tensor.numel())

    # Free host-side copies early when datasets are resident on the training device.
    del train_tokens
    del val_tokens
    del tokens

    _set_trainable_weights(model, args.trainable_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters after applying --trainable-weights")

    optimizer_impl = str(args.optimizer_impl).lower()
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "eps": args.epsilon,
        "weight_decay": args.weight_decay,
    }

    optimizer: torch.optim.Optimizer | None = None
    chosen_optimizer_impl = "default"

    if device.type == "cuda" and optimizer_impl in {"auto", "fused"}:
        try:
            optimizer = torch.optim.AdamW(trainable_params, fused=True, **optimizer_kwargs)
            chosen_optimizer_impl = "fused"
        except TypeError:
            optimizer = None

    if optimizer is None and optimizer_impl in {"auto", "foreach"}:
        try:
            optimizer = torch.optim.AdamW(trainable_params, foreach=True, **optimizer_kwargs)
            chosen_optimizer_impl = "foreach"
        except TypeError:
            optimizer = None

    if optimizer is None:
        optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)
        chosen_optimizer_impl = "default"

    _log(f"Optimizer backend: {chosen_optimizer_impl}", args.quiet)

    epoch_steps = args.epoch_steps
    if epoch_steps <= 0:
        epoch_steps = max(1, int(train_tokens_tensor.numel()) // max(effective_block_size * args.batch_size, 1))

    total_steps: int | None = None
    limit_mode = "unbounded"
    if args.steps > 0:
        total_steps = args.steps
        limit_mode = "steps"
    elif args.max_epochs > 0:
        total_steps = args.max_epochs * epoch_steps
        limit_mode = "max_epochs"

    steps_label = "until terminated" if total_steps is None else f"{total_steps:,}"
    _log(
        f"Training setup: train_tokens={train_token_count:,}, val_tokens={val_token_count:,}, steps={steps_label}, limit_mode={limit_mode}, epoch_steps={epoch_steps:,}, data_device={device}",
        args.quiet,
    )
    scheduler = LRScheduler(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_epochs=args.decay_epochs,
        min_learning_rate=args.min_learning_rate,
        epoch_steps=epoch_steps,
    )

    enable_cuda_amp = bool(args.mixed_precision and device.type == "cuda")
    if args.mixed_precision and device.type != "cuda":
        _log("Mixed precision requested but CUDA is not active; running in full precision", args.quiet)

    selected_mp_dtype = "none"
    autocast_dtype: torch.dtype | None = None
    use_grad_scaler = False
    if enable_cuda_amp:
        req_dtype = str(args.mixed_precision_dtype).lower()
        bf16_supported = bool(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        if req_dtype == "auto":
            req_dtype = "bf16" if bf16_supported else "fp16"
        if req_dtype == "bf16" and not bf16_supported:
            _log("bf16 requested but not supported on this CUDA device; falling back to fp16", args.quiet)
            req_dtype = "fp16"

        if req_dtype == "bf16":
            autocast_dtype = torch.bfloat16
            selected_mp_dtype = "bf16"
            use_grad_scaler = False
        else:
            autocast_dtype = torch.float16
            selected_mp_dtype = "fp16"
            use_grad_scaler = True

    _log(f"Mixed precision mode: {selected_mp_dtype}", args.quiet)

    scaler = _make_grad_scaler(use_grad_scaler)
    manual_loss_scale = max(1.0, args.loss_scaling)
    if use_grad_scaler:
        if args.loss_scaling != 1.0:
            _log("Ignoring --loss-scaling while fp16 GradScaler is active", args.quiet)
        manual_loss_scale = 1.0
    elif enable_cuda_amp and selected_mp_dtype == "bf16" and args.loss_scaling != 1.0:
        _log("Ignoring --loss-scaling for bf16 mixed precision", args.quiet)
        manual_loss_scale = 1.0

    resume_fingerprint = _build_resume_fingerprint(
        args,
        model,
        tokenizer,
        effective_block_size=effective_block_size,
        epoch_steps=epoch_steps,
        chosen_optimizer_impl=chosen_optimizer_impl,
        selected_mp_dtype=selected_mp_dtype,
        enable_cuda_amp=enable_cuda_amp,
    )

    if resume_payload is not None:
        saved_fingerprint = resume_payload.get("fingerprint")
        if not isinstance(saved_fingerprint, dict):
            raise ValueError(
                "Resume sidecar is missing compatibility fingerprint. "
                "Please start fresh without --resume-state-file, or regenerate the sidecar with the current trainer."
            )
        mismatches = _collect_resume_mismatches(saved_fingerprint, resume_fingerprint)
        if mismatches:
            details = "\n".join(f" - {line}" for line in mismatches[:20])
            more = len(mismatches) - 20
            if more > 0:
                details += f"\n - ... and {more} more mismatch(es)"
            raise ValueError(
                "Resume sidecar does not match current model/hyperparameters. "
                "Refusing to restore training state.\n"
                f"{details}"
            )

    resumed_steps = 0
    if resume_training is not None:
        opt_state = resume_training.get("optimizer_state_dict")
        if isinstance(opt_state, dict):
            optimizer.load_state_dict(opt_state)
            _move_optimizer_state_to_device(optimizer, device)
            _log("Restored optimizer state from resume sidecar", args.quiet)

        sched_state = resume_training.get("scheduler")
        if isinstance(sched_state, dict):
            sched_step = sched_state.get("step")
            sched_lr = sched_state.get("learning_rate")
            if isinstance(sched_step, int):
                scheduler.step = int(sched_step)
            if isinstance(sched_lr, (int, float)):
                scheduler.learning_rate = float(sched_lr)

        scaler_state = resume_training.get("scaler_state_dict")
        if use_grad_scaler and isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
            _log("Restored GradScaler state from resume sidecar", args.quiet)

        restored_best = resume_training.get("best_checkpoint_val_loss")
        if isinstance(restored_best, (int, float)):
            meta["best_checkpoint_val_loss"] = float(restored_best)

        rng_state = resume_training.get("rng_state")
        if isinstance(rng_state, dict):
            torch_state = rng_state.get("torch")
            if isinstance(torch_state, torch.Tensor):
                torch.set_rng_state(torch_state)
            cuda_states = rng_state.get("cuda")
            if device.type == "cuda" and isinstance(cuda_states, list) and len(cuda_states) > 0:
                try:
                    torch.cuda.set_rng_state_all([s for s in cuda_states if isinstance(s, torch.Tensor)])
                except Exception as exc:
                    _log(f"warning: failed to restore CUDA RNG state: {exc}", args.quiet)
            py_state = rng_state.get("python_random")
            if isinstance(py_state, tuple):
                random.setstate(py_state)

        restored_steps = resume_training.get("completed_steps", 0)
        if isinstance(restored_steps, int) and restored_steps > 0:
            resumed_steps = int(restored_steps)
            _log(f"Resuming training from step {resumed_steps}", args.quiet)

    metrics = set((args.metrics or "").split(",")) if isinstance(args.metrics, str) else set()
    metrics = {m.strip() for m in metrics if m.strip()}

    start = time.time()
    control: Dict[str, bool] = {"save_requested": False, "stop_requested": False}
    completed_steps = resumed_steps
    last_saved_step = -1
    interrupted = False
    best_checkpoint_val_loss = float("inf")
    existing_best = meta.get("best_checkpoint_val_loss")
    if isinstance(existing_best, (int, float)):
        best_checkpoint_val_loss = float(existing_best)

    def _save_checkpoint(
        reason: str,
        step_value: int,
        *,
        val_loss: float | None = None,
        interrupted_state: bool = False,
    ) -> None:
        nonlocal last_saved_step
        save_meta = dict(meta)
        if reason != "final" and interrupted_state:
            save_meta["phase"] = "training_interrupted"
            save_meta["interrupted"] = True
            save_meta["last_step"] = int(step_value)
            save_meta["save_reason"] = reason
        elif reason != "final":
            save_meta["save_reason"] = reason
            save_meta["last_step"] = int(step_value)

        if val_loss is not None:
            save_meta["checkpoint_val_loss"] = float(val_loss)
        if math.isfinite(best_checkpoint_val_loss):
            save_meta["best_checkpoint_val_loss"] = float(best_checkpoint_val_loss)

        save_model_zip(args.output, model, tokenizer, meta=save_meta, name=args.name)
        if resume_path is not None:
            rng_state_payload: Dict[str, Any] = {
                "torch": torch.get_rng_state().cpu(),
                "python_random": random.getstate(),
            }
            if device.type == "cuda":
                try:
                    rng_state_payload["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
                except Exception:
                    rng_state_payload["cuda"] = []

            sidecar_payload: Dict[str, Any] = {
                "version": 1,
                "model_output": str(args.output),
                "saved_at": float(time.time()),
                "fingerprint": resume_fingerprint,
                "training": {
                    "completed_steps": int(step_value),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": {
                        "step": int(scheduler.step),
                        "learning_rate": float(scheduler.learning_rate),
                    },
                    "scaler_state_dict": scaler.state_dict() if use_grad_scaler else None,
                    "best_checkpoint_val_loss": float(best_checkpoint_val_loss)
                    if math.isfinite(best_checkpoint_val_loss)
                    else float("inf"),
                    "rng_state": rng_state_payload,
                    "data_split": split_info,
                },
            }
            _save_training_state(resume_path, sidecar_payload)
        last_saved_step = step_value
        print(f"Saved checkpoint ({reason}) at step {step_value}: {args.output}")

    _log("Manual checkpoint commands: type 'save' + Enter to save, 'stop' + Enter to save and stop", args.quiet)
    if args.stdin_commands:
        _start_stdin_command_listener(control, args.quiet)

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum, _frame) -> None:
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        _log(f"Received {signame}; saving checkpoint and stopping after current step", args.quiet)
        control["save_requested"] = True
        control["stop_requested"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        step = resumed_steps
        while True:
            step += 1
            x, y = _build_batch(train_tokens_tensor, effective_block_size, args.batch_size, device)

            lr = scheduler.next_lr()
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, enable_cuda_amp, autocast_dtype):
                logits = train_model(
                    x,
                    training=True,
                    dropout=args.dropout,
                    layer_drop=effective_layer_drop,
                    checkpointing=args.gradient_checkpointing,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    label_smoothing=max(0.0, args.label_smoothing),
                )
                scaled_loss = loss * manual_loss_scale

            if use_grad_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            grad_norm = None
            if args.clip_norm is not None and args.clip_norm > 0:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.clip_norm).item())

            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            completed_steps = step

            reached_end = total_steps is not None and step >= total_steps
            should_log = step % args.log_interval == 0 or step == 1 or reached_end
            should_periodic_checkpoint = args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0

            ev: Dict[str, float] | None = None
            has_validation = int(val_tokens_tensor.numel()) >= effective_block_size + 2
            if has_validation and (should_log or should_periodic_checkpoint):
                ev = _evaluate(
                    train_model,
                    val_tokens_tensor,
                    block_size=effective_block_size,
                    batch_size=args.batch_size,
                    batches=max(1, args.validation_batches),
                    device=device,
                )

            if should_periodic_checkpoint:
                if ev is None:
                    _log(
                        f"Skipping periodic checkpoint at step {step}: validation split/data unavailable for val-loss gating",
                        args.quiet,
                    )
                else:
                    current_val_loss = float(ev["loss"])
                    if current_val_loss < best_checkpoint_val_loss:
                        best_checkpoint_val_loss = current_val_loss
                        _save_checkpoint("periodic_best", step, val_loss=current_val_loss)
                    else:
                        _log(
                            f"Skipping periodic checkpoint at step {step}: val_loss={current_val_loss:.6f} best={best_checkpoint_val_loss:.6f}",
                            args.quiet,
                        )

            if control["save_requested"] and step != last_saved_step:
                reason = "manual_stop" if control["stop_requested"] else "manual_save"
                manual_val = float(ev["loss"]) if ev is not None else None
                _save_checkpoint(reason, step, val_loss=manual_val, interrupted_state=control["stop_requested"])
                control["save_requested"] = False

            if control["stop_requested"]:
                interrupted = True
                break

            if should_log:
                train_loss = float(loss.item())
                val_loss = float("nan")
                if ev is not None:
                    val_loss = float(ev["loss"])

                msg = [f"step={step}", f"train_loss={train_loss:.6f}", f"val_loss={val_loss:.6f}"]
                elapsed = max(1e-9, time.time() - start)
                samples_per_second = (step * args.batch_size) / elapsed
                msg.append(f"samples_per_second={samples_per_second:.2f}")
                if "learningRate" in metrics:
                    msg.append(f"lr={lr:.8f}")
                if "gradientNorm" in metrics and grad_norm is not None:
                    msg.append(f"grad_norm={grad_norm:.6f}")
                if "tokensPerSecond" in metrics:
                    tps = (step * args.batch_size * effective_block_size) / elapsed
                    msg.append(f"tokens_per_second={tps:.2f}")

                if ev is not None and ("accuracy" in metrics or "perplexity" in metrics):
                    if "accuracy" in metrics:
                        msg.append(f"val_acc={ev['accuracy']:.4f}")
                    if "perplexity" in metrics:
                        msg.append(f"val_ppl={ev['perplexity']:.4f}")

                if args.prompt:
                    with torch.no_grad():
                        prompt_ids = tokenizer.encode(args.prompt)
                        sample_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.9, top_p=0.9)
                        sample_text = tokenizer.decode(sample_ids[len(prompt_ids) :])
                    msg.append(f"example={sample_text[:80].replace(chr(10), ' ')}")

                print(" ".join(msg))

            if reached_end:
                break
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

    if interrupted:
        if completed_steps != last_saved_step:
            _save_checkpoint("signal_interrupt", completed_steps, interrupted_state=True)
        print(f"Training interrupted at step {completed_steps}. Last checkpoint: {args.output}")
        return

    if total_steps is not None and completed_steps >= total_steps and resumed_steps >= total_steps:
        _log(f"Requested steps already completed in resume state ({completed_steps}/{total_steps})", args.quiet)

    _save_checkpoint("final", completed_steps)
    print(f"Training complete. Saved: {args.output}")


def cmd_generate(args: argparse.Namespace) -> None:
    device = pick_device(args.cpu)
    model, tokenizer, _, _ = load_model_zip(args.model)
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(args.prompt)
    if args.prepend_bos and hasattr(tokenizer, "bos_token"):
        input_ids = [int(getattr(tokenizer, "bos_token"))] + input_ids
    if not input_ids:
        input_ids = [getattr(tokenizer, "bos_token", 0)]

    stop_ids = None
    if not args.allow_special and hasattr(tokenizer, "is_special_token"):
        stop_ids = set()
        for token_id in range(len(getattr(tokenizer, "get_vocab")())):
            if tokenizer.is_special_token(token_id):
                stop_ids.add(token_id)

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=stop_ids,
    )

    if args.only_new_text:
        out_ids = generated_ids[len(input_ids) :]
    else:
        out_ids = generated_ids

    out_text = tokenizer.decode(out_ids)
    print(out_text)


def _extract_config_path(argv: Sequence[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config requires a file path")
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _default_config_path() -> Path:
    return Path(__file__).resolve().with_name("nanogpt.defaults.json")


def _normalize_config_key(key: str) -> str:
    return key.replace("-", "_")


def _load_config_payload(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Config file not found: {config_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object")
    return payload


def _get_subparser_choices(parser: argparse.ArgumentParser) -> Dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _parser_dests(parser: argparse.ArgumentParser) -> set[str]:
    dests: set[str] = set()
    for action in parser._actions:
        if action.dest and action.dest != argparse.SUPPRESS and action.dest != "help":
            dests.add(action.dest)
    return dests


def _resolve_command_name(name: str, command_parsers: Dict[str, argparse.ArgumentParser]) -> str | None:
    if name in command_parsers:
        return name
    swapped_dash = name.replace("_", "-")
    if swapped_dash in command_parsers:
        return swapped_dash
    swapped_underscore = name.replace("-", "_")
    if swapped_underscore in command_parsers:
        return swapped_underscore
    return None


def _filter_defaults(defaults: Dict[str, Any], allowed: set[str], scope: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, value in defaults.items():
        norm = _normalize_config_key(str(key))
        if norm in allowed:
            out[norm] = value
        else:
            unknown.append(str(key))
    if unknown:
        print(f"warning: ignoring unknown config key(s) for {scope}: {', '.join(sorted(unknown))}")
    return out


def _apply_config_defaults(parser: argparse.ArgumentParser, payload: Dict[str, Any]) -> None:
    command_parsers = _get_subparser_choices(parser)
    parser_allowed = _parser_dests(parser)
    for p in command_parsers.values():
        parser_allowed.update(_parser_dests(p))

    global_defaults = payload.get("global", {})
    if global_defaults is not None:
        if not isinstance(global_defaults, dict):
            raise ValueError("Config field 'global' must be an object")
        filtered = _filter_defaults(global_defaults, parser_allowed, "global")
        if filtered:
            parser.set_defaults(**filtered)

    commands_block = payload.get("commands", {})
    if commands_block is not None and not isinstance(commands_block, dict):
        raise ValueError("Config field 'commands' must be an object")

    command_defaults: Dict[str, Dict[str, Any]] = {}
    if isinstance(commands_block, dict):
        for cmd_name, cmd_cfg in commands_block.items():
            if not isinstance(cmd_cfg, dict):
                raise ValueError(f"Config commands.{cmd_name} must be an object")
            resolved = _resolve_command_name(str(cmd_name), command_parsers)
            if resolved is None:
                print(f"warning: ignoring unknown command in config: {cmd_name}")
                continue
            command_defaults[resolved] = cmd_cfg

    for key, value in payload.items():
        if key in {"global", "commands"}:
            continue
        if not isinstance(value, dict):
            continue
        resolved = _resolve_command_name(str(key), command_parsers)
        if resolved is not None and resolved not in command_defaults:
            command_defaults[resolved] = value

    for cmd_name, defaults in command_defaults.items():
        cmd_parser = command_parsers[cmd_name]
        allowed = _parser_dests(cmd_parser)
        filtered = _filter_defaults(defaults, allowed, f"command '{cmd_name}'")
        if filtered:
            cmd_parser.set_defaults(**filtered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Python NanoGPTV2 tools compatible with this repo's zip + custom safetensors format"
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file to overlay on top of nanogpt.defaults.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="Load and validate a model zip")
    p_load.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p_load.add_argument("--model", required=True, help="Path to model zip")
    p_load.set_defaults(func=cmd_load)

    p_save = sub.add_parser("save", help="Load and re-save a model zip")
    p_save.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p_save.add_argument("--model", required=True, help="Input model zip")
    p_save.add_argument("--output", required=True, help="Output model zip")
    p_save.add_argument("--name", default=None, help="Optional meta name")
    p_save.set_defaults(func=cmd_save)

    p_train = sub.add_parser("train", help="Pre-train/fine-tune a model zip on raw text")
    p_train.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p_train.add_argument("--model", help="Input model zip")
    p_train.add_argument("--init-from-scratch", action="store_true", help="Create a new model instead of loading --model")

    p_train.add_argument("--n-layer", type=int, default=6, help="Model depth (new model only)")
    p_train.add_argument("--n-head", type=int, default=4, help="Attention heads (new model only)")
    p_train.add_argument("--n-embed", type=int, default=256, help="Embedding dimension (new model only)")
    p_train.add_argument("--block-size", type=int, default=128, help="Context window size (new model only)")
    p_train.add_argument("--mlp-factor", type=int, default=4, help="MLP expansion factor (new model only)")
    p_train.add_argument("--window-size", default=None, help="Optional window size string for config (new model only)")
    p_train.add_argument("--vocab-size", type=int, default=2000, help="Tokenizer vocab size target (char init)")
    p_train.add_argument("--tokenizer-type", default="char", help="Tokenizer type for new model: char or bpe")
    p_train.add_argument("--tokenizer-vocab-file", default=None, help="JSON vocab list or tokeniser.json for new model")
    p_train.add_argument("--tokenizer-merges-file", default=None, help="Optional JSON merges file for bpe")
    p_train.add_argument("--tokenizer-file", default=None, help="Tokenizer file generated by prepare-data (for pretokenized training)")
    p_train.add_argument("--token-ids-file", default=None, help="Pretokenized ids file generated by prepare-data")
    p_train.add_argument("--text-file", help="Single input file (.txt/.csv/.json/.jsonl)")
    p_train.add_argument("--text-files", nargs="*", default=[], help="Multiple input files (.txt/.csv/.json/.jsonl)")
    p_train.add_argument("--text-dirs", nargs="*", default=[], help="Directories to scan recursively for .txt/.csv/.json/.jsonl")
    p_train.add_argument("--globs", nargs="*", default=[], help="Glob patterns relative to cwd")
    p_train.add_argument("--csv-column", default="text", help="CSV column name to extract")
    p_train.add_argument("--csv-has-header", action="store_true", help="Force CSV header row")
    p_train.add_argument("--csv-no-header", action="store_false", dest="csv_has_header", help="Force CSV no header")
    p_train.set_defaults(csv_has_header=None)
    p_train.add_argument("--output", required=True, help="Output model zip")
    p_train.add_argument(
        "--resume-state-file",
        default=None,
        help="Optional sidecar training state file for seamless resume (optimizer/scheduler/scaler/RNG)",
    )
    p_train.add_argument("--steps", type=int, default=0, help="Training steps cap (0 means no step cap)")
    p_train.add_argument("--max-epochs", type=int, default=0, help="Epoch cap when --steps <= 0 (0 means no epoch cap)")
    p_train.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p_train.add_argument("--validation-split", type=float, default=0.1, help="Validation split fraction")
    p_train.add_argument("--validation-batches", type=int, default=8, help="Validation batches per log")
    p_train.add_argument("--context-scaling", type=float, default=1.0, help="Scale context length in training")

    p_train.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    p_train.add_argument("--min-learning-rate", type=float, default=3e-5, help="Minimum LR for cosine schedule")
    p_train.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    p_train.add_argument("--decay-epochs", type=int, default=100, help="Cosine decay epochs")
    p_train.add_argument("--epoch-steps", type=int, default=0, help="Steps per epoch for LR schedule (0=auto)")

    p_train.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    p_train.add_argument("--beta2", type=float, default=0.99, help="AdamW beta2")
    p_train.add_argument("--epsilon", type=float, default=1e-8, help="AdamW epsilon")
    p_train.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    p_train.add_argument("--loss-scaling", type=float, default=1.0, help="Loss scaling factor")
    p_train.add_argument("--clip-norm", type=float, default=None, help="Gradient clip norm")

    p_train.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    p_train.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training")
    p_train.add_argument(
        "--mixed-precision-dtype",
        default="auto",
        choices=["auto", "bf16", "fp16"],
        help="Mixed precision dtype policy for CUDA",
    )
    p_train.add_argument(
        "--enable-tf32",
        action="store_true",
        default=True,
        help="Enable TF32 matmul/cudnn acceleration on CUDA",
    )
    p_train.add_argument(
        "--disable-tf32",
        action="store_false",
        dest="enable_tf32",
        help="Disable TF32 acceleration",
    )
    p_train.add_argument(
        "--compile-model",
        action="store_true",
        default=False,
        help="Enable torch.compile for the training forward pass",
    )
    p_train.add_argument(
        "--no-compile-model",
        action="store_false",
        dest="compile_model",
        help="Disable torch.compile",
    )
    p_train.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    p_train.add_argument(
        "--optimizer-impl",
        default="auto",
        choices=["auto", "fused", "foreach", "default"],
        help="AdamW backend selection",
    )
    p_train.add_argument(
        "--attention-backend",
        default="auto",
        choices=["auto", "sdpa", "math"],
        help="Attention kernel backend (auto allows flash-capable SDPA when available)",
    )
    p_train.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing")
    p_train.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    p_train.add_argument("--layer-drop", type=float, default=0.0, help="Layer drop rate")
    p_train.add_argument("--trainable-weights", nargs="*", default=[], help="Glob patterns over parameter names")
    p_train.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metrics: accuracy,perplexity,gradientNorm,tokensPerSecond,learningRate",
    )
    p_train.add_argument("--prompt", default=None, help="Optional prompt for periodic sample output")

    p_train.add_argument("--sft-mode", default="full", help="Accepted for TS option compatibility; unused in Python pretraining")
    p_train.add_argument("--masked-loss", action="store_true", help="Accepted for TS option compatibility; unused in Python pretraining")
    p_train.add_argument("--lora-name", default=None, help="Accepted for TS option compatibility; unused in Python pretraining")
    p_train.add_argument("--lora-config", default=None, help="Accepted for TS option compatibility; unused in Python pretraining")

    p_train.add_argument("--log-interval", type=int, default=20, help="Log every N steps")
    p_train.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save output every N steps only when validation loss improves (0 disables periodic checkpointing)",
    )
    p_train.add_argument(
        "--stdin-commands",
        action="store_true",
        default=True,
        help="Enable stdin commands during training: 'save' and 'stop'",
    )
    p_train.add_argument(
        "--no-stdin-commands",
        action="store_false",
        dest="stdin_commands",
        help="Disable stdin command listener",
    )
    p_train.add_argument("--name", default=None, help="Optional meta name for output")
    p_train.add_argument("--progress-interval", type=int, default=1000, help="Progress log interval for preprocessing")
    p_train.add_argument("--quiet", action="store_true", help="Reduce non-essential progress logs")
    p_train.add_argument("--cpu", action="store_true", help="Force CPU")
    p_train.set_defaults(func=cmd_train)

    p_prepare = sub.add_parser("prepare-data", help="Build tokenizer and token-id dataset files for later training")
    p_prepare.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p_prepare.add_argument("--model", default=None, help="Optional base model zip; reuses its tokenizer/vocab")
    p_prepare.add_argument("--text-file", help="Single input file (.txt/.csv/.json/.jsonl)")
    p_prepare.add_argument("--text-files", nargs="*", default=[], help="Multiple input files (.txt/.csv/.json/.jsonl)")
    p_prepare.add_argument("--text-dirs", nargs="*", default=[], help="Directories to scan recursively for .txt/.csv/.json/.jsonl")
    p_prepare.add_argument("--globs", nargs="*", default=[], help="Glob patterns relative to cwd")
    p_prepare.add_argument("--csv-column", default="text", help="CSV column name to extract")
    p_prepare.add_argument("--csv-has-header", action="store_true", help="Force CSV header row")
    p_prepare.add_argument("--csv-no-header", action="store_false", dest="csv_has_header", help="Force CSV no header")
    p_prepare.set_defaults(csv_has_header=None)
    p_prepare.add_argument("--tokenizer-type", default="char", help="Tokenizer type: char or bpe")
    p_prepare.add_argument("--vocab-size", type=int, default=2000, help="Target vocab size")
    p_prepare.add_argument("--tokenizer-vocab-file", default=None, help="Optional existing vocab/tokeniser JSON")
    p_prepare.add_argument("--tokenizer-merges-file", default=None, help="Optional separate merges JSON")
    p_prepare.add_argument("--output-tokenizer-file", default=None, help="Output tokenizer JSON path")
    p_prepare.add_argument("--output-token-ids-file", required=True, help="Output token-id tensor path (.pt)")
    p_prepare.add_argument("--progress-interval", type=int, default=1000, help="Progress log interval for preprocessing")
    p_prepare.add_argument("--quiet", action="store_true", help="Reduce non-essential progress logs")
    p_prepare.set_defaults(func=cmd_prepare_data)

    p_gen = sub.add_parser("generate", help="Generate text from a model zip")
    p_gen.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p_gen.add_argument("--model", required=True, help="Input model zip")
    p_gen.add_argument("--prompt", required=True, help="Prompt text")
    p_gen.add_argument("--max-new-tokens", type=int, default=100, help="Tokens to sample")
    p_gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p_gen.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling threshold")
    p_gen.add_argument(
        "--prepend-bos",
        action="store_true",
        default=True,
        help="Prepend BOS token to prompt (recommended for TS compatibility)",
    )
    p_gen.add_argument(
        "--no-prepend-bos",
        action="store_false",
        dest="prepend_bos",
        help="Do not prepend BOS token",
    )
    p_gen.add_argument(
        "--allow-special",
        action="store_true",
        help="Allow generation to continue through special tokens",
    )
    p_gen.add_argument("--only-new-text", action="store_true", help="Print only generated continuation")
    p_gen.add_argument("--cpu", action="store_true", help="Force CPU")
    p_gen.set_defaults(func=cmd_generate)

    return parser


def main() -> None:
    argv = sys.argv[1:]
    overlay_config_path = _extract_config_path(argv)

    parser = build_parser()
    default_path = _default_config_path()
    default_payload = _load_config_payload(str(default_path))
    _apply_config_defaults(parser, default_payload)

    if overlay_config_path:
        default_resolved = default_path.resolve()
        overlay_resolved = Path(overlay_config_path).resolve()
        if overlay_resolved != default_resolved:
            overlay_payload = _load_config_payload(overlay_config_path)
            _apply_config_defaults(parser, overlay_payload)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
