from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .model import NanoGPTV2, NanoGPTV2Config
from .tokenizer import BPETokenizer, CharTokenizer, Tokenizer, load_tokenizer


def load_custom_safetensors(data: bytes) -> Dict[str, np.ndarray]:
    if len(data) < 8:
        raise ValueError("Invalid safetensors payload: too short")

    header_len = struct.unpack_from("<I", data, 0)[0]
    header_bytes = data[8 : 8 + header_len]
    header = json.loads(header_bytes.decode("utf-8").rstrip(" "))

    out: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta["dtype"]
        shape = tuple(int(v) for v in meta["shape"])
        start, end = meta["data_offsets"]
        start = int(start)
        end = int(end)
        payload = data[8 + header_len + start : 8 + header_len + end]

        if dtype == "F32":
            arr = np.frombuffer(payload, dtype=np.float32)
        elif dtype == "I32":
            arr = np.frombuffer(payload, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if arr.size == 0:
            out[name] = np.empty(shape, dtype=arr.dtype)
        else:
            out[name] = arr.reshape(shape)
    return out


def save_custom_safetensors(tensors: Dict[str, np.ndarray]) -> bytes:
    structure: Dict[str, Dict[str, object]] = {}
    offset = 0

    for name, arr in tensors.items():
        if arr.dtype == np.float32:
            dtype = "F32"
        elif arr.dtype == np.int32:
            dtype = "I32"
        else:
            raise ValueError(f"Unsupported dtype for {name}: {arr.dtype}")

        nbytes = int(arr.size * 4)
        structure[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_json = json.dumps(structure, separators=(",", ":"))
    header = header_json.encode("utf-8")
    pad = (4 - (len(header) % 4)) % 4
    if pad:
        header = header + (b" " * pad)

    chunks = [struct.pack("<I", len(header)), b"\x00\x00\x00\x00", header]

    for _, arr in tensors.items():
        flat = np.ascontiguousarray(arr.reshape(-1))
        chunks.append(flat.tobytes(order="C"))

    return b"".join(chunks)


def load_model_zip(
    path: str | Path,
    require_weights: bool = True,
    strict_weights: bool = True,
) -> Tuple[NanoGPTV2, Tokenizer, Dict[str, object], Dict[str, object]]:
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        config = json.loads(zf.read("config.json").decode("utf-8"))
        tok_spec = json.loads(zf.read("tokeniser.json").decode("utf-8"))
        has_weights = "model.safetensors" in zf.namelist()
        if has_weights:
            weights_data = zf.read("model.safetensors")
        elif require_weights:
            raise ValueError(f"Model weights not found in {p}")
        else:
            weights_data = None
        meta = json.loads(zf.read("meta.json").decode("utf-8")) if "meta.json" in zf.namelist() else {
            "version": 0,
            "application": "",
        }

    cfg = NanoGPTV2Config.from_transformers_config(config)
    model = NanoGPTV2(cfg)

    if weights_data is not None:
        weights = load_custom_safetensors(weights_data)
        model.load_weight_dict(weights, strict=strict_weights)

    tokenizer = load_tokenizer(tok_spec)
    return model, tokenizer, config, meta


def save_model_zip(
    path: str | Path,
    model: NanoGPTV2,
    tokenizer: Tokenizer,
    meta: Dict[str, object] | None = None,
    name: str | None = None,
) -> None:
    p = Path(path)

    weights = model.to_weight_dict()
    weights_blob = save_custom_safetensors(weights)

    cfg = model.config.to_transformers_config()

    if isinstance(tokenizer, CharTokenizer):
        tok_type = "char"
    elif isinstance(tokenizer, BPETokenizer):
        tok_type = "bpe"
    else:
        raise ValueError(f"Unsupported tokenizer instance: {type(tokenizer)}")

    tok_json = {
        "type": tok_type,
        "vocab": tokenizer.get_vocab(),
        "merges": tokenizer.get_merges(),
    }

    meta_out: Dict[str, object] = {
        "version": 2,
        "application": "@genai-fi/nanogpt",
    }
    if meta:
        meta_out.update(meta)
    if name is not None:
        meta_out["name"] = name

    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.safetensors", weights_blob)
        zf.writestr("config.json", json.dumps(cfg, indent=4, ensure_ascii=False))
        zf.writestr("tokeniser.json", json.dumps(tok_json, indent=4, ensure_ascii=False))
        zf.writestr("meta.json", json.dumps(meta_out, indent=4, ensure_ascii=False))
