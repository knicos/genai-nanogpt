from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


@dataclass
class NanoGPTV2Config:
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    block_size: int
    mlpFactor: int
    windowSize: Optional[str] = None

    @property
    def n_embed(self) -> int:
        return self.hidden_size

    @property
    def n_layer(self) -> int:
        return self.num_hidden_layers

    @property
    def n_head(self) -> int:
        return self.num_attention_heads

    @classmethod
    def from_transformers_config(cls, cfg: Dict[str, object]) -> "NanoGPTV2Config":
        model_type = str(cfg.get("model_type", ""))
        if model_type != "GenAI_NanoGPT_v2":
            raise ValueError(f"Only GenAI_NanoGPT_v2 is supported, got: {model_type}")
        return cls(
            model_type=model_type,
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            block_size=int(cfg["block_size"]),
            mlpFactor=int(cfg["mlpFactor"]),
            windowSize=str(cfg["windowSize"]) if cfg.get("windowSize") is not None else None,
        )

    def to_transformers_config(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "model_type": "GenAI_NanoGPT_v2",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "block_size": self.block_size,
            "mlpFactor": self.mlpFactor,
        }
        if self.windowSize is not None:
            data["windowSize"] = self.windowSize
        return data


def rms_norm_no_gamma(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)


class RoPECache:
    def __init__(self, config: NanoGPTV2Config):
        head_dim = config.hidden_size // config.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE")
        self.rotary_dim = head_dim
        self.base = 10000.0
        idx = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
        self.inv_freq = 1.0 / (self.base ** (idx / float(self.rotary_dim)))
        self.cache_len = 0
        self.cos: Optional[torch.Tensor] = None
        self.sin: Optional[torch.Tensor] = None

    def ensure(self, needed: int, device: torch.device, dtype: torch.dtype) -> None:
        if needed <= self.cache_len and self.cos is not None and self.sin is not None:
            if self.cos.device == device and self.cos.dtype == dtype:
                return
        length = max(needed, self.cache_len)
        if length == 0:
            length = needed
        pos = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        freqs = pos * self.inv_freq.to(device=device)
        self.cos = torch.cos(freqs).to(dtype=dtype)
        self.sin = torch.sin(freqs).to(dtype=dtype)
        self.cache_len = length


def apply_rope(x: torch.Tensor, rope_cache: RoPECache, past_len: int) -> torch.Tensor:
    b, h, t, hs = x.shape
    rd = rope_cache.rotary_dim
    rope_cache.ensure(past_len + t, x.device, x.dtype)
    assert rope_cache.cos is not None and rope_cache.sin is not None

    x_rot = x[:, :, :, :rd]
    x_rest = x[:, :, :, rd:] if rd < hs else None

    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]

    cos = rope_cache.cos[past_len : past_len + t].view(1, 1, t, rd // 2)
    sin = rope_cache.sin[past_len : past_len + t].view(1, 1, t, rd // 2)

    even_rot = x_even * cos - x_odd * sin
    odd_rot = x_odd * cos + x_even * sin
    interleaved = torch.stack([even_rot, odd_rot], dim=-1).reshape(b, h, t, rd)

    if x_rest is not None:
        return torch.cat([interleaved, x_rest], dim=-1)
    return interleaved


class CausalSelfAttention(nn.Module):
    def __init__(self, idx: int, config: NanoGPTV2Config):
        super().__init__()
        self.idx = idx
        self.n_embed = config.hidden_size
        self.n_head = config.num_attention_heads
        self.head_dim = self.n_embed // self.n_head
        self.block_size = config.block_size
        self.divisor = 1.0 / (self.head_dim ** 0.5)
        self.attention_backend = "auto"

        self.c_attn = nn.Parameter(torch.randn(self.n_embed, 3 * self.n_embed) * 0.02)
        self.c_proj = nn.Parameter(torch.randn(self.n_embed, self.n_embed) * 0.02)

    def set_attention_backend(self, backend: str) -> None:
        backend = str(backend).lower()
        if backend not in {"auto", "sdpa", "math"}:
            raise ValueError(f"Unsupported attention backend: {backend}")
        self.attention_backend = backend

    def _attention_mask(self, t_cur: int, t_total: int, past_len: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(t_cur, device=device).unsqueeze(1)
        j = torch.arange(t_total, device=device).unsqueeze(0)
        return j <= (i + past_len)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: RoPECache,
        past_kv: Optional[Dict[str, object]] = None,
        training: bool = False,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        b, t_cur, c = x.shape

        qkv = x.view(b * t_cur, c) @ self.c_attn
        qkv = qkv.view(b, t_cur, 3 * c)
        q, k_new, v_new = torch.split(qkv, c, dim=-1)

        q = q.view(b, t_cur, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        k_new = k_new.view(b, t_cur, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v_new = v_new.view(b, t_cur, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        initial_past = int(past_kv.get("cumulativeLength", 0)) if past_kv is not None else 0
        q = apply_rope(q, rope_cache, initial_past)
        k_new = apply_rope(k_new, rope_cache, initial_past)

        q = rms_norm_no_gamma(q)
        k_new = rms_norm_no_gamma(k_new)

        past_len = int(past_kv.get("length", 0)) if past_kv is not None else 0

        if past_kv is not None and not training:
            k_prev = past_kv.get("k")
            v_prev = past_kv.get("v")
            if isinstance(k_prev, torch.Tensor):
                k_total = torch.cat([k_prev, k_new], dim=2)
            else:
                k_total = k_new
            if isinstance(v_prev, torch.Tensor):
                v_total = torch.cat([v_prev, v_new], dim=2)
            else:
                v_total = v_new

            if k_total.shape[2] > self.block_size:
                k_total = k_total[:, :, -self.block_size :, :]
                v_total = v_total[:, :, -self.block_size :, :]
            past_kv["k"] = k_total.detach()
            past_kv["v"] = v_total.detach()
            past_kv["length"] = min(past_len + t_cur, self.block_size)
            past_kv["cumulativeLength"] = int(past_kv.get("cumulativeLength", 0)) + t_cur
        else:
            k_total = k_new
            v_total = v_new

        t_total = k_total.shape[2]
        use_sdpa = self.attention_backend in {"auto", "sdpa"} and hasattr(F, "scaled_dot_product_attention")
        if use_sdpa:
            if past_len == 0 and t_total == t_cur:
                y = F.scaled_dot_product_attention(
                    q,
                    k_total,
                    v_total,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
            else:
                mask = self._attention_mask(t_cur, t_total, past_len, x.device)
                y = F.scaled_dot_product_attention(
                    q,
                    k_total,
                    v_total,
                    attn_mask=mask.view(1, 1, t_cur, t_total),
                    dropout_p=0.0,
                    is_causal=False,
                )
        else:
            scores = (q @ k_total.transpose(-2, -1)) * self.divisor
            mask = self._attention_mask(t_cur, t_total, past_len, x.device)
            scores = scores.masked_fill(~mask.view(1, 1, t_cur, t_total), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            y = probs @ v_total

        y = y.permute(0, 2, 1, 3).contiguous().view(b, t_cur, c)
        out = y.view(b * t_cur, c) @ self.c_proj
        if training and dropout > 0.0:
            out = F.dropout(out, p=dropout, training=True)
        return out.view(b, t_cur, c)


class MLP(nn.Module):
    def __init__(self, idx: int, config: NanoGPTV2Config):
        super().__init__()
        self.idx = idx
        hidden = config.mlpFactor * config.hidden_size

        self.mlp_hidden = nn.Parameter(torch.randn(config.hidden_size, hidden) * 0.02)
        self.mlp_out = nn.Parameter(
            torch.randn(hidden, config.hidden_size) * (0.02 / (2.0 * config.num_hidden_layers) ** 0.5)
        )

    def forward(self, x: torch.Tensor, training: bool = False, dropout: float = 0.0) -> torch.Tensor:
        b, t, c = x.shape
        x2 = x.view(b * t, c)
        h = x2 @ self.mlp_hidden
        h = F.relu(h) ** 2
        out = h @ self.mlp_out
        if training and dropout > 0.0:
            out = F.dropout(out, p=dropout, training=True)
        return out.view(b, t, c)


class TransformerBlock(nn.Module):
    def __init__(self, idx: int, config: NanoGPTV2Config):
        super().__init__()
        self.idx = idx
        self.attn = CausalSelfAttention(idx, config)
        self.mlp = MLP(idx, config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: RoPECache,
        past_kv: Optional[Dict[str, object]] = None,
        training: bool = False,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        norm1 = rms_norm_no_gamma(x)
        attn_out = self.attn(norm1, rope_cache=rope_cache, past_kv=past_kv, training=training, dropout=dropout)
        x = x + attn_out

        norm2 = rms_norm_no_gamma(x)
        mlp_out = self.mlp(norm2, training=training, dropout=dropout)
        x = x + mlp_out
        return x


class NanoGPTV2(nn.Module):
    def __init__(self, config: NanoGPTV2Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Parameter(torch.randn(config.vocab_size, config.hidden_size) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(i, config) for i in range(config.num_hidden_layers)])
        self.rope_cache = RoPECache(config)
        self.attention_backend = "auto"

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = str(backend).lower()
        for block in self.blocks:
            block.attn.set_attention_backend(self.attention_backend)

    def new_cache(self) -> List[Dict[str, object]]:
        return [
            {
                "k": None,
                "v": None,
                "length": 0,
                "cumulativeLength": 0,
            }
            for _ in range(self.config.num_hidden_layers)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[List[Dict[str, object]]] = None,
        training: bool = False,
        skip_logits: bool = False,
        dropout: float = 0.0,
        layer_drop: float = 0.0,
        checkpointing: bool = False,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids rank 2 [B, T], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.config.block_size:
            raise ValueError(
                f"Input sequence length {input_ids.shape[1]} exceeds block size {self.config.block_size}"
            )

        x = F.embedding(input_ids, self.token_embedding)

        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError(f"Cache length {len(cache)} does not match number of blocks {len(self.blocks)}")

        for i, block in enumerate(self.blocks):
            if training and layer_drop > 0.0 and random.random() < (layer_drop * (i / max(len(self.blocks), 1))):
                continue
            past = cache[i] if cache is not None else None
            if training and checkpointing:
                def block_fn(inp: torch.Tensor) -> torch.Tensor:
                    return block(inp, rope_cache=self.rope_cache, past_kv=past, training=training, dropout=dropout)

                x = torch_checkpoint(block_fn, x, use_reentrant=False)
            else:
                x = block(
                    x,
                    rope_cache=self.rope_cache,
                    past_kv=past,
                    training=training,
                    dropout=dropout,
                )

        x = rms_norm_no_gamma(x)
        if skip_logits:
            return x

        logits = x @ self.token_embedding.transpose(0, 1)
        return logits

    def load_weight_dict(self, weights: Dict[str, np.ndarray], strict: bool = True) -> None:
        expected = set(self.expected_weight_names())
        if strict:
            missing = [k for k in sorted(expected) if k not in weights]
            if missing:
                raise ValueError(f"Missing required weights: {missing}")

        def as_writable_tensor(arr: np.ndarray) -> torch.Tensor:
            # Arrays from np.frombuffer can be read-only; copy to writable memory for torch.from_numpy.
            return torch.from_numpy(np.array(arr, copy=True, order="C"))

        with torch.no_grad():
            if "token_embedding" in weights:
                self.token_embedding.copy_(as_writable_tensor(weights["token_embedding"]).to(dtype=torch.float32))
            for i, block in enumerate(self.blocks):
                c_attn = f"block_{i}_cAttn"
                c_proj = f"block_{i}_cProj"
                mlp_h = f"block_{i}_mlpHidden"
                mlp_o = f"block_{i}_mlpOut"
                if c_attn in weights:
                    block.attn.c_attn.copy_(as_writable_tensor(weights[c_attn]).to(dtype=torch.float32))
                if c_proj in weights:
                    block.attn.c_proj.copy_(as_writable_tensor(weights[c_proj]).to(dtype=torch.float32))
                if mlp_h in weights:
                    block.mlp.mlp_hidden.copy_(as_writable_tensor(weights[mlp_h]).to(dtype=torch.float32))
                if mlp_o in weights:
                    block.mlp.mlp_out.copy_(as_writable_tensor(weights[mlp_o]).to(dtype=torch.float32))

    def to_weight_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        out["token_embedding"] = self.token_embedding.detach().cpu().float().numpy().copy()
        for i, block in enumerate(self.blocks):
            out[f"block_{i}_cAttn"] = block.attn.c_attn.detach().cpu().float().numpy().copy()
            out[f"block_{i}_cProj"] = block.attn.c_proj.detach().cpu().float().numpy().copy()
            out[f"block_{i}_mlpHidden"] = block.mlp.mlp_hidden.detach().cpu().float().numpy().copy()
            out[f"block_{i}_mlpOut"] = block.mlp.mlp_out.detach().cpu().float().numpy().copy()
        return out

    def expected_weight_names(self) -> List[str]:
        names = ["token_embedding"]
        for i in range(self.config.num_hidden_layers):
            names.extend(
                [
                    f"block_{i}_cAttn",
                    f"block_{i}_cProj",
                    f"block_{i}_mlpHidden",
                    f"block_{i}_mlpOut",
                ]
            )
        return names

    @torch.no_grad()
    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.9,
        stop_token_ids: Optional[set[int]] = None,
    ) -> List[int]:
        self.eval()
        device = self.token_embedding.device

        if len(input_ids) == 0:
            raise ValueError("input_ids must not be empty")

        cache = self.new_cache()
        idx = torch.tensor([input_ids[-self.config.block_size :]], dtype=torch.long, device=device)
        logits = self.forward(idx, cache=cache, training=False)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        generated = list(input_ids)
        for _ in range(max_new_tokens):
            if top_p is not None and 0.0 < top_p < 1.0:
                probs = torch.softmax(next_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                csum = torch.cumsum(sorted_probs, dim=-1)
                keep = csum <= top_p
                keep[:, 0] = True
                filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                sampled_sorted = torch.multinomial(filtered, num_samples=1)
                next_token = sorted_idx.gather(-1, sampled_sorted)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = int(next_token.item())
            generated.append(token_id)

            if stop_token_ids is not None and token_id in stop_token_ids:
                break

            idx = next_token.view(1, 1)
            logits = self.forward(idx, cache=cache, training=False)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        return generated
