from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union
import unicodedata

SPECIALS: List[str] = [
    "<eos>",
    "<bos>",
    "",
    "<pad>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|system_start|>",
    "<|system_end|>",
]


def _is_punct_symbol_or_space(ch: str) -> bool:
    if ch.isspace():
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def parse_tokens(text: str) -> List[str]:
    normalized = list(text)
    out: List[str] = []
    current = ""
    i = 0
    while i < len(normalized):
        ch = normalized[i]
        if ch == " ":
            next_ch = normalized[i + 1] if i + 1 < len(normalized) else ""
            if next_ch != " ":
                out.append(current)
                current = ch
            else:
                current += ch
        elif _is_punct_symbol_or_space(ch):
            out.append(current)
            run = ch
            while i + 1 < len(normalized) and normalized[i + 1] == ch:
                run += normalized[i + 1]
                i += 1
            out.append(run)
            current = ""
        else:
            current += ch
        i += 1

    if current:
        out.append(current)

    return [t for t in out if len(t) > 0]


def _special_index(vocab: Sequence[str], token: str) -> int | None:
    try:
        return vocab.index(token)
    except ValueError:
        return None


class CharTokenizer:
    def __init__(self, vocab: Sequence[str]):
        if not vocab:
            raise ValueError("Vocab cannot be empty")
        self.vocab: List[str] = list(vocab)

        self.special_tokens: Dict[str, int] = {}
        self.special_token_set = set()
        for tok in SPECIALS:
            idx = _special_index(self.vocab, tok)
            if idx is not None:
                self.special_tokens[tok] = idx
                self.special_token_set.add(idx)

        self.eos_token = self.special_tokens.get("<eos>", 0)
        self.bos_token = self.special_tokens.get("<bos>", self.eos_token)

        unk = self.special_tokens.get("")
        if unk is None:
            for fallback in ("<unk>", "<pad>", "_", " "):
                idx = _special_index(self.vocab, fallback)
                if idx is not None:
                    unk = idx
                    break
        if unk is None:
            unk = self.eos_token
        self.unk_token = unk

        self.vocab = ["" if t == "<pad>" else t for t in self.vocab]
        self.cache: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}

    def is_special_token(self, token_id: int) -> bool:
        return token_id in self.special_token_set

    def get_vocab(self) -> List[str]:
        return list(self.vocab)

    def get_merges(self) -> List[Tuple[str, str]]:
        return []

    def encode(self, text: str) -> List[int]:
        return [self.cache.get(ch, self.unk_token) for ch in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(self.vocab[t] if 0 <= t < len(self.vocab) else "" for t in tokens)


class BPETokenizer:
    def __init__(self, vocab: Sequence[str], merges: Sequence[Sequence[str]] | None = None):
        self.vocab: List[str] = list(vocab)
        self.vocab_index: Dict[str, int] = {v: i for i, v in enumerate(self.vocab)}
        self.merges: List[Tuple[str, str]] = [
            (m[0], m[1]) for m in (merges or []) if len(m) == 2
        ]
        self.pretoken_map: Dict[str, List[str]] = {}

        self.eos_token = self.vocab_index.get("<eos>", 0)
        self.bos_token = self.vocab_index.get("<bos>", 0)
        self.unk_token = self.vocab_index.get("", 1)

        self.special_token_set = set()
        for tok in SPECIALS:
            idx = self.vocab_index.get(tok)
            if idx is not None:
                self.special_token_set.add(idx)

    def is_special_token(self, token_id: int) -> bool:
        return token_id in self.special_token_set

    def get_vocab(self) -> List[str]:
        return list(self.vocab)

    def get_merges(self) -> List[Tuple[str, str]]:
        return list(self.merges)

    @staticmethod
    def _merge_all(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        a, b = pair
        out: List[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def _tokenize_word(self, word: str) -> List[str]:
        tokens = list(word)
        for pair in self.merges:
            tokens = self._merge_all(tokens, pair)
        self.pretoken_map[word] = tokens
        return tokens

    def _tokenize_strings(self, texts: Sequence[str]) -> List[List[str]]:
        out: List[List[str]] = []
        for t in texts:
            pretokens = parse_tokens(t)
            tokens: List[str] = []
            for pt in pretokens:
                if pt in self.pretoken_map:
                    tokens.extend(self.pretoken_map[pt])
                else:
                    tokens.extend(self._tokenize_word(pt))
            out.append(tokens)
        return out

    def encode(self, text: str) -> List[int]:
        token_strs = self._tokenize_strings([text])[0]
        return [self.vocab_index.get(tok, self.unk_token) for tok in token_strs]

    def decode(self, tokens: Sequence[int]) -> str:
        parts = []
        for t in tokens:
            if 0 <= t < len(self.vocab):
                parts.append(self.vocab[t])
        return "".join(parts)


Tokenizer = Union[CharTokenizer, BPETokenizer]


@dataclass
class TokenizerSpec:
    type: str
    vocab: List[str]
    merges: List[Tuple[str, str]]


def load_tokenizer(spec: Dict[str, object]) -> Tokenizer:
    tok_type = str(spec.get("type", "char"))
    vocab = [str(v) for v in spec.get("vocab", [])]
    raw_merges = spec.get("merges", [])
    merges: List[Tuple[str, str]] = []
    if isinstance(raw_merges, list):
        for m in raw_merges:
            if isinstance(m, (list, tuple)) and len(m) == 2:
                merges.append((str(m[0]), str(m[1])))

    if tok_type == "char":
        return CharTokenizer(vocab)
    if tok_type == "bpe":
        return BPETokenizer(vocab, merges)
    raise ValueError(f"Unsupported tokenizer type: {tok_type}")
