"""Lexical retrieval (BM25) and rank fusion (RRF) for hybrid discovery.

Pure-Python, no dependencies. Combined with the embedding ranking in the gateway
this gives hybrid search: embeddings catch paraphrase/semantics, BM25 catches
exact terms (names, ids, rare words) — fused with Reciprocal Rank Fusion, which
needs no score normalisation or tuning.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed document set."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.n
        if not self.avgdl:
            return out
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def rank_order(scores: list[float]) -> list[int]:
    """Indices sorted best-first by score."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: fuse several best-first rankings into one score map."""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return fused
