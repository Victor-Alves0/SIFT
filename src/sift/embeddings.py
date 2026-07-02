"""Embedding backends for the discovery layer.

The default is a fully local, no-API-key embedder via ``fastembed`` (ONNX).
Any object with an ``embed(texts) -> list[vector]`` method can be plugged in
(OpenAI, Cohere, a remote sidecar, etc.).
"""
from __future__ import annotations

import os
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """``embed(texts)`` is required (used for documents). An embedder MAY also
    provide ``embed_query(texts)`` for models trained with an asymmetric
    query/passage scheme (E5-style prefixes); the gateway uses it for queries
    when present and falls back to ``embed`` otherwise."""

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...


class FastEmbedder:
    """Local embeddings via fastembed. Downloads the model on first use."""

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name or os.getenv("SIFT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(list(texts))]

    def embed_query(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Query-side embeddings — applies the model's query prefix where the
        model has one (E5 family; a no-op for bge, verified)."""
        return [np.asarray(v, dtype=np.float32) for v in self._model.query_embed(list(texts))]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
