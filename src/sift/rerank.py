"""Optional cross-encoder reranker.

Reranking re-scores the fused shortlist with a query×document cross-encoder,
which is more accurate than bi-encoder cosine for the final ordering. It's
opt-in (extra latency + a model download), plugged into the gateway:

    from sift import Sift
    from sift.rerank import FastEmbedReranker
    sift = Sift(reranker=FastEmbedReranker())

Any object with ``rerank(query, docs) -> list[float]`` works as a reranker.
"""
from __future__ import annotations

from typing import Sequence


class FastEmbedReranker:
    """Local cross-encoder reranker via fastembed (ONNX, no API key)."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, docs: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(docs))]
