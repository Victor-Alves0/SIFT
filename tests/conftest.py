"""Shared fixtures. Uses a deterministic offline embedder so unit tests never
download a model and stay fast/repeatable."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sift import Sift

_DIM = 256


def _vec(text: str) -> np.ndarray:
    """Stable bag-of-words hashing vector (real token overlap, no network)."""
    v = np.zeros(_DIM, dtype=np.float32)
    for tok in text.lower().replace(".", " ").replace(":", " ").split():
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _DIM
        v[idx] += 1.0
    return v


class FakeEmbedder:
    def embed(self, texts):
        return [_vec(t) for t in texts]


@pytest.fixture
def sift() -> Sift:
    s = Sift(embedder=FakeEmbedder())

    @s.tool(
        "google_workspace.gmail.read",
        description="Read emails from the inbox newest first",
        params={"q": "string:o::search query", "m": "number:o:10:max results"},
        returns=["id", "subject", "from", "snippet", "date"],
    )
    def _read(q="is:unread", m=10):
        return {"id": "1", "subject": "Hi", "from": "a@b.c", "snippet": "...",
                "date": "2026-06-30", "body": "SHOULD BE FILTERED OUT"}

    @s.tool(
        "google_workspace.gmail.send",
        description="Send a new email message",
        params={"to": "string:n::recipient", "subject": "string:n::subject", "body": "string:n::body"},
        returns=["id", "status"],
        risk=True,
    )
    def _send(to, subject, body):
        return {"id": "2", "status": f"sent to {to}"}

    @s.tool(
        "local.filesystem.read",
        description="Read a local text file from disk",
        params={"path": "string:n::absolute file path"},
        returns=["path", "content"],
    )
    def _readfile(path):
        return {"path": path, "content": "hello"}

    s.build_index()
    return s
