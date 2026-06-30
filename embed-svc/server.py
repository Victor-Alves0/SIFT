"""
SIFT — Embedding sidecar.

Microserviço HTTP que expõe embeddings semânticos para o core (Go) usar no
`search_tools`. Roda 100% local via fastembed (ONNX), sem chave de API.

Endpoints:
  GET  /health            -> {"status": "ok", "model": "...", "dim": N}
  POST /embed  {"texts":[...]}  -> {"vectors": [[...], ...], "dim": N}

Config por env:
  SIFT_EMBED_MODEL  (default: BAAI/bge-small-en-v1.5)  modelo fastembed
  SIFT_EMBED_HOST   (default: 127.0.0.1)
  SIFT_EMBED_PORT   (default: 8088)

Por que um sidecar separado? O OpenRouter não expõe endpoint de embeddings
confiável, e os melhores modelos locais de embedding vivem no ecossistema
Python. O core fala com este serviço por HTTP — mantendo o Go leve e o sistema
acoplável (você pode trocar este sidecar por qualquer outro que respeite o
mesmo contrato).
"""
from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

MODEL_NAME = os.getenv("SIFT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
HOST = os.getenv("SIFT_EMBED_HOST", "127.0.0.1")
PORT = int(os.getenv("SIFT_EMBED_PORT", "8088"))

app = FastAPI(title="SIFT Embed Sidecar")

# Carregado preguiçosamente no startup (faz download do modelo na 1ª vez).
_model: TextEmbedding | None = None
_dim: int = 0


class EmbedRequest(BaseModel):
    texts: List[str]


def get_model() -> TextEmbedding:
    global _model, _dim
    if _model is None:
        print(f"[sift-embed] carregando modelo '{MODEL_NAME}' ...", flush=True)
        _model = TextEmbedding(model_name=MODEL_NAME)
        # descobre a dimensão embedando uma string trivial
        probe = next(iter(_model.embed(["probe"])))
        _dim = len(probe)
        print(f"[sift-embed] pronto. dim={_dim}", flush=True)
    return _model


@app.on_event("startup")
def _startup() -> None:
    get_model()


@app.get("/health")
def health() -> dict:
    get_model()
    return {"status": "ok", "model": MODEL_NAME, "dim": _dim}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    model = get_model()
    vectors = [vec.tolist() for vec in model.embed(req.texts)]
    return {"vectors": vectors, "dim": _dim}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
