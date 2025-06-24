from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.faiss"
SNIPPET_PATH = DATA_DIR / "snippets.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_artifacts():
    if not (INDEX_PATH.exists() and SNIPPET_PATH.exists()):
        raise RuntimeError("Index not built. Run `make index` first.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(SNIPPET_PATH, "rb") as f:
        snippets = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    return index, snippets, model


def retrieve(query: str, k: int = 3) -> List[Dict[str, float | str]]:
    """Return top-k snippet dicts: {"text": snippet, "score": similarity}."""
    index, snippets, model = _load_artifacts()
    q_emb = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, idx = index.search(q_emb, k)
    return [
        {"text": snippets[int(i)], "score": float(s)} for i, s in zip(idx[0], scores[0])
    ]
