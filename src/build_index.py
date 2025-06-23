"""
build_index.py
--------------
Ingest one or more PDF doctrine files, chunk into 128-token (~word) windows,
embed with MiniLM-L6 (384-d), and write a FAISS cosine-similarity index plus
pickle of the original snippets.

Outputs are stored under ./data/ :

    data/index.faiss
    data/snippets.pkl
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 128
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INDEX_PATH = DATA_DIR / "index.faiss"
SNIPPET_PATH = DATA_DIR / "snippets.pkl"


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_words(text: str, size: int = CHUNK_TOKENS) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pdfs",
        nargs="*",
        help="PDF paths (defaults to docs/*.pdf)",
    )
    args = ap.parse_args()

    pdf_paths = [Path(p) for p in (args.pdfs or glob.glob("docs/*.pdf"))]
    if not pdf_paths:
        raise SystemExit("No PDF files found -- pass explicit paths or add to docs/.")

    print(f"[INFO] Ingesting {len(pdf_paths)} PDFs ...")

    snippets: List[str] = []
    for pdf in pdf_paths:
        text = pdf_to_text(pdf)
        chunks = chunk_words(text, CHUNK_TOKENS)
        snippets.extend(chunks)
        print(f"    * {pdf.name}: {len(chunks)} chunks")

    model = SentenceTransformer(EMBED_MODEL)
    print("[INFO] Embedding ...")
    embeddings = model.encode(
        snippets, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
    ).astype(
        "float32"
    )  # FAISS prefers float32

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine because vectors are L2-normalized
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with open(SNIPPET_PATH, "wb") as f:
        pickle.dump(snippets, f)

    print(f"[DONE] {len(snippets)} chunks indexed.")
    print(f"       -> {INDEX_PATH}")
    print(f"       -> {SNIPPET_PATH}")


if __name__ == "__main__":
    main()
