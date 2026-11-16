"""Build a hybrid BM25 + semantic-embedding index over
data/processed/documents.jsonl.

History: this environment's Python was initially too new for prebuilt wheels
of the usual embedding stack (onnxruntime / chromadb's default embedding
function hung trying to resolve a matching wheel), so the index started out
BM25-only (rank-bm25, pure Python -- the same ranking algorithm behind
Elasticsearch/Lucene's default search). torch + sentence-transformers later
turned out to already be installed and working (with CUDA), so this now also
builds real sentence embeddings and rag.py fuses both signals (reciprocal
rank fusion) at query time. If sentence-transformers isn't available, this
script still produces a working BM25-only index.

Chunking: most documents here are already short (one caption, one fact, one
news summary) so each document is one chunk; long documents are split on
sentence boundaries to stay under ~800 chars per chunk.

Usage:
    python build_index.py
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python build_index.py   # override model
"""
from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DOCS_PATH = ROOT / "data" / "processed" / "documents.jsonl"
STORE_DIR = ROOT / "vectorstore"
INDEX_PATH = STORE_DIR / "bm25_index.pkl"
EMBEDDINGS_PATH = STORE_DIR / "embeddings.npy"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 800
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def chunk_text(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > MAX_CHARS and current:
            chunks.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def load_documents() -> list[dict]:
    if not DOCS_PATH.exists():
        raise SystemExit(f"{DOCS_PATH} not found -- run `python ingest.py` first.")
    docs = []
    with DOCS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} source documents from {DOCS_PATH}")

    ids, texts, metadatas, tokenized = [], [], [], []
    for doc in tqdm(docs, desc="chunking"):
        for i, chunk in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['id']}_{i}")
            texts.append(chunk)
            meta = {k: v for k, v in doc.items() if k != "text"}
            meta["chunk_index"] = i
            metadatas.append(meta)
            tokenized.append(tokenize(chunk))

    print(f"Building BM25 index over {len(ids)} chunks...")
    bm25 = BM25Okapi(tokenized)

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    embedding_model_name = None
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        embedding_model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding {len(texts)} chunks with '{embedding_model_name}' on {device}...")
        model = SentenceTransformer(embedding_model_name, device=device)
        embeddings = model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Saved embeddings {embeddings.shape} -> {EMBEDDINGS_PATH}")
    except ImportError:
        print("sentence-transformers not available -- building BM25-only index.")
        embeddings = None

    with INDEX_PATH.open("wb") as f:
        pickle.dump(
            {
                "ids": ids,
                "texts": texts,
                "metadatas": metadatas,
                "bm25": bm25,
                "embedding_model": embedding_model_name,
            },
            f,
        )

    print(f"Done. Indexed {len(ids)} chunks -> {INDEX_PATH}")
    if embeddings is not None:
        print("Hybrid (BM25 + embeddings) retrieval is ready.")


if __name__ == "__main__":
    main()
