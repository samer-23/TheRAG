"""Mars RAG core: hybrid retrieval (BM25 + sentence embeddings, fused with
reciprocal rank fusion) and grounded answer generation by an open-source
Hugging Face model run LOCALLY on this machine's GPU/CPU via transformers --
no proprietary API, no Inference Providers billing, just a one-time model
download. HF_TOKEN (env var or .env) is only needed if you point HF_LLM_MODEL
at a gated model; the default is ungated.

    index = load_index(); warmup(index)
    text, hits = answer(index, "Where is the Curiosity landing site?")
"""
from __future__ import annotations

import functools
import os
import pickle
import re
from pathlib import Path

# Hugging Face's Xet CDN is blocked/unreliable on some networks; use plain HTTPS downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402

try:  # optional: HF_TOKEN / HF_LLM_MODEL from a .env next to this file
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from build_index import tokenize, INDEX_PATH, EMBEDDINGS_PATH  # noqa: E402

# Small enough to run locally on an 8GB-class GPU in bf16/fp16. Override with
# HF_LLM_MODEL for a bigger model if you have the VRAM (e.g. Qwen2.5-7B-Instruct).
LLM_MODEL = os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = 300
TOP_K = 5
CHUNK_CHARS = 600  # per-source text sent to the LLM (keeps prompts short and fast)
RRF_K = 60  # reciprocal-rank-fusion smoothing constant

SYSTEM_PROMPT = (
    "You are a Mars mission-planning assistant. Answer using ONLY the numbered source "
    "excerpts provided. Cite sources inline like [1], [2]. If the excerpts do not contain "
    "the answer, say you cannot find it in the sources. Be concise (a few sentences)."
)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise SystemExit("No index found -- run `python ingest.py` then `python build_index.py`.")
    with INDEX_PATH.open("rb") as f:
        index = pickle.load(f)
    index["embeddings"] = np.load(EMBEDDINGS_PATH) if EMBEDDINGS_PATH.exists() else None
    return index


@functools.lru_cache(maxsize=1)
def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=_device())


@functools.lru_cache(maxsize=1)
def get_llm():
    """Load the local generation model+tokenizer once (downloads on first use)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            token=token,
            dtype=torch.bfloat16 if _device() == "cuda" else torch.float32,
        ).to(_device())
        model.eval()
        return tokenizer, model
    except Exception as e:  # normalize to RuntimeError so callers only need one except clause
        raise RuntimeError(
            f"Could not load local model '{LLM_MODEL}': {e}\n"
            "If this is a gated model, put HF_TOKEN=hf_... in a .env file next to main.py."
        ) from e


def warmup(index: dict) -> None:
    """Load the embedding model and the local LLM up front (first call is slow; later ones aren't)."""
    if index.get("embeddings") is not None and index.get("embedding_model"):
        get_embedder(index["embedding_model"])
    get_llm()


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank of each element, best (highest) score first."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    return ranks


def _dedupe_key(text: str) -> str:
    # Near-identical captions (same text, different timestamps/ids) collapse to one key.
    return re.sub(r"[\d:.\-_]+", "", text.lower())[:160]


def retrieve(
    index: dict,
    query: str,
    k: int = TOP_K,
    *,
    mission: str | None = None,
    category: str | None = None,
    source: str | None = None,
    era: str | None = None,
    bm25_only: bool = False,
) -> list[dict]:
    metas = index["metadatas"]
    n = len(metas)
    filters = {"mission": mission, "category": category, "source": source, "era": era}
    active = {f: v for f, v in filters.items() if v}
    cand = (
        np.arange(n)
        if not active
        else np.array([i for i, m in enumerate(metas) if all(m.get(f) == v for f, v in active.items())], dtype=np.int64)
    )
    if cand.size == 0:
        return []

    bm25 = np.asarray(index["bm25"].get_scores(tokenize(query)))[cand]
    emb = index.get("embeddings")
    hybrid = (not bm25_only) and emb is not None and bool(index.get("embedding_model"))

    if hybrid:
        qvec = get_embedder(index["embedding_model"]).encode([query], normalize_embeddings=True)[0]
        cos = (emb if cand.size == n else emb[cand]) @ qvec
        score = 1.0 / (RRF_K + _ranks(bm25) + 1) + 1.0 / (RRF_K + _ranks(cos) + 1)
    else:
        cos = None
        score = np.where(bm25 > 0, bm25, -1.0)

    hits, seen = [], set()
    for j in np.argsort(-score, kind="stable"):
        if not hybrid and score[j] < 0:
            break
        i = int(cand[j])
        key = _dedupe_key(index["texts"][i])
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "text": index["texts"][i],
                "meta": metas[i],
                "score": float(score[j]),
                "bm25_score": float(bm25[j]),
                "embedding_score": float(cos[j]) if cos is not None else None,
            }
        )
        if len(hits) == k:
            break
    return hits


def generate_answer(query: str, hits: list[dict]) -> str:
    import torch

    context = "\n\n".join(
        f"[{i}] ({h['meta'].get('mission')}): {h['text'][:CHUNK_CHARS]}" for i, h in enumerate(hits, 1)
    )
    tokenizer, model = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy -- deterministic, grounded answers
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer(index: dict, query: str, **filters) -> tuple[str, list[dict]]:
    """Retrieve the best sources for `query` and generate a cited answer from them."""
    hits = retrieve(index, query, **filters)
    if not hits:
        return "No matching sources found. Try different keywords.", []
    return generate_answer(query, hits), hits
