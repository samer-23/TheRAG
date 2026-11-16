# Martian Map RAG

A retrieval-augmented generation system over the data sources for the NASA
Space Apps 2026 challenge **"Interplanetary Survival Guide: Martian Map"**,
labeled with a consistent schema (source, mission, category, date/era, tags,
URL) so every answer can be filtered and cited.

## What data it pulls, and from where

| Connector | Source | What it contributes |
|---|---|---|
| `trek.py` | [NASA Mars Trek](https://trek.nasa.gov/mars/) search index | ~2,000 map layers, landing-site bookmarks, named features (nomenclature) and data products, with bounding boxes |
| `usgs_hub.py` | [USGS Astrogeology Hub](https://astrogeology-usgs.hub.arcgis.com/pages/interactive-maps-mars) (ArcGIS Hub API) | ~40 USGS Mars map datasets: geologic maps, landing-site maps, tile/feature services |
| `mars2020_raw.py` | [Mars 2020 raw images](https://mars.nasa.gov/mars2020/multimedia/raw-images/) JSON feed | The newest 500 Perseverance raw-image records (caption, camera, sol, date) |
| `science_nasa.py` | [Curiosity's proposed path up Mount Sharp](https://science.nasa.gov/resource/curiositys-proposed-path-up-mount-sharp/) | Page text describing the planned route |

No API keys are needed. The Trek and USGS pages are JavaScript apps, so the
connectors query the APIs behind them rather than scraping HTML.

## Labeling schema

Every document (see `connectors/common.py:make_doc`) has:

- `source` -- `nasa_mars_trek` | `usgs_astrogeology_hub` | `mars2020_raw_images` | `science_nasa_gov`
- `mission` -- e.g. `"Mars Trek"`, `"Mars 2020 - Perseverance"`, `"USGS Astrogeology Mars maps"`
- `category` -- `trek_nomenclature` | `trek_product` | `trek_dataset` | `trek_bookmark` | `map_dataset` | `raw_image` | `resource_page`
- `date` / `era` -- original date plus a decade bucket (`"2020s"`); Trek items are undated (`"unknown"`)
- `tags`, `url`, `id`, `fetched_at`

These become metadata fields on every indexed chunk, filterable at query time.

## Retrieval: hybrid BM25 + semantic embeddings

This was built against Python 3.14 (very new), where the usual local-embedding
stack (chromadb's bundled `onnxruntime`) has no prebuilt wheel yet and pip
hangs trying to resolve one. The index started out BM25-only --
`rank-bm25` is the same lexical ranking algorithm behind Elasticsearch/Lucene's
default search, pure Python, no compiled dependencies, installs anywhere.

This machine turned out to already have a working `torch` + CUDA install, so
`build_index.py` now also embeds every chunk with
`sentence-transformers/all-MiniLM-L6-v2` (384-dim, GPU-accelerated) and
`rag.py` fuses the BM25 ranking and the embedding-similarity ranking with
**reciprocal rank fusion** (RRF) -- a parameter-light way to combine two
different rankings without needing to normalize/weight their raw scores.
This catches both keyword matches ("Ingenuity final flight") and conceptual
matches with no shared vocabulary ("machines searching for ancient water" ->
Odyssey/MRO/Curiosity water-related articles). Exact metadata filters
(`--mission`, `--category`, `--era`) narrow the candidate pool before fusion.

Pass `--bm25-only` to fall back to pure lexical ranking (also the automatic
fallback if `vectorstore/embeddings.npy` doesn't exist, e.g. `sentence-
transformers` isn't installed). Override the embedding model with
`EMBEDDING_MODEL=<hf-model-id> python build_index.py`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```powershell
python ingest.py          # pull fresh data into data/processed/documents.jsonl
python build_index.py     # build BM25 + embeddings (GPU if available)
python main.py            # ask questions; the answer appears in a frame
```

`main.py` shows a `Question>` prompt and prints the answer in a box, with the
numbered sources underneath. The answer is written by an open-source model
(`LLM_MODEL` in `rag.py`, default `Qwen/Qwen2.5-7B-Instruct`) called through the
Hugging Face Inference API, so nothing large is downloaded. Create a free token
at <https://huggingface.co/settings/tokens> and put it in `.env`:

```
HF_TOKEN=hf_...
```

## Layout

```
connectors/        one module per data source, common.py has the shared schema
data/raw/           untouched JSON responses as fetched (audit trail)
data/processed/     documents.jsonl -- the labeled, flattened corpus
vectorstore/        bm25_index.pkl + embeddings.npy -- built by build_index.py
ingest.py           runs every connector, rebuilds documents.jsonl
build_index.py      chunks documents.jsonl, builds BM25 index + embeddings
rag.py              hybrid retrieval + local LLM answer generation (library)
main.py             question prompt -> framed answer
```

Re-running `python ingest.py` followed by `python build_index.py` refreshes
the whole index with whatever the sources have published since the last run.
