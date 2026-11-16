"""Shared helpers for all NASA Mars data connectors.

Every connector produces a list of "documents" with a common, labeled
schema so the RAG index can filter/cite by source, mission, category,
and time period regardless of which NASA API the text came from.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "mars-rag-ingest/1.0"}


def make_session(retries: int = 4, backoff: float = 1.5) -> requests.Session:
    """A requests session with retries on connect/read timeouts and 5xx/429."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


def era_of(date_str: str | None) -> str:
    """Bucket an ISO-ish date string into a decade label for time filtering."""
    if not date_str:
        return "unknown"
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%B %d, %Y", "%Y"):
        try:
            year = datetime.strptime(date_str[: len(fmt) + 2], fmt).year
            break
        except ValueError:
            continue
    else:
        # last resort: grab the first 4-digit run that looks like a year
        import re

        m = re.search(r"(19|20)\d{2}", date_str)
        if not m:
            return "unknown"
        year = int(m.group(0))
    decade = (year // 10) * 10
    return f"{decade}s"


def doc_id(*parts: str) -> str:
    key = "|".join(parts)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def make_doc(
    *,
    text: str,
    title: str,
    source: str,
    mission: str,
    category: str,
    date: str | None,
    url: str,
    tags: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())  # strip stray HTML
    return {
        "id": doc_id(source, url, title, date or ""),
        "text": text,
        "title": title,
        "source": source,
        "mission": mission,
        "category": category,
        "date": date,
        "era": era_of(date),
        "url": url,
        "tags": sorted(set(t for t in tags if t)),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def save_raw(name: str, payload: Any) -> Path:
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_processed(docs: list[dict[str, Any]], filename: str = "documents.jsonl") -> Path:
    path = PROCESSED_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path
