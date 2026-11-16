"""USGS Astrogeology Hub (https://astrogeology-usgs.hub.arcgis.com/pages/interactive-maps-mars).

The page itself is JavaScript-rendered, so we query the ArcGIS Hub search API
that backs it and keep Mars items published by U.S. Geological Survey accounts.
"""
from __future__ import annotations

from datetime import datetime, timezone

from common import append_processed, make_doc, make_session, save_raw

API = "https://astrogeology-usgs.hub.arcgis.com/api/v3/datasets"
PAGE_URL = "https://astrogeology-usgs.hub.arcgis.com/pages/interactive-maps-mars"
SOURCE = "usgs_astrogeology_hub"
QUERIES = ["mars usgs astrogeology", "mars geologic map", "mars"]
PAGES_PER_QUERY = 3
PAGE_SIZE = 100


def _is_usgs(a: dict) -> bool:
    blob = f"{a.get('orgName') or ''} {a.get('owner') or ''} {a.get('url') or ''}".lower()
    return "usgs" in blob or "geological survey" in blob or "astrogeology" in blob


def _iso(ms) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def run() -> list[dict]:
    session = make_session()
    seen: dict[str, dict] = {}
    for q in QUERIES:
        for page in range(1, PAGES_PER_QUERY + 1):
            r = session.get(
                API,
                params={"q": q, "page[size]": PAGE_SIZE, "page[number]": page},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                break
            for d in data:
                a = d["attributes"]
                if _is_usgs(a) and "mars" in f"{a.get('name')} {a.get('snippet')} {a.get('description')} {a.get('tags')}".lower():
                    seen[d["id"]] = a
    save_raw("usgs_hub_items", list(seen.values()))

    docs = []
    for item_id, a in seen.items():
        name = a.get("name") or a.get("title") or item_id
        text = f"{name}. {a.get('snippet') or ''} {a.get('description') or ''}"
        text += f" Type: {a.get('type')}."
        docs.append(
            make_doc(
                text=text,
                title=name,
                source=SOURCE,
                mission="USGS Astrogeology Mars maps",
                category="map_dataset",
                date=_iso(a.get("modified")),
                url=a.get("url") or f"https://astrogeology-usgs.hub.arcgis.com/datasets/{item_id}",
                tags=[a.get("type"), *(a.get("tags") or [])],
            )
        )
    append_processed(docs)
    print(f"  USGS Astrogeology Hub: {len(docs)} documents")
    return docs
