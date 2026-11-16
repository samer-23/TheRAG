"""NASA Mars Trek (https://trek.nasa.gov/mars/) -- searchable catalog of map layers,
landing-site bookmarks, nomenclature (named features) and products.

Uses the same Solr index the Trek web app queries.
"""
from __future__ import annotations

import re

from common import append_processed, make_doc, make_session, save_raw

SEARCH_URL = "https://trek.nasa.gov/mars/TrekServices/ws/index/eq/searchItems"
PAGE_URL = "https://trek.nasa.gov/mars/"
SOURCE = "nasa_mars_trek"
ROWS = 500


def _clean(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "")


def run() -> list[dict]:
    session = make_session()
    items, start = [], 0
    while True:
        r = session.get(
            SEARCH_URL,
            params={"rows": ROWS, "start": start, "q": "*", "facets": ""},
            timeout=60,
        )
        r.raise_for_status()
        resp = r.json()["response"]
        items.extend(resp["docs"])
        start += ROWS
        if start >= resp["numFound"]:
            break
    save_raw("trek_items", items)

    docs = []
    for it in items:
        label = it.get("productLabel") or it.get("title") or ""
        desc = _clean(it.get("description", ""))
        cats = [it.get(f"productCat{n}") for n in (1, 2, 3)]
        cats = [c for c in cats if c]
        if not (label or desc):
            continue
        text = f"{label}. {desc}"
        if cats:
            text += f" Category: {' > '.join(cats)}."
        if it.get("bbox"):
            text += f" Bounding box (lon/lat): {it['bbox']}."
        item_type = it.get("itemType", "item")
        docs.append(
            make_doc(
                text=text,
                title=label,
                source=SOURCE,
                mission="Mars Trek",
                category=f"trek_{item_type}",
                date=None,
                url=PAGE_URL,
                tags=[item_type, *cats],
                extra={"bbox": it.get("bbox")},
            )
        )
    append_processed(docs)
    print(f"  Mars Trek: {len(docs)} documents")
    return docs
