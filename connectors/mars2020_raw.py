"""Mars 2020 Perseverance raw images (https://mars.nasa.gov/mars2020/multimedia/raw-images/).

Pulls the most recent raw-image records (caption, camera, sol, date) from the
JSON feed that powers that page.
"""
from __future__ import annotations

from common import append_processed, make_doc, make_session, save_raw

FEED = "https://mars.nasa.gov/rss/api/"
PAGE_URL = "https://mars.nasa.gov/mars2020/multimedia/raw-images/"
SOURCE = "mars2020_raw_images"
PER_PAGE = 100
PAGES = 5  # newest 500 images


def run() -> list[dict]:
    session = make_session()
    images = []
    for page in range(PAGES):
        r = session.get(
            FEED,
            params={
                "feed": "raw_images",
                "category": "mars2020",
                "feedtype": "json",
                "num": PER_PAGE,
                "page": page,
                "order": "sol desc",
            },
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json().get("images", [])
        if not batch:
            break
        images.extend(batch)
    save_raw("mars2020_raw_images", images)

    docs = []
    for im in images:
        cam = (im.get("camera") or {}).get("instrument", "unknown camera")
        caption = " ".join((im.get("caption") or "").split())
        sol = im.get("sol")
        date = im.get("date_taken_utc")
        text = f"{caption} Camera: {cam}. Sol {sol}. Credit: {im.get('credit')}."
        files = im.get("image_files") or {}
        docs.append(
            make_doc(
                text=text,
                title=im.get("imageid") or f"sol{sol}",
                source=SOURCE,
                mission="Mars 2020 - Perseverance",
                category="raw_image",
                date=(date or "")[:10] or None,
                url=files.get("large") or files.get("medium") or PAGE_URL,
                tags=[cam, f"sol_{sol}"],
                extra={"sol": sol, "camera": cam},
            )
        )
    append_processed(docs)
    print(f"  Mars 2020 raw images: {len(docs)} documents")
    return docs
