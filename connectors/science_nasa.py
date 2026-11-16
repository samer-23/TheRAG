"""NASA Science resource page: Curiosity's proposed path up Mount Sharp.

https://science.nasa.gov/resource/curiositys-proposed-path-up-mount-sharp/
"""
from __future__ import annotations

from bs4 import BeautifulSoup

from common import append_processed, make_doc, make_session, save_raw

URL = "https://science.nasa.gov/resource/curiositys-proposed-path-up-mount-sharp/"
SOURCE = "science_nasa_gov"


def run() -> list[dict]:
    session = make_session()
    r = session.get(URL, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = (soup.find("h1") or soup.title).get_text(" ", strip=True)
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    parts = []
    if meta and meta.get("content"):
        parts.append(meta["content"])
    main = soup.find("main") or soup
    for p in main.find_all("p"):
        t = p.get_text(" ", strip=True)
        if len(t) > 60:
            parts.append(t)
    text = " ".join(dict.fromkeys(parts))  # dedupe, keep order
    save_raw("science_nasa_curiosity_path", {"url": URL, "title": title, "text": text})

    docs = []
    if text:
        docs.append(
            make_doc(
                text=text,
                title=title,
                source=SOURCE,
                mission="MSL - Curiosity",
                category="resource_page",
                date=None,
                url=URL,
                tags=["Curiosity", "Mount Sharp", "route"],
            )
        )
    append_processed(docs)
    print(f"  science.nasa.gov resource: {len(docs)} documents")
    return docs
