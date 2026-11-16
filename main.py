"""Ask a question about Mars; the answer appears in a frame, written by a local
open-source Hugging Face model (see rag.LLM_MODEL) running on this machine's
GPU/CPU from the retrieved NASA/USGS sources -- no API, no billing.

    python main.py

Rebuild the data first (only when you want fresh data):
    python ingest.py && python build_index.py
"""
from __future__ import annotations

import shutil
import textwrap

import rag


def framed(title: str, text: str) -> None:
    width = min(shutil.get_terminal_size((100, 20)).columns, 100)
    inner = width - 4
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(para, inner, subsequent_indent="    " if para.startswith("[") else "") or [""])
    label = f" {title} "
    print("\n+" + label + "-" * (width - 2 - len(label)) + "+")
    for ln in lines:
        print("| " + ln.ljust(inner) + " |")
    print("+" + "-" * (width - 2) + "+")


def main() -> None:
    print(f"Loading index and {rag.LLM_MODEL} (runs locally; first time downloads it)...")
    index = rag.load_index()
    try:
        rag.warmup(index)
    except RuntimeError as e:  # model failed to load
        raise SystemExit(f"\n{e}")
    print("Ready. Type a question about Mars (blank line or 'exit' to quit).")

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit", "q"}:
            break
        framed("QUESTION", question)
        try:
            text, hits = rag.answer(index, question)
        except Exception as e:  # keep the prompt alive on any model/runtime error
            text, hits = f"Error: {e}", []
        if hits:
            text += "\n\nSources:\n" + "\n".join(
                f"[{i}] {h['meta'].get('mission')} -- {h['meta'].get('url')}" for i, h in enumerate(hits, 1)
            )
        framed("ANSWER", text)


if __name__ == "__main__":
    main()
