"""
Interactive CLI for the Mini RAG chatbot.

Commands:
  help              show commands
  status            show configuration + index size
  reload            rebuild the index from documents/
  mode <name>       switch retrieval mode: dense | bm25 | hybrid | hybrid_rerank
  quit / exit       leave

Anything else is treated as a question.
"""
from __future__ import annotations

import os

from rag import RAG, Config

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def banner() -> None:
    print("\n" + "=" * 62)
    print("        MINI RAG CHATBOT  —  hybrid retrieval + reranking")
    print("=" * 62)
    print("Type 'help' for commands, 'quit' to exit.")
    print("=" * 62)


def show_help() -> None:
    print(
        "\nCommands:\n"
        "  help            show this message\n"
        "  status          show configuration and index size\n"
        "  reload          rebuild the index from the documents/ folder\n"
        "  mode <name>     retrieval mode: dense | bm25 | hybrid | hybrid_rerank\n"
        "  quit / exit     leave\n"
        "\nOr just type a question.\n"
    )


def show_status(rag: RAG, mode: str) -> None:
    cfg = rag.cfg
    print(
        f"\nStatus:\n"
        f"  chat model    : {cfg.chat_model}\n"
        f"  embed model   : {cfg.embed_model}\n"
        f"  ollama url    : {cfg.base_url}\n"
        f"  retrieval mode: {mode}\n"
        f"  chunk size    : {cfg.chunk_size} (overlap {cfg.chunk_overlap})\n"
        f"  top_k / pool  : {cfg.top_k} / {cfg.pool}\n"
        f"  indexed chunks: {len(rag.chunks)}\n"
    )


def render(result: dict) -> None:
    print(f"\nBot  (mode: {result['mode']}):\n")
    print(result["answer"].strip() or "(empty answer)")

    if result.get("contradictions"):
        print("\n⚠ Contradictions detected:")
        for c in result["contradictions"]:
            print(f"  - {c}")

    if result.get("sources"):
        print("\nSources:")
        used = set(result.get("used_sources") or [])
        for s in result["sources"]:
            mark = "*" if s["n"] in used else " "
            print(f" {mark}[{s['n']}] {s['source']} :: {s['section']}")
        if used:
            print(" (* = cited in the answer)")


def main() -> None:
    rag = RAG(Config())
    mode = "hybrid_rerank"

    banner()
    print("\nInitializing index...")
    try:
        n = rag.build_index()
        print(f"Ready — {n} chunk(s) indexed.\n")
    except Exception as exc:  # noqa: BLE001
        print(f"! Could not build index: {exc}")
        print("  Make sure Ollama is running:  ollama serve")
        return

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            low = user.lower()

            if low in ("quit", "exit"):
                print("Goodbye!")
                break
            if low == "help":
                show_help()
                continue
            if low == "status":
                show_status(rag, mode)
                continue
            if low == "reload":
                print("Rebuilding index...")
                rag.build_index(force=True)
                continue
            if low.startswith("mode"):
                parts = user.split()
                if len(parts) == 2 and parts[1] in MODES:
                    mode = parts[1]
                    print(f"Retrieval mode -> {mode}")
                else:
                    print(f"Usage: mode <{' | '.join(MODES)}>")
                continue
            if low == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                banner()
                continue

            try:
                result = rag.answer(user, mode=mode)
                render(result)
                print()
            except Exception as exc:  # noqa: BLE001
                print(f"! Error: {exc}")
                print("  Is Ollama running with the configured models?")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
