"""Command-line interface for the Philippine medical-law RAG system.

Usage:
  python cli.py ingest [--reset]        Build/refresh the index from ./corpus
  python cli.py ask "your question"     Ask a one-off question
  python cli.py chat                    Interactive Q&A loop
  python cli.py status                  Show index + model health
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ragmed import config, llm, ocr, rag, vectorstore
from ragmed.ingest import ingest_corpus

console = Console()


def cmd_ingest(args):
    console.print(f"[bold]Corpus:[/] {config.CORPUS_DIR}")
    if args.reset:
        console.print("[yellow]Resetting collection (full rebuild)...[/]")
    summary = ingest_corpus(reset=args.reset)
    console.print(Panel.fit(
        f"Files found:     {summary['files_found']}\n"
        f"Files processed: {summary['files_processed']}\n"
        f"Chunks indexed:  {summary['chunks_indexed']}\n"
        f"Collection size: {summary['collection_size']}",
        title="Ingestion complete", border_style="green",
    ))
    if summary["skipped"]:
        console.print("[yellow]Skipped:[/]")
        for s in summary["skipped"]:
            console.print(f"  - {s}")


def _print_sources(sources):
    if not sources:
        return
    console.print("\n[bold cyan]Sources[/]")
    for i, s in enumerate(sources, 1):
        law = s.metadata.get("law", "")
        section = s.metadata.get("section", "")
        src = s.metadata.get("source", "")
        label = " · ".join(b for b in (law, section, src) if b)
        console.print(f"  [{i}] {label}  [dim](score {s.score:.3f})[/]")


def _check_llm():
    ok, msg = llm.is_available()
    if not ok:
        console.print(f"[red]LLM unavailable:[/] {msg}")
        return False
    return True


def cmd_ask(args):
    if not _check_llm():
        sys.exit(1)
    question = args.question
    console.print(f"[bold]Q:[/] {question}\n")
    gen, sources = rag.answer(question, stream=True)
    console.print("[bold green]A:[/] ", end="")
    buf = ""
    for token in gen:
        buf += token
        console.print(token, end="")
    console.print()
    _print_sources(sources)


def cmd_chat(args):
    if not _check_llm():
        sys.exit(1)
    console.print(Panel.fit(
        "Interactive mode. Ask about Philippine medical law.\n"
        "Type 'exit' or Ctrl-C to quit.",
        border_style="cyan",
    ))
    while True:
        try:
            question = console.input("[bold]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        gen, sources = rag.answer(question, stream=True)
        console.print("[bold green]Assistant:[/] ", end="")
        for token in gen:
            console.print(token, end="")
        console.print()
        _print_sources(sources)
        console.print()


def cmd_status(args):
    collection = vectorstore.get_collection()
    size = vectorstore.count(collection)
    ok, msg = llm.is_available()
    ocr_ok = ocr.is_available()
    ocr_line = "[green]OK[/]" if ocr_ok else (
        "[yellow]" + ocr.unavailable_reason() + "[/]" if config.OCR_ENABLED
        else "[dim]disabled[/]")
    # A reranker that silently fails to load looks identical to one that is
    # working badly, so report it rather than leaving it to be inferred.
    from ragmed import rerank
    rerank_line = ("[dim]disabled[/]" if not config.RERANK_ENABLED
                   else "[green]OK[/] " + config.RERANK_MODEL
                   if rerank.is_available()
                   else "[yellow]model unavailable — ranking falls back[/]")
    console.print(Panel.fit(
        f"Embedding model: {config.EMBED_MODEL}\n"
        f"LLM model:       {config.LLM_MODEL}\n"
        f"LLM server:      {'[green]OK[/]' if ok else '[red]' + msg + '[/]'}\n"
        f"OCR (scanned):   {ocr_line}\n"
        f"Reranker:        {rerank_line}\n"
        f"Collection:      {config.COLLECTION_NAME}\n"
        f"Chunks indexed:  {size}\n"
        f"Corpus dir:      {config.CORPUS_DIR}\n"
        f"Vector store:    {config.CHROMA_DIR}",
        title="RAG status", border_style="blue",
    ))
    if size == 0:
        console.print("[yellow]Index is empty. Run: python cli.py ingest[/]")


def main():
    parser = argparse.ArgumentParser(description="PH medical-law RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Index the corpus")
    p_ing.add_argument("--reset", action="store_true",
                       help="Wipe and rebuild the collection")
    p_ing.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive loop")
    p_chat.set_defaults(func=cmd_chat)

    p_stat = sub.add_parser("status", help="Show system status")
    p_stat.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
