#!/usr/bin/env python3
import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    truncate_text,
)
from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)
from sklearn.externals.array_api_compat.torch import chunk


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    verify_parser = subparsers.add_parser("verify", help="Verify Model")
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify Embeddings"
    )
    embed_parser = subparsers.add_parser("embed_text", help="Embed Text")
    embed_parser.add_argument("text", type=str, help="Text to Embed")

    embed_query = subparsers.add_parser("embedquery", help="Embed Query")
    embed_query.add_argument("query", type=str, help="Query to Embed")

    search = subparsers.add_parser("search", help="Search")
    search.add_argument("query", type=str, help="Search string")
    search.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit query results",
    )

    chunk = subparsers.add_parser("chunk", help="Chunk Text")
    chunk.add_argument("text", type=str, help="The text to chunk")
    chunk.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_SIZE,
        help="Set chunk size",
    )

    args = parser.parse_args()
    match args.command:
        case "search":
            model = SemanticSearch()
            movies = load_movies()
            model.load_or_create_embeddings(movies)
            results = model.search(args.query, args.limit)
            for idx, result in enumerate(results):
                print(
                    f"{idx + 1}. {result['title']} (score: {result['score']:.4f})\n   {truncate_text(result['description'])}\n"
                )
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size)
            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks):
                print(f"{idx +1}. {chunk}")

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case _:
            parser.print_help()


def chunk_text(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


if __name__ == "__main__":
    main()
