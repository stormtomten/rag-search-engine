#!/usr/bin/env python3
import argparse

from lib.search_utils import (DEFAULT_CHUNK_SIZE, DEFAULT_MAX_CHUNK_SIZE,
                              DEFAULT_OVERLAP_LENGTH, DEFAULT_SEARCH_LIMIT,
                              load_movies, truncate_text)
from lib.semantic_search import (ChunkedSemanticSearch, SemanticSearch,
                                 chunk_text, embed_query_text, embed_text,
                                 search_chunked_command, semantic_chunk,
                                 verify_embeddings, verify_model)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    verify_parser = subparsers.add_parser("verify", help="Verify Model")
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify Embeddings"
    )
    embed_parser = subparsers.add_parser("embed_text", help="Embed Text")
    embed_parser.add_argument("text", type=str, help="Text to Embed")

    embed_query = subparsers.add_parser("embed_query", help="Embed Query")
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
    chunk.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_OVERLAP_LENGTH,
        help="Set overlap length",
    )

    semantic_chunking = subparsers.add_parser("semantic_chunk", help="Semantic Chunking")
    semantic_chunking.add_argument(
            "text",
            type=str,
            help="The text to semanticly chunk"
    )
    semantic_chunking.add_argument(
            "--max-chunk-size", type=int,nargs="?",default=DEFAULT_MAX_CHUNK_SIZE, help="Set Max chunk size")
    semantic_chunking.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_OVERLAP_LENGTH,
        help="Set overlap length",
   )

    subparsers.add_parser("embed_chunks", help="Embedding Chunks")

    search_chunked = subparsers.add_parser("search_chunked", help="Search by Chunks")
    search_chunked.add_argument("query", type=str, help="Search string")
    search_chunked.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit query results",
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

        case "search_chunked":
            results = search_chunked_command(query=args.query, limit=args.limit)

            for i, result in enumerate(results):
                print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
                print(f"   {result["document"]}...")

        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(
                f"Chunking {len(args.text)} characters, with {args.overlap} in overlap"
            )
            for idx, chunk in enumerate(chunks):
                print(f"{idx + 1}. {chunk}")

        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(
                f"Semantically chunking {len(args.text)} characters, with {args.overlap} in overlap"
            )
            for idx, chunk in enumerate(chunks):
                print(f"{idx + 1}. {chunk}")

        case "embed_chunks":
            model = ChunkedSemanticSearch()
            movies = load_movies()
            embeddings = model.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
