#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
    build,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from lib.search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds index cache")

    frequency_parser = subparsers.add_parser(
        "tf", help="Fetche term frequency for a term"
    )
    frequency_parser.add_argument("doc_id", type=int, help="Id of the document")
    frequency_parser.add_argument(
        "term", type=str, help="The term you are searching for"
    )

    idf_parser = subparsers.add_parser(
        "idf", help="Calculates the inverse of document frequency"
    )
    idf_parser.add_argument("term", type=str, help="The term you are looking up")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Fetche term frequency for a term"
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Id of the document")
    tf_idf_parser.add_argument("term", type=str, help="The term you are searching for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit query results",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "tf":
            results = tf_command(args.doc_id, args.term)
            print(results)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":

            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            bm25search = bm25search_command(args.query, args.limit)
            count = 0
            for result in bm25search:
                print(
                    f"{count}. ({result['id']}) {result['title']} - Score: {result['score']:.2f}"
                )
                count += 1

        case "build":
            build()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
