#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    build,
    idf_lookup,
    search_command,
    tf_search,
    tfidf_lookup,
)


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

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "tf":
            results = tf_search(args.doc_id, args.term)
            print(results)
        case "idf":
            idf = idf_lookup(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":

            tf_idf = tfidf_lookup(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "build":
            build()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
