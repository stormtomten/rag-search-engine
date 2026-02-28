#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    verify_parser = subparsers.add_parser("verify", help="Verify Model")
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify Embeddings"
    )
    embed_parser = subparsers.add_parser("embed_text", help="Embed Text")
    embed_parser.add_argument("text", type=str, help="Text to Embed")

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
