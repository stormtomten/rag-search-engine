import argparse

from lib.hybrid_search import (normalize_scores, rff_search_command,
                               weighted_search_command)
from lib.search_utils import DEFAULT_ALPHA, DEFAULT_K, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="Normalizes a list of Scores")
    normalize.add_argument("scores", nargs="*",type=float, help="A list of scores to normalize")

    weighted = subparsers.add_parser("weighted-search", help="Weighted Search")
    weighted.add_argument("query", type=str, help="Search string")
    weighted.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=DEFAULT_ALPHA,
        help="Alpha weighting between search results",
    )
    weighted.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit query results",
    )

    rff_search = subparsers.add_parser("rrf-search", help="A rff ranked Hybrid Search")
    rff_search.add_argument("query", type=str, help="Search string")
    rff_search.add_argument("-k", type=int,nargs="?", default=DEFAULT_K, help="k value weight")
    rff_search.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit query results",
    )


    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            results = weighted_search_command(query=args.query, alpha=args.alpha, limit=args.limit)

            for i, res in enumerate(results["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_rank" in metadata and "semantic_rank" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case "rrf-search":
            results = rff_search_command(query=args.query, k=args.k, limit=args.limit)

            for i, res in enumerate(results["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   RFF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25 Rank: {metadata['bm25_rank']:.3f}, Semantic Rank: {metadata['semantic_rank']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
