import argparse

from lib.evaluate import llm_evaluation
from lib.hybrid_search import (
    normalize_scores,
    rff_search_command,
    weighted_search_command,
)
from lib.search_utils import DEFAULT_ALPHA, DEFAULT_K, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="Normalizes a list of Scores")
    normalize.add_argument(
        "scores", nargs="*", type=float, help="A list of scores to normalize"
    )

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
    rff_search.add_argument(
        "-k", type=int, nargs="?", default=DEFAULT_K, help="k value weight"
    )
    rff_search.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit query results",
    )

    rff_search.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    rff_search.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Resuls reranking method",
    )

    rff_search.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate results using LLM",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            result = weighted_search_command(
                query=args.query, alpha=args.alpha, limit=args.limit
            )

            for i, res in enumerate(result["results"], 1):
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
            result = rff_search_command(
                query=args.query,
                k=args.k,
                enhance=args.enhance,
                rerank=args.rerank_method,
                limit=args.limit,
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )
            if result["rerank_method"]:
                print(
                    f"Re-ranking top {len(result['results'])} results using {result['rerank_method']} method"
                )

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if res.get("individual_score"):
                    print(f"   Re-rank Score: {res.get('individual_score', 0):.3f}")
                if res.get("batch_rank"):
                    print(f"   Re-rank Score: {res.get('batch_rank', 0):.3f}")
                if res.get("cross_encoder_score"):
                    print(
                        f"   Cross Encoder Score: {res.get('cross_encoder_score', 0):.3f}"
                    )
                print(f"   RFF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_rank" in metadata and "semantic_rank" in metadata:
                    print(
                        f"   BM25 Rank: {metadata.get('bm25_rank', 0)}, Semantic Rank: {metadata.get('semantic_rank', 0)}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
            if args.evaluate:
                eval_results = llm_evaluation(result)

                for i, eval in enumerate(eval_results):
                    print(f"{i}. {eval['title']}: {eval['llm_relevance']}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
