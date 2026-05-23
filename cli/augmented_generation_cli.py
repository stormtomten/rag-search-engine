import argparse

from lib.augmented_generation import (
    citations_command,
    question_command,
    rag_command,
    summary_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser(
        "summarize", help="Perform a summary of search results"
    )
    summary_parser.add_argument("query", type=str, help="Search query for Summary")
    summary_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of responses",
    )

    citation_parser = subparsers.add_parser(
        "citations", help="LLM assisted citation search"
    )
    citation_parser.add_argument("query", type=str, help="Search query for Citations")
    citation_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of responses",
    )

    question_parser = subparsers.add_parser("question", help="Ask the LLM a  question")
    question_parser.add_argument("question", type=str, help="Your question")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="The maximum number of documents",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            answer = rag_command(question)

            print("Search Results:")
            for a in answer["search_results"]:
                print(f"- {a['title']}")
            print(f"\nRAG Response:\n{answer['answer']}")

        case "summarize":
            query = args.query
            limit = args.limit
            citations = summary_command(query, limit)

            print("Search Results:")
            for a in citations["search_results"]:
                print(f"- {a['title']}")
            print(f"\nLLM Summary:\n{citations['summary']}\n")

        case "citations":
            question = args.query
            limit = args.limit
            citations = citations_command(question, limit)

            print("Search Results:")
            for a in citations["search_results"]:
                print(f"- {a['title']}")
            print(f"\nLLM Summary:\n{citations['citations']}\n")

        case "question":
            question = args.question
            limit = args.limit
            answer = question_command(query=question, limit=limit)

            print("Search Results:")
            for a in answer["search_results"]:
                print(f"- {a['title']}")
            print(f"\nAnswer:\n{answer['answer']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
