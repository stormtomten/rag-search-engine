import argparse

from lib.hybrid_search import rff_search_command
from lib.search_utils import DEFAULT_K, load_golden


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here

    test_cases = load_golden()


    results = []
    for test_case in test_cases:
        test_result = rff_search_command(test_case.get('query', ''), k=DEFAULT_K, limit=limit)
        retrieved_docs = [ r["title"] for r in test_result.get('results', 'None')]
        relevant_retrieved = len(set(test_case["relevant_docs"]) & set(retrieved_docs))

        precision = relevant_retrieved / limit
        recall = relevant_retrieved / len(test_case["relevant_docs"])
        f1 = 2 * (precision * recall) / (precision + recall)

        results.append({
                "query": test_case["query"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retrieved_docs": retrieved_docs,
                "relevant_docs": test_case["relevant_docs"],
                })

    print(f"k={limit}\n")

    for result in results:
        print(f"- Query: {result.get('query', 'None')}")
        print(f"\t-\tPrecision@{limit}: {result.get('precision', 0):.4f}")
        print(f"\t-\tRecall@{limit}: {result.get('recall', 0):.4f}")
        print(f"\t-\tF1 Score: {result.get('f1', 0):.4f}")
        print(f"\t-\tRetrieved: {", ".join(result.get('retrieved_docs', ''))}")
        print(f"\t-\tRelevant: {", ".join(result.get('relevant_docs', ''))}")
        print("\n")



if __name__ == "__main__":
    main()
