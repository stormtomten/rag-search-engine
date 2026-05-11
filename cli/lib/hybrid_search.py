import os
from typing import List, Optional

from .keyword_search import InvertedIndex
from .query_enhancement import enhance_query
from .reranking import rerank_results
from .search_utils import (DEFAULT_ALPHA, DEFAULT_K, DEFAULT_SEARCH_LIMIT,
                           HYBRID_RESULT_PADDING, format_search_result,
                           load_movies)
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=DEFAULT_SEARCH_LIMIT):
        bm25 = self._bm25_search(query=query, limit=limit * HYBRID_RESULT_PADDING)
        semantic = self.semantic_search.search_chunks(query=query, limit=limit * HYBRID_RESULT_PADDING)

        combined = combine_search_results(bm25_results=bm25, semantic_results=semantic, alpha=alpha)
        return combined[:limit]


    def rrf_search(self, query, k=DEFAULT_K, limit=DEFAULT_SEARCH_LIMIT):
        bm25 = self._bm25_search(query=query, limit=limit * HYBRID_RESULT_PADDING)
        semantic = self.semantic_search.search_chunks(query=query, limit=limit * HYBRID_RESULT_PADDING)

        combined = rff_combine_search_results(bm25_results=bm25, semantic_results=semantic)

        return combined[:limit]



def normalize_scores(scores: List[float]) ->List[float]:
    if not scores:
        return []

    max_score = max(scores)
    min_score = min(scores)
    if min_score == max_score:
        return [1.0000] * len(scores)

    results = []
    for score in scores:
        results.append((score - min_score) / (max_score - min_score))

    return results

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]
    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)



def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rff_score(rank: int, k: int):
    if rank == 0:
        return 0
    return 1 / (k + rank)

def rff_combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], k: int = DEFAULT_K
) -> list[dict]:

    combined_ranks = {}

    for i, result in enumerate(bm25_results):
        doc_id = result["id"]
        if doc_id not in combined_ranks:
            combined_ranks[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": i + 1,
                "semantic_rank": 0,
            }

    for i, result in enumerate(semantic_results):
        doc_id = result["id"]
        if doc_id not in combined_ranks:
            combined_ranks[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": 0,
                "semantic_rank": i + 1,
            }
        else:
            combined_ranks[doc_id]["semantic_rank"] = i + 1

    hybrid_results = []
    for doc_id, data in combined_ranks.items():
        score_value = rff_score(data["bm25_rank"], k) + rff_score(data["semantic_rank"], k)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

def rff_search_command(
        query: str, k: int = DEFAULT_K,enhance: Optional[str] = None,rerank: Optional[str] = None, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query=query, method= enhance)
        query = enhanced_query

    search_limit = limit

    if rerank:
        search_limit = search_limit * 5

    results = searcher.rrf_search(query, k, search_limit)

    if rerank:
        results = rerank_results(query=query, docs=results, method=rerank, limit=limit)
    


    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "rerank_method": rerank,
        "query": query,
        "k": k,
        "results": results[:limit],
    }

