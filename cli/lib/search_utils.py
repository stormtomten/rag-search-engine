import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_LENGTH = 0
DEFAULT_K = 60

DEFAULT_MAX_CHUNK_SIZE = 4
DEFAULT_ALPHA= 0.5
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3
HYBRID_RESULT_PADDING = 500

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORD_PATH, "r") as f:
        return f.read().splitlines()


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."

def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

