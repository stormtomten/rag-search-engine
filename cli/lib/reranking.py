import json
import os
from time import sleep
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from .search_utils import DEFAULT_SEARCH_LIMIT

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-26b-a4b-it"
#model = "gemma-4-31b-it"
#model = "gemini-2.5-flash"


def individual_rerank(query:str, docs: List[Dict], limit: int) -> List[Dict]:
    rerank = []

    for doc in docs:
        prompt = (
            f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:"""
                )
        response = call_with_retry(lambda: client.models.generate_content(model=model, contents=prompt))
        score_text = (response.text or "").strip()
        score = int(score_text)
        rerank.append({**doc, "individual_score": score})
        sleep(8)

    rerank.sort(key=lambda x: x["individual_score"], reverse=True)
    return rerank[:limit]

def batch_rerank(query:str, docs: List[Dict], limit: int) -> List[Dict]:

    lines = []
    for d in docs:
        summary = d["document"][:200] + "..." if len(d["document"]) > 200 else d["summary"]
        lines.append(f"ID: {d['id']} | Title: {d['title']} | Summary: {summary}")

    doc_list_str = "\n".join(lines)

    prompt = (
        f"""Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:"""
                        )
    response = call_with_retry(lambda: client.models.generate_content(model=model, contents=prompt))
    reranked_ids = json.loads(response.text)

    by_id = {r["id"]: r for r in docs}
    for i, doc_id in enumerate(reranked_ids):
        by_id[doc_id]["batch_rank"] = i + 1


    rerank = sorted(by_id.values(), key=lambda x: x["batch_rank"], reverse=False)
    return rerank[:limit]


def rerank_results(query: str, docs: List[Dict], method: Optional[str] = None, limit: int = DEFAULT_SEARCH_LIMIT) -> List[Dict]:
    match method:
        case "individual":
            return individual_rerank(query=query, docs=docs, limit=limit)

        case "batch":
            return batch_rerank(query=query, docs=docs, limit=limit)


        case _:
            return docs


def call_with_retry(fn, retries=3, delay=4):
    for attempt in range(retries):
        try:
            return fn()
        except genai.errors.ServerError:
            if attempt == retries - 1:
                raise
            sleep(delay * (attempt + 1))
