import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import load_movies

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)
model = "gemma-4-26b-a4b-it"


def generate_answer(search_results, query, limit=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {context}

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def rag(query: str):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query=query, k=60, limit=5)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results=search_results, query=query, limit=5)

    return {"query": query, "search_results": search_results, "answer": answer}


def rag_command(query):
    return rag(query)


def generate_summary(search_results, query, limit=5):
    results = ""

    for result in search_results[:limit]:
        results += f"{result['title']}: {result['document']}\n\n"
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {results}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def summary(query, limit):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query=query, k=60, limit=limit)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    summary = generate_summary(search_results=search_results, query=query, limit=5)

    return {"query": query, "search_results": search_results, "summary": summary}


def summary_command(query, limit):
    return summary(query, limit)


def generate_citations(search_results, query, limit=5):
    documents = ""

    for result in search_results[:limit]:
        documents += f"{result['title']}: {result['document']}\n\n"
    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Hoopla, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {documents}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def citations(query, limit):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query=query, k=60, limit=limit)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    citations = generate_citations(search_results=search_results, query=query, limit=5)

    return {"query": query, "search_results": search_results, "citations": citations}


def citations_command(query, limit):
    return citations(query, limit)


def generate_question(search_results, question, limit=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the following question based on the provided documents.

    Question: {question}

    Documents:
    {context}

    General instructions:
    - Answer directly and concisely
    - Use only information from the documents
    - If the answer isn't in the documents, say "I don't have enough information"
    - Cite sources when possible

    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def question(query: str, limit):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query=query, k=60, limit=limit)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_question(
        search_results=search_results, question=query, limit=limit
    )

    return {"query": query, "search_results": search_results, "answer": answer}


def question_command(query, limit):
    return question(query, limit)
