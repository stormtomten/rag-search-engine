import json
import os
import re
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging

from .search_utils import DOCUMENT_PREVIEW_LENGTH, SCORE_PRECISION

transformers_logging.set_verbosity_error()

from .search_utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(
            model_name, cache_folder=CACHE_DIR, token=False
        )
        self.embeddings: np.ndarray | None = None
        self.documents: List[Any] | None = None
        self.document_map: Dict[int, Any] = {}
        self.cache_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embeding(self, text: str) -> np.ndarray:
        if len(text) == 0 or text.isspace():
            raise ValueError("Text is empty")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: List[Any]) -> np.ndarray:
        self.documents = documents

        docstrings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            docstrings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docstrings, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.cache_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: List[Any]) -> Any:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.cache_path):
            self.embeddings = np.load(self.cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit) -> List[Any]:
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        embedded_query = self.generate_embeding(query)

        similarities = []
        for idx, embedding in enumerate(self.embeddings):
            score = cosine_similarity(embedded_query, embedding)
            similarities.append((score, self.documents[idx]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        result = []
        for idx in range(limit):
            score, doc = similarities[idx]
            result.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return result


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: List[Dict] | None = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: List[Any]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        chunks: list[str] = []
        chunks_metadata: List[Dict]= []



        for movie_idx, document in enumerate(self.documents):
            self.document_map[document["id"]] = document

            description = document.get("description", "")
            if not description:
                continue

            doc_chunks = semantic_chunk(description, chunk_size=4,overlap=1)
            chunks.extend(doc_chunks)
            for doc_chunk_idx, _ in enumerate(doc_chunks):
                chunks_metadata.append({"movie_idx": movie_idx, "chunk_idx": doc_chunk_idx, "total_chunks": len(doc_chunks)})

        self.chunk_metadata = chunks_metadata
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)



        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunks_metadata, "total_chunks": len(chunks)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> List[Any]:
        if len(self.chunk_embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embeding(query)
        chunk_score: List[Dict]= []

        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk_embedding, query_embedding)

            metadata = self.chunk_metadata[idx]
            chunk_score.append({"chunk_idx": metadata["chunk_idx"], "movie_idx": metadata["movie_idx"], "score": score})

        movie_scores = {}
        for chunk in chunk_score:
            if chunk["movie_idx"] not in movie_scores or chunk["score"] > movie_scores[chunk["movie_idx"]]:
                movie_scores[chunk["movie_idx"]] = chunk["score"]

        movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in movie_scores[:limit]:
            results.append(
                    {
                        "id": self.documents[movie_idx]["id"],
                        "title": self.documents[movie_idx]["title"],
                        "document": self.documents[movie_idx]["description"][:DOCUMENT_PREVIEW_LENGTH],
                        "score": round(score, SCORE_PRECISION),
                        "metadata": {},
                        
                        })
        return results


        


def verify_model() -> None:
    model = SemanticSearch()

    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")


def embed_text(text: str) -> None:
    model = SemanticSearch()

    embedding = model.generate_embeding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    model = SemanticSearch()

    movies = load_movies()

    embeddings = model.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(model.documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    model = SemanticSearch()

    embedding = model.generate_embeding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)



def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chuck-size")
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0

    while start < len(words):
        remaining = len(words) - start
        if remaining <= overlap:
            break
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def semantic_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chuck-size")
    text = text.strip()
    if text == "":
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return []

    clean_sentences = []

    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned != "":
            clean_sentences.append(cleaned)

    if not clean_sentences:
        return []
    if len(clean_sentences) == 1 and not clean_sentences[0].endswith(('.', '!', '?')):
        return clean_sentences

    chunks = []
    start = 0

    while start < len(clean_sentences):
        remaining = len(clean_sentences) - start
        if remaining <= overlap:
            break
        chunk = " ".join(clean_sentences[start : start + chunk_size])
        chunk = chunk.strip()
        if chunk != "":
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def search_chunked_command(query: str, limit: int) -> List[Any]:
    movies = load_movies()
    model = ChunkedSemanticSearch()
    model.load_or_create_chunk_embeddings(movies)

    return model.search_chunks(query=query, limit=limit)


