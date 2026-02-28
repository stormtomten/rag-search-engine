import os
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

from .search_utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(
            "all-MiniLM-L6-V2", cache_folder=CACHE_DIR, token=False
        )
        self.embeddings: np.ndarray | None = None
        self.documents: List[Any] | None = None
        self.document_map: Dict[int, Any] = {}
        self.cache_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embeding(self, text: str) -> np.ndarray:
        if len(text) == 0 or text.isspace():
            raise ValueError("Text is empty")
        embedding = self.model.encode(list(text))
        return embedding[0]

    def build_embeddings(self, documents: List[Any]) -> np.ndarray:
        self.documents = documents

        docstrings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            docstrings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docstrings, show_progress_bar=True)
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
