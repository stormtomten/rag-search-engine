import os
import pickle
from typing import Any, Dict, List, Set

from .keyword_search import tokenize_text
from .search_utils import CACHE_DIR, load_movies


class InvertedIndex:
    def __init__(self) -> None:
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Any] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = tokenize_text(text)

        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        doc_ids = sorted(self.index.get(term, set()))

        return doc_ids

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
