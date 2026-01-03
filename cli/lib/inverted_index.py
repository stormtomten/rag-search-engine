import pickle
from typing import Any, Dict, List, Set

from .keyword_search import tokenize_text
from .search_utils import DOCMAP_CACHE, INDEX_CACHE, load_movies


class invertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Any] = {}

    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize_text(text)

        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        doc_ids = sorted(self.index.get(term, set()))

        return doc_ids

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self):
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_CACHE, "wb") as f:
            pickle.dump(self.docmap, f)
