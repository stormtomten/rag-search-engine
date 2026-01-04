import math
import os
import pickle
import string
from collections import defaultdict
from typing import Any, Counter, Dict, List, Set

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


class InvertedIndex:
    def __init__(self) -> None:
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.docmap: Dict[int, Any] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies: Dict[int, Counter] = defaultdict(Counter)
        self.term_path = os.path.join(CACHE_DIR, "term_frequencies.pkt")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        doc_ids = sorted(self.index.get(term, set()))

        return doc_ids

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise Exception("more than one token in term")

        return self.term_frequencies[doc_id].get(tokens[0], 0)

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
        except FileNotFoundError:
            print("index not found!")

        try:
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            print("docmap not found!")

        try:
            with open(self.term_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            print("term frequencies not found!")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = InvertedIndex()
    movies.load()
    results = []
    query_tokens = tokenize_text(query)
    seen_ids = set()
    for token in query_tokens:
        if len(results) >= limit:
            break
        docs = movies.get_documents(token)
        if not docs:
            continue
        for doc in docs:
            if doc not in seen_ids:
                seen_ids.add(doc)
                results.append(movies.docmap[doc])
                if len(results) >= limit:
                    break
    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    if any(
        any(query_token in title_token for title_token in title_tokens)
        for query_token in query_tokens
    ):
        return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    # trans_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    stopwords = load_stopwords()

    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(token)
    tokens_stemmed = stem_tokens(valid_tokens)

    return tokens_stemmed


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed = []

    for token in tokens:
        stemmed.append(stemmer.stem(token))

    return stemmed


def tf_search(doc_id: int, term: str) -> int:
    movies = InvertedIndex()
    movies.load()

    return movies.get_tf(doc_id, term)


def idf_lookup(term: str) -> float:
    movies = InvertedIndex()
    movies.load()

    tokens = tokenize_text(term)

    return math.log(
        (len(movies.docmap) + 1) / (len(movies.get_documents(tokens[0])) + 1)
    )


def tfidf_lookup(doc_id: int, term: str) -> float:
    movies = InvertedIndex()
    movies.load()

    tf = tf_search(doc_id, term)
    idf = idf_lookup(term)

    return tf * idf


def build() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
