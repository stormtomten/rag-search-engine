import math
import os
import pickle
import string
from collections import defaultdict
from re import L
from typing import Any, Counter, Dict, List, Set, Tuple

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap: Dict[int, Any] = {}
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.doc_lengths: Dict[int, int] = defaultdict(int)
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies: Dict[int, Counter] = defaultdict(Counter)
        self.term_path = os.path.join(CACHE_DIR, "term_frequencies.pkt")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        return total_length / num_docs

    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        doc_ids = sorted(self.index.get(term, set()))

        return doc_ids

    def get_tf(self, doc_id: int, term: str) -> float:
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("more than one token in term")

        return self.term_frequencies[doc_id].get(tokens[0], 0)

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("more than one token in term")

        return math.log(
            (len(self.docmap) + 1) / (len(self.get_documents(tokens[0])) + 1)
        )

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("more than one token in term")

        n = len(self.docmap)
        df = len(self.get_documents(tokens[0]))

        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[dict]:
        tokens = tokenize_text(query)
        scores: Dict[int, float] = {}

        for doc in self.docmap:
            scores[doc] = 0.0
            for token in tokens:
                scores[doc] += self.bm25(doc, token)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results = []
        for score in sorted_scores:
            if len(results) >= limit:
                break
            doc = self.docmap[score[0]]
            formatted_result = {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"],
                "score": score[1],
            }
            results.append(formatted_result)

        return results

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

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

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

        try:
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
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


def tf_command(doc_id: int, term: str) -> int:
    movies = InvertedIndex()
    movies.load()

    return movies.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    movies = InvertedIndex()
    movies.load()

    return movies.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    movies = InvertedIndex()
    movies.load()

    return movies.get_tfidf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    movies = InvertedIndex()
    movies.load()

    return movies.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    movies = InvertedIndex()
    movies.load()

    return movies.get_bm25_tf(doc_id, term, k1, b)


def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[dict]:
    movies = InvertedIndex()
    movies.load()
    return movies.bm25_search(query, limit)


def build() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
