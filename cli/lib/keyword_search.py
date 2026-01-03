import string

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        titel_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, titel_tokens):
            results.append(movie)
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
