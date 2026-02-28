from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-V2")

    def generate_embeding(self, text):
        if len(text) == 0 or text.isspace():
            raise ValueError("Text is empty")
        embedding = self.model.encode(list(text))
        return embedding[0]


def verify_model():
    model = SemanticSearch()

    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")


def embed_text(text):
    model = SemanticSearch()

    embedding = model.generate_embeding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
