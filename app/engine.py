from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class VectorEngine:
    def __init__(self):
        # Loading a small model for semantic search
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Dimension of this specific model is 384
        self.vector_dim = 384

        # Matrix to hold N vectors: shape(N, 384)
        self.vectors = np.empty((0,self.vector_dim), dtype=np.float32)

        # List to hold the text content
        self.documents = []

    def embed_text(self, text: str) -> np.ndarray:
        """
        Converts text into a vector using the AI model
        """

        vector = self.model.encode(text)
        return vector

    def add_document(self, text: str):

        # Turn the text into vector
        vector = self.embed_text(text)

        # Normalise vector so that dot product = cosine similarity
        # Math: v_norm = v / ||v||

        v_norm = np.linalg.norm(vector)
        if v_norm == 0:
            return
        vector = vector / v_norm

        # Storing the vector
        self.vectors = np.vstack([self.vectors, vector])
        self.documents.append(text)

    def search(self, query: str, k: int = 3) -> List[dict]:

        # Vectorise the query
        query_vector = self.embed_text(query)

        # Normalise the query vector
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        query_vector = query_vector / q_norm

        # Matrix multiplication
        # Matrix A (Database): Shape (N_docs, 384_features)
        # Vector B (Query):    Shape (384_features, 1)
        # Operation:           A . B
        #
        # The inner dimensions (384) match and cancel out.
        # The result is a list of N scores: Shape (N_docs,)
        #
        # Since vectors are normalized, Dot Product == Cosine Similarity.
        scores = np.dot(self.vectors, query_vector)

        # Sort and get top K results
        # argsort returns indices of sorted values (low to high), so we reverse it
        top_k_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.documents[idx],
                "score": float(scores[idx])
            })

        return results














