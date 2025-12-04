from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import os


class VectorEngine:
    def __init__(self):
        # Looking for the model in the cache folder created
        cache_path = os.path.join(os.getcwd(), "model_cache")

        # Check if cache exists to avoid errors on local without build script
        if os.path.exists(cache_path):
            print(f"Loading model from {cache_path}...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
        else:
            print("Downloading model (No cache found)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.vector_dim = 384
        self.vectors = np.empty((0, self.vector_dim), dtype=np.float32)
        self.documents = []

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def add_document(self, text: str):
        vector = self.embed_text(text)
        v_norm = np.linalg.norm(vector)
        if v_norm > 0:
            vector = vector / v_norm
        self.vectors = np.vstack([self.vectors, vector])
        self.documents.append(text)

    def delete_document(self, index: int):
        if 0 <= index < len(self.documents):
            self.documents.pop(index)
            self.vectors = np.delete(self.vectors, index, axis=0)
            return True
        return False

    def clear_database(self):
        self.vectors = np.empty((0, self.vector_dim), dtype=np.float32)
        self.documents = []

    def get_documents(self) -> List[Dict]:
        return [
            {"id": i, "text": doc}
            for i, doc in enumerate(self.documents)
        ]

    def search(self, query: str, k: int = 3) -> List[dict]:
        query_vector = self.embed_text(query)
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0: return []
        query_vector = query_vector / q_norm

        if len(self.documents) == 0: return []

        scores = np.dot(self.vectors, query_vector)
        k = min(k, len(scores))

        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [
            {
                "id": int(idx),
                "text": self.documents[idx],
                "score": float(scores[idx])
            }
            for idx in top_k_indices
        ]