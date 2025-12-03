from sentence_transformers import SentenceTransformer
import numpy as np


class VectorEngine:
    def __init__(self):
        #Loading a small model for semantic search
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        #Dimension of this specific model is 384
        self.vector_dim = 384

        #Matrix to hold N vectors: shape(N, 384)
        self.vectors = np.empty((0,self.vector_dim), dtype=np.float32)

        #List to hold the text content
        self.documents = []

    def embed_text(self, text: str) -> np.ndarray:
        """
        Converts text into a vector using the AI model
        """

        vector = self.model.encode(text)
        return vector

    def add_document(self, text: str):

        #Turn the text into vector
        vector = self.embed_text(text)

        #Normalise vector so that dot product = cosine similarity
        #Math: vector_norm = v / ||v||

        norm = np.linalg.norm(vector)
        if norm == 0:
            return
        vector = vector / norm

        #Storing the vector
        self.vectors = np.vstack([self.vectors, vector])
        self.documents.append(text)









