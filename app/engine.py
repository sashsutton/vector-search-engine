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





