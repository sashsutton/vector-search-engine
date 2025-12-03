import os
from sentence_transformers import SentenceTransformer

# Defining a permanent folder for the model
cache_path = os.path.join(os.getcwd(), "model_cache")
os.makedirs(cache_path, exist_ok=True)

print(f"Downloading model to {cache_path}...")
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
print("Model downloaded successfully!")