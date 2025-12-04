# Vector Search Engine (RAG Backend)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-success)](https://vector-search-engine-frontend.vercel.app/)
[![Frontend Code](https://img.shields.io/badge/Frontend_Repo-React-blue)](https://github.com/sashsutton/vector-search-engine-frontend/tree/main)

A high-performance semantic search engine built from scratch using **Python**, **FastAPI**, and **Linear Algebra**.

It includes a full management API to **Index**, **Search**, **List**, and **Delete** vector embeddings in real-time.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Math](https://img.shields.io/badge/Math-Linear%20Algebra-orange)

## 🚀 Key Features
* **Semantic Search:** Uses Cosine Similarity to find meaning, not just keywords.
* **CRUD Operations:** Full support to Add, List, and Delete documents.
* **Vector Management:** Manages high-dimensional vectors (384-dim) in memory using NumPy.
* **Microservice Ready:** Fully CORS-enabled for frontend integration.

## 🧠 The Math Behind It
1. **Vector Embeddings:** $f(S) \rightarrow \mathbb{R}^{384}$ using BERT Transformers.
2. **Normalization:** Vectors are L2-normalized so $||\vec{v}|| = 1$.
3. **Retrieval:** Similarity is calculated via Matrix Multiplication: $\text{Scores} = \mathbf{B} \cdot \vec{A}$.

## 🌐 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check & document count. |
| `POST` | `/add` | Indexes a new text string into the vector matrix. |
| `POST` | `/search` | Converts query to vector and returns top matches. |
| `GET` | `/documents` | Lists all indexed documents with their IDs. |
| `DELETE` | `/delete/{id}` | Removes a specific document and its vector. |
| `DELETE` | `/delete-all` | Clears the entire database (Text + Vectors). |

## 🛠️ Tech Stack
* **FastAPI:** Async API Framework.
* **NumPy:** Linear Algebra Engine.
* **Sentence-Transformers:** AI Model hosting (`all-MiniLM-L6-v2`).