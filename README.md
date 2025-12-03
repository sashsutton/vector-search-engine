# ⚡ Vector Search Engine (RAG Backend)

A semantic search engine built from scratch using **Python**, **FastAPI**, and **Linear Algebra**. Unlike traditional keyword search (Ctrl+F), this engine understands the *meaning* behind queries by converting text into high-dimensional vectors.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Math](https://img.shields.io/badge/Math-Linear%20Algebra-orange)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-success)](https://vector-search-engine-frontend.vercel.app/)
[![Frontend Code](https://img.shields.io/badge/Frontend_Repo-React-blue)](https://github.com/sashsutton/vector-search-engine-frontend/tree/main)

## 🔗 Project Structure
* **frontend**: [Go to Frontend Folder](https://github.com/sashsutton/vector-search-engine-frontend/tree/main) - React/Vite UI
* **backend**: (This folder) - FastAPI & NumPy Logic
* **🚀 Live Website**:[[Live website](https://vector-search-engine-frontend.vercel.app/)]

## 🚀 How It Works

This project implements **Retrieval-Augmented Generation (RAG)** principles without using external vector databases (like Pinecone). It implements the vector math manually using NumPy.

1.  **Ingestion:** User inputs text (e.g., "The sky is blue").
2.  **Embedding:** Text is passed through a Transformer model (`all-MiniLM-L6-v2`) to generate a 384-dimensional vector.
3.  **Indexing:** The vector is normalized and stored in an in-memory NumPy matrix.
4.  **Retrieval:** We compute **Cosine Similarity** via Matrix Multiplication to find the best match.

## 🧠 The Math Behind It

### 1. Vector Embeddings
We map every sentence $S$ to a vector $\vec{v}$ in a 384-dimensional space:
$$f(S) \rightarrow \mathbb{R}^{384}$$

### 2. Normalization (L2 Norm)
To optimize search speed, we normalize all vectors to unit length. This simplifies Cosine Similarity to a pure Dot Product.
$$\hat{v} = \frac{\vec{v}}{||\vec{v}||}$$

### 3. Matrix Multiplication
We calculate the similarity score between the Query Vector ($\vec{A}$) and Database Matrix ($\mathbf{B}$) in one operation:
$$\text{Scores} = \mathbf{B} \cdot \vec{A}$$

## 🛠️ Tech Stack

-   **Backend:** Python, FastAPI, NumPy, Sentence-Transformers
-   **Frontend:** React, Vite, CSS Modules
-   **Architecture:** REST API with CORS enabled for microservices.

## 💻 Local Setup

1.  **Backend:**
    ```bash
    cd vector-search-engine
    pip install -r requirements.txt
    uvicorn app.main:app --reload
    ```

2.  **Frontend:**
    ```bash
    cd vector-search-engine-frontend
    npm install
    npm run dev
    ```