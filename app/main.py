from fastapi import FastAPI
from app.engine import VectorEngine
from app.models import DocumentRequest, SearchRequest

app = FastAPI()
engine = VectorEngine()
@app.get("/")
def home():
    return {"status": "online", "docs_count": len(engine.documents)}

@app.post("/add")
def add_document(payload: DocumentRequest):
    engine.documents.append(payload.query)
    return {"message": "Document indexed successfully", "total_docs": len(engine.documents)}

@app.post("/search")
def search_documents(payload: SearchRequest):
    results = engine.search(payload.query, payload.k)
    return {"query": payload.query, "results": results}
