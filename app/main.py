from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.engine import VectorEngine
from app.models import DocumentRequest, SearchRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = VectorEngine()

@app.get("/")
def home():
    return {"status": "online", "docs_count": len(engine.documents)}

@app.post("/add")
def add_document(payload: DocumentRequest):
    engine.add_document(payload.text)
    return {"message": "Document indexed successfully", "total_docs": len(engine.documents)}

@app.post("/search")
def search_documents(payload: SearchRequest):
    results = engine.search(payload.query, payload.k)
    return {"query": payload.query, "results": results}

@app.get("/documents")
def list_documents():
    return {"documents": engine.get_documents(), "count": len(engine.documents)}

@app.delete("/delete/{doc_id}")
def delete_document(doc_id: int):
    success = engine.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document index out of range")
    return {"message": "Deleted successfully", "total_docs": len(engine.documents)}

@app.delete("/delete-all")
def delete_all():
    engine.clear_database()
    return {"message": "Database cleared", "total_docs": 0}