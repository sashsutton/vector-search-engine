from fastapi import FastAPI
from app.engine import VectorEngine
from app.models import DocumentRequest, SearchRequest
from fastapi.middleware.cors import CORSMiddleware


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
