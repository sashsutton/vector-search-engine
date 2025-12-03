from pydantic import BaseModel

class DocumentRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    k: int
    