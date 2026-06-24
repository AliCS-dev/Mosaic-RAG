from pydantic import BaseModel


class RetrieverResult(BaseModel):
    document_id: str
    text: str
    score: float
    rank: int
    source_retriever: str
