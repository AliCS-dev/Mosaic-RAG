from pydantic import BaseModel


class RetrieverResult(BaseModel):
    document_id: str
    text: str
    score: float
    rank: int
    source_retriever: str


def serialize_retriever_results(results: list[RetrieverResult]) -> list[dict]:
    return [
        result.model_dump() if hasattr(result, "model_dump") else result.dict()
        for result in results
    ]
