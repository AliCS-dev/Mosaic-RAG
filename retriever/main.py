from fastapi import FastAPI
from pydantic import BaseModel
import logging
from typing import Optional
from interfaces import RetrieverResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("mosaic-rag-retriever")

app = FastAPI(title="Mosaic-RAG Retriever Service")

SOURCE_RETRIEVER = "dense"


class QueryRequest(BaseModel):
    request_id: Optional[str] = None
    query: str
    top_k: int = 3


with open("/data/corpus.txt", "r", encoding="utf-8") as file:
    DOCUMENTS = [line.strip() for line in file.readlines() if line.strip()]


def score_document(query: str, document: str) -> int:
    query_words = set(query.lower().replace("?", "").split())
    document_words = set(document.lower().replace(".", "").split())

    return len(query_words.intersection(document_words))


@app.get("/")
def health_check():
    return {
        "service": "Mosaic-RAG Retriever",
        "status": "running",
        "documents_loaded": len(DOCUMENTS)
    }


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    request_id = request.request_id or "no-request-id"

    logger.info(f"[{request_id}] Received retrieval request for query: {request.query}")
    logger.info(f"[{request_id}] Requested top_k: {request.top_k}")

    results: list[RetrieverResult] = []

    for index, document in enumerate(DOCUMENTS):
        score = score_document(request.query, document)

        results.append(
            RetrieverResult(
                document_id=f"doc-{index}",
                text=document,
                score=float(score),
                rank=0,
                source_retriever=SOURCE_RETRIEVER
            )
        )

    results = sorted(results, key=lambda item: item.score, reverse=True)
    top_results = [
        RetrieverResult(
            document_id=result.document_id,
            text=result.text,
            score=result.score,
            rank=rank,
            source_retriever=result.source_retriever
        )
        for rank, result in enumerate(results[:request.top_k], start=1)
    ]

    logger.info(f"[{request_id}] Returning {len(top_results)} retrieved documents")

    return {
        "request_id": request_id,
        "query": request.query,
        "top_k": request.top_k,
        "retrieved_documents": [result.dict() for result in top_results]
    }
