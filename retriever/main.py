from fastapi import FastAPI
from pydantic import BaseModel
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("mosaic-rag-retriever")

app = FastAPI(title="Mosaic-RAG Retriever Service")


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

    results = []

    for document in DOCUMENTS:
        score = score_document(request.query, document)

        results.append({
            "text": document,
            "score": score
        })

    results = sorted(results, key=lambda item: item["score"], reverse=True)
    top_results = results[:request.top_k]

    logger.info(f"[{request_id}] Returning {len(top_results)} retrieved documents")

    return {
        "request_id": request_id,
        "query": request.query,
        "top_k": request.top_k,
        "retrieved_documents": top_results
    }