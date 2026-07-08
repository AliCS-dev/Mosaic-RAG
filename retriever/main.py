from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional
from bm25 import BM25Retriever
from dense import DenseRetriever
from fusion import reciprocal_rank_fusion
from interfaces import RetrieverResult, serialize_retriever_results

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
    retrieval_mode: str = "dense"


with open("/data/corpus.txt", "r", encoding="utf-8") as file:
    DOCUMENTS = [line.strip() for line in file.readlines() if line.strip()]

BM25 = BM25Retriever(DOCUMENTS)
DENSE = DenseRetriever(DOCUMENTS)


def retrieve_hybrid_results(query: str, top_k: int) -> list[RetrieverResult]:
    dense_results = DENSE.retrieve(query, top_k)
    bm25_results = BM25.retrieve(query, top_k)

    return reciprocal_rank_fusion([dense_results, bm25_results], top_k)


def normalize_retrieval_mode(retrieval_mode: str) -> str:
    normalized_mode = retrieval_mode.lower().strip()

    if normalized_mode == "mosaic":
        return "hybrid"

    if normalized_mode in {"dense", "bm25", "hybrid"}:
        return normalized_mode

    raise HTTPException(
        status_code=400,
        detail="retrieval_mode must be one of: dense, bm25, hybrid, mosaic"
    )


def retrieve_results(query: str, top_k: int, retrieval_mode: str) -> list[RetrieverResult]:
    normalized_mode = normalize_retrieval_mode(retrieval_mode)

    if normalized_mode == "dense":
        return DENSE.retrieve(query, top_k)

    if normalized_mode == "bm25":
        return BM25.retrieve(query, top_k)

    return retrieve_hybrid_results(query, top_k)


@app.get("/")
def health_check():
    return {
        "service": "Mosaic-RAG Retriever",
        "status": "running",
        "documents_loaded": len(DOCUMENTS),
        "dense_backend": DENSE.backend
    }


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    request_id = request.request_id or "no-request-id"
    retrieval_mode = normalize_retrieval_mode(request.retrieval_mode)

    logger.info(f"[{request_id}] Received retrieval request for query: {request.query}")
    logger.info(f"[{request_id}] Requested top_k: {request.top_k}")
    logger.info(f"[{request_id}] Requested retrieval_mode: {retrieval_mode}")

    top_results = retrieve_results(request.query, request.top_k, retrieval_mode)

    logger.info(f"[{request_id}] Returning {len(top_results)} retrieved documents")

    return {
        "request_id": request_id,
        "query": request.query,
        "top_k": request.top_k,
        "retrieval_mode": retrieval_mode,
        "retrieved_documents": serialize_retriever_results(top_results)
    }


@app.post("/retrieve/bm25")
def retrieve_bm25(request: QueryRequest):
    request_id = request.request_id or "no-request-id"

    logger.info(f"[{request_id}] Received BM25 retrieval request for query: {request.query}")
    logger.info(f"[{request_id}] Requested top_k: {request.top_k}")

    top_results = BM25.retrieve(request.query, request.top_k)

    logger.info(f"[{request_id}] Returning {len(top_results)} BM25 retrieved documents")

    return {
        "request_id": request_id,
        "query": request.query,
        "top_k": request.top_k,
        "retrieval_mode": "bm25",
        "retrieved_documents": serialize_retriever_results(top_results)
    }
