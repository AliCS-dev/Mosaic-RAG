from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("mosaic-rag-api")

app = FastAPI(title="Mosaic-RAG API Controller")

RETRIEVER_URL = "http://retriever:8001/retrieve"
GENERATOR_URL = "http://generator:8002/generate"


class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    retrieval_mode: str = "dense"


def validate_retrieval_mode(retrieval_mode: str) -> None:
    normalized_mode = retrieval_mode.lower().strip()

    if normalized_mode not in {"dense", "bm25", "hybrid", "mosaic"}:
        raise HTTPException(
            status_code=400,
            detail="retrieval_mode must be one of: dense, bm25, hybrid, mosaic"
        )


@app.get("/")
def health_check():
    return {
        "service": "Mosaic-RAG API Controller",
        "status": "running"
    }


@app.post("/ask")
def ask(request: AskRequest):
    request_id = str(uuid.uuid4())
    validate_retrieval_mode(request.retrieval_mode)

    logger.info(f"[{request_id}] Received query: {request.query}")
    logger.info(f"[{request_id}] Requested top_k: {request.top_k}")
    logger.info(f"[{request_id}] Requested retrieval_mode: {request.retrieval_mode}")

    try:
        logger.info(f"[{request_id}] Calling Retriever Service")

        retrieval_response = requests.post(
            RETRIEVER_URL,
            json={
                "request_id": request_id,
                "query": request.query,
                "top_k": request.top_k,
                "retrieval_mode": request.retrieval_mode
            },
            timeout=10
        )

        retrieval_response.raise_for_status()
        retrieved_data = retrieval_response.json()

        retrieved_documents = retrieved_data["retrieved_documents"]

        logger.info(f"[{request_id}] Retriever returned {len(retrieved_documents)} documents")
        logger.info(f"[{request_id}] Calling Generator Service")

        generation_response = requests.post(
            GENERATOR_URL,
            json={
                "request_id": request_id,
                "query": request.query,
                "retrieved_documents": retrieved_documents
            },
            timeout=10
        )

        generation_response.raise_for_status()
        generated_data = generation_response.json()

        logger.info(f"[{request_id}] Generator returned answer successfully")

        return {
            "request_id": request_id,
            "query": request.query,
            "retrieval_mode": retrieved_data["retrieval_mode"],
            "retrieved_documents": retrieved_documents,
            "answer": generated_data["answer"]
        }

    except requests.exceptions.RequestException as error:
        logger.error(f"[{request_id}] Service communication failed: {str(error)}")

        raise HTTPException(
            status_code=502,
            detail=f"Service communication failed: {str(error)}"
        )
