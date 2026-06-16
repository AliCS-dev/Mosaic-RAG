from fastapi import FastAPI
from pydantic import BaseModel
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("mosaic-rag-generator")

app = FastAPI(title="Mosaic-RAG Generator Service")


class GenerateRequest(BaseModel):
    request_id: Optional[str] = None
    query: str
    retrieved_documents: list[dict]


@app.get("/")
def health_check():
    return {
        "service": "Mosaic-RAG Generator",
        "status": "running"
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    request_id = request.request_id or "no-request-id"

    logger.info(f"[{request_id}] Received generation request for query: {request.query}")
    logger.info(f"[{request_id}] Received {len(request.retrieved_documents)} retrieved documents")

    context = " ".join(
        [document["text"] for document in request.retrieved_documents]
    )

    answer = (
        f"Question: {request.query}\n\n"
        f"Answer based on retrieved context: {context}"
    )

    logger.info(f"[{request_id}] Generated answer successfully")

    return {
        "request_id": request_id,
        "query": request.query,
        "answer": answer,
        "used_documents": request.retrieved_documents
    }