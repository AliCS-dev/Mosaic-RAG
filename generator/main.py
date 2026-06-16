from fastapi import FastAPI
from pydantic import BaseModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("mosaic-rag-generator")

app = FastAPI(title="Mosaic-RAG Generator Service")


class GenerateRequest(BaseModel):
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
    logger.info(f"Received generation request for query: {request.query}")
    logger.info(f"Received {len(request.retrieved_documents)} retrieved documents")

    context = " ".join(
        [document["text"] for document in request.retrieved_documents]
    )

    answer = (
        f"Question: {request.query}\n\n"
        f"Answer based on retrieved context: {context}"
    )

    logger.info("Generated answer successfully")

    return {
        "query": request.query,
        "answer": answer,
        "used_documents": request.retrieved_documents
    }