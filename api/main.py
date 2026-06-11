from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging

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


@app.get("/")
def health_check():
    return {
        "service": "Mosaic-RAG API Controller",
        "status": "running"
    }


@app.post("/ask")
def ask(request: AskRequest):
    logger.info(f"Received query: {request.query}")
    logger.info(f"Requested top_k: {request.top_k}")

    try:
        logger.info("Calling Retriever Service")

        retrieval_response = requests.post(
            RETRIEVER_URL,
            json={
                "query": request.query,
                "top_k": request.top_k
            },
            timeout=10
        )

        retrieval_response.raise_for_status()
        retrieved_data = retrieval_response.json()

        retrieved_documents = retrieved_data["retrieved_documents"]

        logger.info(f"Retriever returned {len(retrieved_documents)} documents")

        logger.info("Calling Generator Service")

        generation_response = requests.post(
            GENERATOR_URL,
            json={
                "query": request.query,
                "retrieved_documents": retrieved_documents
            },
            timeout=10
        )

        generation_response.raise_for_status()
        generated_data = generation_response.json()

        logger.info("Generator returned answer successfully")

        return {
            "query": request.query,
            "retrieved_documents": retrieved_documents,
            "answer": generated_data["answer"]
        }

    except requests.exceptions.RequestException as error:
        logger.error(f"Service communication failed: {str(error)}")

        raise HTTPException(
            status_code=502,
            detail=f"Service communication failed: {str(error)}"
        )