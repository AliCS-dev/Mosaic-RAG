from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

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
    try:
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

        generation_response = requests.post(
            GENERATOR_URL,
            json={
                "query": request.query,
                "retrieved_documents": retrieved_data["retrieved_documents"]
            },
            timeout=10
        )
        generation_response.raise_for_status()
        generated_data = generation_response.json()

        return {
            "query": request.query,
            "retrieved_documents": retrieved_data["retrieved_documents"],
            "answer": generated_data["answer"]
        }

    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Service communication failed: {str(error)}"
        )
