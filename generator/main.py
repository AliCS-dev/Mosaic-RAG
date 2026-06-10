from fastapi import FastAPI
from pydantic import BaseModel

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
    context = " ".join(
        [document["text"] for document in request.retrieved_documents]
    )

    answer = (
        f"Question: {request.query}\n\n"
        f"Answer based on retrieved context: {context}"
    )

    return {
        "query": request.query,
        "answer": answer,
        "used_documents": request.retrieved_documents
    }
