from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mosaic-RAG Retriever Service")


class QueryRequest(BaseModel):
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
    results = []

    for document in DOCUMENTS:
        score = score_document(request.query, document)

        results.append({
            "text": document,
            "score": score
        })

    results = sorted(results, key=lambda item: item["score"], reverse=True)

    return {
        "query": request.query,
        "top_k": request.top_k,
        "retrieved_documents": results[:request.top_k]
    }