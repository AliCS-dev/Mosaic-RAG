import logging
import os

from interfaces import RetrieverResult


SOURCE_RETRIEVER = "dense"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger("mosaic-rag-dense-retriever")


class DenseRetriever:
    def __init__(self, documents: list[str], model_name: str | None = None):
        self.documents = documents
        self.model_name = model_name or os.getenv("DENSE_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.backend = "keyword"
        self.model = None
        self.index = None
        self.numpy = None

        self._initialize_faiss_backend()

    def _initialize_faiss_backend(self) -> None:
        try:
            import faiss
            import numpy
            from sentence_transformers import SentenceTransformer

            self.numpy = numpy
            self.model = SentenceTransformer(self.model_name)
            document_embeddings = self.model.encode(
                self.documents,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            document_embeddings = document_embeddings.astype("float32")

            self.index = faiss.IndexFlatIP(document_embeddings.shape[1])
            self.index.add(document_embeddings)
            self.backend = "faiss"
            logger.info(f"Initialized FAISS dense retriever with model: {self.model_name}")

        except Exception as error:
            logger.warning(
                "FAISS dense retriever unavailable; falling back to keyword overlap. "
                f"Reason: {error}"
            )

    def _keyword_score(self, query: str, document: str) -> float:
        query_words = set(query.lower().replace("?", "").split())
        document_words = set(document.lower().replace(".", "").split())

        return float(len(query_words.intersection(document_words)))

    def _retrieve_with_keyword_fallback(self, query: str, top_k: int) -> list[RetrieverResult]:
        results = []

        for index, document in enumerate(self.documents):
            results.append(
                RetrieverResult(
                    document_id=f"doc-{index}",
                    text=document,
                    score=self._keyword_score(query, document),
                    rank=0,
                    source_retriever=SOURCE_RETRIEVER
                )
            )

        ranked_results = sorted(results, key=lambda item: item.score, reverse=True)

        return self._rerank_results(ranked_results[:top_k])

    def _retrieve_with_faiss(self, query: str, top_k: int) -> list[RetrieverResult]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                RetrieverResult(
                    document_id=f"doc-{index}",
                    text=self.documents[index],
                    score=float(score),
                    rank=0,
                    source_retriever=SOURCE_RETRIEVER
                )
            )

        return self._rerank_results(results)

    def _rerank_results(self, results: list[RetrieverResult]) -> list[RetrieverResult]:
        return [
            RetrieverResult(
                document_id=result.document_id,
                text=result.text,
                score=result.score,
                rank=rank,
                source_retriever=result.source_retriever
            )
            for rank, result in enumerate(results, start=1)
        ]

    def retrieve(self, query: str, top_k: int) -> list[RetrieverResult]:
        if self.backend == "faiss":
            return self._retrieve_with_faiss(query, top_k)

        return self._retrieve_with_keyword_fallback(query, top_k)
