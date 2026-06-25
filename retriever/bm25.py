import math
import re
from collections import Counter

from interfaces import RetrieverResult


SOURCE_RETRIEVER = "bm25"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_documents = [tokenize(document) for document in documents]
        self.document_lengths = [len(tokens) for tokens in self.tokenized_documents]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )
        self.document_frequencies = self._build_document_frequencies()

    def _build_document_frequencies(self) -> dict[str, int]:
        document_frequencies: dict[str, int] = {}

        for tokens in self.tokenized_documents:
            for token in set(tokens):
                document_frequencies[token] = document_frequencies.get(token, 0) + 1

        return document_frequencies

    def _idf(self, token: str) -> float:
        document_frequency = self.document_frequencies.get(token, 0)
        document_count = len(self.documents)

        return math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def _score_document(self, query_tokens: list[str], document_index: int) -> float:
        document_tokens = self.tokenized_documents[document_index]
        document_length = self.document_lengths[document_index]

        if not document_tokens or self.average_document_length == 0:
            return 0.0

        term_frequencies = Counter(document_tokens)
        score = 0.0

        for token in query_tokens:
            term_frequency = term_frequencies.get(token, 0)

            if term_frequency == 0:
                continue

            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / self.average_document_length
            )
            score += self._idf(token) * (term_frequency * (self.k1 + 1)) / denominator

        return score

    def retrieve(self, query: str, top_k: int) -> list[RetrieverResult]:
        query_tokens = tokenize(query)
        results: list[RetrieverResult] = []

        for index, document in enumerate(self.documents):
            results.append(
                RetrieverResult(
                    document_id=f"doc-{index}",
                    text=document,
                    score=self._score_document(query_tokens, index),
                    rank=0,
                    source_retriever=SOURCE_RETRIEVER
                )
            )

        ranked_results = sorted(results, key=lambda item: item.score, reverse=True)

        return [
            RetrieverResult(
                document_id=result.document_id,
                text=result.text,
                score=result.score,
                rank=rank,
                source_retriever=result.source_retriever
            )
            for rank, result in enumerate(ranked_results[:top_k], start=1)
        ]
