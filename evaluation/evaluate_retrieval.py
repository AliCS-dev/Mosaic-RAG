import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


MODES = ("dense", "bm25", "hybrid")
DEFAULT_K_VALUES = (5, 10)
DEFAULT_RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_corpus(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_qa_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_result(document_id: str, text: str, score: float, rank: int, source: str) -> dict:
    return {
        "document_id": document_id,
        "text": text,
        "score": score,
        "rank": rank,
        "source_retriever": source
    }


def dense_retrieve(query: str, documents: list[str], top_k: int) -> list[dict]:
    query_words = set(query.lower().replace("?", "").split())
    results = []

    for index, document in enumerate(documents):
        document_words = set(document.lower().replace(".", "").split())
        score = float(len(query_words.intersection(document_words)))

        results.append(build_result(f"doc-{index}", document, score, 0, "dense"))

    ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)

    return [
        build_result(
            result["document_id"],
            result["text"],
            result["score"],
            rank,
            result["source_retriever"]
        )
        for rank, result in enumerate(ranked_results[:top_k], start=1)
    ]


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
        document_frequencies = {}

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

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        query_tokens = tokenize(query)
        results = []

        for index, document in enumerate(self.documents):
            results.append(
                build_result(
                    f"doc-{index}",
                    document,
                    self._score_document(query_tokens, index),
                    0,
                    "bm25"
                )
            )

        ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)

        return [
            build_result(
                result["document_id"],
                result["text"],
                result["score"],
                rank,
                result["source_retriever"]
            )
            for rank, result in enumerate(ranked_results[:top_k], start=1)
        ]


def reciprocal_rank_fusion(
    result_sets: list[list[dict]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K
) -> list[dict]:
    scores = {}
    documents_by_id = {}

    for results in result_sets:
        for position, result in enumerate(results, start=1):
            rank = result["rank"] if result["rank"] > 0 else position
            document_id = result["document_id"]

            scores[document_id] = scores.get(document_id, 0.0) + 1 / (rrf_k + rank)
            documents_by_id.setdefault(document_id, result)

    ranked_document_ids = sorted(
        scores,
        key=lambda document_id: (-scores[document_id], document_id)
    )

    return [
        build_result(
            document_id,
            documents_by_id[document_id]["text"],
            scores[document_id],
            rank,
            "rrf"
        )
        for rank, document_id in enumerate(ranked_document_ids[:top_k], start=1)
    ]


def retrieve_for_mode(
    query: str,
    documents: list[str],
    bm25: BM25Retriever,
    mode: str,
    top_k: int
) -> list[dict]:
    if mode == "dense":
        return dense_retrieve(query, documents, top_k)

    if mode == "bm25":
        return bm25.retrieve(query, top_k)

    dense_results = dense_retrieve(query, documents, top_k)
    bm25_results = bm25.retrieve(query, top_k)

    return reciprocal_rank_fusion([dense_results, bm25_results], top_k)


def recall_at_k(results: list[dict], relevant_document_ids: list[str], k: int) -> float:
    relevant_ids = set(relevant_document_ids)
    retrieved_ids = {
        result["document_id"]
        for result in results[:k]
    }

    return len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids)


def evaluate_mode(
    qa_dataset: list[dict],
    documents: list[str],
    mode: str,
    k_values: tuple[int, ...]
) -> dict[int, float]:
    bm25 = BM25Retriever(documents)
    max_k = max(k_values)
    recalls = {k: [] for k in k_values}

    for example in qa_dataset:
        results = retrieve_for_mode(
            example["question"],
            documents,
            bm25,
            mode,
            max_k
        )

        for k in k_values:
            recalls[k].append(
                recall_at_k(results, example["relevant_document_ids"], k)
            )

    return {
        k: sum(values) / len(values)
        for k, values in recalls.items()
    }


def print_results(metrics_by_mode: dict[str, dict[int, float]], k_values: tuple[int, ...]) -> None:
    headers = ["mode", *[f"recall@{k}" for k in k_values]]
    rows = []

    for mode, metrics in metrics_by_mode.items():
        rows.append([mode, *[f"{metrics[k]:.2f}" for k in k_values]])

    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))

    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense, BM25, and hybrid retrieval with Recall@k."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus.txt"),
        help="Path to the corpus text file."
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("data/evaluation_qa.json"),
        help="Path to the evaluation QA JSON file."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_corpus(args.corpus)
    qa_dataset = load_qa_dataset(args.qa)

    metrics_by_mode = {
        mode: evaluate_mode(qa_dataset, documents, mode, DEFAULT_K_VALUES)
        for mode in MODES
    }

    print_results(metrics_by_mode, DEFAULT_K_VALUES)


if __name__ == "__main__":
    main()
