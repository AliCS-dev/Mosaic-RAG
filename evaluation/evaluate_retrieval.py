import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODES = ("dense", "bm25", "hybrid")
DEFAULT_K_VALUES = (5, 10)
DEFAULT_RETRIEVER_URL = "http://localhost:8001"


class EvaluationError(Exception):
    pass


def load_corpus(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_qa_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(
    url: str,
    timeout: float,
    payload: dict | None = None
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET"
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise EvaluationError(
            f"Retriever returned HTTP {error.code} for {url}: {response_body}"
        ) from error
    except URLError as error:
        raise EvaluationError(
            f"Could not connect to the retriever at {url}: {error.reason}"
        ) from error
    except (json.JSONDecodeError, TimeoutError) as error:
        raise EvaluationError(f"Invalid response from {url}: {error}") from error


class RetrieverClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        return request_json(f"{self.base_url}/", self.timeout)

    def retrieve(self, query: str, mode: str, top_k: int, request_id: str) -> list[dict]:
        response = request_json(
            f"{self.base_url}/retrieve",
            self.timeout,
            {
                "request_id": request_id,
                "query": query,
                "top_k": top_k,
                "retrieval_mode": mode
            }
        )
        results = response.get("retrieved_documents")

        if not isinstance(results, list):
            raise EvaluationError(
                "Retriever response is missing the retrieved_documents list."
            )

        return results


def validate_dataset(documents: list[str], qa_dataset: list[dict]) -> None:
    if not documents:
        raise EvaluationError("The corpus is empty.")

    if not qa_dataset:
        raise EvaluationError("The QA dataset is empty.")

    document_ids = {f"doc-{index}" for index in range(len(documents))}

    for index, example in enumerate(qa_dataset, start=1):
        question = example.get("question")
        relevant_ids = example.get("relevant_document_ids")

        if not isinstance(question, str) or not question.strip():
            raise EvaluationError(f"Question {index} has no valid question text.")

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise EvaluationError(f"Question {index} has no relevant document IDs.")

        unknown_ids = set(relevant_ids) - document_ids
        if unknown_ids:
            raise EvaluationError(
                f"Question {index} references unknown documents: "
                f"{', '.join(sorted(unknown_ids))}"
            )


def validate_service(health: dict, expected_document_count: int) -> None:
    if health.get("status") != "running":
        raise EvaluationError("Retriever health check did not report a running service.")

    backend = health.get("dense_backend")
    if backend != "faiss":
        raise EvaluationError(
            f"Dense backend is {backend!r}, not 'faiss'. "
            "Evaluation stopped to avoid reporting placeholder results."
        )

    loaded_count = health.get("documents_loaded")
    if loaded_count != expected_document_count:
        raise EvaluationError(
            f"Retriever loaded {loaded_count} documents, but the evaluation corpus has "
            f"{expected_document_count}. Restart the retriever with the current corpus."
        )


def recall_at_k(results: list[dict], relevant_document_ids: list[str], k: int) -> float:
    relevant_ids = set(relevant_document_ids)
    retrieved_ids = {
        result.get("document_id")
        for result in results[:k]
    }

    return len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids)


def evaluate_mode(
    qa_dataset: list[dict],
    client: RetrieverClient,
    mode: str,
    k_values: tuple[int, ...]
) -> tuple[dict[int, float], list[dict]]:
    max_k = max(k_values)
    strict_k = min(k_values)
    recalls = {k: [] for k in k_values}
    misses = []

    for index, example in enumerate(qa_dataset, start=1):
        results = client.retrieve(
            example["question"],
            mode,
            max_k,
            f"evaluation-{mode}-{index}"
        )

        for k in k_values:
            recalls[k].append(
                recall_at_k(results, example["relevant_document_ids"], k)
            )

        result_ranks = {
            result.get("document_id"): position
            for position, result in enumerate(results, start=1)
        }
        missing_ids = [
            document_id
            for document_id in example["relevant_document_ids"]
            if result_ranks.get(document_id, max_k + 1) > strict_k
        ]

        if missing_ids:
            misses.append(
                {
                    "question": example["question"],
                    "missing_ids": missing_ids,
                    "result_ranks": result_ranks,
                    "max_k": max_k
                }
            )

    metrics = {
        k: sum(values) / len(values)
        for k, values in recalls.items()
    }
    return metrics, misses


def print_results(metrics_by_mode: dict[str, dict[int, float]], k_values: tuple[int, ...]) -> None:
    headers = ["mode", *[f"recall@{k}" for k in k_values]]
    rows = [
        [mode, *[f"{metrics[k]:.2f}" for k in k_values]]
        for mode, metrics in metrics_by_mode.items()
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))

    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_misses(misses_by_mode: dict[str, list[dict]], strict_k: int) -> None:
    print(f"\nMisses at Recall@{strict_k}:")

    for mode, misses in misses_by_mode.items():
        print(f"\n{mode}:")
        if not misses:
            print("  none")
            continue

        for miss in misses:
            statuses = []
            for document_id in miss["missing_ids"]:
                rank = miss["result_ranks"].get(document_id)
                status = f"rank {rank}" if rank else f"not in top {miss['max_k']}"
                statuses.append(f"{document_id}: {status}")

            print(f"  - {miss['question']}")
            print(f"    {', '.join(statuses)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the live retriever API with Recall@k."
    )
    parser.add_argument(
        "--retriever-url",
        default=os.getenv("RETRIEVER_URL", DEFAULT_RETRIEVER_URL),
        help=f"Retriever service base URL (default: {DEFAULT_RETRIEVER_URL})."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus.txt"),
        help="Corpus used by the retriever, for ID and document-count validation."
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("data/evaluation_qa.json"),
        help="Path to the evaluation QA JSON file."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for each retriever request."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_corpus(args.corpus)
    qa_dataset = load_qa_dataset(args.qa)
    validate_dataset(documents, qa_dataset)

    client = RetrieverClient(args.retriever_url, args.timeout)
    health = client.health()
    validate_service(health, len(documents))

    print(
        f"Retriever: {args.retriever_url} | "
        f"backend: {health['dense_backend']} | "
        f"documents: {health['documents_loaded']}"
    )

    metrics_by_mode = {}
    misses_by_mode = {}

    for mode in MODES:
        metrics, misses = evaluate_mode(
            qa_dataset,
            client,
            mode,
            DEFAULT_K_VALUES
        )
        metrics_by_mode[mode] = metrics
        misses_by_mode[mode] = misses

    print()
    print_results(metrics_by_mode, DEFAULT_K_VALUES)
    print_misses(misses_by_mode, min(DEFAULT_K_VALUES))


if __name__ == "__main__":
    try:
        main()
    except (EvaluationError, OSError, json.JSONDecodeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
