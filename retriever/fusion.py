from interfaces import RetrieverResult


SOURCE_RETRIEVER = "rrf"
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_sets: list[list[RetrieverResult]],
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K
) -> list[RetrieverResult]:
    scores: dict[str, float] = {}
    documents_by_id: dict[str, RetrieverResult] = {}

    for results in result_sets:
        for position, result in enumerate(results, start=1):
            rank = result.rank if result.rank > 0 else position

            scores[result.document_id] = (
                scores.get(result.document_id, 0.0) + 1 / (rrf_k + rank)
            )
            documents_by_id.setdefault(result.document_id, result)

    ranked_document_ids = sorted(
        scores,
        key=lambda document_id: (-scores[document_id], document_id)
    )

    fused_results: list[RetrieverResult] = []

    for rank, document_id in enumerate(ranked_document_ids[:top_k], start=1):
        document = documents_by_id[document_id]

        fused_results.append(
            RetrieverResult(
                document_id=document.document_id,
                text=document.text,
                score=scores[document_id],
                rank=rank,
                source_retriever=SOURCE_RETRIEVER
            )
        )

    return fused_results
