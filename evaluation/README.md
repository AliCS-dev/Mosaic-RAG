# Retrieval Evaluation

This folder contains a small retrieval evaluation for Mosaic-RAG. The test data
currently includes 48 RAG-related documents and 25 questions.

The script sends every question to the running retriever API and compares three
retrieval modes:

- `dense`: embedding-based dense retrieval with FAISS
- `bm25`: the BM25 retriever
- `hybrid`: Reciprocal Rank Fusion over dense and BM25 results

## Metrics

`Recall@k` measures how many known relevant documents appear in the top `k` retrieved results.

For one question:

```text
Recall@k = relevant documents found in top k / total relevant documents
```

Higher is better.

`Recall@5` is stricter than `Recall@10` because the relevant document must appear closer to the top of the result list. `Recall@10` is more forgiving and measures whether the system found the relevant document somewhere in a wider set of candidates.

The questions include exact technical terms, natural-language paraphrases, and a
question with more than one relevant document. This makes the comparison less
trivial than the original seven-document test while keeping the labels easy to
inspect manually.

Before evaluating, the script checks that the service is using FAISS and has
loaded the same number of documents as the local corpus. It stops instead of
reporting misleading scores when the service is unavailable, is using the
keyword fallback, or has a stale corpus.

## Run

Start the retriever, then run the evaluation from the repository root:

```bash
docker compose up -d --build retriever
python3 -B evaluation/evaluate_retrieval.py
```

The default retriever URL is `http://localhost:8001`. A different URL can be
provided through `--retriever-url` or the `RETRIEVER_URL` environment variable:

```bash
python3 -B evaluation/evaluate_retrieval.py \
  --retriever-url http://localhost:8001
```

Output format:

```text
mode    recall@5  recall@10
------  --------  ---------
dense   0.XX      0.XX
bm25    0.XX      0.XX
hybrid  0.XX      0.XX
```

After the summary table, the script lists each question that missed at least
one relevant document in the top five. It shows whether that document appeared
at rank 6-10 or was absent from the top ten entirely.

Scores depend on the installed dense model and should be compared on the same
corpus, QA labels, and model version. A higher score is better. Compare
`Recall@5` first because it asks each retriever to place relevant evidence in a
smaller, more useful result set; use `Recall@10` to see whether missed evidence
appears when the result set is widened.
