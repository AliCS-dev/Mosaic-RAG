# Retrieval Evaluation

This folder contains a small retrieval evaluation for Mosaic-RAG. The test data
currently includes 48 RAG-related documents and 25 questions.

The script compares three retrieval modes:

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

The dense retriever uses `sentence-transformers` and `faiss-cpu` when those packages are installed. If they are not available in the local Python environment, the evaluation script falls back to the old keyword placeholder and prints a warning. The retriever Docker image installs the FAISS dependencies from `retriever/requirements.txt`.

## Run

```bash
python3 -B evaluation/evaluate_retrieval.py
```

Output format:

```text
mode    recall@5  recall@10
------  --------  ---------
dense   0.XX      0.XX
bm25    0.XX      0.XX
hybrid  0.XX      0.XX
```

Scores depend on the installed dense model and should be compared on the same
corpus, QA labels, and model version. A higher score is better. Compare
`Recall@5` first because it asks each retriever to place relevant evidence in a
smaller, more useful result set; use `Recall@10` to see whether missed evidence
appears when the result set is widened.
