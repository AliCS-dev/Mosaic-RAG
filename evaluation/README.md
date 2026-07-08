# Retrieval Evaluation

This folder contains a small retrieval evaluation for Mosaic-RAG.

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

With the current tiny corpus, `Recall@10` is not very informative because there are fewer than 10 documents. It is included now so the same script can be reused when the corpus grows.

The dense retriever uses `sentence-transformers` and `faiss-cpu` when those packages are installed. If they are not available in the local Python environment, the evaluation script falls back to the old keyword placeholder and prints a warning. The retriever Docker image installs the FAISS dependencies from `retriever/requirements.txt`.

## Run

```bash
python3 -B evaluation/evaluate_retrieval.py
```

Expected output format:

```text
mode    recall@5  recall@10
------  --------  ---------
dense   1.00      1.00
bm25    1.00      1.00
hybrid  1.00      1.00
```
