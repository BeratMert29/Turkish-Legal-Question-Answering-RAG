# Evaluator Dataset

Place the instructor-provided dataset files in this folder:

| File | Description |
|------|-------------|
| `corpus.jsonl` | Document collection for retrieval (7,579 chunks) |
| `gold_benchmark.json` | 240 questions with gold answers + relevant document IDs |
| `rag_eval.json` | 1,000 question alternative evaluation set |
| `embedding.jsonl` | Embedding model training data (optional, for fine-tuning) |
| `reranker.jsonl` | Reranker training data (optional, for fine-tuning) |
| `llm.jsonl` | LLM training data (optional, for fine-tuning) |

Once files are placed here, run from the `CENG493_Project/` directory:

```powershell
# Windows
$env:PYTHONUTF8="1"
python scripts/14_eval_all_stages.py `
  --corpus ..\Datasets_Ceng493_legal_rag\corpus.jsonl `
  --eval-data ..\Datasets_Ceng493_legal_rag\gold_benchmark.json `
  --stages base,rrf_rerank,emb_ft,llm_ft,full
```

```bash
# macOS / Linux
PYTHONUTF8=1 python scripts/14_eval_all_stages.py \
  --corpus ../Datasets_Ceng493_legal_rag/corpus.jsonl \
  --eval-data ../Datasets_Ceng493_legal_rag/gold_benchmark.json \
  --stages base,rrf_rerank,emb_ft,llm_ft,full
```

See `EVALUATOR.md` for full setup instructions.
