> RAG pipeline for Turkish legal question answering with hybrid retrieval, reranking, and fine-tuned components.

# Turkish Legal QA with RAG

A retrieval-augmented generation (RAG) pipeline for answering Turkish legal questions, built for CENG493 (Information Retrieval). The system combines dense retrieval with BM25, reciprocal rank fusion, cross-encoder reranking, and fine-tuned embedding/LLM components.

---

## Custom Data Evaluation (for Evaluators)

The system is fully parameterized — you can plug in your own document collection and benchmark questions without modifying any code.

### Step 1 — Prerequisites

```bash
pip install -r requirements.txt

# Pull the LLM used for answer generation (requires Ollama)
ollama pull qwen2.5:14b
```

Ollama install: https://ollama.com/download

### Step 2 — Prepare your documents

You can provide documents in two ways:

**Option A — Directory of `.txt` or `.pdf` files (recommended)**

Place your documents in a folder, e.g. `my_docs/`. The system will automatically chunk and index them.

```
my_docs/
├── kanun1.txt
├── kanun2.pdf
└── yonetmelik.txt
```

**Option B — Pre-chunked JSONL corpus**

If your corpus is already chunked, provide a JSONL file where each line is:

```json
{"chunk_id": "doc1_chunk_0", "doc_id": "doc1", "text": "chunk text here", "source": "kanun1.txt", "char_len": 1234}
```

The evaluator's standard format (with top-level `id` and nested `metadata`) is also accepted automatically.

### Step 3 — Prepare your benchmark file

Provide a JSON file with your question-answer pairs. Two formats are accepted:

**Format A — Gold benchmark (with gold chunk IDs)**

```json
[
  {
    "question_id": "q001",
    "question": "Türk Medeni Kanunu'na göre reşit olma yaşı kaçtır?",
    "verified_answer": "18 yaşını dolduran kişi ergin sayılır.",
    "gold_sources": [
      {"source_id": "tmc_chunk_42", "source": "turk_medeni_kanunu.txt"}
    ]
  }
]
```

**Format B — RAG eval format (with gold chunk IDs)**

```json
[
  {
    "query_id": "q001",
    "query": "Türk Medeni Kanunu'na göre reşit olma yaşı kaçtır?",
    "gold_answer_extract": "18 yaşını dolduran kişi ergin sayılır.",
    "gold_chunk_ids": ["tmc_chunk_42"],
    "source": "turk_medeni_kanunu.txt"
  }
]
```

> If you do not have gold chunk IDs, omit `gold_sources` / `gold_chunk_ids`. Retrieval metrics will not be computed but QA and faithfulness metrics will still run.

### Step 4 — Run evaluation

**With a document folder:**

```bash
PYTHONUTF8=1 python scripts/14_eval_all_stages.py \
    --docs-path my_docs/ \
    --eval-data my_benchmark.json \
    --stages base,rrf_rerank
```

**With a pre-chunked corpus JSONL:**

```bash
PYTHONUTF8=1 python scripts/14_eval_all_stages.py \
    --corpus my_corpus.jsonl \
    --eval-data my_benchmark.json \
    --stages base,rrf_rerank
```

**On Windows (PowerShell):**

```powershell
$env:PYTHONUTF8="1"
python scripts/14_eval_all_stages.py `
    --docs-path my_docs\ `
    --eval-data my_benchmark.json `
    --stages base,rrf_rerank
```

### Step 5 — Read results

Results are written to `results/` per stage:

```
results/
├── stage_base/
│   ├── baseline_metrics.json   ← all metrics in one file
│   └── predictions.jsonl       ← per-question predictions
├── stage_rrf_rerank/
│   ├── baseline_metrics.json
│   └── predictions.jsonl
└── ablation_summary.json       ← side-by-side comparison of all stages
```

`baseline_metrics.json` contains:

| Key | Description |
|-----|-------------|
| `retrieval_metrics` | Recall@5, Recall@10, MRR, nDCG@10, Precision@K |
| `qa_metrics` | F1, ROUGE-L, BLEU, Exact Match, Citation Accuracy |
| `hallucination_summary` | NLI-based faithfulness rate |
| `llm_judge_score` | LLM-judged quality (0–1) |
| `semantic_similarity` | Embedding similarity to gold answer |

### Available stages

| Stage ID | Description |
|----------|-------------|
| `base` | BGE-M3 dense retrieval + Qwen2.5:14b |
| `rrf_rerank` | BM25 + dense RRF + BGE reranker + Qwen2.5:14b |
| `emb_ft` | Fine-tuned BGE-M3 + RRF rerank (requires trained model) |
| `llm_ft` | Dense retrieval + fine-tuned LLM (requires Ollama model) |
| `full` | Fine-tuned embedding + rerank + fine-tuned LLM |

Pass multiple stages as comma-separated: `--stages base,rrf_rerank,emb_ft`

---

## Quick Start (development)

### Prerequisites

```bash
pip install -r requirements.txt
ollama pull qwen2.5:14b
```

### Build index and evaluate

```bash
# Build FAISS index from the default corpus
PYTHONUTF8=1 python scripts/02_build_index.py --corpus data/processed/corpus.jsonl

# Run ablation on built-in benchmark
PYTHONUTF8=1 python scripts/14_eval_all_stages.py --stages base,rrf_rerank
```

---

## Architecture

The system includes five evaluation stages that progressively add components:

| Stage | Components | Key Addition |
|-------|-----------|--------------|
| **base** | Dense retrieval (BGE-M3) + Qwen2.5:14b | Baseline system |
| **rrf_rerank** | BM25 + dense RRF + BGE-reranker-v2-m3 | Hybrid retrieval + cross-encoder |
| **emb_ft** | Fine-tuned BGE-M3 + RRF rerank | Task-specific embeddings |
| **llm_ft** | Dense retrieval + QLoRA fine-tuned LLM | Legal QA instruction tuning |
| **full** | Fine-tuned embedding + rerank + fine-tuned LLM | All components combined |

### Retrieval Pipeline

1. **Dual Retrieval**: BM25 (sparse) and FAISS dense search (default: top-50 each)
2. **Rank Fusion**: Reciprocal Rank Fusion (RRF) combines rankings
3. **Reranking**: BGE-reranker-v2-m3 cross-encoder reranks top candidates
4. **Context Assembly**: Top-K passages fed to LLM with question and system prompt

### Models

- **Embedding**: BAAI/bge-m3 (base and fine-tuned variants)
- **Reranker**: BAAI/bge-reranker-v2-m3
- **LLM**: Qwen2.5:14b via Ollama (local inference, no API key required)

---

## Evaluation Metrics

**Retrieval**: Recall@5, Recall@10, MRR, nDCG@10, Source Hit@K, Precision@K

**QA**: F1, ROUGE-L, BLEU, Exact Match, Citation Accuracy, Source-in-Context Rate

**Faithfulness**: NLI-based hallucination detection, semantic similarity, perplexity

**Overall**: LLM judge scores (quality, faithfulness, relevancy, coherence), 3 composite scenario scores

---

## Project Structure

```
CENG493_Project/
├── config.py                  # all hyperparameters and paths
├── requirements.txt
├── utils.py
├── data/
│   ├── corpus_loader.py       # custom doc ingestion (.txt/.pdf → chunks)
│   └── processed/             # chunked corpus, train/eval JSONL files
├── index/                     # FAISS vector index
├── models/
│   ├── bge-m3-turkish-legal/  # fine-tuned embedding model
│   └── qwen25_lora/           # QLoRA adapter weights
├── results/                   # per-stage eval output (metrics, predictions)
├── evaluation/                # metric modules (F1, RAGAS, hallucination, etc.)
├── generation/
│   └── rag_pipeline.py        # retrieve → assemble context → generate
├── retrieval/                 # dense, BM25, RRF, reranker modules
└── scripts/
    ├── 02_build_index.py      # build FAISS index (--corpus or --docs-path)
    ├── 08_finetune_llm.py     # QLoRA fine-tune Qwen2.5
    ├── 12_finetune_embeddings.py  # fine-tune BGE-M3
    ├── 13_export_lora_to_ollama.py
    └── 14_eval_all_stages.py  # main evaluation entry point
```

---

## Fine-tuning

### LLM Fine-tuning (QLoRA)

```bash
# Fine-tune Qwen2.5-3B (safe fp16 LoRA, fits on 16GB GPU)
PYTHONUTF8=1 python scripts/08_finetune_llm.py --backend safe

# Fine-tune Qwen2.5-14B (4-bit QLoRA, requires 40GB+ VRAM)
PYTHONUTF8=1 python scripts/08_finetune_llm.py --backend qlora

# Export LoRA adapter to Ollama
PYTHONUTF8=1 python scripts/13_export_lora_to_ollama.py
```

Training data format (`llm.jsonl`) — each line:

```json
{"id": "sft_001", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Embedding Fine-tuning

```bash
# Build triplet training data from corpus + QA pairs
PYTHONUTF8=1 python scripts/11_build_embedding_triplets.py

# Fine-tune BGE-M3 with contrastive loss
PYTHONUTF8=1 python scripts/12_finetune_embeddings.py
```

Training data format (`embedding.jsonl`) — each line:

```json
{"id": "trip_001", "query": "soru metni", "positive_passage": "ilgili metin", "negative_passage": "alakasız metin"}
```

---

## Configuration

All settings are in `config.py`. Key values:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `qwen2.5:14b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `EMBEDDING_BATCH_SIZE` | `8` | Batch size (increase to 64 on A100) |
| `TOP_K_RETRIEVAL` | 10 | Chunks retrieved per query |
| `TOP_K_FOR_GENERATION` | 5 | Chunks passed to LLM |
| `RERANKER_CANDIDATES` | 50 | Candidates fed to reranker |
| `CHUNK_SIZE` | 1400 | Characters per chunk |

---

## Retrieval Evaluation Notes

### Why source-level metrics are primary

The HMGS gold test set (161 questions) contains Turkish bar-exam questions whose
ground-truth relevance cannot be determined at chunk level: the questions do not
mention a specific law article in most cases, and the correct answer text is not
verbatim in any corpus chunk.  Assigning arbitrary corpus chunks as "relevant"
(e.g. the first N chunks of the source law) inflates Recall/MRR/NDCG with
invented ground-truth.

Instead, two tiers of retrieval metrics are reported:

**Primary — source-level (all queries with a known gold law):**
- `source_hit@5` / `source_hit@10`: fraction of queries where ≥1 top-k chunk is
  from the correct law
- `source_MRR`: mean reciprocal rank of the first chunk from the gold law
- `source_precision@5`: mean fraction of top-5 chunks from the gold law
- Covers all 161 HMGS queries (and all Kaggle queries that have a `source` field)

**Secondary — chunk-level (gold-labeled subset only):**
- Recall@5/10, MRR, nDCG@10 computed only for the subset of queries that have
  verifiable article-level ground-truth (n is printed per stage; for HMGS this
  is typically 1–2 queries).  Treat these numbers as indicative, not definitive.

### Silver lexical labels (optional)

To improve chunk-level coverage on HMGS, a "silver" labeling strategy is
available.  It ranks corpus chunks within the gold law by normalized token
overlap with the question and answer, then labels the top-m above a threshold.
Labels are heuristic (not gold-standard) and are kept separate in coverage stats.

Enable in `CENG493_Project/config.py`:

```python
RELEVANCE_SILVER_LEXICAL = True  # default: False
SILVER_TOP_M = 3                 # chunks labeled per query
SILVER_THRESHOLD = 0.10          # minimum token-overlap score
```

---

## Tech Stack

- **Python 3.11**, PyTorch, HuggingFace (transformers, PEFT, TRL, sentence-transformers)
- **Retrieval**: FAISS (GPU/CPU), rank_bm25
- **Inference**: Ollama (local, no API key)
- **Evaluation**: RAGAS, NLI-based hallucination detection, LLM judge
- **GPU**: Tested on NVIDIA A100 (80GB) and RTX 5070 Ti (16GB)
