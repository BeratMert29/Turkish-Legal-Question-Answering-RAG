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
ollama pull qwen2.5:7b
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
| `base` | BGE-M3 dense retrieval + Qwen2.5:7b |
| `rrf_rerank` | BM25 + dense RRF + BGE reranker + Qwen2.5:7b |
| `emb_ft` | Fine-tuned BGE-M3 + RRF rerank (requires trained model) |
| `llm_ft` | Dense retrieval + fine-tuned LLM (requires Ollama model) |
| `full` | Fine-tuned embedding + rerank + fine-tuned LLM |

Pass multiple stages as comma-separated: `--stages base,rrf_rerank,emb_ft`

---

## Quick Start (development)

### Prerequisites

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

Ollama server settings (set before `ollama serve`):

```bash
# One model resident at a time: generation LLM, then judge LLM, never both
# (a 12 GB GPU cannot hold the 7B generator and the 8B judge together).
export OLLAMA_MAX_LOADED_MODELS=1
# Same context window for the base and the fine-tuned LLM (config.LLM_NUM_CTX);
# the OpenAI-compatible endpoint cannot set num_ctx per request.
export OLLAMA_CONTEXT_LENGTH=8192
```

New model downloads on first use: the NLI model
(`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) and the
multilingual semantic-similarity model
(`paraphrase-multilingual-mpnet-base-v2`).  The embedder and reranker load in
fp16 on CUDA, so rebuild the FAISS index after upgrading.

### Build index and evaluate

```bash
# Build FAISS index from the default corpus
PYTHONUTF8=1 python scripts/02_build_index.py --corpus data/processed/corpus.jsonl

# Run ablation on built-in benchmark
PYTHONUTF8=1 python scripts/14_eval_all_stages.py --stages base,rrf_rerank
```

---

## Local run (12 GB GPU)

Tested target: RTX 4070 Super 12 GB. Run from `CENG493_Project/`; the 7B
generator (`qwen2.5:7b`) and 8B judge (`llama3.1:8b`) are loaded one at a time.
The 14B generator and `llama3.3:70b` judge are options on a larger GPU, but then
the LoRA must be retrained on the matching base.

```bash
# 1. Terminal A: Ollama server, one model resident, 8192-token context
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=8192 ollama serve

# 2. Terminal B: models
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
# optional, only for the llm_ft / full stages:
python scripts/13_export_lora_to_ollama.py

# 3. Python deps
pip install -r requirements.txt

# 4. Optional: standalone index + graph (needed only for run_baseline.py,
#    scripts/03 and the graph stage's graph.json; scripts/14 builds its own
#    in-memory FAISS index and auto-builds graph.json)
PYTHONUTF8=1 python scripts/02_build_index.py --corpus ../results/processed_data/corpus_chunks.jsonl
PYTHONUTF8=1 python scripts/15_build_graph.py

# 5. Smoke run (10 questions, two stages)
PYTHONUTF8=1 python scripts/14_eval_all_stages.py --stages base rrf_rerank --limit 10

# 6. Full run (all stages whose prerequisites exist; the default eval set)
PYTHONUTF8=1 python scripts/14_eval_all_stages.py
```

Re-export the LoRA with `13_export_lora_to_ollama.py` only if the Modelfile
changed (it bakes in `LLM_NUM_CTX` and `LLM_MAX_TOKENS`); the 7B adapter already
matches the base, no retraining is needed.

---

## Architecture

The system includes five evaluation stages that progressively add components:

| Stage | Components | Key Addition |
|-------|-----------|--------------|
| **base** | Dense retrieval (BGE-M3) + Qwen2.5:7b | Baseline system |
| **rrf_rerank** | BM25 + dense RRF + BGE-reranker-v2-m3 | Hybrid retrieval + cross-encoder |
| **emb_ft** | Fine-tuned BGE-M3 + RRF rerank | Task-specific embeddings |
| **llm_ft** | Dense retrieval + QLoRA fine-tuned LLM | Legal QA instruction tuning |
| **full** | Fine-tuned embedding + rerank + fine-tuned LLM | All components combined |

### Ablation design (one factor per step)

- **Base vs fine-tuned LLM** differ only in the LoRA weights: both use the same
  base size (`config.LLM_BASE_FOR_ABLATION` = `qwen2.5:7b`; the committed
  adapter in `results/model_configs/qwen25_lora` is trained on
  `config.LORA_BASE_HF_MODEL` = `Qwen/Qwen2.5-7B-Instruct`, so they match),
  the same `LLM_MAX_TOKENS` and the same `LLM_NUM_CTX`.  Each stage's
  `baseline_metrics.json` records `llm_model`, `llm_max_tokens`, `llm_num_ctx`.
- **rrf vs rrf_rerank**: both fuse the same top-`RERANKER_CANDIDATES` (50)
  dense and BM25 candidates; only the cross-encoder reorder differs.
- **graph** adds neighbours (adjacent chunks) of the top results.  Up to
  `GRAPH_NEIGHBOR_BUDGET` of the `TOP_K_FOR_GENERATION` context slots are
  reserved for them (`GRAPH_CONTEXT_RESERVE`), so the graph can change the
  generated answer.  Retrieval metrics (Recall/MRR/nDCG/source-hit) are always
  computed on the pre-expansion ranking, so graph and rrf_rerank have identical
  retrieval numbers by construction; the effect shows up in QA metrics.
- Failed generations stay in the QA-metric denominator and score 0; the count
  is reported (`qa_metrics.n_generation_failed_scored_zero`).
- Each stage writes `per_query.jsonl` (per-query metric arrays) and
  `confidence_intervals` (95% bootstrap CIs); the ablation table shows an F1 CI.
- Faithfulness (`Ctx-NLI`) is multilingual NLI of the answer against its
  retrieved context; hallucination samples are stratified by whether the gold
  chunk/law was retrieved at rank 1, 2-5 or missed.  Citation accuracy is
  reported for the model's own citations (`native`) and after citation
  injection (`injected`, mostly reflects retrieval overlap).
- The corpus holds out eval rows only for `--eval-set kaggle`.
  `14_eval_all_stages.py` aborts if fewer than 50% of `turkish_legal_rag`
  queries have gold chunk labels (stale index/corpus).

### Retrieval Pipeline

1. **Dual Retrieval**: BM25 (sparse) and FAISS dense search (default: top-50 each)
2. **Rank Fusion**: Reciprocal Rank Fusion (RRF) combines rankings
3. **Reranking**: BGE-reranker-v2-m3 cross-encoder reranks top candidates
4. **Context Assembly**: Top-K passages fed to LLM with question and system prompt

### Models

- **Embedding**: BAAI/bge-m3 (base and fine-tuned variants)
- **Reranker**: BAAI/bge-reranker-v2-m3
- **LLM**: Qwen2.5:7b via Ollama (local inference, no API key required)

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
    ├── 10_build_rag_train_data.py
    ├── 11_build_embedding_triplets.py
    ├── 12_finetune_embeddings.py  # fine-tune BGE-M3
    ├── 12b_finetune_reranker.py
    ├── 13_export_lora_to_ollama.py
    └── 14_eval_all_stages.py  # main evaluation entry point
```

---

## Fine-tuning

### LLM Fine-tuning (QLoRA)

```bash
# Fine-tune Qwen2.5-3B (safe fp16 LoRA, fits on 16GB GPU)
PYTHONUTF8=1 python scripts/08_finetune_llm.py --backend safe

# Fine-tune Qwen2.5-7B (4-bit QLoRA; the base the ablation uses)
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
| `LLM_MODEL` | `qwen2.5:7b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `EMBEDDING_BATCH_SIZE` | `8` | Batch size (increase to 64 on A100) |
| `TOP_K_RETRIEVAL` | 10 | Chunks retrieved per query |
| `TOP_K_FOR_GENERATION` | 5 | Chunks passed to LLM |
| `RERANKER_CANDIDATES` | 50 | Candidates fed to reranker |
| `CHUNK_SIZE` | 1400 | Characters per chunk |
| `LLM_BASE_FOR_ABLATION` | `qwen2.5:7b` | Base LLM; must match the LoRA base |
| `LLM_NUM_CTX` | 8192 | Context window for both LLMs |
| `LLM_JUDGE_SAMPLE_SIZE` | `None` | Judge every answer (int = cap) |
| `NLI_MODEL` | mDeBERTa xnli | Multilingual NLI for faithfulness |
| `TRUST_REMOTE_CODE` | `False` | Passed to HF `from_pretrained` |

---

## Default Evaluation Set: `turkish_legal_rag`

`scripts/14_eval_all_stages.py` and `run_baseline.py` now evaluate on
`turkish_legal_rag` by default: 195 questions with explicit law + article gold
labels, so chunk-level Recall/MRR/nDCG are meaningful.

- **Source:** [`mtntasci/turkish-legal-rag`](https://huggingface.co/datasets/mtntasci/turkish-legal-rag),
  config `qa_benchmark`, split `test` (290 rows).
- **License / attribution:** CC-BY-4.0. Dataset by `mtntasci`; questions derive
  from the Kaggle legal QA data our corpus is built from.
- **Filtering** (290 -> 195 kept):
  - drop 90 rows whose `source_origin` is not `kaggle_batuhankalem` (templated, low quality)
  - drop 0 rows whose law has no chunks in our corpus (all 8 laws are indexed)
  - drop 5 rows whose question appears in a `qa_train*.jsonl` fine-tuning file (leakage; normalized text match)
  - `madde_no` is normalised to the corpus convention (`"3-"` -> `"3"`); 1 kept row has no article
  - 4 kept questions also appear in `qa_eval.jsonl`
- **Gold labels:** `source` + `madde_no` select the corpus chunks of that law and
  article (194/195 labeled this way, 195/195 with at least one gold chunk).
- **Headline metrics:** chunk-level R@5/R@10/MRR/nDCG@10 lead the PRIMARY table
  when at least 50% of queries have gold chunk labels
  (`config.HEADLINE_CHUNK_MIN_LABELED_FRACTION`); otherwise source-hit stays the headline.

Rebuild the set (outputs `results/processed_data/qa_turkish_legal_rag.jsonl` and
`.report.json`; pass `--input rows.json` to use a local copy instead of downloading):

```bash
python CENG493_Project/scripts/16_prepare_turkish_legal_rag.py
```

Run the ablation (default set is `turkish_legal_rag`):

```bash
python scripts/14_eval_all_stages.py --stages base rrf_rerank graph emb_ft full
```

`scripts/10_eval_finetuned.py` was removed (superseded by
`14_eval_all_stages.py --stages llm_ft full`).

HMGS is still available with `--eval-set hmgs` (Kaggle with `--eval-set kaggle`).

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
