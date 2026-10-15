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

Each run writes to its own directory, so runs on different eval sets or quick
`--limit` runs never overwrite each other:

```
results/<eval_set>[_limitN]/          e.g. results/turkish_legal_rag/, results/hmgs_limit10/
├── <stage>/                          base, rrf_rerank, llm_ft, ...
│   ├── baseline_metrics.json         all metrics + hyperparameters + provenance
│   ├── predictions.jsonl             per-question answers, contexts, citation scores
│   ├── per_query.jsonl               per-question metric values (CIs, paired tests)
│   └── judge_raw_<metric>_<run>.jsonl
├── <stage>_FAILED/                   a stage whose generation/judge failure rate was too high
└── ablation_summary.json             {"run": ..., "stages": {...}, "comparisons": {...}}
```

Re-running some stages merges them into the existing `ablation_summary.json`.
`comparisons` holds paired bootstrap differences (same queries, Holm-corrected)
for the one-factor stage pairs.  See [Evaluation Metrics](#evaluation-metrics)
for the keys of `baseline_metrics.json`.

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

Tested target: RTX 4070 Super 12GB. Run from `CENG493_Project/`; the 7B
generator (`qwen2.5:7b`) and 8B judge (`llama3.1:8b`) are loaded one at a time.
The 14B generator and `llama3.3:70b` judge are options on a larger GPU, but then
the LoRA must be retrained on the matching base.

```bash
# 1. Terminal A: Ollama server, one model resident.  The context window,
#    output length, stop sequences and seed are sent with every request
#    (config.LLM_NUM_CTX etc.), so OLLAMA_CONTEXT_LENGTH is no longer required.
OLLAMA_MAX_LOADED_MODELS=1 ollama serve

# 2. Terminal B: models
ollama pull qwen2.5:7b
ollama pull llama3.1:8b          # judge; or run scripts/14 with --no-judge
# llm_ft / full stages: re-export the LoRA so its Modelfile has the current
# stop sequences and context window (older exports produced runaway answers)
python scripts/13_export_lora_to_ollama.py

# 3. Python deps (adds sacrebleu, snowballstemmer, pypdf, requests)
pip install -r requirements.txt

# 4. Optional: re-check the turkish_legal_rag gold labels against the current
#    corpus (needs combined_dataset.csv via `git lfs pull`; the checked labels
#    are committed, so this is only needed after changing the chunker/corpus)
PYTHONUTF8=1 python scripts/17_check_tlr_labels.py --dry-run

# 5. Smoke run (10 questions, two stages) -> results/turkish_legal_rag_limit10/
PYTHONUTF8=1 python scripts/14_eval_all_stages.py --stages base rrf_rerank --limit 10

# 6. Full run (all stages whose prerequisites exist) -> results/turkish_legal_rag/
PYTHONUTF8=1 python scripts/14_eval_all_stages.py
```

`scripts/14` builds its FAISS index, BM25 index and graph in memory from the
corpus, so `02_build_index.py` / `15_build_graph.py` are needed only for
`run_baseline.py` (which refuses a saved index built from a different corpus).
The chunker changed (article titles, short articles kept), so rebuild a saved
index before using `run_baseline.py`.

Perplexity loads the generator's HF weights (`Qwen/Qwen2.5-7B-Instruct`, 4-bit,
plus the LoRA for the fine-tuned stages); set `PERPLEXITY_ENABLED = False` in
`config.py` to skip it.  Useful flags: `--no-judge`, `--limit N`,
`--results-root DIR`, `--eval-set {turkish_legal_rag,hmgs,kaggle}`.

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
  and every request carries the same `num_ctx`, `num_predict`, stop sequences
  and seed (`config.LLM_NUM_CTX`, `LLM_MAX_TOKENS`, `LLM_STOP`, `SEED`).
- **rrf vs rrf_rerank**: both fuse the same top-`RERANKER_CANDIDATES` (50)
  dense and BM25 candidates; only the cross-encoder reorder differs.
- **graph** adds neighbours (adjacent articles) of the kept top results.  Up to
  `GRAPH_NEIGHBOR_BUDGET` context slots go to neighbours, and only when such a
  neighbour exists.  Retrieval metrics are computed on the pre-expansion
  ranking, so graph and rrf_rerank have identical retrieval numbers; the
  effect shows up in answer metrics.
- **Paired comparisons**: `compare_stages` reports later-minus-earlier deltas
  for `base→hybrid`, `base→rrf`, `rrf→rrf_rerank`, `rrf_rerank→graph`,
  `base→llm_ft`, `rrf_rerank→emb_ft`, `emb_ft→full` on the same queries, with
  95% bootstrap CIs and Holm-corrected p-values.
- Failed generations stay in the QA-metric denominator and score 0.  When more
  than `MAX_FAILURE_RATE` fail, the LLM-based metrics are skipped (None) and the
  stage is written to `<stage>_FAILED/`, marked `[FAILED]` in the table.
- Every stage records provenance: git commit, seed, eval-file hash, corpus
  fingerprint, Ollama model digests, package versions, BM25 tokenizer state.

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

**Retrieval** (gold-labeled queries):
- article level — Hit@5/10, MRR, nDCG@10 over (law, article) ids
  (`retrieval_metrics.article_level`); independent of chunk size
- chunk level — Recall@5/10, MRR, nDCG@10, Hit@k, capped recall, Precision@k
  (one deduplicated ranking per query; None when no query is labeled)
- source level — law hit@5/10, MRR, precision@5 for every query with a known law

**Answer quality**: token F1 with separate token precision / recall, chrF++ and
BLEU (sacrebleu, corpus level), ROUGE-L, exact match on token boundaries,
answer containment, semantic similarity (multilingual mpnet), answer length,
truncation and runaway-continuation rates.

**Faithfulness (NLI, every answer)**: `faithfulness_rate` = mean fraction of
answer sentences entailed by the context the generator actually saw
(multilingual mDeBERTa); `gold_claim_recall` = fraction of gold-answer
sentences the answer entails.

**Citations**: the model's own `[Kaynak N]` citations checked against the gold
article — precision, recall, invalid rate — next to the precision of a random
context citation (`qa_metrics.citation_article_level`); law-level citation
accuracy is kept for reference.

**LLM judge** (`llama3.1:8b`, every answer, 0/0.5/1 rubric): answer quality,
faithfulness, relevancy, coherence; parse and call failures are counted, and a
failures-as-zero sensitivity mean is reported.  Calibrate against human labels
with `scripts/18_judge_calibration.py` (Cohen's kappa).

**Perplexity**: answer tokens only, under the generator's own weights.

**Composites**: Scenario 1–3 with their components, weights and n saved under
`scenario_components`; chunk MRR enters Scenario 1 only when most queries are
gold-labeled.  Every per-query metric gets a 95% bootstrap CI.

---

## Project Structure

```
CENG493_Project/
├── config.py                  # all hyperparameters and paths
├── requirements.txt
├── utils.py                   # Turkish normalisation, citation injection, provenance
├── data/
│   ├── data_processor.py      # chunking (article-aware), eval/train sets, gold labels
│   ├── tlr_labels.py          # check turkish_legal_rag gold articles against the law text
│   ├── extra_laws_cleaner.py  # clean the scraped supplementary laws
│   └── corpus_loader.py       # custom doc ingestion (.txt/.pdf → chunks)
├── retrieval/                 # dense (FAISS), BM25, RRF / hybrid fusion, reranker, graph
├── generation/
│   └── rag_pipeline.py        # context assembly → Ollama /api/chat
├── evaluation/                # retrieval, QA, citation, NLI, judge, perplexity, stats
├── pipeline/
│   ├── evaluation.py          # run_stage: one ablation stage end-to-end
│   ├── metric_input.py        # metric rows, per-query records, CIs
│   ├── report.py              # ablation tables, paired stage comparisons
│   └── stages.py              # stage registry
├── results/                   # results/<eval_set>[_limitN]/<stage>/ (scripts/14)
└── scripts/
    ├── 02_build_index.py      # build FAISS index (--corpus or --docs-path)
    ├── 08_finetune_llm.py     # QLoRA fine-tune Qwen2.5
    ├── 10_build_rag_train_data.py
    ├── 11_build_embedding_triplets.py
    ├── 12_finetune_embeddings.py  # fine-tune BGE-M3
    ├── 12b_finetune_reranker.py
    ├── 13_export_lora_to_ollama.py
    ├── 14_eval_all_stages.py  # main evaluation entry point
    ├── 16_prepare_turkish_legal_rag.py
    ├── 17_check_tlr_labels.py # check / fix gold article labels
    └── 18_judge_calibration.py  # judge vs human labels (Cohen's kappa)
```

---

## Fine-tuning

### LLM Fine-tuning (QLoRA)

```bash
# Fine-tune Qwen2.5-7B (4-bit QLoRA; default, the base the ablation uses)
PYTHONUTF8=1 python scripts/08_finetune_llm.py

# Low-VRAM smoke run only: Qwen2.5-3B fp16 LoRA (does not match the ablation base)
PYTHONUTF8=1 python scripts/08_finetune_llm.py --backend safe

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

Training data format (`embedding_triplets.jsonl`, written by script 11) — each line:

```json
{"query": "soru metni", "pos": ["ilgili madde metni"], "neg": ["n1", "...", "n7"], "pos_chunk_id": "...", "pos_strategy": "article"}
```

When a training question names an article ("Anayasa madde 1") its chunk is
the positive; hard negatives exclude only the positive's own article.
Questions of any eval set are dropped from every training file (scripts 10, 11,
12b and the train split).

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
| `HALLUCINATION_SAMPLE_SIZE` | `None` | NLI on every answer (int = stratified sample) |
| `LLM_STOP` / `SEED` | see file | Stop sequences / seed sent with every request |
| `PERPLEXITY_ENABLED` | `True` | Skip the 7B HF perplexity pass when False |
| `TLR_USE_LABEL_FIXES` | `True` | Checked gold articles (False = original HF labels) |
| `RESULTS_ROOT` | `results/` | Root of `<eval_set>[_limitN]/<stage>/` outputs |
| `TRUST_REMOTE_CODE` | `False` | Passed to HF `from_pretrained` |

---

## Default Evaluation Set: `turkish_legal_rag`

`scripts/14_eval_all_stages.py` and `run_baseline.py` evaluate on
`turkish_legal_rag` by default: 194 questions with law + article gold labels,
all of which have gold chunks in the corpus.

- **Source:** [`mtntasci/turkish-legal-rag`](https://huggingface.co/datasets/mtntasci/turkish-legal-rag),
  config `qa_benchmark`, split `test` (290 rows).
- **License / attribution:** CC-BY-4.0. Dataset by `mtntasci`; questions derive
  from the Kaggle legal QA data our corpus is built from.
- **Filtering** (`scripts/16_prepare_turkish_legal_rag.py`, 290 -> 195):
  - drop 90 rows whose `source_origin` is not `kaggle_batuhankalem` (templated, low quality)
  - drop 0 rows whose law has no chunks in the corpus (`unknown_law`)
  - drop 5 rows whose question appears in a `qa_train*.jsonl` fine-tuning file (leakage)
  - `madde_no` is normalised to the corpus convention (`"3-"` -> `"3"`)
  - 195 = 178 rows with clean labels + 17 flagged as label conflicts (written to a
    separate `.label_conflicts.jsonl` file and re-evaluated by the next step)
- **Label check** (`scripts/17_check_tlr_labels.py`, `data/tlr_labels.py`):
  about a quarter of the HF article labels point 1–3 articles before the one
  that holds the answer (İş Kanunu "Ara dinlenmesi" is article 68, labelled 67).
  Each label is checked by how much of the gold answer the labelled article
  covers; 43 labels moved to the article the question names or a neighbour
  that covers the answer, 16 of 17 HF-conflict rows were confirmed or corrected
  and re-admitted, 1 stays out -> 194 rows.  `madde_no_hf` keeps the HF label,
  `label_check` records each decision, and `config.TLR_USE_LABEL_FIXES = False`
  restores the HF labels.
- **Headline metrics:** chunk/article-level metrics lead the PRIMARY table when
  at least 50% of queries have gold chunk labels
  (`config.HEADLINE_CHUNK_MIN_LABELED_FRACTION`); otherwise source-hit does.

HMGS is available with `--eval-set hmgs`.  `--eval-set kaggle` uses 300 kaggle
rows taken round-robin over the 240 distinct contexts (their passages stay in
the index; leakage is controlled on the training side).

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
- **Evaluation**: ranx, sacrebleu (BLEU, chrF++), multilingual NLI, LLM judge, paired bootstrap
- **GPU**: Tested on NVIDIA A100 (80GB) and RTX 4070 Super 12GB
