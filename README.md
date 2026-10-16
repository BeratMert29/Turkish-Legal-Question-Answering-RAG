# Turkish Legal QA with RAG

A retrieval-augmented generation system for Turkish legal questions, built around one idea:
**every component is measured, not assumed.** Hybrid retrieval, reranking, embedding fine-tuning
and LLM fine-tuning are added one factor at a time and compared on the same questions with
paired bootstrap confidence intervals.

Everything runs locally (RTX 4070 Super, 12 GB): Ollama for generation, no API keys.

> Originally built for CENG493 (Information Retrieval); since extended with an evaluation
> framework, article-level gold labels and a reproducible ablation pipeline.

---

## What makes this different from a typical RAG demo

| Typical demo | This project |
|---|---|
| Chunk PDFs, embed, ask an LLM | Article-aware chunking of Turkish statutes (law + article ids preserved) |
| "It seems to work" | 5-stage ablation, one factor per step, paired bootstrap CIs, Holm-corrected p-values |
| Accuracy by eye | Retrieval (Hit@k, MRR, nDCG at article level), NLI faithfulness, citation precision/recall, LLM judge calibrated against human labels (Cohen's kappa) |
| Trust the benchmark labels | Gold labels audited: about a quarter of the public labels pointed 1–3 articles off; 43 corrected and documented |
| Fine-tune and report F1 | Found and diagnosed a citation collapse caused by fine-tuning (see below) |

### A finding worth reading: fine-tuning improved F1 and destroyed citations

QLoRA fine-tuning of Qwen2.5-7B raised answer metrics while citation accuracy collapsed
(F1 0.244 → 0.306, citation accuracy 0.487 → 0.077 in that run). Root cause: the SFT data had
no context blocks and zero `[Kaynak N]` markers, so the model learned never to cite, while
inference always injects sources and a citation policy. Full analysis and fix paths:
[`CENG493_Project/docs/CITATION_FIX.md`](CENG493_Project/docs/CITATION_FIX.md).

---

## Pipeline

```
question
   │
   ├──► BM25 (sparse) ─────┐
   │                        ├─► RRF fusion ─► cross-encoder rerank ─► (+ neighbouring articles)
   └──► BGE-M3 + FAISS ────┘                      BGE-reranker-v2-m3          graph expansion
                                                          │
                                                          ▼
                                   Qwen2.5-7B (base or QLoRA) via Ollama, [Kaynak N] citations
                                                          │
                                                          ▼
                       answer + cited articles ─► NLI faithfulness / citation check / LLM judge
```

### Ablation stages

| Stage | Retrieval | Generator | What it isolates |
|---|---|---|---|
| `base` | BGE-M3 dense | Qwen2.5-7B | Baseline |
| `rrf` / `rrf_rerank` | BM25 + dense (RRF), + cross-encoder | Qwen2.5-7B | Hybrid fusion, then reranking |
| `graph` | `rrf_rerank` + adjacent articles | Qwen2.5-7B | Context expansion |
| `emb_ft` | Fine-tuned BGE-M3 + rerank | Qwen2.5-7B | Embedding fine-tuning |
| `llm_ft` | Dense | Qwen2.5-7B + LoRA | LLM fine-tuning |
| `full` | Fine-tuned embedder + rerank | Qwen2.5-7B + LoRA | Everything combined |

Base and fine-tuned LLM share context window, output length, stop sequences and seed, so the
only difference is the LoRA weights.

### Published models

- Embedding: [`Whis29/bge-m3-turkish-legal`](https://huggingface.co/Whis29/bge-m3-turkish-legal)
- LLM (GGUF + Modelfile): [`Whis29/qwen25-legal-ft`](https://huggingface.co/Whis29/qwen25-legal-ft)

---

## Results

Eval set: `turkish_legal_rag`, 194 questions with law + article gold labels (label-audited).
Generator Qwen2.5-7B, judge `llama3.1:8b`, seed 42, one run of commit `b4b5432` plus the
local config change noted below. Per-stage JSON (with provenance) is written to
`results/turkish_legal_rag/`; it is not committed.

| Stage | Article Hit@5 | Article MRR | Token F1 | Citation precision (model's own) | NLI faithfulness | Gold-claim recall | LLM judge |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` (dense) | 0.928 | 0.833 | 0.507 | 0.773 | 0.661 | 0.626 | 0.987 |
| `hybrid` | 0.918 | 0.858 | 0.484 | 0.798 | 0.661 | 0.594 | 0.982 |
| `rrf` | 0.923 | 0.844 | 0.485 | 0.786 | 0.638 | 0.599 | 0.985 |
| `rrf_rerank` | 0.933 | 0.903 | 0.518 | 0.815 | 0.645 | 0.659 | 0.977 |
| `graph` | 0.933 | 0.903 | 0.526 | 0.812 | 0.631 | 0.651 | 0.987 |
| `emb_ft` | 0.933 | 0.904 | 0.498 | 0.774 | 0.667 | 0.649 | 0.985 |
| `llm_ft` | 0.928 | 0.833 | 0.463 | 0.682 | 0.758 | 0.451 | 0.982 |
| `full` | 0.933 | 0.904 | 0.473 | 0.786 | 0.760 | 0.444 | 0.990 |

### What the paired comparisons say (95% bootstrap CI, Holm-corrected)

- **Reranking is the only retrieval change that clearly helps:** article MRR +0.060
  (CI +0.032 to +0.090, significant). BM25 fusion alone (`hybrid`, `rrf`) did not
  significantly change anything.
- **Embedding fine-tuning added nothing on top of the reranker** (`rrf_rerank` → `emb_ft`:
  MRR +0.001). The reranker already removes most of what a better embedder would fix.
- **Graph expansion did not change retrieval or answer quality** significantly.
- **LLM fine-tuning is a trade-off, not a win:**
  - NLI faithfulness +0.097 (significant): answers stay closer to the context.
  - Gold-claim recall −0.174 (significant): answers cover less of the reference answer.
  - Token F1 −0.044, not significant after Holm correction.
  - The model's own `[Kaynak N]` citations dropped from 39% to 6% of answers, which is the
    citation collapse described above; the pipeline's citation injection (≈99%) hides it
    in the headline numbers, so read the "model's own" column.
- Citation precision is computed only over answers that cite at all, so for `llm_ft` and
  `full` (6–7% of answers cite) it rests on very few answers and is not comparable.
- **The LLM judge scores everything ≈0.98–0.99** and does not separate stages. It is not
  informative until calibrated against human labels (`scripts/18_judge_calibration.py`).

### Caveats

- **Quantization mismatch:** the base generator is `qwen2.5:7b` at Q4_K_M, the fine-tuned
  model was exported at Q8_0. `base → llm_ft` therefore mixes LoRA and quantization effects.
- `PERPLEXITY_ENABLED = False` for this run: the 4-bit HF perplexity pass does not fit next
  to Ollama's resident models on a 12 GB GPU.
- **The eval set is easy.** On average 83% of the gold-answer tokens appear verbatim in the
  gold article (62% of questions: at least 80%), a third of the questions name the law, and
  the corpus is built from the same data the questions come from. Absolute scores are
  therefore high; the harder bar-exam style set (`--eval-set hmgs`) was not run here.
- Single run, single seed; CIs reflect question sampling only.

Reproduce: see [Quick start](#quick-start). Paired stage differences with CIs are in
`results/turkish_legal_rag/ablation_summary.json`.

---

## Quick start

Run everything from `CENG493_Project/`.

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Local LLM (one model resident at a time on a 12 GB GPU)
OLLAMA_MAX_LOADED_MODELS=1 ollama serve        # terminal A
ollama pull qwen2.5:7b                          # generator
ollama pull llama3.1:8b                         # judge (skip with --no-judge)

# 3. Smoke test: 10 questions, two stages
PYTHONUTF8=1 python scripts/14_eval_all_stages.py --stages base rrf_rerank --limit 10

# 4. Full ablation (all stages whose prerequisites exist)
PYTHONUTF8=1 python scripts/14_eval_all_stages.py
```

Fine-tuned stages (`emb_ft`, `llm_ft`, `full`) need the two published models above, or your own
via scripts `08`/`12` (training) and `13` (export LoRA to Ollama).

Evaluate on **your own** documents and questions without touching code:
[`CENG493_Project/docs/REFERENCE.md`](CENG493_Project/docs/REFERENCE.md#custom-data-evaluation-for-evaluators).

---

## Evaluation methodology (short)

- **Retrieval:** article-level Hit@5/10, MRR, nDCG@10 over (law, article) ids, independent of
  chunk size; chunk- and source-level metrics alongside.
- **Faithfulness:** multilingual NLI (mDeBERTa) — share of answer sentences entailed by the
  context the generator actually saw.
- **Citations:** the model's own `[Kaynak N]` markers checked against the gold article
  (precision / recall / invalid rate), next to a random-citation baseline.
- **LLM judge:** `llama3.1:8b`, 0 / 0.5 / 1 rubric, failures counted; calibration against human
  labels via `scripts/18_judge_calibration.py`.
- **Statistics:** 95% bootstrap CIs for every metric; paired bootstrap for stage differences.
- **Provenance:** every stage records git commit, seed, eval-file hash, corpus fingerprint and
  model digests.

Details and every metric key: [`REFERENCE.md`](CENG493_Project/docs/REFERENCE.md#evaluation-metrics).

---

## Repository layout

```
CENG493_Project/
├── config.py            all hyperparameters and paths
├── data/                article-aware chunking, gold labels, label audit, corpus loader
├── retrieval/           BM25, dense (FAISS), RRF fusion, reranker, article graph
├── generation/          context assembly -> Ollama
├── evaluation/          retrieval, QA, citation, NLI, judge, perplexity, statistics
├── pipeline/            ablation stages, reports, paired comparisons
├── scripts/             00–18: data prep, indexing, fine-tuning, evaluation
├── tests/               unit tests (metrics, chunking, fusion, labels, ...)
└── docs/                REFERENCE.md, CITATION_FIX.md, fine-tuning guides
demo.py                  Streamlit demo (streamlit run demo.py)
```

Tests and lint run in CI (`.github/workflows/tests.yml`): `pytest` from `CENG493_Project/`,
`ruff check CENG493_Project demo.py` from the repo root.

---

## Data and attribution

- Evaluation set: [`mtntasci/turkish-legal-rag`](https://huggingface.co/datasets/mtntasci/turkish-legal-rag)
  (`qa_benchmark`, CC-BY-4.0), filtered and label-audited here.
- Corpus: Turkish statutes built from the Kaggle legal QA data plus supplementary laws.
- Course-provided benchmark data (`Datasets_Ceng493_legal_rag/`) is not redistributed.

## Limitations

- Single-jurisdiction, statute-only corpus; no case law.
- The LLM judge is an 8B local model; it is calibrated, not authoritative.
- Not legal advice.
