# Evaluator Note (Instructor / Grader)

How to run **custom document + benchmark** evaluation for CENG493 Term Project (Moodle items 3–4).

All commands must be run from the **`CENG493_Project/`** directory.

---

## Prerequisites

```bash
pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama serve
```

Ollama: https://ollama.com/download

---

## Fine-Tuned Models (Optional — required for emb_ft, llm_ft, full stages)

To run all pipeline stages including fine-tuned components, download the pre-trained models:

### 1. Fine-Tuned Embedding Model (BGE-M3 Turkish Legal)

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Whis29/bge-m3-turkish-legal', local_dir='models/bge-m3-turkish-legal')
print('Embedding model ready.')
"
```

### 2. Fine-Tuned LLM (qwen25-legal-ft) — register in Ollama

```bash
# Download GGUF and Modelfile
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Whis29/qwen25-legal-ft', 'qwen25-legal-ft.gguf', local_dir='models/')
hf_hub_download('Whis29/qwen25-legal-ft', 'Modelfile', local_dir='models/')
print('LLM downloaded.')
"

# Register in Ollama
ollama create qwen25-legal-ft -f models/Modelfile
```

After these steps, **all 6 stages** (including `emb_ft`, `llm_ft`, `full`) will be available.

---

## Full Evaluation (All Stages)

**Required:** use **`--corpus` AND `--eval-data` together.**

Place the dataset folder (`Datasets_Ceng493_legal_rag`) next to the `CENG493_Project` folder after cloning. A placeholder folder already exists in the repository to indicate the correct location.

### Linux / macOS

```bash
cd CENG493_Project

PYTHONUTF8=1 python scripts/14_eval_all_stages.py \
  --corpus ../Datasets_Ceng493_legal_rag/corpus.jsonl \
  --eval-data ../Datasets_Ceng493_legal_rag/gold_benchmark.json \
  --stages base,rrf_rerank,emb_ft,llm_ft,full
```

### Windows (PowerShell)

```powershell
cd CENG493_Project
$env:PYTHONUTF8="1"

python scripts/14_eval_all_stages.py `
  --corpus "..\Datasets_Ceng493_legal_rag\corpus.jsonl" `
  --eval-data "..\Datasets_Ceng493_legal_rag\gold_benchmark.json" `
  --stages base,rrf_rerank,emb_ft,llm_ft,full
```

**Without fine-tuned models** (base stages only — always works):

```powershell
python scripts/14_eval_all_stages.py `
  --corpus "..\Datasets_Ceng493_legal_rag\corpus.jsonl" `
  --eval-data "..\Datasets_Ceng493_legal_rag\gold_benchmark.json" `
  --stages base,rrf_rerank
```

---

## Reference Dataset Format

See `Datasets_Ceng493_legal_rag/` in this repository:

| File | Role |
|------|------|
| `corpus.jsonl` | Pre-chunked document collection (7,579 chunks) |
| `gold_benchmark.json` | 240 questions with gold answers + chunk IDs |
| `rag_eval.json` | 1,000 questions (alternative benchmark) |

Example with the bundled reference folder (from repo root):

```powershell
cd CENG493_Project
$env:PYTHONUTF8="1"

python scripts/14_eval_all_stages.py `
  --corpus "..\Datasets_Ceng493_legal_rag\corpus.jsonl" `
  --eval-data "..\Datasets_Ceng493_legal_rag\gold_benchmark.json" `
  --stages base,rrf_rerank,emb_ft,llm_ft,full
```

---

## Quick Smoke Test (5 questions)

```bash
cd CENG493_Project

PYTHONUTF8=1 python scripts/14_eval_all_stages.py \
  --corpus /path/to/corpus.jsonl \
  --eval-data /path/to/gold_benchmark.json \
  --stages base \
  --limit 5
```

---

## Output

Results are written under:

```
CENG493_Project/results/
├── ablation_summary.json
├── stage_base/
├── stage_reranker/          # rrf_rerank
├── stage_emb_finetuned/     # emb_ft
├── stage_llm_finetuned/     # llm_ft
└── stage_full_optimized/    # full
```

Each stage folder contains `baseline_metrics.json` (retrieval, QA, faithfulness, rubric scenario scores).

---

## Important

1. **`--corpus` and `--eval-data` must be used together.** If only `--eval-data` is given, the system falls back to the default training corpus instead of your documents.
2. Use **`scripts/14_eval_all_stages.py` only** for custom evaluation. Do **not** use `demo.py`, `scripts/03_evaluate_retrieval.py`, or `scripts/04_generate_answers.py` for instructor-provided data.
3. Alternative corpus input: **`--docs-path /path/to/folder`** with `.txt` or `.pdf` files (requires `pip install pypdf` for PDF). Mutually exclusive with `--corpus`.
4. Fine-tuned stages (`emb_ft`, `llm_ft`, `full`) require the models downloaded in the **Fine-Tuned Models** section above. If missing, those stages are skipped automatically and a warning is printed.

---

## Accepted File Formats

**Corpus (`--corpus`):** JSONL, one chunk per line. Native schema or evaluator schema (`id` + `metadata.chunk_id`).

**Benchmark (`--eval-data`):** JSON array.

- **gold_benchmark.json:** `question_id`, `question`, `verified_answer`, `gold_sources[].source_id`
- **rag_eval.json:** `query_id`, `query`, `gold_answer_extract`, `gold_chunk_ids`

See `README.md` → *Custom Data Evaluation (for Evaluators)* for examples.
