from pathlib import Path

BASE_DIR = Path(__file__).parent

# Chunking
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180
CORPUS_DOC_MIN_CHARS = 180

# Data
QA_EVAL_EXPECTED = 300
QA_GOLD_FILE = "qa_eval.jsonl"
RAW_DATA_PATH = BASE_DIR.parent / "combined_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data/processed"
INDEX_DIR = BASE_DIR / "index"
INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
RESULTS_DIR = BASE_DIR / "results/stage1"
RESULTS_DIR_BASE     = BASE_DIR / "results" / "stage_base"
RESULTS_DIR_EMB_FT   = BASE_DIR / "results" / "stage_emb_finetuned"
RESULTS_DIR_RERANK   = BASE_DIR / "results" / "stage_reranker"
RESULTS_DIR_LLM_FT   = BASE_DIR / "results" / "stage_llm_finetuned"
RESULTS_DIR_FULL     = BASE_DIR / "results" / "stage_full_optimized"

# HMGS gold test set
HMGS_DATA_PATH = BASE_DIR.parent / "hmgs_2025_240_only_correct_answers_v2.csv"
HMGS_GOLD_FILE = "qa_hmgs.jsonl"
LLM_SHORT_ANSWER_MAX_TOKENS = 64

# HMGS kaynak -> corpus source name mapping (only laws present in corpus)
HMGS_SOURCE_MAP = {
    # Original corpus laws
    "1982 Anayasası":                     "Türkiye Cumhuriyeti Anayasası",
    "4721 sayılı Türk Medeni Kanunu":     "Türk Medeni Kanunu",
    "5237 sayılı Türk Ceza Kanunu":       "Türk Ceza Kanunu",
    "5271 sayılı Ceza Muhakemesi Kanunu": "Ceza Muhakemesi Kanunu",
    "6098 sayılı Türk Borçlar Kanunu":    "Türk Borçlar Kanunu",
    "4857 sayılı İş Kanunu":              "Türkiye Cumhuriyeti İş Kanunu",
    # Supplementary laws (extra_laws.jsonl)
    "6100 sayılı Hukuk Muhakemeleri Kanunu": "Hukuk Muhakemeleri Kanunu",
    "6102 sayılı Türk Ticaret Kanunu":       "Türk Ticaret Kanunu",
    "2577 sayılı İdari Yargılama Usulü Kanunu": "İdari Yargılama Usulü Kanunu",
    "2004 sayılı İcra ve İflas Kanunu":      "İcra ve İflas Kanunu",
    "213 sayılı Vergi Usul Kanunu":          "Vergi Usul Kanunu",
    "657 sayılı Devlet Memurları Kanunu":    "Devlet Memurları Kanunu",
}
HMGS_EVAL_EXPECTED = 161  # 240 raw - 49 no corpus - 5 VUK (misattributed) - 25 MC-ref; enforced as soft assertion in build_gold_eval_set

# Embedding
EMBEDDING_MODEL = "BAAI/bge-m3"
FINETUNED_EMBEDDING_MODEL = str(BASE_DIR / "models" / "bge-m3-turkish-legal")
HF_PERPLEXITY_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32

# Retrieval
TOP_K_RETRIEVAL = 10
TOP_K_FOR_GENERATION = 5
CONTEXT_WINDOW_CHARS = 14000

# Re-ranker (Stage 2 retrieval)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATES = 10   # initial dense/RRF pool before cross-encoder re-ranking
RRF_K = 60                 # RRF smoothing constant

# LLM (Ollama — free, no API key)
LLM_MODEL = "qwen2.5:14b"
LLM_FINETUNED_MODEL = "qwen25-legal-ft"   # created by scripts/13_export_lora_to_ollama.py
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY = "ollama"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512
LLM_FINETUNED_MAX_TOKENS = 256  # shorter cap for fine-tuned model to reduce runaway generation

KAGGLE_MIN_SCORE = 6

# Evaluation
HALLUCINATION_SAMPLE_SIZE = 150

# Hallucination stratification thresholds (applied to top-1 retrieval score)
HALLUCINATION_HIT_THRESHOLD = 0.7
HALLUCINATION_PARTIAL_THRESHOLD = 0.4

# BM25 tokenization
BM25_MIN_TOKEN_LENGTH = 2

# Oracle relevance (scripts/03_evaluate_retrieval.py)
TOP_K_ORACLE = 5

# Custom corpus / benchmark support
CUSTOM_CORPUS_FILE = "corpus_chunks_custom.jsonl"
SUPPORTED_DOC_EXTENSIONS = (".txt", ".pdf")
