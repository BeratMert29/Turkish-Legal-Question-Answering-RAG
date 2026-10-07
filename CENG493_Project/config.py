from pathlib import Path

BASE_DIR = Path(__file__).parent

# Chunking
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180
CORPUS_DOC_MIN_CHARS = 180
# Minimum chunk character length; shared by _char_chunk and _article_chunk
MIN_CHUNK_CHARS = 180
ARTICLE_CHUNKING_ENABLED = True  # True → split at MADDE boundaries first

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
    "657 sayılı Devlet Memurları Kanunu":    "Devlet Memurları Kanunu",
    # Present in the corpus index (results/index/metadata.jsonl), so mapped here.
    # VUK is mapped but its HMGS rows are excluded via HMGS_DROPPED_SOURCES.
    "213 sayılı Vergi Usul Kanunu":          "Vergi Usul Kanunu",
    "4982 sayılı Bilgi Edinme Hakkı Kanunu": "Bilgi Edinme Kanunu",
    "Türk Bayrağı Tüzüğü":                   "Türk Bayrağı Tüzüğü",
}
# HMGS rows whose kaynak is excluded from the eval set although the corpus has
# the law: the 5 VUK rows are misattributed (3/5 are really Avukatlik Kanunu /
# HMK questions), so their gold source label is wrong.
HMGS_DROPPED_SOURCES = frozenset({"213 sayılı Vergi Usul Kanunu"})
HMGS_EVAL_EXPECTED = 161  # 240 raw - 49 no corpus - 5 VUK (HMGS_DROPPED_SOURCES) - 25 MC-ref; soft assertion in build_gold_eval_set

# turkish_legal_rag eval set (HF mtntasci/turkish-legal-rag, CC-BY-4.0); built by
# scripts/16_prepare_turkish_legal_rag.py and committed under results/processed_data/.
TLR_PROCESSED_DIR = BASE_DIR.parent / "results" / "processed_data"
TLR_METADATA_PATH = BASE_DIR.parent / "results" / "index" / "metadata.jsonl"
TLR_GOLD_FILE = "qa_turkish_legal_rag.jsonl"
TLR_DATA_PATH = TLR_PROCESSED_DIR / TLR_GOLD_FILE
# HF ``kaynak`` spellings that differ from corpus source names
TLR_SOURCE_ALIASES = {
    "Bilgi Edinme Hakkı Kanunu": "Bilgi Edinme Kanunu",
}

# Eval sets selectable via --eval-set; the default applies to scripts/14 and run_baseline.
EVAL_SET_CHOICES = ["turkish_legal_rag", "hmgs", "kaggle"]
DEFAULT_EVAL_SET = "turkish_legal_rag"
# Chunk-level metrics become the headline when at least this fraction of queries
# has gold chunk labels; otherwise source-hit stays the headline.
HEADLINE_CHUNK_MIN_LABELED_FRACTION = 0.5

# Embedding
EMBEDDING_MODEL = "BAAI/bge-m3"
FINETUNED_EMBEDDING_MODEL = str(BASE_DIR / "models" / "bge-m3-turkish-legal")
HF_PERPLEXITY_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 8  # lower = less VRAM; increase to 32 if you have 12GB+ VRAM

# Retrieval
TOP_K_RETRIEVAL = 10
TOP_K_FOR_GENERATION = 5
CONTEXT_WINDOW_CHARS = 14000

# Re-ranker (Stage 2 retrieval)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATES = 50   # fetch more candidates than TOP_K so reranker has a real pool to reorder
RRF_K = 60                 # RRF smoothing constant

GRAPH_FILE = "graph.json"
GRAPH_HOPS = 1
GRAPH_NEIGHBOR_BUDGET = 3
# When True, up to GRAPH_NEIGHBOR_BUDGET of the TOP_K_FOR_GENERATION context
# slots are reserved for graph neighbours of the top-ranked chunks, so graph
# expansion can change generation.  Retrieval metrics (recall/MRR/nDCG/source
# hit) always use the pre-expansion ranking.
GRAPH_CONTEXT_RESERVE = True

# Direct madde lookup: when True, queries that explicitly reference a law article
# (e.g. "TCK 86. madde") inject that article's chunks via _source_madde_lookup.
# Disabled by default; enable only after verifying graph quality on your index.
DIRECT_MADDE_LOOKUP_ENABLED = False

# LLM (Ollama — free, no API key)
# Ablation one-factor rule: the base-LLM stages and the fine-tuned-LLM stages
# must differ ONLY in the LoRA weights.  LLM_BASE_FOR_ABLATION is therefore the
# Ollama tag of the exact base the LoRA adapter was trained on
# (LORA_BASE_HF_MODEL = Qwen/Qwen2.5-7B-Instruct, see
# results/model_configs/qwen25_lora/adapter_config.json); both use
# LLM_MAX_TOKENS and LLM_NUM_CTX.  Sized for a 12 GB GPU (RTX 4070 Super);
# on a larger GPU qwen2.5:14b / llama3.3:70b are options, but the LoRA must
# then be retrained on the matching base.
LLM_BASE_FOR_ABLATION = "qwen2.5:7b"
LORA_BASE_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_MODEL = LLM_BASE_FOR_ABLATION
LLM_FINETUNED_MODEL = "qwen25-legal-ft"   # created by scripts/13_export_lora_to_ollama.py
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY = "ollama"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512
# Context window for BOTH LLMs.  Baked into the fine-tuned Modelfile; for the
# base model start Ollama with OLLAMA_CONTEXT_LENGTH=8192 (the OpenAI-compatible
# endpoint cannot set num_ctx per request).
LLM_NUM_CTX = 8192
# Stage is marked failed when more than this fraction of generations / judge calls fail
MAX_FAILURE_RATE = 0.2
# trust_remote_code executes code shipped with a model repo.  Qwen2.5 and
# BGE-M3 are natively supported by transformers, so keep False.  Set True only
# for a model that really needs custom modelling code.
TRUST_REMOTE_CODE = False
# Judge differs from the generator (Qwen) to avoid self-evaluation bias.
# llama3.3:70b is an option on a larger GPU.
LLM_JUDGE_MODEL = "llama3.1:8b"

# Evaluation
HALLUCINATION_SAMPLE_SIZE = 150

# NLI model for hallucination analysis (multilingual, covers Turkish).
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# Multilingual sentence-embedding model for answer semantic similarity.
SEMANTIC_SIM_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEMANTIC_SIM_MAX_SEQ_LEN = 512  # encoder window; longer answers are chunked
# Number of predictions sampled for each LLM-judge metric call.  None = judge
# every prediction (needed for tight CIs); an int caps cost (Ollama calls).
LLM_JUDGE_SAMPLE_SIZE = None

# Hallucination stratification thresholds (applied to top-1 retrieval score)
HALLUCINATION_HIT_THRESHOLD = 0.7
HALLUCINATION_PARTIAL_THRESHOLD = 0.4

# BM25 tokenization
BM25_MIN_TOKEN_LENGTH = 2

# Custom corpus / benchmark support
CUSTOM_CORPUS_FILE = "corpus_chunks_custom.jsonl"

# Silver lexical labeling (strategy 3.5) — off by default.
# When True, queries with no article-level label receive up to SILVER_TOP_M
# chunks scored by normalized token overlap with (question + answer), restricted
# to the gold source law and above SILVER_THRESHOLD.  Tagged label_strategy="silver_lexical".
RELEVANCE_SILVER_LEXICAL: bool = False
SILVER_TOP_M: int = 3
SILVER_THRESHOLD: float = 0.10
SUPPORTED_DOC_EXTENSIONS = (".txt", ".pdf")
