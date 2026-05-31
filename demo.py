import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent / "CENG493_Project"
sys.path.insert(0, str(_PROJECT_DIR))

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turkish Legal RAG — CENG493",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Presentation-aligned result data ──────────────────────────────────────────
HMGS_RESULTS = [
    {
        "Pipeline": "Base RAG",
        "MRR": 0.8433,
        "nDCG@10": 0.6349,
        "Recall@5": 0.0136,
        "Recall@10": 0.0245,
        "Source Hit@5": 0.919,
        "F1": 0.143,
        "ROUGE-L": 0.187,
        "Faithfulness": 0.980,
        "LLM-Judge": 0.2675,
        "Semantic Sim.": 0.4765,
        "Scenario 1": 0.5918,
        "Scenario 2": 0.2433,
        "Scenario 3": 0.6483,
    },
    {
        "Pipeline": "Hybrid BM25",
        "MRR": 0.8074,
        "nDCG@10": 0.5909,
        "Recall@5": 0.0128,
        "Recall@10": 0.0223,
        "Source Hit@5": 0.863,
        "F1": 0.131,
        "ROUGE-L": 0.176,
        "Faithfulness": 0.980,
        "LLM-Judge": 0.2650,
        "Semantic Sim.": 0.4709,
        "Scenario 1": 0.5607,
        "Scenario 2": 0.2332,
        "Scenario 3": 0.6275,
    },
    {
        "Pipeline": "RRF",
        "MRR": 0.8369,
        "nDCG@10": 0.6207,
        "Recall@5": 0.0139,
        "Recall@10": 0.0235,
        "Source Hit@5": 0.925,
        "F1": 0.133,
        "ROUGE-L": 0.181,
        "Faithfulness": 0.980,
        "LLM-Judge": 0.2000,
        "Semantic Sim.": 0.4643,
        "Scenario 1": 0.5612,
        "Scenario 2": 0.2326,
        "Scenario 3": 0.6033,
    },
    {
        "Pipeline": "RRF + Rerank",
        "MRR": 0.8844,
        "nDCG@10": 0.6400,
        "Recall@5": 0.0146,
        "Recall@10": 0.0235,
        "Source Hit@5": 0.938,
        "F1": 0.154,
        "ROUGE-L": 0.203,
        "Faithfulness": 0.987,
        "LLM-Judge": 0.2900,
        "Semantic Sim.": 0.4802,
        "Scenario 1": 0.6172,
        "Scenario 2": 0.2516,
        "Scenario 3": 0.6442,
    },
    {
        "Pipeline": "FT Embedding",
        "MRR": 0.8829,
        "nDCG@10": 0.6730,
        "Recall@5": 0.0152,
        "Recall@10": 0.0251,
        "Source Hit@5": 0.938,
        "F1": 0.161,
        "ROUGE-L": 0.211,
        "Faithfulness": 0.967,
        "LLM-Judge": 0.2400,
        "Semantic Sim.": 0.4845,
        "Scenario 1": 0.6134,
        "Scenario 2": 0.2581,
        "Scenario 3": 0.6150,
    },
    {
        "Pipeline": "Full Optimized",
        "MRR": 0.8829,
        "nDCG@10": 0.6730,
        "Recall@5": 0.0152,
        "Recall@10": 0.0251,
        "Source Hit@5": 0.888,
        "F1": 0.051,
        "ROUGE-L": 0.082,
        "Faithfulness": 0.940,
        "LLM-Judge": 0.1150,
        "Semantic Sim.": 0.4143,
        "Scenario 1": 0.5631,
        "Scenario 2": 0.1599,
        "Scenario 3": 0.6267,
    },
]

QA300_RESULTS = [
    {
        "Pipeline": "FT LLM",
        "MRR": 0.743,
        "nDCG@10": 0.493,
        "Source Hit@5": 0.853,
        "F1": 0.334,
        "ROUGE-L": 0.362,
        "Faithfulness": 0.927,
        "LLM-Judge": 0.830,
        "Coherence": 0.923,
        "Semantic Sim.": 0.679,
        "Scenario 1": 0.547,
        "Scenario 2": 0.456,
        "Scenario 3": 0.798,
    },
    {
        "Pipeline": "FT Embedding",
        "MRR": 0.870,
        "nDCG@10": 0.547,
        "Source Hit@5": 0.937,
        "F1": 0.265,
        "ROUGE-L": 0.281,
        "Faithfulness": 1.000,
        "LLM-Judge": 0.863,
        "Coherence": 0.920,
        "Semantic Sim.": 0.743,
        "Scenario 1": 0.567,
        "Scenario 2": 0.409,
        "Scenario 3": 0.834,
    },
    {
        "Pipeline": "Full Pipeline",
        "MRR": 0.870,
        "nDCG@10": 0.547,
        "Source Hit@5": 0.937,
        "F1": 0.131,
        "ROUGE-L": 0.140,
        "Faithfulness": 0.880,
        "LLM-Judge": 0.803,
        "Coherence": 0.925,
        "Semantic Sim.": 0.680,
        "Scenario 1": 0.518,
        "Scenario 2": 0.296,
        "Scenario 3": 0.838,
    },
]

SAMPLE_QUESTIONS = [
    "Türk Medeni Kanunu'na göre evlilik için asgari yaş kaçtır?",
    "İş Kanunu'na göre haftalık normal çalışma süresi kaç saattir?",
    "Türk Ceza Kanunu'na göre kasten öldürme suçunun cezası nedir?",
    "Kira sözleşmesi hangi koşullarda sona erdirilebilir?",
    "Tüketici hakları kapsamında ayıplı mal iade süresi ne kadardır?",
    "Türk Borçlar Kanunu'na göre haksız fiil sorumluluğunun şartları nelerdir?",
]

def result_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── Pipeline loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline (BGE-M3 + FAISS)…")
def load_pipeline():
    import config
    from retrieval.embedder import Embedder
    from retrieval.bm25_retriever import BM25Index
    from retrieval.reranker import Reranker
    from retrieval.retriever import Retriever
    from generation.rag_pipeline import RAGPipeline

    embedder = Embedder()
    embedder.load_model()

    retriever = Retriever(
        embedder,
        index_path=config.INDEX_DIR / config.INDEX_FILE,
        metadata_path=config.INDEX_DIR / config.METADATA_FILE,
    )
    bm25 = BM25Index()
    bm25.build(retriever.metadata)

    reranker = Reranker()
    reranker.load_model()

    pipeline = RAGPipeline(retriever)
    return pipeline, bm25, reranker


def ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Turkish Legal RAG System")
st.caption(
    "CENG493 Term Project · Retrieval-Augmented Generation for Turkish Legal Question Answering"
)
st.divider()

tab_demo, tab_ablation, tab_arch = st.tabs(
    ["Live Demo", "Ablation Results", "Architecture"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Ask a Legal Question")

        if "demo_question" not in st.session_state:
            st.session_state.demo_question = ""

        def apply_sample_question():
            selected = st.session_state.demo_sample_question
            if selected != "— type your own —":
                st.session_state.demo_question = selected

        sample = st.selectbox(
            "Sample questions:",
            ["— type your own —"] + SAMPLE_QUESTIONS,
            key="demo_sample_question",
            on_change=apply_sample_question,
        )

        question = st.text_area(
            "Question (Turkish)",
            key="demo_question",
            height=110,
            placeholder="Türkçe hukuki sorunuzu buraya yazın…",
        )

        run = st.button("Get Answer", type="primary", width="stretch")

        ollama_ok = ollama_running()
        if not ollama_ok:
            st.warning(
                "Ollama is not running. Start it with `ollama serve` before querying.",
                icon="⚠️",
            )
        else:
            st.success("Ollama is online", icon="✅")

        st.caption(
            "Stage 3 demo: BGE-M3 + FAISS/BM25 RRF retrieval, cross-encoder reranking, "
            "then Qwen 2.5 running locally."
        )

    with col_out:
        if run and question.strip():
            if not ollama_ok:
                st.error("Cannot generate — Ollama is offline.")
            else:
                with st.spinner("Retrieving passages and generating answer…"):
                    try:
                        pipeline, bm25, reranker = load_pipeline()
                        retrieved = pipeline.retriever.batch_rrf_retrieve(
                            [question],
                            bm25,
                            top_k=config.RERANKER_CANDIDATES,
                        )[0]
                        reranked = reranker.rerank(
                            question,
                            retrieved,
                            top_k=config.TOP_K_RETRIEVAL,
                        )
                        context_used, context_chunks = pipeline.assemble_context(reranked)
                        answer = pipeline.generate(question, context_used)

                        st.subheader("Generated Answer")
                        st.info(answer)

                        st.subheader(f"Retrieved Passages ({len(context_chunks)})")
                        for i, chunk in enumerate(context_chunks, 1):
                            source  = chunk.get("source", "Unknown")
                            score   = chunk.get("score", 0.0)
                            text    = chunk.get("text", "")
                            with st.expander(f"{i}. {source}  —  score: {score:.4f}"):
                                st.caption(f"Characters: {len(text)}")
                                st.text_area(
                                    "Full passage",
                                    value=text,
                                    height=320,
                                    label_visibility="collapsed",
                                )
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")
        elif run:
            st.warning("Please enter a question first.")
        else:
            st.markdown(
                """
                #### How it works

                1. Your question is embedded with **BGE-M3** (1024-dim)
                2. **FAISS** retrieves the top-10 most similar legal passages
                3. Passages are assembled into a context window (≤14,000 chars)
                4. **Qwen 2.5** generates a grounded answer with source citations

                Select a sample question or type your own, then click **Get Answer**.
                """
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABLATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ablation:
    hmgs = result_frame(HMGS_RESULTS)
    qa300 = result_frame(QA300_RESULTS)

    st.subheader("Presentation Results")
    st.caption(
        "Updated from `CENG493_Presentation.html`: HMGS benchmark, QA-300 ablation, "
        "and rubric scenario scores."
    )

    sub_hmgs, sub_qa300, sub_scenarios = st.tabs(
        ["HMGS 2025 (161)", "QA-300 Ablation", "Rubric Scenarios"]
    )

    with sub_hmgs:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Best MRR", "0.884", "RRF + Rerank")
        with c2:
            st.metric("Best F1", "16.1%", "FT Embedding")
        with c3:
            st.metric("Best Faithfulness", "98.7%", "RRF + Rerank")
        with c4:
            st.metric("Best Source Hit@5", "93.8%", "RRF + Rerank / FT Emb.")

        ret_fig = go.Figure()
        for metric in ["MRR", "nDCG@10", "Source Hit@5"]:
            ret_fig.add_trace(
                go.Bar(
                    name=metric,
                    x=hmgs["Pipeline"],
                    y=hmgs[metric],
                    text=[f"{v:.3f}" for v in hmgs[metric]],
                    textposition="outside",
                )
            )
        ret_fig.update_layout(
            title="Retrieval and Source Grounding",
            barmode="group",
            height=390,
            yaxis=dict(range=[0, 1.08], title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=70, b=20),
        )
        st.plotly_chart(ret_fig, width="stretch")

        qa_fig = go.Figure()
        for metric in ["F1", "ROUGE-L", "Faithfulness", "LLM-Judge", "Semantic Sim."]:
            qa_fig.add_trace(
                go.Bar(
                    name=metric,
                    x=hmgs["Pipeline"],
                    y=hmgs[metric],
                    text=[f"{v:.3f}" for v in hmgs[metric]],
                    textposition="outside",
                )
            )
        qa_fig.update_layout(
            title="Answer Quality Metrics",
            barmode="group",
            height=390,
            yaxis=dict(range=[0, 1.12], title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=70, b=20),
        )
        st.plotly_chart(qa_fig, width="stretch")

        st.info(
            "HMGS yorumu: FT Embedding retrieval/semantic metrikleri güçlendirdi. "
            "Full Optimized retrieval'ı koruyor; ancak HMGS kısa cevap beklediği için "
            "açıklamalı FT LLM çıktıları F1, ROUGE-L ve LLM-Judge'ı düşürebiliyor."
        )
        st.dataframe(hmgs, width="stretch", hide_index=True)

    with sub_qa300:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Best F1", "33.4%", "FT LLM")
        with c2:
            st.metric("Best MRR", "0.870", "FT Embedding / Full")
        with c3:
            st.metric("Best Scenario 3", "0.838", "Full Pipeline")

        fig = go.Figure()
        for metric in ["MRR", "F1", "ROUGE-L", "Faithfulness", "LLM-Judge", "Semantic Sim."]:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=qa300["Pipeline"],
                    y=qa300[metric],
                    text=[f"{v:.3f}" for v in qa300[metric]],
                    textposition="outside",
                )
            )
        fig.update_layout(
            title="Fine-Tuning Impact on QA-300",
            barmode="group",
            height=420,
            yaxis=dict(range=[0, 1.12], title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=70, b=20),
        )
        st.plotly_chart(fig, width="stretch")
        st.warning(
            "Evaluation paradox: n-gram metrics such as F1/ROUGE drop when the model "
            "answers in a richer explanatory format. Semantic and judge-based metrics "
            "show a different quality picture."
        )
        st.dataframe(qa300, width="stretch", hide_index=True)

    with sub_scenarios:
        st.markdown(
            """
| Scenario | Formula | Evaluation view |
|---|---|---|
| **Scenario 1** | `0.35*MRR + 0.40*F1 + 0.25*Faithfulness` | Gold question + answer + document |
| **Scenario 2** | `0.70*F1 + 0.30*Semantic Similarity` | Gold question + answer |
| **Scenario 3** | `avg(Relevancy, Faithfulness, Coherence)` | No gold data / LLM-judge view |
"""
        )

        fig_hmgs = go.Figure()
        for metric in ["Scenario 1", "Scenario 2", "Scenario 3"]:
            fig_hmgs.add_trace(
                go.Bar(
                    name=metric,
                    x=hmgs["Pipeline"],
                    y=hmgs[metric],
                    text=[f"{v:.3f}" for v in hmgs[metric]],
                    textposition="outside",
                )
            )
        fig_hmgs.update_layout(
            title="HMGS Scenario Scores",
            barmode="group",
            height=380,
            yaxis=dict(range=[0, 0.75], title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=70, b=20),
        )
        st.plotly_chart(fig_hmgs, width="stretch")

        fig_qa = go.Figure()
        for metric in ["Scenario 1", "Scenario 2", "Scenario 3"]:
            fig_qa.add_trace(
                go.Bar(
                    name=metric,
                    x=qa300["Pipeline"],
                    y=qa300[metric],
                    text=[f"{v:.3f}" for v in qa300[metric]],
                    textposition="outside",
                )
            )
        fig_qa.update_layout(
            title="QA-300 Scenario Scores",
            barmode="group",
            height=380,
            yaxis=dict(range=[0, 0.95], title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=70, b=20),
        )
        st.plotly_chart(fig_qa, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.subheader("Pipeline Flow")
        st.code(
            """\
User Query (Turkish)
        │
        ▼
┌───────────────────┐
│  BGE-M3 Embedder  │  1024-dim dense vector
└────────┬──────────┘
         │
         ▼
┌──────────────────────────┐
│    Hybrid Retrieval       │
│  ┌─────────┐ ┌─────────┐ │
│  │  FAISS  │ │  BM25   │ │  Top-10 candidates each
│  └────┬────┘ └────┬────┘ │
│       └─────┬─────┘      │
│         RRF Fusion        │
│              │            │
│   ┌──────────▼─────────┐  │
│   │  Cross-Encoder      │  │  BGE Reranker v2-m3
│   │  Reranker           │  │
│   └──────────┬─────────┘  │
└──────────────┼────────────┘
               │  Top-5 chunks
               ▼
      Context Assembly  ≤14,000 chars
               │
               ▼
┌──────────────────────────┐
│   Qwen 2.5 via Ollama    │  14B / LoRA fine-tuned
└──────────────┬───────────┘
               │
               ▼
      Answer + Source Citations""",
            language="text",
        )

    with col_b:
        st.subheader("5-Stage Ablation")
        st.markdown(
            """
| Stage | What was added |
|-------|----------------|
| **1 — Baseline** | BGE-M3 + RRF retrieval + Qwen 2.5-7B (no fine-tuning) |
| **2 — Emb. FT** | BGE-M3 contrastive fine-tuning on Turkish legal corpus |
| **3 — Reranker FT** | Cross-encoder fine-tuned for legal passage reranking |
| **4 — LLM FT** | Qwen 2.5 LoRA instruction-tuned on legal QA pairs |
| **5 — Full Optimized** | All three fine-tuned components combined |
"""
        )

        st.subheader("Dataset & Corpus")
        st.markdown(
            """
| | |
|--|--|
| **Corpus** | 12 Turkish laws (Anayasa, TMK, TCK, CMK, TBK, İK, …) |
| **Chunking** | 1,400 chars / 180 char overlap |
| **Index** | FAISS IndexFlatIP (~50k chunks) |
| **Eval set** | 300 questions — HMGS 2025 benchmark |
| **Metrics** | Recall@K, MRR, nDCG · F1, ROUGE-L, BLEU, Citation Acc. |
"""
        )

        st.subheader("Tech Stack")
        st.markdown(
            """
- **Embeddings** — `FlagEmbedding` BGE-M3 (dense + sparse + ColBERT)
- **Vector search** — `faiss-cpu` IndexFlatIP
- **Sparse retrieval** — `rank_bm25` + RRF fusion
- **Reranker** — `BAAI/bge-reranker-v2-m3` cross-encoder
- **LLM serving** — Qwen 2.5-14B via `Ollama` (local)
- **Fine-tuning** — `unsloth` LoRA / QLoRA
- **Evaluation** — `rouge_score`, `sacrebleu`, LLM-as-judge
"""
        )
