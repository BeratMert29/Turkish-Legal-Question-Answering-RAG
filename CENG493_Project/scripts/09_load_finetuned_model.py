import os, sys
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from generation.rag_pipeline import TURKISH_PROMPT, SHORT_ANSWER_PROMPT

HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_DIR = config.BASE_DIR / "models" / "qwen25_lora"


def print_export_steps(adapter_dir: Path) -> None:
    merged_dir = adapter_dir.parent / "qwen25_merged"
    gguf_path = adapter_dir.parent / "qwen25_merged.gguf"
    modelfile_path = adapter_dir.parent / "Modelfile"
    ollama_model_name = "qwen25-legal-ft"

    print("=" * 70)
    print("ADIM 1 — LoRA adaptörünü temel modelle birleştir (Python)")
    print("=" * 70)
    print(f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "{HF_MODEL_ID}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code={config.TRUST_REMOTE_CODE},
)
tokenizer = AutoTokenizer.from_pretrained("{HF_MODEL_ID}", trust_remote_code={config.TRUST_REMOTE_CODE})

model = PeftModel.from_pretrained(base_model, "{adapter_dir}")
model = model.merge_and_unload()

model.save_pretrained("{merged_dir}")
tokenizer.save_pretrained("{merged_dir}")
print("Birlestirilmis model kaydedildi:", "{merged_dir}")
""")

    print("=" * 70)
    print("ADIM 2 — Birleştirilmiş modeli GGUF formatına dönüştür (llama.cpp)")
    print("=" * 70)
    print(f"""
# llama.cpp reposunu klonla veya mevcut kurulumu kullan:
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp && pip install -r requirements.txt

python llama.cpp/convert_hf_to_gguf.py \\
    {merged_dir} \\
    --outfile {gguf_path} \\
    --outtype q4_k_m
""")

    print("=" * 70)
    print("ADIM 3 — Ollama Modelfile oluştur")
    print("=" * 70)
    print(f"""
# {modelfile_path} dosyasını oluştur:

FROM {gguf_path}
PARAMETER temperature 0.0
PARAMETER num_predict {config.LLM_MAX_TOKENS}
SYSTEM "Sen Türk hukuku alanında uzman bir hukuki asistansın."
""")

    print("=" * 70)
    print("ADIM 4 — Ollama'ya modeli kaydet")
    print("=" * 70)
    print(f"""
ollama create {ollama_model_name} -f {modelfile_path}
ollama list
""")

    print("=" * 70)
    print("ADIM 5 — config.py içindeki LLM_MODEL değerini güncelle")
    print("=" * 70)
    print(f"""
# {config.BASE_DIR / 'config.py'} dosyasında şu satırı değiştir:

# Eski:
LLM_MODEL = "{config.LLM_MODEL}"

# Yeni:
LLM_MODEL = "{ollama_model_name}"
""")
    print("=" * 70)


class FinetunedRAGPipeline:
    def __init__(self, retriever, adapter_dir=None, short_answer_mode=False):
        self.retriever = retriever
        self.adapter_dir = Path(adapter_dir) if adapter_dir else DEFAULT_ADAPTER_DIR
        self.short_answer_mode = short_answer_mode
        self._system_prompt = SHORT_ANSWER_PROMPT if short_answer_mode else TURKISH_PROMPT
        self._effective_max_tokens = (
            config.LLM_SHORT_ANSWER_MAX_TOKENS if short_answer_mode else config.LLM_MAX_TOKENS
        )
        self.top_k_for_generation = config.TOP_K_FOR_GENERATION
        self.context_window_chars = config.CONTEXT_WINDOW_CHARS
        self.model = None
        self.tokenizer = None

    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID, trust_remote_code=config.TRUST_REMOTE_CODE
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )

        self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
        self.model.eval()

    def generate(self, question: str, context: str) -> str:
        import torch

        if self.model is None or self.tokenizer is None:
            self.load_model()

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self._effective_max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:].clone()
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Free intermediate tensors to prevent GPU memory accumulation during batch eval
        del inputs, output_ids, generated
        torch.cuda.empty_cache()
        return result

    def _inject_citations(self, answer: str, chunks: list) -> str:
        """Append [Kaynak N] markers to sentences that have significant token overlap
        with the corresponding retrieved chunk.

        Strategy:
        - Split answer into sentences on common sentence-ending punctuation.
        - For each sentence, compute the overlap coefficient against each chunk's text.
        - Attach [Kaynak N] to the sentence that has the highest overlap for that chunk,
          but only when the score exceeds CITATION_THRESHOLD.
        - Each chunk is cited at most once (the best-matching sentence wins).
        """
        import re

        CITATION_THRESHOLD = 0.15

        def tokenize(text: str) -> set:
            # Lowercase, split on whitespace and punctuation, discard empty tokens
            return set(t.lower() for t in re.split(r"[\s\.,;:!?()\[\]{}'\"]+", text) if t)

        def overlap_coefficient(set_a: set, set_b: set) -> float:
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / min(len(set_a), len(set_b))

        # Split into sentences while keeping the delimiter attached
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        raw_sentences = sentence_pattern.split(answer)
        # Filter out empty entries that can arise from edge cases
        sentences = [s for s in raw_sentences if s.strip()]

        if not sentences or not chunks:
            return answer

        # Tokenise every sentence once
        sent_tokens = [tokenize(s) for s in sentences]

        # For each chunk, find the best-scoring sentence index and score
        # pending_citations: list of (sentence_index, chunk_1based_label, score)
        pending_citations = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_tokens = tokenize(chunk.get("text", ""))
            best_score = 0.0
            best_sent_idx = -1
            for sent_idx, stoks in enumerate(sent_tokens):
                score = overlap_coefficient(stoks, chunk_tokens)
                if score > best_score:
                    best_score = score
                    best_sent_idx = sent_idx
            if best_score >= CITATION_THRESHOLD and best_sent_idx >= 0:
                pending_citations.append((best_sent_idx, chunk_idx + 1, best_score))

        if not pending_citations:
            return answer

        # Group by sentence; for each sentence collect the citation labels to append
        from collections import defaultdict
        sent_to_labels: dict = defaultdict(list)
        for sent_idx, label, _score in pending_citations:
            sent_to_labels[sent_idx].append(label)

        # Rebuild the answer, appending citation tags after their target sentence
        result_parts = []
        for i, sentence in enumerate(sentences):
            if i in sent_to_labels:
                tags = " ".join(f"[Kaynak {lbl}]" for lbl in sorted(sent_to_labels[i]))
                result_parts.append(f"{sentence} {tags}")
            else:
                result_parts.append(sentence)

        return " ".join(result_parts)

    def assemble_context(self, chunks: list) -> tuple:
        selected = chunks[:self.top_k_for_generation]
        parts = []
        included = []
        running_len = 0
        for i, chunk in enumerate(selected):
            part = f"[Kaynak {i+1}] ({chunk['source']})\n{chunk['text']}\n\n"
            if running_len + len(part) > self.context_window_chars:
                break
            parts.append(part)
            included.append(chunk)
            running_len += len(part)
        context = "".join(parts)
        return context, included

    def run(self, question: str, top_k_retrieval: int = config.TOP_K_RETRIEVAL) -> dict:
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k_retrieval)
        context_used, context_chunks = self.assemble_context(retrieved_chunks)
        answer = self.generate(question, context_used)
        answer = self._inject_citations(answer, context_chunks)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": context_chunks,
            "context_used": context_used,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tuned LoRA adapter integration helper for Turkish legal RAG"
    )
    p.add_argument(
        "--show-export-steps",
        action="store_true",
        help="Print step-by-step instructions to export the LoRA adapter to Ollama.",
    )
    p.add_argument(
        "--demo",
        metavar="QUESTION",
        type=str,
        default=None,
        help="Run the fine-tuned RAG pipeline on QUESTION and print the result.",
    )
    p.add_argument(
        "--adapter-dir",
        metavar="PATH",
        type=str,
        default=None,
        help=f"Path to the LoRA adapter directory (default: {DEFAULT_ADAPTER_DIR}).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else DEFAULT_ADAPTER_DIR

    if args.show_export_steps:
        print_export_steps(adapter_dir)
        return

    if args.demo:
        from retrieval.embedder import Embedder
        from retrieval.retriever import Retriever

        print(f"Adapter dir : {adapter_dir}")
        print(f"Question    : {args.demo}")
        print()

        embedder = Embedder()
        embedder.load_model()
        retriever = Retriever(embedder,
                              index_path=config.INDEX_DIR / config.INDEX_FILE,
                              metadata_path=config.INDEX_DIR / config.METADATA_FILE)

        pipeline = FinetunedRAGPipeline(
            retriever=retriever,
            adapter_dir=adapter_dir,
            short_answer_mode=False,
        )

        print("Loading fine-tuned model ...")
        pipeline.load_model()
        print("Model loaded.\n")

        result = pipeline.run(args.demo)

        print("Retrieved chunks:")
        for i, chunk in enumerate(result["retrieved_chunks"]):
            score = chunk.get("score", "N/A")
            print(f"  [{i+1}] {chunk['source']}  score={score:.4f}" if isinstance(score, float) else f"  [{i+1}] {chunk['source']}  score={score}")
            print(f"       {chunk['text'][:120].strip()} ...")
        print()
        print("Answer:")
        print(result["answer"])
        return

    build_arg_parser().print_help()


if __name__ == "__main__":
    main()
