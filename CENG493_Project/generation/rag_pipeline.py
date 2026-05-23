import json
from pathlib import Path
import openai
import config

TURKISH_PROMPT = """Sen Türk hukuku alanında uzman bir hukuki asistansın. Görevin, yalnızca aşağıda numaralandırılmış [Kaynak N] bağlamlarını kullanarak soruyu eksiksiz ve doğru biçimde yanıtlamaktır.

ZORUNLU KURALLAR:
1. Yanıtını YALNIZCA verilen [Kaynak N] kaynaklarına dayandır.
2. Hukuki terimleri, kanun adlarını ve madde numaralarını AYNEN koru. Açıklamanı kendi cümlelerinle yap, ancak hukuki kavramları değiştirme veya basitleştirme.
3. İlgili her atıfta kanun adını VE madde numarasını açıkça belirt (örnek: "Türk Medeni Kanunu Madde 997").
4. Birden fazla kaynak ilgiliyse hepsini sentezle ve [Kaynak N] numarasıyla göster.
5. Bağlam soruyu yanıtlamak için yetersizse "Sağlanan bağlam bu soruyu yanıtlamak için yeterli değildir." yaz; asla tahmin yürütme.

YANIT YAPISI:
- İlk cümle: Sorunun doğrudan, kısa yanıtı.
- Devamı: Hukuki dayanak — ilgili kanun adı, madde numarası ve bağlamdan çıkarılan açıklama.
- Gereksiz giriş cümlesi, tekrar veya açıklama ekleme.

Yanıtını yalnızca Türkçe ver."""

SHORT_ANSWER_PROMPT = """Sen Türk hukuku alanında uzman bir hukuki asistansın.

ZORUNLU KURALLAR:
1. Yanıtın yalnızca TEK bir ifade, sayı veya hukuki kavramdan oluşmalıdır — cümle kurma, açıklama yapma, gerekçe gösterme.
2. Hukuki terimleri ve kavramları AYNEN kullan, basitleştirme.
3. Yanıtı öncelikle bağlamdan çıkar. Bağlamda doğrudan cevap yoksa, en ilgili hukuki terimi yaz.
4. Yalnızca bağlam tamamen ilgisizse şunu yaz: Bilgi yok
5. Fazladan kelime, noktalama veya açıklama ekleme.
6. Yanıtının hemen ardına, kullandığın kaynağın numarasını şu formatta ekle: [Kaynak 1]

Yanıtını yalnızca Türkçe ver."""

class ChunkExpander:
    """Merge chunk text with adjacent chunks (same chunk_id prefix) from metadata."""

    def __init__(self, metadata_path: str | Path):
        metadata_path = Path(metadata_path)
        self._by_prefix: dict[str, list[tuple[int, dict]]] = {}
        with open(metadata_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                chunk_id: str = meta.get("chunk_id", "")
                last_sep = chunk_id.rfind("_")
                if last_sep == -1:
                    continue
                prefix = chunk_id[:last_sep]
                try:
                    idx = int(chunk_id[last_sep + 1:])
                except ValueError:
                    continue
                self._by_prefix.setdefault(prefix, []).append((idx, meta))
        for prefix in self._by_prefix:
            self._by_prefix[prefix].sort(key=lambda t: t[0])

    def _lookup(self, chunk_id: str) -> tuple[str, int] | tuple[None, None]:
        last_sep = chunk_id.rfind("_")
        if last_sep == -1:
            return None, None
        prefix = chunk_id[:last_sep]
        try:
            idx = int(chunk_id[last_sep + 1:])
        except ValueError:
            return None, None
        return prefix, idx

    def expand(self, chunk: dict, window: int = 1) -> str:
        """Concatenate neighboring chunk texts within window."""
        chunk_id = chunk.get("chunk_id", "")
        prefix, idx = self._lookup(chunk_id)
        if prefix is None or prefix not in self._by_prefix:
            return chunk.get("text", "")

        sorted_chunks = self._by_prefix[prefix]
        position = next((i for i, (ci, _) in enumerate(sorted_chunks) if ci == idx), None)
        if position is None:
            return chunk.get("text", "")

        start = max(0, position - window)
        end = min(len(sorted_chunks), position + window + 1)
        texts = [meta["text"] for _, meta in sorted_chunks[start:end]]
        if len(texts) == 1:
            return texts[0]

        overlap = config.CHUNK_OVERLAP
        merged = texts[0]
        for t in texts[1:]:
            if overlap and merged.endswith(t[:overlap]):
                merged = merged + t[overlap:]
            elif merged.endswith(t[:min(overlap, len(t))]):
                merged = merged + t[min(overlap, len(t)):]
            else:
                merged = merged + "\n" + t
        return merged


class RAGPipeline:
    def __init__(self, retriever,
                 model: str = config.LLM_MODEL,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS,
                 top_k_for_generation: int = config.TOP_K_FOR_GENERATION,
                 context_window_chars: int = config.CONTEXT_WINDOW_CHARS,
                 short_answer_mode: bool = False,
                 chunk_expander: "ChunkExpander | None" = None):
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = SHORT_ANSWER_PROMPT if short_answer_mode else TURKISH_PROMPT
        self._effective_max_tokens = config.LLM_SHORT_ANSWER_MAX_TOKENS if short_answer_mode else max_tokens
        self.top_k_for_generation = top_k_for_generation
        self.context_window_chars = context_window_chars
        self._chunk_expander = chunk_expander
        self._client = openai.OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )

    def get_llm_client(self) -> openai.OpenAI:
        """Ollama OpenAI-compatible client."""
        return self._client

    def assemble_context(self, chunks: list) -> tuple[str, list]:
        """Numbered sources string and chunks included (respects context_window_chars)."""
        selected = chunks[:self.top_k_for_generation]
        parts = []
        included = []
        running_len = 0
        for i, chunk in enumerate(selected):
            text = (
                self._chunk_expander.expand(chunk)
                if self._chunk_expander is not None
                else chunk["text"]
            )
            part = f"[Kaynak {i+1}] ({chunk['source']})\n{text}\n\n"
            if i > 0 and running_len + len(part) > self.context_window_chars:
                break
            parts.append(part)
            included.append(chunk)
            running_len += len(part)
        context = "".join(parts)
        return context, included

    def generate(self, question: str, context: str) -> str:
        """Single chat completion."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self._effective_max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"},
            ],
        )
        if not response.choices:
            raise ValueError("LLM returned empty choices list")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response")
        cut = len(content)
        for marker in ["Soru:", "Bağlam:", "Question:", "\nQ:"]:
            idx = content.find(marker)
            if idx != -1 and idx >= 10:
                cut = min(cut, idx)
        content = content[:cut]
        return content.strip()

    def run(self, question: str, top_k_retrieval: int = config.TOP_K_RETRIEVAL) -> dict:
        """Retrieve, assemble context, generate."""
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k_retrieval)
        context_used, context_chunks = self.assemble_context(retrieved_chunks)
        answer = self.generate(question, context_used)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": context_chunks,
            "context_used": context_used,
        }
