"""RAG with embeddings (meaning search) and automatic TF-IDF fallback (keyword search).

items = [{"text": what is searched, "doc": what the LLM sees, "meta": dict for filters/display}]
"""
import hashlib, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import llm
from .ingest import chunk

CACHE_DIR = "data/.cache"
DEFAULT_SYSTEM = ("Answer ONLY from the context. Cite item numbers like [2]. "
                  "If the answer is not in the context, say you don't know.")


class RAG:
    def __init__(self, items, use_embeddings=None):
        self.items = items
        self.texts = [i["text"] for i in items]
        want = llm.embeddings_available() if use_embeddings is None else use_embeddings
        self.mode = "tfidf"
        if want:
            try:
                self.mat = self._embed_cached(self.texts)
                self.mode = "embeddings"
            except Exception as e:  # never block the demo
                print("Embeddings failed, falling back to TF-IDF:", e)
        if self.mode == "tfidf":
            self.vec = TfidfVectorizer(stop_words="english")
            self.mat = self.vec.fit_transform(self.texts)

    # ---------- builders ----------
    @classmethod
    def from_texts(cls, texts, **kw):
        items = [{"text": c, "doc": c, "meta": {}} for t in texts for c in chunk(t)]
        return cls(items, **kw)

    @classmethod
    def from_table(cls, df, problem_cols, solution_col, **kw):
        """One row = one item. Only problem_cols are embedded; resolution is returned with the match."""
        items = []
        for n, (_, row) in enumerate(df.iterrows()):
            text = " | ".join(f"{c}: {row[c]}" for c in problem_cols)
            doc = "\n".join(f"{c}: {row[c]}" for c in df.columns)
            items.append({"text": text, "doc": doc, "meta": {c: str(row[c]) for c in df.columns},
                          "solution": str(row[solution_col])})
        return cls(items, **kw)

    # ---------- embeddings with disk cache ----------
    def _embed_cached(self, texts):
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = hashlib.md5((llm.EMBED_MODEL + "\x00".join(texts)).encode()).hexdigest()
        path = f"{CACHE_DIR}/{key}.npy"
        if os.path.exists(path):
            return np.load(path)
        m = np.array(llm.embed(texts), dtype=np.float32)
        m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        np.save(path, m)
        return m

    # ---------- search ----------
    def retrieve(self, q, k=4, where=None):
        """where = optional function(meta_dict)->bool to pre-filter. Returns list of hits."""
        if self.mode == "embeddings":
            qv = np.array(llm.embed([q])[0], dtype=np.float32)
            qv /= (np.linalg.norm(qv) + 1e-9)
            sims = self.mat @ qv
        else:
            sims = cosine_similarity(self.vec.transform([q]), self.mat)[0]
        hits = []
        for i in sims.argsort()[::-1]:
            if where and not where(self.items[i]["meta"]):
                continue
            hits.append({"idx": int(i), "score": float(sims[i]), "item": self.items[i]})
            if len(hits) == k:
                break
        return hits

    def answer(self, q, history=None, k=4, min_score=0.0, where=None, system=DEFAULT_SYSTEM):
        hits = self.retrieve(q, k, where)
        if not hits or hits[0]["score"] < min_score:
            return "No sufficiently similar case found - escalate to a human.", hits
        ctx = "\n\n".join(f"[{h['idx']}] {h['item']['doc']}" for h in hits)
        return chat_with_ctx(q, ctx, history, system), hits


def chat_with_ctx(q, ctx, history, system):
    return llm.chat(f"Context:\n{ctx}\n\nQuestion: {q}", system=system, history=history)
