"""
Mini RAG engine — everything in one module, from scratch.

Pipeline (no langchain / no vector-DB server):
  load -> structure-aware chunking -> dense embeddings + BM25 (local)
       -> hybrid retrieval via Reciprocal Rank Fusion
       -> reranking
       -> contradiction-aware, source-cited answer

Two interchangeable backends (set BACKEND in .env):
  local   -> sentence-transformers embeddings, cross-encoder reranker,
             NLI-based contradiction detection, extractive answers. No server,
             no API key, runs fully offline on CPU.
  ollama  -> Ollama embeddings + a local LLM for reranking and generation.

BM25, chunking, fusion, and the eval metrics never need any model, so the
retrieval harness runs even with no backend available.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_env()


@dataclass
class Config:
    """All tunable knobs; retrieval quality is tuned by changing these."""

    # backend: "local" (offline LSA, no downloads) | "st" (sentence-transformers) | "ollama"
    backend: str = os.getenv("BACKEND", "local").lower()

    # local offline backend (from-scratch LSA embeddings)
    lsa_dim: int = int(os.getenv("LSA_DIM", "128"))

    # st backend (sentence-transformers) models
    st_embed_model: str = os.getenv("ST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-distilroberta-base")
    nli_threshold: float = float(os.getenv("NLI_THRESHOLD", "0.55"))
    # a source only enters contradiction comparison if its salient sentence is
    # this fraction as relevant to the query as the top source (blocks off-topic
    # sources from producing spurious conflicts)
    contradiction_min_ratio: float = float(os.getenv("CONTRADICTION_MIN_RATIO", "0.45"))

    # ollama backend
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_MODEL", "orca-mini:latest")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    # paths
    documents_path: str = os.getenv("DOCUMENTS_PATH", str(ROOT / "documents"))
    index_path: str = os.getenv("VECTOR_STORE_PATH", str(ROOT / "vector_store")) + "/index.pkl"

    # chunking strategy
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # retrieval
    top_k: int = int(os.getenv("TOP_K", "4"))          # final chunks used to answer
    pool: int = int(os.getenv("POOL", "20"))           # candidates per retriever
    rrf_k: int = int(os.getenv("RRF_K", "60"))         # RRF smoothing constant
    rerank: bool = os.getenv("RERANK", "true").lower() == "true"

    bm25_k1: float = 1.5
    bm25_b: float = 0.75


# --------------------------------------------------------------------------- #
# Document loading
# --------------------------------------------------------------------------- #
def load_documents(documents_path: str) -> list[dict]:
    """Return [{source, text}] for every .md/.txt/.pdf file found."""
    docs: list[dict] = []
    base = Path(documents_path)
    if not base.exists():
        return docs
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in (".md", ".txt"):
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif suffix == ".pdf":
                text = _read_pdf(path)
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed to read {path.name}: {exc}")
            continue
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# --------------------------------------------------------------------------- #
# Chunking strategy: section-aware + recursive splitting with overlap
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def chunk_document(text: str, source: str, size: int, overlap: int) -> list[dict]:
    """Split into semantically coherent chunks.

    1. Break along markdown headers so a chunk never straddles two unrelated
       sections (this is what makes retrieval precise).
    2. Recursively split each section on paragraph -> line -> sentence -> word
       boundaries until it fits `size`, then add `overlap` for context bleed.
    """
    sections: list[tuple[str, str]] = []
    title, buf = "(intro)", []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            if buf:
                sections.append((title, "\n".join(buf)))
                buf = []
            title = m.group(1).strip() or "(untitled)"
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf)))
    if not sections:
        sections = [("(root)", text)]

    chunks: list[dict] = []
    for sec_title, body in sections:
        for piece in _split_recursive(body, size):
            piece = piece.strip()
            if piece:
                chunks.append({"source": source, "section": sec_title, "text": piece})
    return _apply_overlap(chunks, overlap)


def _split_recursive(text: str, size: int, seps=("\n\n", "\n", ". ", " ")) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    for sep in seps:
        if sep in text:
            out, cur = [], ""
            for part in text.split(sep):
                candidate = f"{cur}{sep}{part}" if cur else part
                if len(candidate) <= size:
                    cur = candidate
                else:
                    if cur:
                        out.append(cur)
                    if len(part) > size:
                        out.extend(_split_recursive(part, size, seps))
                        cur = ""
                    else:
                        cur = part
            if cur:
                out.append(cur)
            return out
    return [text[i : i + size] for i in range(0, len(text), size)]


def _apply_overlap(chunks: list[dict], overlap: int) -> list[dict]:
    if overlap <= 0:
        return chunks
    for i in range(1, len(chunks)):
        if chunks[i]["source"] != chunks[i - 1]["source"]:
            continue
        tail = chunks[i - 1]["text"][-overlap:]
        # start the overlap at a word boundary so we don't inject a partial word
        sp = tail.find(" ")
        if sp != -1:
            tail = tail[sp + 1 :]
        chunks[i]["text"] = f"{tail} {chunks[i]['text']}".strip()
    return chunks


# Verbs that signal a definition when they immediately follow the term.
_DEF_VERBS = {
    "is", "are", "was", "were", "means", "refers", "stands", "combines",
    "describes", "denotes", "defines", "represents", "improves",
}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 15]


# --------------------------------------------------------------------------- #
# BM25 (sparse lexical retrieval) — from scratch
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Function words carry no domain signal; ignored when judging if a query is
# about the corpus at all.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "at", "for", "and", "or", "but", "with", "as", "by",
    "do", "does", "did", "how", "what", "when", "where", "who", "whom", "which",
    "why", "can", "could", "would", "should", "will", "shall", "may", "might",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "this",
    "that", "these", "those", "there", "here", "get", "got", "many", "much",
    "tell", "give", "about", "from", "into", "per", "any", "some", "have", "has",
}


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(doc) for doc in corpus]
        self.n = len(self.tokens)
        self.doc_len = np.array([len(t) for t in self.tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        df: Counter[str] = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        self.idf = {w: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [Counter(t) for t in self.tokens]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            tf, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores[i] = s
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


# --------------------------------------------------------------------------- #
# Vector maths + rank fusion
# --------------------------------------------------------------------------- #
def _normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def reciprocal_rank_fusion(rank_lists: list[list[tuple[int, float]]], rrf_k: int) -> list[tuple[int, float]]:
    """Fuse ranked lists using only ranks, so dense and BM25 scores never have to
    be normalized onto the same scale."""
    fused: dict[int, float] = {}
    for lst in rank_lists:
        for rank, (idx, _score) in enumerate(lst):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# Backend: embedders
# --------------------------------------------------------------------------- #
class STEmbedder:
    """Local sentence-transformers bi-encoder. CPU, no server, no API key."""

    def __init__(self, cfg: Config):
        from sentence_transformers import SentenceTransformer

        print(f"  loading embedder: {cfg.st_embed_model}")
        self.model = SentenceTransformer(cfg.st_embed_model)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        ).astype(np.float32)


class OllamaEmbedder:
    """Embeddings via the Ollama HTTP API."""

    def __init__(self, cfg: Config):
        import requests

        self.cfg, self._requests = cfg, requests

    def embed(self, text: str) -> np.ndarray:
        r = self._requests.post(
            f"{self.cfg.base_url}/api/embeddings",
            json={"model": self.cfg.ollama_embed_model, "prompt": text},
            timeout=120,
        )
        r.raise_for_status()
        return _normalize(np.array([r.json()["embedding"]], dtype=np.float32))[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i, t in enumerate(texts, 1):
            vecs.append(self.embed(t))
            if i % 10 == 0 or i == len(texts):
                print(f"  embedded {i}/{len(texts)}", end="\r")
        if texts:
            print()
        return np.array(vecs, dtype=np.float32)


class LSAEmbedder:
    """Latent Semantic Analysis embeddings, from scratch (TF-IDF + truncated SVD).

    Needs only numpy — no model download, no server, fully offline. SVD folds
    co-occurring terms onto shared latent dimensions, so it captures a degree of
    'semantic' similarity beyond exact keywords, which is what lets it complement
    BM25 in hybrid search. The fitted projection is persisted with the index so
    query-time vectors live in the same space as the documents.
    """

    def __init__(self, cfg: Config):
        self.dim = cfg.lsa_dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (vocab, k)

    def _tfidf(self, tokens_list: list[list[str]]) -> np.ndarray:
        mat = np.zeros((len(tokens_list), len(self.vocab)), dtype=np.float32)
        for r, toks in enumerate(tokens_list):
            for w, f in Counter(toks).items():
                j = self.vocab.get(w)
                if j is not None:
                    mat[r, j] = f
        return mat * self.idf

    def fit(self, texts: list[str]) -> np.ndarray:
        toks = [tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for t in toks:
            df.update(set(t))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[w])) + 1.0 for w in self.vocab], dtype=np.float32
        )
        x = self._tfidf(toks)
        k = max(1, min(self.dim, min(x.shape) - 1)) if min(x.shape) > 1 else 1
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        self.components = vt[:k].T.astype(np.float32)
        return _normalize(x @ self.components)

    def transform(self, texts: list[str]) -> np.ndarray:
        x = self._tfidf([tokenize(t) for t in texts])
        return _normalize(x @ self.components)

    def embed(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self.fit(texts) if self.components is None else self.transform(texts)

    def save_state(self) -> dict:
        return {"dim": self.dim, "vocab": self.vocab, "idf": self.idf, "components": self.components}

    def load_state(self, state: dict) -> None:
        self.dim = state["dim"]
        self.vocab = state["vocab"]
        self.idf = state["idf"]
        self.components = state["components"]


# --------------------------------------------------------------------------- #
# Backend: rerankers
# --------------------------------------------------------------------------- #
class CrossEncoderReranker:
    """Local cross-encoder reranker — scores (query, passage) jointly."""

    def __init__(self, cfg: Config):
        from sentence_transformers import CrossEncoder

        print(f"  loading reranker: {cfg.cross_encoder_model}")
        self.model = CrossEncoder(cfg.cross_encoder_model)

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c["text"]) for c in chunks])
        order = np.argsort(-np.asarray(scores))
        return [chunks[i] for i in order]


class LLMReranker:
    """Listwise reranking via an Ollama chat model."""

    _SYSTEM = (
        "You are a search reranker. Given a query and numbered candidate passages, "
        "judge how well each answers the query. Respond as JSON "
        '{"ranking": [{"id": <int>, "score": <float 0..1>}, ...]} listing every id, '
        "most to least relevant."
    )

    def __init__(self, cfg: Config):
        self.chat = OllamaChat(cfg)

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        listing = "\n".join(
            f"[{c['id']}] ({c['source']} :: {c['section']}) {c['text'][:400]}" for c in chunks
        )
        try:
            raw = self.chat.complete(self._SYSTEM, f"Query: {query}\n\nCandidates:\n{listing}", json_format=True)
            ranking = json.loads(raw).get("ranking", [])
            by_id = {c["id"]: c for c in chunks}
            ordered = [by_id[int(r["id"])] for r in ranking if int(r.get("id", -1)) in by_id]
            ordered += [c for c in chunks if c not in ordered]
            return ordered
        except Exception as exc:  # noqa: BLE001
            print(f"  ! LLM rerank failed ({exc}); keeping fusion order")
            return chunks


class NoReranker:
    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        return chunks


class FeatureReranker:
    """Offline reranker (no neural model). Re-scores the fused candidate pool by
    combining semantic similarity (embedder cosine) with query-term coverage — a
    signal first-stage fusion ignores — to sharpen precision at the top ranks."""

    def __init__(self, cfg: Config, embedder):
        self.embedder = embedder

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        q = self.embedder.embed(query)
        vecs = self.embedder.embed_batch([c["text"] for c in chunks])
        sem = vecs @ q
        sem = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
        qterms = set(tokenize(query))
        cov = np.array(
            [len(qterms & set(tokenize(c["text"]))) / max(1, len(qterms)) for c in chunks],
            dtype=np.float32,
        )
        # Retain the first-stage fusion signal as a strong prior (candidates
        # arrive in fusion order). On a weak-embedder corpus the fused order is
        # already good, so the reranker should *refine* it, not override it —
        # measured: a lighter prior demoted correct chunks and hurt recall.
        n = len(chunks)
        prior = np.array([1.0 - i / n for i in range(n)], dtype=np.float32)
        score = 0.25 * sem + 0.15 * cov + 0.60 * prior
        order = np.argsort(-score)
        return [chunks[i] for i in order]


# --------------------------------------------------------------------------- #
# Contradiction detection (NLI) + Ollama chat helper
# --------------------------------------------------------------------------- #
class OllamaChat:
    def __init__(self, cfg: Config):
        import requests

        self.cfg, self._requests = cfg, requests

    def complete(self, system: str, user: str, json_format: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.chat_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        if json_format:
            payload["format"] = "json"
        r = self._requests.post(f"{self.cfg.base_url}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


def extract_numbers(text: str) -> set[str]:
    """Numbers in a sentence, digits and spelled-out words alike ('three' -> 3),
    so quantity conflicts are caught however they're written."""
    nums = set(re.findall(r"\d+", text))
    for w in tokenize(text):
        if w in _NUMBER_WORDS:
            nums.add(str(_NUMBER_WORDS[w]))
    return nums


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


class ContradictionDetector:
    """Detects conflicts between the query-relevant statements of different
    sources. Uses an NLI cross-encoder when available, else a light heuristic."""

    _NEG_RE = re.compile(r"\b(no longer|not|never|discontinued|forfeited|without)\b", re.I)

    def __init__(self, cfg: Config, use_nli: bool = False):
        self.cfg = cfg
        self.model = None
        if not use_nli:
            return  # offline mode: skip the (download-requiring) NLI model
        try:
            from sentence_transformers import CrossEncoder

            print(f"  loading NLI model: {cfg.nli_model}")
            self.model = CrossEncoder(cfg.nli_model)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! NLI model unavailable ({exc}); using heuristic contradiction check")

    def _nli_contradiction(self, a: str, b: str) -> float:
        # cross-encoder NLI labels are ordered [contradiction, entailment, neutral]
        logits = self.model.predict([(a, b), (b, a)])
        logits = np.asarray(logits, dtype=np.float64)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(max(probs[0][0], probs[1][0]))

    def _heuristic_contradiction(self, a: str, b: str) -> float:
        # Require shared content words: the two statements must be about the same
        # thing before a difference counts as a real conflict.
        ca = set(tokenize(a)) - STOPWORDS
        cb = set(tokenize(b)) - STOPWORDS
        if not (ca & cb):
            return 0.0
        # A differing quantity on the same topic is a strong, low-noise signal.
        na, nb = extract_numbers(a), extract_numbers(b)
        if na and nb and na != nb:
            return 1.0
        # Opposite polarity only counts when the sentences are near-identical
        # otherwise (e.g. "X is offered" vs "X is no longer offered"); this avoids
        # firing on a stray "not" in two loosely-related sentences.
        negation_differs = bool(self._NEG_RE.search(a)) != bool(self._NEG_RE.search(b))
        if negation_differs and _jaccard(ca, cb) >= 0.5:
            return 1.0
        return 0.0

    def find(self, statements: list[dict]) -> list[str]:
        """statements: [{source, sentence}] — one salient sentence per source."""
        out: list[str] = []
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                a, b = statements[i], statements[j]
                if a["source"] == b["source"]:
                    continue
                if self.model is not None:
                    score = self._nli_contradiction(a["sentence"], b["sentence"])
                else:
                    score = self._heuristic_contradiction(a["sentence"], b["sentence"])
                if score >= self.cfg.nli_threshold:
                    out.append(
                        f"{a['source']} states \"{a['sentence']}\" while "
                        f"{b['source']} states \"{b['sentence']}\" (conflict score {score:.2f})."
                    )
        return out


# --------------------------------------------------------------------------- #
# Backend: answerers
# --------------------------------------------------------------------------- #
class ExtractiveAnswerer:
    """No-LLM answerer: builds the answer from the most query-relevant sentences
    of the retrieved chunks, with citations, and reports contradictions."""

    def __init__(self, cfg: Config, embedder, use_nli: bool = False):
        self.cfg = cfg
        self.embedder = embedder
        self.detector = ContradictionDetector(cfg, use_nli=use_nli)

    def answer(self, query: str, chunks: list[dict], scan_chunks: list[dict] | None = None) -> dict:
        q = self.embedder.embed(query)

        # collect candidate sentences tagged with their citation number
        cands: list[tuple[int, str]] = []
        for n, c in enumerate(chunks, 1):
            for sent in split_sentences(c["text"]):
                cands.append((n, sent))
        if not cands:
            return {"answer": "", "used_sources": [], "contradictions": [], "insufficient": True}

        sent_vecs = self.embedder.embed_batch([s for _, s in cands])
        sims = sent_vecs @ q

        # Combine per-sentence similarity with the chunk's retrieval rank. The
        # reranker already decided which chunks are most relevant; trusting that
        # ordering keeps the answer anchored to the top chunk instead of pulling a
        # stray high-similarity sentence out of a lower-ranked one.
        n_chunks = max(1, len(chunks))
        rng = float(sims.max() - sims.min())
        sims_norm = (sims - sims.min()) / (rng + 1e-9)
        chunk_prior = np.array([1.0 - (cands[i][0] - 1) / n_chunks for i in range(len(cands))], dtype=np.float32)

        # For "what is X" / "define X" questions, prefer the sentence that actually
        # *defines* X: the query term is the subject at the sentence start, or is
        # immediately followed by a defining verb ("RAG is a technique...",
        # "Hybrid search combines..."). This beats a mere mention of the term.
        q_content = [t for t in tokenize(query) if t not in STOPWORDS]
        is_def_query = bool(re.match(r"\s*(what\s+(is|are|does)|define)\b", query.lower())) and q_content
        def_bonus = np.zeros(len(cands), dtype=np.float32)
        if is_def_query:
            for i, (_n, sent) in enumerate(cands):
                toks = tokenize(sent)
                pos = next((j for j, t in enumerate(toks) if t in q_content), None)
                if pos is None:
                    continue
                starts_sentence = pos == 0
                followed_by_verb = any(toks[j] in _DEF_VERBS for j in range(pos + 1, min(pos + 4, len(toks))))
                if starts_sentence or followed_by_verb:
                    def_bonus[i] = 1.0

        if is_def_query:
            score = 0.45 * sims_norm + 0.20 * chunk_prior + 0.35 * def_bonus
        else:
            score = 0.65 * sims_norm + 0.35 * chunk_prior

        used, lines, chosen_tokens = [], [], []
        for idx in np.argsort(-score):
            if len(lines) >= 4:
                break
            if float(sims[idx]) < 0.15:  # relevance floor on raw similarity
                continue
            n, sent = cands[idx]
            sent = sent.strip()
            if sent[:1].islower():  # mid-sentence fragment bled in from overlap
                continue
            toks = set(tokenize(sent)) - STOPWORDS
            if any(_jaccard(toks, prev) >= 0.6 for prev in chosen_tokens):
                continue  # near-duplicate of an already-selected sentence
            chosen_tokens.append(toks)
            lines.append(f"- {sent} [{n}]")
            if n not in used:
                used.append(n)

        # Contradiction scan casts a wider net than the cited chunks, so both
        # sides of a conflict are present even if only one side ranked top-k.
        statements = self._salient_per_source(q, scan_chunks or chunks)
        contradictions = self.detector.find(statements)

        answer = "\n".join(lines) if lines else "The retrieved sources don't clearly answer this."
        return {
            "answer": answer,
            "used_sources": used,
            "contradictions": contradictions,
            "insufficient": not lines,
        }

    def _salient_per_source(self, q: np.ndarray, chunks: list[dict]) -> list[dict]:
        """The single most query-relevant sentence from each distinct source."""
        pairs: list[tuple[str, str]] = []
        for c in chunks:
            for sent in split_sentences(c["text"]):
                if sent[:1].islower():  # skip overlap fragments
                    continue
                pairs.append((c["source"], sent))
        if not pairs:
            return []
        sims = self.embedder.embed_batch([s for _, s in pairs]) @ q
        best: dict[str, tuple[float, str]] = {}
        for idx in np.argsort(-sims):
            src, sent = pairs[idx]
            if src not in best:
                best[src] = (float(sims[idx]), sent)
        if not best:
            return []
        # keep only sources that are genuinely relevant to the query, so
        # off-topic sources can't produce spurious contradictions
        cutoff = max(best.values(), key=lambda v: v[0])[0] * self.cfg.contradiction_min_ratio
        return [{"source": s, "sentence": v[1]} for s, v in best.items() if v[0] >= cutoff]


class LLMAnswerer:
    """LLM answerer (Ollama). The model cites sources and surfaces conflicts."""

    _SYSTEM = """You are a precise retrieval-augmented assistant.
Rules:
- Answer ONLY using the numbered SOURCES. Do not use outside knowledge.
- Cite every claim inline with its source number, e.g. [1] or [2][3].
- If sources disagree, you MUST surface the contradiction, describe each side, and cite each.
- If the sources don't contain the answer, say so.
Respond as compact JSON with keys:
  "answer" (markdown with [n] citations),
  "used_sources" (array of ints),
  "contradictions" (array of strings, [] if none),
  "insufficient" (boolean)."""

    def __init__(self, cfg: Config):
        self.chat = OllamaChat(cfg)

    def answer(self, query: str, chunks: list[dict], scan_chunks: list[dict] | None = None) -> dict:
        context = "\n\n".join(
            f"[{i + 1}] Source: {c['source']} — Section: {c['section']}\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        try:
            raw = self.chat.complete(self._SYSTEM, f"SOURCES:\n{context}\n\nQUESTION: {query}", json_format=True)
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            data = {"answer": f"(generation error: {exc})", "used_sources": [], "contradictions": [], "insufficient": True}
        return {
            "answer": data.get("answer", ""),
            "used_sources": data.get("used_sources", []),
            "contradictions": data.get("contradictions", []),
            "insufficient": bool(data.get("insufficient", False)),
        }


# --------------------------------------------------------------------------- #
# The RAG system
# --------------------------------------------------------------------------- #
class RAG:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.bm25: BM25 | None = None
        self._loaded = False
        self._embedder = None
        self._reranker = None
        self._answerer = None

    # ---- lazy backend construction -------------------------------------- #
    @property
    def embedder(self):
        if self._embedder is None:
            if self.cfg.backend == "local":
                self._embedder = LSAEmbedder(self.cfg)
            elif self.cfg.backend == "st":
                self._embedder = STEmbedder(self.cfg)
            else:
                self._embedder = OllamaEmbedder(self.cfg)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            if not self.cfg.rerank:
                self._reranker = NoReranker()
            elif self.cfg.backend == "local":
                self._reranker = FeatureReranker(self.cfg, self.embedder)
            elif self.cfg.backend == "st":
                self._reranker = CrossEncoderReranker(self.cfg)
            else:
                self._reranker = LLMReranker(self.cfg)
        return self._reranker

    @property
    def answerer(self):
        if self._answerer is None:
            if self.cfg.backend == "ollama":
                self._answerer = LLMAnswerer(self.cfg)
            else:
                # local + st both use extractive answers; NLI only when st (needs download)
                self._answerer = ExtractiveAnswerer(
                    self.cfg, self.embedder, use_nli=(self.cfg.backend == "st")
                )
        return self._answerer

    # ---- index lifecycle ------------------------------------------------- #
    def _corpus_hash(self, docs: list[dict]) -> str:
        import hashlib

        h = hashlib.sha256()
        for d in docs:
            h.update(d["source"].encode())
            h.update(d["text"].encode("utf-8", "ignore"))
        h.update(f"{self.cfg.backend}-{self.cfg.chunk_size}-{self.cfg.chunk_overlap}".encode())
        return h.hexdigest()

    def build_index(self, force: bool = False) -> int:
        docs = load_documents(self.cfg.documents_path)
        if not docs:
            print(f"  ! no documents found in {self.cfg.documents_path}")
            self.chunks, self.embeddings, self.bm25, self._loaded = [], np.zeros((0, 0), np.float32), None, True
            return 0

        digest = self._corpus_hash(docs)
        index_file = Path(self.cfg.index_path)

        if not force and index_file.exists():
            with index_file.open("rb") as fh:
                cache = pickle.load(fh)
            if cache.get("hash") == digest:
                self.chunks = cache["chunks"]
                self.embeddings = cache["embeddings"]
                if hasattr(self.embedder, "load_state") and cache.get("embedder_state") is not None:
                    self.embedder.load_state(cache["embedder_state"])
                self.bm25 = BM25([c["text"] for c in self.chunks], self.cfg.bm25_k1, self.cfg.bm25_b)
                self._loaded = True
                print(f"  loaded cached index ({len(self.chunks)} chunks)")
                return len(self.chunks)

        print("  building index...")
        chunks: list[dict] = []
        for d in docs:
            chunks.extend(chunk_document(d["text"], d["source"], self.cfg.chunk_size, self.cfg.chunk_overlap))
        print(f"  {len(docs)} document(s) -> {len(chunks)} chunk(s); embedding...")
        embeddings = self.embedder.embed_batch([c["text"] for c in chunks])

        self.chunks, self.embeddings = chunks, embeddings
        self.bm25 = BM25([c["text"] for c in chunks], self.cfg.bm25_k1, self.cfg.bm25_b)
        self._loaded = True

        state = self.embedder.save_state() if hasattr(self.embedder, "save_state") else None
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with index_file.open("wb") as fh:
            pickle.dump(
                {"hash": digest, "chunks": chunks, "embeddings": embeddings, "embedder_state": state}, fh
            )
        print(f"  index built and persisted ({len(chunks)} chunks)")
        return len(chunks)

    def ensure_index(self) -> None:
        if not self._loaded:
            self.build_index()

    # ---- retrievers ------------------------------------------------------ #
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.embeddings.size == 0:
            return []
        q = self.embedder.embed(query)
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, k) if self.bm25 else []

    def hybrid_search(self, query: str, k: int) -> list[tuple[int, float]]:
        dense = self.dense_search(query, self.cfg.pool)
        sparse = self.bm25_search(query, self.cfg.pool)
        return reciprocal_rank_fusion([dense, sparse], self.cfg.rrf_k)[:k]

    def _chunk(self, i: int) -> dict:
        return dict(self.chunks[i], id=i)

    def retrieve(self, query: str, mode: str = "hybrid_rerank", k: int | None = None) -> list[dict]:
        """Return top chunks. Modes: dense | bm25 | hybrid | hybrid_rerank."""
        self.ensure_index()
        k = k or self.cfg.top_k
        if mode == "dense":
            chosen = [self._chunk(i) for i, _ in self.dense_search(query, k)]
        elif mode == "bm25":
            chosen = [self._chunk(i) for i, _ in self.bm25_search(query, k)]
        elif mode == "hybrid":
            chosen = [self._chunk(i) for i, _ in self.hybrid_search(query, k)]
        elif mode == "hybrid_rerank":
            pool = [self._chunk(i) for i, _ in self.hybrid_search(query, self.cfg.pool)]
            chosen = self.reranker.rerank(query, pool)[:k]
        else:
            raise ValueError(f"unknown mode: {mode}")
        for rank, c in enumerate(chosen):
            c["rank"] = rank
        return chosen

    def _out_of_domain(self, query: str) -> bool:
        """True if none of the query's content words appear in the corpus at all.
        A cheap, reliable out-of-domain signal for the LSA backend, where
        embedding similarity is fooled by dropped out-of-vocabulary words."""
        vocab = set(self.bm25.idf.keys()) if self.bm25 else set()
        terms = [t for t in tokenize(query) if t not in STOPWORDS]
        if not terms:
            return True
        return not any(t in vocab for t in terms)

    # ---- answer ---------------------------------------------------------- #
    def answer(self, query: str, mode: str = "hybrid_rerank", k: int | None = None) -> dict:
        self.ensure_index()
        if self.cfg.backend == "local" and self._out_of_domain(query):
            return {
                "answer": "I don't have information about that in the knowledge base.",
                "sources": [], "used_sources": [], "contradictions": [], "insufficient": True, "mode": mode,
            }
        chunks = self.retrieve(query, mode=mode, k=k)
        if not chunks:
            return {
                "answer": "The knowledge base is empty. Add documents to documents/ and rebuild.",
                "sources": [], "used_sources": [], "contradictions": [], "insufficient": True, "mode": mode,
            }
        # Wider net for contradiction scanning so both sides of a conflict are
        # seen even when only one side ranks in the cited top-k.
        scan = self.retrieve(query, mode="hybrid", k=max(8, (k or self.cfg.top_k) * 2))
        result = self.answerer.answer(query, chunks, scan_chunks=scan)
        result["sources"] = [
            {"n": i + 1, "source": c["source"], "section": c["section"], "text": c["text"]}
            for i, c in enumerate(chunks)
        ]
        result["mode"] = mode
        return result
