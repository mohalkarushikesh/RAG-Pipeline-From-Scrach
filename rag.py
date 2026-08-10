"""
Mini RAG engine — everything in one module, from scratch.

Pipeline (no langchain / no chroma):
  load -> structure-aware chunking -> dense embeddings (Ollama) + BM25 (local)
       -> hybrid retrieval via Reciprocal Rank Fusion
       -> listwise LLM reranking
       -> contradiction-aware, source-cited generation

Only third-party deps: numpy, requests, pypdf.  Ollama serves embeddings + LLM.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

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
    """All tunable knobs live here; retrieval quality is tuned by changing these."""

    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_MODEL", "orca-mini:latest")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    documents_path: str = os.getenv("DOCUMENTS_PATH", str(ROOT / "documents"))
    index_path: str = os.getenv("VECTOR_STORE_PATH", str(ROOT / "vector_store")) + "/index.pkl"

    # chunking strategy
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # retrieval
    top_k: int = int(os.getenv("TOP_K", "4"))          # final chunks handed to the LLM
    pool: int = int(os.getenv("POOL", "20"))           # candidates pulled per retriever
    rrf_k: int = int(os.getenv("RRF_K", "60"))         # RRF smoothing constant
    rerank: bool = os.getenv("RERANK", "true").lower() == "true"

    bm25_k1: float = 1.5
    bm25_b: float = 0.75


# --------------------------------------------------------------------------- #
# Ollama client (embeddings + chat) over the raw HTTP API
# --------------------------------------------------------------------------- #
class Ollama:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def embed(self, text: str) -> list[float]:
        r = requests.post(
            f"{self.cfg.base_url}/api/embeddings",
            json={"model": self.cfg.embed_model, "prompt": text},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i, t in enumerate(texts, 1):
            vecs.append(self.embed(t))
            if i % 10 == 0 or i == len(texts):
                print(f"  embedded {i}/{len(texts)} chunks", end="\r")
        if texts:
            print()
        return _normalize(np.array(vecs, dtype=np.float32))

    def chat(self, system: str, user: str, json_format: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        if json_format:
            payload["format"] = "json"
        r = requests.post(f"{self.cfg.base_url}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]


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

    1. Break the document along markdown headers so a chunk never straddles two
       unrelated sections (this is what makes retrieval precise).
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
    # no separator worked: hard character split
    return [text[i : i + size] for i in range(0, len(text), size)]


def _apply_overlap(chunks: list[dict], overlap: int) -> list[dict]:
    if overlap <= 0:
        return chunks
    for i in range(1, len(chunks)):
        if chunks[i]["source"] != chunks[i - 1]["source"]:
            continue
        tail = chunks[i - 1]["text"][-overlap:]
        chunks[i]["text"] = f"{tail} {chunks[i]['text']}".strip()
    return chunks


# --------------------------------------------------------------------------- #
# BM25 (sparse lexical retrieval) — implemented from scratch
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
        self.idf = {
            w: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for w, f in df.items()
        }
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
                idf = self.idf.get(w, 0.0)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
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


def reciprocal_rank_fusion(
    rank_lists: list[list[tuple[int, float]]], rrf_k: int
) -> list[tuple[int, float]]:
    """Fuse several ranked candidate lists into one. Rank-based, so dense and
    BM25 scores never have to be normalized onto the same scale."""
    fused: dict[int, float] = {}
    for lst in rank_lists:
        for rank, (idx, _score) in enumerate(lst):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# The RAG system
# --------------------------------------------------------------------------- #
GEN_SYSTEM = """You are a precise retrieval-augmented assistant.
Rules:
- Answer ONLY using the numbered SOURCES provided. Do not use outside knowledge.
- Cite every claim inline with the source number, e.g. [1] or [2][3].
- If the sources disagree with each other, you MUST surface the contradiction
  explicitly, describe each conflicting position, and cite the source for each side.
- If the sources do not contain the answer, say so plainly.
Respond as compact JSON with EXACTLY these keys:
  "answer": string (markdown, with inline [n] citations),
  "used_sources": array of integers (the source numbers you relied on),
  "contradictions": array of strings (each describes one conflict, citing sources; [] if none),
  "insufficient": boolean (true if the sources cannot answer the question)."""

RERANK_SYSTEM = """You are a search reranker. Given a query and numbered candidate
passages, judge how well each passage answers the query. Respond as JSON:
{"ranking": [{"id": <int>, "score": <float 0..1>}, ...]} listing every candidate id,
sorted from most to least relevant."""


class RAG:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.client = Ollama(self.cfg)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.bm25: BM25 | None = None
        self._loaded = False

    # ---- index lifecycle ------------------------------------------------- #
    def _corpus_hash(self, docs: list[dict]) -> str:
        import hashlib

        h = hashlib.sha256()
        for d in docs:
            h.update(d["source"].encode())
            h.update(d["text"].encode("utf-8", "ignore"))
        h.update(f"{self.cfg.chunk_size}-{self.cfg.chunk_overlap}".encode())
        return h.hexdigest()

    def build_index(self, force: bool = False) -> int:
        """Load docs, chunk, embed, and persist. Reuses cache when unchanged."""
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
                self.bm25 = BM25([c["text"] for c in self.chunks], self.cfg.bm25_k1, self.cfg.bm25_b)
                self._loaded = True
                print(f"  loaded cached index ({len(self.chunks)} chunks)")
                return len(self.chunks)

        print("  building index...")
        chunks: list[dict] = []
        for d in docs:
            chunks.extend(chunk_document(d["text"], d["source"], self.cfg.chunk_size, self.cfg.chunk_overlap))
        print(f"  {len(docs)} document(s) -> {len(chunks)} chunk(s); embedding...")
        embeddings = self.client.embed_batch([c["text"] for c in chunks])

        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = BM25([c["text"] for c in chunks], self.cfg.bm25_k1, self.cfg.bm25_b)
        self._loaded = True

        index_file.parent.mkdir(parents=True, exist_ok=True)
        with index_file.open("wb") as fh:
            pickle.dump({"hash": digest, "chunks": chunks, "embeddings": embeddings}, fh)
        print(f"  index built and persisted ({len(chunks)} chunks)")
        return len(chunks)

    def ensure_index(self) -> None:
        if not self._loaded:
            self.build_index()

    # ---- retrievers ------------------------------------------------------ #
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.embeddings.size == 0:
            return []
        q = _normalize(np.array([self.client.embed(query)], dtype=np.float32))[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, k) if self.bm25 else []

    def hybrid_search(self, query: str, k: int) -> list[tuple[int, float]]:
        dense = self.dense_search(query, self.cfg.pool)
        sparse = self.bm25_search(query, self.cfg.pool)
        return reciprocal_rank_fusion([dense, sparse], self.cfg.rrf_k)[:k]

    def rerank(self, query: str, candidate_ids: list[int], k: int) -> list[int]:
        """Listwise LLM reranking of a candidate pool -> top-k chunk ids."""
        if not candidate_ids:
            return []
        listing = "\n".join(
            f"[{cid}] ({self.chunks[cid]['source']} :: {self.chunks[cid]['section']}) "
            f"{self.chunks[cid]['text'][:400]}"
            for cid in candidate_ids
        )
        user = f"Query: {query}\n\nCandidates:\n{listing}"
        try:
            raw = self.client.chat(RERANK_SYSTEM, user, json_format=True)
            ranking = json.loads(raw).get("ranking", [])
            valid = set(candidate_ids)
            ordered = [int(r["id"]) for r in ranking if int(r.get("id", -1)) in valid]
            # append any candidate the model dropped, preserving original order
            ordered += [c for c in candidate_ids if c not in ordered]
            return ordered[:k]
        except Exception as exc:  # noqa: BLE001
            print(f"  ! rerank failed ({exc}); using fusion order")
            return candidate_ids[:k]

    def retrieve(self, query: str, mode: str = "hybrid_rerank", k: int | None = None) -> list[dict]:
        """Return top chunks (each dict gets a 'rank'). Modes:
        dense | bm25 | hybrid | hybrid_rerank."""
        self.ensure_index()
        k = k or self.cfg.top_k
        if mode == "dense":
            ids = [i for i, _ in self.dense_search(query, k)]
        elif mode == "bm25":
            ids = [i for i, _ in self.bm25_search(query, k)]
        elif mode == "hybrid":
            ids = [i for i, _ in self.hybrid_search(query, k)]
        elif mode == "hybrid_rerank":
            pool = [i for i, _ in self.hybrid_search(query, self.cfg.pool)]
            ids = self.rerank(query, pool, k)
        else:
            raise ValueError(f"unknown mode: {mode}")
        out = []
        for rank, cid in enumerate(ids):
            c = dict(self.chunks[cid])
            c["rank"] = rank
            out.append(c)
        return out

    # ---- generation (with contradiction handling) ------------------------ #
    def answer(self, query: str, mode: str = "hybrid_rerank", k: int | None = None) -> dict:
        chunks = self.retrieve(query, mode=mode, k=k)
        if not chunks:
            return {
                "answer": "I couldn't find anything in the knowledge base. Add documents to the "
                "documents/ folder and rebuild the index.",
                "sources": [],
                "contradictions": [],
                "insufficient": True,
                "mode": mode,
            }
        context = "\n\n".join(
            f"[{i + 1}] Source: {c['source']} — Section: {c['section']}\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        user = f"SOURCES:\n{context}\n\nQUESTION: {query}"
        try:
            raw = self.client.chat(GEN_SYSTEM, user, json_format=True)
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            data = {"answer": f"(generation error: {exc})", "used_sources": [], "contradictions": [], "insufficient": True}

        return {
            "answer": data.get("answer", ""),
            "sources": [
                {"n": i + 1, "source": c["source"], "section": c["section"]}
                for i, c in enumerate(chunks)
            ],
            "used_sources": data.get("used_sources", []),
            "contradictions": data.get("contradictions", []),
            "insufficient": bool(data.get("insufficient", False)),
            "mode": mode,
        }
