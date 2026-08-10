# Mini RAG Chatbot — Hybrid Retrieval + Reranking + Eval

A small but complete Retrieval-Augmented Generation system built **from scratch**
(no LangChain, no vector-DB server, no cloud API). It implements the parts that
actually make retrieval good, and an eval harness so quality is *measured*, not
guessed — including catching regressions.

> New here? **[CONCEPT.md](CONCEPT.md)** explains the whole project end to end —
> in plain English first, then a full technical breakdown.

Features:

- **Structure-aware chunking** — splits along markdown headers, then recursively
  on paragraph/line/sentence/word boundaries, with word-boundary overlap.
- **Hybrid search** — dense embeddings **+** BM25 (both from scratch), fused with
  **Reciprocal Rank Fusion**.
- **Reranking** — re-scores the fused pool with semantic similarity + query-term
  coverage, while retaining the first-stage fusion signal as a prior.
- **Contradictory-source handling** — cites every claim and explicitly surfaces
  conflicts between sources; scans a wider net than it cites so both sides are seen.
- **Eval harness** — Recall@k, MRR, nDCG@k across every retrieval mode, plus
  contradiction precision/recall.

## Backends (no compromise on features — pick what your network allows)

Set `BACKEND` in `.env`:

| Backend | Embeddings | Rerank | Contradiction | Answer | Needs |
|---------|-----------|--------|---------------|--------|-------|
| `local` *(default)* | LSA (TF-IDF + SVD), from scratch | feature reranker | numeric/negation heuristic | extractive + citations | **numpy only — fully offline** |
| `st` | sentence-transformers | cross-encoder | NLI cross-encoder | extractive + citations | HuggingFace downloads |
| `ollama` | Ollama embeddings | listwise LLM | LLM | LLM-generated | local Ollama server |

The default `local` backend runs on a locked-down machine (PyPI only, no model
downloads, no API key). Switch to `st` or `ollama` for stronger neural models
when your network allows it — the pipeline and eval are identical.

## Project layout (minimal, flat)

```
rag.py           # the whole engine: load, chunk, embed, BM25, RRF, rerank, answer
app.py           # interactive CLI chatbot
eval.py          # retrieval + contradiction evaluation harness
evalset.jsonl    # labelled questions -> relevant documents
documents/       # sample corpus: RAG concepts + advanced patterns + two
                 #   handbook editions with 6 intentional contradictions
vector_store/    # persisted index (index.pkl), auto-created
.env             # backend selection + tuning knobs
requirements.txt # numpy, pypdf  (+ optional sentence-transformers / requests)
```

## Setup & run (default offline backend)

```bash
pip install -r requirements.txt      # numpy + pypdf
python app.py                        # chatbot
python eval.py --contradict          # metrics + contradiction scoring
```

Commands in the chatbot: `help`, `status`, `reload`, `mode <dense|bm25|hybrid|hybrid_rerank>`, `quit`.

Greetings/small talk get a conversational reply (no retrieval), and questions
whose content words don't appear in the corpus return "I don't have information
about that" instead of forcing an answer from irrelevant chunks.

Try:

- `What is Reciprocal Rank Fusion?` — answered from `rag_concepts.md` with a citation.
- `How many PTO days do employees get?` — the 2023 and 2024 handbooks disagree
  (15 vs 20 days); the bot answers **and flags the contradiction**, citing both.
- `Can employees work remotely?` — another intentional cross-edition conflict.

## Measured results (default `local` backend, this corpus)

```
4 documents -> 36 chunks; 26 eval queries (incl. paraphrase + paraphrased-conflict)

Retrieval quality @ k=3  (higher is better)
mode                 Hit@1    Recall@k       MRR    nDCG@k
----------------------------------------------------------
bm25                 0.769       0.962     0.865     0.890
dense                0.769       0.962     0.865     0.890
hybrid               0.769       0.962     0.865     0.890
hybrid_rerank        0.769       0.962     0.865     0.890

Contradiction handling:  precision=1.000  recall=1.000  (tp=9 fp=0 fn=0 tn=9)
```

**Reading this honestly — why all four retrievers tie:** it is not that hybrid is
useless; it is a measurable property of *this* setup. The default `local`
embedder is LSA (TF-IDF + truncated SVD). With only 36 chunks, SVD keeps ~35
latent dimensions — essentially full rank — so dense cosine collapses to
lexical cosine and ranks almost exactly like BM25. Even paraphrased queries and
the rank-sensitive Hit@1 metric don't separate them. **Offline LSA cannot beat
BM25 on a corpus this small, by construction.** A real dense advantage needs
either a much larger corpus (so SVD actually compresses meaning) or a pretrained
embedding model (`BACKEND=st`) — set that and re-run `eval.py` to see it. This is
exactly what the harness is for: it *proves* where a technique helps instead of
assuming it does.

**Where it shines: contradiction handling — 1.0/1.0.** Nine conflicts across PTO,
remote work, hours, probation, bonus, and reimbursement are all detected, and
nine non-conflicts (including identical policies) are left alone — even when the
question is *paraphrased* away from the document's wording ("holiday allowance",
"trial period", "working from home"). The harness also caught real regressions
during development: a reranker that discarded the fusion signal (fixed with a
fusion prior), a scanner that flagged off-topic sources (query-relevance gate),
and a heuristic that fired on a stray "not" and missed spelled-out numbers
(fixed). The `tn`/`tp` cases guard all of these.

## How it works

```
question
   │
   ├── dense retrieval (embedding cosine, top-POOL)
   ├── BM25 retrieval  (lexical, top-POOL)
   │        └── Reciprocal Rank Fusion ─► candidate pool
   │                                          │
   │                                      reranking ─► top-K chunks
   │                                          │
   └────────────────────────► cited answer + ⚠ contradiction scan (wider net)
```

## Tuning knobs (`.env`)

| Key | Meaning |
|-----|---------|
| `BACKEND` | `local` / `st` / `ollama` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | chunking granularity |
| `TOP_K` | chunks used to answer |
| `POOL` | candidates each retriever contributes before fusion/rerank |
| `RRF_K` | Reciprocal Rank Fusion smoothing constant |
| `RERANK` | enable/disable reranking |
| `LSA_DIM` | latent dimensions for the offline embedder |

Change a knob, rerun `python eval.py`, and confirm the metrics moved the right way.

## Adding your own documents

Drop `.md`, `.txt`, or `.pdf` files into `documents/`, then run `reload` in the
chatbot (or delete `vector_store/index.pkl`). The index rebuilds automatically
whenever the corpus, chunking settings, or backend change.
