# RAG Chatbot — Hybrid Retrieval + Reranking + Eval

A small but complete Retrieval-Augmented Generation system built **from scratch**
(no LangChain, no vector-DB server). It implements the parts that actually make
retrieval good, and an eval harness so improvements are *measured*, not guessed.

Features:

- **Structure-aware chunking** — splits along markdown headers, then recursively
  on paragraph/line/sentence/word boundaries, with configurable overlap.
- **Hybrid search** — dense embeddings (cosine) **+** BM25 (implemented from
  scratch), fused with **Reciprocal Rank Fusion**.
- **Reranking** — a listwise LLM reranker reorders the candidate pool for
  top-rank precision.
- **Contradictory-source handling** — the generator must cite every claim and
  explicitly surface conflicts between sources instead of silently picking one.
- **Eval harness** — Recall@k, MRR, nDCG@k across every retrieval mode, plus
  contradiction precision/recall, so you can prove a change helped.

## Project layout (minimal, flat)

```
rag.py           # the whole engine: load, chunk, embed, BM25, RRF, rerank, generate
app.py           # interactive CLI chatbot
eval.py          # retrieval + contradiction evaluation harness
evalset.jsonl    # labelled questions -> relevant documents
documents/       # knowledge base (ships with a sample corpus + built-in contradictions)
vector_store/    # persisted index (index.pkl), auto-created
.env             # configuration + tuning knobs
requirements.txt # numpy, requests, pypdf
```

## Setup

```bash
pip install -r requirements.txt

# Ollama provides embeddings + the chat model (https://ollama.ai)
ollama pull nomic-embed-text        # embedding model
ollama pull orca-mini:latest        # chat model (any chat model works)
ollama serve                        # keep running
```

## Run the chatbot

```bash
python app.py
```

Commands: `help`, `status`, `reload` (rebuild index), `mode <dense|bm25|hybrid|hybrid_rerank>`, `quit`.

Try these against the bundled corpus:

- `What is Reciprocal Rank Fusion?` — retrieved from `rag_concepts.md`.
- `How many PTO days do employees get?` — the 2023 and 2024 handbooks disagree
  (15 vs 20 days); the bot flags the **contradiction** and cites both.
- `Can employees work remotely?` — another intentional conflict across editions.

## Measure retrieval quality

```bash
python eval.py                # Recall@k / MRR / nDCG@k for every mode
python eval.py --k 5          # change the cut-off
python eval.py --contradict   # also score contradiction detection (runs the LLM)
```

Example shape of the output:

```
Retrieval quality @ k=3  (higher is better)
mode                Recall@k       MRR    nDCG@k
------------------------------------------------
bm25                   0.795     0.841     0.808
dense                  0.727     0.795     0.750
hybrid                 0.886     0.909     0.884
hybrid_rerank          0.955     0.955     0.949
------------------------------------------------
hybrid_rerank vs dense: Recall@k 0.727 -> 0.955  (+31.3%)
```

(Numbers depend on your local models; the point is the harness quantifies the
lift from hybrid + reranking.)

## How it works

```
question
   │
   ├── dense retrieval (embedding cosine, top-POOL)
   ├── BM25 retrieval  (lexical, top-POOL)
   │        └── Reciprocal Rank Fusion ─► candidate pool
   │                                          │
   │                              listwise LLM reranking ─► top-K chunks
   │                                          │
   └────────────────────────► contradiction-aware, source-cited generation
                                              │
                                     answer + citations + ⚠ contradictions
```

## Tuning knobs (`.env`)

| Key | Meaning |
|-----|---------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | chunking granularity |
| `TOP_K` | chunks handed to the LLM |
| `POOL` | candidates each retriever contributes before fusion/rerank |
| `RRF_K` | Reciprocal Rank Fusion smoothing constant |
| `RERANK` | enable/disable the LLM reranker |
| `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL` | chat / embedding models |

Change a knob, rerun `python eval.py`, and confirm the metrics moved the right way.

## Adding your own documents

Drop `.md`, `.txt`, or `.pdf` files into `documents/`, then run `reload` in the
chatbot (or delete `vector_store/index.pkl`). The index re-embeds automatically
whenever the corpus or chunking settings change.
