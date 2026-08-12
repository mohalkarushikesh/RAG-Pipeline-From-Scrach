# Engineering Log — Challenges & How We Solved Them

A record of the real problems hit while building this project, how each was
diagnosed (checking the **data** vs. checking the **code**), the root cause, the
fix, and how it was verified. It doubles as a case study in *measuring instead of
guessing*.

## Method we followed for every issue

1. **Reproduce & observe** the exact symptom (a bad answer, a wrong metric).
2. **Localize** — decide whether the fault is in the **data** (documents, chunks,
   labels) or the **code** (retrieval, scoring, heuristics) by printing the
   intermediate state, not by guessing.
3. **Hypothesize** a root cause from the evidence.
4. **Fix** the smallest thing that addresses the cause.
5. **Verify** with the eval harness and/or a test — and if a fix *didn't* help,
   revert it and re-diagnose. Several "obvious" fixes were rejected this way.

Legend: 🔎 investigation · 🐛 root cause · 🔧 fix · ✅ verification

---

## 1. "No relevant documents found" (the original bug)

**Symptom.** Every query returned the fallback answer; retrieval found 0 documents.

🔎 We read the retrieval path and the on-disk state. The `documents/` folder held
only `.gitkeep`, and `vector_store/` already contained a stray 1-byte
`chroma.sqllite3` file. The loader decided whether to ingest based on *the
directory existing*.

🐛 Root cause — **code + data.** `get_vector_store()` treated "directory exists"
as "index is populated." Because a persistent store directory exists from the
first run, ingestion never ran, so the store stayed empty. Compounded by an empty
corpus.

🔧 Decide ingestion by the **collection count**, not directory existence; add a
sample document. (This later became moot when the project was rebuilt from
scratch per the revised `todo.md`, but it set the tone: *check the actual state,
don't trust a proxy for it.*)

---

## 2. Environment: only PyPI is reachable (no model downloads)

**Symptom.** `BACKEND=st` (sentence-transformers) crashed on first run with a
HuggingFace download error buried under `Cannot send a request, as the client has
been closed`.

🔎 We stopped guessing at the stack trace and **probed the network directly**:

```
curl huggingface.co        -> 000   (blocked)
curl pypi.org              -> 200   (reachable)
curl files.pythonhosted    -> 200   (reachable)
curl google.com / hf-mirror -> 000  (blocked)
```

🐛 Root cause — **environment, not code.** The machine is locked down: PyPI is
whitelisted, everything else (HuggingFace CDN, mirrors, general internet) is
firewalled. So *no* pretrained model can be downloaded — this rules out both
sentence-transformers weights **and** Ollama model pulls.

🔧 Built a third backend — `local` — that needs **zero downloads**: LSA embeddings
(TF-IDF + truncated SVD) in NumPy, a feature reranker, a heuristic contradiction
detector, and extractive answers. Made it the default; kept `st`/`ollama` as
opt-in upgrades for open networks.

✅ Full pipeline + eval run offline with only `numpy` + `pypdf`.

---

## 3. All four retrievers score identically (dense can't beat BM25)

**Symptom.** `bm25`, `dense`, `hybrid`, `hybrid_rerank` returned *identical*
doc-level metrics (Recall@3 = 0.962). Suspicious.

🔎 We added paraphrase queries and a rank-sensitive **Hit@1** metric to try to
separate them — still identical. Then we reasoned about the **math of the data**:
with only 36 chunks, truncated SVD keeps ~35 dimensions (near full rank).

🐛 Root cause — **inherent to the data size, not a bug.** At near-full rank, LSA
cosine ≈ TF-IDF cosine, so "dense" ranks almost exactly like BM25. Offline LSA
*cannot* out-resolve BM25 on a corpus this small, by construction.

🔧 No code fix — this is a real property, so we **documented it honestly** rather
than faking a win, and noted the genuine dense advantage needs a larger corpus or
`BACKEND=st`. We also added **chunk-level** metrics (see #9), which *do*
discriminate.

✅ Reported truthfully in the README; the tie is explained, not hidden.

---

## 4. Contradiction detector: false positive on an unrelated pair

**Symptom.** Asking "What is hybrid search?" flagged a bogus contradiction between
a hybrid-search sentence and an unrelated PTO sentence.

🔎 We printed the salient sentence chosen **per source** and their similarity to
the query. The off-topic source (a handbook) was scoring 0.205 — dragged in by the
wide-net contradiction scan — while the on-topic source scored 0.851.

🐛 Root cause — **code.** The scan compared the best sentence of *every* source in
a wide net, including sources barely related to the question.

🔧 Added a **query-relevance gate** in `_salient_per_source`: a source only enters
the comparison if its salient sentence is ≥ 45% as query-relevant as the top
source. (`CONTRADICTION_MIN_RATIO`.)

✅ False positive gone; added negative-control queries to the eval set so it can't
silently return.

---

## 5. Contradiction detector: crude heuristic (a false positive *and* a false negative)

**Symptom.** On the richer corpus, "How does reranking improve quality?" was
wrongly flagged, and "Can employees work remotely?" was wrongly *not* flagged.

🔎 We dumped the two salient sentences for each:
- FP: *"…is measured…"* vs *"Caching **does not** change retrieval quality…"* — a
  stray "not" tripped the negation rule though the sentences don't conflict.
- FN: *"…up to **three** days…"* vs *"…**five** days…"* — a real conflict expressed
  with **spelled-out numbers** the digit-only detector couldn't see.

🐛 Root cause — **code (heuristic too crude).** Bare negation-difference was too
noisy, and number detection missed word-numbers.

🔧 In `_heuristic_contradiction`: (a) `extract_numbers` now maps words → digits
("three" → 3); (b) flag on a **quantity difference**, or on opposite polarity
**only when the sentences are otherwise near-identical** (Jaccard ≥ 0.5); (c)
require a shared content word (same topic).

✅ Contradiction handling reached **precision 1.0 / recall 1.0** (9 tp, 9 tn),
including paraphrased conflict queries.

---

## 6. Answer didn't lead with the definition ("what is rag")

**Symptom.** "what is rag" returned reranking/other sentences and **never used the
definition sentence**, even though the correct chunk was retrieved and ranked #1.

🔎 We printed the retrieved chunks and their citation numbers. The definition
lived in a *second sub-chunk* of the "What is RAG" section; the reranker had put a
keyword-denser sub-chunk first, and the sentence selector scored purely by weak
LSA similarity — ignoring the reranker's chunk order.

🐛 Root cause — **code.** Sentence selection disagreed with retrieval and had no
notion of "which sentence actually defines the term."

🔧 Two changes in `ExtractiveAnswerer`: (a) score sentences by similarity **+ the
chunk's rank** (trust the reranker); (b) for "what is X / define X" queries, a
**definition boost** for the sentence where the query term is the subject or is
followed by a defining verb ("*RAG is a technique…*", "*Hybrid search combines…*").

✅ Definition-first answers verified for RAG, hybrid search, BM25, HyDE; guarded by
`test_answer_definition_leads_with_definition`.

---

## 7. Garbled answer fragments and near-duplicates

**Symptom.** Answers contained fragments like "s accrue 20 days…" and a
near-duplicate "Corp pays…" vs "Acme Corp pays…".

🔎 Read the chunk text feeding the answerer: the fixed-length **overlap** cut
mid-word, and the dedup key (first 60 chars) differed between "Corp pays…" and
"Acme Corp pays…".

🐛 Root cause — **code.** Overlap wasn't word-aligned; dedup was substring-based.

🔧 Trim overlap to a **word boundary** (`_apply_overlap`); dedup by **token-set
Jaccard ≥ 0.6**; skip sentences that begin mid-sentence (lower-case start).

✅ Clean, de-duplicated answers.

---

## 8. Out-of-domain questions got answered anyway

**Symptom.** "What is the weather on Mars?" produced a confident (wrong) answer
from loosely-matching chunks.

🔎 First hypothesis: add a similarity floor. We **measured** top-sentence
similarity across in- and out-of-domain queries:

```
what is hybrid search   -> 0.851
weather on Mars         -> 0.822   (!)  out-of-domain but HIGH
who won the world cup   -> 0.730
```

The floor idea was **rejected by the data** — LSA drops out-of-vocabulary content
words ("weather", "Mars") and the query collapses onto function words, giving a
spuriously high score. A similarity threshold cannot separate them.

🐛 Root cause — **data/algorithm interaction**, so we changed the *signal*, not the
threshold.

🔧 An **out-of-domain gate** (`_out_of_domain`): if *none* of the query's content
words appear in the corpus vocabulary at all, refuse. Out-of-domain queries all
score 0 content-word coverage; real queries score ≥ 1.

✅ Refuses "weather on Mars" / "world cup" / "tell me a joke"; answers real
questions; guarded by tests.

---

## 9. Doc-level eval was saturated (4 docs make top-3 trivial)

**Symptom.** All modes tied and paraphrase probing didn't help — retrieval quality
felt unmeasurable.

🔎 With only 4 documents, "top-3 of 4" is nearly free; doc-level metrics can't
discriminate ranking quality.

🐛 Root cause — **measurement granularity**, not the retriever.

🔧 Added **chunk-level** metrics driven by `gold` phrases in the eval set: a chunk
is relevant if it's from a labelled source *and* contains a gold answer phrase.
Baseline ≈ k/36, so it's sensitive.

✅ The modes separated immediately (dense 0.795 vs bm25 0.785) — and it exposed the
next bug (#10).

---

## 10. Reranking *lowered* recall — and the first fix made it worse

**Symptom.** The new chunk-level metric showed `hybrid_rerank` Recall@3 = **0.724**,
*below* `hybrid` (0.792). Reranking was hurting.

🔎 **First fix (rejected).** Hypothesis: the reranker collapses both handbook
editions into one, dropping the second. We added a source-diversity top-k rule…
and re-ran the eval: recall fell further to **0.660**. The data said the
hypothesis was wrong, so we **reverted it**.

🔎 **Re-diagnosis.** We printed the reranked lists with gold membership:
- On the PTO query, the reranker already had *both* editions at #0 and #1 — it
  was **not** collapsing editions.
- On the reranking query, the correct chunk (id 24) had fallen out of the **top-6
  entirely** — the reranker's reordering was simply *worse* than the fusion order.

🐛 Root cause — **code.** The `FeatureReranker` weighted its own (weak-LSA) score
too heavily and overrode the already-better fused order.

🔧 Strengthened the **fusion prior**: `0.25·semantic + 0.15·coverage + 0.60·prior`,
so the reranker refines rather than overrides the first stage.

✅ Recall recovered to **0.792** (= hybrid), Hit@1 unchanged, contradiction still
1.0/1.0. Guarded by `test_reranking_does_not_regress_chunk_recall` and
`test_rerank_keeps_both_editions_for_contradiction`. *Two* plausible fixes were
rejected by measurement before the right one landed.

---

## 11. Greetings triggered retrieval

**Symptom.** Typing "Hi" ran retrieval and listed four irrelevant sources.

🔧 Added a small-talk handler (greetings/thanks/bye answered conversationally with
no retrieval) and suppressed the sources block when a result is `insufficient`.

✅ "Hi" → friendly reply; out-of-domain → clean refusal with no misleading sources.

---

## 12. Follow-up expansion mis-fired on a real question

**Symptom.** Multi-turn support expanded "what is BM25" (a complete new question)
with the previous question's context, polluting retrieval.

🔎 The rule was "expand if the query is short." "what is BM25" has one content word
("bm25"), so it looked like a follow-up.

🐛 Root cause — **code (wrong signal).** Brevity ≠ follow-up.

🔧 Detect follow-ups by a **cue word** ("and", "what about") or an **anaphoric
pronoun** ("it", "that") — not length.

✅ "and for 2024?" expands; "what is BM25" does not. Verified across cases +
`test_maybe_expand`.

---

## 13. Operational: EDR flagged `t32.exe` from `pip install`

**Symptom.** A SOC alert flagged a PE binary dropped by `python.exe` during a
`pip install`, with an instruction to delete it.

🔎 We **checked the artifact before acting**: listed the folder (the full,
standard set of `installer` launcher stubs — `t32/t64/w32/w64…`) and confirmed the
owning package (`installer` 0.7.0, a legitimate PyPA wheel installer).

🐛 Root cause — **false positive.** A Python process writing a `.exe` during a
wheel install is expected, benign behavior; the file is a legitimate launcher stub
(the alert itself said so).

🔧 **Did not delete it** — removing a legitimate system file mid-investigation can
look like tampering and breaks `pip`. Instead we provided a factual justification
for the ticket and recommended allowlisting by path + hash.

✅ Same principle as every code bug above: *inspect the actual thing before you act
on a description of it.*

---

## Patterns that kept paying off

- **Print the intermediate state.** Salient sentences, reranked lists with gold
  flags, per-query similarities — almost every root cause was obvious once the
  middle of the pipeline was visible.
- **Let the metric veto the fix.** Source-diversity and a similarity-floor were
  both intuitive and both *wrong*; the eval caught them before they shipped.
- **Change the signal, not the threshold.** For out-of-domain and contradiction,
  the fix was a better signal (content coverage, spelled-number quantity), not a
  tuned cutoff.
- **Prefer honesty over a good-looking number.** The retriever tie and the LSA
  limitation are reported as-is, with the reason.
- **Guard every fix with a test.** 29 offline tests now encode these lessons so
  they can't silently regress.
