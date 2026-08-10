# Advanced RAG Patterns

## Query expansion

Query expansion rewrites or augments the user's question before retrieval to
close the vocabulary gap between how a question is phrased and how the answer is
written. Expansion can add synonyms, spell out acronyms, or generate several
paraphrases of the query and retrieve for each. The union of the results is then
fused. Expansion mainly helps recall on short or ambiguous queries at the cost of
extra retrieval work.

## Hypothetical Document Embeddings (HyDE)

HyDE improves dense retrieval by first asking the language model to write a
hypothetical answer to the question, then embedding that hypothetical answer
instead of the raw question. Because a full answer shares more vocabulary and
structure with the real passage than a terse question does, its embedding often
lands closer to the correct chunk. HyDE trades an extra model call for better
recall and is most useful when questions are much shorter than the documents.

## Maximal Marginal Relevance

Maximal Marginal Relevance (MMR) reranks candidates to balance relevance against
diversity. At each step it picks the candidate that is most relevant to the query
while being least similar to the passages already selected. MMR is valuable when
the top results are near-duplicates, because it replaces redundant chunks with
ones that add new information, giving the generator broader coverage.

## Cross-encoder reranking

A cross-encoder reranker feeds the query and a candidate passage through the model
together and outputs a single relevance score. Because the two texts attend to
each other, a cross-encoder judges relevance far more accurately than comparing
two independently produced embeddings. Cross-encoders are too slow to score a
whole corpus, so they are used only to rerank a small first-stage candidate pool.

## Contextual and semantic chunking

Instead of splitting on a fixed character count, contextual chunking splits at
natural semantic boundaries such as headings, topic shifts, or sentence groups
that discuss one idea. Some systems prepend a short summary of the surrounding
section to each chunk so an isolated passage still carries the context needed to
interpret it. Better chunk boundaries usually help retrieval more than tuning the
embedding model.

## Caching and cost control

Embeddings for unchanged documents should be cached and only recomputed when the
source or the chunking settings change, because embedding a large corpus is the
most expensive step. Query-time embeddings and generated answers can also be
cached keyed by the query text. Caching does not change retrieval quality but it
sharply reduces latency and cost for repeated or similar questions.

## When to use which retriever

Use BM25 alone when the corpus is small and questions share vocabulary with the
documents, because it is exact, cheap, and needs no model. Add dense retrieval
when questions are paraphrased or use different words than the source. Use hybrid
search when both matter, and add reranking when only a few chunks can be shown and
top-rank precision is critical. Always measure on your own evaluation set before
assuming a more complex retriever is better, since added complexity can regress
quality on an easy corpus.
