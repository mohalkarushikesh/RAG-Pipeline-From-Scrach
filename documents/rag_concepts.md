# Retrieval-Augmented Generation Concepts

## What is RAG

Retrieval-Augmented Generation (RAG) is a technique that grounds a language
model's answer in documents retrieved at query time. Instead of relying only on
knowledge baked into the model's weights, a RAG system searches a knowledge base
for passages relevant to the user's question and passes them to the model as
context. This reduces hallucination and lets the assistant answer questions about
private, domain-specific, or up-to-date content without retraining the model.

A RAG pipeline has two halves. The retriever finds the most relevant passages,
and the generator writes an answer grounded in them. The quality of the final
answer is capped by the quality of retrieval: if the right passage is never
retrieved, no amount of prompting will recover the correct answer. This is why
serious RAG work spends most of its effort on making retrieval measurable and
good, not on prompt wording.

## Embeddings

An embedding is a fixed-length vector of numbers that represents the meaning of a
piece of text. Texts with similar meaning map to nearby vectors, so semantic
similarity becomes geometric closeness that can be computed with a dot product.
Embeddings are produced by an embedding model; the same model must be used for
both documents and queries so their vectors live in the same space. Embedding
dimensionality trades quality for cost: larger vectors capture more nuance but
take more memory and are slower to compare.

## Chunking strategy

Documents are too large to embed whole, so they are split into chunks. A good
chunking strategy keeps each chunk semantically coherent. This system first
splits along section headers so a chunk never mixes two unrelated sections, then
recursively splits each section on paragraph, line, sentence, and word
boundaries until it fits the target size. A small overlap is added between
adjacent chunks so that context spanning a boundary is not lost. Chunk size and
overlap are the main quality knobs: chunks that are too large dilute the
embedding and retrieve loosely, while chunks that are too small lose the context
needed to answer a question.

## Dense retrieval

Dense retrieval embeds every chunk into a vector using an embedding model and
finds the chunks whose vectors are closest to the query vector by cosine
similarity. Dense retrieval captures meaning, so it matches paraphrases and
synonyms that share no words with the query. Its weakness is that it can miss
exact identifiers such as error codes, product names, or rare technical terms
that a general embedding model blurs together.

## Sparse retrieval with BM25

BM25 is a sparse lexical ranking function based on term frequency and inverse
document frequency, with length normalization. BM25 is excellent at matching
exact keywords, names, codes, and rare terms, and it needs no training or model.
Its main weakness is the mirror image of dense retrieval: it cannot match a
paraphrase that shares no vocabulary with the query, because it only counts
overlapping terms.

## Hybrid search

Hybrid search combines dense and sparse retrieval so the strengths of each cover
the other's weaknesses: dense handles paraphrase, sparse handles exact terms.
Because dense cosine scores and BM25 scores live on different, incomparable
scales, this system fuses the two ranked lists with Reciprocal Rank Fusion rather
than adding raw scores together.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines several ranked lists using only the rank of
each item, not its score. Each item receives a score of 1 / (k + rank) summed
across the lists it appears in, where k is a smoothing constant, commonly 60.
Items that rank highly in multiple retrievers rise to the top. RRF is robust
precisely because it ignores the incomparable raw scores and depends only on
ordering.

## Reranking

Reranking is a second, more expensive pass over a small candidate pool from the
first-stage retriever. A reranker reads the query together with each candidate
passage and scores true relevance more accurately than first-stage similarity.
Because it only runs on a handful of candidates, the extra cost is affordable.
Reranking mainly improves precision at the top ranks, which is what matters when
only a few chunks are shown to the generator. A good reranker refines the
first-stage order rather than discarding it, so it should keep the fusion ranking
as a prior instead of reordering from scratch.

## Prompt grounding and citations

The generator must be constrained to answer only from the retrieved passages and
to cite the source of each claim. Inline citations let a reader verify every
statement against the source text, and they make hallucinations obvious: a claim
with no citation is a red flag. When the retrieved passages do not contain the
answer, a grounded assistant should say so rather than inventing one.

## Handling contradictory sources

When a knowledge base contains documents from different times or authors, the
retrieved passages can disagree. A robust RAG system must not silently pick one
side. Instead it should detect the conflict, present each position, and cite the
source for each. Detecting a genuine conflict requires that the two statements be
about the same thing; comparing unrelated statements produces false alarms, so a
contradiction check should first confirm the statements are on-topic for the
question.

## Measuring retrieval quality

Retrieval quality is measured with an evaluation set of questions labelled with
the documents that should be retrieved. Recall@k measures how many relevant
documents appear in the top k. Mean Reciprocal Rank rewards placing the first
relevant document high. Normalized Discounted Cumulative Gain rewards ranking all
relevant documents highly, discounting lower positions. Tracking these metrics
lets you prove that a change such as adding reranking actually improved retrieval
rather than guessing, and it catches regressions when a change makes things worse.

## Limitations and failure modes

RAG does not eliminate hallucination; it reduces it when retrieval succeeds. Common
failure modes include retrieving the wrong passage, chunking that separates a
question from its answer, stale documents that contradict newer ones, and queries
that fall outside the knowledge base entirely. An out-of-domain query should be
recognized and refused rather than answered from loosely matching but irrelevant
chunks.
