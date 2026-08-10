# Retrieval-Augmented Generation Concepts

## What is RAG

Retrieval-Augmented Generation (RAG) is a technique that grounds a language
model's answer in documents retrieved at query time. Instead of relying only on
knowledge baked into the model's weights, a RAG system searches a knowledge base
for passages relevant to the user's question and passes them to the model as
context. This reduces hallucination and lets the assistant answer questions about
private, domain-specific, or up-to-date content.

## Chunking strategy

Documents are too large to embed whole, so they are split into chunks. A good
chunking strategy keeps each chunk semantically coherent. This system first
splits along section headers so a chunk never mixes two unrelated sections, then
recursively splits each section on paragraph, line, sentence, and word
boundaries until it fits the target size. A small overlap is added between
adjacent chunks so that context spanning a boundary is not lost. Chunk size and
overlap are the main quality knobs: chunks that are too large dilute the
embedding, while chunks that are too small lose context.

## Dense retrieval

Dense retrieval embeds every chunk into a vector using an embedding model and
finds the chunks whose vectors are closest to the query vector by cosine
similarity. Dense retrieval captures meaning, so it matches paraphrases and
synonyms that share no words with the query.

## Sparse retrieval with BM25

BM25 is a sparse lexical ranking function based on term frequency and inverse
document frequency, with length normalization. BM25 is excellent at matching
exact keywords, names, codes, and rare terms that a dense embedding may blur
together. Its main weakness is that it cannot match a paraphrase that shares no
vocabulary with the query.

## Hybrid search

Hybrid search combines dense and sparse retrieval so the strengths of each cover
the other's weaknesses. Because dense cosine scores and BM25 scores live on
different scales, this system fuses the two ranked lists with Reciprocal Rank
Fusion rather than adding raw scores.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines several ranked lists using only the rank
of each item, not its score. Each item receives a score of 1 / (k + rank) summed
across the lists it appears in, where k is a smoothing constant, commonly 60.
Items that rank highly in multiple retrievers rise to the top. RRF is robust
precisely because it ignores the incomparable raw scores.

## Reranking

Reranking is a second, more expensive pass over a small candidate pool from the
first-stage retriever. A reranker reads the query together with each candidate
passage and scores true relevance more accurately than first-stage similarity.
This system uses a listwise LLM reranker that reads all candidates at once and
returns them sorted by relevance. Reranking mainly improves precision at the top
ranks, which is what matters when only a few chunks are shown to the generator.

## Handling contradictory sources

When a knowledge base contains documents from different times or authors, the
retrieved passages can disagree. A robust RAG system must not silently pick one
side. Instead it should detect the conflict, present each position, and cite the
source for each. This system instructs the generator to surface contradictions
explicitly and to cite the source behind every claim.

## Measuring retrieval quality

Retrieval quality is measured with an evaluation set of questions labelled with
the documents that should be retrieved. Recall@k measures how many relevant
documents appear in the top k. Mean Reciprocal Rank rewards placing the first
relevant document high. Normalized Discounted Cumulative Gain rewards ranking all
relevant documents highly. Tracking these metrics lets you prove that a change
such as adding reranking actually improved retrieval rather than guessing.
