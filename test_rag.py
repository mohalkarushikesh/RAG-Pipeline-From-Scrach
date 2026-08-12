"""
Offline test suite for the Mini RAG chatbot.

Covers the pure logic (chunking, BM25, RRF, number extraction, contradiction
heuristic, out-of-domain gate, eval metrics, follow-up expansion) and a few
end-to-end behaviours on the default `local` backend. No models are downloaded
and no network is used, so `pytest` runs anywhere in a second or two.

    pip install pytest
    pytest -q
"""
from __future__ import annotations

import pytest

import eval as ev
import rag
from rag import (
    BM25,
    Config,
    ContradictionDetector,
    RAG,
    chunk_document,
    extract_numbers,
    reciprocal_rank_fusion,
    split_sentences,
    tokenize,
)
from web import maybe_expand


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_tokenize_lowercases_and_splits():
    assert tokenize("Hybrid Search, RRF-60!") == ["hybrid", "search", "rrf", "60"]


def test_chunk_document_is_section_aware_and_bounded():
    text = "# Intro\n\nAlpha beta gamma.\n\n# Details\n\n" + ("word " * 400)
    chunks = chunk_document(text, "doc.md", size=200, overlap=40)
    assert chunks, "should produce chunks"
    assert {"Intro", "Details"} <= {c["section"] for c in chunks}
    # every chunk stays near the size budget (allowing the overlap prefix)
    assert all(len(c["text"]) <= 200 + 40 + 20 for c in chunks)


def test_chunk_overlap_starts_on_word_boundary():
    text = "para one has several words here.\n\n" + ("alpha bravo charlie delta echo. " * 30)
    chunks = chunk_document(text, "d.md", size=120, overlap=30)
    # no chunk should begin mid-word (overlap trims to a word boundary)
    for c in chunks[1:]:
        assert not c["text"][:1].islower() or " " in c["text"][:40]


def test_split_sentences_drops_tiny_fragments():
    sents = split_sentences("This is a full sentence. Hi. Another complete sentence here.")
    assert "This is a full sentence." in sents
    assert all(len(s) > 15 for s in sents)


# --------------------------------------------------------------------------- #
# BM25 + fusion
# --------------------------------------------------------------------------- #
def test_bm25_ranks_exact_term_match_first():
    corpus = [
        "Reciprocal Rank Fusion merges ranked lists by position.",
        "Dense retrieval uses embeddings and cosine similarity.",
        "BM25 is a sparse lexical ranking function.",
    ]
    bm = BM25(corpus)
    hits = bm.search("sparse lexical BM25", k=3)
    assert hits and hits[0][0] == 2


def test_rrf_prefers_items_ranked_by_both():
    a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    b = [(3, 9.0), (1, 8.0), (9, 7.0)]
    fused = reciprocal_rank_fusion([a, b], rrf_k=60)
    order = [i for i, _ in fused]
    # 1 and 3 appear in both lists -> should outrank items in only one
    assert order[0] in (1, 3) and order[1] in (1, 3)
    assert order.index(2) > order.index(1)


# --------------------------------------------------------------------------- #
# Numbers + contradiction heuristic
# --------------------------------------------------------------------------- #
def test_extract_numbers_handles_digits_and_words():
    assert extract_numbers("up to three days and 15 items") == {"3", "15"}


@pytest.fixture(scope="module")
def detector():
    return ContradictionDetector(Config(), use_nli=False)


def test_contradiction_quantity_conflict(detector):
    a = "Full-time employees accrue 15 days of paid time off per year."
    b = "Full-time employees accrue 20 days of paid time off per year."
    assert detector._heuristic_contradiction(a, b) == 1.0


def test_contradiction_spelled_out_numbers(detector):
    a = "Employees may work remotely up to three days per week."
    b = "All employees are required to work from the office five days per week."
    assert detector._heuristic_contradiction(a, b) == 1.0


def test_contradiction_ignores_stray_negation_on_unrelated_sentences(detector):
    a = "Retrieval quality is measured with an evaluation set of questions."
    b = "Caching does not change retrieval quality but reduces latency and cost."
    assert detector._heuristic_contradiction(a, b) == 0.0


def test_contradiction_requires_shared_topic(detector):
    assert detector._heuristic_contradiction("The sky is blue.", "Bananas cost 5 dollars.") == 0.0


# --------------------------------------------------------------------------- #
# Eval metrics
# --------------------------------------------------------------------------- #
def test_recall_mrr_ndcg():
    ranked = ["b.md", "a.md", "c.md"]
    relevant = {"a.md", "c.md"}
    assert ev.recall_at_k(ranked, relevant, 3) == 1.0
    assert ev.recall_at_k(ranked, relevant, 1) == 0.0
    assert ev.mrr(ranked, relevant) == pytest.approx(0.5)
    assert 0.0 < ev.ndcg_at_k(ranked, relevant, 3) <= 1.0


# --------------------------------------------------------------------------- #
# Follow-up expansion (web)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query,expanded",
    [
        ("and for 2024?", True),
        ("what about it in 2024", True),
        ("what is BM25", False),
        ("how does reranking work", False),
    ],
)
def test_maybe_expand(query, expanded):
    prev = "how many PTO days do employees get"
    out = maybe_expand(query, prev)
    assert (out != query) == expanded


def test_maybe_expand_no_prev_is_noop():
    assert maybe_expand("and for 2024?", None) == "and for 2024?"


# --------------------------------------------------------------------------- #
# End-to-end on the local backend (builds a small offline index once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    cfg = Config()
    cfg.backend = "local"
    cfg.index_path = str(tmp_path_factory.mktemp("vs") / "index.pkl")
    r = RAG(cfg)
    r.build_index()
    return r


def test_index_built(engine):
    assert len(engine.chunks) > 0
    assert engine.embeddings.shape[0] == len(engine.chunks)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_retrieve_returns_ranked_chunks(engine, mode):
    chunks = engine.retrieve("what is hybrid search", mode=mode, k=3)
    assert 0 < len(chunks) <= 3
    assert all({"id", "source", "section", "text", "rank"} <= c.keys() for c in chunks)
    assert [c["rank"] for c in chunks] == list(range(len(chunks)))


def test_answer_definition_leads_with_definition(engine):
    first_line = engine.answer("what is BM25")["answer"].splitlines()[0].lower()
    assert "bm25 is a sparse" in first_line


def test_answer_out_of_domain_refuses(engine):
    res = engine.answer("what is the weather on mars")
    assert res["insufficient"] is True
    assert res["sources"] == []


def test_answer_flags_contradiction_and_cites(engine):
    res = engine.answer("how many PTO days do employees get")
    assert res["contradictions"], "PTO 15 vs 20 should be flagged"
    assert res["sources"] and all("text" in s for s in res["sources"])


def test_out_of_domain_gate(engine):
    assert engine._out_of_domain("who won the world cup") is True
    assert engine._out_of_domain("what is reranking") is False


def test_relevant_chunk_ids_matches_gold(engine):
    row = {"relevant_sources": ["rag_concepts.md"], "gold": ["bm25 is a sparse lexical"]}
    ids = ev.relevant_chunk_ids(engine, row)
    assert ids and all(engine.chunks[i]["source"] == "rag_concepts.md" for i in ids)


def test_reranking_does_not_regress_chunk_recall(engine):
    """Guards the fusion-prior fix: the reranker must refine, not degrade, the
    first-stage order — hybrid_rerank recall must not fall below hybrid/bm25."""
    rows = ev.load_evalset(rag.ROOT / "evalset.jsonl")
    results, usable, unmatched = ev.evaluate_chunks(engine, rows, k=3)
    assert usable > 0 and unmatched == 0, "every gold label should match a chunk"
    rr = results["hybrid_rerank"]["recall"]
    assert rr >= results["hybrid"]["recall"] - 1e-9
    assert rr >= results["bm25"]["recall"] - 1e-9


def test_rerank_keeps_both_editions_for_contradiction(engine):
    """A contradiction query must retrieve chunks from both handbook editions."""
    chunks = engine.retrieve("how many PTO days do employees get", mode="hybrid_rerank", k=3)
    sources = {c["source"] for c in chunks}
    assert {"handbook_2023.md", "handbook_2024.md"} <= sources
