"""
Retrieval + contradiction evaluation harness.

Proves improvements instead of vibe-checking them. Reads evalset.jsonl, runs
every retrieval mode, and reports Recall@k, MRR, and nDCG@k at the *source*
level (a hit = a chunk from a labelled-relevant document appears in the top-k).

It also measures contradiction handling: for questions flagged with
"expected_contradiction", it checks whether the generated answer surfaced a
conflict.

Usage:
  python eval.py               # retrieval metrics for all modes
  python eval.py --contradict  # also run generation + contradiction scoring
  python eval.py --k 5         # override cut-off k

Requires Ollama running (embeddings for dense/hybrid; chat for --contradict).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rag import RAG, Config

ROOT = Path(__file__).parent
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def load_evalset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def retrieved_sources(chunks: list[dict]) -> list[str]:
    """Ordered unique source filenames from a ranked chunk list."""
    seen, ordered = set(), []
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            ordered.append(c["source"])
    return ordered


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for s in ranked[:k] if s in relevant)
    return hits / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, s in enumerate(ranked, 1):
        if s in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, s in enumerate(ranked[:k], 1) if s in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(rag: RAG, rows: list[dict], k: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for mode in MODES:
        hits1, recs, mrrs, ndcgs = [], [], [], []
        for row in rows:
            relevant = set(row["relevant_sources"])
            chunks = rag.retrieve(row["query"], mode=mode, k=max(k, rag.cfg.top_k))
            ranked = retrieved_sources(chunks)
            # Hit@1: is the very first retrieved document relevant? (sensitive to
            # ranking even when Recall@k is saturated on a small corpus)
            hits1.append(1.0 if ranked and ranked[0] in relevant else 0.0)
            recs.append(recall_at_k(ranked, relevant, k))
            mrrs.append(mrr(ranked, relevant))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))
        results[mode] = {
            "hit1": sum(hits1) / len(hits1),
            "recall": sum(recs) / len(recs),
            "mrr": sum(mrrs) / len(mrrs),
            "ndcg": sum(ndcgs) / len(ndcgs),
        }
    return results


def evaluate_contradictions(rag: RAG, rows: list[dict], mode: str) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        expected = bool(row.get("expected_contradiction", False))
        res = rag.answer(row["query"], mode=mode)
        flagged = len(res.get("contradictions", [])) > 0
        if expected and flagged:
            tp += 1
        elif expected and not flagged:
            fn += 1
        elif not expected and flagged:
            fp += 1
        else:
            tn += 1
        status = "OK " if expected == flagged else "MISS"
        print(f"  [{status}] expected={expected!s:5} flagged={flagged!s:5}  {row['query']}")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def print_table(results: dict[str, dict], k: int) -> None:
    print(f"\nRetrieval quality @ k={k}  (higher is better)")
    print(f"{'mode':<16}{'Hit@1':>10}{'Recall@k':>12}{'MRR':>10}{'nDCG@k':>10}")
    print("-" * 58)
    for mode in MODES:
        r = results[mode]
        print(f"{mode:<16}{r['hit1']:>10.3f}{r['recall']:>12.3f}{r['mrr']:>10.3f}{r['ndcg']:>10.3f}")

    base = results["dense"]["recall"]
    best = results["hybrid_rerank"]["recall"]
    if base > 0:
        lift = (best - base) / base * 100
        print("-" * 58)
        print(f"hybrid_rerank vs dense: Recall@k {base:.3f} -> {best:.3f}  ({lift:+.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="cut-off k for metrics")
    ap.add_argument("--contradict", action="store_true", help="also score contradiction handling")
    ap.add_argument("--evalset", default=str(ROOT / "evalset.jsonl"))
    args = ap.parse_args()

    rag = RAG(Config())
    print("Building / loading index...")
    rag.build_index()

    rows = load_evalset(Path(args.evalset))
    print(f"Loaded {len(rows)} eval queries.")

    results = evaluate_retrieval(rag, rows, args.k)
    print_table(results, args.k)

    if args.contradict:
        print("\nContradiction handling (mode: hybrid_rerank)")
        contra_rows = [r for r in rows if "expected_contradiction" in r]
        stats = evaluate_contradictions(rag, contra_rows, "hybrid_rerank")
        print(
            f"\n  precision={stats['precision']:.3f}  recall={stats['recall']:.3f}  "
            f"(tp={stats['tp']} fp={stats['fp']} fn={stats['fn']} tn={stats['tn']})"
        )


if __name__ == "__main__":
    main()
