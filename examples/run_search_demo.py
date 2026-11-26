"""Interactive live demonstration of PgVector-HybridEngine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure package root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_engine.models import Document, RetrievalConfig
from hybrid_engine.pipeline import HybridSearchPipeline


def main():
    print("=" * 80)
    print("🚀 PgVector-HybridEngine: Live Search & Retrieval Demonstration")
    print("=" * 80)

    # Load benchmark dataset
    data_path = Path(__file__).resolve().parent.parent / "benchmarks" / "dataset.json"
    with open(data_path, "r", encoding="utf-8") as f:
        docs = [Document(**item) for item in json.load(f)]

    print(f"\nIndexed {len(docs)} knowledge documents with dual dense (HNSW) and sparse (BM25) representation.\n")
    pipeline = HybridSearchPipeline(docs)

    sample_queries = [
        "How do agents recover from failures and resume state?",
        "ERR_PG_40012 lock timeout",
        "PostgreSQL 16 pgvector HNSW m=16 parameters",
        "ABN AMRO Royal Bank of Canada reconciliation",
    ]

    for q in sample_queries:
        print("-" * 80)
        print(f"🔍 QUERY: \"{q}\"")
        res = pipeline.search(q, RetrievalConfig(final_top_k=2))
        print(f"⏱️  Execution Time: {res.execution_time_ms} ms")
        print("\nTop Retrieved Contexts (Hybrid RRF + Cross-Encoder Re-ranked):")
        for i, cand in enumerate(res.final_candidates, start=1):
            doc = cand.document
            print(f"  [{i}] Title: {doc.title}")
            print(f"      ID: {doc.id} | RRF Score: {cand.rrf_score} | Re-rank Score: {cand.rerank_score}")
            print(f"      Dense Rank: {cand.dense_rank} | Sparse Rank: {cand.sparse_rank}")
            print(f"      Content: {doc.content[:110]}...")

    print("\n" + "=" * 80)
    print("✅ Hybrid Engine verification finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
