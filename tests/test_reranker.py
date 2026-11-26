"""Tests for Cross-Encoder token re-ranking."""
import pytest
from hybrid_engine.models import Document, SearchCandidate
from hybrid_engine.search.reranker import CrossEncoderReranker


def test_reranker_boosts_exact_phrase_matches():
    doc1 = Document(
        id="d1",
        title="PostgreSQL 16",
        content="Guide on HNSW index parameter tuning with m=16.",
    )
    doc2 = Document(
        id="d2",
        title="General Guide",
        content="General discussion mentioning indexing and tuning.",
    )

    candidates = [
        SearchCandidate(document=doc2, score=0.03, rrf_score=0.03),
        SearchCandidate(document=doc1, score=0.02, rrf_score=0.02),
    ]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("HNSW index parameter tuning", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].document.id == "d1"  # doc1 gets exact phrase match bonus
    assert reranked[0].rerank_score > reranked[1].rerank_score
