"""Tests for Reciprocal Rank Fusion (RRF) rank aggregation."""
import pytest
from hybrid_engine.models import Document, SearchCandidate
from hybrid_engine.search.rrf import reciprocal_rank_fusion


def test_rrf_combines_ranks():
    doc_a = Document(id="doc_a", title="Doc A", content="A")
    doc_b = Document(id="doc_b", title="Doc B", content="B")
    doc_c = Document(id="doc_c", title="Doc C", content="C")

    # Dense ranks: doc_a (#1), doc_b (#2)
    dense = [
        SearchCandidate(document=doc_a, score=0.95, dense_rank=1),
        SearchCandidate(document=doc_b, score=0.80, dense_rank=2),
    ]

    # Sparse ranks: doc_b (#1), doc_c (#2)
    sparse = [
        SearchCandidate(document=doc_b, score=12.5, sparse_rank=1),
        SearchCandidate(document=doc_c, score=8.1, sparse_rank=2),
    ]

    # RRF with k=60
    # doc_b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522 (Winner!)
    # doc_a: 1/(60+1) + 0        = 0.016393
    # doc_c: 0 + 1/(60+2)        = 0.016129

    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=3)

    assert len(fused) == 3
    assert fused[0].document.id == "doc_b"  # Strongest combined agreement
    assert fused[1].document.id == "doc_a"
    assert fused[2].document.id == "doc_c"
    assert fused[0].rrf_score > fused[1].rrf_score
