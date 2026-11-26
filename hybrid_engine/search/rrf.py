"""Reciprocal Rank Fusion (RRF) algorithm combining dense and sparse rankings."""
from __future__ import annotations

from typing import Dict, List, Optional
from hybrid_engine.models import Document, SearchCandidate


def reciprocal_rank_fusion(
    dense_results: List[SearchCandidate],
    sparse_results: List[SearchCandidate],
    k: int = 60,
    top_n: int = 10,
) -> List[SearchCandidate]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion:
    RRF_Score(d) = Σ [ 1 / (k + rank_i(d)) ]
    
    This eliminates score scaling disparities between cosine similarity [0, 1]
    and unbounded BM25 scores.
    """
    doc_map: Dict[str, Document] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}
    rrf_scores: Dict[str, float] = {}

    # Process Dense ranks
    for item in dense_results:
        doc_id = item.document.id
        doc_map[doc_id] = item.document
        dense_ranks[doc_id] = item.dense_rank or 999
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (item.dense_rank or 999)))

    # Process Sparse ranks
    for item in sparse_results:
        doc_id = item.document.id
        doc_map[doc_id] = item.document
        sparse_ranks[doc_id] = item.sparse_rank or 999
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (item.sparse_rank or 999)))

    # Sort descending by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_candidates: List[SearchCandidate] = []
    for doc_id, score in sorted_items[:top_n]:
        fused_candidates.append(
            SearchCandidate(
                document=doc_map[doc_id],
                score=round(score, 6),
                dense_rank=dense_ranks.get(doc_id),
                sparse_rank=sparse_ranks.get(doc_id),
                rrf_score=round(score, 6),
            )
        )

    return fused_candidates
