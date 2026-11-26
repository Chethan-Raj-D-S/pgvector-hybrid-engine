"""Unified Hybrid Retrieval Pipeline coordinating Dense, Sparse, RRF, and Reranking."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from hybrid_engine.embeddings import generate_deterministic_embedding
from hybrid_engine.models import Document, RetrievalConfig, SearchCandidate
from hybrid_engine.search.dense import DenseVectorSearcher
from hybrid_engine.search.reranker import CrossEncoderReranker
from hybrid_engine.search.rrf import reciprocal_rank_fusion
from hybrid_engine.search.sparse import BM25SparseSearcher


class HybridSearchResult(BaseModel):
    query: str
    final_candidates: List[SearchCandidate]
    dense_candidates: List[SearchCandidate]
    sparse_candidates: List[SearchCandidate]
    fused_candidates: List[SearchCandidate]
    execution_time_ms: float


class HybridSearchPipeline:
    """Production-grade hybrid search orchestrator combining pgvector HNSW with BM25."""

    def __init__(self, documents: List[Document]):
        self.documents = list(documents)
        # Populate embeddings if missing
        for doc in self.documents:
            if not doc.embedding:
                doc.embedding = generate_deterministic_embedding(f"{doc.title} {doc.content}")

        self.dense_searcher = DenseVectorSearcher(self.documents)
        self.sparse_searcher = BM25SparseSearcher(self.documents)
        self.reranker = CrossEncoderReranker()

    def add_document(self, document: Document) -> None:
        if not document.embedding:
            document.embedding = generate_deterministic_embedding(f"{document.title} {document.content}")
        self.documents.append(document)
        self.dense_searcher = DenseVectorSearcher(self.documents)
        self.sparse_searcher = BM25SparseSearcher(self.documents)

    def search(self, query: str, config: Optional[RetrievalConfig] = None) -> HybridSearchResult:
        cfg = config or RetrievalConfig()
        start = time.perf_counter()

        # 1. Dense Semantic Search
        query_embedding = generate_deterministic_embedding(query)
        dense_results = self.dense_searcher.search(
            query_embedding=query_embedding,
            top_k=cfg.match_count,
            filter_metadata=cfg.filter_metadata,
        )

        # 2. Sparse Lexical Search (BM25)
        sparse_results = self.sparse_searcher.search(
            query=query,
            top_k=cfg.match_count,
            filter_metadata=cfg.filter_metadata,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=cfg.rrf_k,
            top_n=cfg.rerank_candidates,
        )

        # 4. Cross-Encoder Token Reranking
        final_results = self.reranker.rerank(
            query=query,
            candidates=fused_results,
            top_k=cfg.final_top_k,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return HybridSearchResult(
            query=query,
            final_candidates=final_results,
            dense_candidates=dense_results,
            sparse_candidates=sparse_results,
            fused_candidates=fused_results,
            execution_time_ms=round(elapsed_ms, 2),
        )
