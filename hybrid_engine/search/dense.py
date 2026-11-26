"""Dense vector similarity search mimicking pgvector HNSW cosine query execution."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from hybrid_engine.models import Document, SearchCandidate


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseVectorSearcher:
    """Executes dense vector search over documents with metadata pre-filtering."""

    def __init__(self, documents: List[Document]):
        self.documents = documents

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchCandidate]:
        scored: List[Tuple[Document, float]] = []

        for doc in self.documents:
            if not doc.embedding:
                continue

            # Relational metadata filtering (WHERE clause)
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            sim = cosine_similarity(query_embedding, doc.embedding)
            scored.append((doc, sim))

        # Sort descending by cosine similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc, score) in enumerate(scored[:top_k], start=1):
            results.append(SearchCandidate(document=doc, score=round(score, 4), dense_rank=rank))

        return results
