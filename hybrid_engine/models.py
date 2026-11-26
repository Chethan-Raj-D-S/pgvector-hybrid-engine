"""Pydantic data models for documents, search results, and retrieval configurations."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class SearchCandidate(BaseModel):
    document: Document
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RetrievalConfig(BaseModel):
    match_count: int = 10
    rrf_k: int = 60
    rerank_candidates: int = 20
    final_top_k: int = 3
    filter_metadata: Optional[Dict[str, Any]] = None
