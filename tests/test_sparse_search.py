"""Tests for BM25 Sparse Lexical search."""
import pytest
from hybrid_engine.models import Document
from hybrid_engine.search.sparse import BM25SparseSearcher


def test_bm25_exact_token_match():
    doc1 = Document(
        id="d1",
        title="PostgreSQL Errors",
        content="Encountered fatal code ERR_PG_40012 lock timeout on table partitions.",
    )
    doc2 = Document(
        id="d2",
        title="General Database Errors",
        content="Connection pool exhausted on port 5432.",
    )

    searcher = BM25SparseSearcher([doc1, doc2])
    results = searcher.search("ERR_PG_40012", top_k=2)

    assert len(results) == 1
    assert results[0].document.id == "d1"
    assert results[0].sparse_rank == 1


def test_bm25_metadata_filtering():
    doc1 = Document(
        id="d1",
        title="Config Guide",
        content="HNSW indexing tuning parameters.",
        metadata={"category": "database"},
    )
    doc2 = Document(
        id="d2",
        title="API Guide",
        content="HNSW endpoint retrieval parameters.",
        metadata={"category": "api"},
    )

    searcher = BM25SparseSearcher([doc1, doc2])
    results = searcher.search("HNSW", filter_metadata={"category": "database"})

    assert len(results) == 1
    assert results[0].document.id == "d1"
