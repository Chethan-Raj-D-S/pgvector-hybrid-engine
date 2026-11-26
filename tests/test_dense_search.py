"""Tests for Dense Vector search and metadata filtering."""
import pytest
from hybrid_engine.embeddings import generate_deterministic_embedding
from hybrid_engine.models import Document
from hybrid_engine.search.dense import DenseVectorSearcher


def test_dense_vector_similarity():
    doc1 = Document(
        id="d1",
        title="Agentic AI State Machines",
        content="LangGraph state graphs enable cycle detection and checkpointing.",
        embedding=generate_deterministic_embedding("Agentic AI State Machines checkpointing"),
    )
    doc2 = Document(
        id="d2",
        title="Baking Sourdough Bread",
        content="Flour water salt and yeast fermentation temperature.",
        embedding=generate_deterministic_embedding("Baking Sourdough Bread yeast"),
    )

    searcher = DenseVectorSearcher([doc1, doc2])
    query_vec = generate_deterministic_embedding("State graphs in agentic systems")

    results = searcher.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0].document.id == "d1"
    assert results[0].score > results[1].score


def test_dense_metadata_filtering():
    doc1 = Document(
        id="d1",
        title="Public Doc",
        content="Accessible to everyone.",
        metadata={"tenant_id": "tenant_a"},
        embedding=generate_deterministic_embedding("Public Doc"),
    )
    doc2 = Document(
        id="d2",
        title="Private Doc",
        content="Restricted data.",
        metadata={"tenant_id": "tenant_b"},
        embedding=generate_deterministic_embedding("Private Doc"),
    )

    searcher = DenseVectorSearcher([doc1, doc2])
    query_vec = generate_deterministic_embedding("Doc")

    # Filter for tenant_a only
    results = searcher.search(query_vec, top_k=5, filter_metadata={"tenant_id": "tenant_a"})
    assert len(results) == 1
    assert results[0].document.id == "d1"
