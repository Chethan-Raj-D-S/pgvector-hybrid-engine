"""End-to-end integration tests for HybridSearchPipeline."""
import pytest
from hybrid_engine.models import Document, RetrievalConfig
from hybrid_engine.pipeline import HybridSearchPipeline


def test_hybrid_pipeline_end_to_end():
    docs = [
        Document(
            id="doc_1",
            title="Postgres HNSW Vectors",
            content="PostgreSQL 16 pgvector HNSW index provides 95%+ recall on dynamic updates.",
            metadata={"domain": "database"},
        ),
        Document(
            id="doc_2",
            title="Agentic State Machines",
            content="LangGraph state graphs for agent execution, cycle detection, and checkpointing.",
            metadata={"domain": "ai"},
        ),
        Document(
            id="doc_3",
            title="API Error Codes",
            content="Error code ERR_SEC_9901 denotes unauthorized BOLA access attempt.",
            metadata={"domain": "security"},
        ),
    ]

    pipeline = HybridSearchPipeline(docs)

    # 1. Test hybrid query
    res = pipeline.search("PostgreSQL 16 pgvector HNSW index", RetrievalConfig(final_top_k=2))

    assert len(res.final_candidates) > 0
    assert res.final_candidates[0].document.id == "doc_1"
    assert res.execution_time_ms > 0
    assert len(res.dense_candidates) > 0
    assert len(res.sparse_candidates) > 0

    # 2. Test exact token query
    res_err = pipeline.search("ERR_SEC_9901", RetrievalConfig(final_top_k=1))
    assert res_err.final_candidates[0].document.id == "doc_3"

    # 3. Test dynamic document addition
    pipeline.add_document(
        Document(
            id="doc_4",
            title="Financial Reconciliation",
            content="ABN AMRO Royal Bank of Canada Intellimatch double-entry reconciliation.",
            metadata={"domain": "fintech"},
        )
    )
    res_fin = pipeline.search("Intellimatch reconciliation", RetrievalConfig(final_top_k=1))
    assert res_fin.final_candidates[0].document.id == "doc_4"
