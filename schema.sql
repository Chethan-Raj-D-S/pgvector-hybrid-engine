-- ==============================================================================
-- PostgreSQL 16 + pgvector Production Schema: Hybrid Dense-Sparse Retrieval
-- Author: Chethan Raj D S (chethanraj.d.s@gmail.com)
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create knowledge documents table with dual representation
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Lexical Search: Generated tsvector column
    tsv_content TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) STORED,

    -- Semantic Search: 384-dimensional or 1536-dimensional vector column
    embedding VECTOR(384) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Create Indexes for High-Concurrency Production Performance

-- (A) Sparse Lexical Index (GIN)
CREATE INDEX IF NOT EXISTS idx_docs_tsv 
ON knowledge_documents USING GIN (tsv_content);

-- (B) Dense Vector Index (HNSW for high recall under dynamic inserts)
CREATE INDEX IF NOT EXISTS idx_docs_hnsw_embedding 
ON knowledge_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- (C) Relational Metadata Index for pre-filtering (Composite B-Tree)
CREATE INDEX IF NOT EXISTS idx_docs_metadata_tenant 
ON knowledge_documents ((metadata->>'tenant_id'), created_at DESC);

-- ==============================================================================
-- 4. Reciprocal Rank Fusion (RRF) Hybrid Search Query
-- Combines Dense and Sparse ranks using: RRF_Score = 1/(k + rank_dense) + 1/(k + rank_sparse)
-- ==============================================================================
CREATE OR REPLACE FUNCTION hybrid_search_rrf(
    query_text TEXT,
    query_embedding VECTOR(384),
    match_count INT DEFAULT 10,
    k_constant INT DEFAULT 60
)
RETURNS TABLE (
    id UUID,
    title VARCHAR(255),
    content TEXT,
    metadata JSONB,
    dense_rank BIGINT,
    sparse_rank BIGINT,
    rrf_score DOUBLE PRECISION
)
LANGUAGE sql STABLE
AS $$
WITH dense_matches AS (
    SELECT 
        kd.id,
        ROW_NUMBER() OVER (ORDER BY kd.embedding <=> query_embedding) AS rank
    FROM knowledge_documents kd
    ORDER BY kd.embedding <=> query_embedding
    LIMIT match_count * 2
),
sparse_matches AS (
    SELECT 
        kd.id,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(kd.tsv_content, plainto_tsquery('english', query_text)) DESC) AS rank
    FROM knowledge_documents kd
    WHERE kd.tsv_content @@ plainto_tsquery('english', query_text)
    ORDER BY ts_rank_cd(kd.tsv_content, plainto_tsquery('english', query_text)) DESC
    LIMIT match_count * 2
),
combined_ranks AS (
    SELECT 
        coalesce(d.id, s.id) AS id,
        d.rank AS dense_rank,
        s.rank AS sparse_rank,
        (coalesce(1.0 / (k_constant + d.rank), 0.0) + 
         coalesce(1.0 / (k_constant + s.rank), 0.0)) AS rrf_score
    FROM dense_matches d
    FULL OUTER JOIN sparse_matches s ON d.id = s.id
)
SELECT 
    kd.id,
    kd.title,
    kd.content,
    kd.metadata,
    c.dense_rank,
    c.sparse_rank,
    c.rrf_score
FROM combined_ranks c
JOIN knowledge_documents kd ON c.id = kd.id
ORDER BY c.rrf_score DESC
LIMIT match_count;
$$;
