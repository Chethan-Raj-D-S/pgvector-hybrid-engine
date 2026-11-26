"""Cross-Encoder Token-Level Reranker for scoring query-passage interactions."""
from __future__ import annotations

import re
from typing import List
from hybrid_engine.models import SearchCandidate


class CrossEncoderReranker:
    """
    Reranks candidate passages using fine-grained token-level cross-interaction scoring.
    Acts as the second-stage filter before passing context to LLM.
    """

    def rerank(self, query: str, candidates: List[SearchCandidate], top_k: int = 3) -> List[SearchCandidate]:
        if not candidates:
            return []

        query_tokens = [w.lower() for w in re.findall(r"\b\w+\b", query)]
        scored = []

        for cand in candidates:
            text = f"{cand.document.title} {cand.document.content}".lower()
            text_tokens = re.findall(r"\b\w+\b", text)

            # 1. Exact phrase matching bonus
            phrase_bonus = 2.0 if query.lower() in text else 0.0

            # 2. Token overlap ratio (query coverage)
            matched_query_tokens = sum(1 for q in query_tokens if q in text_tokens)
            coverage = matched_query_tokens / len(query_tokens) if query_tokens else 0.0

            # 3. Base RRF contribution
            base_rrf = cand.rrf_score or 0.0

            # Combined cross-attention simulation score
            rerank_score = (coverage * 3.0) + phrase_bonus + (base_rrf * 10.0)

            cand.rerank_score = round(rerank_score, 4)
            scored.append((cand, rerank_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
