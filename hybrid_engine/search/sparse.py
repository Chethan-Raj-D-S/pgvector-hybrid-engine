"""BM25 Sparse Lexical Search mimicking PostgreSQL tsvector and ts_rank_cd behavior."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from hybrid_engine.models import Document, SearchCandidate


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\b[A-Za-z0-9_#-]+\b", text)]


class BM25SparseSearcher:
    """Computes BM25 full-text lexical ranking over document collection."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.corpus_size = len(documents)

        # Build inverted index & document length statistics
        self.doc_lengths: Dict[str, int] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: Dict[str, Counter[str]] = {}

        total_length = 0
        for doc in documents:
            tokens = tokenize(f"{doc.title} {doc.content}")
            self.doc_lengths[doc.id] = len(tokens)
            total_length += len(tokens)
            tf = Counter(tokens)
            self.term_freqs[doc.id] = tf
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = (total_length / self.corpus_size) if self.corpus_size > 0 else 1.0

    def _idf(self, term: str) -> float:
        n = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.corpus_size - n + 0.5) / (n + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchCandidate]:
        query_terms = tokenize(query)
        scores: List[Tuple[Document, float]] = []

        for doc in self.documents:
            # Metadata pre-filtering
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            doc_len = self.doc_lengths.get(doc.id, 0)
            tf_map = self.term_freqs.get(doc.id, Counter())
            doc_score = 0.0

            for term in query_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self._idf(term)
                # BM25 formula
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                doc_score += idf * (num / denom)

            if doc_score > 0.0:
                scores.append((doc, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc, score) in enumerate(scores[:top_k], start=1):
            results.append(SearchCandidate(document=doc, score=round(score, 4), sparse_rank=rank))

        return results
