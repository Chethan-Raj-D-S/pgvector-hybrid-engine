"""Vector Embedding generator providing deterministic 384-dim semantic embeddings."""
from __future__ import annotations

import hashlib
import math
from typing import List


def generate_deterministic_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Generates a deterministic L2-normalized dense vector from text tokens.
    Provides reproducible high-dimensional embeddings for local tests and offline benchmarks.
    """
    words = text.lower().split()
    vec = [0.0] * dim

    for i, word in enumerate(words):
        h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
        for d in range(dim):
            # Deterministic projection per dimension
            weight = math.sin((h >> (d % 32)) + d * 0.1)
            vec[d] += weight / (1.0 + i * 0.05)

    # L2 Normalization
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [1.0 / math.sqrt(dim)] * dim
    return [round(x / norm, 6) for x in vec]
