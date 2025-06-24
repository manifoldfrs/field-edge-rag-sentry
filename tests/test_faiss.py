import faiss
import numpy as np
import pytest


def test_faiss_smoke():
    """Smoke test for FAISS - index random vectors and query nearest neighbors."""
    # 2 k × 384‑d vectors
    dim, n = 384, 2_000
    np.random.seed(0)
    data = np.random.random((n, dim)).astype("float32")

    index = faiss.IndexFlatL2(dim)
    index.add(data)

    # query first 5 vectors for 5 NNs
    D, I = index.search(data[:5], 5)

    # Verify results - first result should be the query vector itself
    for i in range(5):
        assert I[i, 0] == i, f"Expected first result for query {i} to be itself"

    assert I.shape == (5, 5), "Expected 5x5 result matrix"
    assert D.shape == (5, 5), "Expected 5x5 distance matrix"
