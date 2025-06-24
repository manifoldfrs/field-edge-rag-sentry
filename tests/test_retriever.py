import pytest

from src.retriever import retrieve


def test_retriever():
    """Test the retriever with a few sample queries."""
    queries = [
        "What is mission-type orders?",
        "How to conduct defensive operations?",
        "Field fortification techniques",
    ]

    for query in queries:
        results = retrieve(query, k=3)

        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "score" in r
            assert isinstance(r["text"], str)
            assert isinstance(r["score"], float)
            assert len(r["text"]) > 0
            assert 0 <= r["score"] <= 1  # Cosine similarity is between 0 and 1
