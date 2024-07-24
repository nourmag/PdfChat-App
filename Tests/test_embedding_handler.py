import pytest
from src.embedding_handler import EmbeddingHandler

def test_generate_embeddings():
    test_chunks = ["This is a sample chunk."] * 5
    embedding_handler = EmbeddingHandler()
    embeddings = embedding_handler.generate_embeddings(test_chunks)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(test_chunks)
    assert all(isinstance(embedding, list) for embedding in embeddings)

def test_embedding_generation_error_handling():
    invalid_chunks = [None] * 5
    embedding_handler = EmbeddingHandler()
    try:
        embedding_handler.generate_embeddings(invalid_chunks)
    except Exception as e:
        assert isinstance(e, Exception)
