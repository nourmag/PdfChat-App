import pytest
from src.vector_store_handler import VectorStoreHandler
from src.embedding_handler import EmbeddingHandler

def test_store_embeddings():
    test_chunks = ["This is a sample chunk."] * 5
    embedding_handler = EmbeddingHandler()
    embeddings = embedding_handler.generate_embeddings(test_chunks)
    vector_store_handler = VectorStoreHandler()
    vector_store = vector_store_handler.store_embeddings(test_chunks, embeddings)
    assert vector_store is not None
    assert hasattr(vector_store, 'similarity_search')
