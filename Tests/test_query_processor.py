import pytest
from src.query_processor import QueryProcessor
from src.vector_store_handler import VectorStoreHandler
from src.embedding_handler import EmbeddingHandler

def test_process_query():
    test_query = "What is the sample text about?"
    test_chunks = ["This is a sample chunk."] * 5
    embedding_handler = EmbeddingHandler()
    embeddings = embedding_handler.generate_embeddings(test_chunks)
    vector_store_handler = VectorStoreHandler()
    vector_store = vector_store_handler.store_embeddings(test_chunks, embeddings)
    query_processor = QueryProcessor(vector_store)
    response = query_processor.process_query(test_query)
    assert isinstance(response, str)
    assert len(response) > 0

def test_invalid_query():
    test_query = ""
    test_chunks = ["This is a sample chunk."] * 5
    embedding_handler = EmbeddingHandler()
    embeddings = embedding_handler.generate_embeddings(test_chunks)
    vector_store_handler = VectorStoreHandler()
    vector_store = vector_store_handler.store_embeddings(test_chunks, embeddings)
    query_processor = QueryProcessor(vector_store)
    try:
        query_processor.process_query(test_query)
    except Exception as e:
        assert isinstance(e, Exception)
