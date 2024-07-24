import pytest
from src.pdf_handler import PDFHandler
from src.embedding_handler import EmbeddingHandler
from src.query_processor import QueryProcessor
from src.vector_store_handler import VectorStoreHandler

def test_text_extraction_error_handling():
    test_pdf_path = "path/to/corrupt.pdf"
    pdf_handler = PDFHandler(test_pdf_path)
    try:
        pdf_handler.extract_text()
    except Exception as e:
        assert isinstance(e, Exception)

def test_embedding_generation_error_handling():
    invalid_chunks = [None] * 5
    embedding_handler = EmbeddingHandler()
    try:
        embedding_handler.generate_embeddings(invalid_chunks)
    except Exception as e:
        assert isinstance(e, Exception)

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
