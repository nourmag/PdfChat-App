import pytest
from app import main
from src.pdf_handler import PDFHandler
from src.embedding_handler import EmbeddingHandler
from src.vector_store_handler import VectorStoreHandler
from src.query_processor import QueryProcessor
import streamlit as st

def test_app_integration():
    # Setup code to simulate the app behavior
    test_pdf_path = "path/to/test.pdf"
    pdf_handler = PDFHandler(test_pdf_path)
    embedding_handler = EmbeddingHandler()
    vector_store_handler = VectorStoreHandler()
    query_processor = QueryProcessor(vector_store_handler)

    try:
        text = pdf_handler.extract_text()
        chunks = PDFHandler.split_text(text)
        embeddings = embedding_handler.generate_embeddings(chunks)
        vector_store = vector_store_handler.store_embeddings(chunks, embeddings)
        response = query_processor.process_query("What is in the PDF?")
        assert isinstance(response, str)
        assert len(response) > 0
    except Exception as e:
        assert False, f"Integration test failed: {e}"
