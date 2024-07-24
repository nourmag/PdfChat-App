import pytest
from src.pdf_handler import PDFHandler

def test_extract_text_from_pdf():
    test_pdf_path = "path/to/test.pdf"
    pdf_handler = PDFHandler(test_pdf_path)
    extracted_text = pdf_handler.extract_text()
    assert isinstance(extracted_text, str)
    assert len(extracted_text) > 0

def test_split_text():
    test_text = "This is a sample text " * 100
    chunks = PDFHandler.split_text(test_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_empty_pdf():
    test_pdf_path = "path/to/empty.pdf"
    pdf_handler = PDFHandler(test_pdf_path)
    extracted_text = pdf_handler.extract_text()
    assert extracted_text == ""

def test_text_extraction_error_handling():
    test_pdf_path = "path/to/corrupt.pdf"
    pdf_handler = PDFHandler(test_pdf_path)
    try:
        pdf_handler.extract_text()
    except Exception as e:
        assert isinstance(e, Exception)

def test_large_pdf():
    test_pdf_path = "path/to/large.pdf"
    pdf_handler = PDFHandler(test_pdf_path)
    extracted_text = pdf_handler.extract_text()
    assert isinstance(extracted_text, str)
    assert len(extracted_text) > 0

    chunks = PDFHandler.split_text(extracted_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_split_empty_text():
    empty_text = ""
    chunks = PDFHandler.split_text(empty_text)
    assert chunks == []

def test_split_short_text():
    short_text = "This is a short text."
    chunks = PDFHandler.split_text(short_text)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0] == short_text

