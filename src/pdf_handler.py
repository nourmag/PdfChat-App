import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class PDFHandler:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        try:
            reader = PdfReader(self.pdf_path)
            text = "".join([page.extract_text() for page in reader.pages])
            logging.info("Text extraction completed.")
            logging.debug(f"Extracted text: {text[:500]}")  # Log the first 500 characters
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            raise

    @staticmethod
    def split_text(text):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            logging.info(f"Text split into {len(chunks)} chunks.")
            logging.debug(f"Chunks: {chunks[:5]}")  # Log the first 5 chunks for verification
            return chunks
        except Exception as e:
            logging.error(f"Error splitting text: {e}")
            raise
