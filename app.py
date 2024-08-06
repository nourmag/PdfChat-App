import streamlit as st
from dotenv import load_dotenv
import os
import logging

from src.pdf_handler import PDFHandler
from src.vector_store_handler import VectorStoreHandler
from src.query_processor import QueryProcessor

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Custom function to add vertical space
def add_vertical_space(lines):
    for _ in range(lines):
        st.write("")

# Load environment variables from .env file
load_dotenv()

# Main Streamlit app function
def main():
    st.header("Chat with PDF 💬")

    # Sidebar information
    with st.sidebar:
        st.title('Pdf-Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Hugging Face](https://huggingface.co/) API
        ''')
        add_vertical_space(5)
        st.write('Made by Nour Maged & Naira Mohammed')

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_handler = PDFHandler(pdf)
        vector_store_handler = VectorStoreHandler()

        try:
            text = pdf_handler.extract_text()
            if not text.strip():
                st.error("The PDF does not contain any extractable text. Please upload a different PDF.")
                return

            chunks = PDFHandler.split_text(text)
            if not chunks:
                st.error("Failed to split the text into chunks. Please check the content of the PDF.")
                return

            vector_store = vector_store_handler.store_embeddings(chunks)
            st.write('Embeddings Computation Completed.')

            # Accept user's questions/query
            query = st.text_input('Ask questions about your PDF:')
            if query:
                query_processor = QueryProcessor(vector_store)
                response = query_processor.process_query(query)
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
