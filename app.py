import streamlit as st
from dotenv import load_dotenv
import os
import logging

from src.pdf_handler import PDFHandler
from src.embedding_handler import EmbeddingHandler
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

# Ensure the OpenAI API key is set
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("The OpenAI API key must be set in the environment variables or in the .env file.")
    st.stop()  # Stop the app if the API key is missing

# Set the OpenAI API key in the environment
os.environ['OPENAI_API_KEY'] = openai_api_key
st.write(f"Using OpenAI API Key: {openai_api_key[:4]}...")

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
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
        ''')
        add_vertical_space(5)
        st.write('Made by Nour Maged & Naira Mohammed')

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_handler = PDFHandler(pdf)

        try:
            text = pdf_handler.extract_text()
            if not text.strip():
                st.error("The PDF does not contain any extractable text. Please upload a different PDF.")
                return

            chunks = PDFHandler.split_text(text)
            if not chunks:
                st.error("Failed to split the text into chunks. Please check the content of the PDF.")
                return
            
            # Accept user's questions/query
            query = st.text_input('Ask questions about your PDF:')
            if query:
                embedding_handler = EmbeddingHandler()
                vector_store_handler = VectorStoreHandler()

                # Generate embeddings
                embeddings = embedding_handler.generate_embeddings(chunks)
                if not embeddings:
                    st.error("Failed to generate embeddings. Please check the content of the PDF.")
                    return

                # Store embeddings
                vector_store = vector_store_handler.store_embeddings(chunks, embeddings)
                st.write('Embeddings Computation Completed.')

                # Process the query
                query_processor = QueryProcessor(vector_store)
                response = query_processor.process_query(query)
                st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
