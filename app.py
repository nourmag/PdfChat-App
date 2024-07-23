import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import logging

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

# Check if the `openai` package is installed
try:
    import openai
except ImportError:
    st.error("The `openai` package is not installed. Please install it with `pip install openai`.")
    st.stop()  # Stop the app if the package is missing

# Check if the `chromadb` package is installed
try:
    import chromadb
except ImportError:
    st.error("The `chromadb` package is not installed. Please install it with `pip install chromadb`.")
    st.stop()  # Stop the app if the package is missing

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logging.info("Text extraction completed.")
        logging.debug(f"Extracted text: {text[:500]}")  # Log the first 500 characters of extracted text for verification
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise

# Function to split text into chunks
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

# Function to generate embeddings for text chunks
def generate_embeddings(chunks):
    try:
        embeddings = OpenAIEmbeddings()
        embedded_chunks = embeddings.embed_documents(chunks)
        logging.info(f"Generated embeddings for {len(chunks)} chunks.")
        logging.debug(f"Embeddings: {embedded_chunks[:5]}")  # Log the first 5 embeddings for verification
        return embedded_chunks
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

# Function to store embeddings in ChromaDB
def store_embeddings(chunks, embeddings):
    try:
        client = chromadb.Client()
        
        class OpenAIEmbeddingFunction:
            def __init__(self, embeddings):
                self.embeddings = embeddings

            def __call__(self, input):
                return self.embeddings.embed_documents(input)

        embedding_function = OpenAIEmbeddingFunction(embeddings)
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding_function=embedding_function,
            client=client
        )
        logging.info("Embeddings stored in ChromaDB.")
        return vector_store
    except Exception as e:
        logging.error(f"Error storing embeddings in ChromaDB: {e}")
        raise

# Function to process user query and retrieve relevant information
def process_query(query, vector_store):
    try:
        # Perform the similarity search
        docs = vector_store.similarity_search(query=query, k=3)
        logging.debug(f"Documents retrieved: {docs}")
        
        if not docs:
            logging.error("No documents retrieved. Check your ChromaDB query and data indexing.")
            st.error("No relevant documents found. Try rephrasing your query or uploading a different PDF.")
            return "No relevant documents found."

        # Correct function for creating a chat completion
        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type='stuff')

        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            logging.info(f"OpenAI callback: {cb}")
        return response
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        st.error(f"An error occurred while processing your query: {e}")
        raise

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
        try:
            text = extract_text_from_pdf(pdf)
            if not text.strip():
                st.error("The PDF does not contain any extractable text. Please upload a different PDF.")
                return
            
            chunks = split_text(text)
            if not chunks:
                st.error("Failed to split the text into chunks. Please check the content of the PDF.")
                return
            
            embeddings = generate_embeddings(chunks)
            if not embeddings:
                st.error("Failed to generate embeddings. Please check the content of the PDF.")
                return
            
            vector_store = store_embeddings(chunks, embeddings)
            st.write('Embeddings Computation Completed.')

            # Accept user's questions/query
            query = st.text_input('Ask questions about your PDF:')
            if query:
                response = process_query(query, vector_store)
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
