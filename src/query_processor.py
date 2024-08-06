from transformers import pipeline
from langchain.chains.question_answering import load_qa_chain
import logging
import streamlit as st

class QueryProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    def process_query(self, query):
        try:
            docs = self.vector_store.similarity_search(query=query, k=3)
            logging.debug(f"Documents retrieved: {docs}")

            if not docs:
                logging.error("No documents retrieved. Check your ChromaDB query and data indexing.")
                st.error("No relevant documents found. Try rephrasing your query or uploading a different PDF.")
                return "No relevant documents found."

            # Generate context from the retrieved documents
            context = " ".join([doc["text"] for doc in docs])

            # Create QA input
            qa_input = {
                "question": query,
                "context": context
            }

            response = self.qa_pipeline(qa_input)
            return response['answer']
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            st.error(f"An error occurred while processing your query: {e}")
            raise
