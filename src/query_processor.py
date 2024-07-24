from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import logging
import streamlit as st

class QueryProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = OpenAI(model_name="gpt-3.5-turbo")

    def process_query(self, query):
        try:
            docs = self.vector_store.similarity_search(query=query, k=3)
            logging.debug(f"Documents retrieved: {docs}")

            if not docs:
                logging.error("No documents retrieved. Check your ChromaDB query and data indexing.")
                st.error("No relevant documents found. Try rephrasing your query or uploading a different PDF.")
                return "No relevant documents found."

            chain = load_qa_chain(llm=self.llm, chain_type='stuff')

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                logging.info(f"OpenAI callback: {cb}")
            return response
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            st.error(f"An error occurred while processing your query: {e}")
            raise
