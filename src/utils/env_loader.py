from dotenv import load_dotenv
import os
import streamlit as st

def load_env_variables():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        st.error("The OpenAI API key must be set in the environment variables or in the .env file.")
        st.stop()
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Check if the openai package is installed
    try:
        import openai
    except ImportError:
        st.error("The openai package is not installed. Please install it with pip install openai.")
        st.stop()

    # Check if the chromadb package is installed
    try:
        import chromadb
    except ImportError:
        st.error("The chromadb package is not installed. Please install it with pip install chromadb.")
        st.stop()
