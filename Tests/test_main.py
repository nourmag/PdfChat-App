import pytest
import os
from dotenv import load_dotenv

def test_environment_variables():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    assert openai_api_key is not None
    assert len(openai_api_key) > 0

def test_missing_packages():
    try:
        import openai
        import chromadb
    except ImportError as e:
        assert False, f"Required package is not installed: {e}"
