import unittest
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb

class TestChromaDB(unittest.TestCase):
    def setUp(self):
        self.client = chromadb.Client()
        self.embeddings = OpenAIEmbeddings()
        self.texts = ["Hello world", "How are you?"]

    def test_chroma_vector_store(self):
        embedding_function = lambda texts: self.embeddings.embed_documents(texts)
        vector_store = Chroma.from_texts(
            texts=self.texts,
            embedding_function=embedding_function,
            client=self.client
        )
        self.assertIsNotNone(vector_store)

    def test_similarity_search(self):
        embedding_function = lambda texts: self.embeddings.embed_documents(texts)
        vector_store = Chroma.from_texts(
            texts=self.texts,
            embedding_function=embedding_function,
            client=self.client
        )
        query = "Hello"
        docs = vector_store.similarity_search(query=query, k=1)
        self.assertGreater(len(docs), 0)

if __name__ == '__main__':
    unittest.main()
