import unittest
from src.embeddings.openai_embeddings import OpenAIEmbeddingsManager

class TestOpenAIEmbeddingsManager(unittest.TestCase):
    def setUp(self):
        self.manager = OpenAIEmbeddingsManager()

    def test_create_vector_store(self):
        texts = ["Hello world", "How are you?"]
        vector_store = self.manager.create_vector_store(texts)
        self.assertIsNotNone(vector_store)

    def test_embedding_function(self):
        embedding_function = self.manager._embedding_function()
        embeddings = embedding_function(["Hello world"])
        self.assertIsInstance(embeddings, list)
        self.assertGreater(len(embeddings), 0)

if __name__ == '__main__':
    unittest.main()
