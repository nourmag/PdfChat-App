import unittest
from unittest.mock import MagicMock
from src.chat.chat_service import ChatService
from src.chat.vector_store import VectorStore

class TestChatService(unittest.TestCase):
    def setUp(self):
        self.vector_store = MagicMock(spec=VectorStore)
        self.chat_service = ChatService(self.vector_store)

    def test_answer_query(self):
        self.vector_store.similarity_search.return_value = ["doc1", "doc2", "doc3"]
        query = "What is the main topic?"
        response = self.chat_service.answer_query(query)
        self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main()
