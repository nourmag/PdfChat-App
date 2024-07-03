import unittest
from langchain.llms import OpenAI

class TestLangChain(unittest.TestCase):
    def test_llm_initialization(self):
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
        self.assertIsNotNone(llm)

if __name__ == '__main__':
    unittest.main()
