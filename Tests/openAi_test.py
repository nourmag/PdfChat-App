import unittest
import openai
import os

class TestOpenAI(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key

    def test_openai_api_call(self):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Say this is a test",
            max_tokens=5
        )
        self.assertIsNotNone(response.choices[0].text.strip())

if __name__ == '__main__':
    unittest.main()

