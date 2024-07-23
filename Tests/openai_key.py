import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("The OpenAI API key must be set in the environment variables or in the .env file.")
    exit()

os.environ['OPENAI_API_KEY'] = openai_api_key

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, OpenAI!"}
        ]
    )
    print("API Key is valid and working!")
    print(response)
except openai.OpenAIError as e:
    print(f"Error: {e}")
