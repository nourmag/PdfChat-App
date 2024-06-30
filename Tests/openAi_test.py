import openai

# Ensure that OpenAI API key is set in the environment
openai.api_key = 'your_openai_api_key_here'

# Test OpenAI API call
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Say this is a test",
    max_tokens=5
)

print(response.choices[0].text.strip())
