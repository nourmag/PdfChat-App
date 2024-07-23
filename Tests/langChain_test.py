from langchain.llms import OpenAI

# Initialize a LangChain model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
print("LangChain LLM initialized successfully.")
