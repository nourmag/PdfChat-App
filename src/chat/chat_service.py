from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

class ChatService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
        self.chain = load_qa_chain(llm=self.llm, chain_type='stuff')

    def answer_query(self, query):
        docs = self.vector_store.similarity_search(query=query, k=3)
        with get_openai_callback() as cb:
            response = self.chain.run(input_documents=docs, question=query)
            print(cb)
        return response
