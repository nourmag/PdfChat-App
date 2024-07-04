from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import pickle

class OpenAIEmbeddingsManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def create_vector_store(self, texts):
        client = chromadb.Client()
        embedding_function = self._embedding_function()
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding_function=embedding_function,
            client=client
        )

        # Save the vector store
        store_name = "vector_store"
        with open(f'{store_name}.pkl', 'wb') as f:
            pickle.dump(vector_store, f)
        return vector_store

    def _embedding_function(self):
        class EmbeddingFunction:
            def __init__(self, embeddings):
                self.embeddings = embeddings

            def __call__(self, input):
                return self.embeddings.embed_documents(input)

        return EmbeddingFunction(self.embeddings)
