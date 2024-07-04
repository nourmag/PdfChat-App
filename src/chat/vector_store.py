from langchain.vectorstores import Chroma
import chromadb
import pickle

class VectorStore:
    def __init__(self, texts, embeddings_manager):
        self.texts = texts
        self.embeddings_manager = embeddings_manager
        self.vector_store = None

    def create_vector_store(self):
        client = chromadb.Client()
        embedding_function = self.embeddings_manager.get_embedding_function()
        self.vector_store = Chroma.from_texts(
            texts=self.texts,
            embedding_function=embedding_function,
            client=client
        )

        # Save the vector store
        store_name = "vector_store"
        with open(f'{store_name}.pkl', 'wb') as f:
            pickle.dump(self.vector_store, f)
        
        return self.vector_store

    def load_vector_store(self, store_name):
        with open(f'{store_name}.pkl', 'rb') as f:
            self.vector_store = pickle.load(f)
        
        return self.vector_store
