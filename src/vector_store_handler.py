from langchain.vectorstores import Chroma
import chromadb
import logging

class VectorStoreHandler:
    def __init__(self):
        self.client = chromadb.Client()

    def store_embeddings(self, chunks, embeddings):
        try:
            class OpenAIEmbeddingFunction:
                def __init__(self, embeddings):
                    self.embeddings = embeddings

                def __call__(self, input):
                    return self.embeddings.embed_documents(input)

            embedding_function = OpenAIEmbeddingFunction(embeddings)
            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding_function=embedding_function,
                client=self.client
            )
            logging.info("Embeddings stored in ChromaDB.")
            return vector_store
        except Exception as e:
            logging.error(f"Error storing embeddings in ChromaDB: {e}")
            raise
