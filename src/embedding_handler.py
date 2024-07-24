from langchain.embeddings.openai import OpenAIEmbeddings
import logging

class EmbeddingHandler:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def generate_embeddings(self, chunks):
        try:
            embedded_chunks = self.embeddings.embed_documents(chunks)
            logging.info(f"Generated embeddings for {len(chunks)} chunks.")
            logging.debug(f"Embeddings: {embedded_chunks[:5]}")  # Log the first 5 embeddings for verification
            return embedded_chunks
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise
