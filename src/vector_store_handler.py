from langchain.vectorstores import Chroma
import chromadb
import logging
from transformers import AutoModel, AutoTokenizer
import torch

class VectorStoreHandler:
    def __init__(self):
        self.client = chromadb.Client()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def embedding_function(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
        return embeddings

    def store_embeddings(self, chunks):
        try:
            # Debugging: Print a sample chunk and the embedding function
            logging.debug(f"Sample chunk: {chunks[0]}")
            logging.debug(f"Embedding function: {self.embedding_function}")
            
            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding_function=self.embedding_function,  # Pass the embedding function
                client=self.client
            )
            logging.info("Embeddings stored in ChromaDB.")
            return vector_store
        except Exception as e:
            logging.error(f"Error storing embeddings in ChromaDB: {e}")
            raise

