from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb

# Initialize the Chroma client
client = chromadb.Client()

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Example texts
texts = ["Hello world", "How are you?"]

# Define the embedding function
class OpenAIEmbeddingFunction:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, input):
        return self.embeddings.embed_documents(input)

embedding_function = OpenAIEmbeddingFunction(embeddings)

# Initialize the Chroma Vector Store
vector_store = Chroma.from_texts(
    texts=texts,
    embedding_function=embedding_function,
    client=client
)

# Perform a similarity search
query = "Hello"
docs = vector_store.similarity_search(query=query, k=1)
print(docs)
