import pandas as pd
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.config import Settings

client = chromadb.Client()

class CustomEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = [generate_embeddings(text) for text in texts]
        # Tutaj możesz dostosować wymiary osadzeń
        # Na przykład, jeśli osadzenia mają wymiar 384, możesz je powielić, aby uzyskać wymiar 9600
        embeddings = [embedding.repeat(25) for embedding in embeddings]
        return embeddings

def add_data_to_collection(documents, embeddings, table_name):
    collection = client.get_or_create_collection(table_name, embedding_function=CustomEmbeddingFunction())
    collection.add(documents = documents, embeddings = embeddings, metadatas=[{"source": "student info"}], ids=['id_1'])

def query_collection(table_name, query):
    collection = client.get_collection(name=table_name)
    return collection.query(query_texts=[query])

