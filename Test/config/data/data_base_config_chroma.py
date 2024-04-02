import pandas as pd
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.config import Settings

client = chromadb.PersistentClient(path="data/chroma_db")


def add_data_to_collection(document, table_name, metadata, ida, embeddings):
    collection = client.get_or_create_collection(name=table_name)
    collection.add(documents=document, metadatas=metadata, ids=ida, embeddings=embeddings)


def query_collection(table_name, query, n_results):
    collection = client.get_or_create_collection(name=table_name)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return results



