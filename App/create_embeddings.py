import pandas
import translators as ts
from transformers import BertTokenizer, BertModel
import torch
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

tokenizer = BertTokenizer.from_pretrained('sentence-transformers/paraphrase-TinyBERT-L6-v2')
model = BertModel.from_pretrained('sentence-transformers/paraphrase-TinyBERT-L6-v2')

# Load the model weights from the file via loading the state_dict
model.load_state_dict(torch.load('Model/Data/model_weights.pth'))


# Script that translates the text from polish to english and then tokenizes and embeds it
def translate_token_embed(text):
    translated_text = ts.translate_text(query_text=text, translator='google', from_language='pl', to_language='en')
    tokenized_text = tokenizer.encode_plus(translated_text, add_special_tokens=True, return_tensors='pt')
    embedded_text = model(**tokenized_text)
    return embedded_text.last_hidden_state.mean(dim=1).flatten().tolist()


# Custom embedding function
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = [translate_token_embed(doc) for doc in input]
        return embeddings


def get_embedding_function():
    return MyEmbeddingFunction()

def build_database():
    # create instance of chroma
    client = chromadb.PersistentClient(path="Model/Data/Chroma")

    # Deleting collection
    client.delete_collection(name="embeddings")

    # Creating collection
    collection = client.create_collection(name="embeddings", embedding_function=get_embedding_function())

    # Load data
    df = pandas.read_csv('Model/Data/unaugmented_data.csv')

    # Add document
    collection.add(ids=df['category'].tolist(), documents=df['text'].tolist())

if __name__ == '__main__':
    # create instance of chroma
    client = chromadb.PersistentClient(path="Model/Data/Chroma")

    # Deleting collection
    client.delete_collection(name="embeddings")

    # Creating collection
    collection = client.create_collection(name="embeddings", embedding_function=get_embedding_function())

    # Load data
    df = pandas.read_csv('Model/Data/unaugmented_data.csv')

    # Add document
    collection.add(ids=df['category'].tolist(), documents=df['text'].tolist())

    print(collection)