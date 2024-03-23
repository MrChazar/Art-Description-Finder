from torch import cuda
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import prompt as pr
import data.data_base_config as db


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence']

# Prompts
prompts = []
for a in sentences:
    prompts.append(pr.create_prompt_func(a))

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers-testing/stsb-bert-tiny-safetensors')
model = AutoModel.from_pretrained('sentence-transformers-testing/stsb-bert-tiny-safetensors')

# Tokenize prompts
encoded_inputs = []
for a in prompts:
    encoded_inputs.append(tokenizer(sentences, padding=True, truncation=True, return_tensors='pt'))

# Compute token embeddings
feature_extraction = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

# Extract embeddings
embeddings = feature_extraction(prompts)

# Save embeddings to database
for embedding in embeddings:
    print(f"Embedding: {embedding}")
    db.save_list_to_database(embedding, 'embeddings')









