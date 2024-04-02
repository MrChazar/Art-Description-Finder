from transformers import AutoTokenizer, AutoModel
import torch
import config.data.data_base_config_chroma as db

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers-testing/stsb-bert-tiny-safetensors')
model = AutoModel.from_pretrained('sentence-transformers-testing/stsb-bert-tiny-safetensors')

# Tokenize prompts
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_inputs)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Calculate mean pooling
embeddings = mean_pooling(model_output, encoded_inputs['attention_mask'])

# Convert embeddings to list
embeddings = embeddings.tolist()

# Save embeddings to database
db.add_data_to_collection(sentences, embeddings, "test_table_7")

# Query database
print(db.query_collection("test_table_7", sentences[0]))
