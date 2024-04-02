from transformers import AutoTokenizer, AutoModel
import torch
import config.data.data_base_config_chroma as db

# Sentences we want sentence embeddings for
sentences = ["Nestled among towering pines, the tranquil lake reflects the clear blue sky, offering a serene escape from the bustling world beyond. Its shores, peppered with cozy cabins, invite nature lovers to revel in peaceful solitude."]

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

# Add data to collection
db.add_data_to_collection(sentences, "test_b", [{"chapter": "1"}], ["id1"], embeddings=embeddings)

# Query collection
print(db.query_collection("test_a", "among", 1))
