import os
import torch
from huggingface_hub.hf_api import HfFolder
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from constants import Constants

# Set CUDA and environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# login to hugging face
HfFolder.save_token(Constants.HF_KEY)

# Load model directly
text_tokenizer = AutoTokenizer.from_pretrained(Constants.TEXT_MODEL_CHECKPOINT)
text_model = AutoModelForSeq2SeqLM.from_pretrained(Constants.TEXT_MODEL_CHECKPOINT)
classi_tokenizer = LongformerTokenizer.from_pretrained(Constants.CLASSIFICATIN_MODEL_CHECKPOINT)
classi_model = LongformerForSequenceClassification.from_pretrained(Constants.CLASSIFICATIN_MODEL_CHECKPOINT)

# Make sure to set the model to evaluation mode
text_model.eval()
text_model.to(device) 
classi_model.to(device) #type:ignore

# tokenize the input for classification
def preprocess_data(query:str, document:str) -> dict:
    """ Preprocess the input data and prepare it for the model. """
    text = f"Instruction: {Constants.CLASSI_PREFIX}query: {query} document: {document}"
    inputs = classi_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in inputs.items()}
    return inputs

# predict the label
def predict_relevance(query:str, document:str) -> float:
    """ Make a prediction for a single query-document pair. """
    model_inputs = preprocess_data(query, document)
    with torch.no_grad():
        outputs = classi_model(**model_inputs) #type:ignore
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[:, 1].item()  # Return the probability of "Relevant"

# preparte the input for justification generation
def prepare_input(query:str, document:str, relevancy:str) -> dict:
    input_text = f"Instruction: {Constants.TEXT_PREFIX}\nQuery: {query}\nDocument: {document}\nRelevancy: {relevancy}\nOutput:"
    inputs = text_tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input to GPU if using
    return inputs

# generate justification
def generate_text(input_ids:dict) -> str:
    # Generate outputs
    attention_mask = input_ids['attention_mask']
    outputs = text_model.generate(input_ids=input_ids['input_ids'], 
                             attention_mask=attention_mask,
                             max_length=1024, 
                            #  num_beams=5, 
                            #  no_repeat_ngram_size=2, 
                             early_stopping=True)
    
    # Decode the output to text
    output_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text


# user input
query:str = input("Query: ")
document:str = input("Document: ")

# Make a prediction
relevance_prob = predict_relevance(query, document)
relevancy = "Relevant" if relevance_prob > 0.5 else "Non-Relevant"

# Prepare input and generate text
input_ids = prepare_input(query, document, relevancy)
generated_text = generate_text(input_ids)

print(f"Relevance Probability: {relevance_prob:.2f}")
print(f"Relevancy: {relevancy}")
print("Generated Justification:", generated_text)
