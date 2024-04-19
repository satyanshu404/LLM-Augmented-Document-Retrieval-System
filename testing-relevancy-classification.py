import os
from matplotlib.scale import scale_factory
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments
from datasets import DatasetDict
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Set CUDA and environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_checkpoint = "satyanshu404/relevance-classification-v1"
prefix = "For the given query and a document, find document's relevance to the given query."
load_dir = "Dataset/top_1000"
save_dir = f"Results/classification-results-{model_checkpoint.split('-')[-1]}.csv"

# Load tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained(model_checkpoint)
model = LongformerForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.to(device) #type:ignore

def preprocess_data(query, document):
    """ Preprocess the input data and prepare it for the model. """
    text = f"Instruction: {prefix}\nQuery: {query}\nDocument: {document}"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs

def predict_relevance(query, document):
    """ Make a prediction for a single query-document pair. """
    model_inputs = preprocess_data(query, document)
    with torch.no_grad():
        outputs = model(**model_inputs) #type:ignore
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[:, 1].item()  # Return the probability of "Relevant"


dataset = DatasetDict.load_from_disk(load_dir)
result = []


for data in dataset['test']:
    relevance_prob = predict_relevance(data["query"], data["document"]) #type:ignore
    relevancy = "Relevant" if relevance_prob >= 0.5 else "Non-Relevant" 

    result.append([data['query'], data['relevancy'], relevancy, data["document"]]) #type:ignore


df = df = pd.DataFrame(result, columns=['query', 'actual_relevancy', 'generated_relevancy', 'doc'])
df.to_csv(save_dir)
print("all data saved")

df = pd.read_csv(save_dir)

# Construct the confusion matrix
conf_matrix = confusion_matrix(df['actual_relevancy'], df['generated_relevancy'], labels=["Relevant", "Non-Relevant"])

# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Generate classification report
report = classification_report(df['actual_relevancy'], df['generated_relevancy'], target_names=["Relevant", "Non-Relevant"])

# Print the classification report
print("Classification Report:\n", report)