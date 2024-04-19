import os
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments
from datasets import DatasetDict
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import numpy as np

# Set CUDA and environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration and settings
model_checkpoint = 'allenai/longformer-base-4096'
load_dir = "Dataset/top_1000"
version = "v1"
num_epochs = 100
learning_rate = 2e-5
batch_size = 1
prefix = "For the given query and a document, find document's relevance to the given query."

# Load tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained(model_checkpoint)
model = LongformerForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.to(device) #type:ignore

# Load dataset
dataset = DatasetDict.load_from_disk(load_dir)

def preprocess_function(examples):
    # Concatenate query and document texts and encode them
    texts = [f"Instruction: {prefix}query: {query} document: {doc}" for query, doc in zip(examples['query'], examples['document'])]
    model_inputs = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")

    # Labels: Convert text labels to binary labels
    labels = [1 if label == 'Relevant' else 0 for label in examples['relevancy']]
    model_inputs['labels'] = torch.tensor(labels)
    return model_inputs

# Prepare datasets
processed_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = processed_dataset["train"]
eval_dataset = processed_dataset["validation"]

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"{model_checkpoint}-finetuned-relevance-classification-{version}",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Initialize the Trainer with the correct data collator
data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model, #type:ignore
    args=training_args,
    train_dataset=train_dataset, #type:ignore
    eval_dataset=eval_dataset, #type:ignore
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # Pass the metric computation function
)

# Start training
trainer.train()

# Push to hub and save locally
trainer.push_to_hub()
model.save_pretrained(f"Results/longformer-relevance-classification-{version}") #type:ignore
tokenizer.save_pretrained(f"Results/longformer-relevance-classification-{version}")

