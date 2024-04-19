import os
import torch
import nltk
import pandas as pd
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_metric

nltk.download('punkt')

# Set CUDA and environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_checkpoint = "satyanshu404/gpt2-finetuned-justification-v5"
prefix = "For the given query, document and document's relevancy, Provide a justification for the document's relevance to the given query."
load_dir = "Dataset/top_1000"
save_dir = f"Results/justification-results-{model_checkpoint.split('-')[-1]}.csv"

# Load model directly
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
rouge = load_metric("rouge")

# Make sure to set the model to evaluation mode
model.eval()
model.to(device) 

def prepare_input(query, document, relevancy):
    input_text = f"Instruction: {prefix}\nQuery: {query}\nDocument: {document}\nRelevancy: {relevancy}\nOutput:"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input to GPU if using
    return inputs


def generate_text(query, document, relevancy):
    input_ids = prepare_input(query, document, relevancy)
    # Generate outputs
    attention_mask = input_ids['attention_mask']
    outputs = model.generate(input_ids=input_ids['input_ids'], 
                             attention_mask=attention_mask,
                             max_length=1024, 
                            #  num_beams=5, 
                            #  no_repeat_ngram_size=2, 
                            #  early_stopping=True
                            )
    
    # Decode the output to text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def compute_df_rouge_scores(df, tokenizer, rouge):
    results = []
    for _, row in df.iterrows():
        # Encode the texts using the tokenizer
        pred_tokens = tokenizer.encode(row['generated_justification'], return_tensors="pt", padding="max_length", truncation=True).squeeze().tolist()
        label_tokens = tokenizer.encode(row['actual_justification'], return_tensors="pt", padding="max_length", truncation=True).squeeze().tolist()
        
        decoded_pred = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_tokens, skip_special_tokens=True)
        
        decoded_pred = "\n".join(nltk.sent_tokenize(decoded_pred.strip()))
        decoded_label = "\n".join(nltk.sent_tokenize(decoded_label.strip()))

        # Compute ROUGE scores
        result = rouge.compute(predictions=[decoded_pred], references=[decoded_label], use_stemmer=True)

        simplified_result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        results.append(simplified_result)
    
    return results

def average_rouge_scores(rouge_scores:list):
    # Initialize sums of ROUGE scores
    rouge_totals = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0
    }
    num_entries = len(df)
    
    # Sum up all ROUGE scores from each entry
    for scores in rouge_scores:
        rouge_totals['rouge1'] += scores['rouge1']
        rouge_totals['rouge2'] += scores['rouge2']
        rouge_totals['rougeL'] += scores['rougeL']
    
    # Calculate average by dividing total by number of entries
    average_scores = {key: total / num_entries for key, total in rouge_totals.items()}
    return average_scores


dataset = DatasetDict.load_from_disk(load_dir)
result = []

for data in dataset['test']:
    justification = generate_text(data["query"], data["document"], data['relevancy']) #type:ignore
    
    result.append([data['query'], data['justification'], justification, data["document"]]) #type:ignore

df = df = pd.DataFrame(result, columns=['query', 'actual_justification', 'generated_justification', 'doc'])
df.to_csv(save_dir)
print("all data saved")

df = pd.read_csv(save_dir)

# find the rouge score for each instance
rouge_scores = compute_df_rouge_scores(df, tokenizer, rouge)

# Compute the average ROUGE scores
average_scores = average_rouge_scores(rouge_scores)
print("Average ROUGE Scores:", average_scores)