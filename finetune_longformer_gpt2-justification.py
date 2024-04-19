import os
import torch
import numpy as np
import nltk
from transformers import LongformerModel, GPT2Config, GPT2LMHeadModel, EncoderDecoderModel, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import DatasetDict
from datasets import load_metric
from transformers import DataCollatorForSeq2Seq

nltk.download('punkt')

# Set CUDA devices and PyTorch environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration and settings
model_checkpoint_encoder = 'allenai/longformer-base-4096'
model_checkpoint_decoder = 'gpt2'
load_dir = "Dataset/top_1000"
version = "v5"
num_epochs = 200
learning_rate = 2e-5
batch_size = 1
prefix = "For the given query, document and document's relevancy, Provide a justification for the document's relevance to the given query."


# Initialize encoder and decoder
encoder = LongformerModel.from_pretrained(model_checkpoint_encoder)
decoder_config = GPT2Config.from_pretrained(model_checkpoint_decoder, add_cross_attention=True)
decoder = GPT2LMHeadModel.from_pretrained(model_checkpoint_decoder, config=decoder_config)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_decoder)
tokenizer.pad_token = tokenizer.eos_token

# Configure the EncoderDecoder model
model = EncoderDecoderModel(encoder=encoder, decoder=decoder) #type:ignore
model.config.decoder_start_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.tie_encoder_decoder_embeddings = True #type:ignore
model.to(device) #type:ignore

# Load dataset
dataset = DatasetDict.load_from_disk(load_dir)

# Preprocessing function
def preprocess_function(examples):
    inputs = [f"Instruction: {prefix}\n Query: {query}\nDocument: {doc}\nRelevancy: {relevancy}\nOutput:" for query, doc, relevancy in zip(examples['query'], examples["document"], examples["relevancy"])]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(examples['justification'], max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["decoder_input_ids"] = labels.input_ids
    model_inputs["labels"] = labels.input_ids
    return model_inputs

processed_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_checkpoint_decoder}-finetuned-justification-{version}",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    push_to_hub=True,
)

rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) #type:ignore
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return result

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"], #type:ignore
    eval_dataset=processed_dataset["validation"], #type:ignore
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics 
)


# Start training
trainer.train()

# Push to hub and save locally
trainer.push_to_hub()

model.save_pretrained(f"Results/longformer-gpt2-justification-generation-{version}")
tokenizer.save_pretrained(f"Results/longformer-gpt2-justification-generation-{version}")
