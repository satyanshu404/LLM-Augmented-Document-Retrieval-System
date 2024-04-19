import os
import pandas as pd
from datasets import Dataset, DatasetDict


file_path = "Dataset/triples_with_justifications.tsv.gz"
save_dir = "top_1000"
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# load the dataset
df = pd.read_csv(file_path,sep='\t', encoding='utf8')
print(df.head())

data = []

for index, row in df.iterrows():
    if str(row['rdocJusti']).lower() != 'nan':
        data.append([row['query'], row['rdoc'], 'Relevant', row['rdocJusti']])
    if str(row['nrdocJusti']).lower() != 'nan':
        data.append([row['query'], row['ndoc'], 'Non-Relevant', row['nrdocJusti']])
        
# create new dataframe 
df = pd.DataFrame(data, columns=['query', 'document', 'relevancy', 'justification'])


# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Shuffle the dataset
dataset = dataset.shuffle()

# Split dataset into train, validation, and test sets
train_dataset = dataset.select(range(int(len(dataset) * train_ratio)))
remaining = dataset.select(range(int(len(dataset) * train_ratio), len(dataset)))
val_dataset = remaining.select(range(int(len(remaining) * (val_ratio / (val_ratio + test_ratio)))))
test_dataset = remaining.select(range(int(len(remaining) * (val_ratio / (val_ratio + test_ratio))), len(remaining)))

# Create DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print(dataset_dict)

# Save the DatasetDict to disk
dataset_dict.save_to_disk(save_dir)

# Check if the directory exists, and print a message indicating where the dataset is saved
if os.path.exists(save_dir):
    print(f"Dataset saved successfully at: {save_dir}")
else:
    print("Error: Dataset could not be saved.")