# !pip install datasets
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import re

dataset = load_dataset("databricks/databricks-dolly-15k",split="train")
print(dataset)
df = pd.DataFrame(dataset)
filtered_df = df[
        (df['context'].apply(len) < 5000) &
        (df['response'].apply(len) < 500)
        ]

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

list_df = []

for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing rows"):
    row = {
        'context': clean_text(row['context']),
        'question': clean_text(row['instruction']),
        'response': clean_text(row['response']) 
    }
    list_df.append(row)

new_dataset = pd.DataFrame(list_df)
new_dataset.to_csv('/scratch/mm12799/we_com/datasets/dolly/dolly.csv', index=False)
print(len(new_dataset))
print(new_dataset.head())