from datasets import load_dataset
import pandas as pd
import os

print("Downloading PopQA dataset...")

# Download the dataset from Hugging Face
dataset = load_dataset("akariasai/PopQA")

# PopQA has only one split: test
df = dataset["test"].to_pandas()

# Create the raw data folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save the dataset as a CSV file
df.to_csv("data/raw/popqa.csv", index=False)

print("\nDataset downloaded successfully!")
print(f"Total Questions: {len(df)}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())
