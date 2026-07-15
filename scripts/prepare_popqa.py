import os
import pandas as pd

print("=" * 50)
print("Preparing PopQA Dataset")
print("=" * 50)

input_path = "data/raw/popqa.csv"

df = pd.read_csv(input_path)

print(f"Loaded {len(df)} records.")


text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].str.strip()

print("Whitespace removed from text columns.")


print("\nMissing Values:")

print(df.isnull().sum())


os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/popqa_processed.csv"

df.to_csv(output_path, index=False)

print("\nProcessed dataset saved successfully!")

print(f"Location: {output_path}")
