import pandas as pd

print("=" * 50)
print("Validating Processed PopQA Dataset")
print("=" * 50)

df = pd.read_csv("data/processed/popqa_processed.csv")

print(f"\nTotal Records : {len(df)}")
print(f"Total Columns : {len(df.columns)}")

print("\nChecking Duplicate IDs...")
duplicate_ids = df["id"].duplicated().sum()
print(f"Duplicate IDs : {duplicate_ids}")

print("\nChecking Empty Questions...")
empty_questions = (df["question"].str.strip() == "").sum()
print(f"Empty Questions : {empty_questions}")

print("\nChecking Empty Answers...")
empty_answers = (df["possible_answers"].str.strip() == "").sum()
print(f"Empty Answers : {empty_answers}")

print("\nChecking Missing Values...")
print(df.isnull().sum())

print("\nValidation Complete!")
