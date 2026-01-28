import pandas as pd

# Hugging Face parquet path
path = "hf://datasets/becklabash/rectangular-patch-antenna-freq-response/data/train-00000-of-00001.parquet"

print("Loading dataset...")
df = pd.read_parquet(path)

print(f"Dataset loaded. Shape: {df.shape}")

# Save to CSV
output_csv = "antenna_dataset.csv"
df.to_csv(output_csv, index=False)

print(f"Saved dataset to: {output_csv}")
