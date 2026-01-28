import pandas as pd
import numpy as np
import re
from tqdm import tqdm

input_csv = "antenna_dataset.csv"
output_csv = "antenna_long_format_fixed.csv"

df = pd.read_csv(input_csv)

def parse_numpy_array(s):
    s = s.replace("[", "").replace("]", "").replace("\n", " ")
    values = re.split(r"\s+", s.strip())
    return np.array([float(v) for v in values])

rows_out = []
row_counter = 0

print("Expanding frequency rows...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    freqs = parse_numpy_array(row["frequencies"])
    s11 = parse_numpy_array(row["s11"])

    for f, s in zip(freqs, s11):
        rows_out.append({
            "row_id": row_counter,
            "antenna_id": row["id"],
            "length": row["length"],
            "width": row["width"],
            "feed_y": row["feed_y"],
            "frequency": f,
            "s11": s
        })
        row_counter += 1

out_df = pd.DataFrame(rows_out)
out_df.to_csv(output_csv, index=False)

print(f"\nSaved expanded dataset to: {output_csv}")
print("Total rows:", len(out_df))
