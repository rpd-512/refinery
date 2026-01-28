import pandas as pd
import numpy as np
import re

INPUT_CSV = "antenna_dataset.csv"
OUTPUT_CSV = "antenna_dataset_short.csv"
TARGET_POINTS = 200

df = pd.read_csv(INPUT_CSV)

compressed_rows = []

def parse_array(text):
    # Remove brackets
    text = text.strip().strip("[]")
    # Split by whitespace or comma
    values = re.split(r"[\s,]+", text)
    return np.array([float(v) for v in values if v])

for _, row in df.iterrows():
    freqs = parse_array(row["frequencies"])
    s11 = parse_array(row["s11"])

    if len(freqs) < 2:
        continue

    # Sort if needed
    order = np.argsort(freqs)
    freqs = freqs[order]
    s11 = s11[order]

    # New frequency grid
    new_freqs = np.linspace(freqs.min(), freqs.max(), TARGET_POINTS)

    # Interpolate
    new_s11 = np.interp(new_freqs, freqs, s11)

    compressed_rows.append({
        "id": row["id"],
        "length": row["length"],
        "width": row["width"],
        "feed_y": row["feed_y"],
        "frequencies": new_freqs.tolist(),
        "s11": new_s11.tolist()
    })

out_df = pd.DataFrame(compressed_rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved compressed dataset → {OUTPUT_CSV}")
print(f"Each antenna now has {TARGET_POINTS} frequency–S11 pairs.")
