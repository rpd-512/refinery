import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

csv_path = "antenna_dataset.csv"
df = pd.read_csv(csv_path)

# Pick random row
row = df.sample(1).iloc[0]

length = row["length"]
width = row["width"]
feed_y = row["feed_y"]
sample_id = row["id"]

print(row)
def parse_numpy_array(s):
    # Remove brackets and newlines
    s = s.replace("[", "").replace("]", "").replace("\n", " ")
    # Split by whitespace
    values = re.split(r"\s+", s.strip())
    return np.array([float(v) for v in values])

# Parse arrays
freqs = parse_numpy_array(row["frequencies"])
s11 = parse_numpy_array(row["s11"])

# Convert frequency to GHz
freqs_ghz = freqs / 1e9

# Plot
plt.figure(figsize=(10, 5))
plt.plot(freqs_ghz, s11)
plt.axhline(-10, linestyle="--", alpha=0.5)

plt.title(f"S11 Response | ID={sample_id}\nL={length}, W={width}, FeedY={feed_y}")
plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.grid(True)

plt.show()
