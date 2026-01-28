import subprocess
import numpy as np
import matplotlib.pyplot as plt

subprocess.run(["openEMS", "model.xml"], check=True)

freq = []
s11 = []

with open("results/antenna.s1p") as f:
    for line in f:
        if line.startswith("!") or line.startswith("#"):
            continue
        f0, re, im = map(float, line.split())
        freq.append(f0)
        s11.append(re + 1j*im)

freq = np.array(freq)
s11 = np.array(s11)

plt.plot(freq/1e9, 20*np.log10(np.abs(s11)))
plt.xlabel("Frequency (GHz)")
plt.ylabel("|S11| (dB)")
plt.grid(True)
plt.show()
