import pandas as pd
from matplotlib import pyplot as plt

f = "./candidates-Analgesics-2026-02-12T09_07_19.500Z.tsv"

df = pd.read_csv(f, sep="\t")

# print(df.head())

df["Binding Confidence"].hist(range=(0.3, 0.9), bins=25, grid=False)

plt.title("Histogram of BoltzLab Confidence Scores")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")

plt.show()
