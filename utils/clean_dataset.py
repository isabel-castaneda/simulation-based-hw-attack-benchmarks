import pandas as pd
import numpy as np
import re

df = pd.read_csv("filtered_dataset.csv")

# Extract the first number (integer or decimal) in situations like ‘552 100.00% 100.00%’, extracts only 552.0
def clean_value(v):
    if isinstance(v, str):
        match = re.search(r"[-+]?\d*\.?\d+", v)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return np.nan
        return np.nan
    return v

for col in df.columns:
    if col not in ["id", "label"]:
        df[col] = df[col].apply(clean_value)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(0, inplace=True)

num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].clip(lower=0, upper=1e12)

df.to_csv("filtered_dataset_clean.csv", index=False)
print("Cleaned dataset saved as filtered_dataset_clean.csv")

