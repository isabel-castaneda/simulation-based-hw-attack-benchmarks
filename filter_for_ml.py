import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python filter_counters_for_ml.py <path_to_full_dataset.csv>")
    sys.exit(1)

input_csv_path = sys.argv[1]

df = pd.read_csv(input_csv_path)

useful_keywords = [
    "inst", "tick", "load", "store", "miss", "access", "latency",
    "hit", "fetch", "branch", "mem", "commit", "flush", "exec", "cycles"
]

available_counters = df.columns.tolist()
available_counters = [c for c in available_counters if c not in ["id", "label"]]

selected_counters = [
    col for col in available_counters
    if any(keyword in col.lower() for keyword in useful_keywords)
]

print("Selected counters for ML:", selected_counters)

filtered_df = df[["id", "label"] + selected_counters]

output_csv_path = "filtered_dataset.csv"
filtered_df.to_csv(output_csv_path, index=False)
print(f"\nFiltered dataset saved as: {output_csv_path}")
