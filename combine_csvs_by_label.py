import os
import glob
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python combine_csvs_by_label.py <benign|malicious>")
    sys.exit(1)

label = sys.argv[1]
folder = label

all_rows = []
csv_files = glob.glob(f"{folder}/*/dataset/csvs/*.csv")

if not csv_files:
    print(f"No CSV files found in {folder}/.../dataset/csvs")
    sys.exit(1)

for file in csv_files:
    df = pd.read_csv(file)

    # Remove "_stats.csv" from filename to get program name
    program_name = os.path.basename(file).replace("_stats.csv", "")

    # Convert to one row: stat values as columns
    row = df.set_index("counter")["value"].to_dict()
    row["id"] = program_name
    row["label"] = label
    all_rows.append(row)

# Combine all rows
merged_df = pd.DataFrame(all_rows)

# Reorder columns: id, label, then the rest
cols = ["id", "label"] + [c for c in merged_df.columns if c not in ["id", "label"]]
merged_df = merged_df[cols]

# Save
output_file = f"{label}_dataset.csv"
merged_df.to_csv(output_file, index=False)
print(f"Dataset saved as {output_file}")
