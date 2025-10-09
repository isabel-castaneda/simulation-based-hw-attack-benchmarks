import os
import sys
import csv

def convert_stats_to_csv(stats_path):
    base_name = os.path.splitext(stats_path)[0]
    output_path = f"{base_name}.csv"

    with open(stats_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['counter', 'value', 'description'])

        for line in infile:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # # Skip comments and empty lines

            # Skip the start and end lines of statistics
            if line.startswith('---------- Begin Simulation Statistics') or \
               line.startswith('---------- End Simulation Statistics'):
                continue

            # Separate description (if any)
            parts = line.split('#', 1)
            description = parts[1].strip() if len(parts) > 1 else ''
            left = parts[0].strip().split()

            if len(left) >= 2:
                counter = left[0]
                value = ' '.join(left[1:])  # Support multi-part values
                csv_writer.writerow([counter, value, description])

    print(f"Converted: {stats_path} → {output_path}")

def convert_all_stats_in_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('_stats.txt'):
                stats_path = os.path.join(root, file)
                convert_stats_to_csv(stats_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_stats_to_csv.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Error: Provided path is not a valid directory.")
        sys.exit(1)

    convert_all_stats_in_directory(directory)

