import os
import pandas as pd

# Paths
root_dir = os.path.dirname(os.path.dirname(__file__))  # Root directory of the project
data_folder = os.path.join(root_dir, "data", "mimic")
processed_file_path = os.path.join(data_folder, "processed_mimic_df.csv")
updated_insurance_file_path = os.path.join(data_folder, "bq-results-20240624-155227-1719244407963.csv")
output_file_path = os.path.join(data_folder, "updated_processed_mimic_df.csv")
figures_folder = os.path.join(root_dir, "fig", "mimic", "cxr-foundation", "metadata")
os.makedirs(figures_folder, exist_ok=True)

# Read processed data
print("Reading processed data...")
processed_df = pd.read_csv(processed_file_path)

# Read updated insurance data
print("Reading updated insurance data...")
updated_insurance_df = pd.read_csv(updated_insurance_file_path)

# Display data info
print("Processed data info:")
processed_df.info()
print("Updated insurance data info:")
updated_insurance_df.info()

# Merge with updated insurance data
print("Merging updated insurance data...")
merged_df = processed_df.drop(columns=['insurance'], errors='ignore')
merged_df = pd.merge(merged_df, updated_insurance_df, on='subject_id', how='left', suffixes=('', '_new'))

# Ensure 'insurance' column is in the correct position
if 'insurance' in merged_df.columns:
    insurance_col = merged_df.pop('insurance')
    merged_df.insert(merged_df.columns.get_loc('subject_id') + 1, 'insurance', insurance_col)

print("Sample rows from merged data:")
print(merged_df.head(2))

# Save the updated dataframe
print(f"Saving the updated dataframe to {output_file_path}...")
merged_df.to_csv(output_file_path, index=False)

print("Updated dataframe saved successfully.")


