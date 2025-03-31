import os
import sys
import pandas as pd
import numpy as np
import random as python_random
from IPython.display import display
import re

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Import the required modules from cxr_foundation
from cxr_foundation.mimic import parse_embedding_file_pattern

# Import utility functions
from utils import read_tfrecord_values, read_sha256_sums, clean_and_merge_data

# Set random seed for reproducibility
seed = 19
np.random.seed(seed)
python_random.seed(seed)

# Paths
root_dir = os.path.dirname(os.path.dirname(__file__))  # Root directory of the project
zip_file_path = os.path.join(root_dir, "data", "mimic", "generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0.zip")
unzipped_folder_path = os.path.join(root_dir, "data", "mimic",
                                    "generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0")
data_folder = os.path.join(root_dir, "data")
embeddings_file_name = "generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/SHA256SUMS.txt"
figures_folder = os.path.join(root_dir, "fig", "mimic", "cxr-foundation", "metadata")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

# Read SHA256SUMS.txt
df_embeddings = read_sha256_sums(zip_file_path, unzipped_folder_path, embeddings_file_name)

df_embeddings = df_embeddings[[1]].rename(columns={1: "embeddings_file"})
df_embeddings[["subject_id", "study_id", "dicom_id"]] = df_embeddings.apply(
    lambda x: parse_embedding_file_pattern(x["embeddings_file"]), axis=1, result_type="expand")

# Construct the full path for the embeddings file and normalize it
df_embeddings["embeddings_file"] = df_embeddings["embeddings_file"].apply(
    lambda x: os.path.normpath("generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/" + x).replace("\\",
                                                                                                                  "/"))

# Ensure the paths are correct
print("Sample rows from df_embeddings:")
print(df_embeddings.head(2))

# Example tfrecord
tfrecord_path = df_embeddings.embeddings_file.iloc[80000]
print(f"Trying to open TFRecord file: {tfrecord_path}")

example = read_tfrecord_values(tfrecord_path, zip_file_path, root_dir)

print("Example tfrecord:")
print(example)

# Read additional data
print("Reading metadata...")
df_metadata = pd.read_csv(os.path.join(data_folder, "mimic-cxr-2.0.0-metadata.csv.gz"), compression="gzip")
print("Reading CheXpert labels...")
MIMIC_CXR_Labels_df = pd.read_csv(os.path.join(data_folder, "mimic-cxr-2.0.0-chexpert.csv.gz"), compression="gzip")
print("Reading demographic data...")
demographic_df = pd.read_csv(os.path.join(data_folder, "admissions.csv.gz"), compression="gzip")
print("Reading patient data...")
patients_df = pd.read_csv(os.path.join(data_folder, "patients.csv.gz"), compression="gzip")

# Display data info
print("Metadata info:")
df_metadata.info()
print("CheXpert labels info:")
MIMIC_CXR_Labels_df.info()
print("Demographic data info:")
demographic_df.info()
print("Patient data info:")
patients_df.info()

# Clean and merge data
print("Cleaning and merging data...")
data_df = clean_and_merge_data(df_metadata, MIMIC_CXR_Labels_df, demographic_df, patients_df, df_embeddings)

# Convert age to age_decile
data_df['anchor_age'] = data_df['anchor_age'].astype('int64')
data_df.insert(data_df.columns.get_loc('anchor_age') + 1, 'age_decile',
               pd.cut(data_df['anchor_age'], bins=[0, 20, 40, 60, 80, float('inf')],
                      labels=['0-20', '20-40', '40-60', '60-80', '80+'], right=False))
data_df.drop(columns=['anchor_age'], inplace=True)

print("Sample rows from data_df:")
display(data_df.head(2))

# Extract the base extraction folder for replacement
base_extraction_path = os.path.join("generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0")

# Create a regular expression pattern that is cross-platform
pattern = re.escape(base_extraction_path) + r'[\\/]?'

# Replace the base path with an empty string to make it relative
data_df['path'] = data_df['path'].str.replace(pattern, '', regex=True)

print("Saving processed dataframes to CSV files...")
data_df.to_csv(os.path.join(data_folder, "processed_mimic_df.csv"), index=False)
