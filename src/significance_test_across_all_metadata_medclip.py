import os
import sys
import pandas as pd
import numpy as np
import random as python_random
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp
import pickle
from pickle import load

# Add the parent directory to the system path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Import utility functions
# from utils import read_tfrecord_values

# Set random seed for reproducibility
seed = 19
np.random.seed(seed)
python_random.seed(seed)

single = False

# Paths
root_dir = os.path.dirname(os.path.dirname(__file__))  # Root directory of the project
zip_file_path = os.path.join(root_dir, "data", "mimic",
                             "generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0.zip")
unzipped_folder_path = os.path.join(root_dir, "data", "mimic",
                                    "generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0")
data_folder = os.path.join(root_dir, "data")
os.makedirs(data_folder, exist_ok=True)
if single:
    medclip_embedding_folder = os.path.join(root_dir, "data", "mimic", "medclip-embedding-single")
else:
    medclip_embedding_folder = os.path.join(root_dir, "data", "mimic", "medclip-embedding-all")
medclip_embedding_path = os.path.join(medclip_embedding_folder, "embedding_from_medclip.pkl")

# Load data
print("Loading data...")
data_df = pd.read_csv(os.path.join(data_folder, 'df_final_for_metadata_prediction.csv'))
if single:
    data_df.drop_duplicates(subset='subject_id', inplace=True)
    data_df = data_df.reset_index()

# Update file paths
print("Updating file paths...")
data_df['path'] = data_df['path'].apply(
    lambda p: os.path.normpath(os.path.join("generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0", p)))
# data_df = data_df.sample(n=200)
# print(f"length of the dataset before removing duped entries {len(data_df)}")
# print(f"length of the dataset before removing duped entries {len(data_df)}")

data_df['grouped_race'] = data_df['grouped_race'].str.replace('/', '_')

if os.path.exists(medclip_embedding_path):
    with open(medclip_embedding_path, 'rb') as f:
        X, y, valid_id, ids = load(f)
print(len(X))
data_df["embedding"] = [row for row in X]
data_df = data_df.iloc[valid_id]

# data_df.drop_duplicates(subset='subject_id', inplace=True)
# data_df=data_df.reset_index()

# metadata to analyze
# metadata=["gender","age_decile","insurance","race"]
metadata = ["grouped_race", "gender", "age_decile", "insurance", "No Finding"]
# metadata = ["grouped_race","marital_status","language_binary"]
wo = False
alpha = 0.05
len_emb = 512

if single:
    results_folder = "mimic/medclip-embedding-single"
else:
    results_folder = "mimic/medclip-embedding-all"

for col in metadata:
    # Split the DataFrame
    unique_values = data_df[col].unique()
    unique_values.sort()
    unique_values = unique_values.tolist()
    if wo:
        if "UNKNOWN" in unique_values:
            unique_values.remove("UNKNOWN")

        if "UNABLE TO OBTAIN" in unique_values:
            unique_values.remove("UNABLE TO OBTAIN")

    embeddings_subset = [np.array(data_df[data_df[col] == val]["embedding"].values.tolist()) for val in unique_values]
    print(len(embeddings_subset[0]))
    if len(embeddings_subset) == 2:
        p_values = []
        count = 0
        for i in range(len_emb):
            U_stat, p_value = mannwhitneyu(embeddings_subset[0][:, i], embeddings_subset[1][:, i],
                                           alternative='two-sided')
            p_values.append(np.round(p_value, 4))
            if p_value <= alpha:
                count += 1
        print(
            f"{unique_values[0]} vs {unique_values[1]}: {count} features have a p_value under {alpha} according to the U-test")
        # Save the list to a file
        file_path = os.path.join(root_dir, "results", results_folder, "Utest", col,
                                 f"{unique_values[0]} vs {unique_values[1]}.pkl")
        os.makedirs(os.path.join(root_dir, "results", results_folder, "Utest", col), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(p_values, file)

        p_values = []
        count = 0
        for i in range(len_emb):
            t_statistic, p_value = ttest_ind(embeddings_subset[0][:, i], embeddings_subset[1][:, i])
            p_values.append(np.round(p_value, 4))
            if p_value <= alpha:
                count += 1
        print(
            f"{unique_values[0]} vs {unique_values[1]}: {count} features have a p_value under {alpha} according to the t-test")
        # Save the list to a file
        file_path = os.path.join(root_dir, "results", results_folder, "ttest", col,
                                 f"{unique_values[0]} vs {unique_values[1]}.pkl")
        os.makedirs(os.path.join(root_dir, "results", results_folder, "ttest", col), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(p_values, file)

        p_values = []
        count = 0
        for i in range(len_emb):
            t_statistic, p_value = ks_2samp(embeddings_subset[0][:, i], embeddings_subset[1][:, i])
            p_values.append(np.round(p_value, 4))
            if p_value <= alpha:
                count += 1
        print(
            f"{unique_values[0]} vs {unique_values[1]}: {count} features have a p_value under {alpha} according to the ks-test")
        # Save the list to a file
        file_path = os.path.join(root_dir, "results", results_folder, "kstest", col,
                                 f"{unique_values[0]} vs {unique_values[1]}.pkl")
        os.makedirs(os.path.join(root_dir, "results", results_folder, "kstest", col), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(p_values, file)
    else:
        # make comparisons one vs one
        if col == "race" and wo:
            col_name = col + "_wo"
        else:
            col_name = col
        for i in range(len(unique_values) - 1):
            for j in range(i + 1, len(unique_values)):
                count = 0
                p_values = []
                for k in range(len_emb):
                    U_stat, p_value = mannwhitneyu(embeddings_subset[i][:, k], embeddings_subset[j][:, k],
                                                   alternative='two-sided')
                    p_values.append(np.round(p_value, 4))
                    if p_value <= alpha:
                        count += 1
                print(
                    f"{unique_values[i]} vs {unique_values[j]}: {count} features have a p_value under {alpha} according to the U-test")
                file_path = os.path.join(root_dir, "results", results_folder, "Utest", col_name,
                                         f"{unique_values[i]} vs {unique_values[j]}.pkl")
                os.makedirs(os.path.join(root_dir, "results", results_folder, "Utest", col_name), exist_ok=True)
                with open(file_path, "wb") as file:
                    pickle.dump(p_values, file)

                count = 0
                p_values = []
                for k in range(len_emb):
                    t_statistic, p_value = ttest_ind(embeddings_subset[i][:, k], embeddings_subset[j][:, k])
                    p_values.append(np.round(p_value, 4))
                    if p_value <= alpha:
                        count += 1
                print(
                    f"{unique_values[i]} vs {unique_values[j]}: {count} features have a p_value under {alpha} according to the t-test")
                file_path = os.path.join(root_dir, "results", results_folder, "ttest", col_name,
                                         f"{unique_values[i]} vs {unique_values[j]}.pkl")
                os.makedirs(os.path.join(root_dir, "results", results_folder, "ttest", col_name), exist_ok=True)
                with open(file_path, "wb") as file:
                    pickle.dump(p_values, file)

                count = 0
                p_values = []
                for k in range(len_emb):
                    t_statistic, p_value = ks_2samp(embeddings_subset[i][:, k], embeddings_subset[j][:, k])
                    p_values.append(np.round(p_value, 4))
                    if p_value <= alpha:
                        count += 1
                print(
                    f"{unique_values[i]} vs {unique_values[j]}: {count} features have a p_value under {alpha} according to the ks-test")
                file_path = os.path.join(root_dir, "results", results_folder, "kstest", col_name,
                                         f"{unique_values[i]} vs {unique_values[j]}.pkl")
                os.makedirs(os.path.join(root_dir, "results", results_folder, "kstest", col_name), exist_ok=True)
                with open(file_path, "wb") as file:
                    pickle.dump(p_values, file)

        # make comparisons one vs all
        for i in range(len(unique_values)):
            # remove the entry that has is the ONE and create the ALL by concatenating the remaing
            temp = embeddings_subset[:]
            temp.pop(i)
            concat_subset = np.concatenate(temp, axis=0)

            count = 0
            p_values = []
            for k in range(len_emb):
                stat, p_value = mannwhitneyu(embeddings_subset[i][:, k], concat_subset[:, k], alternative='two-sided')
                p_values.append(np.round(p_value, 4))
                if p_value <= alpha:
                    count += 1
            print(f"{unique_values[i]} vs ALL: {count} features have a p_value under {alpha} according to the U-test")
            file_path = os.path.join(root_dir, "results", results_folder, "Utest", col_name,
                                     f"{unique_values[i]} vs ALL.pkl")
            os.makedirs(os.path.join(root_dir, "results", results_folder, "Utest", col_name), exist_ok=True)
            with open(file_path, "wb") as file:
                pickle.dump(p_values, file)

            count = 0
            p_values = []
            for k in range(len_emb):
                t_statistic, p_value = ttest_ind(embeddings_subset[i][:, k], concat_subset[:, k])
                p_values.append(np.round(p_value, 4))
                if p_value <= alpha:
                    count += 1
            print(f"{unique_values[i]} vs ALL: {count} features have a p_value under {alpha} according to the t-test")
            file_path = os.path.join(root_dir, "results", results_folder, "ttest", col_name,
                                     f"{unique_values[i]} vs ALL.pkl")
            os.makedirs(os.path.join(root_dir, "results", results_folder, "ttest", col_name), exist_ok=True)
            with open(file_path, "wb") as file:
                pickle.dump(p_values, file)

            count = 0
            p_values = []
            for k in range(len_emb):
                t_statistic, p_value = ks_2samp(embeddings_subset[i][:, k], concat_subset[:, k])
                p_values.append(np.round(p_value, 4))
                if p_value <= alpha:
                    count += 1
            print(f"{unique_values[i]} vs ALL: {count} features have a p_value under {alpha} according to the ks-test")
            file_path = os.path.join(root_dir, "results", results_folder, "kstest", col_name,
                                     f"{unique_values[i]} vs ALL.pkl")
            os.makedirs(os.path.join(root_dir, "results", results_folder, "kstest", col_name), exist_ok=True)
            with open(file_path, "wb") as file:
                pickle.dump(p_values, file)
