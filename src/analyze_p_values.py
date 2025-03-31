import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add the parent directory to the system path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

root_dir = os.path.dirname(os.path.dirname(__file__))  # Root directory of the project
test_folders = ["Utest", "ttest"]
# metadata_folders = ["gender", "age_decile", "insurance", "race","race-wo"]
metadata_folders = ["grouped_race", "gender", "age_decile", "insurance"]
figures_folder = os.path.join(root_dir, "fig", "mimic", "p-values")
os.makedirs(figures_folder, exist_ok=True)

levels_of_signifcance = [0.001, 0.05]

for test in test_folders:
    for metadata in metadata_folders:
        directory = os.path.join(root_dir, "results", "mimic", test, metadata)
        file_list = os.listdir(directory)
        if len(file_list) == 1:
            print(file_list)
            with open(os.path.join(directory, file_list[0]), 'rb') as f:
                p_values = pickle.load(f)
                print(len(p_values))

                # Define bin edges
                bins = [0]
                bins.extend(levels_of_signifcance)
                bins.extend([0.1, 0.2, 0.5, 1])

                # Create histogram
                plt.hist(p_values, bins=bins, edgecolor='black')

                # Add title and labels
                title = f"{test} for gender with p-value 0.05"
                # plt.title(title)
                plt.xscale('log')
                plt.xlabel('p-value')
                plt.ylabel('# diff. features')
                # plt.show()
                plt.savefig(os.path.join(figures_folder, title + '.png'))
                plt.close()
        else:
            # divide file list into one vs one and one vs all

            # Initialize two lists to hold the divided entries
            one_vs_all = []
            one_vs_one = []

            # Loop through each entry in the list
            for each_file in file_list:
                # Check if "ALL" is in the entry
                if "ALL" in each_file:
                    one_vs_all.append(each_file)
                else:
                    one_vs_one.append(each_file)
            print(one_vs_one)
            print(one_vs_all)

            # Create an empty DataFrame with three columns
            col_names = ['Group 1', 'Group 2']
            col_names.extend([f'{test} -Under-{i}' for i in levels_of_signifcance])
            df = pd.DataFrame(columns=col_names)

            for each_file in one_vs_one:
                parts = each_file.split(" vs ")
                with open(os.path.join(directory, each_file), 'rb') as f:
                    p_values = pickle.load(f)
                new_row = [parts[0], parts[1][:-4]]
                new_row2 = [parts[1][:-4], parts[0]]
                p_values = np.array(p_values)
                for alpha in levels_of_signifcance:
                    temp = np.sum(p_values <= alpha)/len(p_values)
                    new_row.append(temp)
                    new_row2.append(temp)

                df.loc[len(df)] = new_row
                df.loc[len(df)] = new_row2

            for each_file in one_vs_all:
                parts = each_file.split(" vs ")
                with open(os.path.join(directory, each_file), 'rb') as f:
                    p_values = pickle.load(f)
                new_row = [parts[0], parts[1][:-4]]
                p_values = np.array(p_values)
                for alpha in levels_of_signifcance:
                    temp = np.sum(p_values <= alpha)/len(p_values)
                    new_row.append(temp)
                df.loc[len(df)] = new_row
            print(df.head())
            n_groups = len(df["Group 1"].unique())

            for k in range(len(df.columns) - 2):
                l = -2
                matrix = np.zeros([n_groups, n_groups + 1])
                for i in range(n_groups - 1):
                    for j in range(i + 1, n_groups):
                        l += 2
                        matrix[i, j] = df.iloc[l, k + 2]
                        matrix[j, i] = matrix[i, j]

                for i in range(n_groups):
                    l += 1
                    matrix[i, -1] = df.iloc[l, k + 2]
                #matrix = matrix.astype(int)
                print(matrix)
                # Create the heatmap
                row_labels = df['Group 1'].unique()
                column_labels = df['Group 2'].unique()
                # Swap the first two entries
                column_labels[0], column_labels[1] = column_labels[1], column_labels[0]
                for i in range(len(column_labels)):
                    column_labels[i] = column_labels[i].replace('_', '\n')
                for i in range(len(row_labels)):
                    row_labels[i] = row_labels[i].replace('_', '\n')
                # Generate the heatmap
                mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
                sns.heatmap(matrix, annot=True, fmt='f', cmap='Blues', yticklabels=row_labels,
                            xticklabels=column_labels, vmin=0, vmax=1376, mask=~mask)
                title = f"{test} for {metadata} with p-values {levels_of_signifcance[k]}" 
                plt.title(title)
                plt.tight_layout()
                # Display the heatmap
                # plt.show()
                plt.savefig(os.path.join(figures_folder, title + '.png'))
                plt.close()
