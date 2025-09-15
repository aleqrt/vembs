import copy
import os
import sys
from pickle import load, dump
from random import randint

import numpy as np
import pandas as pd
import joblib  # For loading scikit-learn models
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Adjust sys.path if necessary
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of 'features-analysis' (which is 'app')
app_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Get the parent directory of 'app' (which is your project root)
project_root = os.path.abspath(os.path.join(app_dir, '..'))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
root_dir = os.path.dirname(os.path.dirname('__file__'))

# define metadata
metas = ["gender", "race-bin", "race-cls", "age_decile", "insurance", "disease"]
embedding = "biomedclip-embedding"  # choose one of ["generalized-image-embedding", "medclip-embedding", "biomedclip-embedding"]
clip = "clip" in embedding

importance_folder = os.path.join(root_dir, "data", "mimic", embedding, "task-analysis-rf")
os.makedirs(importance_folder, exist_ok=True)

# Dictionaries to store feature importances and models for each metadata
metadata_importances = {}
models = {}

# Load data, models, and compute feature importances
for metadata in metas:
    data_folder = os.path.join(root_dir, "data", "mimic", embedding, metadata)
    models_folder = os.path.join(root_dir, "models", "mimic", embedding, metadata)
    importance_file = os.path.join(importance_folder, f'feature_importances_{metadata}.pkl')

    # Load train data
    print(f"Loading train data {metadata}...")
    train_set_path = os.path.join(data_folder, 'train_dataset_std.pkl')
    with open(train_set_path, 'rb') as f:
        if clip:
            X_train, y_train, train_all_ids, _ = load(f)
        else:
            X_train, y_train, train_all_ids = load(f)

    # print('Train set after processing - Max value:', X_train.max(), ' Min value:', X_train.min())
    # print('Train set after processing -  shape:', X_train.shape)
    # print('Train set after processing -  classes:', np.unique(y_train))

    # Load test data
    print(f"Loading test data {metadata}...")
    test_set_path = os.path.join(data_folder, 'test_dataset_std.pkl')
    with open(test_set_path, 'rb') as f:
        X_test, y_test = load(f)

    # print('Test set after processing - Max value:', X_test.max(), ' Min value:', X_test.min())
    # print('Test set after processing -  shape:', X_test.shape)
    # print('Test set after processing -  classes:', np.unique(y_test))

    n_class = len(np.unique(y_train))

    # Load all Random Forest models for the current metadata
    models[metadata] = []
    for fold in range(1, 11):
        model_path = os.path.join(models_folder, f'random_forest_fold_{fold}.pkl')
        model = joblib.load(model_path)
        models[metadata].append(model)
    print(f"{len(models[metadata])} Random Forest models for {metadata} loaded and stored.")

    # Check if feature importances are already computed and saved
    if os.path.exists(importance_file):
        print(f"Loading precomputed feature importances for {metadata} from {importance_file}...")
        with open(importance_file, 'rb') as f:
            importances_data = load(f)
            metadata_importances[metadata] = importances_data
    else:
        # Compute feature importances across all models
        importances_all_models = []
        for model in models[metadata]:
            importances_all_models.append(model.feature_importances_)
        importances_all_models = np.array(importances_all_models)  # Shape: (n_models, n_features)

        # Save feature importances
        metadata_importances[metadata] = importances_all_models

        # Save feature importances to file for future use
        with open(importance_file, 'wb') as f:
            dump(importances_all_models, f)
        print(f"Feature importances for {metadata} saved to {importance_file}.")

    print(f"Number of features in {metadata}: {metadata_importances[metadata].shape[1]}")


# Function to compute average feature importances for each task
def compute_average_importance(metadata):
    # Average over all models
    avg_importance = np.mean(metadata_importances[metadata], axis=0)
    return avg_importance


# Compute and store average feature importances for each task
feature_importances = {}
for metadata in metas:
    feature_importances[metadata] = compute_average_importance(metadata)


# Function to evaluate model metrics
def evaluate_model_metrics(models_list, X_test, y_test):
    metrics_list = []
    for model in models_list:
        # Predict probabilities
        y_test_pred_prob = model.predict_proba(X_test)

        if y_test_pred_prob.shape[1] != len(np.unique(y_test)):
            raise ValueError(f"Model {model} predicts {y_test_pred_prob.shape[1]} classes, "
                             f"but y_test has {len(np.unique(y_test))} classes.")

        # Flatten true labels
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # Assuming y_test is one-hot encoded
            y_test_flat = np.argmax(y_test, axis=1)
        else:
            y_test_flat = y_test.flatten()

        # Encode labels to consecutive integers starting from 0
        label_encoder = LabelEncoder()
        y_test_mapped = label_encoder.fit_transform(y_test_flat)
        classes = label_encoder.classes_
        n_classes = len(classes)

        if n_classes == 2:
            # Binary classification
            y_test_pred = model.predict(X_test)
            y_test_pred_mapped = y_test_pred

            # Compute ROC-AUC
            try:
                roc_auc = roc_auc_score(y_test_mapped, y_test_pred_prob[:, 1])
            except ValueError:
                roc_auc = None
        else:
            # Multiclass classification
            y_test_pred = model.predict(X_test)
            y_test_pred_mapped = label_encoder.transform(y_test_pred)

            # Binarize the true labels
            y_test_binarized = label_binarize(y_test_mapped, classes=np.arange(n_classes))

            # Compute ROC-AUC
            try:
                roc_auc = roc_auc_score(y_test_binarized, y_test_pred_prob, average="macro", multi_class="ovr")
            except ValueError:
                roc_auc = None

        # Calculate other metrics
        accuracy = accuracy_score(y_test_mapped, y_test_pred_mapped)
        precision = precision_score(y_test_mapped, y_test_pred_mapped, average='macro', zero_division=0)
        recall = recall_score(y_test_mapped, y_test_pred_mapped, average='macro', zero_division=0)
        f1 = f1_score(y_test_mapped, y_test_pred_mapped, average='macro', zero_division=0)

        metrics_list.append({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': roc_auc
        })

    # Now compute mean and std for each metric
    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    return mean_metrics, std_metrics


# Function to modify top-k important features
def modify_top_k_features(X_test, important_features_indices):
    X_modified = X_test.copy()
    X_modified[:, important_features_indices] = 0.0
    return X_modified


def retrain_rf_models(models_list, X_train, y_train, n_splits=10):
    # Initialize list to store retrained models and accuracies for each fold
    retrained_models = []
    fold_accuracies = []

    # Cross-validation setup
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\nTraining on fold {fold + 1}/{n_splits}...")

        # Split data for the current fold
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Retrain each model in the list for each fold
        # for n, model in enumerate(models_list):
        n = randint(0, 9)
        model = models_list[n]

        # Clone the model to ensure each fold uses an untrained model
        cloned_model = copy.deepcopy(model)

        # Train the cloned model on the fold's training data
        cloned_model.fit(X_train_fold, y_train_fold)

        # Validate the model on the validation fold
        y_val_pred = cloned_model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} - Model {n + 1} Accuracy: {accuracy:.4f}")

        # Append the retrained model to the list
        retrained_models.append(cloned_model)

    print(f"\nAverage validation accuracy across folds: {np.mean(fold_accuracies):.2f}")

    return retrained_models


# Function to evaluate performance drop for varying k
def evaluate_performance_drop(metadata, k_values):
    data_folder_metadata = os.path.join(root_dir, "data", embedding, metadata)

    # Load training data for the current metadata task
    train_set_path_metadata = os.path.join(data_folder_metadata, 'train_dataset_std.pkl')
    with open(train_set_path_metadata, 'rb') as f:
        if clip:
            X_train, y_train, train_all_ids, _ = load(f)
        else:
            X_train, y_train, train_all_ids = load(f)

    # Load test data for the current metadata task
    test_set_path_metadata = os.path.join(data_folder_metadata, 'test_dataset_std.pkl')
    with open(test_set_path_metadata, 'rb') as f:
        X_test, y_test = load(f)

    # Load labels for the 'disease' task (use the same embeddings)
    data_folder_disease = os.path.join(root_dir, "data", embedding, 'disease')
    train_set_path_disease = os.path.join(data_folder_disease, 'train_dataset_std.pkl')
    test_set_path_disease = os.path.join(data_folder_disease, 'test_dataset_std.pkl')

    with open(train_set_path_disease, 'rb') as f:
        if clip:
            X_train_disease, y_train_disease, _, _ = load(f)
        else:
            X_train_disease, y_train_disease, _ = load(f)
    with open(test_set_path_disease, 'rb') as f:
        X_test_disease, y_test_disease = load(f)

    # Convert y_train and y_train_disease to 1D arrays if needed
    y_train = y_train.ravel()
    y_train_disease = y_train_disease.ravel()

    y_test = y_test.ravel()
    y_test_disease = y_test_disease.ravel()

    importances = feature_importances[metadata]
    sorted_indices = np.argsort(-importances)

    performance_results = []

    for k in k_values:
        print(f"\nEvaluating performance for top-{k}% features modified in {metadata}...")
        k_percent = int(len(sorted_indices) * k / 100)
        top_k_indices = sorted_indices[:k_percent]
        num_features_modified = len(top_k_indices)

        print(f"k: {k}, num_features_modified: {num_features_modified}")
        # Modify train set
        X_train_modified = modify_top_k_features(X_train, top_k_indices)
        X_train_disease_modified = modify_top_k_features(X_train_disease, top_k_indices)

        retrained_models = {}

        # No features modified, use original models
        if k == 0:
            retrained_models[metadata] = models[metadata]
            retrained_models['disease'] = models['disease']
        else:
            # if thera are at least 1% fetures modified, retrain the model
            retrained_models[metadata] = retrain_rf_models(models[metadata], X_train_modified, y_train)
            retrained_models['disease'] = retrain_rf_models(models['disease'], X_train_disease_modified, y_train_disease)

        # Modify test set
        X_test_modified = modify_top_k_features(X_test, top_k_indices)
        X_test_disease_modified = modify_top_k_features(X_test_disease, top_k_indices)

        # Compute metrics for modified test set
        mean_metrics_metadata, std_metrics_metadata = evaluate_model_metrics(
            retrained_models[metadata], X_test_modified, y_test
        )
        mean_metrics_disease, std_metrics_disease = evaluate_model_metrics(
            retrained_models['disease'], X_test_disease_modified, y_test_disease
        )

        performance_drop = {
            'k': k,

            'Accuracy Mean Disease': mean_metrics_disease['Accuracy'],
            'Accuracy Std Disease': std_metrics_disease['Accuracy'],
            'Accuracy Mean': mean_metrics_metadata['Accuracy'],
            'Accuracy Std': std_metrics_metadata['Accuracy'],

            'Precision Mean Disease': mean_metrics_disease['Precision'],
            'Precision Std Disease': std_metrics_disease['Precision'],
            'Precision Mean': mean_metrics_metadata['Precision'],
            'Precision Std': std_metrics_metadata['Precision'],

            'Recall Mean Disease': mean_metrics_disease['Recall'],
            'Recall Std Disease': std_metrics_disease['Recall'],
            'Recall Mean': mean_metrics_metadata['Recall'],
            'Recall Std': std_metrics_metadata['Recall'],

            'F1 Score Mean Disease': mean_metrics_disease['F1 Score'],
            'F1 Score Std Disease': std_metrics_disease['F1 Score'],
            'F1 Score Mean': mean_metrics_metadata['F1 Score'],
            'F1 Score Std': std_metrics_metadata['F1 Score'],

            'AUC Mean Disease': mean_metrics_disease['AUC'],
            'AUC Std Disease': std_metrics_disease['AUC'],
            'AUC Mean': mean_metrics_metadata['AUC'],
            'AUC Std': std_metrics_metadata['AUC']
        }

        performance_results.append(performance_drop)
        print(f"Performance after retrain: {performance_drop}")

    return performance_results


# Set up k values
k_values = [0, 1, 5, 10, 20, 50]

# Run experiments for each task
# all_results = {}
#
# for metadata in metas[:-1]:
#     print(f"\nRunning experiments for task: {metadata}")
#     performance_results = evaluate_performance_drop(metadata, k_values)
#     all_results[metadata] = performance_results
#
#     # Convert to DataFrame and save to CSV
#     df_results = pd.DataFrame(performance_results)
#     csv_filename = f'performance_drop_{metadata}.csv'
#     csv_filepath = os.path.join(importance_folder, csv_filename)
#     df_results.to_csv(csv_filepath, index=False)
#     print(f"Results saved to {csv_filepath}")

all_results = {}

for metadata in metas[:-1]:
    print(f"\nRunning experiments for task: {metadata}")
    performance_results = evaluate_performance_drop(metadata, k_values)
    all_results[metadata] = performance_results

    df_results = pd.DataFrame(performance_results)
    csv_filename = f'performance_retrain_{metadata}_cv.csv'
    csv_filepath = os.path.join(importance_folder, csv_filename)
    df_results.to_csv(csv_filepath, index=False)
    print(f"Results saved to {csv_filepath}")


# # Function to plot performance drop and save figures
# def plot_performance_drop(metadata, metric_name):
#     performance_results = all_results[metadata]
#     ks = [res['k'] for res in performance_results]
#     metric_mean_disease = [res[f'{metric_name} Mean Disease'] for res in performance_results]
#     metric_mean = [res[f'{metric_name} Mean'] for res in performance_results]
#
#     metric_std_disease = [res[f'{metric_name} Std Disease'] for res in performance_results]
#     metric_std = [res[f'{metric_name} Std'] for res in performance_results]
#
#     plt.figure(figsize=(10, 6))
#     plt.errorbar(ks, metric_mean_disease, yerr=metric_std_disease, label=f'Disease Prediction', marker='o',
#                  capsize=5)
#     plt.errorbar(ks, metric_mean, yerr=metric_std, label=f'{metric_name} Prediction', marker='o',
#                  capsize=5)
#     plt.xlabel('Number of Top-k per cent Features Modified')
#     plt.ylabel(metric_name)
#     plt.title(f'{metadata.capitalize()} - {metric_name} vs. Number of Features Modified')
#     plt.legend()
#     plt.grid(True)
#
#     # Create a directory to save the figures
#     figures_folder = os.path.join(importance_folder, 'figures')
#     os.makedirs(figures_folder, exist_ok=True)
#
#     # Save the figure
#     figure_filename = f'{metadata}_{metric_name}_performance_drop.png'
#     figure_filepath = os.path.join(figures_folder, figure_filename)
#     plt.savefig(figure_filepath)
#     plt.close()
#     print(f"Figure saved to {figure_filepath}")
#
#
# # Plot performance drop for each task and metric
# metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
#
# for metadata in metas[:-1]:
#     for metric in metrics_to_plot:
#         plot_performance_drop(metadata, metric)
