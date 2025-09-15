import copy
import os
import sys
from pickle import load, dump
from random import randint

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Adjust sys.path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root = os.path.abspath(os.path.join(app_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
root_dir = os.path.dirname(os.path.dirname('__file__'))

# define metadata
metas = ["gender", "race-bin", "race-cls", "age_decile", "insurance", "disease"]
embedding = "generalized-image-embedding"  # choose one of ["generalized-image-embedding", "medclip-embedding", "biomedclip-embedding"]
clip = "clip" in embedding

gradient_folder = os.path.join(root_dir, "data", "mimic", embedding, "task-analysis-lr")
os.makedirs(gradient_folder, exist_ok=True)

# Dictionaries to store gradients and models for each metadata
metadata_gradients = {}
models = {}


# Function to compute gradients for Logistic Regression
def get_logistic_gradients(X, model):
    """
    Compute the gradient of the predicted probabilities with respect to the input features for Logistic Regression.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        model (LogisticRegression): Trained Logistic Regression model.

    Returns:
        gradients (np.ndarray): Gradients of shape (n_samples, n_features).
    """
    # Get model coefficients
    coef = model.coef_  # Shape: (n_classes, n_features) or (1, n_features) for binary

    # Compute predicted probabilities
    probs = model.predict_proba(X)  # Shape: (n_samples, n_classes)

    n_samples, n_features = X.shape

    if coef.shape[0] == 1:
        # Binary classification
        coef = coef.flatten()  # Shape: (n_features,)
        probs_pos = probs[:, 1]  # Probability of positive class
        # Compute gradients
        gradients = np.outer(probs_pos * (1 - probs_pos), coef)
    else:
        # Multiclass classification
        n_classes = coef.shape[0]
        gradients = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            P = probs[i]  # Shape: (n_classes,)
            grad_sample = np.zeros(n_features)
            for k in range(n_classes):
                grad_sample += P[k] * (coef[k] - np.dot(P, coef))
            gradients[i] = grad_sample
    return gradients


# Load data, models, and compute gradients
for metadata in metas:
    data_folder = os.path.join(root_dir, "data", embedding, metadata)
    models_folder = os.path.join(root_dir, "models", embedding, metadata)
    gradients_file = os.path.join(gradient_folder, f'gradients_{metadata}.pkl')

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

    # Load all logistic regression models for the current metadata
    models[metadata] = []
    for fold in range(1, 11):
        model_path = os.path.join(models_folder, f'logistic_regression_fold_{fold}.pkl')
        model = joblib.load(model_path)
        models[metadata].append(model)
    print(f"{len(models[metadata])} Logistic Regression models for {metadata} loaded and stored.")

    # Check if gradients are already computed and saved
    if os.path.exists(gradients_file):
        print(f"Loading precomputed gradients for {metadata} from {gradients_file}...")
        with open(gradients_file, 'rb') as f:
            gradients_data = load(f)
            metadata_gradients[metadata] = gradients_data['gradients_all_models']
    else:
        # Define subset size for gradient calculation
        print('Computing gradients for subsamples...')
        subset_size = min(10000, len(X_train))  # Adjust as necessary
        subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_subset = X_train[subset_indices]

        # Initialize an array to store gradients from all models and samples
        n_models = len(models[metadata])
        n_samples = subset_size
        n_features = X_train.shape[1]
        grads_all_models = np.zeros((n_models, n_samples, n_features))

        # Loop over models
        for m_idx, model in enumerate(models[metadata]):
            grads = get_logistic_gradients(X_subset, model)
            grads_all_models[m_idx] = grads

        # Save gradients by metadata
        metadata_gradients[metadata] = grads_all_models

        # Save computed gradients to file for future use
        with open(gradients_file, 'wb') as f:
            dump({'gradients_all_models': grads_all_models}, f)
        print(f"Gradients for {metadata} saved to {gradients_file}.")

    print(f"Number of gradients in {metadata}: {metadata_gradients[metadata].shape}")


# Function to compute feature importances for each task
def compute_feature_importance(metadata):
    n_features = metadata_gradients[metadata].shape[-1]
    grads = metadata_gradients[metadata].reshape(-1, n_features)
    avg_grads_abs = np.mean(np.abs(grads), axis=0)
    return avg_grads_abs


# Compute and store feature importances for each task
feature_importances = {}
for metadata in metas:
    feature_importances[metadata] = compute_feature_importance(metadata)


# Function to evaluate model metrics
def evaluate_model_metrics(models_list, X_test, y_test):
    metrics_list = []
    for model in models_list:
        # Predict probabilities
        y_test_pred_prob = model.predict_proba(X_test)

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


# Function to retrain models on modified data
def retrain_lr_models(models_list, X_train_modified, y_train, n_splits=10, seed=42):
    # Ensure y_train is a 1D array
    print(f"Before reshaping, y_train shape: {y_train.shape}")
    y_train = y_train.ravel()
    print(f"After reshaping, y_train shape: {y_train.shape}")

    retrained_models = []
    fold_accuracies = []

    # # Determine if the task is binary or multi-class
    # n_classes = len(np.unique(y_train))

    # Cross-validation setup
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_modified)):
        print(f"\nTraining on fold {fold + 1}/{n_splits}...")

        # Split data for the current fold
        X_train_fold, X_val_fold = X_train_modified[train_idx], X_train_modified[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Select a random model from the list for retraining
        n = randint(0, len(models_list) - 1)
        model = models_list[n]

        # Clone the model to ensure each fold uses an untrained model
        cloned_model = copy.deepcopy(model)

        # Adjust model parameters based on number of classes
        cloned_model.set_params(solver='lbfgs')

        # Train the model on the fold's training data
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
    # Load test data for the current metadata task
    data_folder_metadata = os.path.join(root_dir, "data", embedding, metadata)
    test_set_path_metadata = os.path.join(data_folder_metadata, 'test_dataset_std.pkl')
    with open(test_set_path_metadata, 'rb') as f:
        X_test, y_test = load(f)

    # Load training data for the current metadata task
    train_set_path_metadata = os.path.join(data_folder_metadata, 'train_dataset_std.pkl')
    with open(train_set_path_metadata, 'rb') as f:
        if clip:
            X_train, y_train, train_all_ids, _ = load(f)
        else:
            X_train, y_train, train_all_ids = load(f)

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

    # Get feature importances
    importances = feature_importances[metadata]
    # Get indices of features sorted by importance (descending order)
    sorted_indices = np.argsort(-importances)

    # Initialize a list to store results
    performance_results = []

    for k in k_values:
        print(f"\nEvaluating performance for top-{k}% features modified in {metadata}...")
        # Select top-k important features
        k_percent = int(len(sorted_indices) * k / 100)
        top_k_indices = sorted_indices[:k_percent]
        num_features_modified = len(top_k_indices)
        print(f"k: {k}, num_features_modified: {num_features_modified}")

        # Modify train set
        X_train_modified = modify_top_k_features(X_train, top_k_indices)
        X_train_disease_modified = modify_top_k_features(X_train_disease, top_k_indices)

        retrained_models = {}

        if k != 0:
            # Retrain models for current metadata
            retrained_models[metadata] = retrain_lr_models(
                models[metadata], X_train_modified, y_train
            )

            # Retrain models for 'disease'
            retrained_models['disease'] = retrain_lr_models(
                models['disease'], X_train_disease_modified, y_train_disease
            )
        else:
            # No features modified, use original models
            retrained_models[metadata] = models[metadata]
            retrained_models['disease'] = models['disease']

        # Modify test set
        X_test_modified = modify_top_k_features(X_test, top_k_indices)
        X_test_disease_modified = modify_top_k_features(X_test_disease, top_k_indices)

        # Evaluate model performance on modified data
        mean_metrics, std_metrics = evaluate_model_metrics(
            retrained_models[metadata], X_test_modified, y_test
        )
        mean_metrics_disease, std_metrics_disease = evaluate_model_metrics(
            retrained_models['disease'], X_test_disease_modified, y_test_disease
        )

        # Calculate performance drop
        performance_drop = {
            'k': k,
            'Accuracy Mean Disease': mean_metrics_disease['Accuracy'],
            'Accuracy Std Disease': std_metrics_disease['Accuracy'],
            'Accuracy Mean': mean_metrics['Accuracy'],
            'Accuracy Std': std_metrics['Accuracy'],

            'Precision Mean Disease': mean_metrics_disease['Precision'],
            'Precision Std Disease': std_metrics_disease['Precision'],
            'Precision Mean': mean_metrics['Precision'],
            'Precision Std': std_metrics['Precision'],

            'Recall Mean Disease': mean_metrics_disease['Recall'],
            'Recall Std Disease': std_metrics_disease['Recall'],
            'Recall Mean': mean_metrics['Recall'],
            'Recall Std': std_metrics['Recall'],

            'F1 Score Mean Disease': mean_metrics_disease['F1 Score'],
            'F1 Score Std Disease': std_metrics_disease['F1 Score'],
            'F1 Score Mean': mean_metrics['F1 Score'],
            'F1 Score Std': std_metrics['F1 Score'],

            'AUC Mean Disease': mean_metrics_disease['AUC'],
            'AUC Std Disease': std_metrics_disease['AUC'],
            'AUC Mean': mean_metrics['AUC'],
            'AUC Std': std_metrics['AUC']
        }

        performance_results.append(performance_drop)
        print(f"Performance after retrain: {performance_drop}")

    return performance_results


# Set up k values
k_values = [0, 1, 5, 10, 20, 50]

# Run experiments for each task
all_results = {}

# for metadata in metas[:-1]:
#     print(f"\nRunning experiments for task: {metadata}")
#     performance_results = evaluate_performance_drop(metadata, k_values)
#     all_results[metadata] = performance_results
#
#     # Convert to DataFrame and save to CSV
#     df_results = pd.DataFrame(performance_results)
#     csv_filename = f'performance_drop_{metadata}.csv'
#     csv_filepath = os.path.join(gradient_folder, csv_filename)
#     df_results.to_csv(csv_filepath, index=False)
#     print(f"Results saved to {csv_filepath}")

for metadata in metas[:-1]:
    print(f"\nRunning experiments with retraining for task: {metadata}")
    performance_results = evaluate_performance_drop(metadata, k_values)
    all_results[metadata] = performance_results

    # Convert to DataFrame and save to CSV
    df_results = pd.DataFrame(performance_results)
    csv_filename = f'performance_retrain_{metadata}_cv.csv'
    csv_filepath = os.path.join(gradient_folder, csv_filename)
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
#     plt.errorbar(ks, metric_mean_disease, yerr=metric_std_disease, label='Disease Prediction', marker='o',
#                  capsize=5)
#     plt.errorbar(ks, metric_mean, yerr=metric_std, label=f'{metric_name} Prediction', marker='o',
#                  capsize=5)
#     plt.xlabel('Number of Top-k Percent Features Modified')
#     plt.ylabel(metric_name)
#     plt.title(f'{metadata.capitalize()} - {metric_name} vs. Number of Features Modified')
#     plt.legend()
#     plt.grid(True)
#
#     # Create a directory to save the figures
#     figures_folder = os.path.join(gradient_folder, 'figures')
#     os.makedirs(figures_folder, exist_ok=True)
#
#     # Save the figure
#     figure_filename = f'{metadata}_{metric_name}_performance_drop_retrain.png'
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
