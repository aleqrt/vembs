import os
import sys
from pickle import load, dump
from random import randint

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Add the parent directory to the system path for module import
# Use the actual __file__ variable rather than a string literal.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

# Get the directory of the current script
# (already computed above as current_dir)
# Get the parent directory of 'features-analysis' (which is 'app')
app_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Get the parent directory of 'app' (which is 'explain-error')
project_root = os.path.abspath(os.path.join(app_dir, '..'))

# Add the project root to sys.path if not already included
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# For further usage, we define root_dir as project_root.
root_dir = project_root

from app.integrated_gradient import get_gradients
from app.models import create_mlp

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# define metadata
metas = ["gender", "age_decile", "disease"]  # e.g. "race-bin", "race-cls", "insurance"
embedding = "cxr-foundation"  # choose one of ["cxr-foundation", "medclip", "biomedclip"]
clip = "clip" in embedding

gradient_folder = os.path.join(root_dir, "data", "chexpert", embedding, "task-analysis-mlp")
os.makedirs(gradient_folder, exist_ok=True)

# Dictionary to store gradients for each metadata and models
metadata_gradients = {}
models = {}

# Load data, models, and compute gradients
for metadata in metas:
    data_folder = os.path.join(root_dir, "data", "chexpert", embedding, metadata)
    figures_folder = os.path.join(root_dir, "fig", "chexpert", embedding, f"predict-{metadata}")
    models_folder = os.path.join(root_dir, "models", "chexpert", embedding, metadata)
    gradients_file = os.path.join(gradient_folder, f'gradients_{metadata}.pkl')

    # Load train data
    print(f"Loading train data for {metadata}...")
    train_set_path = os.path.join(data_folder, 'train_dataset_std.pkl')
    with open(train_set_path, 'rb') as f:
        if clip:
            X_train, y_train, train_all_ids, _ = load(f)
        else:
            X_train, y_train, train_all_ids = load(f)

    # Load test data
    print(f"Loading test data for {metadata}...")
    test_set_path = os.path.join(data_folder, 'test_dataset_std.pkl')
    with open(test_set_path, 'rb') as f:
        X_test, y_test = load(f)

    n_class = len(np.unique(y_train))

    # Load all 10 models for the current metadata
    models[metadata] = []
    for fold in range(1, 11):
        model = create_mlp(X_train.shape[1], n_class=n_class)
        weight_path = os.path.join(models_folder, f'mlp_fold_{fold}.keras')
        model.load_weights(weight_path)
        models[metadata].append(model)
    print(f"{len(models[metadata])} models for {metadata} loaded and stored.")

    # Check if gradients are already computed and saved
    if os.path.exists(gradients_file):
        print(f"Loading precomputed gradients for {metadata} from {gradients_file}...")
        with open(gradients_file, 'rb') as f:
            gradients_data = load(f)
            metadata_gradients[metadata] = gradients_data['grads_all_models']
    else:
        # Define subset size for gradient calculation
        print('Compute gradients for subsamples...')
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
            y_train_pred_prob = model.predict(X_train)

            # Ensure top_pred_idx is always an array
            if y_train_pred_prob.ndim > 1 and y_train_pred_prob.shape[1] > 1:
                top_pred_idx = np.argmax(y_train_pred_prob, axis=1)
            else:
                top_pred_idx = (y_train_pred_prob > 0.5).astype(int).ravel()

            top_pred_subset = top_pred_idx[subset_indices]

            for i, sample in enumerate(X_subset):
                grads = get_gradients([sample], model, top_pred_subset[i])
                grads_all_models[m_idx, i, :] = grads[0].numpy()

        # Save gradients by metadata
        metadata_gradients[metadata] = grads_all_models

        # Save computed gradients to file for future use
        with open(gradients_file, 'wb') as f:
            dump({'grads_all_models': grads_all_models}, f)
        print(f"Gradients for {metadata} saved to {gradients_file}.")

    print(f"Number of gradients in {metadata}: {np.array(metadata_gradients[metadata]).shape}")


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
        y_test_pred_prob = model.predict(X_test)

        # Flatten true labels
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_flat = np.argmax(y_test, axis=1)
        else:
            y_test_flat = y_test.flatten()

        # Encode labels to consecutive integers starting from 0
        label_encoder = LabelEncoder()
        y_test_mapped = label_encoder.fit_transform(y_test_flat)
        classes = label_encoder.classes_
        n_classes = len(classes)

        if n_classes == 2:
            if y_test_pred_prob.ndim > 1:
                y_test_pred_prob = y_test_pred_prob.flatten()
            y_test_pred = (y_test_pred_prob > 0.5).astype(int)
            y_test_pred_mapped = y_test_pred

            try:
                roc_auc = roc_auc_score(y_test_mapped, y_test_pred_prob)
            except ValueError:
                roc_auc = None
        else:
            y_test_pred = np.argmax(y_test_pred_prob, axis=1)
            y_test_pred_mapped = label_encoder.transform(y_test_pred)
            y_test_binarized = label_binarize(y_test_mapped, classes=np.arange(n_classes))

            if y_test_pred_prob.shape[1] != n_classes:
                raise ValueError(
                    "Number of classes in predicted probabilities does not match number of unique classes.")

            try:
                roc_auc = roc_auc_score(y_test_binarized, y_test_pred_prob, average="macro", multi_class="ovr")
            except ValueError:
                roc_auc = None

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

    metrics_df = pd.DataFrame(metrics_list)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    return mean_metrics, std_metrics


# Function to modify top-k important features
def modify_top_k_features(X, important_features_indices):
    X_modified = X.copy()
    X_modified[:, important_features_indices] = 0.0
    return X_modified


def retrain_mlp_models(models_list, X_train_modified, y_train, task_name, max_epochs=10, n_splits=10):
    retrained_models = []
    fold_accuracies = []

    # Prepare labels based on task
    if task_name == 'disease':
        n_class = 2
        y_train_categorical = y_train.flatten() if len(y_train.shape) == 1 else y_train[:, 0]
    else:
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_categorical = y_train
            n_class = y_train.shape[1]
        else:
            y_train = y_train.flatten()
            n_class = len(np.unique(y_train))
            if n_class == 2:
                y_train_categorical = y_train
            else:
                y_train_categorical = to_categorical(y_train, num_classes=n_class)

    print(f"Task: {task_name}, Number of classes: {n_class}")
    print(f"y_train shape before encoding: {y_train.shape}")
    if n_class != 2 or task_name != 'disease':
        print(f"y_train shape after one-hot encoding: {y_train_categorical.shape}")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_modified)):
        print(f"\nTraining on fold {fold + 1}/{n_splits}...")

        X_train_fold, X_val_fold = X_train_modified[train_idx], X_train_modified[val_idx]
        y_train_fold, y_val_fold = y_train_categorical[train_idx], y_train_categorical[val_idx]

        # Currently selecting a random model from models_list for retraining.
        # If intended to retrain each model, consider iterating over all models instead.
        n = randint(0, len(models_list) - 1)
        model = models_list[n]
        print(f'Training model {n} for {task_name}...')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=max_epochs, batch_size=256,
            callbacks=[early_stopping], verbose=1
        )

        retrained_models.append(model)
        fold_accuracies.append(history.history['val_accuracy'][-1])

    print(f"\nAverage validation accuracy across folds: {np.mean(fold_accuracies):.2f}")
    return retrained_models


# Function to evaluate performance drop for varying k
def evaluate_performance_drop(metadata, k_values):
    # Updated paths to include the "chexpert" directory
    data_folder_metadata = os.path.join(root_dir, "data", "chexpert", embedding, metadata)
    train_set_path_metadata = os.path.join(data_folder_metadata, 'train_dataset_std.pkl')
    with open(train_set_path_metadata, 'rb') as f:
        if clip:
            X_train, y_train, train_all_ids, _ = load(f)
        else:
            X_train, y_train, train_all_ids = load(f)

    test_set_path_metadata = os.path.join(data_folder_metadata, 'test_dataset_std.pkl')
    with open(test_set_path_metadata, 'rb') as f:
        X_test, y_test = load(f)

    # Update disease folder paths to include "chexpert"
    data_folder_disease = os.path.join(root_dir, "data", "chexpert", embedding, 'disease')
    train_set_path_disease = os.path.join(data_folder_disease, 'train_dataset_std.pkl')
    test_set_path_disease = os.path.join(data_folder_disease, 'test_dataset_std.pkl')

    with open(train_set_path_disease, 'rb') as f:
        if clip:
            X_train_disease, y_train_disease, _, _ = load(f)
        else:
            X_train_disease, y_train_disease, _ = load(f)
    with open(test_set_path_disease, 'rb') as f:
        X_test_disease, y_test_disease = load(f)

    n_class = len(np.unique(y_train))

    # Get feature importances and sorted indices (descending order)
    importances = feature_importances[metadata]
    sorted_indices = np.argsort(-importances)

    performance_results = []

    for k in k_values:
        print(f"Evaluating performance for top-{k}% features modified in {metadata}...")
        k_percent = int(len(sorted_indices) * k / 100)
        top_k_indices = sorted_indices[:k_percent]

        X_train_modified = modify_top_k_features(X_train, top_k_indices)
        X_train_disease_modified = modify_top_k_features(X_train_disease, top_k_indices)

        retrained_models = {}

        if k != 0:
            # Clone and compile models for current metadata
            models_to_retrain_metadata = []
            for model in models[metadata]:
                cloned_model = tf.keras.models.clone_model(model)
                cloned_model.set_weights(model.get_weights())
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                loss_function = 'binary_crossentropy' if n_class == 2 else 'categorical_crossentropy'
                cloned_model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
                models_to_retrain_metadata.append(cloned_model)

            # Clone and compile models for disease
            models_to_retrain_disease = []
            for model in models['disease']:
                cloned_model = tf.keras.models.clone_model(model)
                cloned_model.set_weights(model.get_weights())
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                cloned_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                models_to_retrain_disease.append(cloned_model)

            retrained_models[metadata] = retrain_mlp_models(
                models_to_retrain_metadata, X_train_modified, y_train, metadata
            )

            retrained_models['disease'] = retrain_mlp_models(
                models_to_retrain_disease, X_train_disease_modified, y_train_disease, 'disease'
            )
        else:
            retrained_models[metadata] = models[metadata]
            retrained_models['disease'] = models['disease']

        X_test_modified = modify_top_k_features(X_test, top_k_indices)
        X_test_disease_modified = modify_top_k_features(X_test_disease, top_k_indices)

        mean_metrics, std_metrics = evaluate_model_metrics(
            retrained_models[metadata], X_test_modified, y_test
        )
        mean_metrics_disease, std_metrics_disease = evaluate_model_metrics(
            retrained_models['disease'], X_test_disease_modified, y_test_disease
        )

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

# Modify the experiment loop to include retraining
all_results_with_retraining = {}

for metadata in metas[:-1]:
    print(f"\nRunning experiments with retraining for task: {metadata}")
    performance_results_with_retraining = evaluate_performance_drop(metadata, k_values)
    all_results_with_retraining[metadata] = performance_results_with_retraining

    df_results_retraining = pd.DataFrame(performance_results_with_retraining)
    csv_filename_retraining = f'performance_retrain_{metadata}_cv.csv'
    csv_filepath_retraining = os.path.join(gradient_folder, csv_filename_retraining)
    df_results_retraining.to_csv(csv_filepath_retraining, index=False)
    print(f"Results with retraining saved to {csv_filepath_retraining}")

# # (Optional plotting functions are commented out below)
# # Function to plot performance drop and save figures
# def plot_performance_drop(metadata, metric_name):
#     # performance_results = all_results[metadata]
#     performance_results = all_results_with_retraining[metadata]
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
