import os
import pickle

import joblib
import zipfile
import pandas as pd
import tempfile
import numpy as np
from io import TextIOWrapper
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import load, dump

import sys

import torch
from PIL import Image

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, recall_score, \
    precision_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from umap.parametric_umap import ParametricUMAP
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from cxr_foundation import embeddings_data
import tensorflow as tf

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Paths
root_dir = os.path.dirname(os.path.abspath('__file__'))


def save_figure(fig, filename, figures_folder):
    """
    Save the given figure to the specified directory with the given filename.

    Parameters:
    - fig: matplotlib.figure.Figure, the figure object to save.
    - filename: str, the name of the file.
    - figures_folder: str, the folder where the figure should be saved.
    """
    fig.savefig(os.path.join(figures_folder, filename))
    plt.close(fig)


def read_tfrecord_values_from_zip(zip_file, path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        with zip_ref.open(path) as tfrecord_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(tfrecord_file.read())
                tmp_file_path = tmp_file.name

            example = embeddings_data.read_tfrecord_values(tmp_file_path)
            os.remove(tmp_file_path)
            return example


def read_tfrecord_values(path, zip_file_path, root_dir):
    unzipped_path = os.path.join(root_dir, "data", path)
    if os.path.exists(unzipped_path):
        return embeddings_data.read_tfrecord_values(unzipped_path)

    else:
        return read_tfrecord_values_from_zip(zip_file_path, path)


def read_sha256_sums(zip_file_path, unzipped_folder_path, embeddings_file_name):
    if os.path.exists(unzipped_folder_path):
        sha256_file_path = os.path.join(unzipped_folder_path, "SHA256SUMS.txt")
        df_embeddings = pd.read_csv(sha256_file_path, delimiter=" ", header=None, skiprows=[0])
    else:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            with zip_ref.open(embeddings_file_name) as file:
                df_embeddings = pd.read_csv(TextIOWrapper(file), delimiter=" ", header=None, skiprows=[0])
    return df_embeddings


def clean_and_merge_data(df_metadata, MIMIC_CXR_Labels_df, demographic_df, patients_df, df_embeddings):
    MIMIC_CXR_Labels_df.replace(np.nan, 0, inplace=True)
    MIMIC_CXR_Labels_df.replace(-1, 0, inplace=True)
    demographic_df = demographic_df.drop_duplicates(subset='subject_id')
    ethnicity_df = demographic_df[['subject_id', 'ethnicity']].drop_duplicates()
    inconsistent_race = ethnicity_df[ethnicity_df.subject_id.isin(
        ethnicity_df.subject_id.value_counts().loc[lambda x: x.gt(1)].index)].subject_id.unique()
    data_df = df_metadata.merge(demographic_df, on='subject_id').merge(patients_df, on='subject_id')
    data_df = data_df.drop(columns=['anchor_year', 'anchor_year_group', 'dod', 'hadm_id', 'admittime', 'dischtime',
                                    'deathtime', 'admission_type', 'admission_location', 'discharge_location',
                                    'language', 'marital_status', 'edregtime', 'edouttime', 'hospital_expire_flag',
                                    'PerformedProcedureStepDescription', 'ViewPosition', 'Rows', 'Columns',
                                    'StudyDate', 'StudyTime', 'ProcedureCodeSequence_CodeMeaning',
                                    'ViewCodeSequence_CodeMeaning', 'PatientOrientationCodeSequence_CodeMeaning'])
    data_df = data_df[~data_df.subject_id.isin(inconsistent_race)].rename(columns={"ethnicity": "race"})
    data_df = data_df.merge(MIMIC_CXR_Labels_df, on=['study_id', 'subject_id'])
    data_df = df_embeddings.merge(data_df, on='dicom_id', how='left').dropna().rename(
        columns={'subject_id_x': 'subject_id', 'study_id_x': 'study_id'})
    data_df = data_df[['embeddings_file', 'subject_id', 'study_id', 'dicom_id', 'gender', 'insurance',
                       'anchor_age', 'race', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                       'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']]
    data_df.insert(4, "split", "none", True)
    data_df.rename(columns={'embeddings_file': 'path'}, inplace=True)
    return data_df


# Function to remove correlated features
def remove_correlated_features(X, threshold=0.75, removed_features=None, figures_folder=None):
    if removed_features is None:
        corr_matrix = np.corrcoef(X, rowvar=False)
        upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
        correlated_pairs = [(i, j) for i, j in zip(*upper_triangle_indices) if abs(corr_matrix[i, j]) > threshold]

        features_to_remove = set()
        for i, j in correlated_pairs:
            features_to_remove.add(j)

        X_reduced = np.delete(X, list(features_to_remove), axis=1)

        # Save the correlation matrix plot
        if figures_folder:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
            plt.title('Feature Correlation Matrix')
            plt.savefig(os.path.join(figures_folder, 'correlation_matrix.png'))
            plt.close()

        return X_reduced, list(features_to_remove)
    else:
        X_reduced = np.delete(X, removed_features, axis=1)
        return X_reduced, removed_features


# Plotting CV results (accuracy and loss)
def plot_loss_acc_cv(train_scores, val_scores, metric_name, figures_folder):
    min_length = min(len(scores) for scores in train_scores + val_scores)
    train_scores = [scores[:min_length] for scores in train_scores]
    val_scores = [scores[:min_length] for scores in val_scores]

    train_mean = np.mean(train_scores, axis=0)
    train_std = np.std(train_scores, axis=0)
    val_mean = np.mean(val_scores, axis=0)
    val_std = np.std(val_scores, axis=0)
    epochs = range(1, min_length + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_mean, label=f'Train {metric_name}', color='blue')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.plot(epochs, val_mean, label=f'Validation {metric_name}', color='orange')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.savefig(os.path.join(figures_folder, f'cv_{metric_name.lower()}.png'))
    plt.close()


def calculate_roc_metrics(y_true, y_pred_prob, multi_class=False):
    """
    Calculate ROC metrics for binary or multi-class classification.

    Parameters:
    - y_true: True labels
    - y_pred_prob: Predicted probabilities
    - multi_class: Boolean indicating if it's a multi-class problem

    Returns:
    - tprs: True Positive Rates
    - aucs: Area Under Curves
    - mean_fpr: Mean False Positive Rate
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    if multi_class:
        for class_idx in range(y_pred_prob.shape[1]):
            if class_idx not in y_true:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, class_idx], pos_label=class_idx)
            roc_auc = roc_auc_score(y_true == class_idx, y_pred_prob[:, class_idx])
            aucs.append(roc_auc)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
    else:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = roc_auc_score(y_true, y_pred_prob)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    return tprs, aucs, mean_fpr


def plot_roc_curves(models_folder, model_names, X_folds, y_folds, figures_folder, suffix=''):
    """
    Plots ROC curves for multiple models over multiple folds.

    Parameters:
    - models_folder (str): Path to the folder where the models are saved.
    - model_names (list): List of model names.
    - X_folds (list of ndarray): List of validation feature sets for each fold.
    - y_folds (list of ndarray): List of validation target sets for each fold.
    - figures_folder (str): Path to the folder where the figures are saved.
    - suffix (str): Suffix for the output file.
    """
    plt.figure(figsize=(10, 8))

    mean_fpr = np.linspace(0, 1, 100)

    for model_name in model_names:
        all_tprs = []
        all_aucs = []

        for fold_idx in range(1, 10):
            model_path = os.path.join(models_folder,
                                      f'{model_name}_fold_{fold_idx}.keras' if model_name == 'mlp' else f'{model_name}_fold_{fold_idx}.pkl')

            if model_name == 'mlp':
                model = tf.keras.models.load_model(model_path)
                y_pred_prob = model.predict(X_folds[fold_idx - 1])
            else:
                model = joblib.load(model_path)
                y_pred_prob = model.predict_proba(X_folds[fold_idx - 1])

            multi_class = y_pred_prob.shape[1] > 1
            if model_name == 'mlp' and not multi_class:
                y_pred_prob = y_pred_prob.ravel()
            elif not multi_class:
                y_pred_prob = y_pred_prob[:, 1]

            tprs, aucs, _ = calculate_roc_metrics(y_folds[fold_idx - 1], y_pred_prob, multi_class)
            all_tprs.extend(tprs)
            all_aucs.extend(aucs)

        mean_tpr = np.mean(all_tprs, axis=0)
        std_tpr = np.std(all_tprs, axis=0)
        mean_auc = np.mean(all_aucs)

        plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f})')
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(figures_folder, f'roc_curves_{suffix}.png'))
    plt.close()


def plot_roc_curves_test(models, X_test, y_test, figures_folder, suffix=''):
    """
    Plots ROC curves for multiple models.

    Parameters:
    - models (dict): Dictionary containing lists of models to evaluate.
    - X_test (ndarray): Test feature set.
    - y_test (ndarray): Test target set.
    - figures_folder (str): Path to the folder where the figures are saved.
    - suffix (str): Suffix for the output file.
    """
    plt.figure(figsize=(10, 8))

    for model_name, model_list in models.items():
        all_tprs = []
        all_aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for model in model_list:
            if model_name == 'mlp':
                y_pred_prob = model.predict(X_test)
            else:
                y_pred_prob = model.predict_proba(X_test)

            multi_class = y_pred_prob.shape[1] > 1
            if model_name == 'mlp' and not multi_class:
                y_pred_prob = y_pred_prob.ravel()
            elif not multi_class:
                y_pred_prob = y_pred_prob[:, 1]

            tprs, aucs, _ = calculate_roc_metrics(y_test, y_pred_prob, multi_class)
            all_tprs.extend(tprs)
            all_aucs.extend(aucs)

        if not all_tprs:
            print(f'No valid models found for {model_name}. Skipping...')
            continue

        mean_tpr = np.mean(all_tprs, axis=0)
        std_tpr = np.std(all_tprs, axis=0)
        mean_auc = np.mean(all_aucs)

        plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f})')
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2)

    plt.rcParams.update({'font.size': 22})
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(figures_folder, f'roc_curves_{suffix}.png'))
    plt.close()


def plot_boxplot(results, figures_folder, suffix=''):
    """
    Plots boxplots for performance metrics.

    Parameters:
    - results (dict): Dictionary containing the performance metrics for each model.
    - figures_folder (str): Path to the folder where the figures are saved.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 15))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    for ax, metric in zip(axes, metrics):
        data = [results[m][metric] for m in results.keys()]
        ax.boxplot(data, labels=results.keys())
        ax.set_title(metric.capitalize())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)

    plt.savefig(os.path.join(figures_folder, f'metrics_boxplot_{suffix}.png'))
    plt.close()


# Function to evaluate models and print metrics
def evaluate_models(models, model_name, X_test, y_test, n_class, figures_folder):
    print(f'Evaluating {model_name}...')

    # Lists to store metrics for each model in the ensemble
    accuracies, precisions, recalls, f1s, roc_aucs = [], [], [], [], []

    for model in models[model_name]:
        if model_name == 'mlp':
            y_test_pred_prob = model.predict(X_test)
        else:
            y_test_pred_prob = model.predict_proba(X_test)

        if n_class == 2:
            if model_name == 'mlp':
                y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()
                y_test_pred_prob = y_test_pred_prob.flatten()  # Ensure correct shape
            else:
                y_test_pred = (y_test_pred_prob[:, 1] > 0.5).astype(int)
                y_test_pred_prob = y_test_pred_prob[:, 1]  # Use probabilities of the positive class
            y_test_flat = y_test.flatten()
        else:
            y_test_pred = np.argmax(y_test_pred_prob, axis=1)
            y_test_flat = y_test

        # Calculate metrics for this model
        accuracies.append(accuracy_score(y_test_flat, y_test_pred))
        precisions.append(precision_score(y_test_flat, y_test_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test_flat, y_test_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test_flat, y_test_pred, average='macro', zero_division=0))

        # Calculate ROC-AUC for this model
        if n_class == 2:
            roc_auc = roc_auc_score(y_test_flat, y_test_pred_prob)
        else:
            roc_auc = roc_auc_score(y_test_flat, y_test_pred_prob, multi_class='ovr', average='macro')
        roc_aucs.append(roc_auc)

    # Calculate mean and std for each metric
    test_accuracy = (np.mean(accuracies), np.std(accuracies))
    test_precision = (np.mean(precisions), np.std(precisions))
    test_recall = (np.mean(recalls), np.std(recalls))
    test_f1 = (np.mean(f1s), np.std(f1s))
    test_roc_auc = (np.mean(roc_aucs), np.std(roc_aucs))

    # Print test metrics
    print(f'Test Accuracy: {test_accuracy[0]:.4f} ± {test_accuracy[1]:.4f}')
    print(f'Test Precision: {test_precision[0]:.4f} ± {test_precision[1]:.4f}')
    print(f'Test Recall: {test_recall[0]:.4f} ± {test_recall[1]:.4f}')
    print(f'Test F1 Score: {test_f1[0]:.4f} ± {test_f1[1]:.4f}')
    print(f'Test ROC AUC: {test_roc_auc[0]:.4f} ± {test_roc_auc[1]:.4f}')

    # For the confusion matrix, we'll use the ensemble prediction
    if model_name == 'mlp':
        y_test_pred_prob_ensemble = np.mean([model.predict(X_test) for model in models[model_name]], axis=0)
    else:
        y_test_pred_prob_ensemble = np.mean([model.predict_proba(X_test) for model in models[model_name]], axis=0)

    if n_class == 2:
        if model_name == 'mlp':
            y_test_pred_ensemble = (y_test_pred_prob_ensemble > 0.5).astype(int).flatten()
        else:
            y_test_pred_ensemble = (y_test_pred_prob_ensemble[:, 1] > 0.5).astype(int)
    else:
        y_test_pred_ensemble = np.argmax(y_test_pred_prob_ensemble, axis=1)

    # Plot confusion matrix for test set
    cm = confusion_matrix(y_test_flat, y_test_pred_ensemble)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix - {model_name.capitalize()} Test Set')
    plt.savefig(os.path.join(figures_folder, f'confusion_matrix_test_{model_name}.png'))
    plt.close()

    return accuracies, precisions, recalls, f1s, roc_aucs


def get_embedding_and_metadata_chexpert(df):
    # Make a copy so we don't mutate the original df
    df = df.copy()

    # Prepare target dict
    target = dict()

    # 1) Encode gender: Male = 1, Female = 0
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({'M': 1, 'F': 0})
        target["gender"] = df["gender"].values.reshape(-1, 1)

    # 2) Encode age_decile: 20-40=0, 40-60=1, 60-80=2, 80+=3
    if "age_decile" in df.columns:
        df["age_decile"] = df["age_decile"].replace({
            '20-40': 0,
            '40-60': 1,
            '60-80': 2,
            '80+': 3
        })
        target["age_decile"] = df["age_decile"].values.reshape(-1, 1)

    # 3) Encode race-cls and race-bin
    if "grouped_race" in df.columns:
        df["race-cls"] = df["grouped_race"].replace({
            'White': 0,
            'Black': 1,
            'Hispanic/Latino': 2,
            'Other': 3
        })
        target["race-cls"] = df["race-cls"].values.reshape(-1, 1)

        # For binary race encoding: White=0, else=1
        df["race-bin"] = df["grouped_race"].replace({
            'White': 0,
            'Black': 1,
            'Hispanic/Latino': 1,
            'Other': 1
        })
        target["race-bin"] = df["race-bin"].values.reshape(-1, 1)

    # 4) Encode disease (assuming "No Finding" is the target)
    if "No Finding" in df.columns:
        target["disease"] = df["No Finding"].values.reshape(-1, 1)

    # 5) Handle subject ID: remove "patient" prefix and convert to integer, return as 1D array
    if "subject_id" in df.columns:
        subject_id = df["subject_id"].astype(str).apply(
            lambda x: int(x.lower().replace("patient", "")) if x.lower().startswith("patient") else int(x)
        ).values  # Return as a 1D array
    else:
        subject_id = np.arange(len(df))

    # 6) Extract the actual embeddings from the "embedding" column.
    if "embedding" not in df.columns:
        raise ValueError("DataFrame must have an 'embedding' column with the actual embeddings.")
    embedding_data = np.array(list(df["embedding"]))

    return embedding_data, target, subject_id


# Function to generate X-Y metadata
def get_embedding_and_metadata(df, data_folder, r_dir):
    target = dict()

    # Encode gender: Male = 1, Female = 0
    df["gender"] = df['gender'].replace({'M': 1, 'F': 0})
    target["gender"] = np.array(list(df["gender"].values)).reshape(-1, 1)

    # Convert 'insurance' column to numerical values
    df["insurance"] = df['insurance'].replace({'Medicare': 0, 'Medicaid': 1, 'Private': 2})
    target["insurance"] = np.array(list(df["insurance"].values)).reshape(-1, 1)

    # Encode age_decile: 20-40 = 0, 40-60 = 1, 60-80 = 2, 80+ = 3
    df["age_decile"] = df['age_decile'].replace({
        '20-40': 0, '40-60': 1, '60-80': 2, '80+': 3
    })
    target["age_decile"] = np.array(list(df["age_decile"].values)).reshape(-1, 1)

    # Encode race into classes without 'UNKNOWN' and 'UNABLE TO OBTAIN'
    df["race-cls"] = df['grouped_race'].replace({
        'White': 0, 'Black': 1, 'Hispanic/Latino': 2,
        'Other': 3
    })
    target["race-cls"] = np.array(list(df["race-cls"].values)).reshape(-1, 1)

    # Encode race into classes without 'UNKNOWN' and 'UNABLE TO OBTAIN'
    df["race-bin"] = df['grouped_race'].replace({
        'White': 0, 'Black': 1, 'Hispanic/Latino': 1,
        'Other': 1
    })
    target["race-bin"] = np.array(list(df["race-bin"].values)).reshape(-1, 1)

    target["disease"] = np.array(list(df["No Finding"].values)).reshape(-1, 1)

    # Extract embedding data
    # Normalize the path for consistency across different operating systems
    df['path'] = df['path'].apply(lambda p: os.path.normpath(
        os.path.join("generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0", p)))

    # Read TFRecord values for each row
    df["embedding"] = df.apply(lambda x: read_tfrecord_values(x["path"], data_folder, r_dir), axis=1)

    embedding_data = np.array(list(df["embedding"].values))
    subject_id = np.array(list(df["subject_id"].values)).reshape(-1, 1)
    return embedding_data, target, subject_id


def get_embedding_and_metadata_from_pickle(pkl_path):
    """
    Load precomputed embeddings, metadata dictionary, and subject IDs
    from a single pickle file.  Assumes the pickle contains a tuple
    of the form (X, y_dict, subject_ids) or similar.

    Parameters
    ----------
    pkl_path : str
        Full path to the pickle file containing the embeddings and metadata.

    Returns
    -------
    X : np.ndarray
        2D array of embeddings (e.g., shape (N, D)).
    y_dict : dict
        Dictionary mapping metadata keys (e.g., "gender", "disease", etc.) to 1D arrays of labels.
    subject_ids : np.ndarray
        1D array of subject/patient IDs corresponding to rows in X.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {pkl_path}")

    print(f"Loading embeddings and metadata from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Adjust unpacking to match how you stored data. For example:
    # data may be (X, y_dict, subject_ids) or (X, y, subject_ids).
    # Example below assumes (X, y_dict, subject_ids).
    if not isinstance(data, tuple) or len(data) < 3:
        raise ValueError("Pickle file does not contain the expected tuple (X, y_dict, subject_ids).")

    X, y_dict, subject_ids = data[0], data[1], data[2]

    # (Optional) add any checks or prints here
    print(f"  Loaded embeddings of shape: {X.shape}")
    if isinstance(y_dict, dict):
        print(f"  Found label dictionary with keys: {list(y_dict.keys())}")
    print(f"  Found {len(subject_ids)} subject_ids.")

    return X, y_dict, subject_ids


def get_embedding_from_biomedclip_and_metadata(df, r_dir, model,
                                               preprocess, device):
    print("Extract targets and embedding with BiomedCLIP...")
    target = dict()

    # Encode gender: Male = 1, Female = 0
    df["gender"] = df['gender'].replace({'M': 1, 'F': 0})
    target["gender"] = np.array(list(df["gender"].values)).reshape(-1, 1)

    # Convert 'insurance' column to numerical values
    df["insurance"] = df['insurance'].replace({'Medicare': 0, 'Medicaid': 1, 'Private': 2})
    target["insurance"] = np.array(list(df["insurance"].values)).reshape(-1, 1)

    # Encode age_decile: 20-40 = 0, 40-60 = 1, 60-80 = 2, 80+ = 3
    df["age_decile"] = df['age_decile'].replace({
        '20-40': 0, '40-60': 1, '60-80': 2, '80+': 3
    })
    target["age_decile"] = np.array(list(df["age_decile"].values)).reshape(-1, 1)

    # Encode race into classes without 'UNKNOWN' and 'UNABLE TO OBTAIN'
    df["race-cls"] = df['grouped_race'].replace({
        'White': 0, 'Black': 1, 'Hispanic/Latino': 2,
        'Other': 3
    })
    target["race-cls"] = np.array(list(df["race-cls"].values)).reshape(-1, 1)

    # Encode race into classes without 'UNKNOWN' and 'UNABLE TO OBTAIN'
    df["race-bin"] = df['grouped_race'].replace({
        'White': 0, 'Black': 1, 'Hispanic/Latino': 1,
        'Other': 1
    })
    target["race-bin"] = np.array(list(df["race-bin"].values)).reshape(-1, 1)

    target["disease"] = np.array(list(df["No Finding"].values)).reshape(-1, 1)

    # Extract embedding data
    # Normalize the path for consistency across different operating systems
    df['path'] = df['path'].apply(lambda p: os.path.normpath(p))

    embeddings = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows()):
        image_path = os.path.join(r_dir, "data", "mimic_jpg", row['path'].replace('.tfrecord', '.jpg'))
        try:
            # print(f"Compute embedding biomedclip for: {image_path}")
            image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
            embeddings.append(embedding.cpu().numpy())
            valid_indices.append(idx)
        except FileNotFoundError:
            print(f"Image {image_path} not found.")
            embeddings.append(np.zeros((1, 512)))
        except OSError as e:
            print(f"Error processing image {image_path}: {e}")
            embeddings.append(np.zeros((1, 512)))

    embedding_data = np.vstack(embeddings)
    subject_id = np.array(list(df["subject_id"].values)).reshape(-1, 1)
    return embedding_data, target, valid_indices, subject_id


def get_embedding_from_medclip_and_metadata(df, r_dir, model, preprocess, device):
    print("Extract targets and embedding with medCLIP...")
    target = dict()

    # Encode gender: Male = 1, Female = 0
    df["gender"] = df['gender'].replace({'M': 1, 'F': 0})
    target["gender"] = np.array(list(df["gender"].values)).reshape(-1, 1)

    # Convert 'insurance' column to numerical values
    df["insurance"] = df['insurance'].replace({'Medicare': 0, 'Medicaid': 1, 'Private': 2})
    target["insurance"] = np.array(list(df["insurance"].values)).reshape(-1, 1)

    # Encode age_decile: 20-40 = 0, 40-60 = 1, 60-80 = 2, 80+ = 3
    df["age_decile"] = df['age_decile'].replace({
        '20-40': 0, '40-60': 1, '60-80': 2, '80+': 3
    })
    target["age_decile"] = np.array(list(df["age_decile"].values)).reshape(-1, 1)

    # Encode race into classes without 'UNKNOWN' and 'UNABLE TO OBTAIN'
    df["race-cls"] = df['grouped_race'].replace({
        'White': 0, 'Black': 1, 'Hispanic/Latino': 2,
        'Other': 3
    })
    target["race-cls"] = np.array(list(df["race-cls"].values)).reshape(-1, 1)

    # Encode race into classes without 'UNKNOWN' and 'UNABLE TO OBTAIN'
    df["race-bin"] = df['grouped_race'].replace({
        'White': 0, 'Black': 1, 'Hispanic/Latino': 1,
        'Other': 1
    })
    target["race-bin"] = np.array(list(df["race-bin"].values)).reshape(-1, 1)

    target["disease"] = np.array(list(df["No Finding"].values)).reshape(-1, 1)

    # Extract embedding data
    # Normalize the path for consistency across different operating systems
    df['path'] = df['path'].apply(lambda p: os.path.normpath(p))

    embeddings = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_path = os.path.join(r_dir, "data", "mimic_jpg", row['path'].replace('.tfrecord', '.jpg'))
        try:
            # Manually handle image conversion to RGB
            image = Image.open(image_path).convert('RGB')
            # Preprocess image using the CLIPImageProcessor
            image_tensor = preprocess(images=image, return_tensors="pt")['pixel_values'].to(device)
            with torch.no_grad():
                embedding = model.encode_image(image_tensor)
            embeddings.append(embedding.cpu().numpy())
            valid_indices.append(idx)
        except FileNotFoundError:
            print(f"Image {image_path} not found.")
            embeddings.append(np.zeros((1, 512)))
        except OSError as e:
            print(f"Error processing image {image_path}: {e}")
            embeddings.append(np.zeros((1, 512)))

    embedding_data = np.vstack(embeddings)
    subject_id = np.array(list(df["subject_id"].values)).reshape(-1, 1)
    return embedding_data, target, valid_indices, subject_id


def calculate_metrics(predictions, true_values):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for pred in predictions:
        accuracies.append(accuracy_score(true_values, pred))
        precisions.append(precision_score(true_values, pred, average='macro', zero_division=0))
        recalls.append(recall_score(true_values, pred, average='macro', zero_division=0))
        f1s.append(f1_score(true_values, pred, average='macro', zero_division=0))
    return accuracies, precisions, recalls, f1s


def create_baselines(target_train, target_test):
    target_test = np.transpose(target_test)[0].astype(int)
    target_train = np.transpose(target_train)[0].astype(int)
    cardinality_test = len(target_test)
    cardinality_train = len(target_train)
    n_classes = max(target_train) + 1

    counts = np.bincount(target_train)
    n_folds = 10

    probabilities = counts / cardinality_train

    random_cls = [np.random.randint(0, n_classes, cardinality_test) for _ in range(n_folds)]
    random_cls_ck = [np.random.choice(np.arange(n_classes), size=cardinality_test, p=probabilities) for _ in
                     range(n_folds)]

    random_metrics = calculate_metrics(random_cls, target_test)
    random_ck_metrics = calculate_metrics(random_cls_ck, target_test)

    baseline_results = {
        'random': {
            'accuracy': random_metrics[0],
            'precision': random_metrics[1],
            'recall': random_metrics[2],
            'f1': random_metrics[3]
        },
        'random_cardinality': {
            'accuracy': random_ck_metrics[0],
            'precision': random_ck_metrics[1],
            'recall': random_ck_metrics[2],
            'f1': random_ck_metrics[3]
        }
    }

    return baseline_results


def monitor_embeddings(model, dataset, labels, epoch, save_folder, phase='Training', device='cuda'):
    """
    Funzione per monitorare gli embeddings durante l'addestramento.

    Parameters:
    - model: Il modello PyTorch addestrato
    - dataset: Dataset di input su cui monitorare gli embeddings
    - labels: Etichette corrispondenti al dataset
    - epoch: Epoca corrente
    - save_folder: Cartella in cui salvare le immagini
    - phase: Fase del processo (es. 'Training', 'Validation', 'Test')
    - device: Dispositivo su cui viene eseguito il modello (default: 'cuda')

    Returns:
    - None: Le immagini degli embeddings vengono salvate nella cartella specificata
    """
    model.eval()
    with torch.no_grad():
        dataset_tensor = dataset.clone().detach().unsqueeze(1).float().to(device)
        embeddings = model(dataset_tensor).cpu().numpy()

        # Se gli embeddings hanno più di una dimensione, riducili con PCA
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        # Salva gli embeddings prima e dopo l'addestramento
        plot_embeddings(
            embeddings,
            labels,
            f'{phase} Embeddings (Epoch {epoch})',
            os.path.join(save_folder, f'{phase.lower()}_embeddings_epoch_{epoch}.png'), None,
            save_pca_path=os.path.join(save_folder, f'PCA.pkl')
        )


def plot_embeddings(embeddings, labels, title, save_path, load_pca_path=None, save_pca_path=None):
    """
    Funzione per plottare gli embeddings.

    Parameters:
    - embeddings: Gli embeddings da plottare
    - labels: Etichette corrispondenti agli embeddings
    - title: Titolo del grafico
    - save_path: Percorso per salvare l'immagine

    Returns:
    - None: Salva il grafico degli embeddings come immagine
    """
    if load_pca_path is not None:
        with open(load_pca_path, 'rb') as f:
            pca = load(f)
        embeddings_pca = pca.transform(embeddings)
    else:
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        if save_pca_path is not None:
            with open(save_pca_path, 'wb') as f:
                dump(pca, f)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], hue=labels, palette="deep")
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(save_path)
    plt.close()


def create_gif(image_folder, gif_name="embedding_evolution.gif", duration=500):
    """
    Create a GIF from images in a folder.

    Args:
        image_folder (str): Path to the folder containing images.
        gif_name (str): Name of the output GIF file.
        duration (int): Duration between frames in milliseconds.
    """
    # Get all image files in the folder
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]

    # Load images
    frames = [Image.open(os.path.join(image_folder, img)) for img in images]

    # Save as GIF
    frames[0].save(
        os.path.join(image_folder, gif_name),
        format="GIF",
        append_images=frames[1:],  # List of images to append after the first image
        save_all=True,  # Save all images
        duration=duration,  # Time between frames in milliseconds
        loop=0  # 0 means loop indefinitely
    )

    print(f"GIF saved as {gif_name}")


def plot_embeddings_tsne(embeddings, labels, title, save_path):
    """
    Funzione per plottare gli embeddings.

    Parameters:
    - embeddings: Gli embeddings da plottare
    - labels: Etichette corrispondenti agli embeddings
    - title: Titolo del grafico
    - save_path: Percorso per salvare l'immagine

    Returns:
    - None: Salva il grafico degli embeddings come immagine
    """

    tsne = TSNE(n_components=2, perplexity=30)
    embeddings_tsne = tsne.fit_transform(embeddings, y=labels)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], hue=labels, palette="deep")
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(save_path)
    plt.close()


def plot_embeddings_umap(embeddings, labels, title, save_path):
    """
    Funzione per plottare gli embeddings.

    Parameters:
    - embeddings: Gli embeddings da plottare
    - labels: Etichette corrispondenti agli embeddings
    - title: Titolo del grafico
    - save_path: Percorso per salvare l'immagine

    Returns:
    - None: Salva il grafico degli embeddings come immagine
    """

    tsne = umap.UMAP(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(embeddings, y=labels)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette="deep")
    plt.title(title)
    plt.xlabel('Umap Component 1')
    plt.ylabel('Umap Component 2')
    plt.savefig(save_path)
    plt.close()


def plot_embeddings_pumap(embeddings, labels, title, save_path):
    """
    Funzione per plottare gli embeddings.

    Parameters:
    - embeddings: Gli embeddings da plottare
    - labels: Etichette corrispondenti agli embeddings
    - title: Titolo del grafico
    - save_path: Percorso per salvare l'immagine

    Returns:
    - None: Salva il grafico degli embeddings come immagine
    """

    tsne = ParametricUMAP(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(embeddings, y=labels)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette="deep")
    plt.title(title)
    plt.xlabel('Umap Component 1')
    plt.ylabel('Umap Component 2')
    plt.savefig(save_path)
    plt.close()
