import gc
import json
import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add the parent directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), os.pardir)))

# Import utility functions and model definitions from your project
from utils import (
    plot_loss_acc_cv, remove_correlated_features,
    get_embedding_and_metadata_chexpert, plot_roc_curves, plot_boxplot
)
from models import create_mlp

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# ----------------------------------------------------------------
# PATHS AND DATA MERGING (CheXpert-specific)
# ----------------------------------------------------------------
root_dir = os.path.dirname(os.path.dirname('__file__'))
chexpert_data_folder = os.path.join(root_dir, "data", "chexpert")
embedding_folder = os.path.join(chexpert_data_folder, "biomedclip")
extracted_embedding_folder = os.path.join(chexpert_data_folder, "biomedclip")
os.makedirs(extracted_embedding_folder, exist_ok=True)

# Paths to metadata CSV and embedding pickle
metadata_csv = os.path.join(chexpert_data_folder, 'metadata.csv')
embed_pkl = os.path.join(embedding_folder, "embedding_from_biomedclip_Chexpert.pkl")

# List of metadata columns (classification tasks)
metas = ["gender", "age_decile", "race-bin", "race-cls", "disease"]

print("Loading CheXpert metadata...")
df_meta = pd.read_csv(metadata_csv)

print("Loading CheXpert embeddings from BiomedCLIP...")
with open(embed_pkl, 'rb') as f:
    embed_tuple = pickle.load(f)

# The tuple structure is as follows:
#   element 0: list of embeddings
#   element 2: list of indices for each instance (not used here)
#   element 3: list of paths
embeddings_list = embed_tuple[0]
indices_list = embed_tuple[2]  # (optional; not used in this merge)
paths_list = embed_tuple[3]


# Create a DataFrame from embeddings and paths.
def extract_path(p):
    """Recursively extract the innermost element from a nested list and return it as a string."""
    # If p is a numpy array, convert to list
    if isinstance(p, np.ndarray):
        p = p.tolist()
    # If p is a list and non-empty, extract the first element
    if isinstance(p, list) and len(p) > 0:
        # Check recursively in case the element is still nested
        return extract_path(p[0].replace("-small", ""))
    else:
        return str(p).replace("-small", "")


# Process embeddings and paths
embeddings_fixed = [np.array(e).flatten() for e in embeddings_list]
paths_fixed = [extract_path(p) for p in paths_list]

df_embed = pd.DataFrame({
    "embedding": embeddings_fixed,
    "Path": paths_fixed
})

print("Merging metadata and embeddings on 'Path'...")
df_merged = pd.merge(df_meta, df_embed, on='Path', how='inner')
df_merged.rename(columns={"Path": "path", "PATIENT": "subject_id"}, inplace=True)
df_merged.drop(columns={"Unnamed: 0"}, inplace=True)
df_merged.info()

print("Extracting embeddings & metadata...")
# This function should return:
#    X_all: a 2D np.array of all embeddings,
#    y_dict: a dictionary of label arrays (one per metadata task),
#    all_ids: an array of patient IDs (one per row) used for splitting
X_all, y_dict, all_ids = get_embedding_and_metadata_chexpert(
    df_merged
)

print("Dataset shapes:")
print("  X_all:", X_all.shape)
for key in y_dict:
    print(f"  y_dict['{key}'] shape:", y_dict[key].shape)

# ----------------------------------------------------------------
# SPLIT DATA BY PATIENT ID
# ----------------------------------------------------------------
extracted_id_folder = os.path.join(chexpert_data_folder, "medclip")  # Use the same train/test split of Medclip
id_split_path = os.path.join(extracted_id_folder, "train_test_ids.pkl")

if os.path.exists(id_split_path):
    print("Loading saved train/test IDs...")
    with open(id_split_path, "rb") as f:
        train_all_ids, test_all_ids = pickle.load(f)
    train_indices = [i for i, pid in enumerate(all_ids) if pid in train_all_ids]
    test_indices = [i for i, pid in enumerate(all_ids) if pid in test_all_ids]
else:
    print("No saved train/test IDs found. Computing new split...")
    unique_ids = np.unique(all_ids)
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.20, random_state=seed)
    train_indices = [i for i, pid in enumerate(all_ids) if pid in train_ids]
    test_indices = [i for i, pid in enumerate(all_ids) if pid in test_ids]
    train_all_ids = np.array([pid for i, pid in enumerate(all_ids) if pid in train_ids])
    test_all_ids = np.array([pid for i, pid in enumerate(all_ids) if pid in test_ids])
    with open(id_split_path, "wb") as f:
        pickle.dump((train_all_ids, test_all_ids), f)
    print("Train/test IDs saved to", id_split_path)

X_train, X_test = X_all[train_indices], X_all[test_indices]

# Split the labels per task accordingly
y_train_dict = {}
y_test_dict = {}
for meta in metas:
    y_train_dict[meta] = y_dict[meta][train_indices]
    y_test_dict[meta] = y_dict[meta][test_indices]

# ----------------------------------------------------------------
# TRAINING & CROSS-VALIDATION PER METADATA TASK
# ----------------------------------------------------------------
# Hyperparameters for model training
K = 10
EPOCHS = 100
BATCH_SIZE = 256
learning_rate = 0.01

# Loop over each metadata prediction task
for meta in metas:
    print(f"\n========== Processing metadata: {meta} ==========")

    # Set up directories for saving data, models, and figures
    data_folder = os.path.join(extracted_embedding_folder, meta)
    models_folder = os.path.join(root_dir, "models", "chexpert", "biomedclip", meta)
    figures_folder = os.path.join(root_dir, "fig", "chexpert", "biomedclip", f"predict-{meta}")

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)

    # Paths to store preprocessed datasets
    train_dataset_path = os.path.join(data_folder, 'train_dataset_std.pkl')
    test_dataset_path = os.path.join(data_folder, 'test_dataset_std.pkl')

    if os.path.exists(train_dataset_path):
        print("Loaded preprocessed train dataset from pickle file.")
        with open(train_dataset_path, 'rb') as f:
            X_train_std, y_train, saved_train_ids = pickle.load(f)
        # Use the saved patient IDs for CV splitting:
        train_all_ids = saved_train_ids
    else:
        # Compute train_all_ids as before
        train_all_ids = np.array([pid for i, pid in enumerate(all_ids) if i in train_indices])

        # Remove correlated features using only training data
        print("Removing correlated features...")
        X_train_clean, removed_features = remove_correlated_features(X_train, threshold=0.75,
                                                                     figures_folder=figures_folder)
        # Apply the same feature removal to test set
        X_test_clean = np.delete(X_test, removed_features, axis=1)

        print("Features removed:", removed_features)
        with open(os.path.join(data_folder, 'removed_features.pkl'), 'wb') as f:
            pickle.dump(removed_features, f)

        # Data normalization using StandardScaler
        print("Standardizing data...")
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_clean)
        X_test_std = scaler.transform(X_test_clean)

        # Save the scaler for future use
        with open(os.path.join(data_folder, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        # Save the preprocessed train and test splits
        with open(train_dataset_path, 'wb') as f:
            pickle.dump((X_train_std, y_train_dict[meta], train_all_ids), f)
        print("Train dataset saved.")
        with open(test_dataset_path, 'wb') as f:
            pickle.dump((X_test_std, y_test_dict[meta]), f)
        print("Test dataset saved.")

        y_train = y_train_dict[meta]
        y_train = y_train.ravel()

    # Define the models to evaluate
    models = {
        'logistic_regression': LogisticRegression(max_iter=5000),
        'random_forest': RandomForestClassifier(n_estimators=250, max_depth=8,
                                                min_samples_leaf=10, random_state=seed,
                                                n_jobs=-1),
        'xgboost': XGBClassifier(eval_metric='logloss', random_state=seed),
        'mlp': 'mlp'
    }

    # Update the results dictionary to include roc_auc
    results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []} for name in models.keys()}
    train_accuracies, val_accuracies, train_losses, val_losses = [], [], [], []
    X_val_folds, y_val_folds = [], []

    # Set up K-Fold cross-validation based on unique patient IDs in the training split.
    train_unique_ids = np.unique(train_all_ids)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    # For each model and each CV fold, split the training set by patient ID
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name} for metadata: {meta}")
        fold_num = 1
        for train_fold_idx, val_fold_idx in kf.split(train_unique_ids):
            # Get the patient IDs for this fold
            fold_train_patient_ids = train_unique_ids[train_fold_idx]
            fold_val_patient_ids = train_unique_ids[val_fold_idx]

            # Find indices in train_all_ids that belong to the current fold’s patients
            cv_train_indices = [i for i, pid in enumerate(train_all_ids) if pid in fold_train_patient_ids]
            cv_val_indices = [i for i, pid in enumerate(train_all_ids) if pid in fold_val_patient_ids]

            print("Length of X_train_std:", X_train_std.shape[0])
            print("Max index in cv_train_indices:", max(cv_train_indices) if cv_train_indices else None)

            X_train_fold = X_train_std[cv_train_indices]
            X_val_fold = X_train_std[cv_val_indices]
            y_train_fold = y_train_dict[meta][cv_train_indices]
            y_val_fold = y_train_dict[meta][cv_val_indices]

            y_train_fold = y_train_fold.ravel()
            y_val_fold = y_val_fold.ravel()

            # Save validation folds for later ROC curve plotting
            X_val_folds.append(X_val_fold)
            y_val_folds.append(y_val_fold)

            # Determine number of classes in the current fold
            n_class = len(np.unique(y_train_fold))

            if model_name == 'mlp':
                input_dim = X_train_fold.shape[1]
                model_instance = create_mlp(input_dim, n_class=n_class)
                if n_class == 2:
                    loss = 'binary_crossentropy'
                else:
                    # Convert labels to categorical if multiclass
                    y_train_fold_cat = to_categorical(y_train_fold, num_classes=n_class)
                    y_val_fold_orig = y_val_fold
                    y_val_fold_cat = to_categorical(y_val_fold, num_classes=n_class)
                    loss = 'categorical_crossentropy'

                model_instance.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                       loss=loss,
                                       metrics=['accuracy', 'AUC'])

                # Define callbacks for early stopping and learning rate reduction
                checkpoint = ModelCheckpoint(os.path.join(models_folder, f'mlp_fold_{fold_num}.keras'),
                                             monitor='val_loss', save_best_only=True, mode='min')
                early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

                # Train the MLP model
                if n_class == 2:
                    history = model_instance.fit(X_train_fold, y_train_fold,
                                                 validation_data=(X_val_fold, y_val_fold),
                                                 epochs=EPOCHS,
                                                 batch_size=BATCH_SIZE,
                                                 callbacks=[checkpoint, early_stopping, reduce_lr],
                                                 verbose=0)
                    y_pred_prob = model_instance.predict(X_val_fold)
                    # For binary classification, threshold probabilities at 0.5
                    y_pred = (y_pred_prob > 0.5).astype(int)
                    y_pred_prob = y_pred_prob.ravel()  # For ROC-AUC computation
                    roc_auc = roc_auc_score(y_val_fold, y_pred_prob)
                else:
                    history = model_instance.fit(X_train_fold, y_train_fold_cat,
                                                 validation_data=(X_val_fold, y_val_fold_cat),
                                                 epochs=EPOCHS,
                                                 batch_size=BATCH_SIZE,
                                                 callbacks=[checkpoint, early_stopping, reduce_lr],
                                                 verbose=0)
                    y_pred_prob = model_instance.predict(X_val_fold)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    y_val_fold = y_val_fold_orig
                    roc_auc = roc_auc_score(y_val_fold, y_pred_prob, multi_class='ovr', average='macro')

                # Save training history for CV plots
                train_accuracies.append(history.history['accuracy'])
                val_accuracies.append(history.history['val_accuracy'])
                train_losses.append(history.history['loss'])
                val_losses.append(history.history['val_loss'])

                del model_instance
                gc.collect()
            else:
                # For classical machine learning models
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                y_pred_prob = model.predict_proba(X_val_fold)

                # For binary classification, use probability for the positive class (column index 1)
                if n_class == 2:
                    roc_auc = roc_auc_score(y_val_fold, y_pred_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_val_fold, y_pred_prob, multi_class='ovr', average='macro')

                # Save the trained model for this fold
                with open(os.path.join(models_folder, f'{model_name}_fold_{fold_num}.pkl'), 'wb') as f:
                    pickle.dump(model, f)

            # Calculate performance metrics for the current fold
            acc = accuracy_score(y_val_fold, y_pred)
            precision = precision_score(y_val_fold, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_val_fold, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)

            results[model_name]['accuracy'].append(acc)
            results[model_name]['precision'].append(precision)
            results[model_name]['recall'].append(recall)
            results[model_name]['f1'].append(f1)
            results[model_name]['roc_auc'].append(roc_auc)

            print(
                f"Fold {fold_num} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            fold_num += 1

    # Plot cross-validation loss/accuracy for MLP (if applicable)
    if 'mlp' in models:
        plot_loss_acc_cv(train_accuracies, val_accuracies, 'Accuracy', figures_folder)
        plot_loss_acc_cv(train_losses, val_losses, 'Loss', figures_folder)

    # Save CV results to a JSON file
    cv_result_path = os.path.join(figures_folder, 'performance_cv.json')
    with open(cv_result_path, 'w') as fp:
        json.dump(results, fp)
    print(f"Saved CV results to {cv_result_path}")

    # Plot boxplots and ROC curves for the CV folds
    plot_boxplot(results, figures_folder, suffix='cv')
    plot_roc_curves(models_folder=models_folder, model_names=models.keys(),
                    X_folds=X_val_folds, y_folds=y_val_folds,
                    figures_folder=figures_folder, suffix='cv')

    print(f"Completed processing for metadata: {meta}")

print("\nAll tasks completed.")
