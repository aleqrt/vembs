import gc
import json
import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pickle import dump, load

# Add the parent directory to the system path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), os.pardir)))

# Import utility functions
from utils import plot_loss_acc_cv, remove_correlated_features, \
    get_embedding_and_metadata, plot_roc_curves, plot_boxplot
from models import create_mlp

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Paths
root_dir = os.path.dirname(os.path.dirname('__file__'))  # Root directory of the project

metas = ["gender", "race-cls", "race-bin", "age_decile", "insurance", "disease"]

extracted_embedding_folder = os.path.join(root_dir, "data",  "mimic", "cxr-foundation")
train_test_idx_path = os.path.join(extracted_embedding_folder, "train_test_idx.pkl")
all_embedding_metadata_path = os.path.join(extracted_embedding_folder, "embedding_metadata.pkl")
df_path = os.path.join(root_dir, "data", "mimic", 'df_final_for_metadata_prediction.csv')

os.makedirs(extracted_embedding_folder, exist_ok=True)

for metadata in metas:
    data_folder = os.path.join(root_dir, extracted_embedding_folder, metadata)
    models_folder = os.path.join(root_dir, "models", "mimic", "cxr-foundation", metadata)
    figures_folder = os.path.join(root_dir, "fig", "mimic", "cxr-foundation", f"predict-{metadata}")

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)

    # Load data
    print(f"Start {metadata}...")

    # Path for save train/test split
    train_dataset_path = os.path.join(data_folder, 'train_dataset_std.pkl')
    test_dataset_path = os.path.join(data_folder, 'test_dataset_std.pkl')

    if os.path.exists(train_dataset_path):
        print("Loaded dataset for training from pickle file.")
        with open(train_dataset_path, 'rb') as f:
            X_train_std, y_train, train_all_ids = load(f)
        train_ids = np.unique(train_all_ids)
    else:
        # Select patients for training
        print(f'Select patients...')

        data_df = pd.read_csv(df_path)
        
        data_df.drop_duplicates(subset='subject_id', inplace=True)
        data_df = data_df.reset_index()

        # get unique subject ids
        unique_ids = np.array(list(data_df["subject_id"].unique())).reshape(-1, 1)
        # Load embedding and target
        print('Load embedding and target...')
        if os.path.exists(all_embedding_metadata_path):
            with open(all_embedding_metadata_path, 'rb') as f:
                X, y, ids = load(f)
        else:
            # Get embeddings
            X, y, ids = get_embedding_and_metadata(data_df, data_folder, root_dir)

            print('Saving embedding and target variables in raw format...')
            with open(all_embedding_metadata_path, 'wb') as f:
                dump((X, y, ids), f)

        if os.path.exists(train_test_idx_path):
            with open(train_test_idx_path, 'rb') as f:
                train_indices, test_indices, train_all_ids, test_all_ids = load(f)
            print("train and test indices loaded from pickle file.")
            train_ids = np.unique(train_all_ids)
            test_ids = np.unique(test_all_ids)
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[metadata][train_indices], y[metadata][test_indices]
        else:
            ####
            train_ids, test_ids = train_test_split(unique_ids, test_size=0.20, random_state=seed)
            train_indices = [i for i, value in enumerate(ids) if value in train_ids]
            test_indices = [i for i, value in enumerate(ids) if value in test_ids]
            train_all_ids = [value for i, value in enumerate(ids) if value in train_ids]
            test_all_ids = [value for i, value in enumerate(ids) if value in test_ids]
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[metadata][train_indices], y[metadata][test_indices]
            #####

            print('Save train and test indices...')
            with open(train_test_idx_path, 'wb') as f:
                dump((train_indices, test_indices, train_all_ids, test_all_ids), f)

        print('Train set - Max value:', X_train.max(), ' Min value:', X_train.min())
        print('Train set -  shape:', X_train.shape)
        print('Train set -  classes:', np.unique(y_train))

        print('Test set - Max value:', X_test.max(), ' Min value:', X_test.min())
        print('Test set -  shape:', X_test.shape)
        print('Test set -  classes:', np.unique(y_test))

        # Remove correlated features
        print('Remove correlated features...')
        X_train_clean, removed_features = remove_correlated_features(X_train,
                                                                     threshold=0.75,
                                                                     figures_folder=figures_folder)
        X_test_clean = np.delete(X_test, removed_features, axis=1)

        print('features removed:', removed_features)
        removed_features_path = os.path.join(data_folder, 'removed_features.pkl')
        with open(removed_features_path, 'wb') as f:
            dump(removed_features, f)

        # Data normalization
        print('Standardization process...')
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_clean)
        X_test_std = scaler.transform(X_test_clean)

        print('Train set after processing - Max value:', X_train_std.max(), ' Min value:', X_train_std.min())
        print('Train set after processing -  shape:', X_train_std.shape)
        print('Train set after processing -  classes:', np.unique(y_train))

        print('Test set after processing - Max value:', X_test_std.max(), ' Min value:', X_test_std.min())
        print('Test set after processing -  shape:', X_test_std.shape)
        print('Test set after processing -  classes:', np.unique(y_test))

        # Save the scaler
        with open(os.path.join(data_folder, 'scaler.pkl'), 'wb') as f:
            dump(scaler, f)

        # Save train_set as pickle
        with open(train_dataset_path, 'wb') as f:
            dump((X_train_std, y_train, train_all_ids), f)
        print("Train split saved to pickle file.")

        # Save test_set as pickle
        with open(test_dataset_path, 'wb') as f:
            dump((X_test_std, y_test), f)
        print("Test split saved to pickle file.")

    ###############################################################################
    ###############################################################################
    # Define the models to evaluate
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                                min_samples_leaf=10, random_state=seed,
                                                n_jobs=-1),
        'xgboost': XGBClassifier(eval_metric='logloss', random_state=seed),
        'mlp': 'mlp'
    }

    # K-Fold Cross Validation
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    # Metrics and histories for CV
    results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for name in models.keys()}
    train_accuracies, val_accuracies, train_losses, val_losses = [], [], [], []
    X_val_folds, y_val_folds = [], []

    # Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 256
    learning_rate = 0.01

    print(f'{K}-fold cross validation...')

    for model_name, model in models.items():
        print(f'Training {model_name}...')
        for i, (train_ids_idx, val_ids_idx) in enumerate(kf.split(train_ids), 1):
            print(f'Fold {i}...')
            train_index = [j for j, value in enumerate(train_all_ids) if value in train_ids[train_ids_idx]]
            val_index = [j for j, value in enumerate(train_all_ids) if value in train_ids[val_ids_idx]]
            X_train_fold, X_val_fold = X_train_std[train_index], X_train_std[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # store validation folds
            X_val_folds.append(X_val_fold)
            y_val_folds.append(y_val_fold)

            # get the number of classes
            n_class = len(np.unique(y_train_fold))

            if len(y_train_fold.shape) > 1 and y_train_fold.shape[1] == 1:
                y_train_fold = y_train_fold.ravel()
            if len(y_val_fold.shape) > 1 and y_val_fold.shape[1] == 1:
                y_val_fold = y_val_fold.ravel()

            if model_name == 'mlp':
                input_dim = X_train_fold.shape[1]
                model = create_mlp(input_dim, n_class=n_class)

                if n_class == 2:
                    loss = 'binary_crossentropy'
                else:
                    y_train_fold = to_categorical(y_train_fold, num_classes=n_class)
                    y_val_fold_orig = y_val_fold
                    y_val_fold = to_categorical(y_val_fold, num_classes=n_class)
                    loss = 'categorical_crossentropy'

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                              loss=loss,
                              metrics=['accuracy', 'AUC'])

                # Define callbacks
                checkpoint = ModelCheckpoint(os.path.join(models_folder, f'mlp_fold_{i}.keras'),
                                             monitor='val_loss',
                                             save_best_only=True, mode='min')
                early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

                # Train the model
                history = model.fit(X_train_fold, y_train_fold,
                                    validation_data=(X_val_fold, y_val_fold),
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    callbacks=[checkpoint, early_stopping, reduce_lr])

                # Append history for CV plots
                train_accuracies.append(history.history['accuracy'])
                val_accuracies.append(history.history['val_accuracy'])
                train_losses.append(history.history['loss'])
                val_losses.append(history.history['val_loss'])

                # Evaluate the model on the validation set
                y_pred_prob = model.predict(X_val_fold)

                if n_class == 2:
                    # Binary classification
                    y_pred = (y_pred_prob > 0.5).astype(int)
                    y_pred_prob = y_pred_prob.ravel()  # For ROC-AUC
                else:
                    # Multiclass classification
                    y_pred = np.argmax(y_pred_prob, axis=1)
                    y_val_fold = y_val_fold_orig

                del model
                gc.collect()
            else:
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                y_pred_prob = model.predict_proba(X_val_fold)

                with open(os.path.join(models_folder, f'{model_name}_fold_{i}.pkl'), 'wb') as f:
                    dump(model, f)

            # Calculate metrics
            results[model_name]['accuracy'].append(float(accuracy_score(y_val_fold, y_pred)))
            results[model_name]['precision'].append(
                float(precision_score(y_val_fold, y_pred, average='macro', zero_division=0)))
            results[model_name]['recall'].append(
                float(recall_score(y_val_fold, y_pred, average='macro', zero_division=0)))
            results[model_name]['f1'].append(float(f1_score(y_val_fold, y_pred, average='macro', zero_division=0)))

    # Plot CV results for MLP
    if 'mlp' in models:
        plot_loss_acc_cv(train_accuracies, val_accuracies, 'Accuracy', figures_folder)
        plot_loss_acc_cv(train_losses, val_losses, 'Loss', figures_folder)

    # Print metrics for validation
    print('Plotting boxplot and ROC curves train...')
    with open(os.path.join(figures_folder, 'result_cv.json'), 'w') as fp:
        json.dump(results, fp)

    # Plot boxplots and AUC-ROC curves
    plot_boxplot(results, figures_folder, suffix='cv')
    plot_roc_curves(models_folder=models_folder, model_names=models.keys(),
                    X_folds=X_val_folds, y_folds=y_val_folds,
                    figures_folder=figures_folder, suffix='cv')
