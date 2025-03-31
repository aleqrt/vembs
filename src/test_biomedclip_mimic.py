import os
import sys
import numpy as np
from pickle import load
import joblib
import tensorflow as tf
import json

# Add the parent directory to the system path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), os.pardir)))

# Import utility functions
from utils import evaluate_models, plot_roc_curves_test, plot_boxplot, create_baselines
from models import create_mlp

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Metadata
metas = ["race-bin", "race-cls", "gender", "age_decile", "insurance", "disease"]
# metas = ["disease"]

for metadata in metas:
    root_dir = os.path.dirname(os.path.dirname('__file__'))
    data_folder = os.path.join(root_dir, "data", "mimic", "biomedclip-embedding", metadata)
    figures_folder = os.path.join(root_dir, "fig", "mimic", "biomedclip-embedding", f"predict-{metadata}")
    models_folder = os.path.join(root_dir, "models", "mimic", "biomedclip-embedding", metadata)

    # Load train data
    print(f"Loading BiomedCLIP {metadata} train split data...")
    train_set_path = os.path.join(data_folder, 'train_dataset_std.pkl')
    with open(train_set_path, 'rb') as f:
        X_train, y_train, train_all_ids, train_all_ids_f = load(f)

    print(f'{metadata} train set after processing - Max value:', X_train.max(), ' Min value:', X_train.min())
    print(f'{metadata} train set after processing -  shape:', X_train.shape)
    print(f'{metadata} train set after processing -  classes:', np.unique(y_train))

    # Load test data
    print(f"Loading BiomedCLIP {metadata} test split data...")
    test_set_path = os.path.join(data_folder, 'test_dataset_std.pkl')
    with open(test_set_path, 'rb') as f:
        X_test, y_test = load(f)

    print(f'{metadata} test set after processing - Max value:', X_test.max(), ' Min value:', X_test.min())
    print(f'{metadata} test set after processing -  shape:', X_test.shape)
    print(f'{metadata} test set after processing -  classes:', np.unique(y_test))

    # Load models for each fold
    print(f'Load {metadata} models...')
    models = {
        'logistic_regression': [],
        'random_forest': [],
        'xgboost': [],
        'mlp': []
    }
    n_class = len(np.unique(y_test))
    for model_name in models.keys():
        for i in range(1, 11):
            if model_name == 'mlp':
                model = create_mlp(X_test.shape[1], n_class=n_class)
                model.load_weights(os.path.join(models_folder, f'{model_name}_fold_{i}.keras'))
                models[model_name].append(model)
            else:
                model_path = os.path.join(models_folder, f'{model_name}_fold_{i}.pkl')
                models[model_name].append(joblib.load(model_path))

    # Collect results
    print(f'Compute metrics results for {metadata}...')
    results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc': []} for name in models.keys()}

    # Compute Confusion matrix and print on console results
    print(f'Plot boxplot and ROC-AUC for {metadata}...')
    for model_name in models:
        test_accuracy, test_precision, test_recall, test_f1, test_roc = evaluate_models(models, model_name,
                                                                                        X_test, y_test, n_class,
                                                                                        figures_folder)
        results[model_name]['accuracy'] = test_accuracy
        results[model_name]['precision'] = test_precision
        results[model_name]['recall'] = test_recall
        results[model_name]['f1'] = test_f1
        results[model_name]['roc'] = test_roc

    # Compute random model performance
    baseline_results = create_baselines(y_train, y_test)

    # Add random model results to existing results
    results.update(baseline_results)

    # Save and plot the results
    with open(os.path.join(figures_folder, 'result_test.json'), 'w') as fp:
        json.dump(results, fp)

    # plot_boxplot(results, figures_folder, suffix='test')
    # plot_roc_curves_test(models, X_test, y_test,
    #                      figures_folder, suffix='test')
