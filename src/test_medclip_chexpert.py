import os
import sys
import numpy as np
from pickle import load
import joblib
import tensorflow as tf
import json

# Add parent folder to module import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), os.pardir)))

# Import utility functions and model definition
from utils import compute_confidence_interval, evaluate_models, plot_roc_curves_test, plot_boxplot, create_baselines
from models import create_mlp

# Set the seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# The metadata tasks used for CheXpert
metas = ["gender", "age_decile", "race-bin", "race-cls", "disease"]

for metadata in metas:
    # Define folders for MedCLIP
    root_dir = os.path.dirname(os.path.dirname('__file__'))
    data_folder = os.path.join(root_dir, "data", "chexpert", "medclip", metadata)
    figures_folder = os.path.join(root_dir, "fig", "chexpert", "medclip", f"predict-{metadata}")
    models_folder = os.path.join(root_dir, "models", "chexpert", "medclip", metadata)

    # Load training data (saved in pickle format)
    print(f"Loading {metadata} train split data...")
    train_set_path = os.path.join(data_folder, 'train_dataset_std.pkl')
    with open(train_set_path, 'rb') as f:
        X_train, y_train, train_all_ids = load(f)

    print(f'{metadata} train set - Max value:', X_train.max(), 'Min value:', X_train.min())
    print(f'{metadata} train set - shape:', X_train.shape)
    print(f'{metadata} train set - classes:', np.unique(y_train))

    # Load test data (saved in pickle format)
    print(f"Loading {metadata} test split data...")
    test_set_path = os.path.join(data_folder, 'test_dataset_std.pkl')
    with open(test_set_path, 'rb') as f:
        X_test, y_test = load(f)

    print(f'{metadata} test set - Max value:', X_test.max(), 'Min value:', X_test.min())
    print(f'{metadata} test set - shape:', X_test.shape)
    print(f'{metadata} test set - classes:', np.unique(y_test))

    # Load models (trained on MedCLIP embeddings), for each fold
    print(f'Loading {metadata} models...')
    models = {
        'logistic_regression': [],
        'random_forest': [],
        'xgboost': [],
        'mlp': []
    }
    n_class = len(np.unique(y_test))
    for model_name in models.keys():
        for i in range(1, 11):  # assuming 10-fold training
            if model_name == 'mlp':
                model = create_mlp(X_test.shape[1], n_class=n_class)
                model.load_weights(os.path.join(models_folder, f'{model_name}_fold_{i}.keras'))
                models[model_name].append(model)
            else:
                model_path = os.path.join(models_folder, f'{model_name}_fold_{i}.pkl')
                models[model_name].append(joblib.load(model_path))

    # Collect results
    print(f'Evaluating models for {metadata}...')
    results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc': []} for name in models.keys()}

    # Evaluate each model on the test set
    for model_name in models:
        (test_accuracy, test_precision,
         test_recall, test_f1,
         test_roc) = evaluate_models(models, model_name, X_test, y_test, n_class, figures_folder)

        results[model_name]['accuracy'] = test_accuracy
        results[model_name]['precision'] = test_precision
        results[model_name]['recall'] = test_recall
        results[model_name]['f1'] = test_f1
        results[model_name]['roc'] = test_roc

    # Process raw results to get mean and confidence intervals
    results_with_ci = {model_name: {} for model_name in models.keys()}
    for model_name, metrics in results.items():
        for metric_name, scores in metrics.items():
            if scores: # Ensure scores list is not empty
                mean_score, ci = compute_confidence_interval(scores)
                results_with_ci[model_name][metric_name] = {
                    'mean': mean_score,
                    'confidence_interval': list(ci)
                }
            else:
                results_with_ci[model_name][metric_name] = {
                    'mean': None,
                    'confidence_interval': [None, None]
                }

    # Compute random model performance as a baseline
    #baseline_results = create_baselines(y_train, y_test)
    #results.update(baseline_results)

    # Save results in JSON
    with open(os.path.join(figures_folder, 'performance_test.json'), 'w') as fp:
        json.dump(results, fp)
    
    # Save results in JSON
    with open(os.path.join(figures_folder, 'confidence_intervals_test.json'), 'w') as fp:
        json.dump(results_with_ci, fp)

    # Optionally, plot boxplots and ROC curves
    # plot_boxplot(results, figures_folder, suffix='test')
    # plot_roc_curves_test(models, X_test, y_test, figures_folder, suffix='test')

    print(f"Completed testing for metadata: {metadata}")

print("\nAll tests completed.")
