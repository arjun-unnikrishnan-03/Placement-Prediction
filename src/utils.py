import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV

# Import classification metrics (CHANGED from regression metrics)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Placeholder for CustomException (assuming it's in src.exception)
# This block is for standalone execution/testing; in your actual project,
# 'from src.exception import CustomException' should work directly if setup.
try:
    from src.exception import CustomException
except ImportError:
    class CustomException(Exception):
        def __init__(self, message, sys_info=None):
            super().__init__(message)
            self.sys_info = sys_info
            print(f"Error: {message}")

# --- Utility functions (structure unchanged) ---

def save_object(file_path, obj):
    """
    Saves a Python object to a specified file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        # Assuming you have a logging setup in your project (e.g., from src.logger)
        # logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple classification models using GridSearchCV for hyperparameter tuning.
    Returns a dictionary of model names and their test F1-scores.
    (Refactored for Classification Project)
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            parameters = param.get(model_name, {}) # Use .get() to safely retrieve params for a model

            print(f"\n--- Evaluating Model: {model_name} ---")

            # GridSearchCV for hyperparameter tuning
            # 'scoring' is now set to 'f1' for classification problems.
            # You can change 'f1' to 'roc_auc' or 'accuracy' if preferred.
            gs = GridSearchCV(model, parameters, cv=3, scoring='f1', n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Set the model to its best parameters found by GridSearchCV
            model.set_params(**gs.best_params_)

            # Train the model with the best parameters on the full training data
            model.fit(X_train, y_train)

            # Make predictions on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate classification metrics (CHANGED from r2_score)
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            # zero_division=0 handles cases where there are no positive predictions or true positives
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)

            # Attempt to calculate ROC-AUC if the model supports predict_proba
            try:
                if hasattr(model, "predict_proba"):
                    y_test_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class
                    test_roc_auc = roc_auc_score(y_test, y_test_proba)
                else:
                    test_roc_auc = None # Model does not have predict_proba
            except Exception as e:
                test_roc_auc = None
                print(f"Warning: Could not calculate ROC-AUC for {model_name}: {e}")

            # Store the primary evaluation metric (Test F1-Score) in the report
            report[model_name] = test_f1

            print(f"  Best parameters: {gs.best_params_}")
            print(f"  Train F1-Score: {train_f1:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test Precision: {test_precision:.4f}")
            print(f"  Test Recall: {test_recall:.4f}")
            print(f"  Test F1-Score: {test_f1:.4f}")
            if test_roc_auc is not None:
                print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
            else:
                print(f"  Test ROC-AUC: N/A (Model does not support predict_proba)")

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a specified file path using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)