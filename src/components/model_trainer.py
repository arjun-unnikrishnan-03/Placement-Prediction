import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

try:
    from src.exception import CustomException
    from src.logger import logging
    from src.utils import save_object, evaluate_models 
except ImportError:
    class CustomException(Exception):
        def __init__(self, message, sys_info=None):
            super().__init__(message)
            self.sys_info = sys_info
            print(f"Error: {message}")
    class Logger:
        def info(self, message):
            print(f"INFO: {message}")
        def error(self, message):
            print(f"ERROR: {message}")
    logging = Logger()
    def save_object(file_path, obj):
        import pickle
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
            logging.info(f"Object saved to {file_path}")
        except Exception as e:
            raise CustomException(f"Error saving object: {e}", sys)

    def evaluate_models(X_train, y_train, X_test, y_test, models, param):
        report = {}
        for i, (model_name, model) in enumerate(models.items()):
            logging.info(f"Training and evaluating model: {model_name}")
            model_params = param.get(model_name, {}) 
            from sklearn.model_selection import GridSearchCV
            gs = GridSearchCV(model, model_params, cv=3, scoring='f1', n_jobs=-1, verbose=0) 
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_f1
            logging.info(f"Model {model_name} trained. Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return report

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
                "XGBClassifier": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False, random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "K-Neighbors Classifier": KNeighborsClassifier(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'], 
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                "Gradient Boosting": {
                    'learning_rate': [.01, .05, .1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },
                "XGBClassifier": {
                    'learning_rate': [.01, .05, .1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                },
                "CatBoosting Classifier": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [.01, .05, .1],
                    'n_estimators': [50, 100, 200],
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)
            print("\nModel Report (Test F1-Scores):")
            print(model_report)
            print("-------------------------------")
   
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with F1-score >= 0.6. Consider adjusting threshold or models/params.")
            logging.info(f"Best found model on both training and testing dataset: {best_model_name} with F1-score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            final_metric = f1_score(y_test, predicted)
            logging.info(f"Best model '{best_model_name}' achieved a test F1-score of {final_metric:.4f}")
            return final_metric

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    num_samples_train = 80
    num_samples_test = 20
    num_features = 8

    mock_X_train = np.random.randn(num_samples_train, num_features)
    mock_y_train = np.random.randint(0, 2, num_samples_train)
    mock_train_arr = np.c_[mock_X_train, mock_y_train]

    mock_X_test = np.random.randn(num_samples_test, num_features)
    mock_y_test = np.random.randint(0, 2, num_samples_test)
    mock_test_arr = np.c_[mock_X_test, mock_y_test]

    logging.info("Mock train_arr and test_arr created for ModelTrainer testing.")
    print(f"Mock train_arr shape: {mock_train_arr.shape}")
    print(f"Mock test_arr shape: {mock_test_arr.shape}")

    modeltrainer = ModelTrainer()
    best_model_f1_score = modeltrainer.initiate_model_trainer(mock_train_arr, mock_test_arr)
    print(f"\nOverall best model's F1-score on test set: {best_model_f1_score:.4f}")
