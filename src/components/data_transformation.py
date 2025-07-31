import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder # Import LabelEncoder for target encoding

# Placeholder for custom exception and logging if you don't have src folder set up
try:
    from src.exception import CustomException
    from src.logger import logging
    from src.utils import save_object # Assuming save_object is defined in src/utils.py
except ImportError:
    class CustomException(Exception):
        def __init__(self, message, sys_info=None):
            super().__init__(message)
            self.sys_info = sys_info
            print(f"Error: {message}") # Basic error handling for demonstration
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

    print("Warning: 'src.exception', 'src.logger', or 'src.utils' not found. Using placeholder classes. "
          "Please ensure your 'src' directory is properly configured for your project.")


import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        It defines preprocessing pipelines for numerical and categorical features.
        '''
        try:
            # --- MODIFIED COLUMN NAMES FOR COLLEGE PLACEMENT DATASET ---
            # Numerical columns based on df.info() and df.describe() from previous inspection
            # Academic_Performance, Extra_Curricular_Score, Communication_Skills, Projects_Completed are int64,
            # but are scores/counts. Treating them as numerical for scaling is typical.
            numerical_columns = [
                "IQ",
                "Prev_Sem_Result",
                "CGPA",
                "Academic_Performance",
                "Extra_Curricular_Score",
                "Communication_Skills",
                "Projects_Completed"
            ]
            # Categorical columns based on df.info() from previous inspection
            categorical_columns = [
                "Internship_Experience"
            ]
            # -------------------------------------------------------------

            # Numerical Pipeline (only StandardScaler, no Imputer needed as per initial inspection)
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()) # Only scaling needed
                ]
            )

            # Categorical Pipeline (OneHotEncoder and StandardScaler for sparse output)
            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')), # handle_unknown='ignore' prevents error for new categories
                    ("scaler", StandardScaler(with_mean=False)) # with_mean=False for sparse matrices after OneHotEncoder
                ]
            )

            logging.info(f"Categorical columns identified for transformation: {categorical_columns}")
            logging.info(f"Numerical columns identified for transformation: {numerical_columns}")

            # Combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ],
                remainder='drop' # Drops any columns not specified in numerical_columns or categorical_columns
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            # Drop 'College_ID' if it still exists in the split datasets (it should have been dropped in DataIngestion)
            # This is a safeguard; if DataIngestion consistently drops it, this won't be strictly necessary here.
            if 'College_ID' in train_df.columns:
                train_df = train_df.drop('College_ID', axis=1)
            if 'College_ID' in test_df.columns:
                test_df = test_df.drop('College_ID', axis=1)
            logging.info("Removed 'College_ID' column from train/test data if present.")

            logging.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            # --- MODIFIED TARGET COLUMN NAME FOR COLLEGE PLACEMENT DATASET ---
            target_column_name = "Placement"
            # -----------------------------------------------------------------

            # Apply LabelEncoder to the target column (Placement)
            # This is crucial because OneHotEncoder is for features, not targets.
            # And many classification models expect numerical labels (0 and 1).
            le_placement = LabelEncoder()
            train_df[target_column_name] = le_placement.fit_transform(train_df[target_column_name])
            test_df[target_column_name] = le_placement.transform(test_df[target_column_name])
            logging.info(f"Target column '{target_column_name}' encoded.")
            logging.info(f"Placement mapping: {list(le_placement.classes_)} -> {le_placement.transform(le_placement.classes_)}")


            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Fit and transform the training data, transform the test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate processed features with the target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                preprocessing_obj, # Returning the preprocessor object itself, not just its path
            )
        except Exception as e:
            raise CustomException(e, sys)

# Example usage (assuming DataIngestion has already run and saved CSVs)
if __name__ == "__main__":
    # Ensure 'artifacts' directory exists for DataIngestion to save files
    os.makedirs('artifacts', exist_ok=True)

    # Mock DataIngestion output for testing DataTransformation independently
    # In a real setup, you'd import and run DataIngestion
    # For demonstration, let's create dummy train/test CSVs with expected columns
    dummy_data = {
        'IQ': np.random.randint(70, 130, 100),
        'Prev_Sem_Result': np.random.uniform(5.0, 9.0, 100),
        'CGPA': np.random.uniform(5.0, 9.0, 100),
        'Academic_Performance': np.random.randint(1, 11, 100),
        'Internship_Experience': np.random.choice(['Yes', 'No'], 100),
        'Extra_Curricular_Score': np.random.randint(0, 11, 100),
        'Communication_Skills': np.random.randint(1, 11, 100),
        'Projects_Completed': np.random.randint(0, 6, 100),
        'Placement': np.random.choice(['Yes', 'No'], 100)
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_train, dummy_test = train_test_split(dummy_df, test_size=0.2, random_state=42, stratify=dummy_df['Placement'])

    train_path_mock = os.path.join('artifacts', 'train.csv')
    test_path_mock = os.path.join('artifacts', 'test.csv')

    dummy_train.to_csv(train_path_mock, index=False)
    dummy_test.to_csv(test_path_mock, index=False)
    logging.info("Dummy train/test CSVs created for DataTransformation testing.")


    data_transformation = DataTransformation()
    # Assuming DataIngestion has already saved train.csv and test.csv in 'artifacts'
    train_arr, test_arr, preprocessor_obj = data_transformation.initiate_data_transformation(train_path_mock, test_path_mock)

    print("\nShape of transformed train_arr:", train_arr.shape)
    print("Shape of transformed test_arr:", test_arr.shape)
    print("\nFirst 5 rows of transformed train_arr:")
    print(train_arr[:5])