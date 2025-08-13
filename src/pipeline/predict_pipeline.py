import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 iq: float,
                 prev_sem_result: float,
                 cgpa: float,
                 academic_performance: float,
                 internship_experience: str,
                 extra_curricular_score: float,
                 communication_skills: float,
                 projects_completed: float):
        self.iq = iq
        self.prev_sem_result = prev_sem_result
        self.cgpa = cgpa
        self.academic_performance = academic_performance
        self.internship_experience = internship_experience
        self.extra_curricular_score = extra_curricular_score
        self.communication_skills = communication_skills
        self.projects_completed = projects_completed

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "IQ": [self.iq],
                "Prev_Sem_Result": [self.prev_sem_result],
                "CGPA": [self.cgpa],
                "Academic_Performance": [self.academic_performance],
                "Internship_Experience": [self.internship_experience],
                "Extra_Curricular_Score": [self.extra_curricular_score],
                "Communication_Skills": [self.communication_skills],
                "Projects_Completed": [self.projects_completed],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
