from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


@app.route('/')
def index():
    
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    
    if request.method == 'GET':
 
        return render_template('home.html')
    else:

        data = CustomData(
            iq=float(request.form.get('iq')),
            prev_sem_result=float(request.form.get('prev_sem_result')),
            cgpa=float(request.form.get('cgpa')),
            academic_performance=float(request.form.get('academic_performance')),
            internship_experience=request.form.get('internship_experience'),
            extra_curricular_score=float(request.form.get('extra_curricular_score')),
            communication_skills=float(request.form.get('communication_skills')),
            projects_completed=float(request.form.get('projects_completed'))
        )
        
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
   
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
       
        placement_result = 'Yes' if results[0] == 1 else 'No'

        
        return render_template('home.html', results=placement_result)

if __name__ == "__main__":
 
    app.run(host="0.0.0.0", debug=True)
