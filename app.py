from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Flask app initialization
application = Flask(__name__)
app = application

# =============================================================
# Route for the home page (index.html)
# =============================================================
@app.route('/')
def index():
    """Renders the main index page."""
    return render_template('index.html')

# =============================================================
# Route for the prediction page (home.html)
# =============================================================
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handles data input and model prediction."""
    if request.method == 'GET':
        # If the request is a GET, just render the form page
        return render_template('home.html')
    else:
        # If the request is a POST, process the form data
        # Instantiate CustomData with new project features
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
        
        # Convert the CustomData object into a pandas DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Instantiate the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        # Get the prediction from the model
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        # The model's output is a numerical label (e.g., 0 or 1).
        # We convert it back to a human-readable string ('No' or 'Yes').
        # Assuming 1 is 'Yes' and 0 is 'No' based on our LabelEncoder setup.
        placement_result = 'Yes' if results[0] == 1 else 'No'

        # Render the home.html template with the prediction result
        return render_template('home.html', results=placement_result)

if __name__ == "__main__":
    # The port is changed to a non-standard one to avoid conflicts
    # app.run(host="0.0.0.0", debug=True, port=5001)
    app.run(host="0.0.0.0", debug=True)
