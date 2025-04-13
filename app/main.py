from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model import train_model, load_model
from typing import Optional
import pickle
import pandas as pd
import io
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create a directory for temporary files if it doesn't exists
os.makedirs("temp", exist_ok=True)

# Train model when the app starts
train_model()
model_data = load_model()
loaded_model = model_data["model"]
feature_names = model_data["feature_names"]

# Load encoders
with open("./models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.get("/")
def home():
    return {"message": "FastAPI Customer Churn Prediction App"}

@app.post("/predict")
def predict(input_data: dict):
    """Make predictions from input data"""
    input_data_df = pd.DataFrame([input_data])

    # Encode categorical features
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])

    # Make prediction
    prediction = loaded_model.predict(input_data_df)
    probability = loaded_model.predict_proba(input_data_df)

    return {
        "prediction": "Customer will churn" if prediction[0] == 1 else "Customer will not churn",
        "probability": probability.tolist()
    }


@app.post("/predict-batch")
async def predict_batct(file: UploadFile = File(...)):
    """
        Process a CSV file with customer data and return predictions
        Returns an augmented CSV with additional columns for churn prediction amd probability
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=404, detail="Only CSV files are supported")
    
    try:
        # Read the CSV file

        content = await file.read()
        input_df = pd.read_csv(io.StringIO(content.decode('utf-8')))

        # Make a copy of the input dataframe to avoid modifying it 
        df_copy = input_df.copy()

        # Check if required columns are present 
        missing_columns = [col for col in feature_names if col not in df_copy.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
    
        # Encode categorical features 
        for column, encoder in encoders.items():
            if column in df_copy.columns:
                df_copy[column] = encoder.transform(df_copy[column])

        # Make predictions for each row 
        predictions = loaded_model.predict(df_copy)
        probabilities = loaded_model.predict_proba(df_copy)

        # Add predictions to original dataframe
        input_df['churn_prediction'] = ["Yes" if pred == 1 else "No" for pred in predictions]
        input_df['churn_probability'] = [round(prob[1] * 100, 2) for prob in probabilities]

        # Save the results in temporary file
        result_filename = f"temp/result_{uuid.uuid4()}.csv"
        input_df.to_csv(result_filename, index=False)

        # Return the file 
        return FileResponse(
            path=result_filename,
            filename="churn_predictions.csv",
            media_type="text/csv",
            background=BackgroundTask(lambda: os.remove(result_filename))
            
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Processing file: {str(e)}")
    
# Background task to clean up temporary files
from starlette.background import BackgroundTask


    
