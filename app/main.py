from fastapi import FastAPI
from app.model import train_model, load_model
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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
